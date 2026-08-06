"""Run the immutable dense EWC-coupled MNIST experiment."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset

from mnist_experiment.run_experiment import (
    _ReferenceOracle,
    _calculate_online_statistics,
    _clone_at_vector,
    _cpu_tree,
    _materialize_batch,
    _statistics_to_device,
    _synchronize,
)
from src.artifacts import RunStore
from src.config import ExperimentConfig, load_config
from src.coupled_trajectory import (
    COUPLED_DENSE_METHODS,
    DenseFisherTracker,
    dense_fisher_metrics,
)
from src.ewc import build_optimizer, mixture_ewc_strength, take_ewc_proposal
from src.fixed_trajectory import OnlineStepStatistics
from src.initialization import evaluate_classifier, load_replica_bundle_for_config
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import configure_torch_runtime, resolve_device, resolve_dtype
from src.parameters import ParameterLayout
from src.reference import ReferenceFisherStore, relative_frobenius_error

PHASE6_METRIC_SCHEMA_VERSION = 1
PHASE6_TRAJECTORY_SCHEMA_VERSION = 1
PHASE6_MATRIX_SCHEMA_VERSION = 1
PHASE6_REFERENCE_PLAN_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_phase6_config(config: ExperimentConfig) -> None:
    if config.estimator.representation != "dense":
        raise ValueError("Phase 6 requires estimator.representation='dense'")
    if config.estimator.fresh_fisher_cadence is None:
        raise ValueError("Phase 6 requires estimator.fresh_fisher_cadence")
    if config.estimator.ridge_half_life_steps is None:
        raise ValueError("Phase 6 smoke coverage requires ridge settings")
    if config.controller.policy != "fixed":
        raise ValueError("Phase 6 requires controller.policy='fixed'")
    pi_value = config.controller.fixed_pi
    if not 0.0 < pi_value <= config.controller.pi_max < 1.0:
        raise ValueError(
            "Phase 6 requires 0 < controller.fixed_pi <= pi_max < 1"
        )


def _tensor_hash(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _trajectory_hash(
    method: str,
    parameters: Tensor,
    displacements: Tensor,
    stream_plan_hash: str,
) -> str:
    payload = json.dumps(
        {
            "method": method,
            "parameters_hash": _tensor_hash(parameters),
            "displacements_hash": _tensor_hash(displacements),
            "stream_plan_hash": stream_plan_hash,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prefix_mapping(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _online_metrics(
    statistics: OnlineStepStatistics,
    reference: Tensor,
    direction: Tensor,
    previous_reference: Tensor | None,
) -> dict[str, Any]:
    estimate = statistics.estimate
    ac_norm = torch.linalg.matrix_norm(estimate.amari_chentsov, ord="fro")
    residual_norm = torch.linalg.matrix_norm(estimate.residual, ord="fro")
    denominator = ac_norm * residual_norm
    reference_increment = (
        None if previous_reference is None else reference - previous_reference
    )
    return {
        "direction_norm": float(torch.linalg.vector_norm(direction)),
        "direct_fisher_relative_error": relative_frobenius_error(
            estimate.fisher,
            reference,
        ),
        "direct_fisher_fro": float(
            torch.linalg.matrix_norm(estimate.fisher, ord="fro")
        ),
        "ac_fro": float(ac_norm),
        "residual_fro": float(residual_norm),
        "full_lfu_fro": float(
            torch.linalg.matrix_norm(estimate.full, ord="fro")
        ),
        "ac_residual_alignment": (
            None
            if float(denominator) == 0.0
            else float(
                (estimate.amari_chentsov * estimate.residual).sum()
                / denominator
            )
        ),
        "reference_increment_fro": (
            None
            if reference_increment is None
            else float(torch.linalg.matrix_norm(reference_increment, ord="fro"))
        ),
        "ac_increment_relative_error": (
            None
            if reference_increment is None
            else relative_frobenius_error(
                estimate.amari_chentsov,
                reference_increment,
            )
        ),
        "full_increment_relative_error": (
            None
            if reference_increment is None
            else relative_frobenius_error(
                estimate.full,
                reference_increment,
            )
        ),
        "score_gradient_count": statistics.score_gradient_count,
        "hvp_count": statistics.hvp_count,
        "derivative_elapsed_seconds": statistics.elapsed_seconds,
    }


@dataclasses.dataclass(frozen=True)
class _ConditionRun:
    method: str
    trajectory_hash: str
    parameters: Tensor
    displacements: Tensor
    rows: tuple[dict[str, Any], ...]
    references: tuple[dict[str, Any], ...]
    checkpoints: dict[str, Any]
    reference_plans: dict[str, Any]
    optimizer_state: dict[str, Any]
    elapsed_seconds: float


def _run_condition(
    method: str,
    config: ExperimentConfig,
    initial_model: nn.Module,
    initial_template: nn.Module,
    train_dataset: Dataset,
    evaluation_dataset: Dataset,
    stream_plan,
    oracle: _ReferenceOracle,
    *,
    device: torch.device,
    training_dtype: torch.dtype,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> _ConditionRun:
    model = copy.deepcopy(initial_model).to(
        device=device,
        dtype=training_dtype,
    )
    layout = ParameterLayout.from_module(model)
    optimizer = build_optimizer(model, config.optimizer)
    initial_vector = layout.flatten_module(model, detach=True)
    initial_record = oracle.estimate(0, stream_plan.p_values[0], initial_vector)
    initial_fisher = initial_record.matrix.to(
        device=device,
        dtype=matrix_dtype,
    )
    tracker = DenseFisherTracker(
        method,
        initial_fisher,
        ema_gain=config.estimator.ema_gain,
        fresh_fisher_cadence=config.estimator.fresh_fisher_cadence,
        ridge_half_life_steps=config.estimator.ridge_half_life_steps,
        ridge_amplitude_epsilon=config.estimator.ridge_amplitude_epsilon,
        ridge_coherence_threshold=config.estimator.ridge_coherence_threshold,
    )
    adaptation_weight = config.controller.fixed_pi
    effective_strength = mixture_ewc_strength(
        adaptation_weight,
        multiplier=config.optimizer.ewc_strength,
    )
    checkpoint_indices = {
        0,
        len(stream_plan.p_values) - 1,
        *range(
            config.estimator.fresh_fisher_cadence,
            len(stream_plan.p_values),
            config.estimator.fresh_fisher_cadence,
        ),
    }
    parameters = []
    displacements = []
    rows = []
    reference_rows = []
    checkpoints = {}
    reference_plans = {}
    lagged_displacement = torch.zeros_like(initial_vector).cpu()
    previous_reference = None
    started = time.perf_counter()

    for step, p_value in enumerate(stream_plan.p_values):
        parameter_before = layout.flatten_module(model, detach=True)
        parameters.append(parameter_before.cpu())
        parameter_hash = _tensor_hash(parameter_before)
        record = oracle.estimate(step, p_value, parameter_before)
        reference = record.matrix.to(device=device, dtype=matrix_dtype)
        reference_plans[str(step)] = record.plan.to_mapping()

        derivative_model, derivative_layout = _clone_at_vector(
            initial_template,
            parameter_before,
            device=device,
            dtype=derivative_dtype,
        )
        derivative_inputs, derivative_targets = _materialize_batch(
            train_dataset,
            stream_plan.observation_indices[step],
            device=device,
            dtype=derivative_dtype,
        )
        expected_targets = torch.as_tensor(
            stream_plan.class_labels[step],
            dtype=torch.long,
            device=device,
        )
        if not torch.equal(derivative_targets, expected_targets):
            raise RuntimeError("paired stream labels do not match the dataset")
        try:
            statistics_cpu = _calculate_online_statistics(
                derivative_model,
                derivative_layout,
                derivative_inputs,
                derivative_targets,
                lagged_displacement.to(device=device, dtype=derivative_dtype),
                matrix_dtype=matrix_dtype,
                include_hvp=step > 0,
                device=device,
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"{method} failed derivative estimation at step {step}, "
                f"p={p_value:g}: {exc}"
            ) from exc
        statistics = _statistics_to_device(
            [statistics_cpu],
            device,
            matrix_dtype,
        )[0]
        direction = lagged_displacement.to(device=device, dtype=matrix_dtype)
        try:
            update = tracker.update(step, statistics, direction, reference)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"{method} failed Fisher tracking at step {step}, "
                f"p={p_value:g}: {exc}"
            ) from exc

        before_evaluation = evaluate_classifier(
            model,
            evaluation_dataset,
            batch_size=config.initialization.batch_size,
            device=device,
            dtype=training_dtype,
            num_workers=config.initialization.num_workers,
        )
        proposal_mapping = None
        after_evaluation = before_evaluation
        accepted_displacement = None
        if step + 1 < len(stream_plan.p_values):
            training_inputs, training_targets = _materialize_batch(
                train_dataset,
                stream_plan.observation_indices[step],
                device=device,
                dtype=training_dtype,
            )
            proposal = take_ewc_proposal(
                model,
                layout,
                training_inputs,
                training_targets,
                update.estimate.to(dtype=training_dtype),
                config.optimizer,
                optimizer,
                adaptation_weight=adaptation_weight,
            )
            accepted_displacement = (
                layout.flatten_module(model, detach=True) - parameter_before
            ).cpu()
            agreement_error = float(
                torch.linalg.vector_norm(
                    accepted_displacement - proposal.displacement
                )
            )
            agreement_tolerance = (
                10.0
                * torch.finfo(accepted_displacement.dtype).eps
                * max(
                    float(torch.linalg.vector_norm(accepted_displacement)),
                    1.0,
                )
            )
            if agreement_error > agreement_tolerance:
                raise RuntimeError(
                    "proposal displacement does not equal the accepted move"
                )
            if proposal.effective_ewc_strength != effective_strength:
                raise RuntimeError("proposal used the wrong EWC odds coefficient")
            displacements.append(accepted_displacement)
            lagged_displacement = accepted_displacement
            proposal_mapping = {
                **proposal.metrics_mapping(),
                "accepted_displacement_norm": proposal.displacement_norm,
                "accepted_displacement_agreement_error": agreement_error,
                "accepted_displacement_agreement_tolerance": (
                    agreement_tolerance
                ),
                "post_optimization_scaling_applied": False,
            }
            after_evaluation = evaluate_classifier(
                model,
                evaluation_dataset,
                batch_size=config.initialization.batch_size,
                device=device,
                dtype=training_dtype,
                num_workers=config.initialization.num_workers,
            )

        tracking = dense_fisher_metrics(update, reference, direction)
        online = _online_metrics(
            statistics,
            reference,
            direction,
            previous_reference,
        )
        rows.append(
            {
                "method": method,
                "step": step,
                "p": p_value,
                "adaptation_weight": adaptation_weight,
                "pi_max": config.controller.pi_max,
                "effective_ewc_strength": effective_strength,
                "objective_normalization": (
                    "mean_new_loss_plus_old_to_new_odds"
                ),
                "parameter_hash": parameter_hash,
                "parameter_norm": float(
                    torch.linalg.vector_norm(parameter_before)
                ),
                "distance_from_initial": float(
                    torch.linalg.vector_norm(parameter_before - initial_vector)
                ),
                "reference_cache_digest": record.cache_digest,
                "reference_plan_hash": record.plan.content_hash,
                "reference_parameter_hash": parameter_hash,
                "proposal": proposal_mapping,
                **_prefix_mapping("before", before_evaluation),
                **_prefix_mapping("after", after_evaluation),
                **online,
                **tracking,
            }
        )
        reference_rows.append(
            {
                "method": method,
                "step": step,
                "p": p_value,
                "parameter_hash": parameter_hash,
                **record.metrics_mapping(step),
            }
        )
        if step in checkpoint_indices:
            checkpoints[str(step)] = {
                "parameter": parameter_before,
                "reference": reference,
                "estimate": update.estimate,
                "prediction": update.prediction,
                "candidate": update.candidate,
                "direct_fisher": statistics.estimate.fisher,
                "amari_chentsov": statistics.estimate.amari_chentsov,
                "residual": statistics.estimate.residual,
                "lagged_direction": direction,
                "accepted_displacement": accepted_displacement,
                "ridge_state": update.ridge_state,
                "reference_cache_digest": record.cache_digest,
                "reference_plan_hash": record.plan.content_hash,
                "parameter_hash": parameter_hash,
            }
        previous_reference = reference

    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    if not torch.equal(
        displacement_tensor,
        parameter_tensor[1:] - parameter_tensor[:-1],
    ):
        raise RuntimeError("coupled trajectory displacements are inconsistent")
    content_hash = _trajectory_hash(
        method,
        parameter_tensor,
        displacement_tensor,
        stream_plan.content_hash,
    )
    return _ConditionRun(
        method=method,
        trajectory_hash=content_hash,
        parameters=parameter_tensor,
        displacements=displacement_tensor,
        rows=tuple(rows),
        references=tuple(reference_rows),
        checkpoints=checkpoints,
        reference_plans=reference_plans,
        optimizer_state=optimizer.state_dict(),
        elapsed_seconds=time.perf_counter() - started,
    )


def _path_divergence_rows(
    conditions: Sequence[_ConditionRun],
    p_values: Sequence[float],
) -> list[dict[str, Any]]:
    rows = []
    for step, p_value in enumerate(p_values):
        distances = []
        for left_index, left in enumerate(conditions):
            for right in conditions[left_index + 1 :]:
                distances.append(
                    float(
                        torch.linalg.vector_norm(
                            left.parameters[step] - right.parameters[step]
                        )
                    )
                )
        rows.append(
            {
                "step": step,
                "p": p_value,
                "maximum_pairwise_parameter_distance": (
                    max(distances) if distances else 0.0
                ),
                "mean_pairwise_parameter_distance": (
                    sum(distances) / len(distances) if distances else 0.0
                ),
            }
        )
    return rows


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    _validate_phase6_config(config)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=(
            config.runtime.deterministic_algorithms
            and config.runtime.device in {"cuda", "auto"}
        ),
    )
    cache_parent = Path(config.cache_root).parent
    data_root = arguments.data_root or cache_parent / "datasets"
    replica_root = arguments.replica_root or cache_parent / "replicas"
    reference_root = arguments.reference_root or cache_parent / "references"
    output_root = arguments.output_root or Path(config.cache_root)
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.training_dtype)
    derivative_dtype = resolve_dtype(config.reference.derivative_dtype)
    matrix_dtype = resolve_dtype(config.runtime.matrix_dtype)

    session = RunStore(output_root).begin(
        config,
        Path(__file__).parents[1],
        resume=arguments.resume,
    )
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=False)
    train_targets = dataset_targets(train_dataset)
    loaded = load_replica_bundle_for_config(
        replica_root,
        config,
        device=device,
    )
    evaluation_dataset = Subset(test_dataset, loaded.partitions.evaluation)
    initial_template = copy.deepcopy(loaded.model).cpu()
    oracle = _ReferenceOracle(
        config,
        initial_template,
        train_dataset,
        train_targets,
        loaded.partitions,
        ReferenceFisherStore(reference_root),
        device=device,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    conditions = []
    for method in COUPLED_DENSE_METHODS:
        print(f"phase6 condition started: {method}", flush=True)
        condition = _run_condition(
            method,
            config,
            loaded.model,
            initial_template,
            train_dataset,
            evaluation_dataset,
            loaded.stream_plan,
            oracle,
            device=device,
            training_dtype=training_dtype,
            derivative_dtype=derivative_dtype,
            matrix_dtype=matrix_dtype,
        )
        conditions.append(condition)
        print(
            f"phase6 condition completed: {method} "
            f"({condition.elapsed_seconds:.3f}s)",
            flush=True,
        )
    _synchronize(device)
    total_elapsed = time.perf_counter() - started
    path_divergence = _path_divergence_rows(
        conditions,
        loaded.stream_plan.p_values,
    )
    initial_parameter_hashes = {
        _tensor_hash(condition.parameters[0]) for condition in conditions
    }
    if len(initial_parameter_hashes) != 1:
        raise RuntimeError("paired conditions did not share initialization")
    methods = [condition.method for condition in conditions]
    trajectory_artifact = {
        "schema_version": PHASE6_TRAJECTORY_SCHEMA_VERSION,
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "p_values": loaded.stream_plan.p_values,
        "observation_indices": loaded.stream_plan.observation_indices,
        "class_labels": loaded.stream_plan.class_labels,
        "parameter_layout": loaded.layout.metadata(),
        "conditions": {
            condition.method: {
                "content_hash": condition.trajectory_hash,
                "parameters": condition.parameters,
                "displacements": condition.displacements,
                "optimizer_state": condition.optimizer_state,
            }
            for condition in conditions
        },
    }
    checkpoint_artifact = {
        "schema_version": PHASE6_MATRIX_SCHEMA_VERSION,
        "checkpoint_policy": "initial, final, and periodic-fresh indices",
        "conditions": {
            condition.method: condition.checkpoints
            for condition in conditions
        },
    }
    reference_plan_artifact = {
        "schema_version": PHASE6_REFERENCE_PLAN_SCHEMA_VERSION,
        "plans": conditions[0].reference_plans,
    }
    metrics = {
        "phase6_metric_schema_version": PHASE6_METRIC_SCHEMA_VERSION,
        "replica_bundle_id": loaded.metadata["bundle_id"],
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "device": str(device),
        "training_dtype": str(training_dtype),
        "derivative_dtype": str(derivative_dtype),
        "matrix_dtype": str(matrix_dtype),
        "methods": methods,
        "adaptation": {
            "environmental_variable": "p",
            "adaptation_weight": config.controller.fixed_pi,
            "pi_max": config.controller.pi_max,
            "ewc_multiplier": config.optimizer.ewc_strength,
            "effective_ewc_strength": mixture_ewc_strength(
                config.controller.fixed_pi,
                multiplier=config.optimizer.ewc_strength,
            ),
            "objective_normalization": (
                "mean_new_loss_plus_old_to_new_odds"
            ),
            "post_optimization_scaling": False,
        },
        "pairing": {
            "shared_initialization": True,
            "shared_observation_stream": True,
            "initial_parameter_hash": next(iter(initial_parameter_hashes)),
            "path_diverged": (
                path_divergence[-1]["maximum_pairwise_parameter_distance"] > 0
            ),
        },
        "path_hashes": {
            condition.method: condition.trajectory_hash
            for condition in conditions
        },
        "path_divergence": path_divergence,
        "condition_steps": [
            row for condition in conditions for row in condition.rows
        ],
        "references": [
            row for condition in conditions for row in condition.references
        ],
        "condition_elapsed_seconds": {
            condition.method: condition.elapsed_seconds
            for condition in conditions
        },
        "total_elapsed_seconds": total_elapsed,
        "total_score_gradient_count": sum(
            int(row["score_gradient_count"])
            for condition in conditions
            for row in condition.rows
        ),
        "total_hvp_count": sum(
            int(row["hvp_count"])
            for condition in conditions
            for row in condition.rows
        ),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }
    trajectory_path = session.write_torch(
        "phase6_trajectories.pt",
        _cpu_tree(trajectory_artifact),
    )
    checkpoint_path = session.write_torch(
        "phase6_checkpoints.pt",
        _cpu_tree(checkpoint_artifact),
    )
    reference_plan_path = session.write_torch(
        "phase6_reference_plans.pt",
        reference_plan_artifact,
    )
    metrics["artifact_files_bytes"] = {
        "phase6_trajectories.pt": trajectory_path.stat().st_size,
        "phase6_checkpoints.pt": checkpoint_path.stat().st_size,
        "phase6_reference_plans.pt": reference_plan_path.stat().st_size,
    }
    session.write_json("phase6_metrics.json", metrics)
    destination = session.complete(
        [
            "phase6_metrics.json",
            "phase6_trajectories.pt",
            "phase6_checkpoints.pt",
            "phase6_reference_plans.pt",
        ]
    )
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "path": str(destination),
                "methods": methods,
                "steps": len(loaded.stream_plan.p_values),
                "effective_ewc_strength": metrics["adaptation"][
                    "effective_ewc_strength"
                ],
                "path_diverged": metrics["pairing"]["path_diverged"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
