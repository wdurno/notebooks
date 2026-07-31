"""Run the Phase 7 coupled dense/structured Fisher comparison."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import resource
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset

from mnist_experiment.run_coupled import (
    _online_metrics,
    _path_divergence_rows,
    _prefix_mapping,
    _tensor_hash,
    _trajectory_hash,
)
from mnist_experiment.run_experiment import (
    _ReferenceOracle,
    _calculate_online_statistics,
    _clone_at_vector,
    _cpu_tree,
    _materialize_batch,
    _statistics_to_device,
    _synchronize,
)
from mnist_experiment.run_representations import _fixed_probes
from src.artifacts import RunStore
from src.config import ExperimentConfig, load_config
from src.coupled_trajectory import DenseFisherTracker, dense_fisher_metrics
from src.ewc import build_optimizer, mixture_ewc_strength, take_ewc_proposal
from src.initialization import evaluate_classifier, load_replica_bundle_for_config
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import configure_torch_runtime, resolve_device, resolve_dtype
from src.parameters import ParameterLayout
from src.reference import ReferenceFisherStore
from src.representations import DiagonalFisher, LowRankDiagonalFisher
from src.seeding import derive_component_seed
from src.structured_trajectory import (
    DiagonalFisherTracker,
    LowRankDiagonalFisherTracker,
    structured_representation_metrics,
)

PHASE7_COUPLED_METRIC_SCHEMA_VERSION = 1
PHASE7_COUPLED_TRAJECTORY_SCHEMA_VERSION = 1
PHASE7_COUPLED_CHECKPOINT_SCHEMA_VERSION = 1
PHASE7_COUPLED_REFERENCE_PLAN_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_config(config: ExperimentConfig) -> int:
    if config.schema_version != 6:
        raise ValueError("Phase 7 coupled runs require schema version 6")
    if config.estimator.representation != "low_rank_diagonal":
        raise ValueError(
            "Phase 7 coupled runs require a low_rank_diagonal estimator"
        )
    if (
        config.estimator.low_rank_grid is None
        or config.estimator.low_rank_grid
        != [0, config.estimator.low_rank]
    ):
        raise ValueError(
            "coupled low_rank_grid must contain only zero and the selected rank"
        )
    if config.estimator.low_rank is None or config.estimator.low_rank < 1:
        raise ValueError("coupled runs require a positive selected rank")
    if config.estimator.fresh_fisher_cadence is None:
        raise ValueError("coupled runs require fresh_fisher_cadence")
    if config.estimator.ridge_half_life_steps is None:
        raise ValueError("coupled runs require directional-ridge settings")
    if config.controller.policy != "fixed":
        raise ValueError("coupled runs require controller.policy='fixed'")
    if not (
        0.0
        < config.controller.fixed_pi
        <= config.controller.pi_max
        < 1.0
    ):
        raise ValueError("coupled runs require 0 < fixed_pi <= pi_max < 1")
    return config.estimator.low_rank


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


def _structured_tracking_metrics(
    representation: DiagonalFisher | LowRankDiagonalFisher,
    reference: Tensor,
    probes: Tensor,
    direction: Tensor,
) -> dict[str, Any]:
    metrics = structured_representation_metrics(
        representation,
        reference,
        reference,
        probes,
        direction,
    )
    return {
        "relative_frobenius_error": (
            metrics["relative_frobenius_error_to_reference"]
        ),
        "fixed_probe_matvec_relative_error": (
            metrics["fixed_probe_matvec_relative_error"]
        ),
        "mean_probe_quadratic_relative_error": (
            metrics["mean_probe_quadratic_relative_error"]
        ),
        "maximum_probe_quadratic_relative_error": (
            metrics["maximum_probe_quadratic_relative_error"]
        ),
        "directional_quadratic_relative_error": (
            metrics["directional_quadratic_relative_error"]
        ),
        "leading_eigenvalue_relative_error": (
            metrics["leading_eigenvalue_relative_error"]
        ),
        "leading_eigenvector_alignment": (
            metrics["leading_eigenvector_alignment"]
        ),
        "representation_storage_bytes": (
            metrics["representation_storage_bytes"]
        ),
        "dense_storage_bytes": metrics["dense_storage_bytes"],
    }


def _run_condition(
    method: str,
    selected_rank: int,
    config: ExperimentConfig,
    initial_model: nn.Module,
    initial_template: nn.Module,
    train_dataset: Dataset,
    evaluation_dataset: Dataset,
    stream_plan,
    oracle: _ReferenceOracle,
    probes: Tensor,
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
    tracker_arguments = {
        "initial_fisher": initial_fisher,
        "ema_gain": config.estimator.ema_gain,
        "ridge_half_life_steps": config.estimator.ridge_half_life_steps,
        "ridge_amplitude_epsilon": (
            config.estimator.ridge_amplitude_epsilon
        ),
        "ridge_coherence_threshold": (
            config.estimator.ridge_coherence_threshold
        ),
    }
    if method == "dense_ridge_full":
        tracker = DenseFisherTracker(
            "ridge_full_lfu",
            fresh_fisher_cadence=config.estimator.fresh_fisher_cadence,
            **tracker_arguments,
        )
    elif method == "diagonal_ridge_full":
        tracker = DiagonalFisherTracker(**tracker_arguments)
    elif method == f"low_rank_diagonal_r{selected_rank}":
        tracker = LowRankDiagonalFisherTracker(
            **tracker_arguments,
            rank=selected_rank,
        )
    else:
        raise ValueError(f"unsupported structured coupled method: {method}")

    adaptation_weight = config.controller.fixed_pi
    effective_strength = mixture_ewc_strength(
        adaptation_weight,
        multiplier=config.optimizer.ewc_strength,
    )
    checkpoint_indices = {
        0,
        len(stream_plan.p_values) // 2,
        len(stream_plan.p_values) - 1,
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
        statistics = _statistics_to_device(
            [statistics_cpu],
            device,
            matrix_dtype,
        )[0]
        direction = lagged_displacement.to(device=device, dtype=matrix_dtype)
        update_started = time.perf_counter()
        if method == "dense_ridge_full":
            update = tracker.update(step, statistics, direction, reference)
            fisher = update.estimate
            tracking = dense_fisher_metrics(update, reference, direction)
            representation_mapping = {
                "kind": "dense",
                "matrix": update.estimate,
            }
            update_diagnostics = {
                "projection": dataclasses.asdict(update.projection),
                "ridge": update.ridge_metrics,
            }
            ridge_state = update.ridge_state
        elif method == "diagonal_ridge_full":
            update = tracker.update(step, statistics, direction)
            fisher = update.representation
            tracking = _structured_tracking_metrics(
                fisher,
                reference,
                probes,
                direction,
            )
            update_diagnostics = {
                "projection": dataclasses.asdict(update.projection),
                "ridge": update.ridge_metrics,
            }
            representation_mapping = fisher.artifact_mapping()
            ridge_state = update.ridge_state
        else:
            update = tracker.update(
                step,
                statistics,
                direction,
                lanczos_seed=derive_component_seed(
                    config.replica_seed,
                    f"phase7_coupled_lanczos:{method}:step={step}",
                ),
            )
            fisher = update.representation
            tracking = _structured_tracking_metrics(
                fisher,
                reference,
                probes,
                direction,
            )
            update_diagnostics = {
                "candidate_minimum_eigenvalue": (
                    update.candidate_minimum_eigenvalue
                ),
                "candidate_negative_eigenvalue_count": (
                    update.candidate_negative_eigenvalue_count
                ),
                "lanczos": update.lanczos.mapping(),
                "ridge": update.ridge_metrics,
            }
            representation_mapping = fisher.artifact_mapping()
            ridge_state = update.ridge_state
        _synchronize(device)
        update_elapsed = time.perf_counter() - update_started

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
            proposal_fisher = (
                fisher.to(dtype=training_dtype)
                if not isinstance(fisher, Tensor)
                else fisher.to(dtype=training_dtype)
            )
            proposal = take_ewc_proposal(
                model,
                layout,
                training_inputs,
                training_targets,
                proposal_fisher,
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
                "representation": representation_mapping["kind"],
                "requested_rank": (
                    selected_rank
                    if method.startswith("low_rank_diagonal")
                    else 0 if method == "diagonal_ridge_full" else None
                ),
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
                "proposal": proposal_mapping,
                "update_elapsed_seconds": update_elapsed,
                **_prefix_mapping("before", before_evaluation),
                **_prefix_mapping("after", after_evaluation),
                **online,
                **tracking,
                "update_diagnostics": update_diagnostics,
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
                "representation": representation_mapping,
                "direct_fisher": statistics.estimate.fisher,
                "amari_chentsov": statistics.estimate.amari_chentsov,
                "residual": statistics.estimate.residual,
                "lagged_direction": direction,
                "accepted_displacement": accepted_displacement,
                "ridge_state": ridge_state,
                "update_diagnostics": update_diagnostics,
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


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    selected_rank = _validate_config(config)
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
    if selected_rank > loaded.layout.total_numel:
        raise ValueError("selected rank exceeds the parameter count")
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
    probe_seed = derive_component_seed(
        config.replica_seed,
        "phase7_coupled_fixed_probes",
    )
    probes = _fixed_probes(
        loaded.layout.total_numel,
        count=8,
        seed=probe_seed,
        device=device,
        dtype=matrix_dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    methods = (
        "dense_ridge_full",
        "diagonal_ridge_full",
        f"low_rank_diagonal_r{selected_rank}",
    )
    started = time.perf_counter()
    conditions = []
    for method in methods:
        print(f"phase7 coupled condition started: {method}", flush=True)
        condition = _run_condition(
            method,
            selected_rank,
            config,
            loaded.model,
            initial_template,
            train_dataset,
            evaluation_dataset,
            loaded.stream_plan,
            oracle,
            probes,
            device=device,
            training_dtype=training_dtype,
            derivative_dtype=derivative_dtype,
            matrix_dtype=matrix_dtype,
        )
        conditions.append(condition)
        print(
            f"phase7 coupled condition completed: {method} "
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

    trajectory_artifact = {
        "schema_version": PHASE7_COUPLED_TRAJECTORY_SCHEMA_VERSION,
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
        "schema_version": PHASE7_COUPLED_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_policy": "initial, midpoint, and final indices",
        "conditions": {
            condition.method: condition.checkpoints
            for condition in conditions
        },
    }
    reference_plan_artifact = {
        "schema_version": PHASE7_COUPLED_REFERENCE_PLAN_SCHEMA_VERSION,
        "plans": conditions[0].reference_plans,
    }
    metrics = {
        "phase7_coupled_metric_schema_version": (
            PHASE7_COUPLED_METRIC_SCHEMA_VERSION
        ),
        "run_kind": "structured_coupled",
        "replica_bundle_id": loaded.metadata["bundle_id"],
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "parameter_count": loaded.layout.total_numel,
        "device": str(device),
        "training_dtype": str(training_dtype),
        "derivative_dtype": str(derivative_dtype),
        "matrix_dtype": str(matrix_dtype),
        "methods": list(methods),
        "selected_rank": selected_rank,
        "probe_count": probes.shape[1],
        "probe_seed": probe_seed,
        "probe_hash": _tensor_hash(probes),
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
            "shared_probe_vectors": True,
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
        "phase7_coupled_trajectories.pt",
        _cpu_tree(trajectory_artifact),
    )
    checkpoint_path = session.write_torch(
        "phase7_coupled_checkpoints.pt",
        _cpu_tree(checkpoint_artifact),
    )
    reference_plan_path = session.write_torch(
        "phase7_coupled_reference_plans.pt",
        reference_plan_artifact,
    )
    metrics["artifact_files_bytes"] = {
        "phase7_coupled_trajectories.pt": trajectory_path.stat().st_size,
        "phase7_coupled_checkpoints.pt": checkpoint_path.stat().st_size,
        "phase7_coupled_reference_plans.pt": (
            reference_plan_path.stat().st_size
        ),
    }
    session.write_json("phase7_coupled_metrics.json", metrics)
    destination = session.complete(
        [
            "phase7_coupled_metrics.json",
            "phase7_coupled_trajectories.pt",
            "phase7_coupled_checkpoints.pt",
            "phase7_coupled_reference_plans.pt",
        ]
    )
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "path": str(destination),
                "methods": list(methods),
                "selected_rank": selected_rank,
                "steps": len(loaded.stream_plan.p_values),
                "path_diverged": metrics["pairing"]["path_diverged"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
