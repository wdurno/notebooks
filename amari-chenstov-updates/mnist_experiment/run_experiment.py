"""Run the immutable dense fixed-trajectory MNIST experiment."""

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
from torch.utils.data import Dataset

from src.artifacts import RunStore
from src.config import ExperimentConfig, load_config
from src.derivatives import per_sample_derivatives
from src.fisher import LFUBatchEstimate, dense_lfu_estimate, empirical_fisher
from src.fixed_trajectory import (
    DenseConditionResult,
    FixedTrajectory,
    OnlineStepStatistics,
    common_step_metrics,
    generate_fixed_trajectory,
    lagged_directions,
    replay_dense_conditions,
)
from src.initialization import load_replica_bundle_for_config
from src.mnist_data import (
    ReferenceSamplePlan,
    dataset_targets,
    generate_reference_sample_plan,
    load_mnist_datasets,
)
from src.mnist_model import (
    configure_torch_runtime,
    mnist_nll,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.reference import (
    adaptive_reference_fisher,
    ReferenceFisherEstimate,
    ReferenceFisherStore,
    chunked_reference_fisher,
    reference_cache_key,
    relative_frobenius_error,
)
from src.representations import project_psd_frobenius
from src.seeding import derive_component_seed

PHASE4_METRIC_SCHEMA_VERSION = 2
PHASE4_MATRIX_SCHEMA_VERSION = 2
PHASE4_REFERENCE_PLAN_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_phase4_config(config: ExperimentConfig) -> None:
    if config.estimator.representation != "dense":
        raise ValueError("Phase 4 requires estimator.representation='dense'")
    if config.estimator.fresh_fisher_cadence is None:
        raise ValueError("Phase 4 requires estimator.fresh_fisher_cadence")
    if config.controller.policy != "uncontrolled":
        raise ValueError("Phase 4 fixed trajectories require an uncontrolled policy")
    if config.controller.fixed_pi != 1.0:
        raise ValueError("Phase 4 fixed trajectories require controller.fixed_pi=1")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _materialize_batch(
    dataset: Dataset,
    indices: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    rows = [dataset[int(index)] for index in indices]
    inputs = torch.stack([row[0] for row in rows]).to(
        device=device,
        dtype=dtype,
    )
    targets = torch.as_tensor(
        [int(row[1]) for row in rows],
        dtype=torch.long,
        device=device,
    )
    return inputs, targets


def _clone_at_vector(
    template: nn.Module,
    vector: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, ParameterLayout]:
    model = copy.deepcopy(template).to(device=device, dtype=dtype)
    layout = ParameterLayout.from_module(model)
    layout.copy_vector_to_module(
        model,
        vector.to(device=device, dtype=dtype),
    )
    return model, layout


@dataclasses.dataclass(frozen=True)
class _ReferenceRecord:
    matrix: Tensor
    estimate: ReferenceFisherEstimate
    cache_digest: str
    plan: ReferenceSamplePlan

    def metrics_mapping(self, step: int) -> dict[str, Any]:
        return {
            "step": step,
            "p": self.plan.p,
            "cache_digest": self.cache_digest,
            "plan_hash": self.plan.content_hash,
            "plan_seed": self.plan.seed,
            **self.estimate.diagnostics_mapping(),
        }


class _ReferenceOracle:
    def __init__(
        self,
        config: ExperimentConfig,
        template: nn.Module,
        dataset: Dataset,
        train_targets: Tensor,
        partitions,
        store: ReferenceFisherStore,
        *,
        device: torch.device,
        derivative_dtype: torch.dtype,
        matrix_dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.template = copy.deepcopy(template).cpu()
        self.dataset = dataset
        self.train_targets = train_targets
        self.partitions = partitions
        self.store = store
        self.device = device
        self.derivative_dtype = derivative_dtype
        self.matrix_dtype = matrix_dtype
        self._plans: dict[int, ReferenceSamplePlan] = {}
        self._records: dict[tuple[int, str, int], _ReferenceRecord] = {}

    def plan(self, step: int, p_value: float) -> ReferenceSamplePlan:
        existing = self._plans.get(step)
        if existing is not None:
            if existing.p != p_value:
                raise RuntimeError("reference step was requested with a different p")
            return existing
        seed = derive_component_seed(
            self.config.replica_seed,
            f"phase4_reference:step={step}:p={p_value:.17g}",
        )
        plan = generate_reference_sample_plan(
            self.train_targets,
            self.partitions,
            self.config.data,
            p=p_value,
            sample_size=self.config.reference.sample_size,
            seed=seed,
            pool="reference",
        )
        self._plans[step] = plan
        return plan

    def estimate(
        self,
        step: int,
        p_value: float,
        parameter_vector: Tensor,
        *,
        sample_size: int | None = None,
    ) -> _ReferenceRecord:
        full_plan = self.plan(step, p_value)
        plan = (
            full_plan
            if sample_size is None
            else full_plan.prefix(sample_size)
        )
        model, layout = _clone_at_vector(
            self.template,
            parameter_vector,
            device=self.device,
            dtype=self.derivative_dtype,
        )
        maximum_key = reference_cache_key(
            model,
            layout,
            plan,
            derivative_dtype=self.derivative_dtype,
            matrix_dtype=self.matrix_dtype,
        )
        record_key = (
            step,
            maximum_key.target_checkpoint_hash,
            plan.sample_size,
        )
        existing = self._records.get(record_key)
        if existing is not None:
            return existing
        if self.config.schema_version >= 8:
            estimate = adaptive_reference_fisher(
                model,
                self.dataset,
                plan,
                mnist_nll,
                layout,
                chunk_size=self.config.reference.chunk_size,
                minimum_chunks=self.config.reference.convergence_min_chunks,
                sigma=self.config.reference.convergence_sigma,
                relative_epsilon=(
                    self.config.reference.convergence_relative_epsilon
                ),
                absolute_epsilon=(
                    self.config.reference.convergence_absolute_epsilon
                ),
                device=self.device,
                derivative_dtype=self.derivative_dtype,
                matrix_dtype=self.matrix_dtype,
                strategy="vmap",
                num_workers=self.config.initialization.num_workers,
            )
            plan = plan.prefix(estimate.sample_count)
            key = reference_cache_key(
                model,
                layout,
                plan,
                derivative_dtype=self.derivative_dtype,
                matrix_dtype=self.matrix_dtype,
            )
            if self.store.exists(key):
                cached = self.store.load(key, layout)
                if cached.convergence == estimate.convergence:
                    estimate = cached
            else:
                self.store.save(key, estimate)
        else:
            key = maximum_key
            if self.store.exists(key):
                estimate = self.store.load(key, layout)
            else:
                estimate = chunked_reference_fisher(
                    model,
                    self.dataset,
                    plan,
                    mnist_nll,
                    layout,
                    chunk_size=min(
                        self.config.reference.chunk_size,
                        plan.sample_size,
                    ),
                    device=self.device,
                    derivative_dtype=self.derivative_dtype,
                    matrix_dtype=self.matrix_dtype,
                    strategy="vmap",
                    num_workers=self.config.initialization.num_workers,
                )
                self.store.save(key, estimate)
        record = _ReferenceRecord(
            matrix=estimate.matrix,
            estimate=estimate,
            cache_digest=key.digest,
            plan=plan,
        )
        self._records[record_key] = record
        return record


def _calculate_online_statistics(
    model: nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    direction: Tensor,
    *,
    matrix_dtype: torch.dtype,
    include_hvp: bool,
    device: torch.device,
) -> OnlineStepStatistics:
    _synchronize(device)
    started = time.perf_counter()
    derivatives = per_sample_derivatives(
        model,
        inputs,
        targets,
        mnist_nll,
        layout,
        direction=direction if include_hvp else None,
        strategy="vmap",
    )
    gradients = derivatives.gradients.to(dtype=matrix_dtype)
    if include_hvp:
        estimate = dense_lfu_estimate(
            gradients,
            derivatives.hvps.to(dtype=matrix_dtype),
            direction.to(dtype=matrix_dtype),
        )
        hvp_count = inputs.shape[0]
    else:
        zero = torch.zeros(
            layout.total_numel,
            layout.total_numel,
            dtype=matrix_dtype,
            device=device,
        )
        estimate = LFUBatchEstimate(
            fisher=empirical_fisher(gradients),
            amari_chentsov=zero,
            residual=zero.clone(),
        )
        hvp_count = 0
    _synchronize(device)
    return OnlineStepStatistics(
        estimate=LFUBatchEstimate(
            fisher=estimate.fisher.detach().cpu(),
            amari_chentsov=estimate.amari_chentsov.detach().cpu(),
            residual=estimate.residual.detach().cpu(),
        ),
        score_gradient_count=inputs.shape[0],
        hvp_count=hvp_count,
        elapsed_seconds=time.perf_counter() - started,
    )


def _all_online_statistics(
    trajectory: FixedTrajectory,
    template: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> list[OnlineStepStatistics]:
    results = []
    for step, parameter_vector in enumerate(trajectory.parameters):
        model, layout = _clone_at_vector(
            template,
            parameter_vector,
            device=device,
            dtype=derivative_dtype,
        )
        inputs, targets = _materialize_batch(
            dataset,
            trajectory.observation_indices[step],
            device=device,
            dtype=derivative_dtype,
        )
        direction = (
            torch.zeros(
                layout.total_numel,
                device=device,
                dtype=derivative_dtype,
            )
            if step == 0
            else trajectory.displacements[step - 1].to(
                device=device,
                dtype=derivative_dtype,
            )
        )
        results.append(
            _calculate_online_statistics(
                model,
                layout,
                inputs,
                targets,
                direction,
                matrix_dtype=matrix_dtype,
                include_hvp=step > 0,
                device=device,
            )
        )
    return results


def _statistics_to_device(
    statistics: Sequence[OnlineStepStatistics],
    device: torch.device,
    dtype: torch.dtype,
) -> list[OnlineStepStatistics]:
    return [
        OnlineStepStatistics(
            estimate=LFUBatchEstimate(
                fisher=row.estimate.fisher.to(device=device, dtype=dtype),
                amari_chentsov=row.estimate.amari_chentsov.to(
                    device=device,
                    dtype=dtype,
                ),
                residual=row.estimate.residual.to(device=device, dtype=dtype),
            ),
            score_gradient_count=row.score_gradient_count,
            hvp_count=row.hvp_count,
            elapsed_seconds=row.elapsed_seconds,
        )
        for row in statistics
    ]


def _condition_ranking(
    conditions: dict[str, DenseConditionResult],
) -> list[str]:
    averages = {
        method: sum(
            row["relative_frobenius_error"]
            for row in result.metrics[1:]
        )
        / max(len(result.metrics) - 1, 1)
        for method, result in conditions.items()
    }
    return sorted(averages, key=lambda method: (averages[method], method))


def _statistics_disagreement(
    first: Sequence[OnlineStepStatistics],
    second: Sequence[OnlineStepStatistics],
) -> dict[str, Any]:
    rows = []
    maximum = 0.0
    for step, (left, right) in enumerate(zip(first, second, strict=True)):
        values = {
            name: relative_frobenius_error(
                getattr(right.estimate, name),
                getattr(left.estimate, name),
            )
            for name in ("fisher", "amari_chentsov", "residual", "full")
        }
        maximum = max(maximum, *values.values())
        rows.append({"step": step, **values})
    return {"maximum_relative_error": maximum, "steps": rows}


def _cpu_gpu_prefix_audit(
    trajectory: FixedTrajectory,
    template: nn.Module,
    dataset: Dataset,
    gpu_statistics: Sequence[OnlineStepStatistics],
    *,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> dict[str, Any]:
    step = 1 if len(trajectory.p_values) > 1 else 0
    model, layout = _clone_at_vector(
        template,
        trajectory.parameters[step],
        device=torch.device("cpu"),
        dtype=derivative_dtype,
    )
    inputs, targets = _materialize_batch(
        dataset,
        trajectory.observation_indices[step],
        device=torch.device("cpu"),
        dtype=derivative_dtype,
    )
    direction = (
        torch.zeros(layout.total_numel, dtype=derivative_dtype)
        if step == 0
        else trajectory.displacements[step - 1].to(dtype=derivative_dtype)
    )
    cpu = _calculate_online_statistics(
        model,
        layout,
        inputs,
        targets,
        direction,
        matrix_dtype=matrix_dtype,
        include_hvp=step > 0,
        device=torch.device("cpu"),
    )
    gpu = gpu_statistics[step]
    errors = {
        name: relative_frobenius_error(
            getattr(gpu.estimate, name),
            getattr(cpu.estimate, name),
        )
        for name in ("fisher", "amari_chentsov", "residual", "full")
    }
    return {
        "step": step,
        "p": trajectory.p_values[step],
        "maximum_relative_error": max(errors.values()),
        **errors,
    }


def _cuda_audit(
    config: ExperimentConfig,
    trajectory: FixedTrajectory,
    template: nn.Module,
    dataset: Dataset,
    oracle: _ReferenceOracle,
    references: Sequence[_ReferenceRecord],
    statistics: Sequence[OnlineStepStatistics],
    conditions: dict[str, DenseConditionResult],
    *,
    device: torch.device,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> tuple[dict[str, Any] | None, list[OnlineStepStatistics] | None]:
    if device.type != "cuda":
        return None, None

    repeated = _all_online_statistics(
        trajectory,
        template,
        dataset,
        device=device,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
    )
    repeatability = _statistics_disagreement(statistics, repeated)
    cpu_gpu = _cpu_gpu_prefix_audit(
        trajectory,
        template,
        dataset,
        statistics,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
    )
    audit_step = len(trajectory.p_values) // 2
    full_record = references[audit_step]
    half_size = max(1, full_record.plan.sample_size // 2)
    half_record = oracle.estimate(
        audit_step,
        trajectory.p_values[audit_step],
        trajectory.parameters[audit_step],
        sample_size=half_size,
    )
    reference_disagreement = relative_frobenius_error(
        half_record.matrix,
        full_record.matrix,
    )
    tolerance = 0.1 * reference_disagreement

    replay_repeated = replay_dense_conditions(
        references[0].matrix.to(device=device, dtype=matrix_dtype),
        [record.matrix.to(device=device, dtype=matrix_dtype) for record in references],
        _statistics_to_device(repeated, device, matrix_dtype),
        lagged_directions(
            trajectory,
            device=device,
            dtype=matrix_dtype,
        ),
        trajectory.p_values,
        ema_gain=config.estimator.ema_gain,
        fresh_fisher_cadence=config.estimator.fresh_fisher_cadence,
        ridge_half_life_steps=config.estimator.ridge_half_life_steps,
        ridge_amplitude_epsilon=config.estimator.ridge_amplitude_epsilon,
        ridge_coherence_threshold=(
            config.estimator.ridge_coherence_threshold
        ),
    )
    original_ranking = _condition_ranking(conditions)
    repeated_ranking = _condition_ranking(replay_repeated)
    repeatability_accepted = (
        repeatability["maximum_relative_error"] <= tolerance
    )
    cpu_gpu_accepted = cpu_gpu["maximum_relative_error"] <= tolerance
    ranking_stable = original_ranking == repeated_ranking
    return (
        {
            "adaptive_avg_pool2d_backward_cuda_is_nondeterministic": True,
            "reference_audit_step": audit_step,
            "reference_half_sample_size": half_size,
            "reference_full_sample_size": full_record.plan.sample_size,
            "reference_half_cache_digest": half_record.cache_digest,
            "reference_full_cache_digest": full_record.cache_digest,
            "reference_monte_carlo_disagreement": reference_disagreement,
            "acceptance_tolerance": tolerance,
            "gpu_repeatability": repeatability,
            "cpu_gpu_prefix": cpu_gpu,
            "original_condition_ranking": original_ranking,
            "repeated_condition_ranking": repeated_ranking,
            "condition_ranking_stable": ranking_stable,
            "repeatability_accepted": repeatability_accepted,
            "cpu_gpu_quality_accepted": cpu_gpu_accepted,
            "accepted": (
                repeatability_accepted
                and cpu_gpu_accepted
                and ranking_stable
            ),
        },
        repeated,
    )


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _checkpoint_artifact(
    config: ExperimentConfig,
    references: Sequence[_ReferenceRecord],
    statistics: Sequence[OnlineStepStatistics],
    conditions: dict[str, DenseConditionResult],
) -> dict[str, Any]:
    cadence = config.estimator.fresh_fisher_cadence
    indices = sorted(
        {
            0,
            len(references) - 1,
            *range(cadence, len(references), cadence),
        }
    )
    ridge_result = conditions.get("ridge_full_lfu")
    ridge_states = None if ridge_result is None else ridge_result.ridge_states
    return {
        "schema_version": PHASE4_MATRIX_SCHEMA_VERSION,
        "checkpoint_indices": indices,
        "checkpoints": {
            str(step): {
                "reference": references[step].matrix,
                "online_fisher": statistics[step].estimate.fisher,
                "amari_chentsov": statistics[step].estimate.amari_chentsov,
                "residual": statistics[step].estimate.residual,
                "directional_ridge_state": (
                    None
                    if ridge_states is None
                    else ridge_states[step]
                ),
                "conditions": {
                    method: {
                        "estimate": result.estimates[step],
                        "prediction": result.predictions[step],
                        "candidate": result.candidates[step],
                    }
                    for method, result in conditions.items()
                },
            }
            for step in indices
        },
    }


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    _validate_phase4_config(config)
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
    train_dataset, _ = load_mnist_datasets(data_root, download=False)
    train_targets = dataset_targets(train_dataset)
    loaded = load_replica_bundle_for_config(
        replica_root,
        config,
        device=device,
    )
    initial_template = copy.deepcopy(loaded.model).cpu()
    store = ReferenceFisherStore(reference_root)
    oracle = _ReferenceOracle(
        config,
        initial_template,
        train_dataset,
        train_targets,
        loaded.partitions,
        store,
        device=device,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def driver_fisher_provider(
        step: int,
        p_value: float,
        model: nn.Module,
        layout: ParameterLayout,
    ) -> tuple[Tensor, dict[str, Any]]:
        vector = layout.flatten_module(model, detach=True)
        record = oracle.estimate(step, p_value, vector)
        projected = project_psd_frobenius(
            record.matrix.to(device=device, dtype=matrix_dtype)
        )
        return (
            projected.projected.to(dtype=training_dtype),
            {
                "cache_digest": record.cache_digest,
                "plan_hash": record.plan.content_hash,
                "sample_size": record.plan.sample_size,
                "minimum_eigenvalue": (
                    projected.diagnostics.minimum_eigenvalue
                ),
                "projection_distance_fro": (
                    projected.diagnostics.projection_distance_fro
                ),
            },
        )

    started = time.perf_counter()
    trajectory = generate_fixed_trajectory(
        loaded.model,
        loaded.layout,
        train_dataset,
        loaded.stream_plan,
        config.optimizer,
        fisher_cadence=config.estimator.fresh_fisher_cadence,
        fisher_provider=driver_fisher_provider,
        device=device,
        dtype=training_dtype,
    )
    trajectory_elapsed = time.perf_counter() - started

    references = [
        oracle.estimate(step, p_value, trajectory.parameters[step])
        for step, p_value in enumerate(trajectory.p_values)
    ]
    statistics = _all_online_statistics(
        trajectory,
        initial_template,
        train_dataset,
        device=device,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
    )
    replay_references = [
        record.matrix.to(device=device, dtype=matrix_dtype)
        for record in references
    ]
    replay_statistics = _statistics_to_device(
        statistics,
        device,
        matrix_dtype,
    )
    directions = lagged_directions(
        trajectory,
        device=device,
        dtype=matrix_dtype,
    )
    replay_started = time.perf_counter()
    conditions = replay_dense_conditions(
        replay_references[0],
        replay_references,
        replay_statistics,
        directions,
        trajectory.p_values,
        ema_gain=config.estimator.ema_gain,
        fresh_fisher_cadence=config.estimator.fresh_fisher_cadence,
        ridge_half_life_steps=config.estimator.ridge_half_life_steps,
        ridge_amplitude_epsilon=config.estimator.ridge_amplitude_epsilon,
        ridge_coherence_threshold=(
            config.estimator.ridge_coherence_threshold
        ),
    )
    _synchronize(device)
    replay_elapsed = time.perf_counter() - replay_started
    common_rows = common_step_metrics(
        replay_references,
        replay_statistics,
        directions,
        trajectory.p_values,
    )
    cuda_audit, repeated_statistics = _cuda_audit(
        config,
        trajectory,
        initial_template,
        train_dataset,
        oracle,
        references,
        statistics,
        conditions,
        device=device,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
    )
    methods = tuple(conditions)

    metrics = {
        "phase4_metric_schema_version": PHASE4_METRIC_SCHEMA_VERSION,
        "replica_bundle_id": loaded.metadata["bundle_id"],
        "trajectory_hash": trajectory.content_hash,
        "device": str(device),
        "training_dtype": str(training_dtype),
        "derivative_dtype": str(derivative_dtype),
        "matrix_dtype": str(matrix_dtype),
        "methods": list(methods),
        "ema_gain": config.estimator.ema_gain,
        "fresh_fisher_cadence": config.estimator.fresh_fisher_cadence,
        "directional_ridge": {
            "enabled": config.estimator.ridge_half_life_steps is not None,
            "half_life_steps": config.estimator.ridge_half_life_steps,
            "amplitude_epsilon": (
                config.estimator.ridge_amplitude_epsilon
            ),
            "coherence_threshold": (
                config.estimator.ridge_coherence_threshold
            ),
        },
        "driver": {
            "kind": "oracle_assisted_fixed_trajectory",
            "treatment_independent": True,
            "ewc_anchor": "parameter_at_start_of_step",
            "reference_fisher_cadence": (
                config.estimator.fresh_fisher_cadence
            ),
            "optimizer": dataclasses.asdict(config.optimizer),
            "controller_policy": config.controller.policy,
            "trajectory_elapsed_seconds": trajectory_elapsed,
            "steps": list(trajectory.driver_steps),
        },
        "references": [
            record.metrics_mapping(step)
            for step, record in enumerate(references)
        ],
        "common_steps": common_rows,
        "condition_steps": [
            row
            for method in methods
            for row in conditions[method].metrics
        ],
        "condition_ranking": _condition_ranking(conditions),
        "replay_elapsed_seconds": replay_elapsed,
        "checkpoint_policy": (
            "initial, final, and periodic-fresh replacement indices"
        ),
        "total_score_gradient_count": sum(
            row.score_gradient_count for row in statistics
        ),
        "total_hvp_count": sum(row.hvp_count for row in statistics),
        "cuda_repeatability_audit": cuda_audit,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }
    reference_plan_artifact = {
        "schema_version": PHASE4_REFERENCE_PLAN_SCHEMA_VERSION,
        "plans": {
            str(step): record.plan.to_mapping()
            for step, record in enumerate(references)
        },
    }
    checkpoint_artifact = _checkpoint_artifact(
        config,
        references,
        statistics,
        conditions,
    )
    if repeated_statistics is not None:
        checkpoint_artifact["cuda_repeat_statistics"] = {
            str(step): {
                "fisher": row.estimate.fisher,
                "amari_chentsov": row.estimate.amari_chentsov,
                "residual": row.estimate.residual,
            }
            for step, row in enumerate(repeated_statistics)
            if step in checkpoint_artifact["checkpoint_indices"]
        }

    trajectory_path = session.write_torch(
        "phase4_trajectory.pt",
        _cpu_tree(trajectory.artifact_mapping()),
    )
    checkpoint_path = session.write_torch(
        "phase4_checkpoints.pt",
        checkpoint_artifact,
    )
    reference_plan_path = session.write_torch(
        "phase4_reference_plans.pt",
        reference_plan_artifact,
    )
    metrics["artifact_files_bytes"] = {
        "phase4_trajectory.pt": trajectory_path.stat().st_size,
        "phase4_checkpoints.pt": checkpoint_path.stat().st_size,
        "phase4_reference_plans.pt": reference_plan_path.stat().st_size,
    }
    metrics["artifact_payload_bytes_before_metrics"] = sum(
        metrics["artifact_files_bytes"].values()
    )
    session.write_json("phase4_metrics.json", metrics)
    destination = session.complete(
        [
            "phase4_metrics.json",
            "phase4_trajectory.pt",
            "phase4_checkpoints.pt",
            "phase4_reference_plans.pt",
        ]
    )
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "path": str(destination),
                "trajectory_hash": trajectory.content_hash,
                "steps": len(trajectory.p_values),
                "methods": list(methods),
                "cuda_audit_accepted": (
                    None if cuda_audit is None else cuda_audit["accepted"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
