"""Run the immutable Phase 7 fixed-trajectory representation sweep."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from mnist_experiment.run_coupled import _tensor_hash
from mnist_experiment.run_experiment import (
    _ReferenceOracle,
    _all_online_statistics,
    _cpu_tree,
    _statistics_to_device,
    _synchronize,
)
from src.artifacts import RunStore
from src.config import ExperimentConfig, load_config
from src.fixed_trajectory import (
    generate_fixed_trajectory,
    lagged_directions,
    replay_dense_conditions,
)
from src.initialization import load_replica_bundle_for_config
from src.lanczos_wrapper import (
    LEGACY_LANCZOS_SHA256,
    legacy_lanczos_source_hash,
)
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import configure_torch_runtime, resolve_device, resolve_dtype
from src.parameters import ParameterLayout
from src.reference import ReferenceFisherStore
from src.representations import project_psd_frobenius
from src.seeding import derive_component_seed
from src.structured_trajectory import (
    DiagonalFisherTracker,
    LowRankDiagonalFisherTracker,
    structured_representation_metrics,
)

PHASE7_METRIC_SCHEMA_VERSION = 1
PHASE7_REPRESENTATION_SCHEMA_VERSION = 1
PHASE7_DENSE_CHECKPOINT_SCHEMA_VERSION = 1
PHASE7_REFERENCE_PLAN_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_phase7_config(config: ExperimentConfig) -> tuple[int, ...]:
    if config.schema_version != 6:
        raise ValueError("Phase 7 requires configuration schema version 6")
    if config.estimator.representation != "low_rank_diagonal":
        raise ValueError(
            "Phase 7 requires estimator.representation='low_rank_diagonal'"
        )
    if config.estimator.low_rank_grid is None:
        raise ValueError("Phase 7 requires estimator.low_rank_grid")
    if config.estimator.fresh_fisher_cadence is None:
        raise ValueError("Phase 7 requires estimator.fresh_fisher_cadence")
    if config.estimator.ridge_half_life_steps is None:
        raise ValueError("Phase 7 requires directional-ridge settings")
    if config.controller.policy != "uncontrolled":
        raise ValueError(
            "Phase 7 fixed trajectories require controller.policy='uncontrolled'"
        )
    if config.controller.fixed_pi != 1.0:
        raise ValueError(
            "Phase 7 fixed trajectories require controller.fixed_pi=1"
        )
    return tuple(config.estimator.low_rank_grid)


def _fixed_probes(
    parameter_count: int,
    *,
    count: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probes = torch.randn(
        parameter_count,
        count,
        generator=generator,
        dtype=torch.float64,
    )
    probes /= torch.linalg.vector_norm(probes, dim=0).clamp_min(
        torch.finfo(probes.dtype).eps
    )
    return probes.to(device=device, dtype=dtype)


def _prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _rank_name(rank: int) -> str:
    return "diagonal" if rank == 0 else f"low_rank_diagonal_r{rank}"


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    rank_grid = _validate_phase7_config(config)
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
    parameter_count = loaded.layout.total_numel
    if rank_grid[-1] > parameter_count:
        raise ValueError(
            f"largest rank {rank_grid[-1]} exceeds {parameter_count} parameters"
        )
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

    def driver_fisher_provider(
        step: int,
        p_value: float,
        model: nn.Module,
        layout: ParameterLayout,
    ) -> tuple[Tensor, dict[str, Any]]:
        record = oracle.estimate(
            step,
            p_value,
            layout.flatten_module(model, detach=True),
        )
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

    total_started = time.perf_counter()
    trajectory_started = time.perf_counter()
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
    trajectory_elapsed = time.perf_counter() - trajectory_started
    references = [
        oracle.estimate(step, p_value, trajectory.parameters[step])
        for step, p_value in enumerate(trajectory.p_values)
    ]
    statistics_cpu = _all_online_statistics(
        trajectory,
        initial_template,
        train_dataset,
        device=device,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
    )
    references_device = [
        record.matrix.to(device=device, dtype=matrix_dtype)
        for record in references
    ]
    statistics = _statistics_to_device(
        statistics_cpu,
        device,
        matrix_dtype,
    )
    directions = lagged_directions(
        trajectory,
        device=device,
        dtype=matrix_dtype,
    )
    dense_started = time.perf_counter()
    dense_conditions = replay_dense_conditions(
        references_device[0],
        references_device,
        statistics,
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
    dense_target = dense_conditions["ridge_full_lfu"]
    dense_elapsed = time.perf_counter() - dense_started

    tracker_arguments = {
        "initial_fisher": references_device[0],
        "ema_gain": config.estimator.ema_gain,
        "ridge_half_life_steps": config.estimator.ridge_half_life_steps,
        "ridge_amplitude_epsilon": (
            config.estimator.ridge_amplitude_epsilon
        ),
        "ridge_coherence_threshold": (
            config.estimator.ridge_coherence_threshold
        ),
    }
    diagonal = DiagonalFisherTracker(**tracker_arguments)
    rank_zero_audit = LowRankDiagonalFisherTracker(
        **tracker_arguments,
        rank=0,
    )
    rank_trackers = {
        rank: LowRankDiagonalFisherTracker(
            **tracker_arguments,
            rank=rank,
        )
        for rank in rank_grid
        if rank > 0
    }
    probe_seed = derive_component_seed(
        config.replica_seed,
        "phase7_fixed_probes",
    )
    probes = _fixed_probes(
        parameter_count,
        count=8,
        seed=probe_seed,
        device=device,
        dtype=matrix_dtype,
    )
    rows = []
    representation_steps: dict[str, dict[str, Any]] = {
        _rank_name(rank): {} for rank in rank_grid
    }
    status = {
        _rank_name(rank): {
            "requested_rank": rank,
            "status": "pending",
            "failure_step": None,
            "failure": None,
        }
        for rank in rank_grid
    }
    active_ranks = set(rank for rank in rank_grid if rank > 0)
    structured_started = time.perf_counter()

    for step, p_value in enumerate(trajectory.p_values):
        target = dense_target.estimates[step].to(
            device=device,
            dtype=matrix_dtype,
        )
        reference = references_device[step]
        direction = directions[step]
        update_started = time.perf_counter()
        diagonal_update = diagonal.update(step, statistics[step], direction)
        rank_zero_update = rank_zero_audit.update(
            step,
            statistics[step],
            direction,
            lanczos_seed=derive_component_seed(
                config.replica_seed,
                f"phase7_lanczos:rank=0:step={step}",
            ),
        )
        _synchronize(device)
        diagonal_elapsed = time.perf_counter() - update_started
        torch.testing.assert_close(
            rank_zero_update.representation.residual_diagonal,
            diagonal_update.representation.values,
            rtol=1e-10,
            atol=1e-12,
        )
        diagonal_metrics = structured_representation_metrics(
            diagonal_update.representation,
            target,
            reference,
            probes,
            direction,
        )
        diagonal_name = _rank_name(0)
        rows.append(
            {
                "condition": diagonal_name,
                "representation": "diagonal",
                "requested_rank": 0,
                "realized_rank": 0,
                "status": "completed",
                "step": step,
                "p": p_value,
                "update_elapsed_seconds": diagonal_elapsed,
                "lanczos_elapsed_seconds": 0.0,
                **diagonal_metrics,
                **_prefixed(
                    "projection",
                    dataclasses.asdict(diagonal_update.projection),
                ),
                **_prefixed("ridge", diagonal_update.ridge_metrics),
            }
        )
        representation_steps[diagonal_name][str(step)] = {
            "representation": (
                diagonal_update.representation.artifact_mapping()
            ),
            "correction_diagonal": diagonal_update.correction,
            "projection": dataclasses.asdict(diagonal_update.projection),
            "ridge": diagonal_update.ridge_metrics,
        }

        for rank in sorted(tuple(active_ranks)):
            name = _rank_name(rank)
            update_started = time.perf_counter()
            seed = derive_component_seed(
                config.replica_seed,
                f"phase7_lanczos:rank={rank}:step={step}",
            )
            try:
                update = rank_trackers[rank].update(
                    step,
                    statistics[step],
                    direction,
                    lanczos_seed=seed,
                )
                _synchronize(device)
                update_elapsed = time.perf_counter() - update_started
                representation_metrics = structured_representation_metrics(
                    update.representation,
                    target,
                    reference,
                    probes,
                    direction,
                )
            except (RuntimeError, ValueError) as exc:
                status[name].update(
                    {
                        "status": "failed",
                        "failure_step": step,
                        "failure": str(exc),
                    }
                )
                active_ranks.remove(rank)
                rows.append(
                    {
                        "condition": name,
                        "representation": "low_rank_diagonal",
                        "requested_rank": rank,
                        "realized_rank": None,
                        "status": "failed",
                        "step": step,
                        "p": p_value,
                        "failure": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "condition": name,
                    "representation": "low_rank_diagonal",
                    "requested_rank": rank,
                    "realized_rank": update.representation.rank,
                    "status": "completed",
                    "step": step,
                    "p": p_value,
                    "update_elapsed_seconds": update_elapsed,
                    "lanczos_elapsed_seconds": (
                        update.lanczos.elapsed_seconds
                    ),
                    "candidate_minimum_eigenvalue": (
                        update.candidate_minimum_eigenvalue
                    ),
                    "candidate_negative_eigenvalue_count": (
                        update.candidate_negative_eigenvalue_count
                    ),
                    **representation_metrics,
                    **_prefixed("lanczos", update.lanczos.mapping()),
                    **_prefixed("ridge", update.ridge_metrics),
                }
            )
            representation_steps[name][str(step)] = {
                "representation": update.representation.artifact_mapping(),
                "candidate_minimum_eigenvalue": (
                    update.candidate_minimum_eigenvalue
                ),
                "candidate_negative_eigenvalue_count": (
                    update.candidate_negative_eigenvalue_count
                ),
                "lanczos": update.lanczos.mapping(),
                "ridge": update.ridge_metrics,
            }

    status[diagonal_name]["status"] = "completed"
    for rank in active_ranks:
        status[_rank_name(rank)]["status"] = "completed"
    structured_elapsed = time.perf_counter() - structured_started
    _synchronize(device)

    checkpoint_indices = sorted(
        {0, len(trajectory.p_values) // 2, len(trajectory.p_values) - 1}
    )
    representation_artifact = {
        "schema_version": PHASE7_REPRESENTATION_SCHEMA_VERSION,
        "parameter_layout": loaded.layout.metadata(),
        "trajectory_hash": trajectory.content_hash,
        "rank_grid": list(rank_grid),
        "conditions": representation_steps,
    }
    dense_checkpoint_artifact = {
        "schema_version": PHASE7_DENSE_CHECKPOINT_SCHEMA_VERSION,
        "trajectory_hash": trajectory.content_hash,
        "checkpoint_indices": checkpoint_indices,
        "checkpoints": {
            str(step): {
                "reference": references_device[step],
                "dense_ridge_full_target": dense_target.estimates[step].to(
                    device=device,
                    dtype=matrix_dtype,
                ),
                "direct_fisher": statistics[step].estimate.fisher,
                "lagged_direction": directions[step],
            }
            for step in checkpoint_indices
        },
    }
    reference_plan_artifact = {
        "schema_version": PHASE7_REFERENCE_PLAN_SCHEMA_VERSION,
        "plans": {
            str(step): record.plan.to_mapping()
            for step, record in enumerate(references)
        },
    }
    metrics = {
        "phase7_metric_schema_version": PHASE7_METRIC_SCHEMA_VERSION,
        "run_kind": "fixed_rank_sweep",
        "replica_bundle_id": loaded.metadata["bundle_id"],
        "trajectory_hash": trajectory.content_hash,
        "parameter_count": parameter_count,
        "device": str(device),
        "training_dtype": str(training_dtype),
        "derivative_dtype": str(derivative_dtype),
        "matrix_dtype": str(matrix_dtype),
        "rank_grid": list(rank_grid),
        "condition_status": status,
        "dense_target": "ridge_full_lfu",
        "probe_count": probes.shape[1],
        "probe_seed": probe_seed,
        "probe_hash": _tensor_hash(probes),
        "legacy_lanczos_source_hash": legacy_lanczos_source_hash(),
        "expected_legacy_lanczos_source_hash": LEGACY_LANCZOS_SHA256,
        "structured_state": {
            "persistent": True,
            "dense_candidate_materialized_for_validation": True,
            "dense_candidate_excluded_from_representation_storage_bytes": True,
        },
        "directional_ridge": {
            "half_life_steps": config.estimator.ridge_half_life_steps,
            "amplitude_epsilon": (
                config.estimator.ridge_amplitude_epsilon
            ),
            "coherence_threshold": (
                config.estimator.ridge_coherence_threshold
            ),
        },
        "condition_steps": rows,
        "references": [
            record.metrics_mapping(step)
            for step, record in enumerate(references)
        ],
        "trajectory_elapsed_seconds": trajectory_elapsed,
        "dense_replay_elapsed_seconds": dense_elapsed,
        "structured_replay_elapsed_seconds": structured_elapsed,
        "total_elapsed_seconds": time.perf_counter() - total_started,
        "total_score_gradient_count": sum(
            row.score_gradient_count for row in statistics_cpu
        ),
        "total_hvp_count": sum(row.hvp_count for row in statistics_cpu),
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
        "phase7_trajectory.pt",
        _cpu_tree(trajectory.artifact_mapping()),
    )
    representation_path = session.write_torch(
        "phase7_representations.pt",
        _cpu_tree(representation_artifact),
    )
    dense_checkpoint_path = session.write_torch(
        "phase7_dense_checkpoints.pt",
        _cpu_tree(dense_checkpoint_artifact),
    )
    reference_plan_path = session.write_torch(
        "phase7_reference_plans.pt",
        reference_plan_artifact,
    )
    metrics["artifact_files_bytes"] = {
        "phase7_trajectory.pt": trajectory_path.stat().st_size,
        "phase7_representations.pt": representation_path.stat().st_size,
        "phase7_dense_checkpoints.pt": dense_checkpoint_path.stat().st_size,
        "phase7_reference_plans.pt": reference_plan_path.stat().st_size,
    }
    session.write_json("phase7_metrics.json", metrics)
    destination = session.complete(
        [
            "phase7_metrics.json",
            "phase7_trajectory.pt",
            "phase7_representations.pt",
            "phase7_dense_checkpoints.pt",
            "phase7_reference_plans.pt",
        ]
    )
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "path": str(destination),
                "trajectory_hash": trajectory.content_hash,
                "rank_grid": list(rank_grid),
                "condition_status": status,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
