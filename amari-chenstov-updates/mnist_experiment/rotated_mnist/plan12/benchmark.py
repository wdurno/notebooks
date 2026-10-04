"""Measured Plan 12 Phase 0 cost benchmark and compute projection."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch

from ..artifacts import _read_json
from ..transform import rotate_mnist_batch
from .artifacts import UnitStore
from .assets import ASSET_REQUIRED, ensure_replica_assets, runtime
from .local_response import _evaluate, _fit_once, local_conditions
from .trajectory import TRAJECTORY_REQUIRED, TrajectoryCondition, run_trajectory


LOCAL_BENCHMARK_REQUIRED = ("summary.json",)
PROJECTION_REQUIRED = ("projection.json",)


def benchmark_condition() -> TrajectoryCondition:
    return TrajectoryCondition("gauge_no_ridge", chart=True)


def run_benchmark_trajectory(
    store: UnitStore,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    return run_trajectory(
        store,
        "phase0_benchmark",
        1,
        "linear",
        benchmark_condition(),
        data_root=data_root,
        resume=resume,
    )


def run_local_benchmark(
    store: UnitStore,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    asset_path = ensure_replica_assets(
        store,
        "phase0_benchmark",
        1,
        data_root=data_root,
        resume=True,
    )
    conditions = local_conditions(store.study.ridge_scale_ratios)
    unit = store.unit(
        "phase0",
        "local_benchmark",
        1,
        detail={"conditions": [condition.mapping() for condition in conditions]},
    )
    session = store.begin(unit, LOCAL_BENCHMARK_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, LOCAL_BENCHMARK_REQUIRED)
        assert completed is not None
        return completed

    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    config = store.study.protocol_for_replica("phase0_benchmark", 1)
    device, training_dtype, _ = runtime(store.study)
    evaluation_count = min(256 if store.study.smoke else 2048, assets["base_inputs"].shape[0])
    evaluation_inputs = assets["base_inputs"][:evaluation_count]
    evaluation_targets = assets["base_targets"][:evaluation_count]
    batch = assets["streams"]["linear"]
    anchor: dict[str, Any] = {
        "anchor_index": 1,
        "chart_state": assets["chart_initial_state"],
        "chart_parameter": assets["chart_parameter_layout"],
        "chart_fisher": assets["chart_initial_fisher"],
        "chart_dense_archive": assets["chart_initial_dense_fisher"],
        "raw_state": assets["raw_initial_state"],
        "raw_parameter": assets["raw_parameter_layout"],
        "raw_fisher": assets["raw_initial_fisher"],
        "raw_dense_archive": assets["raw_initial_dense_fisher"],
        "samples": {
            "nine_prevalence": assets["nine_prevalence"],
            "current_evaluation_inputs": evaluation_inputs,
            "evaluation_targets": evaluation_targets,
            "retention_000_inputs": evaluation_inputs,
            "retention_015_inputs": rotate_mnist_batch(evaluation_inputs, 15.0, config.rotation),
            "retention_030_inputs": rotate_mnist_batch(evaluation_inputs, 30.0, config.rotation),
        },
    }
    # Parameter vectors, unlike layout metadata, are needed as penalty anchors.
    from .gauge import build_gauge_fixed_model
    from src.mnist_model import build_canonical_model

    chart_model, chart_layout = build_gauge_fixed_model(
        config.replica_seed, device=device, dtype=training_dtype
    )
    chart_model.load_state_dict(assets["chart_initial_state"])
    raw_model, raw_layout = build_canonical_model(
        config.replica_seed, device=device, dtype=training_dtype
    )
    raw_model.load_state_dict(assets["raw_initial_state"])
    anchor["chart_parameter"] = chart_layout.flatten_module(chart_model, detach=True).cpu()
    anchor["raw_parameter"] = raw_layout.flatten_module(raw_model, detach=True).cpu()

    rows = []
    inputs = batch["inputs"][0]
    targets = batch["targets"][0]
    for condition in conditions:
        started = time.perf_counter()
        model, _, health, kappa, _ = _fit_once(
            store,
            anchor,
            condition,
            inputs,
            targets,
        )
        metrics, _ = _evaluate(model, anchor, device=device, dtype=training_dtype)
        rows.append(
            {
                "condition": condition.mapping(),
                "kappa": kappa,
                "wall_time_seconds": time.perf_counter() - started,
                "fit_health": health,
                "current_nll": metrics["current_nll"],
            }
        )
    session.write_json(
        "summary.json",
        {
            "conditions": rows,
            "mean_seconds": statistics.fmean(row["wall_time_seconds"] for row in rows),
            "median_seconds": statistics.median(row["wall_time_seconds"] for row in rows),
            "total_seconds": sum(row["wall_time_seconds"] for row in rows),
        },
    )
    return store.finish(session, LOCAL_BENCHMARK_REQUIRED)


def run_compute_projection(store: UnitStore, *, resume: bool) -> Path:
    condition = benchmark_condition()
    trajectory_unit = store.unit(
        "phase0_benchmark",
        "trajectory",
        1,
        schedule="linear",
        condition=condition.name,
        detail=condition.mapping(),
    )
    trajectory_path = store.completed(trajectory_unit, TRAJECTORY_REQUIRED)
    local_unit = store.unit(
        "phase0",
        "local_benchmark",
        1,
        detail={"conditions": [item.mapping() for item in local_conditions(store.study.ridge_scale_ratios)]},
    )
    local_path = store.completed(local_unit, LOCAL_BENCHMARK_REQUIRED)
    asset_unit = store.unit("phase0_benchmark", "assets", 1)
    asset_path = store.completed(asset_unit, ASSET_REQUIRED)
    if trajectory_path is None or local_path is None or asset_path is None:
        raise RuntimeError("Phase 0 cost projection requires completed benchmark units")
    unit = store.unit("phase0", "compute_projection", 1)
    session = store.begin(unit, PROJECTION_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, PROJECTION_REQUIRED)
        assert completed is not None
        return completed

    trajectory = _read_json(trajectory_path / "summary.json")
    local = _read_json(local_path / "summary.json")
    asset = _read_json(asset_path / "summary.json")
    trajectory_seconds = float(trajectory["total_wall_time_seconds"])
    branch_seconds = float(local["median_seconds"])
    asset_seconds = float(asset["wall_time_seconds"])
    condition_count = len(local["conditions"])
    study = store.study
    phase1_branches = 2 * study.anchors_per_schedule * study.local_batches_per_anchor * condition_count
    estimates = {
        "phase1_hours": (phase1_branches * branch_seconds + 3600.0) / 3600.0,
        "phase2_hours": (
            study.phase2_replicas * asset_seconds
            + study.phase2_replicas * 2 * 5 * trajectory_seconds
        ) / 3600.0,
        "phase3_hours": 4.5 if not study.smoke else 0.05,
        "phase4_hours": (
            study.phase4_replicas * asset_seconds
            + study.phase4_replicas * 2 * 4 * trajectory_seconds
        ) / 3600.0,
    }
    estimates["remaining_total_hours"] = sum(estimates.values())
    session.write_json(
        "projection.json",
        {
            "measured": {
                "asset_seconds": asset_seconds,
                "trajectory_seconds": trajectory_seconds,
                "median_local_branch_seconds": branch_seconds,
                "local_condition_count": condition_count,
                "benchmark_artifact_bytes": sum(
                    path.stat().st_size
                    for root in (trajectory_path, local_path, asset_path)
                    for path in root.iterdir()
                    if path.is_file()
                ),
            },
            "projected": estimates,
            "notes": [
                "Phase 1 includes a one-hour allowance for anchors, targets, references, and analysis.",
                "Phase 3 retains the preregistered midpoint because resampling cost is not represented by a local branch.",
                "Projections are throughput estimates, not changes to frozen unit counts.",
            ],
        },
    )
    return store.finish(session, PROJECTION_REQUIRED)
