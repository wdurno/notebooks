"""Immutable paired artifacts for the Plan 5 Phase 4 memory screen."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

import torch

from src.seeding import SEED_SCHEMA_VERSION, derive_seed_map

from .artifacts import (
    MANIFEST_SCHEMA_VERSION,
    RotatedArtifactError,
    RotatedCompletedRunError,
    RotatedIncompleteRunError,
    RotatedRunSession,
    _read_json,
    _utc_now,
    _write_json,
    runtime_metadata,
)
from .data import RotatedPartitions, RotatedStreamPlan
from .phase4_config import (
    PHASE4_ARTIFACT_SCHEMA_VERSION,
    PHASE4_CONDITIONS,
    PHASE4_METRIC_SCHEMA_VERSION,
    RotatedPhase4Config,
)


PHASE4_REQUIRED_ARTIFACTS = (
    "partitions.json",
    "stream_plan.json",
    "stream_tensors.pt",
    "initialization_metrics.json",
    "initial_fisher.pt",
    "evaluation_panel.json",
    "trajectory_metrics.json",
    "trajectories.pt",
    "model_states.pt",
    "operational_checks.json",
    "run_summary.json",
)
PHASE4_SEED_COMPONENTS = (
    "plan5_model_initialization",
    "plan5_data_partition",
    "plan5_initialization_loader",
    "plan5_online_stream",
    "plan5_reference_stream",
    "plan5_evaluation_panel",
    "plan5_optimizer",
    "plan5_numerical_randomization",
    "plan5_phase4_initial_lanczos",
    "plan5_phase4_ewc_lanczos",
    "plan5_phase4_hybrid_archive_lanczos",
)


class RotatedPhase4RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"

    def begin(
        self,
        config: RotatedPhase4Config,
        repo_root: str | Path,
        *,
        resume: bool = False,
    ) -> RotatedRunSession:
        config.validate()
        final_path = self.root / config.run_id
        working_path = self.incomplete_root / config.run_id
        if final_path.exists():
            if (final_path / "COMPLETED").is_file():
                raise RotatedCompletedRunError(
                    f"Phase 4 run already completed: {config.run_id}"
                )
            raise RotatedArtifactError(
                f"final Phase 4 run exists without COMPLETED: {final_path}"
            )
        if working_path.exists():
            if not resume:
                raise RotatedIncompleteRunError(
                    f"incomplete Phase 4 run exists; pass --resume: {config.run_id}"
                )
            if _read_json(working_path / "config.json") != config.to_mapping():
                raise RotatedArtifactError("incomplete Phase 4 config differs")
            manifest = _read_json(working_path / "manifest.json")
            if manifest.get("status") != "incomplete":
                raise RotatedArtifactError("incomplete Phase 4 manifest is invalid")
            return RotatedRunSession(config.run_id, working_path, final_path)

        working_path.mkdir(parents=True, exist_ok=False)
        _write_json(working_path / "config.json", config.to_mapping())
        _write_json(
            working_path / "manifest.json",
            {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "artifact_schema_version": config.artifact_schema_version,
                "metric_schema_version": config.metric_schema_version,
                "seed_schema_version": SEED_SCHEMA_VERSION,
                "run_kind": "phase4_repeated_path_memory_screen",
                "run_id": config.run_id,
                "experiment": config.experiment,
                "replica_id": config.replica_id,
                "config_hash": config.config_hash,
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
                "seeds": derive_seed_map(
                    config.replica_seed, PHASE4_SEED_COMPONENTS
                ),
                "runtime": runtime_metadata(Path(repo_root), config),
            },
        )
        return RotatedRunSession(config.run_id, working_path, final_path)


@dataclasses.dataclass(frozen=True)
class LoadedRotatedPhase4Run:
    path: Path
    config: RotatedPhase4Config
    manifest: dict[str, Any]
    partitions: RotatedPartitions
    stream_plan: RotatedStreamPlan
    initialization_metrics: dict[str, Any]
    evaluation_panel: dict[str, Any]
    trajectory_metrics: dict[str, tuple[dict[str, Any], ...]]
    operational_checks: dict[str, Any]
    run_summary: dict[str, Any]


def load_completed_phase4_run(path: str | Path) -> LoadedRotatedPhase4Run:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(f"Phase 4 run is incomplete: {run_path}")
    for name in PHASE4_REQUIRED_ARTIFACTS:
        if not (run_path / name).is_file():
            raise RotatedArtifactError(f"completed Phase 4 run is missing {name}")
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("Phase 4 config artifact must be an object")
    config = RotatedPhase4Config.from_mapping(config_value)
    manifest = _read_json(run_path / "manifest.json")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_schema_version")
        != PHASE4_ARTIFACT_SCHEMA_VERSION
        or manifest.get("metric_schema_version") != PHASE4_METRIC_SCHEMA_VERSION
        or manifest.get("run_kind") != "phase4_repeated_path_memory_screen"
        or manifest.get("status") != "completed"
        or manifest.get("config_hash") != config.config_hash
        or manifest.get("run_id") != config.run_id
        or run_path.name != config.run_id
    ):
        raise RotatedArtifactError("Phase 4 manifest is incompatible")
    partitions = RotatedPartitions.from_mapping(
        _read_json(run_path / "partitions.json")
    )
    stream_plan = RotatedStreamPlan.from_mapping(
        _read_json(run_path / "stream_plan.json")
    )
    if stream_plan.partition_hash != partitions.content_hash:
        raise RotatedArtifactError("Phase 4 stream and partition hashes differ")
    metrics = _read_json(run_path / "trajectory_metrics.json")
    if not isinstance(metrics, Mapping) or set(metrics) != set(PHASE4_CONDITIONS):
        raise RotatedArtifactError("Phase 4 treatment metrics are incompatible")
    trajectory_metrics: dict[str, tuple[dict[str, Any], ...]] = {}
    for condition in PHASE4_CONDITIONS:
        rows = metrics[condition]
        if not isinstance(rows, list) or len(rows) != stream_plan.schedule.num_points:
            raise RotatedArtifactError("Phase 4 trajectory length is incompatible")
        for step, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or row.get("step") != step
                or row.get("condition") != condition
                or row.get("angle_degrees")
                != stream_plan.schedule.angles_degrees[step]
                or "current_environment_accuracy" not in row
                or "current_per_class_recall" not in row
                or "current_confusion_matrix" not in row
            ):
                raise RotatedArtifactError("Phase 4 metric contract failed")
        trajectory_metrics[condition] = tuple(rows)
    initialization = _read_json(run_path / "initialization_metrics.json")
    evaluation_panel = _read_json(run_path / "evaluation_panel.json")
    checks = _read_json(run_path / "operational_checks.json")
    summary = _read_json(run_path / "run_summary.json")
    if not all(
        isinstance(value, dict)
        for value in (initialization, evaluation_panel, checks, summary)
    ):
        raise RotatedArtifactError("Phase 4 scalar artifacts are incompatible")
    if (
        summary.get("schedule_hash") != stream_plan.schedule.content_hash
        or summary.get("stream_plan_hash") != stream_plan.content_hash
        or summary.get("partition_hash") != partitions.content_hash
        or tuple(summary.get("conditions", ())) != PHASE4_CONDITIONS
        or evaluation_panel.get("evaluation_indices_hash")
        != summary.get("evaluation_indices_hash")
    ):
        raise RotatedArtifactError("Phase 4 summary provenance is incompatible")
    if checks.get("all_finite") is not True:
        raise RotatedArtifactError("Phase 4 completed with nonfinite diagnostics")
    try:
        fisher = torch.load(
            run_path / "initial_fisher.pt", map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError) as exc:
        raise RotatedArtifactError(f"could not load initial Fisher: {exc}") from exc
    if not isinstance(fisher, dict) or set(fisher) != {
        "dense",
        "representation",
        "sample_indices",
    }:
        raise RotatedArtifactError("Phase 4 initial Fisher artifact is invalid")
    dense = fisher["dense"]
    if (
        dense.ndim != 2
        or dense.shape[0] != dense.shape[1]
        or not torch.isfinite(dense).all()
    ):
        raise RotatedArtifactError("Phase 4 dense initial Fisher is invalid")
    return LoadedRotatedPhase4Run(
        path=run_path,
        config=config,
        manifest=manifest,
        partitions=partitions,
        stream_plan=stream_plan,
        initialization_metrics=initialization,
        evaluation_panel=evaluation_panel,
        trajectory_metrics=trajectory_metrics,
        operational_checks=checks,
        run_summary=summary,
    )
