"""Immutable paired artifacts for the Plan 5 Phase 3 transfer pilot."""

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
from .phase3_config import (
    PHASE3_ARTIFACT_SCHEMA_VERSION,
    PHASE3_CONDITIONS,
    PHASE3_METRIC_SCHEMA_VERSION,
    RotatedPhase3Config,
)


PHASE3_REQUIRED_ARTIFACTS = (
    "partitions.json",
    "stream_plan.json",
    "stream_tensors.pt",
    "initialization_metrics.json",
    "initial_fisher.pt",
    "trajectory_metrics.json",
    "trajectories.pt",
    "model_states.pt",
    "run_summary.json",
)
PHASE3_SEED_COMPONENTS = (
    "plan5_model_initialization",
    "plan5_data_partition",
    "plan5_initialization_loader",
    "plan5_online_stream",
    "plan5_reference_stream",
    "plan5_evaluation_panel",
    "plan5_optimizer",
    "plan5_numerical_randomization",
    "plan5_phase3_initial_lanczos",
    "plan5_phase3_online_lanczos",
)


class RotatedPhase3RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"

    def begin(
        self,
        config: RotatedPhase3Config,
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
                    f"Phase 3 run already completed: {config.run_id}"
                )
            raise RotatedArtifactError(
                f"final Phase 3 run exists without COMPLETED: {final_path}"
            )
        if working_path.exists():
            if not resume:
                raise RotatedIncompleteRunError(
                    f"incomplete Phase 3 run exists; pass --resume: {config.run_id}"
                )
            if _read_json(working_path / "config.json") != config.to_mapping():
                raise RotatedArtifactError("incomplete Phase 3 config differs")
            manifest = _read_json(working_path / "manifest.json")
            if manifest.get("status") != "incomplete":
                raise RotatedArtifactError("incomplete Phase 3 manifest is invalid")
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
                "run_kind": "phase3_low_data_transfer_pilot",
                "run_id": config.run_id,
                "experiment": config.experiment,
                "replica_id": config.replica_id,
                "config_hash": config.config_hash,
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
                "seeds": derive_seed_map(
                    config.replica_seed, PHASE3_SEED_COMPONENTS
                ),
                "runtime": runtime_metadata(Path(repo_root), config),
            },
        )
        return RotatedRunSession(config.run_id, working_path, final_path)


@dataclasses.dataclass(frozen=True)
class LoadedRotatedPhase3Run:
    path: Path
    config: RotatedPhase3Config
    manifest: dict[str, Any]
    partitions: RotatedPartitions
    stream_plan: RotatedStreamPlan
    initialization_metrics: dict[str, Any]
    trajectory_metrics: dict[str, tuple[dict[str, Any], ...]]
    run_summary: dict[str, Any]


def load_completed_phase3_run(path: str | Path) -> LoadedRotatedPhase3Run:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(f"Phase 3 run is incomplete: {run_path}")
    for name in PHASE3_REQUIRED_ARTIFACTS:
        if not (run_path / name).is_file():
            raise RotatedArtifactError(f"completed Phase 3 run is missing {name}")
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("Phase 3 config artifact must be an object")
    config = RotatedPhase3Config.from_mapping(config_value)
    manifest = _read_json(run_path / "manifest.json")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_schema_version")
        != PHASE3_ARTIFACT_SCHEMA_VERSION
        or manifest.get("metric_schema_version") != PHASE3_METRIC_SCHEMA_VERSION
        or manifest.get("run_kind") != "phase3_low_data_transfer_pilot"
        or manifest.get("status") != "completed"
        or manifest.get("config_hash") != config.config_hash
        or manifest.get("run_id") != config.run_id
        or run_path.name != config.run_id
    ):
        raise RotatedArtifactError("Phase 3 manifest is incompatible")
    partitions = RotatedPartitions.from_mapping(
        _read_json(run_path / "partitions.json")
    )
    stream_plan = RotatedStreamPlan.from_mapping(
        _read_json(run_path / "stream_plan.json")
    )
    if stream_plan.partition_hash != partitions.content_hash:
        raise RotatedArtifactError("Phase 3 stream and partition hashes differ")
    metrics = _read_json(run_path / "trajectory_metrics.json")
    if not isinstance(metrics, Mapping) or tuple(metrics) != PHASE3_CONDITIONS:
        raise RotatedArtifactError("Phase 3 treatment metrics are incompatible")
    trajectory_metrics: dict[str, tuple[dict[str, Any], ...]] = {}
    for condition in PHASE3_CONDITIONS:
        rows = metrics[condition]
        if not isinstance(rows, list) or len(rows) != stream_plan.schedule.num_points:
            raise RotatedArtifactError("Phase 3 trajectory length is incompatible")
        for step, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or row.get("step") != step
                or row.get("condition") != condition
                or row.get("angle_degrees")
                != stream_plan.schedule.angles_degrees[step]
                or "current_environment_accuracy" not in row
                or "current_per_class_recall" not in row
            ):
                raise RotatedArtifactError("Phase 3 metric contract failed")
        trajectory_metrics[condition] = tuple(rows)
    initialization = _read_json(run_path / "initialization_metrics.json")
    summary = _read_json(run_path / "run_summary.json")
    if not isinstance(initialization, dict) or not isinstance(summary, dict):
        raise RotatedArtifactError("Phase 3 scalar artifacts are incompatible")
    if (
        summary.get("schedule_hash") != stream_plan.schedule.content_hash
        or summary.get("stream_plan_hash") != stream_plan.content_hash
        or summary.get("partition_hash") != partitions.content_hash
        or tuple(summary.get("conditions", ())) != PHASE3_CONDITIONS
    ):
        raise RotatedArtifactError("Phase 3 summary provenance is incompatible")
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
        raise RotatedArtifactError("Phase 3 initial Fisher artifact is invalid")
    dense = fisher["dense"]
    if (
        dense.ndim != 2
        or dense.shape[0] != dense.shape[1]
        or not torch.isfinite(dense).all()
    ):
        raise RotatedArtifactError("Phase 3 dense initial Fisher is invalid")
    return LoadedRotatedPhase3Run(
        path=run_path,
        config=config,
        manifest=manifest,
        partitions=partitions,
        stream_plan=stream_plan,
        initialization_metrics=initialization,
        trajectory_metrics=trajectory_metrics,
        run_summary=summary,
    )
