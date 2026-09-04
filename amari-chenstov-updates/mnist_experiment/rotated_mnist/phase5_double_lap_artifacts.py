"""Immutable artifacts for the Phase 5 double-lap closed-loop retry."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

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
from .phase5_double_lap_config import (
    DOUBLE_LAP_ARTIFACT_SCHEMA_VERSION,
    DOUBLE_LAP_CONDITIONS,
    DOUBLE_LAP_METRIC_SCHEMA_VERSION,
    DOUBLE_LAP_SCHEDULES,
    RotatedDoubleLapConfig,
)


DOUBLE_LAP_REQUIRED_ARTIFACTS = (
    "partitions.json",
    "stream_plans.json",
    "stream_tensors.pt",
    "initialization_metrics.json",
    "initial_fisher.pt",
    "evaluation_panel.json",
    "trajectory_metrics.json",
    "trajectories.pt",
    "model_states.pt",
    "controller_states.pt",
    "operational_checks.json",
    "run_summary.json",
)
DOUBLE_LAP_SEED_COMPONENTS = (
    "plan5_model_initialization",
    "plan5_data_partition",
    "plan5_initialization_loader",
    "plan5_online_stream",
    "plan5_reference_stream",
    "plan5_evaluation_panel",
    "plan5_optimizer",
    "plan5_double_lap_initial_lanczos",
    "plan5_double_lap_update_lanczos",
)


class RotatedDoubleLapRunStore:
    def __init__(
        self,
        root: str | Path,
        *,
        run_kind: str = "phase5_double_lap_closed_loop_retry",
        run_label: str = "double-lap",
        seed_components: tuple[str, ...] = DOUBLE_LAP_SEED_COMPONENTS,
    ):
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"
        self.run_kind = run_kind
        self.run_label = run_label
        self.seed_components = seed_components

    def begin(
        self,
        config: RotatedDoubleLapConfig,
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
                    f"{self.run_label} run already completed: {config.run_id}"
                )
            raise RotatedArtifactError(
                f"final {self.run_label} run exists without COMPLETED: {final_path}"
            )
        if working_path.exists():
            if not resume:
                raise RotatedIncompleteRunError(
                    f"incomplete {self.run_label} run exists; pass --resume: "
                    f"{config.run_id}"
                )
            if _read_json(working_path / "config.json") != config.to_mapping():
                raise RotatedArtifactError(
                    f"incomplete {self.run_label} config differs"
                )
            manifest = _read_json(working_path / "manifest.json")
            if manifest.get("status") != "incomplete":
                raise RotatedArtifactError(
                    f"incomplete {self.run_label} manifest is invalid"
                )
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
                "run_kind": self.run_kind,
                "run_id": config.run_id,
                "experiment": config.experiment,
                "replica_id": config.replica_id,
                "config_hash": config.config_hash,
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
                "seeds": derive_seed_map(
                    config.replica_seed, self.seed_components
                ),
                "runtime": runtime_metadata(repo_root, config),
            },
        )
        return RotatedRunSession(config.run_id, working_path, final_path)


@dataclasses.dataclass(frozen=True)
class LoadedDoubleLapRun:
    path: Path
    config: RotatedDoubleLapConfig
    manifest: dict[str, Any]
    partitions: RotatedPartitions
    stream_plans: dict[str, RotatedStreamPlan]
    trajectory_metrics: dict[str, dict[str, tuple[dict[str, Any], ...]]]
    operational_checks: dict[str, Any]
    run_summary: dict[str, Any]


def load_completed_double_lap_run(path: str | Path) -> LoadedDoubleLapRun:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(f"double-lap run is incomplete: {run_path}")
    for name in DOUBLE_LAP_REQUIRED_ARTIFACTS:
        if not (run_path / name).is_file():
            raise RotatedArtifactError(f"completed double-lap run is missing {name}")
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("double-lap config artifact must be an object")
    config = RotatedDoubleLapConfig.from_mapping(config_value)
    manifest = _read_json(run_path / "manifest.json")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_schema_version")
        != DOUBLE_LAP_ARTIFACT_SCHEMA_VERSION
        or manifest.get("metric_schema_version") != DOUBLE_LAP_METRIC_SCHEMA_VERSION
        or manifest.get("run_kind") != "phase5_double_lap_closed_loop_retry"
        or manifest.get("status") != "completed"
        or manifest.get("config_hash") != config.config_hash
        or manifest.get("run_id") != config.run_id
        or run_path.name != config.run_id
    ):
        raise RotatedArtifactError("double-lap manifest is incompatible")
    partitions = RotatedPartitions.from_mapping(
        _read_json(run_path / "partitions.json")
    )
    plan_values = _read_json(run_path / "stream_plans.json")
    if not isinstance(plan_values, Mapping) or set(plan_values) != set(
        DOUBLE_LAP_SCHEDULES
    ):
        raise RotatedArtifactError("double-lap stream plans are incompatible")
    plans = {
        name: RotatedStreamPlan.from_mapping(plan_values[name])
        for name in DOUBLE_LAP_SCHEDULES
    }
    metrics = _read_json(run_path / "trajectory_metrics.json")
    if not isinstance(metrics, Mapping) or set(metrics) != set(DOUBLE_LAP_SCHEDULES):
        raise RotatedArtifactError("double-lap schedules are incomplete")
    converted = {}
    for schedule in DOUBLE_LAP_SCHEDULES:
        values = metrics[schedule]
        if not isinstance(values, Mapping) or set(values) != set(
            DOUBLE_LAP_CONDITIONS
        ):
            raise RotatedArtifactError("double-lap conditions are incomplete")
        converted[schedule] = {}
        for condition in DOUBLE_LAP_CONDITIONS:
            rows = values[condition]
            if not isinstance(rows, list) or len(rows) != plans[schedule].schedule.num_points:
                raise RotatedArtifactError("double-lap trajectory length is invalid")
            converted[schedule][condition] = tuple(rows)
    checks = _read_json(run_path / "operational_checks.json")
    summary = _read_json(run_path / "run_summary.json")
    if (
        not isinstance(checks, dict)
        or not isinstance(summary, dict)
        or checks.get("all_finite") is not True
        or summary.get("config_hash") != config.config_hash
    ):
        raise RotatedArtifactError("double-lap run summary is incompatible")
    return LoadedDoubleLapRun(
        path=run_path,
        config=config,
        manifest=manifest,
        partitions=partitions,
        stream_plans=plans,
        trajectory_metrics=converted,
        operational_checks=checks,
        run_summary=summary,
    )
