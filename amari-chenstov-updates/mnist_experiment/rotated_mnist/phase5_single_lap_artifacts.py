"""Immutable artifacts for the Phase 5 slow-trend single-lap retry."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    MANIFEST_SCHEMA_VERSION,
    RotatedArtifactError,
    RotatedIncompleteRunError,
    _read_json,
)
from .data import RotatedPartitions, RotatedStreamPlan
from .phase5_double_lap_artifacts import (
    DOUBLE_LAP_REQUIRED_ARTIFACTS,
    RotatedDoubleLapRunStore,
)
from .phase5_single_lap_config import (
    SINGLE_LAP_ARTIFACT_SCHEMA_VERSION,
    SINGLE_LAP_CONDITIONS,
    SINGLE_LAP_METRIC_SCHEMA_VERSION,
    SINGLE_LAP_SCHEDULES,
    RotatedSlowSingleLapConfig,
)


SINGLE_LAP_REQUIRED_ARTIFACTS = DOUBLE_LAP_REQUIRED_ARTIFACTS
SINGLE_LAP_SEED_COMPONENTS = (
    "plan5_model_initialization",
    "plan5_data_partition",
    "plan5_initialization_loader",
    "plan5_online_stream",
    "plan5_reference_stream",
    "plan5_evaluation_panel",
    "plan5_optimizer",
    "plan5_single_lap_initial_lanczos",
    "plan5_single_lap_update_lanczos",
)
SINGLE_LAP_RUN_KIND = "phase5_slow_single_lap_closed_loop_retry"


class RotatedSingleLapRunStore(RotatedDoubleLapRunStore):
    def __init__(self, root: str | Path):
        super().__init__(
            root,
            run_kind=SINGLE_LAP_RUN_KIND,
            run_label="slow single-lap",
            seed_components=SINGLE_LAP_SEED_COMPONENTS,
        )


@dataclasses.dataclass(frozen=True)
class LoadedSingleLapRun:
    path: Path
    config: RotatedSlowSingleLapConfig
    manifest: dict[str, Any]
    partitions: RotatedPartitions
    stream_plans: dict[str, RotatedStreamPlan]
    trajectory_metrics: dict[str, dict[str, tuple[dict[str, Any], ...]]]
    operational_checks: dict[str, Any]
    run_summary: dict[str, Any]


def load_completed_single_lap_run(path: str | Path) -> LoadedSingleLapRun:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(
            f"slow single-lap run is incomplete: {run_path}"
        )
    for name in SINGLE_LAP_REQUIRED_ARTIFACTS:
        if not (run_path / name).is_file():
            raise RotatedArtifactError(
                f"completed slow single-lap run is missing {name}"
            )
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("slow single-lap config must be an object")
    config = RotatedSlowSingleLapConfig.from_mapping(config_value)
    manifest = _read_json(run_path / "manifest.json")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_schema_version")
        != SINGLE_LAP_ARTIFACT_SCHEMA_VERSION
        or manifest.get("metric_schema_version") != SINGLE_LAP_METRIC_SCHEMA_VERSION
        or manifest.get("run_kind") != SINGLE_LAP_RUN_KIND
        or manifest.get("status") != "completed"
        or manifest.get("config_hash") != config.config_hash
        or manifest.get("run_id") != config.run_id
        or run_path.name != config.run_id
    ):
        raise RotatedArtifactError("slow single-lap manifest is incompatible")
    partitions = RotatedPartitions.from_mapping(
        _read_json(run_path / "partitions.json")
    )
    plan_values = _read_json(run_path / "stream_plans.json")
    if not isinstance(plan_values, Mapping) or set(plan_values) != set(
        SINGLE_LAP_SCHEDULES
    ):
        raise RotatedArtifactError("slow single-lap stream plans are incompatible")
    plans = {
        name: RotatedStreamPlan.from_mapping(plan_values[name])
        for name in SINGLE_LAP_SCHEDULES
    }
    metrics = _read_json(run_path / "trajectory_metrics.json")
    if not isinstance(metrics, Mapping) or set(metrics) != set(
        SINGLE_LAP_SCHEDULES
    ):
        raise RotatedArtifactError("slow single-lap schedules are incomplete")
    converted = {}
    for schedule in SINGLE_LAP_SCHEDULES:
        values = metrics[schedule]
        if not isinstance(values, Mapping) or set(values) != set(
            SINGLE_LAP_CONDITIONS
        ):
            raise RotatedArtifactError("slow single-lap conditions are incomplete")
        converted[schedule] = {}
        for condition in SINGLE_LAP_CONDITIONS:
            rows = values[condition]
            if (
                not isinstance(rows, list)
                or len(rows) != plans[schedule].schedule.num_points
            ):
                raise RotatedArtifactError(
                    "slow single-lap trajectory length is invalid"
                )
            converted[schedule][condition] = tuple(rows)
    checks = _read_json(run_path / "operational_checks.json")
    summary = _read_json(run_path / "run_summary.json")
    if (
        not isinstance(checks, dict)
        or not isinstance(summary, dict)
        or checks.get("all_finite") is not True
        or summary.get("config_hash") != config.config_hash
    ):
        raise RotatedArtifactError("slow single-lap run summary is incompatible")
    return LoadedSingleLapRun(
        path=run_path,
        config=config,
        manifest=manifest,
        partitions=partitions,
        stream_plans=plans,
        trajectory_metrics=converted,
        operational_checks=checks,
        run_summary=summary,
    )
