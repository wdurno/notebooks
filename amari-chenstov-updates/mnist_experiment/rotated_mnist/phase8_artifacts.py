"""Immutable artifacts for the decomposed-EDR rechallenge."""

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
from .phase8_config import (
    PHASE8_ARTIFACT_SCHEMA_VERSION,
    PHASE8_METRIC_SCHEMA_VERSION,
    Phase8Config,
)


PHASE8_RUN_KIND = "phase8_decomposed_edr_rechallenge"
PHASE8_REQUIRED_ARTIFACTS = (
    "source_contract.json",
    "trajectory_metrics.json",
    "trajectories.pt",
    "model_states.pt",
    "controller_states.pt",
    "operational_checks.json",
    "run_summary.json",
)
PHASE8_SEED_COMPONENTS = (
    "plan8_update_lanczos",
    "plan8_analysis_bootstrap",
)


class Phase8RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"

    def begin(
        self,
        config: Phase8Config,
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
                    f"Plan 8 run already completed: {config.run_id}"
                )
            raise RotatedArtifactError(
                f"final Plan 8 run exists without COMPLETED: {final_path}"
            )
        if working_path.exists():
            if not resume:
                raise RotatedIncompleteRunError(
                    f"incomplete Plan 8 run exists; pass --resume: {config.run_id}"
                )
            if _read_json(working_path / "config.json") != config.to_mapping():
                raise RotatedArtifactError("incomplete Plan 8 config differs")
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
                "run_kind": PHASE8_RUN_KIND,
                "run_id": config.run_id,
                "experiment": config.experiment,
                "replica_id": config.replica_id,
                "config_hash": config.config_hash,
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
                "seeds": derive_seed_map(
                    config.replica_seed, PHASE8_SEED_COMPONENTS
                ),
                "runtime": runtime_metadata(Path(repo_root), config),
            },
        )
        return RotatedRunSession(config.run_id, working_path, final_path)


@dataclasses.dataclass(frozen=True)
class LoadedPhase8Run:
    path: Path
    config: Phase8Config
    manifest: dict[str, Any]
    source_contract: dict[str, Any]
    trajectory_metrics: dict[str, dict[str, tuple[dict[str, Any], ...]]]
    operational_checks: dict[str, Any]
    run_summary: dict[str, Any]


def load_completed_phase8_run(path: str | Path) -> LoadedPhase8Run:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(f"Plan 8 run is incomplete: {run_path}")
    for name in PHASE8_REQUIRED_ARTIFACTS:
        if not (run_path / name).is_file():
            raise RotatedArtifactError(f"completed Plan 8 run is missing {name}")
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("Plan 8 config must be an object")
    config = Phase8Config.from_mapping(config_value)
    manifest = _read_json(run_path / "manifest.json")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_schema_version")
        != PHASE8_ARTIFACT_SCHEMA_VERSION
        or manifest.get("metric_schema_version") != PHASE8_METRIC_SCHEMA_VERSION
        or manifest.get("run_kind") != PHASE8_RUN_KIND
        or manifest.get("status") != "completed"
        or manifest.get("config_hash") != config.config_hash
        or manifest.get("run_id") != config.run_id
        or run_path.name != config.run_id
    ):
        raise RotatedArtifactError("Plan 8 manifest is incompatible")
    source = _read_json(run_path / "source_contract.json")
    metrics = _read_json(run_path / "trajectory_metrics.json")
    checks = _read_json(run_path / "operational_checks.json")
    summary = _read_json(run_path / "run_summary.json")
    if not all(isinstance(value, dict) for value in (source, metrics, checks, summary)):
        raise RotatedArtifactError("Plan 8 scalar artifacts are invalid")
    if checks.get("all_finite") is not True:
        raise RotatedArtifactError("Plan 8 completed with nonfinite values")
    converted: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {}
    for schedule, by_condition in metrics.items():
        if not isinstance(by_condition, Mapping) or set(by_condition) != set(
            config.conditions
        ):
            raise RotatedArtifactError("Plan 8 condition metrics are incomplete")
        converted[schedule] = {}
        for condition, rows in by_condition.items():
            if not isinstance(rows, list) or not rows:
                raise RotatedArtifactError("Plan 8 trajectory is empty")
            converted[schedule][condition] = tuple(rows)
    return LoadedPhase8Run(
        path=run_path,
        config=config,
        manifest=manifest,
        source_contract=source,
        trajectory_metrics=converted,
        operational_checks=checks,
        run_summary=summary,
    )
