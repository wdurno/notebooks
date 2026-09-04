"""Immutable artifacts for the Plan 5 Phase 5 EDR challenge."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

from src.seeding import SEED_SCHEMA_VERSION

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
from .phase4_config import RotatedPhase4Config
from .phase5_config import (
    PHASE5_ARTIFACT_SCHEMA_VERSION,
    PHASE5_CONDITIONS,
    PHASE5_METRIC_SCHEMA_VERSION,
    RotatedPhase5Config,
)


PHASE5_RECONSTRUCTION_REQUIRED = (
    "source.json",
    "coefficient_trajectory.json",
    "fisher_reconstruction.json",
    "gate.json",
    "run_summary.json",
)
PHASE5_CHALLENGE_REQUIRED = (
    "source.json",
    "trajectory_metrics.json",
    "trajectories.pt",
    "model_states.pt",
    "controller_states.pt",
    "operational_checks.json",
    "run_summary.json",
)
PHASE5_SENSITIVITY_REQUIRED = (
    "source.json",
    "sensitivity_trajectories.json",
    "sensitivity_summary.json",
    "gate.json",
    "run_summary.json",
)


class RotatedPhase5RunStore:
    def __init__(self, root: str | Path, *, run_kind: str):
        if run_kind not in {
            "phase5_edr_reconstruction",
            "phase5_edr_closed_loop_challenge",
            "phase5_edr_sensitivity",
        }:
            raise ValueError("unsupported Phase 5 run kind")
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"
        self.run_kind = run_kind

    def begin(
        self,
        config: RotatedPhase5Config,
        source_config: RotatedPhase4Config,
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
                    f"Phase 5 run already completed: {config.run_id}"
                )
            raise RotatedArtifactError(
                f"final Phase 5 run exists without COMPLETED: {final_path}"
            )
        if working_path.exists():
            if not resume:
                raise RotatedIncompleteRunError(
                    f"incomplete Phase 5 run exists; pass --resume: {config.run_id}"
                )
            if _read_json(working_path / "config.json") != config.to_mapping():
                raise RotatedArtifactError("incomplete Phase 5 config differs")
            manifest = _read_json(working_path / "manifest.json")
            if (
                manifest.get("status") != "incomplete"
                or manifest.get("run_kind") != self.run_kind
            ):
                raise RotatedArtifactError("incomplete Phase 5 manifest is invalid")
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
                "source_phase4_run_id": config.source_phase4_run_id,
                "source_phase4_config_hash": config.source_phase4_config_hash,
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
                "runtime": runtime_metadata(Path(repo_root), source_config),
            },
        )
        return RotatedRunSession(config.run_id, working_path, final_path)


@dataclasses.dataclass(frozen=True)
class LoadedPhase5Reconstruction:
    path: Path
    config: RotatedPhase5Config
    manifest: dict[str, Any]
    source: dict[str, Any]
    coefficient_trajectory: tuple[dict[str, Any], ...]
    fisher_reconstruction: dict[str, Any]
    gate: dict[str, Any]
    run_summary: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class LoadedPhase5Challenge:
    path: Path
    config: RotatedPhase5Config
    manifest: dict[str, Any]
    source: dict[str, Any]
    trajectory_metrics: dict[str, tuple[dict[str, Any], ...]]
    controller_states: dict[str, Any]
    operational_checks: dict[str, Any]
    run_summary: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class LoadedPhase5Sensitivity:
    path: Path
    config: RotatedPhase5Config
    manifest: dict[str, Any]
    source: dict[str, Any]
    trajectories: dict[str, tuple[dict[str, Any], ...]]
    summaries: dict[str, dict[str, Any]]
    gate: dict[str, Any]
    run_summary: dict[str, Any]


def _load_base(
    path: str | Path,
    *,
    run_kind: str,
    required: tuple[str, ...],
) -> tuple[Path, RotatedPhase5Config, dict[str, Any]]:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(f"Phase 5 run is incomplete: {run_path}")
    for name in required:
        if not (run_path / name).is_file():
            raise RotatedArtifactError(f"completed Phase 5 run is missing {name}")
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("Phase 5 config artifact must be an object")
    config = RotatedPhase5Config.from_mapping(config_value)
    manifest = _read_json(run_path / "manifest.json")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_schema_version")
        != PHASE5_ARTIFACT_SCHEMA_VERSION
        or manifest.get("metric_schema_version") != PHASE5_METRIC_SCHEMA_VERSION
        or manifest.get("run_kind") != run_kind
        or manifest.get("status") != "completed"
        or manifest.get("config_hash") != config.config_hash
        or manifest.get("run_id") != config.run_id
        or run_path.name != config.run_id
    ):
        raise RotatedArtifactError("Phase 5 manifest is incompatible")
    return run_path, config, manifest


def load_completed_phase5_reconstruction(
    path: str | Path,
) -> LoadedPhase5Reconstruction:
    run_path, config, manifest = _load_base(
        path,
        run_kind="phase5_edr_reconstruction",
        required=PHASE5_RECONSTRUCTION_REQUIRED,
    )
    source = _read_json(run_path / "source.json")
    trajectory = _read_json(run_path / "coefficient_trajectory.json")
    fisher = _read_json(run_path / "fisher_reconstruction.json")
    gate = _read_json(run_path / "gate.json")
    summary = _read_json(run_path / "run_summary.json")
    if (
        not isinstance(source, dict)
        or not isinstance(trajectory, list)
        or not isinstance(fisher, dict)
        or not isinstance(gate, dict)
        or not isinstance(summary, dict)
    ):
        raise RotatedArtifactError("Phase 5 reconstruction artifacts are invalid")
    if (
        source.get("run_id") != config.source_phase4_run_id
        or source.get("config_hash") != config.source_phase4_config_hash
        or summary.get("trajectory_rows") != len(trajectory)
        or gate.get("recommendation") not in {"proceed", "stop"}
    ):
        raise RotatedArtifactError("Phase 5 reconstruction provenance is invalid")
    return LoadedPhase5Reconstruction(
        path=run_path,
        config=config,
        manifest=manifest,
        source=source,
        coefficient_trajectory=tuple(trajectory),
        fisher_reconstruction=fisher,
        gate=gate,
        run_summary=summary,
    )


def load_completed_phase5_challenge(path: str | Path) -> LoadedPhase5Challenge:
    run_path, config, manifest = _load_base(
        path,
        run_kind="phase5_edr_closed_loop_challenge",
        required=PHASE5_CHALLENGE_REQUIRED,
    )
    source = _read_json(run_path / "source.json")
    metrics = _read_json(run_path / "trajectory_metrics.json")
    # Controller tensors live in the torch artifact; scalar summaries stay in JSON.
    controller_summary = _read_json(run_path / "run_summary.json").get(
        "controller_summaries", {}
    )
    checks = _read_json(run_path / "operational_checks.json")
    summary = _read_json(run_path / "run_summary.json")
    if not isinstance(metrics, Mapping) or set(metrics) != set(PHASE5_CONDITIONS):
        raise RotatedArtifactError("Phase 5 challenge conditions are incompatible")
    converted = {}
    expected_rows = summary.get("num_points")
    for condition in PHASE5_CONDITIONS:
        rows = metrics[condition]
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise RotatedArtifactError("Phase 5 challenge trajectory is incomplete")
        converted[condition] = tuple(rows)
    if (
        source.get("run_id") != config.source_phase4_run_id
        or checks.get("all_finite") is not True
        or not isinstance(controller_summary, dict)
    ):
        raise RotatedArtifactError("Phase 5 challenge provenance is invalid")
    return LoadedPhase5Challenge(
        path=run_path,
        config=config,
        manifest=manifest,
        source=source,
        trajectory_metrics=converted,
        controller_states=controller_summary,
        operational_checks=checks,
        run_summary=summary,
    )


def load_completed_phase5_sensitivity(
    path: str | Path,
) -> LoadedPhase5Sensitivity:
    run_path, config, manifest = _load_base(
        path,
        run_kind="phase5_edr_sensitivity",
        required=PHASE5_SENSITIVITY_REQUIRED,
    )
    source = _read_json(run_path / "source.json")
    trajectories = _read_json(run_path / "sensitivity_trajectories.json")
    summaries = _read_json(run_path / "sensitivity_summary.json")
    gate = _read_json(run_path / "gate.json")
    summary = _read_json(run_path / "run_summary.json")
    if not all(
        isinstance(value, dict)
        for value in (source, trajectories, summaries, gate, summary)
    ):
        raise RotatedArtifactError("Phase 5 sensitivity artifacts are invalid")
    expected = set(summary.get("settings", ()))
    if (
        not expected
        or set(trajectories) != expected
        or set(summaries) != expected
        or source.get("run_id") != config.source_phase4_run_id
        or source.get("config_hash") != config.source_phase4_config_hash
        or gate.get("recommendation") not in {"reopen", "retain_stop"}
    ):
        raise RotatedArtifactError("Phase 5 sensitivity provenance is invalid")
    converted: dict[str, tuple[dict[str, Any], ...]] = {}
    expected_rows = summary.get("trajectory_rows_per_setting")
    for setting, rows in trajectories.items():
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise RotatedArtifactError("Phase 5 sensitivity trajectory is incomplete")
        converted[setting] = tuple(rows)
    return LoadedPhase5Sensitivity(
        path=run_path,
        config=config,
        manifest=manifest,
        source=source,
        trajectories=converted,
        summaries={name: dict(value) for name, value in summaries.items()},
        gate=gate,
        run_summary=summary,
    )
