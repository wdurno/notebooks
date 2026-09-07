"""Immutable artifacts for the Plan 7 movement-premium audit."""

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
from .phase7_movement_config import (
    MOVEMENT_ARTIFACT_SCHEMA_VERSION,
    MOVEMENT_METRIC_SCHEMA_VERSION,
    MovementPremiumAuditConfig,
)


MOVEMENT_RUN_KIND = "phase7_artifact_only_movement_premium_audit"
MOVEMENT_REQUIRED_ARTIFACTS = (
    "source_contract.json",
    "movement_rows.json",
    "audit_summary.json",
)


class MovementPremiumAuditStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"

    def begin(
        self,
        config: MovementPremiumAuditConfig,
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
                    f"movement-premium audit already completed: {config.run_id}"
                )
            raise RotatedArtifactError(
                f"movement-premium final path lacks COMPLETED: {final_path}"
            )
        if working_path.exists():
            if not resume:
                raise RotatedIncompleteRunError(
                    f"movement-premium audit is incomplete; pass --resume: {config.run_id}"
                )
            if _read_json(working_path / "config.json") != config.to_mapping():
                raise RotatedArtifactError("incomplete movement-audit config differs")
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
                "run_kind": MOVEMENT_RUN_KIND,
                "run_id": config.run_id,
                "experiment": config.experiment,
                "replica_id": config.replica_id,
                "config_hash": config.config_hash,
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
                "seeds": {},
                "runtime": runtime_metadata(repo_root, config),
            },
        )
        return RotatedRunSession(config.run_id, working_path, final_path)


@dataclasses.dataclass(frozen=True)
class LoadedMovementPremiumAudit:
    path: Path
    config: MovementPremiumAuditConfig
    manifest: dict[str, Any]
    source_contract: dict[str, Any]
    movement_rows: tuple[dict[str, Any], ...]
    audit_summary: dict[str, Any]


def load_completed_movement_premium_audit(
    path: str | Path,
) -> LoadedMovementPremiumAudit:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(
            f"movement-premium audit is incomplete: {run_path}"
        )
    for name in ("config.json", "manifest.json", *MOVEMENT_REQUIRED_ARTIFACTS):
        if not (run_path / name).is_file():
            raise RotatedArtifactError(
                f"completed movement-premium audit is missing {name}"
            )
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("movement-premium config must be an object")
    config = MovementPremiumAuditConfig.from_mapping(config_value)
    manifest = _read_json(run_path / "manifest.json")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_schema_version")
        != MOVEMENT_ARTIFACT_SCHEMA_VERSION
        or manifest.get("metric_schema_version") != MOVEMENT_METRIC_SCHEMA_VERSION
        or manifest.get("run_kind") != MOVEMENT_RUN_KIND
        or manifest.get("status") != "completed"
        or manifest.get("config_hash") != config.config_hash
        or manifest.get("run_id") != config.run_id
        or run_path.name != config.run_id
    ):
        raise RotatedArtifactError("movement-premium manifest is incompatible")
    source = _read_json(run_path / "source_contract.json")
    rows = _read_json(run_path / "movement_rows.json")
    summary = _read_json(run_path / "audit_summary.json")
    if (
        not isinstance(source, dict)
        or not isinstance(rows, list)
        or not isinstance(summary, dict)
        or summary.get("config_hash") != config.config_hash
        or summary.get("row_count") != len(rows)
    ):
        raise RotatedArtifactError("movement-premium contents are incompatible")
    return LoadedMovementPremiumAudit(
        path=run_path,
        config=config,
        manifest=manifest,
        source_contract=source,
        movement_rows=tuple(rows),
        audit_summary=summary,
    )
