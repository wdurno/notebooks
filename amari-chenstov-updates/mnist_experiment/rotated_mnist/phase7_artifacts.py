"""Immutable artifact lifecycle for the Plan 7 coefficient audit."""

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
from .phase7_config import (
    PHASE7_ARTIFACT_SCHEMA_VERSION,
    PHASE7_METRIC_SCHEMA_VERSION,
    Phase7AuditConfig,
)


PHASE7_REQUIRED_ARTIFACTS = (
    "source_contract.json",
    "branch_diagnostics.json",
    "coefficient_rows.json",
    "audit_summary.json",
)
PHASE7_RUN_KIND = "phase7_artifact_only_anchor_coefficient_audit"


class Phase7RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"

    def begin(
        self,
        config: Phase7AuditConfig,
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
                    f"Plan 7 audit already completed: {config.run_id}"
                )
            raise RotatedArtifactError(
                f"final Plan 7 audit exists without COMPLETED: {final_path}"
            )
        if working_path.exists():
            if not resume:
                raise RotatedIncompleteRunError(
                    f"incomplete Plan 7 audit exists; pass --resume: {config.run_id}"
                )
            if _read_json(working_path / "config.json") != config.to_mapping():
                raise RotatedArtifactError("incomplete Plan 7 config differs")
            manifest = _read_json(working_path / "manifest.json")
            if manifest.get("status") != "incomplete":
                raise RotatedArtifactError("incomplete Plan 7 manifest is invalid")
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
                "run_kind": PHASE7_RUN_KIND,
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
class LoadedPhase7AuditRun:
    path: Path
    config: Phase7AuditConfig
    manifest: dict[str, Any]
    source_contract: dict[str, Any]
    branch_diagnostics: dict[str, Any]
    coefficient_rows: tuple[dict[str, Any], ...]
    audit_summary: dict[str, Any]


def load_completed_phase7_audit(path: str | Path) -> LoadedPhase7AuditRun:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(f"Plan 7 audit is incomplete: {run_path}")
    for name in ("config.json", "manifest.json", *PHASE7_REQUIRED_ARTIFACTS):
        if not (run_path / name).is_file():
            raise RotatedArtifactError(f"completed Plan 7 audit is missing {name}")
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("Plan 7 config must be an object")
    config = Phase7AuditConfig.from_mapping(config_value)
    manifest = _read_json(run_path / "manifest.json")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_schema_version")
        != PHASE7_ARTIFACT_SCHEMA_VERSION
        or manifest.get("metric_schema_version") != PHASE7_METRIC_SCHEMA_VERSION
        or manifest.get("run_kind") != PHASE7_RUN_KIND
        or manifest.get("status") != "completed"
        or manifest.get("config_hash") != config.config_hash
        or manifest.get("run_id") != config.run_id
        or run_path.name != config.run_id
    ):
        raise RotatedArtifactError("Plan 7 manifest is incompatible")
    source_contract = _read_json(run_path / "source_contract.json")
    branch = _read_json(run_path / "branch_diagnostics.json")
    rows = _read_json(run_path / "coefficient_rows.json")
    summary = _read_json(run_path / "audit_summary.json")
    if (
        not isinstance(source_contract, dict)
        or not isinstance(branch, dict)
        or not isinstance(rows, list)
        or not isinstance(summary, dict)
        or source_contract.get("source_run_id") != config.source_run_id
        or source_contract.get("debias_run_id") != config.debias_run_id
        or source_contract.get("oracle_run_id") != config.oracle_run_id
        or summary.get("config_hash") != config.config_hash
        or summary.get("row_count") != len(rows)
    ):
        raise RotatedArtifactError("Plan 7 artifact contents are incompatible")
    return LoadedPhase7AuditRun(
        path=run_path,
        config=config,
        manifest=manifest,
        source_contract=source_contract,
        branch_diagnostics=branch,
        coefficient_rows=tuple(rows),
        audit_summary=summary,
    )
