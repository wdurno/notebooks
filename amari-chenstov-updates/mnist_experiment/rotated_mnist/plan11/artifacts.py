"""Atomic Plan 11 units; a completed trajectory is never overwritten."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from src.seeding import derive_seed_map

from ..artifacts import (
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
from .config import SOURCE_FILES, Study, canonical_hash


SEED_COMPONENTS = (
    "plan5_model_initialization",
    "plan5_data_partition",
    "plan5_initialization_loader",
    "plan5_online_stream",
    "plan5_reference_stream",
    "plan5_evaluation_panel",
    "plan11_initial_lanczos",
    "plan11_update_lanczos",
)


def source_hashes(repo_root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UnitStore:
    def __init__(self, root: Path, study: Study, repo_root: Path):
        self.root = root
        self.study = study
        self.repo_root = repo_root
        self.sources = source_hashes(repo_root)

    def unit(self, phase: str, index: int, schedule: str | None, policy: dict[str, float] | None) -> dict[str, Any]:
        return {
            "study_hash": self.study.config_hash,
            "phase": phase,
            "replica_index": index,
            "replica_seed": self.study.replica_seed(phase, index),
            "schedule": schedule,
            "policy": policy,
            "source_hashes": self.sources,
        }

    def paths(self, unit: dict[str, Any]) -> tuple[Path, Path]:
        digest = canonical_hash(unit)[:16]
        label = "assets" if unit["schedule"] is None else f'{unit["schedule"]}__c{unit["policy"]["anchor"]:.3f}_g{unit["policy"]["gain"]:.3f}'
        run_id = f'{unit["phase"]}__replica-{unit["replica_index"]:04d}__{label}__{digest}'
        final = self.root / unit["phase"] / run_id
        working = self.root / unit["phase"] / ".incomplete" / run_id
        return final, working

    def completed(self, unit: dict[str, Any], required: tuple[str, ...]) -> Path | None:
        final, _ = self.paths(unit)
        if not final.exists():
            return None
        if not (final / "COMPLETED").is_file():
            raise RotatedArtifactError(f"final Plan 11 path lacks COMPLETED: {final}")
        if _read_json(final / "config.json") != unit:
            raise RotatedArtifactError(f"completed Plan 11 configuration differs: {final}")
        manifest = _read_json(final / "manifest.json")
        if (
            manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
            or manifest.get("artifact_schema_version") != 1
            or manifest.get("metric_schema_version") != 1
            or manifest.get("status") != "completed"
            or manifest.get("config_hash") != canonical_hash(unit)
            or manifest.get("run_id") != final.name
        ):
            raise RotatedArtifactError(f"completed Plan 11 manifest is invalid: {final}")
        if any(not (final / name).is_file() for name in required):
            raise RotatedArtifactError(f"completed Plan 11 unit has missing artifacts: {final}")
        integrity_path = final / "integrity.json"
        if not integrity_path.is_file():
            raise RotatedArtifactError(f"completed Plan 11 unit lacks integrity ledger: {final}")
        integrity = _read_json(integrity_path)
        if set(integrity) != set(required) or any(
            integrity[name] != file_hash(final / name) for name in required
        ):
            raise RotatedArtifactError(f"completed Plan 11 unit has corrupt artifacts: {final}")
        return final

    def begin(self, unit: dict[str, Any], required: tuple[str, ...], *, resume: bool) -> RotatedRunSession | None:
        complete = self.completed(unit, required)
        if complete is not None:
            if not resume:
                raise RotatedCompletedRunError(f"Plan 11 unit already complete: {complete}")
            return None
        final, working = self.paths(unit)
        if working.exists():
            if not resume:
                raise RotatedIncompleteRunError(f"Plan 11 unit incomplete; pass --resume: {working}")
            try:
                existing = _read_json(working / "config.json")
                manifest = _read_json(working / "manifest.json")
            except RotatedArtifactError:
                existing, manifest = None, None
            if existing is not None and existing != unit:
                raise RotatedArtifactError(f"incomplete Plan 11 configuration differs: {working}")
            if existing == unit and isinstance(manifest, dict) and manifest.get("status") == "incomplete":
                return RotatedRunSession(final.name, working, final)
            working.rename(working.with_name(f"{working.name}.quarantine-{time.time_ns()}"))
        working.mkdir(parents=True, exist_ok=False)
        _write_json(working / "config.json", unit)
        _write_json(
            working / "manifest.json",
            {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "artifact_schema_version": 1,
                "metric_schema_version": 1,
                "run_kind": "plan11_asset" if unit["schedule"] is None else "plan11_trajectory",
                "run_id": final.name,
                "config_hash": canonical_hash(unit),
                "study_hash": self.study.config_hash,
                "phase": unit["phase"],
                "replica_id": f'replica-{unit["replica_index"]:04d}',
                "seeds": derive_seed_map(unit["replica_seed"], SEED_COMPONENTS),
                "runtime": runtime_metadata(self.repo_root, self.study.protocol),
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
            },
        )
        return RotatedRunSession(final.name, working, final)

    def finish(self, session: RotatedRunSession, required: tuple[str, ...]) -> Path:
        integrity = {name: file_hash(session.working_path / name) for name in required}
        session.write_json("integrity.json", integrity)
        return session.complete((*required, "integrity.json"))
