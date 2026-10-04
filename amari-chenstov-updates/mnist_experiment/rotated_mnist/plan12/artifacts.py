"""Immutable, content-validated units for the Plan 12 study."""

from __future__ import annotations

import hashlib
import json
import os
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
from .config import ARTIFACT_SCHEMA_VERSION, METRIC_SCHEMA_VERSION, Plan12Study, canonical_hash


CORE_SOURCE_FILES = (
    "mnist_experiment/rotated_mnist/run.py",
    "mnist_experiment/rotated_mnist/run_phase3.py",
    "mnist_experiment/rotated_mnist/run_phase5_double_lap.py",
    "mnist_experiment/rotated_mnist/run_phase8.py",
    "mnist_experiment/rotated_mnist/data.py",
    "mnist_experiment/rotated_mnist/phase4_metrics.py",
    "mnist_experiment/rotated_mnist/schedule.py",
    "mnist_experiment/rotated_mnist/transform.py",
    "src/derivatives.py",
    "src/ewc.py",
    "src/hybrid.py",
    "src/initialization.py",
    "src/lanczos_wrapper.py",
    "src/mnist_model.py",
    "src/parameters.py",
    "src/representations.py",
)

SEED_COMPONENTS = (
    "plan5_model_initialization",
    "plan5_data_partition",
    "plan5_initialization_loader",
    "plan5_online_stream",
    "plan5_reference_stream",
    "plan5_evaluation_panel",
    "plan12_initial_lanczos",
    "plan12_update_lanczos",
    "plan12_local_batch",
    "plan12_target_fit",
    "plan12_mp_resample",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(repo_root: Path) -> dict[str, str]:
    plan12 = sorted(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "mnist_experiment/rotated_mnist/plan12").glob("*.py")
    )
    names = tuple(dict.fromkeys((*CORE_SOURCE_FILES, *plan12)))
    missing = [name for name in names if not (repo_root / name).is_file()]
    if missing:
        raise RotatedArtifactError(f"Plan 12 source manifest has missing files: {missing}")
    return {name: file_hash(repo_root / name) for name in names}


def _safe_label(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "-" for character in value)


class UnitStore:
    def __init__(self, root: Path, study: Plan12Study, repo_root: Path):
        self.root = root
        self.study = study
        self.repo_root = repo_root
        self.sources = source_hashes(repo_root)
        self._validated_completed: set[tuple[Path, tuple[str, ...]]] = set()

    def unit(
        self,
        phase: str,
        kind: str,
        index: int,
        *,
        schedule: str | None = None,
        condition: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "study_hash": self.study.config_hash,
            "phase": phase,
            "kind": kind,
            "index": int(index),
            "schedule": schedule,
            "condition": condition,
            "detail": {} if detail is None else detail,
            "source_hashes": self.sources,
        }

    def paths(self, unit: dict[str, Any]) -> tuple[Path, Path]:
        digest = canonical_hash(unit)[:16]
        pieces = [unit["phase"], unit["kind"], f'{unit["index"]:05d}']
        if unit["schedule"] is not None:
            pieces.append(unit["schedule"])
        if unit["condition"] is not None:
            pieces.append(unit["condition"])
        run_id = _safe_label("__".join(pieces)) + f"__{digest}"
        final = self.root / unit["phase"] / run_id
        working = self.root / unit["phase"] / ".incomplete" / run_id
        return final, working

    def completed(self, unit: dict[str, Any], required: tuple[str, ...]) -> Path | None:
        final, _ = self.paths(unit)
        if not final.exists():
            return None
        cache_key = (final, required)
        if cache_key in self._validated_completed:
            return final
        if not (final / "COMPLETED").is_file():
            raise RotatedArtifactError(f"final Plan 12 path lacks COMPLETED: {final}")
        if _read_json(final / "config.json") != unit:
            raise RotatedArtifactError(f"completed Plan 12 configuration differs: {final}")
        manifest = _read_json(final / "manifest.json")
        expected = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "status": "completed",
            "config_hash": canonical_hash(unit),
            "run_id": final.name,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RotatedArtifactError(f"completed Plan 12 manifest is invalid: {final}")
        if any(not (final / name).is_file() for name in required):
            raise RotatedArtifactError(f"completed Plan 12 unit has missing artifacts: {final}")
        integrity = _read_json(final / "integrity.json")
        if set(integrity) != set(required) or any(
            integrity[name] != file_hash(final / name) for name in required
        ):
            raise RotatedArtifactError(f"completed Plan 12 unit has corrupt artifacts: {final}")
        self._validated_completed.add(cache_key)
        return final

    def begin(
        self,
        unit: dict[str, Any],
        required: tuple[str, ...],
        *,
        resume: bool,
    ) -> RotatedRunSession | None:
        complete = self.completed(unit, required)
        if complete is not None:
            if not resume:
                raise RotatedCompletedRunError(f"Plan 12 unit already complete: {complete}")
            return None
        final, working = self.paths(unit)
        if working.exists():
            if not resume:
                raise RotatedIncompleteRunError(f"Plan 12 unit incomplete; pass --resume: {working}")
            try:
                existing = _read_json(working / "config.json")
                manifest = _read_json(working / "manifest.json")
            except RotatedArtifactError:
                existing, manifest = None, None
            if existing == unit and isinstance(manifest, dict) and manifest.get("status") == "incomplete":
                return RotatedRunSession(final.name, working, final)
            working.rename(working.with_name(f"{working.name}.quarantine-{time.time_ns()}"))
        working.mkdir(parents=True, exist_ok=False)
        _write_json(working / "config.json", unit)
        replica_seed = self.study.seed(f'{unit["phase"]}:{unit["kind"]}', unit["index"])
        _write_json(
            working / "manifest.json",
            {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "run_kind": f'plan12_{unit["kind"]}',
                "run_id": final.name,
                "config_hash": canonical_hash(unit),
                "study_hash": self.study.config_hash,
                "phase": unit["phase"],
                "replica_id": f'{unit["kind"]}-{unit["index"]:05d}',
                "seeds": derive_seed_map(replica_seed, SEED_COMPONENTS),
                "runtime": runtime_metadata(self.repo_root, self.study.protocol),
                "pid": os.getpid(),
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

    def record_failure(self, unit: dict[str, Any], error: BaseException) -> Path:
        path = self.root / "failures" / f"{canonical_hash(unit)}.json"
        _write_json(
            path,
            {
                "unit": unit,
                "error_type": type(error).__name__,
                "error": str(error),
                "recorded_at": _utc_now(),
            },
        )
        return path


def freeze_json(path: Path, value: dict[str, Any], *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise RotatedArtifactError(f"Plan 12 ledger exists; pass --resume: {path}")
        if _read_json(path) != value:
            raise RotatedArtifactError(f"Plan 12 ledger differs from frozen run: {path}")
        return
    _write_json(path, value)
