"""Immutable Plan 9 artifact lifecycle with content validation."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from ..artifacts import RotatedArtifactError, _git_metadata, _utc_now, runtime_metadata


PLAN9_MANIFEST_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, allow_nan=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedArtifactError(f"could not read {path}: {exc}") from exc


def _safe_path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    if value.is_absolute() or value == Path(".") or ".." in value.parts:
        raise RotatedArtifactError("artifact paths must be nonempty and relative")
    return root / value


@dataclasses.dataclass
class Plan9Session:
    run_id: str
    working_path: Path
    final_path: Path
    _completed: bool = False

    def write_json(self, relative: str | Path, value: Any) -> Path:
        if self._completed:
            raise RotatedArtifactError("cannot mutate a completed Plan 9 run")
        destination = _safe_path(self.working_path, relative)
        write_json(destination, value)
        return destination

    def write_torch(self, relative: str | Path, value: Any) -> Path:
        if self._completed:
            raise RotatedArtifactError("cannot mutate a completed Plan 9 run")
        destination = _safe_path(self.working_path, relative)
        buffer = io.BytesIO()
        torch.save(value, buffer)
        _atomic_write(destination, buffer.getvalue())
        return destination

    def mark_interrupted(self, reason: str) -> None:
        write_json(
            self.working_path / "INTERRUPTED.json",
            {"interrupted_at": _utc_now(), "reason": reason},
        )

    def complete(self, required: Iterable[str]) -> Path:
        names = tuple(required)
        missing = [name for name in names if not _safe_path(self.working_path, name).is_file()]
        if missing:
            raise RotatedArtifactError(f"cannot complete Plan 9 run; missing {missing}")
        hashes = {name: file_sha256(_safe_path(self.working_path, name)) for name in names}
        manifest = read_json(self.working_path / "manifest.json")
        manifest.update(
            {
                "status": "completed",
                "completed_at": _utc_now(),
                "artifact_sha256": hashes,
            }
        )
        write_json(self.working_path / "manifest.json", manifest)
        _atomic_write(self.working_path / "COMPLETED", b"")
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.working_path, self.final_path)
        self._completed = True
        return self.final_path


class Plan9RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"

    def begin(
        self,
        *,
        run_id: str,
        run_kind: str,
        config: Mapping[str, Any],
        config_hash: str,
        repo_root: str | Path,
        runtime_config: Any,
        resume: bool = False,
    ) -> Plan9Session:
        if not run_id or Path(run_id).name != run_id:
            raise RotatedArtifactError("invalid Plan 9 run ID")
        final = self.root / run_id
        working = self.incomplete_root / run_id
        if (final / "COMPLETED").is_file():
            raise RotatedArtifactError(f"Plan 9 run already completed: {run_id}")
        if final.exists():
            raise RotatedArtifactError("Plan 9 final path lacks COMPLETED")
        if working.exists():
            if not resume:
                raise RotatedArtifactError(f"incomplete Plan 9 run exists: {run_id}")
            stored = read_json(working / "config.json")
            if stored != dict(config):
                raise RotatedArtifactError("incomplete Plan 9 config differs")
            interrupted = working / "INTERRUPTED.json"
            interrupted.unlink(missing_ok=True)
            return Plan9Session(run_id, working, final)
        working.mkdir(parents=True, exist_ok=False)
        write_json(working / "config.json", dict(config))
        write_json(
            working / "manifest.json",
            {
                "manifest_schema_version": PLAN9_MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
                "run_kind": run_kind,
                "config_hash": config_hash,
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
                "runtime": runtime_metadata(repo_root, runtime_config),
                "source_git": _git_metadata(Path(repo_root)),
            },
        )
        return Plan9Session(run_id, working, final)


def validate_completed(path: str | Path, *, required: Iterable[str]) -> dict[str, Any]:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise RotatedArtifactError(f"Plan 9 run is incomplete: {run_path}")
    manifest = read_json(run_path / "manifest.json")
    if manifest.get("status") != "completed" or manifest.get("run_id") != run_path.name:
        raise RotatedArtifactError("Plan 9 manifest is incompatible")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise RotatedArtifactError("Plan 9 manifest lacks artifact hashes")
    for name in required:
        artifact = _safe_path(run_path, name)
        if not artifact.is_file() or hashes.get(name) != file_sha256(artifact):
            raise RotatedArtifactError(f"Plan 9 artifact failed validation: {name}")
    return manifest


def discard_incomplete(path: str | Path) -> None:
    """Remove only an explicitly identified incomplete test run."""

    value = Path(path)
    if (value / "COMPLETED").exists():
        raise RotatedArtifactError("refusing to discard a completed run")
    shutil.rmtree(value)
