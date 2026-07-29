"""Immutable run lifecycle and versioned metadata manifests."""

from __future__ import annotations

import json
import io
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import ExperimentConfig
from .seeding import SEED_SCHEMA_VERSION, derive_seed_map

MANIFEST_SCHEMA_VERSION = 1


class RunStateError(RuntimeError):
    """Raised when a run directory is in an invalid lifecycle state."""


class CompletedRunError(RunStateError):
    """Raised when code attempts to mutate or replace a completed run."""


class IncompleteRunError(RunStateError):
    """Raised when an incomplete run exists but resume was not requested."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunStateError(f"could not read run artifact {path}: {exc}") from exc


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def collect_runtime_metadata(
    repo_root: str | Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    try:
        import numpy
    except ImportError:
        numpy_version = None
    else:
        numpy_version = numpy.__version__

    try:
        import torch
    except ImportError:
        torch_metadata: dict[str, Any] = {"version": None}
    else:
        cuda_available = torch.cuda.is_available()
        torch_metadata = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "gpu_capability": (
                list(torch.cuda.get_device_capability(0))
                if cuda_available
                else None
            ),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": numpy_version,
        "torch": torch_metadata,
        "requested_device": config.runtime.device,
        "training_dtype": config.runtime.training_dtype,
        "matrix_dtype": config.runtime.matrix_dtype,
        "git": _git_metadata(Path(repo_root)),
    }


def _safe_artifact_path(run_path: Path, relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError("artifact path must be a nonempty relative path")
    return run_path / relative


@dataclass
class RunSession:
    run_id: str
    working_path: Path
    final_path: Path
    _completed: bool = False

    @property
    def path(self) -> Path:
        return self.final_path if self._completed else self.working_path

    def write_json(self, relative_path: str | Path, value: Any) -> Path:
        if self._completed:
            raise CompletedRunError(f"run {self.run_id} is complete")
        destination = _safe_artifact_path(self.working_path, relative_path)
        if destination.name in {"config.json", "manifest.json", "COMPLETED"}:
            raise RunStateError(f"{destination.name} is managed by the run lifecycle")
        _atomic_write_json(destination, value)
        return destination

    def write_torch(self, relative_path: str | Path, value: Any) -> Path:
        if self._completed:
            raise CompletedRunError(f"run {self.run_id} is complete")
        destination = _safe_artifact_path(self.working_path, relative_path)
        if destination.name in {"config.json", "manifest.json", "COMPLETED"}:
            raise RunStateError(f"{destination.name} is managed by the run lifecycle")
        try:
            import torch
        except ImportError as exc:
            raise RunStateError("PyTorch is required for tensor artifacts") from exc
        buffer = io.BytesIO()
        torch.save(value, buffer)
        _atomic_write_bytes(destination, buffer.getvalue())
        return destination

    def complete(self, required_artifacts: Iterable[str | Path] = ()) -> Path:
        if self._completed:
            raise CompletedRunError(f"run {self.run_id} is complete")
        if self.final_path.exists():
            raise CompletedRunError(
                f"completed run destination already exists: {self.final_path}"
            )

        missing = [
            str(relative_path)
            for relative_path in required_artifacts
            if not _safe_artifact_path(
                self.working_path, relative_path
            ).exists()
        ]
        if missing:
            raise RunStateError(f"cannot complete run; missing artifacts: {missing}")

        manifest_path = self.working_path / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["status"] = "completed"
        manifest["completed_at"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        _atomic_write_bytes(self.working_path / "COMPLETED", b"")

        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.working_path, self.final_path)
        self._completed = True
        return self.final_path


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"

    def begin(
        self,
        config: ExperimentConfig,
        repo_root: str | Path,
        *,
        resume: bool = False,
    ) -> RunSession:
        config.validate()
        run_id = config.run_id
        final_path = self.root / run_id
        working_path = self.incomplete_root / run_id

        if final_path.exists():
            if (final_path / "COMPLETED").is_file():
                raise CompletedRunError(f"run already completed: {run_id}")
            raise RunStateError(
                f"final run directory exists without COMPLETED: {final_path}"
            )

        if working_path.exists():
            if not resume:
                raise IncompleteRunError(
                    f"incomplete run exists; pass resume=True: {run_id}"
                )
            stored_config = _read_json(working_path / "config.json")
            if stored_config != config.to_mapping():
                raise RunStateError(
                    f"incomplete run configuration does not match: {run_id}"
                )
            manifest = _read_json(working_path / "manifest.json")
            if manifest.get("status") != "incomplete":
                raise RunStateError(
                    f"incomplete run has invalid manifest status: {run_id}"
                )
            return RunSession(run_id, working_path, final_path)

        working_path.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(working_path / "config.json", config.to_mapping())
        manifest = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact_schema_version": config.artifact_schema_version,
            "metric_schema_version": config.metric_schema_version,
            "seed_schema_version": SEED_SCHEMA_VERSION,
            "run_id": run_id,
            "experiment": config.experiment,
            "replica_id": config.replica_id,
            "config_hash": config.config_hash,
            "status": "incomplete",
            "started_at": _utc_now(),
            "completed_at": None,
            "seeds": derive_seed_map(config.replica_seed),
            "runtime": collect_runtime_metadata(repo_root, config),
        }
        _atomic_write_json(working_path / "manifest.json", manifest)
        return RunSession(run_id, working_path, final_path)
