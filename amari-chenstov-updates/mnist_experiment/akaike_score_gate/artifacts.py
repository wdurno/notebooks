"""Immutable units and crash-safe checkpoints using the existing artifact API."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mnist_experiment.rotated_mnist.artifacts import (
    RotatedArtifactError, RotatedRunSession, _read_json, _write_json, _utc_now,
    runtime_metadata,
)
from .config import REPO_ROOT, Study, canonical_hash


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    paths = list((REPO_ROOT / "mnist_experiment/akaike_score_gate").glob("*.py"))
    paths.extend(REPO_ROOT / p for p in (
        "src/parameters.py", "src/derivatives.py", "src/seeding.py", "src/mnist_data.py", "src/mnist_model.py",
        "mnist_experiment/rotated_mnist/artifacts.py", "mnist_experiment/rotated_mnist/transform.py",
        "mnist_experiment/rotated_mnist/plan12/gauge.py"))
    return {p.relative_to(REPO_ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def freeze_json(path: Path, value: Any, *, resume: bool = True) -> None:
    if path.exists():
        if not resume or _read_json(path) != value:
            raise RotatedArtifactError(f"frozen configuration collision: {path}")
    else:
        _write_json(path, value)


class StopExecution(Exception):
    """A requested bounded invocation ended at a recoverable boundary."""


class RunLimit:
    def __init__(self, seconds: float | None = None, max_units: int | None = None):
        self.started = time.monotonic()
        self.seconds = seconds
        self.max_units = max_units
        self.units = 0
        self.stop_requested = False

    def check(self) -> None:
        if self.stop_requested or (self.seconds is not None and time.monotonic() - self.started >= self.seconds):
            raise StopExecution("wall limit or stop request; accepted work is checkpointed")
        if self.max_units is not None and self.units >= self.max_units:
            raise StopExecution("completed-unit limit reached")

    def completed(self) -> None:
        self.units += 1

    @contextlib.contextmanager
    def signal_handlers(self):
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        def request_stop(signum, frame):
            self.stop_requested = True
        try:
            for sig in old:
                signal.signal(sig, request_stop)
            yield self
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)


@contextlib.contextmanager
def run_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("a Plan 14 writer already owns this run") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class UnitStore:
    def __init__(self, root: Path, study: Study, *, limit: RunLimit | None = None):
        self.root = Path(root)
        self.study = study
        self.sources = source_hashes()
        self.limit = limit or RunLimit()
        self._validated: set[tuple[Path, tuple[str, ...]]] = set()

    def unit(self, phase: str, kind: str, index: int, *, environment: str = "shared",
             policy: str | None = None, detail: dict[str, Any] | None = None) -> dict[str, Any]:
        sources = self.sources
        if kind == "validation_profile":
            # Reporting and later phase dispatch do not execute in a timing
            # primitive. Keep its compatibility boundary at its actual code.
            modules = {"__init__.py", "phase0.py", "model.py", "evidence.py", "fisher.py",
                       "gate.py", "optimizer.py", "budget.py", "config.py", "artifacts.py"}
            sources = {name: digest for name, digest in sources.items()
                       if "akaike_score_gate/" not in name or Path(name).name in modules}
        return {"schema_version": 1, "study_hash": self.study.config_hash,
                "phase": phase, "kind": kind, "index": index,
                "environment": environment, "policy": policy,
                "detail": {} if detail is None else detail, "source_hashes": sources}

    def paths(self, unit: dict[str, Any]) -> tuple[Path, Path]:
        label = "__".join(str(unit[k]) for k in ("kind", "index", "environment", "policy"))
        name = f"{label}__{canonical_hash(unit)[:20]}"
        return self.root / unit["phase"] / name, self.root / unit["phase"] / ".incomplete" / name

    def completed(self, unit: dict[str, Any], required: tuple[str, ...]) -> Path | None:
        final, _ = self.paths(unit)
        if not final.exists():
            return None
        key = (final, required)
        if key in self._validated:
            return final
        if not (final / "COMPLETED").is_file() or _read_json(final / "config.json") != unit:
            raise RotatedArtifactError(f"invalid completed unit: {final}")
        manifest = _read_json(final / "manifest.json")
        if (manifest.get("status") != "completed" or manifest.get("config_hash") != canonical_hash(unit)
                or manifest.get("artifact_schema_version") != 1):
            raise RotatedArtifactError(f"incompatible completed manifest: {final}")
        integrity = _read_json(final / "integrity.json")
        if not set(required).issubset(integrity):
            raise RotatedArtifactError(f"required artifacts not sealed: {final}")
        for name, digest in integrity.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise RotatedArtifactError("invalid integrity path")
            if not (final / name).is_file() or file_hash(final / name) != digest:
                raise RotatedArtifactError(f"corrupt immutable artifact: {final / name}")
        self._validated.add(key)
        return final

    def begin(self, unit: dict[str, Any], required: tuple[str, ...], *, resume: bool) -> RotatedRunSession | None:
        if self.completed(unit, required) is not None:
            if not resume:
                raise RotatedArtifactError("completed unit exists; use --resume to skip")
            return None
        self.limit.check()
        final, working = self.paths(unit)
        if working.exists():
            if not resume:
                raise RotatedArtifactError("incomplete unit exists; use --resume")
            # A halt between sealing and rename is recoverable without rewriting contents.
            if (working / "COMPLETED").is_file():
                if _read_json(working / "config.json") != unit:
                    raise RotatedArtifactError("sealed staging configuration mismatch")
                final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(working, final)
                self.completed(unit, required)
                return None
            if not (working / "config.json").exists() or not (working / "manifest.json").exists():
                working.rename(working.with_name(working.name + f".quarantine-{time.time_ns()}"))
            else:
                if _read_json(working / "config.json") != unit:
                    raise RotatedArtifactError("incomplete unit identity mismatch")
                manifest = _read_json(working / "manifest.json")
                if manifest["status"] == "completed":
                    # Manifest was flushed but the completion marker was not; revalidate on finish.
                    manifest["status"] = "incomplete"
                    manifest["completed_at"] = None
                    _write_json(working / "manifest.json", manifest)
                return RotatedRunSession(final.name, working, final)
        working.mkdir(parents=True, exist_ok=False)
        _write_json(working / "config.json", unit)
        runtime = runtime_metadata(REPO_ROOT, SimpleNamespace(runtime=SimpleNamespace(
            device=self.study.device, dtype=self.study.dtype)))
        _write_json(working / "manifest.json", {
            "manifest_schema_version": 1, "artifact_schema_version": 1,
            "metric_schema_version": 1, "config_hash": canonical_hash(unit),
            "run_id": final.name, "replica_id": f'{unit["phase"]}-{unit["index"]}',
            "study_hash": self.study.config_hash, "phase": unit["phase"],
            "environment": unit["environment"], "runtime": runtime,
            "seeds": {component: self.study.seed(f'{unit["phase"]}:{unit["environment"]}:{component}', unit["index"])
                      for component in ("initialization", "stream", "reference", "model_labels", "bootstrap")},
            "status": "incomplete", "started_at": _utc_now(), "completed_at": None,
        })
        return RotatedRunSession(final.name, working, final)

    def finish(self, session: RotatedRunSession, required: tuple[str, ...]) -> Path:
        names = tuple(dict.fromkeys((*required, "config.json")))
        session.write_json("integrity.json", {n: file_hash(session.path / n) for n in names})
        path = session.complete((*names, "integrity.json"))
        self.limit.completed()
        return path

    def log(self, message: str, **detail: Any) -> None:
        path = self.root / "lab_log.json"
        rows = _read_json(path) if path.exists() else []
        rows.append({"time": _utc_now(), "message": message, **detail})
        _write_json(path, rows)
