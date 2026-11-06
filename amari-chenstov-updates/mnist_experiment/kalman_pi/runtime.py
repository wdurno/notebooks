"""Plan 17 identity, family lease, and shared bounded immutable storage."""

from __future__ import annotations

import fcntl
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from mnist_experiment.fisher_memory.artifacts import Store as BaseStore, ArtifactError
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, utc_now
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget as BaseRuntimeBudget,
    HEARTBEAT_SECONDS,
    LEASE_SECONDS,
    file_bytes,
    file_digest,
)
from .config import PROTOCOL


REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "cache/mnist_experiment/plan17"
ALLOCATION_SECONDS = 20 * 3600.
MAX_BYTES = 500_000_000_000
FREE_RESERVE_BYTES = 20_000_000_000


def read_json(path):
    def invalid(value):
        raise ArtifactError(f"nonfinite JSON value {value}: {path}")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read valid JSON: {path}") from exc


def source_manifest(check=lambda: None):
    paths = [p for p in Path(__file__).parent.glob("*.py")
             if p.name not in {"report.py", "analysis.py", "control.py"}]
    paths += [REPO / p for p in (
        "mnist_experiment/diagonal_lfu/__init__.py",
        "mnist_experiment/diagonal_lfu/config.py", "mnist_experiment/diagonal_lfu/data.py",
        "mnist_experiment/diagonal_lfu/trajectory.py", "mnist_experiment/diagonal_lfu/curvature.py",
        "mnist_experiment/diagonal_lfu/learner.py",
        "mnist_experiment/rotated_mnist/plan12/gauge.py", "src/parameters.py",
        "src/seeding.py", "src/mnist_model.py", "src/mnist_data.py",
        "mnist_experiment/fisher_memory/artifacts.py",
        "mnist_experiment/fisher_memory/control.py",
        "mnist_experiment/fisher_memory_regression/__init__.py",
        "mnist_experiment/fisher_memory_regression/config.py",
        "mnist_experiment/fisher_memory_regression/runtime.py")]
    return {str(p.relative_to(REPO)): file_digest(p, check) for p in sorted(set(paths))}


def provenance(cfg, data_root, check=lambda: None):
    files = {}
    for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                 "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        check()
        path = Path(data_root) / "MNIST/raw" / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"cached MNIST missing; no automatic download: {path}")
        files[name] = {"sha256": file_digest(path, check), "bytes": path.stat().st_size}

    def git(*args):
        check()
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

    return {"protocol": PROTOCOL, "allocation_family": "plan17", "schema": 1,
            "metric_schema_version": 1, "artifact_schema_version": 1,
            "config": cfg.mapping(), "config_hash": cfg.scientific_hash,
            "sources": source_manifest(check), "dataset": files,
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty_at_creation": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": cfg.device, "dtype": cfg.dtype,
            "fisher_dtype": cfg.fisher_dtype,
            "device_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else "CPU"}


def setup(cfg):
    cfg.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan17-mpl")
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use direct GPU access, not a silent CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _seconds(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ArtifactError(f"invalid {label} time ledger")
    return float(value)


def _charged_seconds(root):
    runtime, lease = Path(root) / "runtime.json", Path(root) / "timing_lease.json"
    try:
        value = _seconds(read_json(runtime)["cumulative_active_seconds"], "runtime") if runtime.exists() else 0.
        if lease.exists():
            reservation = read_json(lease)
            if type(reservation["running"]) is not bool:
                raise ArtifactError("invalid crash-time reservation state")
            charged = _seconds(reservation["charged_until"], "crash-time reservation")
            # A clean lease may have flushed just before the runtime heartbeat.
            value = max(value, charged)
        return value
    except (KeyError, TypeError) as exc:
        raise ArtifactError(f"invalid time ledger: {root}") from exc


def peer_seconds(root, check=lambda: None):
    """All Plan 17 versions and unrooted acceptance overhead share 20 hours."""
    total = 0.
    if FAMILY.exists():
        for peer in sorted(FAMILY.iterdir()):
            check()
            if peer.resolve() == Path(root).resolve() or not (peer / "identity.json").is_file():
                continue
            if peer.is_symlink():
                raise ArtifactError("allocation peer must not be a symlink")
            info = read_json(peer / "identity.json")["identity"]
            if info.get("allocation_family", "plan17") == "plan17":
                total += _charged_seconds(peer)
    engineering = FAMILY / "acceptance_compute.json"
    if engineering.exists():
        for item in read_json(engineering).get("entries", []):
            check()
            total += _seconds(item["seconds"], "acceptance")
    return total


def external_bytes(root, data_root, check=lambda: None):
    """Count the union of both study families and the dataset, excluding self."""
    roots = [FAMILY, REPO / "cache/mnist_experiment/plan16", Path(data_root)]
    for path in roots:
        if path.is_symlink():
            raise ArtifactError(f"storage dependency must not be a symlink: {path}")
    roots = sorted(set(path.resolve() for path in roots), key=lambda path: len(path.parts))
    outer = []
    for path in roots:
        if not any(path.is_relative_to(parent) for parent in outer):
            outer.append(path)
    return sum(file_bytes(path, exclude=Path(root), check=check) for path in outer)


class Store(BaseStore):
    """Keep the tested Store API and release its additional family lock on close."""

    def _release_lock(self):
        try:
            super()._release_lock()
        finally:
            lock = getattr(self, "plan17_allocation_lock", None)
            if lock is not None and not lock.closed:
                lock.close()


class RuntimeBudget(BaseRuntimeBudget):
    """Reuse crash recovery, but cap reservations at admitted invocation bounds."""

    def __init__(self, store, *, other_seconds=0., allocation_seconds=ALLOCATION_SECONDS):
        if (type(allocation_seconds) not in (int, float) or not math.isfinite(allocation_seconds)
                or not 0 <= allocation_seconds <= ALLOCATION_SECONDS):
            raise ValueError("Plan 17 allocation_seconds must be finite and between zero and 72000")
        super().__init__(store, other_seconds=other_seconds, allocation_seconds=allocation_seconds)

    def check(self):
        self.store.check()
        if self.used >= self.allocation_seconds:
            raise ResourceLimit("aggregate Plan 17 compute allocation exhausted")
        self.pulse()

    def pulse(self, *, force=False):
        if self._pulsing or (not force and time.monotonic() - self.last_pulse < HEARTBEAT_SECONDS):
            return
        self._pulsing = True
        try:
            store = self.store
            if hasattr(store, "plan17_data_root"):
                store.external_bytes = external_bytes(store.root, store.plan17_data_root, store.control.check)
            active = store.control.active_seconds
            until = min(active + LEASE_SECONDS, self.allocation_seconds - self.other_seconds)
            if store.control.max_active_seconds is not None:
                until = min(until, store.control.max_active_seconds)
            if store.control.max_wall_seconds is not None:
                until = min(until, store.control.base_seconds + store.control.max_wall_seconds)
            until = max(active, until)
            store.write_json(None, "timing_lease.json", {
                "schema_version": 1, "running": True, "pid": os.getpid(),
                "charged_until": until, "heartbeat": utc_now(),
                "reservation_seconds": until - active,
            }, finalizing=True)
            store.heartbeat(force=True)
            self.last_pulse = time.monotonic()
        finally:
            self._pulsing = False


def open_store(cfg, root, data_root, *, resume, clear_stop, allocation_hours, max_wall_seconds):
    started = time.monotonic()
    if (type(allocation_hours) not in (int, float) or not math.isfinite(allocation_hours)
            or not 0 < allocation_hours <= ALLOCATION_SECONDS / 3600):
        raise ValueError("Plan 17 allocation_hours must be finite, positive, and at most 20")
    if max_wall_seconds is not None and (
            type(max_wall_seconds) not in (int, float) or not math.isfinite(max_wall_seconds)
            or max_wall_seconds < 0):
        raise ValueError("max_wall_seconds must be finite and nonnegative")
    if clear_stop and not resume:
        raise ValueError("clearing STOP requires resume=True")
    root, data_root = Path(root), Path(data_root)
    if root.is_symlink() or FAMILY.is_symlink():
        raise ArtifactError("study and allocation roots must not be symlinks")
    FAMILY.mkdir(parents=True, exist_ok=True)
    lock_path = FAMILY / ".allocation.lock"
    if lock_path.is_symlink():
        raise ArtifactError("allocation lock must not be a symlink")
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise ArtifactError("another Plan 17 process owns the cumulative allocation") from exc
    store = None
    try:
        own = _charged_seconds(root)
        peers = peer_seconds(root)
        limit = allocation_hours * 3600

        def check():
            if (root / "STOP").exists() and not clear_stop:
                raise StopRequested("stop requested; pass --resume --clear-stop explicitly")
            elapsed = time.monotonic() - started
            if max_wall_seconds is not None and elapsed >= max_wall_seconds:
                raise ResourceLimit("invocation wall-clock allocation exhausted")
            if own + peers + elapsed >= limit:
                raise ResourceLimit("aggregate Plan 17 compute allocation exhausted")

        check()
        setup(cfg)
        identity = provenance(cfg, data_root, check)
        identity_path = root / "identity.json"
        if identity_path.exists():
            prior = read_json(identity_path)["identity"]
            for key in ("git_commit", "git_dirty_at_creation"):
                identity[key] = prior[key]
        outside = external_bytes(root, data_root, check)
        check()
        remaining_wall = None if max_wall_seconds is None else max_wall_seconds - (time.monotonic() - started)
        store = Store(root, identity, resume=resume, clear_stop=clear_stop,
                      max_bytes=MAX_BYTES, free_reserve_bytes=FREE_RESERVE_BYTES,
                      external_bytes=outside, finalization_reserve_bytes=64 * 1024**2,
                      max_wall_seconds=remaining_wall, max_active_seconds=max(0., limit - peers))
        store.plan17_allocation_lock = lock
        store.plan17_data_root = data_root
        store.control.restore_active_seconds(own + time.monotonic() - started)
        budget = RuntimeBudget(store, other_seconds=peers, allocation_seconds=limit)
        budget.check()
        return store, budget
    except BaseException:
        if store is not None:
            store.close()
        lock.close()
        raise
