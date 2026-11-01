"""Plan 16 provenance and bounded runtime using the tested immutable unit store."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from mnist_experiment.fisher_memory.artifacts import Store, ArtifactError
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, utc_now
from mnist_experiment.fisher_memory_regression.runtime import RuntimeBudget, file_bytes, file_digest
from . import PROTOCOL


REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "cache/mnist_experiment/plan16"


def read_json(path):
    return json.loads(Path(path).read_text())


def source_manifest():
    paths = [p for p in Path(__file__).parent.glob("*.py")
             if p.name not in {"report.py", "analysis.py", "control.py"}]
    paths += [REPO / p for p in (
        "mnist_experiment/rotated_mnist/plan12/gauge.py", "src/parameters.py",
        "src/seeding.py", "src/mnist_model.py", "src/mnist_data.py",
        "mnist_experiment/fisher_memory/artifacts.py",
        "mnist_experiment/fisher_memory/control.py",
        "mnist_experiment/fisher_memory_regression/runtime.py")]
    return {str(p.relative_to(REPO)): file_digest(p) for p in sorted(paths)}


def provenance(cfg, data_root):
    files = {}
    for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                 "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        path = Path(data_root) / "MNIST/raw" / name
        if not path.is_file():
            raise FileNotFoundError(f"cached MNIST missing: {path}")
        files[name] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    return {"protocol": PROTOCOL, "allocation_family": "plan16", "schema": 1,
            "config": cfg.mapping(), "config_hash": cfg.scientific_hash,
            "sources": source_manifest(), "dataset": files,
            "git_commit": git("rev-parse", "HEAD"), "git_dirty_at_creation": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": cfg.device, "dtype": cfg.dtype,
            "device_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else "CPU"}


def setup(cfg):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan16-mpl")
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use direct GPU access, not a silent CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def peer_seconds(root):
    total = 0.
    if FAMILY.exists():
        for peer in FAMILY.iterdir():
            if peer.resolve() == Path(root).resolve() or not (peer / "identity.json").is_file():
                continue
            runtime = peer / "runtime.json"
            if runtime.exists():
                value = read_json(runtime).get("cumulative_active_seconds", 0.)
                lease = peer / "timing_lease.json"
                if lease.exists():
                    value = max(value, read_json(lease).get("charged_until", 0.))
                total += value
    engineering = FAMILY / "acceptance_compute.json"
    if engineering.exists():
        total += sum(item["seconds"] for item in read_json(engineering).get("entries", []))
    return total


def open_store(cfg, root, data_root, *, resume, clear_stop, allocation_hours, max_wall_seconds):
    setup(cfg)
    FAMILY.mkdir(parents=True, exist_ok=True)
    lock = (FAMILY / ".allocation.lock").open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ArtifactError("another Plan 16 process owns the cumulative allocation")
    try:
        return _open_locked(cfg, root, data_root, resume=resume, clear_stop=clear_stop,
                            allocation_hours=allocation_hours, max_wall_seconds=max_wall_seconds,
                            allocation_lock=lock)
    except BaseException:
        lock.close()
        raise


def _open_locked(cfg, root, data_root, *, resume, clear_stop, allocation_hours,
                 max_wall_seconds, allocation_lock):
    identity_path = Path(root) / "identity.json"
    identity = provenance(cfg, data_root)
    if identity_path.exists():
        prior = read_json(identity_path)["identity"]
        # Git dirtiness/commit are provenance, scientific files are identity.
        identity["git_commit"] = prior["git_commit"]
        identity["git_dirty_at_creation"] = prior["git_dirty_at_creation"]
    peers = peer_seconds(root)
    parent_bytes = file_bytes(FAMILY, exclude=Path(root))
    parent_bytes += file_bytes(REPO / "cache/mnist_experiment/plan17")
    parent_bytes += file_bytes(Path(data_root))
    store = Store(Path(root), identity, resume=resume, clear_stop=clear_stop,
                  max_bytes=500_000_000_000, free_reserve_bytes=20_000_000_000,
                  external_bytes=parent_bytes, finalization_reserve_bytes=64 * 1024**2,
                  max_wall_seconds=max_wall_seconds,
                  max_active_seconds=max(0., allocation_hours * 3600 - peers))
    budget = RuntimeBudget(store, other_seconds=peers, allocation_seconds=allocation_hours * 3600)
    store.plan16_allocation_lock = allocation_lock
    return store, budget
