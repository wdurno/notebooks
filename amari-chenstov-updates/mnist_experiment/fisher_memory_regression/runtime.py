"""Phase 8 provenance, shared accounting, and bounded crash-time reservations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from mnist_experiment.fisher_memory.artifacts import Store, ArtifactError
from mnist_experiment.fisher_memory.control import ResourceLimit, utc_now

from . import PROTOCOL
from .config import ROOT, PACKAGE, Config, content_hash

LEASE_SECONDS = 60.
HEARTBEAT_SECONDS = 10.


def read_json(path: Path):
    def invalid(value):
        raise ArtifactError(f"nonfinite JSON value {value}: {path}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def file_digest(path: Path, check=lambda: None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            check()
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def file_bytes(root: Path, exclude: Path | None = None, check=lambda: None) -> int:
    """Count regular-file bytes with bounded stop checks and no symlink traversal.

    Missing/non-directory roots count as zero, matching the legacy helper.
    Reject root and encountered entry symlinks, including directory links and
    dangling links. Excluded subtrees are pruned, not enumerated. Enumeration
    errors propagate rather than silently undercounting the storage budget.
    """
    check()
    root = Path(root)
    if root.is_symlink():
        raise ValueError(f"budget dependency must not be a symlink: {root}")
    if not root.is_dir():
        return 0
    excluded = Path(exclude).resolve() if exclude is not None else None
    if excluded is not None and root.resolve() == excluded:
        return 0
    total, pending = 0, [root]
    while pending:
        check()
        directory = pending.pop()
        if directory.is_symlink():
            raise ValueError(f"budget dependency must not be a symlink: {directory}")
        with os.scandir(directory) as entries:
            for entry in entries:
                check()
                if entry.is_symlink():
                    raise ValueError(f"budget dependency must not be a symlink: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    path = Path(entry.path)
                    if excluded is None or path.resolve() != excluded:
                        pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
    check()
    return total


def scientific_sources() -> dict:
    paths = [p for p in PACKAGE.glob("*.py") if p.name not in ("report.py", "acceptance.py")]
    paths.extend(ROOT / name for name in (
        "mnist_experiment/fisher_memory/artifacts.py", "mnist_experiment/fisher_memory/control.py",
        "mnist_experiment/fisher_memory/learner.py", "mnist_experiment/fisher_memory/model.py",
        "mnist_experiment/fisher_memory/data.py", "mnist_experiment/fisher_memory/config.py",
        "mnist_experiment/fisher_memory/initialize.py",
        "mnist_experiment/fisher_memory/run.py", "mnist_experiment/rotated_mnist/plan12/gauge.py",
        "mnist_experiment/rotated_mnist/transform.py", "mnist_experiment/rotated_mnist/config.py",
        "src/parameters.py", "src/seeding.py", "src/mnist_data.py", "src/mnist_model.py",
        "src/lanczos_wrapper.py", "src/lanczos.py", "src/representations.py"))
    return {str(p.relative_to(ROOT)): file_digest(p) for p in sorted(set(paths))}


def dataset_manifest(root: Path, check=lambda: None) -> dict:
    result = {}
    for stem in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                 "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        check()
        path = root / "MNIST" / "raw" / stem
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"cached MNIST required; no automatic download: {path}")
        result[str(path.relative_to(root))] = {"bytes": path.stat().st_size,
                                              "sha256": file_digest(path, check)}
    return result


def provenance(cfg: Config, data: dict) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    sources = scientific_sources()
    return dict(schema_version=1, protocol=PROTOCOL, config=cfg.mapping(),
                config_hash=cfg.scientific_hash, source_hash=content_hash(sources), sources=sources,
                dataset_manifest=data, data_hash=content_hash(data),
                git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
                python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                device=cfg.device, dtype=cfg.dtype,
                device_name=torch.cuda.get_device_name(0) if cfg.device == "cuda" else "CPU",
                dependencies={name: importlib.metadata.version(name)
                              for name in ("numpy", "torchvision", "nbformat", "matplotlib")})


def other_compute_seconds(budget_root: Path, root: Path) -> float:
    """Production/acceptance roots, including superseded versions, share 20 hours.

    Smoke runtime is charged once by its acceptance record, not again here.
    Scientific code versions may differ while retaining this allocation family.
    """
    total = 0.
    for peer in budget_root.iterdir():
        if peer.resolve() == root.resolve() or not peer.is_dir():
            continue
        identity = peer / "identity.json"
        if not identity.exists():
            continue
        info = read_json(identity).get("identity", {})
        if info.get("allocation_family") != "plan15-phase8" or info.get("role") == "smoke":
            continue
        runtime = peer / "runtime.json"
        if runtime.exists():
            value = read_json(runtime)["cumulative_active_seconds"]
            lease = peer / "timing_lease.json"
            if lease.exists():
                value = max(value, read_json(lease).get("charged_until", value))
            if not math.isfinite(value) or value < 0:
                raise ArtifactError("invalid peer Phase 8 time ledger")
            total += value
    return total


class RuntimeBudget:
    """Reserve the next bounded interval before work; recover it after hard kills."""

    def __init__(self, store: Store, *, other_seconds=0., allocation_seconds=72_000.):
        self.store = store
        self.other_seconds = float(other_seconds)
        self.allocation_seconds = float(allocation_seconds)
        for name, value in (("other_seconds", self.other_seconds),
                            ("allocation_seconds", self.allocation_seconds)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.last_pulse = 0.
        self._pulsing = False
        path = store.root / "timing_lease.json"
        if path.exists():
            prior = read_json(path)
            if prior.get("running"):
                until = prior["charged_until"]
                if not math.isfinite(until) or until < 0:
                    raise ArtifactError("invalid crash-time reservation")
                store.control.restore_active_seconds(until)
        self.pulse(force=True)

    @property
    def used(self):
        return self.other_seconds + self.store.control.active_seconds

    @property
    def remaining(self):
        return max(0., self.allocation_seconds - self.used)

    def check(self):
        self.store.check()
        if self.used >= self.allocation_seconds:
            raise ResourceLimit("aggregate Phase 8 compute allocation exhausted")
        self.pulse()

    def pulse(self, *, force=False):
        if self._pulsing or (not force and time.monotonic() - self.last_pulse < HEARTBEAT_SECONDS):
            return
        self._pulsing = True
        try:
            self.store.write_json(None, "timing_lease.json", {
                "schema_version": 1, "running": True, "pid": os.getpid(),
                "charged_until": self.store.control.active_seconds + LEASE_SECONDS,
                "heartbeat": utc_now(), "reservation_seconds": LEASE_SECONDS,
            }, finalizing=True)
            self.store.heartbeat(force=True)
            self.last_pulse = time.monotonic()
        finally:
            self._pulsing = False

    def close(self):
        self.store.write_json(None, "timing_lease.json", {
            "schema_version": 1, "running": False,
            "charged_until": self.store.control.active_seconds,
            "heartbeat": utc_now(), "reservation_seconds": 0.,
        }, finalizing=True)
        self.store.heartbeat(force=True)


def setup_torch(cfg: Config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan15-regression-matplotlib")
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; production must not silently fall back to CPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
