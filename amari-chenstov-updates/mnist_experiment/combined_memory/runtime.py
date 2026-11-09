"""Independent Plan 18 allocation using the established atomic artifact store."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import platform
import subprocess
import time

import torch

from mnist_experiment.fisher_memory.artifacts import Store as BaseStore, ArtifactError
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, utc_now
from mnist_experiment.fisher_memory_regression.runtime import file_bytes, file_digest, read_json
from mnist_experiment.kalman_pi.runtime import RuntimeBudget as BaseBudget, _charged_seconds
from . import PROTOCOL


REPO = Path(__file__).resolve().parents[2]
FAMILY = REPO / "cache/mnist_experiment/plan18"
DATA_ROOT = REPO / "cache/mnist_experiment/datasets"


def setup(cfg):
    cfg.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan18-matplotlib")
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; Plan 18 production requires direct GPU access")


def sources(check=lambda: None):
    files = [p for p in Path(__file__).parent.glob("*.py")
             if p.name not in {"report.py", "acceptance.py"}]
    files.extend(REPO / name for name in (
        "src/parameters.py", "src/seeding.py", "src/mnist_data.py",
        "mnist_experiment/fisher_memory/artifacts.py", "mnist_experiment/fisher_memory/control.py",
        "mnist_experiment/fisher_memory/model.py", "mnist_experiment/fisher_memory/learner.py",
        "mnist_experiment/fisher_memory/data.py", "mnist_experiment/fisher_memory_registration/model.py",
        "mnist_experiment/diagonal_lfu/data.py", "mnist_experiment/diagonal_lfu/config.py",
        "mnist_experiment/diagonal_lfu/trajectory.py", "mnist_experiment/kalman_pi/runtime.py",
        "mnist_experiment/fisher_memory_regression/runtime.py",
        "mnist_experiment/fisher_memory_grid_registration/model.py",
        "mnist_experiment/fisher_memory_stm_capacity/model.py",
        "mnist_experiment/fisher_memory_registration/data.py",
        "mnist_experiment/kalman_pi/policies.py", "mnist_experiment/kalman_pi/observations.py",
        "mnist_experiment/kalman_pi/registration/data.py",
        "mnist_experiment/kalman_pi/discrepancy/config.py",
        "mnist_experiment/kalman_pi/discrepancy/data.py"))
    result = {}
    for path in sorted(set(files)):
        check()
        result[str(path.relative_to(REPO))] = file_digest(path, check)
    return result


def provenance(cfg, data_root, check=lambda: None):
    dataset = {}
    for path in sorted((Path(data_root) / "MNIST/raw").glob("*-ubyte")):
        check()
        dataset[path.name] = dict(sha256=file_digest(path, check), bytes=path.stat().st_size)
    if len(dataset) != 4:
        raise FileNotFoundError("four cached MNIST IDX files required; downloading is disabled")
    return dict(protocol=PROTOCOL, allocation_family="plan18", schema=1,
        metric_schema_version=1, artifact_schema_version=1,
        config=cfg.mapping(), config_hash=cfg.scientific_hash, sources=sources(check), dataset=dataset,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        git_dirty_at_creation=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        device=cfg.device, device_name=torch.cuda.get_device_name(0) if cfg.device == "cuda" else "CPU",
        dtype=cfg.dtype, fisher_dtype="float64", parameter_count=3376, ltm_count=3268, stm_count=108)


class Store(BaseStore):
    def _release_lock(self):
        try:
            super()._release_lock()
        finally:
            lock = getattr(self, "family_lock", None)
            if lock is not None and not lock.closed:
                lock.close()


def peer_seconds(root):
    total = 0.
    if FAMILY.exists():
        for path in FAMILY.iterdir():
            if path.resolve() != Path(root).resolve() and (path / "identity.json").is_file():
                if read_json(path / "identity.json")["identity"].get("allocation_family") == "plan18":
                    total += _charged_seconds(path)
    ledger = FAMILY / "acceptance_compute.json"
    if ledger.exists():
        record = read_json(ledger)
        total += float(record.get("charged_seconds", sum(float(e["seconds"]) for e in record.get("entries", []))))
    return total


class RuntimeBudget(BaseBudget):
    def check(self):
        self.store.check()
        if self.used >= self.allocation_seconds:
            raise ResourceLimit("Plan 18 cumulative 20-hour allocation exhausted")
        self.pulse()

    def pulse(self, *, force=False):
        if self._pulsing or (not force and time.monotonic() - self.last_pulse < 10):
            return
        self._pulsing = True
        try:
            store = self.store
            if time.monotonic() - getattr(self, "last_storage_scan", 0) > 60:
                store.external_bytes = (file_bytes(FAMILY, exclude=store.root)
                                        + file_bytes(store.data_root))
                self.last_storage_scan = time.monotonic()
            active = store.control.active_seconds
            until = min(active + 60, self.allocation_seconds - self.other_seconds)
            if store.control.max_wall_seconds is not None:
                until = min(until, store.control.base_seconds + store.control.max_wall_seconds)
            store.write_json(None, "timing_lease.json", dict(schema_version=1, running=True,
                pid=os.getpid(), charged_until=max(active, until), heartbeat=utc_now(),
                reservation_seconds=max(0., until-active)), finalizing=True)
            store.heartbeat(force=True)
            self.last_pulse = time.monotonic()
        finally:
            self._pulsing = False


def open_store(cfg, root, data_root=DATA_ROOT, *, resume=False, clear_stop=False,
               allocation_hours=20., max_wall_seconds=None):
    started = time.monotonic()
    if not 0 < allocation_hours <= 20:
        raise ValueError("Plan 18 allocation must be in (0,20] hours")
    FAMILY.mkdir(parents=True, exist_ok=True)
    family_lock = (FAMILY / ".allocation.lock").open("a+b")
    store = None
    try:
        fcntl.flock(family_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path(root)
        peers, own = peer_seconds(root), _charged_seconds(root)
        limit = allocation_hours * 3600
        if peers + own >= limit:
            raise ResourceLimit("Plan 18 allocation already exhausted")
        setup(cfg)
        identity = provenance(cfg, data_root)
        if (root / "identity.json").exists():
            previous = read_json(root / "identity.json")["identity"]
            for key in ("git_commit", "git_dirty_at_creation"):
                identity[key] = previous[key]
        store = Store(root, identity, resume=resume, clear_stop=clear_stop,
            max_bytes=cfg.max_bytes, free_reserve_bytes=cfg.min_free_bytes,
            external_bytes=file_bytes(FAMILY, exclude=root) + file_bytes(Path(data_root)),
            finalization_reserve_bytes=64 * 1024**2, max_wall_seconds=max_wall_seconds,
            max_active_seconds=limit-peers)
        store.family_lock, store.data_root = family_lock, Path(data_root)
        store.control.restore_active_seconds(own + time.monotonic() - started)
        budget = RuntimeBudget(store, other_seconds=peers, allocation_seconds=limit)
        budget.check()
        snapshot_metadata = dict(kind="source_snapshot", identity_hash=store.identity_hash)
        if store.begin("source-snapshot", snapshot_metadata) is not None:
            snapshot = {}
            for name, digest in identity["sources"].items():
                file = REPO / name
                if file_digest(file, budget.check) != digest:
                    raise ArtifactError("source changed while opening the scientific run")
                snapshot[name] = file.read_text()
            store.write_json("source-snapshot", "sources.json", snapshot)
            store.write_json("source-snapshot", "plan.json", {
                "document": (REPO/"mnist_experiment/plan18.md").read_text()})
            store.complete("source-snapshot")
        return store, budget
    except BaseException:
        if store is not None:
            store.close()
        family_lock.close()
        raise
