"""Independent Phase 6 allocation with the shared immutable-store contract."""

from __future__ import annotations

import fcntl
import math
import os
from pathlib import Path
import time

from mnist_experiment.fisher_memory.artifacts import ArtifactError
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, utc_now
from mnist_experiment.kalman_pi import runtime as historical


REPO = historical.REPO
FAMILY = historical.FAMILY
ALLOCATION_FAMILY = "plan17-phase6"
MAX_SECONDS = 20 * 3600.
ACCEPTANCE_LEDGER = FAMILY / "phase6_acceptance_compute.json"
read_json = historical.read_json
file_digest = historical.file_digest
Store = historical.Store


def scientific_sources(check=lambda: None):
    paths = [p for p in Path(__file__).parent.glob("*.py") if p.name not in ("report.py", "acceptance.py")]
    paths += [REPO / name for name in (
        "mnist_experiment/kalman_pi/policies.py",
        "mnist_experiment/kalman_pi/trajectory.py",
        "mnist_experiment/kalman_pi/observations.py",
        "mnist_experiment/kalman_pi/runtime.py",
        "mnist_experiment/diagonal_lfu/data.py",
        "mnist_experiment/diagonal_lfu/config.py",
        "mnist_experiment/diagonal_lfu/trajectory.py",
        "mnist_experiment/diagonal_lfu/curvature.py",
        "mnist_experiment/diagonal_lfu/learner.py",
        "mnist_experiment/rotated_mnist/plan12/gauge.py",
        "src/mnist_model.py",
        "mnist_experiment/fisher_memory/artifacts.py",
        "mnist_experiment/fisher_memory/control.py",
        "mnist_experiment/fisher_memory/data.py",
        "mnist_experiment/fisher_memory_regression/runtime.py",
        "mnist_experiment/fisher_memory_regression/config.py",
        "mnist_experiment/fisher_memory_registration/data.py",
        "mnist_experiment/fisher_memory_registration/__init__.py",
        "src/parameters.py", "src/seeding.py", "src/mnist_data.py")]
    return {str(p.relative_to(REPO)): file_digest(p, check) for p in sorted(set(paths))}


def provenance(cfg, data_root, check=lambda: None):
    result = historical.provenance(cfg, data_root, check)
    result.update(protocol=cfg.protocol, allocation_family=ALLOCATION_FAMILY,
                  phase=6, sources=scientific_sources(check),
                  likelihood="six-coordinate-Gaussian-covariance-4I",
                  parameter_count=448)
    return result


def phase_seconds(root, check=lambda: None):
    total = 0.
    if FAMILY.exists():
        for peer in sorted(FAMILY.iterdir()):
            check()
            if peer.resolve() == Path(root).resolve() or not (peer / "identity.json").is_file():
                continue
            if peer.is_symlink():
                raise ArtifactError("phase allocation peer cannot be a symlink")
            identity = read_json(peer / "identity.json")["identity"]
            if identity.get("allocation_family") == ALLOCATION_FAMILY:
                total += historical._charged_seconds(peer)
    if ACCEPTANCE_LEDGER.exists():
        for entry in read_json(ACCEPTANCE_LEDGER).get("entries", []):
            check()
            total += historical._seconds(entry["seconds"], "Phase 6 acceptance")
    return total


def external_bytes(root, check=lambda: None):
    return historical.file_bytes(REPO / "cache", exclude=Path(root), check=check)


class RuntimeBudget(historical.RuntimeBudget):
    def pulse(self, *, force=False):
        if self._pulsing or (not force and time.monotonic() - self.last_pulse < 10):
            return
        self._pulsing = True
        try:
            store = self.store
            if time.monotonic() - getattr(self, "last_storage_scan", 0.) >= 60:
                store.external_bytes = external_bytes(store.root, store.control.check)
                self.last_storage_scan = time.monotonic()
            active = store.control.active_seconds
            until = min(active + 60., self.allocation_seconds - self.other_seconds)
            if store.control.max_wall_seconds is not None:
                until = min(until, store.control.base_seconds + store.control.max_wall_seconds)
            store.write_json(None, "timing_lease.json", {
                "schema_version": 1, "running": True, "pid": os.getpid(),
                "charged_until": max(active, until), "heartbeat": utc_now(),
                "reservation_seconds": max(0., until - active)}, finalizing=True)
            store.heartbeat(force=True)
            self.last_pulse = time.monotonic()
        finally:
            self._pulsing = False


def open_store(cfg, root, data_root, *, resume=False, clear_stop=False,
               allocation_hours=20., max_wall_seconds=None):
    started = time.monotonic()
    if (type(allocation_hours) not in (int, float) or not math.isfinite(allocation_hours)
            or not 0 < allocation_hours <= 20):
        raise ValueError("Phase 6 allocation must be in (0,20] hours")
    if max_wall_seconds is not None and (not math.isfinite(max_wall_seconds) or max_wall_seconds < 0):
        raise ValueError("invocation limit must be finite and nonnegative")
    if clear_stop and not resume:
        raise ValueError("clearing STOP requires --resume")
    root, data_root = Path(root), Path(data_root)
    if root.is_symlink() or FAMILY.is_symlink():
        raise ArtifactError("study/family roots must not be symlinks")
    FAMILY.mkdir(parents=True, exist_ok=True)
    lock_path = FAMILY / ".allocation.lock"
    if lock_path.is_symlink():
        raise ArtifactError("allocation lock must not be a symlink")
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise ArtifactError("another Plan 17 worker owns the allocation lock") from error
    store = None
    try:
        own = historical._charged_seconds(root)
        peers = phase_seconds(root)
        limit = allocation_hours * 3600

        def check():
            elapsed = time.monotonic() - started
            if (root / "STOP").exists() and not clear_stop:
                raise StopRequested("stop requested; resume with --clear-stop explicitly")
            if max_wall_seconds is not None and elapsed >= max_wall_seconds:
                raise ResourceLimit("invocation allocation exhausted")
            if own + peers + elapsed >= limit:
                raise ResourceLimit("Phase 6 cumulative allocation exhausted")

        check()
        historical.setup(cfg)
        identity = provenance(cfg, data_root, check)
        if (root / "identity.json").exists():
            prior = read_json(root / "identity.json")["identity"]
            for key in ("git_commit", "git_dirty_at_creation"):
                identity[key] = prior[key]
        outside = external_bytes(root, check)
        remaining = None if max_wall_seconds is None else max_wall_seconds - (time.monotonic() - started)
        store = Store(root, identity, resume=resume, clear_stop=clear_stop,
                      max_bytes=min(cfg.max_bytes, 500_000_000_000),
                      free_reserve_bytes=max(cfg.min_free_bytes, 20_000_000_000),
                      external_bytes=outside, finalization_reserve_bytes=64 * 1024**2,
                      max_wall_seconds=remaining, max_active_seconds=max(0., limit - peers))
        store.plan17_allocation_lock = lock
        store.control.restore_active_seconds(own + time.monotonic() - started)
        budget = RuntimeBudget(store, other_seconds=peers, allocation_seconds=limit)
        budget.historical_seconds = historical.peer_seconds(root, check)
        budget.check()
        return store, budget
    except BaseException:
        if store is not None:
            store.close()
        lock.close()
        raise
