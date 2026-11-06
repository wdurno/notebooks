"""Phase 5 identity with the existing cumulative family lease and resource caps."""

from __future__ import annotations

import fcntl
import math
import time
from pathlib import Path

from mnist_experiment.fisher_memory.artifacts import ArtifactError
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested
from mnist_experiment.kalman_pi import runtime as original
from .config import PROTOCOL


REPO = original.REPO


def provenance(cfg, data_root, check=lambda: None):
    result = original.provenance(cfg, data_root, check)
    result["protocol"] = PROTOCOL
    sources = dict(result["sources"])
    for p in sorted(Path(__file__).parent.glob("*.py")):
        if p.name != "report.py":
            sources[str(p.relative_to(REPO))] = original.file_digest(p, check)
    # These otherwise report-only helpers determine Phase 5's calibration summaries.
    for relative in ("mnist_experiment/kalman_pi/analysis.py", "mnist_experiment/diagonal_lfu/analysis.py"):
        sources[relative] = original.file_digest(REPO / relative, check)
    result["sources"] = sources
    return result


def open_store(cfg, root, data_root, *, resume, clear_stop, allocation_hours, max_wall_seconds):
    started = time.monotonic()
    if (type(allocation_hours) not in (int, float) or not math.isfinite(allocation_hours)
            or not 0 < allocation_hours <= 20):
        raise ValueError("Plan 17 cumulative allocation must be in (0,20] hours")
    if max_wall_seconds is not None and (not math.isfinite(max_wall_seconds) or max_wall_seconds < 0):
        raise ValueError("invocation limit must be finite and nonnegative")
    if clear_stop and not resume:
        raise ValueError("clearing STOP requires --resume")
    root, data_root = Path(root), Path(data_root)
    if root.is_symlink() or original.FAMILY.is_symlink():
        raise ArtifactError("study/family roots must not be symlinks")
    original.FAMILY.mkdir(parents=True, exist_ok=True)
    lock_path = original.FAMILY / ".allocation.lock"
    if lock_path.is_symlink():
        raise ArtifactError("allocation lock must not be a symlink")
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise ArtifactError("another Plan 17 process owns the allocation") from error
    store = None
    try:
        own, peers = original._charged_seconds(root), original.peer_seconds(root)
        limit = allocation_hours * 3600

        def check():
            if (root / "STOP").exists() and not clear_stop:
                raise StopRequested("stop requested; resume with --clear-stop explicitly")
            elapsed = time.monotonic() - started
            if max_wall_seconds is not None and elapsed >= max_wall_seconds:
                raise ResourceLimit("invocation wall-clock allocation exhausted")
            if own + peers + elapsed >= limit:
                raise ResourceLimit("aggregate Plan 17 compute allocation exhausted")

        check()
        original.setup(cfg)
        identity = provenance(cfg, data_root, check)
        if (root / "identity.json").exists():
            prior = original.read_json(root / "identity.json")["identity"]
            for key in ("git_commit", "git_dirty_at_creation"):
                identity[key] = prior[key]
        outside = original.external_bytes(root, data_root, check)
        check()
        remaining = None if max_wall_seconds is None else max_wall_seconds - (time.monotonic() - started)
        store = original.Store(root, identity, resume=resume, clear_stop=clear_stop,
            max_bytes=original.MAX_BYTES, free_reserve_bytes=original.FREE_RESERVE_BYTES,
            external_bytes=outside, finalization_reserve_bytes=64 * 1024**2,
            max_wall_seconds=remaining, max_active_seconds=max(0., limit - peers))
        store.plan17_allocation_lock = lock
        store.plan17_data_root = data_root
        store.control.restore_active_seconds(own + time.monotonic() - started)
        budget = original.RuntimeBudget(store, other_seconds=peers, allocation_seconds=limit)
        budget.check()
        return store, budget
    except BaseException:
        if store is not None:
            store.close()
        lock.close()
        raise
