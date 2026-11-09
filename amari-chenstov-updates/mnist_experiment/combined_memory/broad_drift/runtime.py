"""Separate scientific identity with the original cumulative Plan 18 budget."""

from __future__ import annotations

import fcntl
from pathlib import Path
import time

from mnist_experiment.fisher_memory.artifacts import ArtifactError
from mnist_experiment.fisher_memory.control import ResourceLimit
from .. import runtime as original

REPO, FAMILY, DATA_ROOT = original.REPO, original.FAMILY, original.DATA_ROOT
Store, RuntimeBudget = original.Store, original.RuntimeBudget
read_json, setup = original.read_json, original.setup


def sources(check=lambda: None):
    result = original.sources(check)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name == "report.py":
            continue
        check()
        result[str(path.relative_to(REPO))] = original.file_digest(path, check)
    return result


def provenance(cfg, data_root, check=lambda: None):
    result = original.provenance(cfg, data_root, check)
    result.update(protocol=cfg.protocol, sources=sources(check),
        acquisition_clock="unique_outbound_grids; stationary has no acquisition",
        legacy_scalar_aliases={"novel_count": "outbound acquisition count",
            "H100": "old RMS deterioration at configured outbound budget",
            "novel_mse_100": "rotated MSE at configured outbound budget",
            "old_mse_100": "old MSE at configured outbound budget",
            "learning_seconds_100": "learning seconds at configured outbound budget"})
    return result


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
        peers, own = original.peer_seconds(root), original._charged_seconds(root)
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
            external_bytes=original.file_bytes(FAMILY, exclude=root) + original.file_bytes(Path(data_root)),
            finalization_reserve_bytes=64 * 1024**2, max_wall_seconds=max_wall_seconds,
            max_active_seconds=limit-peers)
        store.family_lock, store.data_root = family_lock, Path(data_root)
        store.control.restore_active_seconds(own + time.monotonic() - started)
        budget = RuntimeBudget(store, other_seconds=peers, allocation_seconds=limit)
        budget.check()
        metadata = dict(kind="source_snapshot", identity_hash=store.identity_hash)
        if store.begin("source-snapshot", metadata) is not None:
            snapshot = {}
            for name, digest in identity["sources"].items():
                file = REPO / name
                if original.file_digest(file, budget.check) != digest:
                    raise ArtifactError("source changed while opening the scientific run")
                snapshot[name] = file.read_text()
            store.write_json("source-snapshot", "sources.json", snapshot)
            store.write_json("source-snapshot", "plan.json", {
                "document": (REPO / "mnist_experiment/plan18.md").read_text()})
            store.complete("source-snapshot")
        return store, budget
    except BaseException:
        if store is not None:
            store.close()
        family_lock.close()
        raise
