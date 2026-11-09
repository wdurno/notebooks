"""Continue a passing Phase 6 allocation under the explicitly approved ceiling.

The immutable learner and scheduler remain the same. An operational decision
archives this wrapper and the user's resource authorization separately from
the frozen scientific identity. Only a passing four-member seal permits entry.
"""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from mnist_experiment.diagonal_lfu.data import load_data
from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, utc_now
from mnist_experiment.fisher_memory_regression.runtime import RuntimeBudget as LeaseBudget
from ... import runtime as original
from ...config import ARMS, Recipe
from .. import calibration
from ..config import Config
from ..run import Scheduler
from ..runtime import (DEFAULT_ROOT, HISTORY_ROOT, DATA_ROOT, REPO, checked_scalar,
                       read_exported_records, provenance)


APPROVED_CEILING_HOURS = 21.5


def require_calibrated(root, cfg):
    identity = artifacts._identity(Path(root))
    selected, digest = checked_scalar(root, "decision-selection", "decision.json")
    if (not selected.get("frozen") or selected.get("calibration_status") != "passed"
            or selected.get("source_identity") != identity["identity_hash"]
            or selected.get("members") != [0, 1, 2, 3]):
        raise artifacts.ArtifactError("a passing frozen four-member calibration is required for the extension")
    records, _ = read_exported_records(root)
    rows = {arm: calibration.aggregate(records, Recipe(**selected["recipes"][arm])) for arm in ARMS}
    actual = calibration.matching(rows, cfg.retention_tolerance)
    if not actual["passed"] or actual != selected["matching"] or rows != selected["calibration"]:
        raise artifacts.ArtifactError("the calibration evidence does not support the approved conditional extension")
    if selected["main_replicas"] != cfg.main_replicas or selected["stationary_replicas"] != cfg.stationary_replicas:
        raise artifacts.ArtifactError("the approved confirmation allocation must remain unchanged")
    return selected, digest


class ExtendedBudget(original.RuntimeBudget):
    def __init__(self, store, *, other_seconds=0., allocation_seconds=21.5*3600):
        if allocation_seconds != APPROVED_CEILING_HOURS*3600:
            raise ValueError("only the explicitly approved 21.5-hour ceiling is supported")
        # Reuse the common crash-recovery lease without Plan 17's fixed 20-hour cap.
        LeaseBudget.__init__(self, store, other_seconds=other_seconds,
                             allocation_seconds=allocation_seconds)

    def check(self):
        self.store.check()
        if self.used >= self.allocation_seconds:
            raise ResourceLimit("the approved 21.5-hour cumulative Plan 18 ceiling is exhausted")
        self.pulse()


def open_extended_store(cfg, root, history_root, *, approved_hours, approval_note, clear_stop=False):
    started = time.monotonic()
    if approved_hours != APPROVED_CEILING_HOURS or not approval_note.strip():
        raise ValueError("the explicit 21.5-hour user approval and its note are required")
    selected, selection_digest = require_calibrated(root, cfg)
    original.FAMILY.mkdir(parents=True, exist_ok=True)
    lock = (original.FAMILY / ".allocation.lock").open("a+b")
    store = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path(root)
        peers, own = original.peer_seconds(root), original._charged_seconds(root)
        limit = approved_hours * 3600
        if peers + own >= limit:
            raise ResourceLimit("the approved cumulative allocation is already exhausted")
        original.setup(cfg)
        identity = provenance(cfg, DATA_ROOT, history_root)
        previous = artifacts._identity(root)["identity"]
        for key in ("git_commit", "git_dirty_at_creation"):
            identity[key] = previous[key]
        store = original.Store(root, identity, resume=True, clear_stop=clear_stop,
            max_bytes=cfg.max_bytes, free_reserve_bytes=cfg.min_free_bytes,
            external_bytes=original.file_bytes(original.FAMILY, exclude=root)+original.file_bytes(DATA_ROOT),
            finalization_reserve_bytes=64*1024**2, max_active_seconds=limit-peers)
        store.family_lock, store.data_root = lock, DATA_ROOT
        store.control.restore_active_seconds(own+time.monotonic()-started)
        budget = ExtendedBudget(store, other_seconds=peers, allocation_seconds=limit)
        budget.check()
        name = "resource-extension-21p5"
        uid = "decision-"+name
        source = Path(__file__).read_bytes()
        source_hash = hashlib.sha256(source).hexdigest()
        if store.begin(uid, dict(kind="decision", name=name)) is not None:
            store.write_json(uid, "decision.json", dict(
                approved_at=utc_now(), approval_note=approval_note,
                previous_ceiling_hours=20., approved_ceiling_hours=approved_hours,
                condition="passing four-member development calibration; unchanged fresh allocation",
                selection_sha256=selection_digest, source_identity=store.identity_hash,
                config_hash=cfg.scientific_hash, main_replicas=selected["main_replicas"],
                stationary_replicas=selected["stationary_replicas"],
                continuation_source_sha256=source_hash, continuation_source=source.decode()))
            store.complete(uid)
        else:
            decision, _ = checked_scalar(root, uid, "decision.json", budget.check)
            if (decision["approved_ceiling_hours"] != approved_hours
                    or decision["selection_sha256"] != selection_digest
                    or decision["continuation_source_sha256"] != source_hash):
                raise artifacts.ArtifactError("resource continuation differs from its recorded approval")
        return store, budget
    except BaseException:
        if store is not None:
            store.close()
        lock.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--approved-ceiling-hours", type=float, required=True)
    parser.add_argument("--approval-note", required=True)
    parser.add_argument("--clear-stop", action="store_true")
    args = parser.parse_args(argv)
    cfg = Config.read(args.config)
    if (args.root/"study_completed.json").exists():
        print(json.dumps(dict(status="already_completed")),flush=True)
        return 0
    store = budget = scheduler = None
    try:
        store, budget = open_extended_store(cfg, args.root, args.history_root,
            approved_hours=args.approved_ceiling_hours, approval_note=args.approval_note,
            clear_stop=args.clear_stop)
        with store.control.signal_handlers():
            scheduler = Scheduler(cfg,store,budget,load_data(DATA_ROOT),history_root=args.history_root)
            scheduler.validate_selection()
            scheduler.run()
    except (StopRequested, ResourceLimit) as error:
        if scheduler:
            scheduler.publish("paused" if isinstance(error,StopRequested) else "resource_limited",figures=False)
        print(json.dumps(dict(status="paused",reason=str(error))),flush=True)
    finally:
        if budget: budget.close()
        if store: store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
