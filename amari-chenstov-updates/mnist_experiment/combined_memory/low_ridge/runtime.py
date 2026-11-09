"""Phase 7 identities, verified development imports, and incremental allowance."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import ResourceLimit, utc_now
from mnist_experiment.fisher_memory_regression.runtime import RuntimeBudget as LeaseBudget
from .. import runtime as original
from ..retention_match import runtime as history
from ..retention_match.config import Config as HistoryConfig
from . import PROTOCOL

REPO, FAMILY, DATA_ROOT = original.REPO, original.FAMILY, original.DATA_ROOT
DEFAULT_ROOT = FAMILY / "low_ridge_v1"
HISTORY_ROOT = FAMILY / "retention_match_v1"
Store, read_json = original.Store, original.read_json
checked_scalar, scalar_record = history.checked_scalar, history.scalar_record
load_readonly_checkpoint = history.load_readonly_checkpoint
AUTHORIZATION = FAMILY / "phase7_authorization.json"


def authorize():
    """Pin the user-approved incremental baseline once, before acceptance work."""
    FAMILY.mkdir(parents=True, exist_ok=True)
    with (FAMILY / ".allocation.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if AUTHORIZATION.exists():
            return authorization()
        baseline = original.peer_seconds(DEFAULT_ROOT) + original._charged_seconds(DEFAULT_ROOT)
        payload = dict(schema=1, protocol=PROTOCOL, authorized_at=utc_now(),
            approval="User: Please execute Phase 7. (2026-11-08)",
            baseline_seconds=baseline, incremental_allowance_seconds=72000.,
            cumulative_ceiling_seconds=baseline + 72000.)
        envelope = dict(payload=payload, sha256=artifacts._hash_json(payload))
        fd, temporary = tempfile.mkstemp(prefix=".phase7-authorization-", dir=FAMILY)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(artifacts._json_bytes(envelope))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, AUTHORIZATION)
            artifacts._sync_directory(FAMILY)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return authorization()


def authorization():
    value = read_json(AUTHORIZATION)
    payload = value.get("payload", {})
    if (value.get("sha256") != artifacts._hash_json(payload)
            or payload.get("protocol") != PROTOCOL
            or payload.get("incremental_allowance_seconds") != 72000.
            or payload.get("baseline_seconds", -1) < 0
            or payload.get("cumulative_ceiling_seconds") != payload["baseline_seconds"] + 72000.):
        raise artifacts.ArtifactError("invalid Phase 7 compute authorization")
    return value


def sources(check=lambda: None):
    result = history.sources(check)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name != "report.py":
            check()
            result[str(path.relative_to(REPO))] = original.file_digest(path, check)
    return result


def _historical_identity(root, cfg, check=lambda: None):
    identity = artifacts._identity(Path(root))
    expected = identity["identity"]
    if expected.get("protocol") != "plan18-retention-match-v1":
        raise artifacts.ArtifactError("Phase 7 requires a Phase 6 dependency")
    if expected["config"].get("seed") != HistoryConfig().seed:
        raise artifacts.ArtifactError("historical development seed differs from the pinned Phase 6 members")
    for name, digest in expected["sources"].items():
        check()
        if original.file_digest(REPO / name, check) != digest:
            raise artifacts.ArtifactError(f"historical scientific source changed: {name}")
    old = HistoryConfig.from_mapping(expected["config"]).broad_mapping()
    new = cfg.broad_mapping()
    for key in ("protocol", "schema_version", "seed", "main_replicas", "stationary_replicas"):
        old.pop(key, None)
        new.pop(key, None)
    if old != new:
        raise artifacts.ArtifactError("Phase 7 learning/data policy differs from historical dependency")
    return identity


def provenance(cfg, data_root, history_root, check=lambda: None):
    identity = _historical_identity(history_root, cfg, check)
    result = original.provenance(cfg, data_root, check)
    if result["dataset"] != identity["identity"]["dataset"]:
        raise artifacts.ArtifactError("historical cached dataset changed")
    result.update(protocol=PROTOCOL, sources=sources(check),
        historical_development=dict(root=str(Path(history_root).resolve().relative_to(REPO)),
                                    identity_hash=identity["identity_hash"]),
        authorization_sha256=authorization()["sha256"], development_members=list(range(4)))
    return result


def read_exported_records(root, check=lambda: None):
    root = Path(root)
    if not (root / "identity.json").exists():
        return [], dict(files={}, imports=[])
    identity = artifacts._identity(root)
    if identity["identity"].get("protocol") != PROTOCOL:
        raise artifacts.ArtifactError("incompatible low-ridge report identity")
    records, seen = [], set()
    provenance = dict(files={}, imports=[], identity_hash=identity["identity_hash"])
    for unit in artifacts.iter_units(root, completed_only=True, verify=False, check=check):
        if unit["metadata"].get("kind") == "trajectory":
            record, seals = scalar_record(root, unit["unit_id"], check)
            records.append(record)
            provenance["files"][unit["unit_id"]] = seals
        elif unit["metadata"].get("kind") == "historical_development_import":
            imported, digest = checked_scalar(root, unit["unit_id"], "references.json", check)
            for reference in imported["units"]:
                origin = REPO / reference["root"]
                key = (str(origin.resolve()), reference["unit_id"])
                if key in seen:
                    raise artifacts.ArtifactError("duplicate transitive historical trajectory")
                seen.add(key)
                if artifacts._identity(origin)["identity_hash"] != reference["identity_hash"]:
                    raise artifacts.ArtifactError("historical development identity changed")
                record, seals = scalar_record(origin, reference["unit_id"], check)
                if seals != reference["seals"]:
                    raise artifacts.ArtifactError("historical development checksum changed")
                meta = record["metadata"]
                if meta.get("cohort") != "development" or meta.get("replica") not in range(4):
                    raise artifacts.ArtifactError("historical confirmation cannot enter calibration")
                alias = "imported-" + artifacts._hash_json(dict(root=reference["root"],
                    unit=reference["unit_id"]))[:20]
                record["unit_id"] = alias
                record["metadata"] = dict(meta, imported=True, origin_root=reference["root"],
                                          origin_unit=reference["unit_id"])
                records.append(record)
            provenance["imports"].append(dict(unit=unit["unit_id"], sha256=digest, dependency=imported))
    return records, provenance


class RuntimeBudget(original.RuntimeBudget):
    def __init__(self, store, *, other_seconds=0., allocation_seconds):
        if allocation_seconds != authorization()["payload"]["cumulative_ceiling_seconds"]:
            raise ValueError("budget must equal the sealed Phase 7 cumulative authorization")
        LeaseBudget.__init__(self, store, other_seconds=other_seconds,
                             allocation_seconds=allocation_seconds)

    def check(self):
        self.store.check()
        if time.monotonic() - getattr(self, "last_peer_refresh", 0.) >= 60:
            self.other_seconds = original.peer_seconds(self.store.root)
            self.store.control.max_active_seconds = self.allocation_seconds - self.other_seconds
            self.last_peer_refresh = time.monotonic()
        if self.used >= self.allocation_seconds:
            raise ResourceLimit("Phase 7 approved incremental compute allowance exhausted")
        self.pulse()


def open_store(cfg, root, history_root=HISTORY_ROOT, data_root=DATA_ROOT, *, resume=False,
               clear_stop=False, max_wall_seconds=None):
    started = time.monotonic()
    approved = authorization()
    lock = (FAMILY / ".allocation.lock").open("a+b")
    store = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path(root)
        peers, own = original.peer_seconds(root), original._charged_seconds(root)
        limit = approved["payload"]["cumulative_ceiling_seconds"]
        if peers + own >= limit:
            raise ResourceLimit("Phase 7 cumulative allowance already exhausted")
        original.setup(cfg)
        identity = provenance(cfg, data_root, history_root)
        if (root / "identity.json").exists():
            old = read_json(root / "identity.json")["identity"]
            for key in ("git_commit", "git_dirty_at_creation"):
                identity[key] = old[key]
        store = Store(root, identity, resume=resume, clear_stop=clear_stop,
            max_bytes=cfg.max_bytes, free_reserve_bytes=cfg.min_free_bytes,
            external_bytes=original.file_bytes(FAMILY, exclude=root) + original.file_bytes(Path(data_root)),
            finalization_reserve_bytes=64 * 1024**2, max_wall_seconds=max_wall_seconds,
            max_active_seconds=limit-peers)
        store.family_lock, store.data_root = lock, Path(data_root)
        store.control.restore_active_seconds(own + time.monotonic() - started)
        budget = RuntimeBudget(store, other_seconds=peers, allocation_seconds=limit)
        budget.authorization = approved["payload"]
        budget.check()
        if store.begin("source-snapshot", dict(kind="source_snapshot", identity_hash=store.identity_hash)) is not None:
            snapshot = {}
            for name, digest in identity["sources"].items():
                if original.file_digest(REPO / name, budget.check) != digest:
                    raise artifacts.ArtifactError("scientific source changed at startup")
                snapshot[name] = (REPO / name).read_text()
            store.write_json("source-snapshot", "sources.json", snapshot)
            store.write_json("source-snapshot", "plan.json", dict(document=(REPO / "mnist_experiment/plan18.md").read_text()))
            store.complete("source-snapshot")
        if store.begin("decision-authorization", dict(kind="decision", name="authorization")) is not None:
            store.write_json("decision-authorization", "decision.json", approved)
            store.complete("decision-authorization")
        if read_json(store.unit_path("decision-authorization") / "decision.json") != approved:
            raise artifacts.ArtifactError("sealed compute authorization changed")
        return store, budget
    except BaseException:
        if store is not None:
            store.close()
        lock.close()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(authorize()), flush=True)
