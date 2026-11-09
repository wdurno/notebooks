"""Immutable Phase 6 identity, read-only imports, and the Plan 18 time ledger."""

import fcntl
import hashlib
import json
from pathlib import Path
import time

import torch

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import ResourceLimit
from .. import runtime as original
from ..broad_drift import runtime as broad
from . import PROTOCOL

REPO, FAMILY, DATA_ROOT = original.REPO, original.FAMILY, original.DATA_ROOT
DEFAULT_ROOT = FAMILY / "retention_match_v1"
HISTORY_ROOT = FAMILY / "broad_drift_v1"
Store, RuntimeBudget = original.Store, original.RuntimeBudget
read_json = original.read_json


def sources(check=lambda: None):
    result = broad.sources(check)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name != "report.py":
            check()
            result[str(path.relative_to(REPO))] = original.file_digest(path, check)
    return result


def checked_scalar(root, uid, name, check=lambda: None):
    """Verify an exported scalar file without opening any model checkpoint."""
    root = Path(root)
    identity = artifacts._identity(root)
    unit = artifacts._unit_record(root, uid, identity["identity_hash"], {}, check, verify=False)
    marker = unit["completion"]
    if marker is None or name not in marker["files"]:
        raise artifacts.ArtifactError(f"missing completed scalar export: {uid}/{name}")
    path = unit["path"] / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise artifacts.ArtifactError(f"invalid or oversized scalar export: {path}")
    check()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = marker["files"][name]
    if expected != dict(bytes=len(data), sha256=digest):
        raise artifacts.ArtifactError(f"completed scalar checksum mismatch: {path}")
    value = json.loads(data)
    artifacts._json_bytes(value)  # Reject nonfinite JSON, including exponent overflow.
    return value, digest


def scalar_record(root, uid, check=lambda: None):
    identity = artifacts._identity(Path(root))
    unit = artifacts._unit_record(Path(root), uid, identity["identity_hash"], {}, check, verify=False)
    metadata, manifest_hash = checked_scalar(root, uid, "manifest.json", check)
    if metadata != unit["manifest"]:
        raise artifacts.ArtifactError("trajectory manifest changed during scalar validation")
    summary, sh = checked_scalar(root, uid, "summary.json", check)
    metrics, mh = checked_scalar(root, uid, "metrics.json", check)
    diagnostics, dh = checked_scalar(root, uid, "diagnostics.json", check)
    return dict(unit_id=uid, metadata=unit["metadata"], status="completed",
                summary=summary, metrics=metrics, diagnostics=diagnostics), dict(
                    manifest=manifest_hash, summary=sh, metrics=mh, diagnostics=dh,
                    completion=original.file_digest(unit["path"] / "COMPLETED", check))


def load_readonly_checkpoint(root, uid, check=lambda: None):
    """Load a trusted sealed dependency without restoring its historical clock."""
    root = Path(root)
    identity = artifacts._identity(root)
    unit = artifacts._unit_record(root, uid, identity["identity_hash"], {}, check)
    if unit["completion"] is None:
        raise artifacts.ArtifactError("historical checkpoint is not complete")
    latest, _ = checked_scalar(root, uid, "latest.json", check)
    descriptor = latest["current"]
    name = artifacts._safe_name(descriptor["file"])
    path = unit["path"] / name
    if original.file_digest(path, check) != descriptor["sha256"]:
        raise artifacts.ArtifactError("historical checkpoint checksum mismatch")
    envelope = torch.load(path, map_location="cpu", weights_only=False)
    if (envelope["identity_hash"] != identity["identity_hash"] or envelope["unit_id"] != uid
            or envelope["manifest_hash"] != artifacts._hash_json(unit["manifest"])):
        raise artifacts.ArtifactError("historical checkpoint identity mismatch")
    return envelope["payload"]


def read_exported_records(root, check=lambda: None):
    """Read only completed scalar exports, including explicitly sealed imports."""
    root = Path(root)
    if not (root / "identity.json").exists():
        return [], dict(files={}, imports=[])
    identity = artifacts._identity(root)
    if identity["identity"].get("protocol") != PROTOCOL:
        raise artifacts.ArtifactError("incompatible retention-match report identity")
    records, provenance = [], dict(files={}, imports=[], identity_hash=identity["identity_hash"])
    for unit in artifacts.iter_units(root, completed_only=True, verify=False, check=check):
        if unit["metadata"].get("kind") == "trajectory":
            record, seals = scalar_record(root, unit["unit_id"], check)
            records.append(record)
            provenance["files"][unit["unit_id"]] = seals
        elif unit["metadata"].get("kind") == "historical_development_import":
            dependency, seal = checked_scalar(root, unit["unit_id"], "references.json", check)
            history = REPO / dependency["root"]
            historical = artifacts._identity(history)
            if historical["identity_hash"] != dependency["identity_hash"]:
                raise artifacts.ArtifactError("historical dependency identity changed")
            for reference in dependency["units"]:
                record, seals = scalar_record(history, reference["unit_id"], check)
                if seals != reference["seals"]:
                    raise artifacts.ArtifactError("historical scalar reference changed")
                meta = record["metadata"]
                if meta["cohort"] != "development" or meta["replica"] not in (0, 1):
                    raise artifacts.ArtifactError("historical confirmation is not development evidence")
                record["unit_id"] = "imported-" + record["unit_id"]
                record["metadata"] = dict(meta, imported=True,
                    origin_root=dependency["root"], origin_unit=reference["unit_id"])
                records.append(record)
            provenance["imports"].append(dict(unit=unit["unit_id"], sha256=seal,
                                            dependency=dependency))
    return records, provenance


def provenance(cfg, data_root, history_root, check=lambda: None):
    result = broad.provenance(cfg, data_root, check)
    history = artifacts._identity(Path(history_root))
    expected = history["identity"]
    if expected.get("protocol") != "plan18-broad-drift-v1":
        raise artifacts.ArtifactError("Phase 6 requires a Phase 5 historical dependency")
    for name, digest in expected["sources"].items():
        check()
        if original.file_digest(REPO / name, check) != digest:
            raise artifacts.ArtifactError(f"historical scientific source changed: {name}")
    old_cfg = dict(expected["config"])
    new_cfg = cfg.broad_mapping()
    for key in ("protocol", "seed", "main_replicas", "stationary_replicas",
                "retention_tolerance", "fd_dtype", "max_iter", "max_eval"):
        old_cfg.pop(key, None)
        new_cfg.pop(key, None)
    if old_cfg != new_cfg:
        raise artifacts.ArtifactError("historical and current learning/data configurations differ")
    result.update(protocol=PROTOCOL, sources=sources(check),
        historical_development=dict(root=str(Path(history_root).resolve().relative_to(REPO)),
                                    identity_hash=history["identity_hash"]),
        development_members=[dict(replica=i, source="historical" if i < 2 else "new")
                             for i in range(4)])
    return result


def open_store(cfg, root, history_root=HISTORY_ROOT, data_root=DATA_ROOT, *, resume=False,
               clear_stop=False, allocation_hours=20., max_wall_seconds=None):
    started = time.monotonic()
    if not 0 < allocation_hours <= 20:
        raise ValueError("Plan 18 cumulative allocation must be in (0,20] hours")
    FAMILY.mkdir(parents=True, exist_ok=True)
    lock = (FAMILY / ".allocation.lock").open("a+b")
    store = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path(root)
        peers, own = original.peer_seconds(root), original._charged_seconds(root)
        limit = allocation_hours * 3600
        if peers + own >= limit:
            raise ResourceLimit("Plan 18 cumulative allocation already exhausted")
        original.setup(cfg)
        identity = provenance(cfg, data_root, history_root)
        if (root / "identity.json").exists():
            previous = read_json(root / "identity.json")["identity"]
            for key in ("git_commit", "git_dirty_at_creation"):
                identity[key] = previous[key]
        store = Store(root, identity, resume=resume, clear_stop=clear_stop,
            max_bytes=cfg.max_bytes, free_reserve_bytes=cfg.min_free_bytes,
            external_bytes=original.file_bytes(FAMILY, exclude=root) + original.file_bytes(Path(data_root)),
            finalization_reserve_bytes=64 * 1024**2, max_wall_seconds=max_wall_seconds,
            max_active_seconds=limit-peers)
        store.family_lock, store.data_root = lock, Path(data_root)
        store.control.restore_active_seconds(own + time.monotonic() - started)
        budget = RuntimeBudget(store, other_seconds=peers, allocation_seconds=limit)
        budget.check()
        metadata = dict(kind="source_snapshot", identity_hash=store.identity_hash)
        if store.begin("source-snapshot", metadata) is not None:
            snapshot = {}
            for name, digest in identity["sources"].items():
                if original.file_digest(REPO / name, budget.check) != digest:
                    raise artifacts.ArtifactError("source changed during run startup")
                snapshot[name] = (REPO / name).read_text()
            store.write_json("source-snapshot", "sources.json", snapshot)
            store.write_json("source-snapshot", "plan.json", {
                "document": (REPO / "mnist_experiment/plan18.md").read_text()})
            store.complete("source-snapshot")
        return store, budget
    except BaseException:
        if store is not None:
            store.close()
        lock.close()
        raise
