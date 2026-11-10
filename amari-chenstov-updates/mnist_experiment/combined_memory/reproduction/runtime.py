"""One durable incremental allowance and frozen historical dependencies."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import utc_now
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, read_json, file_digest, file_bytes, dataset_manifest,
)
from mnist_experiment.combined_memory import runtime as parent
from . import PROTOCOL
from .config import REPO, FAMILY, HISTORY, ROOT

AUTHORIZATION = FAMILY / "phase8_authorization.json"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(artifacts._json_bytes(value))
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        artifacts._sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def authorize():
    FAMILY.mkdir(parents=True, exist_ok=True)
    with (FAMILY / ".phase8-authorization.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not AUTHORIZATION.exists():
            baseline = parent.peer_seconds(ROOT) + parent._charged_seconds(ROOT)
            value = dict(protocol=PROTOCOL, approved_at=utc_now(),
                approval="User: Excellent. You are set to Sol. Please execute Phase 8.",
                baseline_seconds=baseline, additional_seconds=72000.,
                cumulative_ceiling_seconds=baseline + 72000.)
            atomic_json(AUTHORIZATION, dict(payload=value, sha256=artifacts._hash_json(value)))
    value = read_json(AUTHORIZATION)
    if value["sha256"] != artifacts._hash_json(value["payload"]) or value["payload"]["additional_seconds"] != 72000.:
        raise artifacts.ArtifactError("invalid Phase 8 authorization")
    return value


def historical_sources(history=HISTORY, check=lambda: None):
    details = read_json(Path(history) / "provenance.json")
    for name, digest in details["sources"].items():
        check()
        if file_digest(REPO / name, check) != digest:
            raise artifacts.ArtifactError(f"historical scientific source changed: {name}")
    return details


def sources(history=HISTORY):
    result = dict(historical_sources(history)["sources"])
    for path in Path(__file__).parent.glob("*.py"):
        if path.name != "report.py":
            result[str(path.relative_to(REPO))] = file_digest(path)
    return dict(sorted(result.items()))


def peer_seconds(root):
    charged = sum(parent._charged_seconds(p) for p in FAMILY.iterdir()
               if p.resolve() != Path(root).resolve() and (p / "identity.json").exists()
               and read_json(p / "identity.json")["identity"].get("protocol") == PROTOCOL)
    ledger = FAMILY / "phase8_preparation_compute.json"
    if ledger.exists():
        value = read_json(ledger)
        if value["protocol"] != PROTOCOL or value["charged_seconds"] < 0:
            raise ValueError("invalid preparation charge")
        charged += value["charged_seconds"]
    return charged


def check_historical_unit(unit, history=HISTORY, check=lambda: None):
    history = Path(history)
    return artifacts._unit_record(history, unit, artifacts._identity(history)["identity_hash"],
                                  {}, check, verify=True)


def external_bytes(root, data_root):
    return file_bytes(FAMILY, exclude=root) + file_bytes(HISTORY) + file_bytes(data_root)
