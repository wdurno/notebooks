"""Durable additional allowance and historical scientific dependency checks."""

import fcntl
import math
from pathlib import Path

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import utc_now
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, read_json, file_digest, file_bytes, dataset_manifest,
)
from mnist_experiment.combined_memory import runtime as parent
from mnist_experiment.combined_memory.reproduction import runtime as previous
from mnist_experiment.combined_memory.reproduction.runtime import atomic_json
from . import PROTOCOL
from .config import REPO, FAMILY, ROOT, HISTORY


def authorize():
    FAMILY.mkdir(parents=True, exist_ok=True)
    path = FAMILY / "phase9_authorization.json"
    with (FAMILY / ".phase9-authorization.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            prior = read_json(FAMILY / "phase8_authorization.json")["payload"]
            baseline = prior["baseline_seconds"] + previous.peer_seconds(ROOT)
            payload = dict(protocol=PROTOCOL, approved_at=utc_now(),
                approval="User: You are set to Sol. Please execute your new phase.",
                baseline_seconds=baseline, additional_seconds=72000.,
                cumulative_ceiling_seconds=baseline+72000.)
            atomic_json(path, dict(payload=payload, sha256=artifacts._hash_json(payload)))
    value = read_json(path)
    if value["sha256"] != artifacts._hash_json(value["payload"]) or value["payload"]["additional_seconds"] != 72000.:
        raise artifacts.ArtifactError("invalid Phase 9 authorization")
    return value


def sources():
    details = read_json(HISTORY / "provenance.json")
    result = dict(details["sources"])
    for name, digest in result.items():
        if file_digest(REPO/name) != digest:
            raise artifacts.ArtifactError(f"historical source changed: {name}")
    # Parent helpers used for model merging, scores and covariance recursions.
    for directory in (Path(__file__).parent, REPO/"mnist_experiment/combined_memory",
                      REPO/"mnist_experiment/kalman_pi"):
        for path in directory.glob("*.py"):
            if path.name not in ("report.py", "acceptance.py"):
                result[str(path.relative_to(REPO))] = file_digest(path)
    path = REPO/"mnist_experiment/combined_memory/broad_drift/observer.py"
    result[str(path.relative_to(REPO))] = file_digest(path)
    return dict(sorted(result.items()))


def peer_seconds(root):
    total = 0.
    for p in FAMILY.iterdir():
        if p.resolve() != Path(root).resolve() and (p/"identity.json").exists():
            if read_json(p/"identity.json")["identity"].get("protocol") == PROTOCOL:
                total += parent._charged_seconds(p)
    ledger = FAMILY / "phase9_preparation_compute.json"
    if ledger.exists():
        value=read_json(ledger)
        if value.get('protocol')!=PROTOCOL or not math.isfinite(value['charged_seconds']) or value['charged_seconds']<0:
            raise artifacts.ArtifactError('invalid preparation compute ledger')
        total += value["charged_seconds"]
    return total


def charge_preparation(seconds, description):
    if not math.isfinite(seconds) or seconds<0:
        raise ValueError('preparation seconds must be finite and nonnegative')
    path = FAMILY / "phase9_preparation_compute.json"
    with (FAMILY/".phase9-preparation.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = read_json(path) if path.exists() else dict(protocol=PROTOCOL, charged_seconds=0., entries=[])
        value["entries"].append(dict(seconds=float(seconds), description=description, recorded_at=utc_now()))
        value["charged_seconds"] += float(seconds)
        atomic_json(path, value)


def external_bytes(root, data):
    return file_bytes(FAMILY, exclude=root) + file_bytes(data)
