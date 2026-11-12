"""Linked compute allowance, immutable external assets and frozen sources."""

import fcntl
import math
from pathlib import Path

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import utc_now
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, read_json, file_digest, file_bytes,
)
from mnist_experiment.combined_memory import runtime as family_runtime
from mnist_experiment.combined_memory.reproduction.runtime import atomic_json
from mnist_experiment.combined_memory.reproduction.audit import load_checkpoint
from mnist_experiment.combined_memory.replicated_combination import runtime as previous
from . import PROTOCOL
from .config import REPO, FAMILY, PREVIOUS


def authorize():
    original = previous.authorize()
    path = FAMILY / "phase9e_authorization.json"
    with (FAMILY / ".phase9e-authorization.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not path.exists():
            payload = dict(protocol=PROTOCOL, approved_at=utc_now(),
                approval="User: Okay, you are set to Sol. Please execute this phase.",
                original_authorization_sha256=original["sha256"],
                additional_seconds=0., shared_phase9_seconds=72000.,
                original_baseline_seconds=original["payload"]["baseline_seconds"])
            atomic_json(path, dict(payload=payload, sha256=artifacts._hash_json(payload)))
    value = read_json(path)
    if (value["sha256"] != artifacts._hash_json(value["payload"])
            or value["payload"]["original_authorization_sha256"] != original["sha256"]
            or value["payload"]["shared_phase9_seconds"] != 72000.
            or value["payload"]["additional_seconds"] != 0.):
        raise artifacts.ArtifactError("invalid shared Phase 9E allowance")
    return value


def sources():
    original = read_json(PREVIOUS / "provenance.json")["sources"]
    for name, digest in original.items():
        if file_digest(REPO / name) != digest:
            raise artifacts.ImmutableError("frozen Phase 9 source changed: " + name)
    result = dict(original)
    for path in Path(__file__).parent.glob("*.py"):
        if path.name not in ("report.py", "acceptance.py"):
            result[str(path.relative_to(REPO))] = file_digest(path)
    return dict(sorted(result.items()))


def peer_seconds(root):
    total = previous.peer_seconds(root)
    for path in FAMILY.iterdir():
        if path.resolve() != Path(root).resolve() and (path / "identity.json").is_file():
            if read_json(path / "identity.json")["identity"].get("protocol") == PROTOCOL:
                total += family_runtime._charged_seconds(path)
    ledger = FAMILY / "phase9e_preparation_compute.json"
    if ledger.exists():
        value = read_json(ledger)
        if value["protocol"] != PROTOCOL or not math.isfinite(value["charged_seconds"]) or value["charged_seconds"] < 0:
            raise artifacts.ArtifactError("invalid extension preparation ledger")
        total += value["charged_seconds"]
    return total


def charge_preparation(seconds, description):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("nonnegative compute charge required")
    path = FAMILY / "phase9e_preparation_compute.json"
    with (FAMILY / ".phase9e-preparation.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        value = read_json(path) if path.exists() else dict(protocol=PROTOCOL, charged_seconds=0., entries=[])
        value["entries"].append(dict(seconds=seconds, description=description, recorded_at=utc_now()))
        value["charged_seconds"] += seconds
        atomic_json(path, value)


def check_external(unit, check=lambda: None):
    return artifacts._unit_record(PREVIOUS, unit, artifacts._identity(PREVIOUS)["identity_hash"],
                                  {}, check, True)


class ReuseStore:
    """Read sealed development assets in place; write only into the new store."""

    def __init__(self, base, *, reuse=True):
        self.base, self.reuse = base, reuse
        self.verified = set()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def external(self, unit):
        if not self.reuse or not unit or not unit.startswith(("development-shared--", "acceptance-shared--")):
            return False
        path = PREVIOUS / "units" / unit
        if not (path / "COMPLETED").exists():
            return False
        if unit not in self.verified:
            check_external(unit, self.base.check)
            self.verified.add(unit)
        return True

    def unit_path(self, unit):
        return PREVIOUS / "units" / unit if self.external(unit) else self.base.unit_path(unit)

    def is_complete(self, unit, **kwargs):
        return self.external(unit) or self.base.is_complete(unit, **kwargs)

    def load_checkpoint(self, unit):
        return load_checkpoint(PREVIOUS, unit) if self.external(unit) else self.base.load_checkpoint(unit)


def external_bytes(root, data):
    return file_bytes(FAMILY, exclude=root) + file_bytes(data)
