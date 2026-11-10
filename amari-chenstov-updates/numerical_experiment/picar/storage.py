"""Plan19 transactional local checkpoints, immutable frames and replay ledgers.

Only CURRENT.json commits evidence. JSON/CSV readers never unpickle a model.
The runner owns numerical events and must checkpoint learner/RNG/replay/pending
labels/simulator/queues together, rolling back its in-memory event on failure.
Torch payloads may be loaded only from a trusted local run, after verification.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence
import uuid

from . import ARTIFACT_SCHEMA_VERSION, METRIC_SCHEMA_VERSION
from .config import canonical_json
from .runtime import ArchiveUnavailable, BudgetExceeded, check_storage, runtime_metadata, sha256_file, utc_now
from .schemas import METRIC_COLUMNS, TransitionRecord, metric_row

STORAGE_SCHEMA_VERSION = 1
JOURNAL_SCHEMA_VERSION = 1
JOURNAL_CHECKPOINT_POLICY = {"name": "journal", "version": JOURNAL_SCHEMA_VERSION,
                             "recovery_snapshots": 2}


class IntegrityError(ValueError):
    pass


class RunCollisionError(FileExistsError):
    pass


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_fsynced(temporary, data)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    if path.is_symlink():
        raise IntegrityError(f"symlink is not a run artifact: {path}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"missing or invalid JSON artifact: {path}") from exc


def _safe_child(root: Path, name: str) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise IntegrityError("unsafe artifact path")
    result = root / relative
    if any(p.is_symlink() for p in (result, *result.parents) if p != root.parent):
        raise IntegrityError("symlink artifact paths are unsupported")
    return result


def _verify_file(path: Path, entry: Mapping[str, Any]) -> None:
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]
            or sha256_file(path) != entry["sha256"]):
        raise IntegrityError(f"artifact checksum mismatch: {path}")


class _StorageGuard:
    """One size inventory on opening, incremental accounting on each write."""

    def __init__(self, root: Path, *, quota_bytes: int, reserve_bytes: int):
        if root.is_symlink() or not root.is_dir():
            raise ArchiveUnavailable(f"artifact root unavailable: {root}")
        self.root = root
        info = root.stat()
        self.identity = (info.st_dev, info.st_ino)
        admission = check_storage(root, quota_bytes=quota_bytes, reserve_bytes=reserve_bytes)
        self.used_bytes = admission["used_bytes"]
        self.quota_bytes = quota_bytes
        self.reserve_bytes = admission["reserve_bytes"]

    def check(self, incoming_bytes: int = 0) -> None:
        try:
            info = self.root.stat()
        except OSError as exc:
            raise ArchiveUnavailable(f"artifact root vanished: {self.root}") from exc
        if (self.root.is_symlink() or not self.root.is_dir()
                or (info.st_dev, info.st_ino) != self.identity):
            raise ArchiveUnavailable(f"artifact root identity changed: {self.root}")
        if incoming_bytes < 0:
            raise ValueError("incoming bytes cannot be negative")
        if self.used_bytes + incoming_bytes > self.quota_bytes:
            raise BudgetExceeded("artifact allocation would exceed its byte quota")
        if shutil.disk_usage(self.root).free - incoming_bytes < self.reserve_bytes:
            raise BudgetExceeded("filesystem free-space reserve would be violated")


class FrameStore:
    """Write-once content-addressed NumPy RGB blobs; no pickle or external paths."""

    def __init__(self, root: str | Path, *, quota_bytes: int = 100_000_000_000,
                 reserve_bytes: int = 0, readonly: bool = False,
                 _guard: _StorageGuard | None = None, _writable_check=None):
        self.root = Path(root)
        self.readonly = readonly
        self._guard = _guard
        self._writable_check = _writable_check
        if not readonly:
            if any((parent / "COMPLETED").exists() for parent in (self.root, *self.root.parents)):
                raise RuntimeError("cannot write frames under a completed artifact")
            if self._guard is not None:
                self._guard.check()
            self.root.mkdir(parents=True, exist_ok=True)
            _fsync_directory(self.root.parent)
            if self._guard is None:
                self._guard = _StorageGuard(self.root, quota_bytes=quota_bytes, reserve_bytes=reserve_bytes)

    def _path(self, reference: str) -> Path:
        if not isinstance(reference, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", reference):
            raise IntegrityError("invalid frame reference")
        return _safe_child(self.root, reference[7:] + ".npy")

    def put(self, rgb) -> str:
        if self.readonly:
            raise RuntimeError("completed frame store is immutable")
        if self._writable_check is not None:
            self._writable_check()
        import numpy as np
        array = np.asarray(rgb)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or min(array.shape) < 1:
            raise ValueError("frame must be nonempty HWC uint8 RGB")
        if array.nbytes > 32_000_000:
            raise ValueError("frame exceeds bounded blob size")
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
        data = buffer.getvalue()
        reference = "sha256:" + _digest(data)
        path = self._path(reference)
        self._guard.check()
        if path.exists():
            self.verify(reference)
            return reference
        self._guard.check(len(data))
        temporary = self.root / f".{uuid.uuid4().hex}.tmp"
        try:
            _write_fsynced(temporary, data)
            try:
                os.link(temporary, path)  # Atomic no-overwrite publication.
            except FileExistsError:
                self.verify(reference)
            else:
                self._guard.used_bytes += len(data)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)
        return reference

    def verify(self, reference: str) -> Path:
        if self._guard is not None:
            self._guard.check()
        path = self._path(reference)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != reference[7:]:
            raise IntegrityError(f"missing/corrupt frame: {reference}")
        return path

    def get(self, reference: str):
        import numpy as np
        path = self.verify(reference)
        with path.open("rb") as stream:
            result = np.load(stream, allow_pickle=False)
        if result.dtype != np.uint8 or result.ndim != 3 or result.shape[2] != 3:
            raise IntegrityError("frame has incompatible RGB schema")
        return result


class FIFOReplay:
    """Exactly-once evidence identities; overflow stays eligible until retirement.

    admit() is for initially ready records. add_pending()/finalize() explicitly
    bridge delayed labels. Retirement only updates this ledger: the caller
    first performs learner.retire(), then checkpoints both states atomically.
    A checkpoint failure requires caller rollback of both in-memory objects.
    """

    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("replay capacity must be a positive integer")
        self.capacity = capacity
        self._eligible: OrderedDict[str, TransitionRecord] = OrderedDict()
        self._pending: OrderedDict[str, TransitionRecord] = OrderedDict()
        self._retired: dict[str, str] = {}
        self._transactions: dict[str, tuple[str, ...]] = {}
        self._arrivals: dict[int, str] = {}

    def __len__(self) -> int:
        return len(self._eligible)

    @property
    def eligible_ids(self) -> tuple[str, ...]:
        return tuple(self._eligible)

    @property
    def pending_ids(self) -> tuple[str, ...]:
        return tuple(self._pending)

    @property
    def retired_ids(self) -> tuple[str, ...]:
        return tuple(self._retired)

    def eligible(self) -> tuple[TransitionRecord, ...]:
        return tuple(self._eligible.values())

    def pending(self) -> tuple[TransitionRecord, ...]:
        return tuple(self._pending.values())

    def _new(self, record: TransitionRecord) -> None:
        key = record.transition_id
        if key in self._eligible or key in self._pending or key in self._retired:
            raise ValueError("duplicate transition admission")
        if record.arrival_index in self._arrivals:
            raise ValueError("arrival index already belongs to another transition")

    def _insert(self, record: TransitionRecord) -> None:
        self._eligible[record.transition_id] = record
        self._eligible = OrderedDict(sorted(self._eligible.items(), key=lambda item: item[1].arrival_index))

    def admit(self, record: TransitionRecord) -> tuple[str, ...]:
        self._new(record)
        if not record.ready or record.truncated:
            raise ValueError("pending/truncated transitions cannot enter replay")
        self._arrivals[record.arrival_index] = record.transition_id
        self._insert(record)
        return self.retirement_candidates()

    def add_pending(self, record: TransitionRecord) -> None:
        self._new(record)
        if record.ready or record.truncated:
            raise ValueError("pending queue requires an unlabeled, nontruncated transition")
        self._arrivals[record.arrival_index] = record.transition_id
        self._pending[record.transition_id] = record

    def finalize(self, record: TransitionRecord) -> tuple[str, ...]:
        original = self._pending.get(record.transition_id)
        if original is None or not record.ready:
            raise ValueError("finalization requires an existing pending ID and complete labels")
        if original.labeled(record.return_value, record.return_bin) != record:
            raise ValueError("finalization cannot change recorded experience")
        del self._pending[record.transition_id]
        self._insert(record)
        return self.retirement_candidates()

    def retirement_candidates(self) -> tuple[str, ...]:
        return self.eligible_ids[:max(0, len(self) - self.capacity)]

    def retire(self, ids: Iterable[str], *, transaction_id: str) -> bool:
        ids = tuple(ids)
        if not isinstance(transaction_id, str) or not transaction_id or not ids or len(set(ids)) != len(ids):
            raise ValueError("retirement requires unique IDs and a transaction ID")
        if transaction_id in self._transactions:
            if self._transactions[transaction_id] != ids:
                raise ValueError("retirement transaction ID conflict")
            return False
        if self.eligible_ids[:len(ids)] != ids:
            raise ValueError("retirement must remove an eligible FIFO prefix, never pending/retired IDs")
        for key in ids:
            del self._eligible[key]
            self._retired[key] = transaction_id
        self._transactions[transaction_id] = ids
        return True

    def sample(self, count: int, *, rng) -> tuple[TransitionRecord, ...]:
        if type(count) is not int or not 0 <= count <= len(self):
            raise ValueError("sample exceeds eligible replay")
        records = self.eligible()
        if isinstance(rng, random.Random):
            return tuple(rng.sample(records, count))
        indices = rng.choice(len(self), size=count, replace=False)
        return tuple(records[int(i)] for i in indices)

    def state_dict(self) -> dict:
        return {"schema_version": STORAGE_SCHEMA_VERSION, "capacity": self.capacity,
                "eligible": [record.to_dict() for record in self._eligible.values()],
                "pending": [record.to_dict() for record in self._pending.values()],
                "retired": dict(self._retired),
                "transactions": {key: list(ids) for key, ids in self._transactions.items()},
                "arrivals": [[index, key] for index, key in self._arrivals.items()]}

    def load_state_dict(self, state: dict) -> None:
        if state["schema_version"] != STORAGE_SCHEMA_VERSION or state["capacity"] != self.capacity:
            raise IntegrityError("replay schema/capacity mismatch")
        restored = FIFOReplay(self.capacity)
        for value in state["eligible"]:
            restored.admit(TransitionRecord.from_dict(value))
        for value in state["pending"]:
            restored.add_pending(TransitionRecord.from_dict(value))
        retired = dict(state["retired"])
        transactions = {key: tuple(ids) for key, ids in state["transactions"].items()}
        flattened = [key for ids in transactions.values() for key in ids]
        if (len(flattened) != len(set(flattened)) or set(flattened) != set(retired)
                or any(retired[key] != tx for tx, ids in transactions.items() for key in ids)
                or set(retired) & (set(restored._eligible) | set(restored._pending))):
            raise IntegrityError("invalid exactly-once retirement ledger")
        arrivals = dict(state["arrivals"])
        if (len(arrivals) != len(state["arrivals"]) or len(set(arrivals.values())) != len(arrivals)
                or set(arrivals.values()) != set(flattened) | set(restored._eligible) | set(restored._pending)
                or any(arrivals.get(record.arrival_index) != record.transition_id
                       for record in (*restored.eligible(), *restored.pending()))):
            raise IntegrityError("invalid replay arrival ledger")
        restored._retired, restored._transactions, restored._arrivals = retired, transactions, arrivals
        self.__dict__.update(restored.__dict__)


class RunStore:
    """One immutable run directory, single-writer lock and pointer transactions.

    checkpoint() accepts opaque full runner state. Include learner, named RNGs,
    replay.state_dict(), simulator.snapshot(), pending labels/context/actions,
    queues and cumulative clocks. records/metrics are JSON-only reporting data.
    Reuse an explicit transaction_id when retrying an interrupted publication.
    Root progress.json is a convenience mirror; CURRENT's progress is canonical.

    checkpoint_policy="journal-v1" opts into a separately versioned, hash-linked
    event journal. It adds checkpoint_policy to the resolved config identity;
    config_hash, if supplied, must hash that resolved config (available as
    store.config). Resume must explicitly select the same policy.
    Only current/previous and explicit milestone state.pt payloads are retained.
    Evidence, progress and transaction manifests are never pruned, nor are
    uncommitted orphans. Quotas must allow staging one full state before pruning.
    The default legacy layout and schema-1 reporting envelope are unchanged.
    """

    def __init__(self, root: str | Path, config: Mapping[str, Any], config_hash: str | None = None,
                 *, resume: bool = False, metadata: Mapping[str, Any] | None = None,
                 archive_root: str | Path | None = None,
                 local_quota_bytes: int = 100_000_000_000,
                 archive_quota_bytes: int = 1_000_000_000_000, reserve_bytes: int = 0,
                 allow_same_filesystem_archive: bool = False,
                 checkpoint_policy: str = "legacy"):
        self.root = Path(root).absolute()
        sealed_parent = any((parent / "COMPLETED").exists() for parent in self.root.parents)
        if sealed_parent and not (resume and (self.root / "COMPLETED").is_file()):
            raise RunCollisionError("cannot create or resume incomplete data under a completed artifact")
        self.config = json.loads(canonical_json(dict(config)))
        configured_policy = self.config.get("checkpoint_policy")
        if checkpoint_policy not in {"legacy", "journal-v1"}:
            raise ValueError("checkpoint_policy must be legacy or journal-v1")
        if configured_policy is not None and (checkpoint_policy != "journal-v1"
                                              or configured_policy != JOURNAL_CHECKPOINT_POLICY):
            raise ValueError("incompatible checkpoint policy in resolved configuration")
        self.checkpoint_policy = checkpoint_policy
        if checkpoint_policy == "journal-v1":
            self.config["checkpoint_policy"] = dict(JOURNAL_CHECKPOINT_POLICY)
        self.config_hash = _digest(canonical_json(self.config).encode())
        if config_hash is not None and config_hash != self.config_hash:
            raise ValueError("config hash does not match resolved configuration")
        self._lock = None
        self._closed = False
        self.completed = False
        self._current = None
        self._records: OrderedDict[str, TransitionRecord] = OrderedDict()
        self._metrics: list[dict] = []
        self._transactions: dict[str, dict] = {}
        self._recovery: list[dict] = []
        self._prunable: list[dict] = []
        self._verified_frames: set[str] = set()
        created = False
        try:
            try:
                self.root.mkdir(parents=True, exist_ok=False)
                created = True
            except FileExistsError:
                if not resume:
                    raise RunCollisionError(f"run already exists; use resume or a new directory: {self.root}")
                if self.root.is_symlink() or not self.root.is_dir():
                    raise RunCollisionError("run path is not a real directory")
            lock_flags = os.O_RDWR | os.O_NOFOLLOW
            if not (self.root / "COMPLETED").exists():
                lock_flags |= os.O_CREAT
            fd = os.open(self.root / ".writer.lock", lock_flags, 0o600)
            self._lock = os.fdopen(fd, "r+b")
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunCollisionError("another writer owns this run") from exc
            if created:
                (self.root / "transactions").mkdir()
                _fsync_directory(self.root)
            else:
                resolved = _read_json(self.root / "config.json")
                manifest = _read_json(self.root / "manifest.json")
                if (resolved != self.config or manifest["config_hash"] != self.config_hash
                        or manifest["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION
                        or manifest["metric_schema_version"] != METRIC_SCHEMA_VERSION
                        or manifest["storage_schema_version"] != STORAGE_SCHEMA_VERSION
                        or manifest.get("checkpoint_policy") != self.config.get("checkpoint_policy")):
                    raise RunCollisionError("run configuration or schemas differ")
                self.completed = (self.root / "COMPLETED").exists()
                if manifest["status"] == "completed" and not self.completed:
                    # Completion publication may have stopped before its marker;
                    # it remains resumable, never silently treated as complete.
                    self.completed = False
            self._local_guard = _StorageGuard(self.root, quota_bytes=local_quota_bytes, reserve_bytes=reserve_bytes)
            if archive_root is None and not created:
                stored_archive = manifest.get("archive")
                archive_root = None if stored_archive is None else stored_archive["root"]
            if archive_root is not None:
                archive = Path(archive_root).absolute()
                if archive.is_symlink() or not archive.is_dir():
                    raise ArchiveUnavailable(f"archive must already exist: {archive}")
                if archive.stat().st_dev == self.root.stat().st_dev and not allow_same_filesystem_archive:
                    raise ArchiveUnavailable("external archive is on the local filesystem; refusing silent SSD spill")
                self._archive_guard = _StorageGuard(archive, quota_bytes=archive_quota_bytes,
                                                     reserve_bytes=reserve_bytes)
                archive_info = {"root": str(archive), "identity": list(self._archive_guard.identity)}
                if not created and archive_info != manifest.get("archive"):
                    raise ArchiveUnavailable("archive identity or location differs from the run")
                frames_root = archive / "frames"
            else:
                self._archive_guard = self._local_guard
                archive_info = None
                if not created and manifest.get("archive") is not None:
                    raise ArchiveUnavailable("external archive cannot fall back to local storage")
                frames_root = self.root / "frames"
            self.frames = FrameStore(frames_root, readonly=self.completed, _guard=self._archive_guard,
                                     _writable_check=self._writable)
            if created:
                manifest = {"storage_schema_version": STORAGE_SCHEMA_VERSION,
                            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                            "metric_schema_version": METRIC_SCHEMA_VERSION,
                            "config_hash": self.config_hash, "started_at": utc_now(),
                            "completed_at": None, "status": "incomplete", "archive": archive_info,
                            "runtime": runtime_metadata(), "metadata": dict(metadata or {})}
                if self.checkpoint_policy == "journal-v1":
                    manifest["checkpoint_policy"] = dict(JOURNAL_CHECKPOINT_POLICY)
                for name, value in (("config.json", self.config), ("manifest.json", manifest)):
                    self._write_root(name, _json_bytes(value))
            self.manifest = manifest
            if (self.root / "CURRENT.json").exists():
                self._load_current()
            if self.completed:
                self.verify_completed()
            elif self.checkpoint_policy == "journal-v1":
                self._prune_journal()
        except BaseException:
            self.close()
            raise

    def _writable(self) -> None:
        if self._closed or self.completed or (self.root / "COMPLETED").exists():
            raise RuntimeError("run is closed or completed and immutable")
        if any((parent / "COMPLETED").exists() for parent in self.root.parents):
            raise RuntimeError("parent artifact is completed and immutable")
        self._local_guard.check()
        self._archive_guard.check()

    def _write_root(self, name: str, data: bytes) -> None:
        path = self.root / name
        self._local_guard.check(len(data))
        previous = path.stat().st_size if path.exists() else 0
        try:
            _atomic_bytes(path, data)
        except BaseException:
            if self.checkpoint_policy == "journal-v1":
                self._refresh_local_usage()
            raise
        self._local_guard.used_bytes += len(data) - previous

    def _refresh_local_usage(self) -> None:
        # Failed writes/renames can leave partial or unpointed payloads behind.
        self._local_guard.used_bytes = sum(path.stat().st_size for path in self.root.rglob("*")
                                           if path.is_file() and not path.is_symlink())

    def _load_current(self) -> None:
        pointer = _read_json(self.root / "CURRENT.json")
        if pointer.get("storage_schema_version") != STORAGE_SCHEMA_VERSION:
            raise IntegrityError("incompatible checkpoint pointer schema")
        if self.checkpoint_policy == "journal-v1":
            self._load_journal_current(pointer)
            return
        commit = pointer["commit_id"]
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{32}", commit):
            raise IntegrityError("invalid checkpoint ID")
        directory = _safe_child(self.root, f"transactions/{commit}")
        _verify_file(directory / "manifest.json", pointer["manifest"])
        manifest = _read_json(directory / "manifest.json")
        if (manifest["storage_schema_version"] != STORAGE_SCHEMA_VERSION
                or manifest["config_hash"] != self.config_hash or manifest["commit_id"] != commit):
            raise IntegrityError("checkpoint schema/config identity mismatch")
        for name, entry in manifest["files"].items():
            _verify_file(_safe_child(directory, name), entry)
        records = _read_json(directory / "records.json")
        self._records = OrderedDict()
        for value in records:
            record = TransitionRecord.from_dict(value)
            if record.transition_id in self._records:
                raise IntegrityError("duplicate committed transition")
            self._records[record.transition_id] = record
        self._metrics = [metric_row(**row) for row in _read_json(directory / "metrics.json")]
        self._transactions = manifest["transactions"]
        self._current = {"pointer": pointer, "manifest": manifest, "directory": directory}

    def _load_journal_current(self, pointer: dict) -> None:
        chain = []
        seen = set()
        while pointer is not None:
            commit = pointer.get("commit_id")
            if (pointer.get("storage_schema_version") != STORAGE_SCHEMA_VERSION
                    or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{32}", commit)
                    or commit in seen):
                raise IntegrityError("invalid or cyclic journal checkpoint pointer")
            seen.add(commit)
            directory = _safe_child(self.root, f"transactions/{commit}")
            _verify_file(directory / "manifest.json", pointer["manifest"])
            manifest = _read_json(directory / "manifest.json")
            if (manifest.get("storage_schema_version") != STORAGE_SCHEMA_VERSION
                    or manifest.get("journal_schema_version") != JOURNAL_SCHEMA_VERSION
                    or manifest.get("checkpoint_policy") != JOURNAL_CHECKPOINT_POLICY
                    or manifest.get("config_hash") != self.config_hash
                    or manifest.get("commit_id") != commit
                    or type(manifest.get("milestone")) is not bool
                    or set(manifest["files"]) != {"records.json", "metrics.json", "progress.json"}):
                raise IntegrityError("journal checkpoint schema/config identity mismatch")
            for name, entry in manifest["files"].items():
                _verify_file(_safe_child(directory, name), entry)
            state_path = _safe_child(directory, "state.pt")
            # Old non-milestone state is expendable; its sealed event is not.
            if len(chain) < 2 or manifest["milestone"] or state_path.exists():
                _verify_file(state_path, manifest["recovery_state"])
            parent = manifest["parent"]
            if manifest["parent_commit"] != (None if parent is None else parent.get("commit_id")):
                raise IntegrityError("journal parent identity mismatch")
            chain.append({"pointer": pointer, "manifest": manifest, "directory": directory})
            pointer = parent
        records: OrderedDict[str, TransitionRecord] = OrderedDict()
        metrics = []
        transactions = {}
        for sequence, entry in enumerate(reversed(chain), start=1):
            manifest, directory = entry["manifest"], entry["directory"]
            tx = manifest["transaction_id"]
            if (type(manifest["sequence"]) is not int or manifest["sequence"] != sequence
                    or not isinstance(tx, str) or not 0 < len(tx) <= 160
                    or tx in transactions or set(manifest["transactions"]) != {tx}):
                raise IntegrityError("invalid journal sequence/transaction identity")
            values = _read_json(directory / "records.json")
            event_metrics = [metric_row(**row) for row in _read_json(directory / "metrics.json")]
            fingerprint = _digest(_json_bytes({"state_sha256": manifest["recovery_state"]["sha256"],
                "records": values, "metrics": event_metrics,
                "progress": _read_json(directory / "progress.json"), "milestone": manifest["milestone"]}))
            receipt = {"commit_id": manifest["commit_id"], "fingerprint": fingerprint}
            if manifest["transactions"][tx] != receipt:
                raise IntegrityError("journal transaction fingerprint mismatch")
            incoming_ids = set()
            for value in values:
                record = TransitionRecord.from_dict(value)
                old = records.get(record.transition_id)
                if (record.transition_id in incoming_ids or old is not None and
                        (old.ready or not record.ready
                         or old.labeled(record.return_value, record.return_bin) != record)):
                    raise IntegrityError("invalid journal transition finalization")
                incoming_ids.add(record.transition_id)
                records[record.transition_id] = record
            metrics.extend(event_metrics)
            transactions[tx] = receipt
        self._records, self._metrics, self._transactions = records, metrics, transactions
        self._current = chain[0]
        self._recovery = list(reversed(chain[:2]))
        self._prunable = [entry for entry in chain[2:] if not entry["manifest"]["milestone"]
                          and (entry["directory"] / "state.pt").exists()]

    def _prune_journal(self) -> None:
        self._writable()
        if not self._prunable:
            return
        # Re-establish durability even if a prior CURRENT replacement raised
        # before its directory fsync. No pre-commit space reclamation is safe.
        _fsync_directory(self.root)
        while self._prunable:
            entry = self._prunable[0]
            path = _safe_child(entry["directory"], "state.pt")
            try:
                if path.exists():
                    _verify_file(path, entry["manifest"]["recovery_state"])
                    size = path.stat().st_size
                    path.unlink()
                    self._local_guard.used_bytes -= size
                _fsync_directory(path.parent)
            except BaseException:
                self._refresh_local_usage()
                raise
            self._prunable.pop(0)

    def load_checkpoint(self, *, trusted: bool = False) -> dict | None:
        if not trusted:
            raise ValueError("Torch checkpoints are trusted-local-only; pass trusted=True explicitly")
        if self._current is None:
            return None
        # Verify again immediately before unpickling, not only during resume.
        path = self._current["directory"] / "state.pt"
        manifest = self._current["manifest"]
        seal = manifest["recovery_state"] if self.checkpoint_policy == "journal-v1" else manifest["files"]["state.pt"]
        _verify_file(path, seal)
        import torch
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise IntegrityError("checkpoint runner state must be a dictionary")
        return state

    def records(self) -> tuple[TransitionRecord, ...]:
        return tuple(TransitionRecord.from_dict(record.to_dict()) for record in self._records.values())

    def metrics(self) -> list[dict]:
        return [dict(row) for row in self._metrics]

    def progress(self) -> dict:
        return {} if self._current is None else _read_json(self._current["directory"] / "progress.json")

    def checkpoint(self, state: dict, *, records: Sequence[TransitionRecord] = (),
                   metrics: Sequence[Mapping[str, Any]] = (), progress: dict | None = None,
                   transaction_id: str | None = None, milestone: bool = False) -> str:
        self._writable()
        if type(milestone) is not bool or milestone and self.checkpoint_policy != "journal-v1":
            raise ValueError("milestone requires the journal checkpoint policy and a boolean flag")
        if not isinstance(state, dict):
            raise ValueError("checkpoint requires full runner state as a dictionary")
        tx = transaction_id or uuid.uuid4().hex
        if not isinstance(tx, str) or not 0 < len(tx) <= 160:
            raise ValueError("transaction ID must be a bounded nonempty string")
        new_metrics = [metric_row(**dict(row)) for row in metrics]
        progress = json.loads(canonical_json(progress or {}))
        import torch
        buffer = io.BytesIO()
        torch.save(state, buffer)
        state_data = buffer.getvalue()
        event_identity = {"state_sha256": _digest(state_data),
                          "records": [record.to_dict() for record in records],
                          "metrics": new_metrics, "progress": progress}
        if self.checkpoint_policy == "journal-v1":
            event_identity["milestone"] = milestone
        fingerprint = _digest(_json_bytes(event_identity))
        if tx in self._transactions:
            previous = self._transactions[tx]
            if previous["fingerprint"] != fingerprint:
                raise ValueError("checkpoint transaction ID conflict")
            if self.checkpoint_policy == "journal-v1":
                self._prune_journal()
            return previous["commit_id"]
        new_records = OrderedDict(self._records)
        incoming_ids = set()
        for record in records:
            if record.transition_id in incoming_ids:
                raise ValueError("duplicate transition in checkpoint event")
            incoming_ids.add(record.transition_id)
            old = new_records.get(record.transition_id)
            if old is not None:
                if old == record:
                    raise ValueError("transition already committed; retry the transaction ID instead")
                if old.ready or not record.ready or old.labeled(record.return_value, record.return_bin) != record:
                    raise ValueError("committed experience may only finalize its pending label once")
            new_records[record.transition_id] = TransitionRecord.from_dict(record.to_dict())
        new_frame_refs = {reference for record in records for reference in
                          (record.observation_ref, record.next_observation_ref, *record.context_refs)}
        for reference in sorted(new_frame_refs - self._verified_frames):
            self.frames.verify(reference)
        commit = uuid.uuid4().hex
        transactions = {**self._transactions, tx: {"commit_id": commit, "fingerprint": fingerprint}}
        journal = self.checkpoint_policy == "journal-v1"
        files = {"state.pt": state_data,
                 "records.json": _json_bytes(event_identity["records"] if journal else
                                            [record.to_dict() for record in new_records.values()]),
                 "metrics.json": _json_bytes(new_metrics if journal else self._metrics + new_metrics),
                 "progress.json": _json_bytes(progress)}
        manifest = {"storage_schema_version": STORAGE_SCHEMA_VERSION, "config_hash": self.config_hash,
                    "commit_id": commit, "transaction_id": tx, "committed_at": utc_now(),
                    "parent_commit": None if self._current is None else self._current["pointer"]["commit_id"],
                    "transactions": transactions,
                    "files": {name: {"sha256": _digest(data), "bytes": len(data)} for name, data in files.items()}}
        if journal:
            manifest.update({"checkpoint_policy": dict(JOURNAL_CHECKPOINT_POLICY),
                             "journal_schema_version": JOURNAL_SCHEMA_VERSION,
                             "sequence": 1 if self._current is None else self._current["manifest"]["sequence"] + 1,
                             "parent": None if self._current is None else self._current["pointer"],
                             "milestone": milestone, "transactions": {tx: transactions[tx]},
                             "recovery_state": {"sha256": _digest(state_data), "bytes": len(state_data)},
                             "files": {name: {"sha256": _digest(data), "bytes": len(data)}
                                       for name, data in files.items() if name != "state.pt"}})
        manifest_data = _json_bytes(manifest)
        incoming = sum(map(len, files.values())) + len(manifest_data)
        self._local_guard.check(incoming + 16384)
        temporary = self.root / "transactions" / f".tmp-{commit}"
        temporary.mkdir()
        try:
            for name, data in files.items():
                _write_fsynced(temporary / name, data)
            _write_fsynced(temporary / "manifest.json", manifest_data)
            _fsync_directory(temporary)
            self._archive_guard.check()
            self._local_guard.check()
            destination = self.root / "transactions" / commit
            os.rename(temporary, destination)
            _fsync_directory(destination.parent)
            self._local_guard.used_bytes += incoming
            pointer = {"storage_schema_version": STORAGE_SCHEMA_VERSION, "commit_id": commit,
                       "manifest": {"sha256": _digest(manifest_data), "bytes": len(manifest_data)}}
            self._write_root("CURRENT.json", _json_bytes(pointer))
        except BaseException:
            # No cleanup is needed for safety: unpointed directories are orphans.
            # Inventory their bytes now so retries cannot undercount the quota.
            if self.checkpoint_policy == "journal-v1":
                self._refresh_local_usage()
            elif temporary.exists():
                self._local_guard.used_bytes += sum(path.stat().st_size for path in temporary.iterdir() if path.is_file())
            current_path = self.root / "CURRENT.json"
            if current_path.exists() and _read_json(current_path).get("commit_id") == commit:
                self._load_current()
            raise
        self._records, self._metrics = new_records, self._metrics + new_metrics
        self._transactions = transactions
        self._verified_frames.update(new_frame_refs)
        self._current = {"pointer": pointer, "manifest": manifest, "directory": destination}
        if self.checkpoint_policy == "journal-v1":
            self._recovery.append(self._current)
            if len(self._recovery) > 2:
                superseded = self._recovery.pop(0)
                if not superseded["manifest"]["milestone"]:
                    self._prunable.append(superseded)
            self._prune_journal()
        # Non-authoritative mirror is recoverable from CURRENT on read. Its
        # failure must not turn a successful commit into a repeated event.
        try:
            self._write_root("progress.json", _json_bytes(progress))
        except (OSError, BudgetExceeded):
            pass
        return commit

    def complete(self, summary: Mapping[str, Any] | None = None) -> dict:
        self._writable()
        if self._current is None:
            raise ValueError("cannot complete a run without a committed checkpoint")
        self._load_current()
        if self.checkpoint_policy == "journal-v1":
            self._prune_journal()
        if any(not record.ready or record.truncated for record in self._records.values()):
            raise ValueError("pending/truncated transition labels prevent scientific completion")
        references = {reference for record in self._records.values() for reference in
                      (record.observation_ref, record.next_observation_ref, *record.context_refs)}
        for reference in sorted(references):
            self.frames.verify(reference)
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(self._metrics)
        summary_data = _json_bytes(dict(summary or {}))
        metrics_data = csv_buffer.getvalue().encode()
        self._write_root("metrics.csv", metrics_data)
        self._write_root("summary.json", summary_data)
        final = {**self.manifest, "status": "completed", "completed_at": utc_now(),
                 "current": self._current["pointer"], "frame_count": len(references),
                 "final_files": {"metrics.csv": {"sha256": _digest(metrics_data), "bytes": len(metrics_data)},
                                 "summary.json": {"sha256": _digest(summary_data), "bytes": len(summary_data)}}}
        self._write_root("manifest.json", _json_bytes(final))
        self._archive_guard.check()
        # COMPLETED is the last durable publication; nothing writes after it.
        self._write_root("COMPLETED", _json_bytes({"manifest_sha256": _digest(_json_bytes(final))}))
        self.manifest, self.completed, self.frames.readonly = final, True, True
        return final

    def verify_completed(self) -> dict:
        marker = _read_json(self.root / "COMPLETED")
        manifest_path = self.root / "manifest.json"
        if marker["manifest_sha256"] != sha256_file(manifest_path):
            raise IntegrityError("completed manifest checksum mismatch")
        manifest = _read_json(manifest_path)
        if manifest["status"] != "completed" or manifest["config_hash"] != self.config_hash:
            raise IntegrityError("invalid completed manifest identity/status")
        for name, entry in manifest["final_files"].items():
            _verify_file(_safe_child(self.root, name), entry)
        if self._current is None or manifest["current"] != self._current["pointer"]:
            raise IntegrityError("completed pointer differs from the manifest")
        return manifest

    def close(self) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None
        self._closed = True

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *args) -> None:
        self.close()
