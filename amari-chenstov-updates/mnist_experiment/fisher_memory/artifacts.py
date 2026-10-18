"""Bounded, locked Plan 15 artifact writes and immutable scientific units.

The caller supplies the complete scientific source/configuration identity and
unit provenance. Keep accepted metrics inside the checkpoint payload, not in a
separately appended log. Only export them after accepting that transaction.
"""

from __future__ import annotations

import contextlib
import csv
import errno
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from .control import ResourceLimit, RuntimeControl, StopRequested, utc_now

SCHEMA = 1
CHUNK_BYTES = 1024 * 1024
SYSTEM_FILES = {"identity.json", "runtime.json", "byte_ledger.json", "validation_index.json",
                ".run.lock", "STOP", "manifest.json", "latest.json", "COMPLETED"}


class ArtifactError(RuntimeError):
    """Missing, incompatible, or corrupt scientific artifacts."""


class ImmutableError(ArtifactError):
    """An existing scientific identity must not be overwritten."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read valid JSON: {path}") from exc


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_hash(path: Path, check: Callable[[], None] = lambda: None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            check()
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> list[int]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"artifact must be a regular file: {path}")
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _safe_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"invalid artifact component: {name!r}")
    return name


def _files(path: Path, check: Callable[[], None] = lambda: None) -> list[Path]:
    found = []
    for parent, directories, names in os.walk(path, followlinks=False):
        check()
        for name in directories:
            if (Path(parent) / name).is_symlink():
                raise ArtifactError(f"symlink directory is not an artifact: {Path(parent) / name}")
        for name in names:
            check()
            file = Path(parent) / name
            _fingerprint(file)
            found.append(file)
    return sorted(found)


def _load_index(root: Path) -> dict:
    path = root / "validation_index.json"
    if not path.exists():
        return {}
    # This is only an acceleration index. Corruption means revalidation, not repair of data.
    try:
        value = _read_json(path)
        if value["schema"] == SCHEMA and _hash_json(value["entries"]) == value["sha256"]:
            return value["entries"]
    except (ArtifactError, KeyError, TypeError):
        pass
    return {}


def _verify(root: Path, path: Path, record: dict, index: dict,
            check: Callable[[], None]) -> None:
    check()
    signature = _fingerprint(path)
    key = path.relative_to(root).as_posix()
    expected = {"fingerprint": signature, "sha256": record.get("sha256")}
    if signature[2] != record.get("bytes"):
        raise ArtifactError(f"artifact size mismatch: {path}")
    if index.get(key) != expected:
        if _file_hash(path, check) != record.get("sha256"):
            raise ArtifactError(f"artifact checksum mismatch: {path}")
        if _fingerprint(path) != signature:
            raise ArtifactError(f"artifact changed during validation: {path}")
        index[key] = expected


def _identity(root: Path) -> dict:
    value = _read_json(root / "identity.json")
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or value.get("identity_hash") != _hash_json(value.get("identity"))):
        raise ArtifactError("invalid store identity")
    return value


def _unit_record(root: Path, unit_id: str, identity_hash: str, index: dict,
                 check: Callable[[], None], verify: bool = True) -> dict:
    path = root / "units" / _safe_name(unit_id)
    manifest = _read_json(path / "manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("unit_id") != unit_id
            or manifest.get("identity_hash") != identity_hash
            or manifest.get("metadata_hash") != _hash_json(manifest.get("metadata"))):
        raise ArtifactError(f"unit identity mismatch: {path}")
    marker = path / "COMPLETED"
    completion = _read_json(marker) if marker.exists() else None
    if completion is not None:
        if (completion.get("schema") != SCHEMA or completion.get("unit_id") != unit_id
                or completion.get("identity_hash") != identity_hash
                or not isinstance(completion.get("files"), dict)
                or "manifest.json" not in completion["files"]
                or completion.get("seal_hash") != _hash_json(
                    {k: v for k, v in completion.items() if k != "seal_hash"})):
            raise ArtifactError(f"invalid completion marker: {path}")
        if verify:
            files = {p.relative_to(path).as_posix(): p for p in _files(path, check)
                     if p.name != "COMPLETED"}
            if set(files) != set(completion["files"]):
                raise ArtifactError(f"completed file set changed: {path}")
            for name, file in files.items():
                _verify(root, file, completion["files"][name], index, check)
    return {"unit_id": unit_id, "path": path, "metadata": manifest["metadata"],
            "manifest": manifest, "completion": completion,
            "status": "completed" if completion is not None else "incomplete"}


def iter_units(root: Path, *, completed_only: bool = False, verify: bool = True,
               check: Callable[[], None] | None = None, full: bool = False):
    """Read-only report iterator; validated artifacts only by default, no rewrites."""
    root = Path(root)
    check = check or (lambda: None)
    identity = _identity(root)
    index = {} if full else _load_index(root)
    for path in sorted((root / "units").glob("*")):
        check()
        if path.is_symlink() or not path.is_dir():
            raise ArtifactError(f"invalid unit path: {path}")
        record = _unit_record(root, path.name, identity["identity_hash"], index, check, verify)
        if not completed_only or record["status"] == "completed":
            yield record


class _CountWriter:
    def __init__(self, check: Callable[[], None]):
        self.position = 0
        self.check = check

    def write(self, data):
        self.check()
        size = len(data)
        self.position += size
        return size

    def tell(self):
        return self.position

    def flush(self):
        pass


class _BoundedWriter(_CountWriter):
    def __init__(self, handle, limit: int, check: Callable[[], None]):
        super().__init__(check)
        self.handle, self.limit = handle, limit
        self.error = None

    def write(self, data):
        try:
            if self.position + len(data) > self.limit:
                raise ResourceLimit("serialization exceeded its admitted byte bound")
            size = super().write(data)
            view = memoryview(data)
            for offset in range(0, size, CHUNK_BYTES):
                self.check()
                self.handle.write(view[offset:offset + CHUNK_BYTES])
            return size
        except BaseException as exc:
            self.error = exc
            raise

    def flush(self):
        self.handle.flush()


class Store:
    """One exclusive writer, held until close; reports use module-level iter_units.

    ``external_bytes`` charges shared dependencies once without copying them.
    ``reserve`` admits bounded external work; do not nest ordinary Store writes
    inside that reservation, since Store writes reserve their own peak bytes.
    Call ``heartbeat`` at accepted boundaries and during lengthy computation.
    """

    def __init__(self, root: Path, identity: dict, *, max_bytes: int = 500_000_000_000,
                 free_reserve_bytes: int = 268_435_456, resume: bool = False,
                 finalization_reserve_bytes: int = 1_048_576, external_bytes: int = 0,
                 max_wall_seconds: float | None = None,
                 max_active_seconds: float | None = None, clear_stop: bool = False):
        for value in (max_bytes, free_reserve_bytes, finalization_reserve_bytes, external_bytes):
            if not isinstance(value, int) or value < 0:
                raise ValueError("byte limits must be nonnegative integers")
        if max_bytes > 500_000_000_000:
            raise ValueError("Plan 15 cannot exceed the hard 500 GB artifact cap")
        if clear_stop and not resume:
            raise ValueError("clearing STOP requires resume=True")
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise ArtifactError("store root must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.free_reserve_bytes = max_bytes, free_reserve_bytes
        self.finalization_reserve_bytes, self.external_bytes = finalization_reserve_bytes, external_bytes
        self.resume = resume
        self.identity = json.loads(_json_bytes(identity))
        self.identity_hash = _hash_json(self.identity)
        self._closed = False
        self._reservations: dict[str, int] = {}
        self._sizes: dict[str, int] = {}
        self._used = 0
        self._index: dict = {}
        self._checkpoint_index: dict = {}
        self._persisted_index_hash = None
        self._last_reconcile = time.monotonic()
        if (self.root / ".run.lock").is_symlink():
            raise ArtifactError("run lock must not be a symlink")
        self._lock = (self.root / ".run.lock").open("a+b")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            self._closed = True
            raise ArtifactError("another writer owns this Plan 15 store") from exc
        try:
            path = self.root / "identity.json"
            if path.exists():
                if _identity(self.root)["identity"] != self.identity:
                    raise ImmutableError("scientific source/config identity changed; use a new root")
                if not resume:
                    raise ImmutableError("store already exists; pass resume=True")
            elif clear_stop:
                raise ArtifactError("clearing STOP requires an existing matching store identity")
            saved = _read_json(self.root / "runtime.json") if (self.root / "runtime.json").exists() else {}
            self.control = RuntimeControl(self.root, active_seconds=saved.get("cumulative_active_seconds", 0),
                                          max_wall_seconds=max_wall_seconds,
                                          max_active_seconds=max_active_seconds)
            if clear_stop:
                (self.root / "STOP").unlink(missing_ok=True)
                _sync_directory(self.root)
            self._cleanup_stale_temps()
            self.reconcile(persist=False)
            self._admit(0, finalizing=True)
            if not path.exists():
                self._atomic_bytes(path, _json_bytes({"schema": SCHEMA, "identity": self.identity,
                                                     "identity_hash": self.identity_hash}))
            (self.root / "units").mkdir(exist_ok=True)
            _sync_directory(self.root)
            self._index = _load_index(self.root)
            self.heartbeat(force=True)
        except BaseException:
            self._release_lock()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except (OSError, ResourceLimit):
            if exc is None:
                raise

    def _release_lock(self):
        if not self._closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._closed = True

    def close(self):
        if self._closed:
            return
        try:
            self.heartbeat(force=True)
        finally:
            self._release_lock()

    def _ensure_open(self):
        if self._closed:
            raise ArtifactError("store is closed")

    def check(self):
        self._ensure_open()
        self.control.check()

    def _cleanup_stale_temps(self):
        # Only our unpublished staging files are disposable after acquiring the lock.
        for path in _files(self.root, self.control.check):
            if path.name.startswith(".tmp-"):
                parts = path.relative_to(self.root).parts
                if (len(parts) >= 3 and parts[0] == "units"
                        and (self.root / "units" / parts[1] / "COMPLETED").exists()):
                    raise ArtifactError(f"temporary file inside completed unit: {path}")
                path.unlink()
                _sync_directory(path.parent)
        for unit in (self.root / "units").glob("*"):
            self.control.check()
            if (unit / "COMPLETED").exists() or not (unit / "manifest.json").exists():
                continue
            pointer = unit / "latest.json"
            latest = _read_json(pointer) if pointer.exists() else {}
            keep = {entry["file"] for entry in (latest.get("current"), latest.get("previous")) if entry}
            for checkpoint in unit.glob("checkpoint-*.pt"):
                if checkpoint.name not in keep:
                    checkpoint.unlink()
                    _sync_directory(unit)

    def _sync_size(self, path: Path):
        key = path.relative_to(self.root).as_posix()
        old = self._sizes.pop(key, 0)
        size = path.stat().st_size if path.exists() else 0
        self._used += size - old
        if path.exists():
            self._sizes[key] = size

    def reconcile(self, *, persist: bool = True, finalizing: bool = False):
        self._ensure_open()
        paths = _files(self.root, (lambda: None) if finalizing else self.control.check)
        self._sizes = {p.relative_to(self.root).as_posix(): p.stat().st_size for p in paths}
        self._used = sum(self._sizes.values())
        self._last_reconcile = time.monotonic()
        if persist:
            self._persist_ledger()
        return self.storage_status()

    def storage_status(self) -> dict:
        return {"used_bytes": self._used, "external_bytes": self.external_bytes,
                "reserved_bytes": sum(self._reservations.values()), "max_bytes": self.max_bytes,
                "finalization_reserve_bytes": self.finalization_reserve_bytes,
                "free_reserve_bytes": self.free_reserve_bytes,
                "filesystem_free_bytes": shutil.disk_usage(self.root).free}

    def _admit(self, size: int, *, finalizing: bool = False):
        if not isinstance(size, int) or size < 0:
            raise ValueError("reservation must be a nonnegative byte count")
        reserve = 0 if finalizing else self.finalization_reserve_bytes
        needed = size + sum(self._reservations.values()) + reserve
        if self._used + self.external_bytes + needed > self.max_bytes:
            raise ResourceLimit("Plan 15 artifact byte allocation would be exceeded")
        if shutil.disk_usage(self.root).free < needed + self.free_reserve_bytes:
            raise ResourceLimit("insufficient filesystem space including OS safety reserve")

    @contextlib.contextmanager
    def reserve(self, num_bytes: int, *, finalizing: bool = False):
        """Admit bounded direct report/scratch writes, then reconcile their sizes.

        External writers must obey this byte bound and stage under ``root``.
        They must not touch system files or scientific unit contents.
        """
        with self._reservation(num_bytes, finalizing=finalizing):
            try:
                yield
            finally:
                self.reconcile(persist=False, finalizing=True)

    @contextlib.contextmanager
    def _reservation(self, num_bytes: int, *, finalizing: bool = False):
        self._ensure_open()
        self._admit(num_bytes, finalizing=finalizing)
        token = uuid.uuid4().hex
        self._reservations[token] = num_bytes
        try:
            self._persist_ledger()
            yield token
        finally:
            self._reservations.pop(token, None)
            self._persist_ledger()

    def _persist_ledger(self):
        path = self.root / "byte_ledger.json"
        payload = {"schema": SCHEMA, "artifact_bytes": self._used - self._sizes.get(path.name, 0),
                   "external_bytes": self.external_bytes, "reserved_bytes": sum(self._reservations.values()),
                   "reservations": self._reservations, "max_bytes": self.max_bytes,
                   "finalization_reserve_bytes": self.finalization_reserve_bytes,
                   "updated_at": utc_now()}
        self._raw_atomic(path, _json_bytes(payload))

    def _raw_atomic(self, path: Path, data: bytes):
        # Small operational metadata uses the held finalization margin, avoiding recursion.
        self._admit(len(data), finalizing=True)
        temporary = path.with_name(f".tmp-{uuid.uuid4().hex}")
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            _sync_directory(path.parent)
        except OSError as exc:
            if exc.errno in (errno.ENOSPC, errno.EDQUOT):
                raise ResourceLimit("filesystem rejected artifact write") from exc
            raise
        finally:
            temporary.unlink(missing_ok=True)
            self._sync_size(path)

    def _atomic_bytes(self, path: Path, data: bytes, *, finalizing: bool = False):
        with self._reservation(len(data), finalizing=finalizing):
            # The data reservation already covers this write; metadata has separate headroom.
            temporary = path.with_name(f".tmp-{uuid.uuid4().hex}")
            try:
                with temporary.open("xb") as handle:
                    for offset in range(0, len(data), CHUNK_BYTES):
                        if not finalizing:
                            self.check()
                        handle.write(data[offset:offset + CHUNK_BYTES])
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                _sync_directory(path.parent)
            except OSError as exc:
                if exc.errno in (errno.ENOSPC, errno.EDQUOT):
                    raise ResourceLimit("filesystem rejected artifact write") from exc
                raise
            finally:
                temporary.unlink(missing_ok=True)
                self._sync_size(path)

    def heartbeat(self, *, force: bool = False):
        self._ensure_open()
        # A final checkpoint/close must not start a potentially long directory audit.
        if not force and time.monotonic() - self._last_reconcile >= 60:
            self.reconcile(persist=False)
        else:
            self._sync_size(self.root / "STOP")
        self._raw_atomic(self.root / "runtime.json", _json_bytes(self.control.snapshot()))
        if self._index:
            digest = _hash_json(self._index)
            if self._persisted_index_hash != digest:
                value = {"schema": SCHEMA, "entries": self._index, "sha256": digest}
                self._atomic_bytes(self.root / "validation_index.json", _json_bytes(value), finalizing=True)
                self._persisted_index_hash = digest
        self._persist_ledger()

    def unit_path(self, unit_id: str) -> Path:
        path = self.root / "units" / _safe_name(unit_id)
        if path.is_symlink():
            raise ArtifactError("unit path must not be a symlink")
        return path

    def _record(self, unit_id: str, *, finalizing: bool = False):
        self.unit_path(unit_id)
        return _unit_record(self.root, unit_id, self.identity_hash, self._index,
                            (lambda: None) if finalizing else self.check)

    def is_complete(self, unit_id: str, *, full: bool = False) -> bool:
        if not self.unit_path(unit_id).exists():
            return False
        if full:
            prefix = f"units/{unit_id}/"
            self._index = {key: value for key, value in self._index.items() if not key.startswith(prefix)}
        return self._record(unit_id)["status"] == "completed"

    def iter_units(self, *, completed_only: bool = False):
        for path in sorted((self.root / "units").iterdir()):
            self.check()
            record = self._record(path.name)
            if not completed_only or record["status"] == "completed":
                yield record

    def begin(self, unit_id: str, metadata: dict) -> Path | None:
        self.check()
        path = self.unit_path(unit_id)
        if (path / "manifest.json").exists():
            record = self._record(unit_id)
            if record["metadata"] != metadata:
                raise ImmutableError(f"unit metadata changed: {unit_id}")
            if not self.resume:
                raise ImmutableError(f"unit already exists; pass resume=True: {unit_id}")
            return None if record["status"] == "completed" else path
        if path.exists() and any(path.iterdir()):
            raise ArtifactError(f"unit lacks its identity manifest: {path}")
        path.mkdir(parents=True, exist_ok=True)
        _sync_directory(path.parent)
        manifest = {"schema": SCHEMA, "unit_id": unit_id, "identity_hash": self.identity_hash,
                    "metadata": metadata, "metadata_hash": _hash_json(metadata), "started_at": utc_now()}
        self._atomic_bytes(path / "manifest.json", _json_bytes(manifest))
        return path

    def _writable_unit(self, unit_id: str, *, finalizing: bool = False) -> Path:
        self._ensure_open()
        record = self._record(unit_id, finalizing=finalizing)
        if record["status"] == "completed":
            raise ImmutableError(f"completed unit is immutable: {unit_id}")
        return self.unit_path(unit_id)

    def _export_path(self, unit_id: str | None, name: str, *, finalizing: bool) -> Path:
        _safe_name(name)
        if name in SYSTEM_FILES or name.startswith("checkpoint-"):
            raise ValueError(f"reserved artifact filename: {name}")
        parent = self.root if unit_id is None else self._writable_unit(unit_id, finalizing=finalizing)
        return parent / name

    def write_json(self, unit_id: str | None, name: str, value: Any, *, finalizing: bool = False):
        path = self._export_path(unit_id, name, finalizing=finalizing)
        self._atomic_bytes(path, _json_bytes(value), finalizing=finalizing)
        return path

    def write_csv(self, unit_id: str | None, name: str, rows: Iterable[dict], *,
                  fieldnames: list[str] | None = None, finalizing: bool = False):
        rows = list(rows)
        fields = fieldnames if fieldnames is not None else list(dict.fromkeys(k for row in rows for k in row))
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        path = self._export_path(unit_id, name, finalizing=finalizing)
        self._atomic_bytes(path, output.getvalue().encode("utf-8"), finalizing=finalizing)
        return path

    def save_checkpoint(self, unit_id: str, payload: Any, *, finalizing: bool = False):
        path = self._writable_unit(unit_id, finalizing=finalizing)
        check = (lambda: None) if finalizing else self.check
        pointer = path / "latest.json"
        old = _read_json(pointer) if pointer.exists() else None
        if old is not None:
            self._checkpoint_record(unit_id, old["current"], check)
        envelope = {"schema": SCHEMA, "identity_hash": self.identity_hash, "unit_id": unit_id,
                    "manifest_hash": _hash_json(_read_json(path / "manifest.json")),
                    "active_seconds": self.control.active_seconds, "payload": payload}
        counter = _CountWriter(check)
        torch.save(envelope, counter)
        destination = path / f"checkpoint-{uuid.uuid4().hex}.pt"
        temporary = path / f".tmp-{uuid.uuid4().hex}"
        try:
            with self._reservation(counter.position, finalizing=finalizing):
                try:
                    with temporary.open("xb") as handle:
                        writer = _BoundedWriter(handle, counter.position, check)
                        try:
                            torch.save(envelope, writer)
                        except BaseException:
                            # PyTorch's ZIP finalizer can mask the underlying write failure.
                            if writer.error is not None:
                                raise writer.error
                            raise
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, destination)
                    _sync_directory(path)
                except OSError as exc:
                    if exc.errno in (errno.ENOSPC, errno.EDQUOT):
                        raise ResourceLimit("filesystem rejected checkpoint write") from exc
                    raise
                finally:
                    temporary.unlink(missing_ok=True)
                    self._sync_size(destination)
            current = {"file": destination.name, "bytes": destination.stat().st_size,
                       "sha256": _file_hash(destination, check)}
            latest = {"schema": SCHEMA, "current": current,
                      "previous": old["current"] if old else None}
            self._atomic_bytes(pointer, _json_bytes(latest), finalizing=finalizing)
        except BaseException:
            # A committed pointer, including a crash after rename, owns its target.
            committed = _read_json(pointer) if pointer.exists() else {}
            if committed.get("current", {}).get("file") != destination.name:
                destination.unlink(missing_ok=True)
                self._sync_size(destination)
            raise
        keep = {current["file"]}
        if latest["previous"]:
            keep.add(latest["previous"]["file"])
        for checkpoint in path.glob("checkpoint-*.pt"):
            if checkpoint.name not in keep:
                checkpoint.unlink()
                self._sync_size(checkpoint)
                self._checkpoint_index.pop(checkpoint.relative_to(self.root).as_posix(), None)
        _sync_directory(path)
        self.heartbeat(force=finalizing)
        return destination

    def _checkpoint_record(self, unit_id: str, descriptor: dict, check: Callable[[], None]) -> Path:
        name = _safe_name(descriptor["file"])
        if not name.startswith("checkpoint-") or not name.endswith(".pt"):
            raise ArtifactError("invalid checkpoint descriptor")
        path = self.unit_path(unit_id) / name
        # Frequent incomplete checkpoints must not rewrite the full completed-unit index.
        key = path.relative_to(self.root).as_posix()
        index = self._index if key in self._index else self._checkpoint_index
        _verify(self.root, path, descriptor, index, check)
        return path

    def load_checkpoint(self, unit_id: str, *, map_location="cpu", previous: bool = False):
        path = self.unit_path(unit_id)
        if not path.exists():
            return None
        self._record(unit_id)
        pointer = path / "latest.json"
        if not pointer.exists():
            return None
        latest = _read_json(pointer)
        if latest.get("schema") != SCHEMA:
            raise ArtifactError("checkpoint pointer schema mismatch")
        descriptor = latest.get("previous" if previous else "current")
        if descriptor is None:
            return None
        checkpoint = self._checkpoint_record(unit_id, descriptor, self.check)
        # Trusted local artifacts may contain optimizer and numpy RNG state.
        envelope = torch.load(checkpoint, map_location=map_location, weights_only=False)
        if (envelope.get("schema") != SCHEMA or envelope.get("identity_hash") != self.identity_hash
                or envelope.get("unit_id") != unit_id
                or envelope.get("manifest_hash") != _hash_json(_read_json(path / "manifest.json"))):
            raise ArtifactError("checkpoint identity mismatch")
        self.control.restore_active_seconds(envelope["active_seconds"])
        return envelope["payload"]

    def complete(self, unit_id: str, metadata: dict | None = None) -> Path:
        path = self._writable_unit(unit_id)
        if (path / "latest.json").exists():
            latest = _read_json(path / "latest.json")
            for descriptor in (latest["current"], latest.get("previous")):
                if descriptor:
                    self._checkpoint_record(unit_id, descriptor, self.check)
        records = {}
        for file in _files(path, self.check):
            if file.name.startswith(".tmp-"):
                raise ArtifactError("cannot complete unit with unpublished temporary files")
            signature = _fingerprint(file)
            digest = _file_hash(file, self.check)
            if _fingerprint(file) != signature:
                raise ArtifactError(f"artifact changed during sealing: {file}")
            records[file.relative_to(path).as_posix()] = {"bytes": signature[2], "sha256": digest}
            self._index[file.relative_to(self.root).as_posix()] = {
                "fingerprint": signature, "sha256": digest}
        marker = {"schema": SCHEMA, "unit_id": unit_id, "identity_hash": self.identity_hash,
                  "completed_at": utc_now(), "metadata": metadata or {}, "files": records}
        marker["seal_hash"] = _hash_json(marker)
        self._atomic_bytes(path / "COMPLETED", _json_bytes(marker))
        self._record(unit_id)
        self.heartbeat()
        return path
