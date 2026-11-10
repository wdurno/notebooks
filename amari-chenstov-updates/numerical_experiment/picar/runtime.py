"""Bounded scheduling, resource admission, and reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_metadata() -> dict:
    versions = {}
    for name in ("torch", "numpy", "transformers", "peft", "bitsandbytes", "pybullet", "gymnasium", "imageio", "imageio-ffmpeg", "flask", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "git_commit": git("rev-parse", "HEAD"),
        "dirty_worktree": None if status is None else bool(status),
        "recorded_at": utc_now(),
    }


def source_fingerprint() -> dict[str, str]:
    roots = [Path("numerical_experiment/picar"), Path("src/continual"), Path("src/robotics"), Path("demo/picar")]
    return {str(p): sha256_file(p) for root in roots for p in sorted(root.rglob("*.py"))}


class BudgetExceeded(RuntimeError):
    pass


class ArchiveUnavailable(RuntimeError):
    pass


def check_storage(root: Path, *, quota_bytes: int, reserve_bytes: int, incoming_bytes: int = 0) -> dict:
    if min(quota_bytes, reserve_bytes, incoming_bytes) < 0:
        raise ValueError("storage limits cannot be negative")
    if not root.is_dir():
        raise ArchiveUnavailable(f"artifact root unavailable: {root}")
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    free = shutil.disk_usage(root).free
    if used + incoming_bytes > quota_bytes:
        raise BudgetExceeded("artifact allocation would exceed its byte quota")
    reserve = max(reserve_bytes, int(0.1 * quota_bytes))
    if free - incoming_bytes < reserve:
        raise BudgetExceeded("filesystem free-space reserve would be violated")
    return {"used_bytes": used, "free_bytes": free, "quota_bytes": quota_bytes, "reserve_bytes": reserve}


@dataclass
class ComputeLedger:
    limit_s: float
    charged_s: float = 0.0

    def admit(self, estimated_s: float = 0.0) -> None:
        if estimated_s < 0 or self.charged_s + estimated_s >= self.limit_s:
            raise BudgetExceeded("active wall-clock allowance exhausted")

    @contextmanager
    def charge(self) -> Iterator[None]:
        self.admit()
        start = time.monotonic()
        try:
            yield
        finally:
            self.charged_s += time.monotonic() - start

    def state_dict(self) -> dict:
        return asdict(self)


class StopController:
    """Signals and STOP files request a bounded checkpoint, not an abrupt exit."""

    def __init__(self, root: Path):
        self.root = root
        self.requested = threading.Event()
        self._previous = {}

    def check(self) -> bool:
        return self.requested.is_set() or (self.root / "STOP").exists()

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal)

    def _signal(self, signum, frame) -> None:
        self.requested.set()

    def close(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()


class ModelAccess:
    """One GPU role at a time; queued control inference precedes new updates.

    An update keeps its writer transaction across cooperate() calls. The owner
    may yield only with unchanged actor parameters and no live functional
    override. After optimizer mutation it must publish without yielding again.
    Runtime owns no device API: callers supply a synchronization barrier when
    asynchronous device work needs draining at either side of a handoff.

    This cannot preempt an in-flight GPU kernel. A separate physics worker must
    enforce command deadlines while either role holds the lease.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._active = False
        self._active_owner: int | None = None
        self._update_owner: int | None = None
        self._cooperating = self._published = False
        self._waiting_inference = 0
        self._waiting_updates = 0
        self._stats = {"inference_leases": 0, "update_leases": 0,
                       "cooperate_calls": 0, "cooperative_yields": 0,
                       "inference_wait_s": 0., "max_inference_wait_s": 0.,
                       "update_wait_s": 0., "max_update_wait_s": 0.,
                       "cooperate_wait_s": 0., "max_cooperate_wait_s": 0.}
        self.version = 0

    def _reject_nested(self, owner: int) -> None:
        if owner in (self._active_owner, self._update_owner):
            raise RuntimeError("nested model leases are not allowed")

    def _record_wait(self, role: str, started: float) -> None:
        elapsed = time.monotonic() - started
        self._stats[f"{role}_wait_s"] += elapsed
        key = f"max_{role}_wait_s"
        self._stats[key] = max(self._stats[key], elapsed)

    def diagnostics(self) -> dict:
        """Constant-space counters and total/max waits, plus current admission state."""
        with self._condition:
            return {**self._stats, "waiting_inference": self._waiting_inference,
                    "waiting_updates": self._waiting_updates,
                    "update_active": self._update_owner is not None,
                    "inference_active": self._active and self._active_owner != self._update_owner,
                    "version": self.version}

    @contextmanager
    def inference(self) -> Iterator[int]:
        owner, started = threading.get_ident(), time.monotonic()
        with self._condition:
            self._reject_nested(owner)
            self._waiting_inference += 1
            try:
                self._condition.wait_for(lambda: not self._active)
                self._active = True
                self._active_owner = owner
                self._stats["inference_leases"] += 1
                self._record_wait("inference", started)
                version = self.version
            finally:
                self._waiting_inference -= 1
                self._condition.notify_all()
        try:
            yield version
        finally:
            with self._condition:
                self._active = False
                self._active_owner = None
                self._condition.notify_all()

    @contextmanager
    def update(self) -> Iterator[Callable[[], None]]:
        owner, started = threading.get_ident(), time.monotonic()
        with self._condition:
            self._reject_nested(owner)
            self._waiting_updates += 1
            try:
                self._condition.wait_for(lambda: not self._active and not self._waiting_inference
                                         and self._update_owner is None)
                self._active = True
                self._active_owner = self._update_owner = owner
                self._published = False
                self._stats["update_leases"] += 1
                self._record_wait("update", started)
            finally:
                self._waiting_updates -= 1

        lease_active = True
        def publish() -> None:
            with self._condition:
                if (not lease_active or threading.get_ident() != owner or self._update_owner != owner
                        or self._active_owner != owner or self._cooperating):
                    raise RuntimeError("publication requires the active update owner")
                if self._published:
                    raise RuntimeError("an update lease may publish only once")
                self.version += 1
                self._published = True

        try:
            yield publish
        finally:
            with self._condition:
                lease_active = False
                self._active = False
                self._active_owner = self._update_owner = None
                self._condition.notify_all()

    def cooperate(self, barrier: Callable[[], None] | None = None) -> bool:
        """Drain queued inference at a caller-certified graph-safe boundary.

        Only the active, unpublished update owner may call this. Return False
        without invoking barrier when no inference is queued. Otherwise barrier
        runs before releasing GPU access and after reclaiming it, if supplied.
        The writer transaction remains exclusive, even on barrier/inference
        failure; the caller regains GPU access before any exception propagates.
        """
        owner = threading.get_ident()
        if barrier is not None and not callable(barrier):
            raise TypeError("barrier must be callable")
        with self._condition:
            if self._update_owner != owner or self._active_owner != owner or self._cooperating:
                raise RuntimeError("cooperate requires the active update owner and cannot be nested")
            if self._published:
                raise RuntimeError("cannot cooperate after publication")
            self._stats["cooperate_calls"] += 1
            if not self._waiting_inference:
                return False
            self._cooperating = True
        try:
            if barrier is not None:
                barrier()
            try:
                with self._condition:
                    started = time.monotonic()
                    self._active = False
                    self._active_owner = None
                    self._stats["cooperative_yields"] += 1
                    self._condition.notify_all()
                    ready = lambda: not self._active and not self._waiting_inference
                    try:
                        self._condition.wait_for(ready)
                    finally:
                        # Rollback must not race inference, even on interruption.
                        self._condition.wait_for(ready)
                        self._active = True
                        self._active_owner = owner
                        self._record_wait("cooperate", started)
            finally:
                if barrier is not None:
                    barrier()
            return True
        finally:
            with self._condition:
                self._cooperating = False


class BoundedWorkQueue:
    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("queue capacity must be a positive integer")
        self._queue = queue.Queue(maxsize=capacity)

    def offer(self, value) -> bool:
        try:
            self._queue.put_nowait(value)
            return True
        except queue.Full:
            return False

    def take(self, timeout: float = 0.0):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __len__(self) -> int:
        return self._queue.qsize()
