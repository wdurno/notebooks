"""Cooperative Plan 15 runtime limits and a read-only status/stop CLI."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


class ResourceLimit(RuntimeError):
    """A storage or runtime allocation cannot admit more work."""


class StopRequested(RuntimeError):
    """Stop at a recoverable boundary; do not discard accepted state."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuntimeControl:
    def __init__(self, root: Path, *, active_seconds: float = 0.0,
                 max_wall_seconds: float | None = None,
                 max_active_seconds: float | None = None):
        for name, value in (("active_seconds", active_seconds),
                            ("max_wall_seconds", max_wall_seconds),
                            ("max_active_seconds", max_active_seconds)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        self.root = Path(root)
        self.started = time.monotonic()
        self.base_seconds = float(active_seconds)
        self.max_wall_seconds = max_wall_seconds
        self.max_active_seconds = max_active_seconds
        self.stop_requested = False

    @property
    def active_seconds(self) -> float:
        return self.base_seconds + time.monotonic() - self.started

    def restore_active_seconds(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid checkpoint compute counter")
        self.base_seconds += max(0.0, value - self.active_seconds)

    def snapshot(self) -> dict:
        return {"cumulative_active_seconds": self.active_seconds,
                "invocation_active_seconds": time.monotonic() - self.started,
                "heartbeat": utc_now(), "pid": os.getpid(),
                "max_wall_seconds": self.max_wall_seconds,
                "max_active_seconds": self.max_active_seconds,
                "stop_requested": self.stop_requested or (self.root / "STOP").exists()}

    def check(self) -> None:
        if self.stop_requested or (self.root / "STOP").exists():
            raise StopRequested("stop requested; checkpoint accepted state before exiting")
        if (self.max_wall_seconds is not None
                and time.monotonic() - self.started >= self.max_wall_seconds):
            raise ResourceLimit("invocation wall-clock allocation exhausted")
        if self.max_active_seconds is not None and self.active_seconds >= self.max_active_seconds:
            raise ResourceLimit("cumulative active-compute allocation exhausted")

    @contextlib.contextmanager
    def signal_handlers(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

        def request_stop(signum, frame):
            self.stop_requested = True

        try:
            for sig in previous:
                signal.signal(sig, request_stop)
            yield self
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def status(root: Path) -> dict:
    """Read small operational files only, without a writer or artifact replay."""
    root = Path(root)
    if not (root / "identity.json").is_file():
        raise FileNotFoundError(f"not a Plan 15 store: {root}")
    result = {"root": str(root), "stop_requested": (root / "STOP").exists(),
              "writer_active": False}
    lock_path = root / ".run.lock"
    if lock_path.exists():
        with lock_path.open("rb") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result["writer_active"] = True
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
    for key, name in (("runtime", "runtime.json"), ("storage", "byte_ledger.json")):
        path = root / name
        if path.exists():
            result[key] = json.loads(path.read_text(encoding="utf-8"))
    if "storage" in result:
        result["storage"]["used_bytes"] = (
            result["storage"]["artifact_bytes"] + (root / "byte_ledger.json").stat().st_size)
    return result


def request_stop(root: Path) -> None:
    """An empty durable marker consumes no artifact bytes; never clears a stop."""
    root = Path(root)
    if not (root / "identity.json").is_file():
        raise FileNotFoundError(f"not a Plan 15 store: {root}")
    try:
        descriptor = os.open(root / "STOP", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "stop"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "stop":
        request_stop(args.root)
    print(json.dumps(status(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
