"""Run bounded acceptance commands and charge their elapsed compute to Plan 18."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .runtime import FAMILY


def charge(label, start, end, returncode, log=None):
    FAMILY.mkdir(parents=True, exist_ok=True)
    with (FAMILY / ".acceptance.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = FAMILY / "acceptance_compute.json"
        ledger = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "entries": []}
        ledger["entries"].append(dict(label=label, start=start, end=end,
            seconds=end-start, returncode=returncode, log=log))
        intervals = sorted((entry["start"], entry["end"]) for entry in ledger["entries"])
        total, until = 0., 0.
        for lo, hi in intervals:
            total += max(0., hi-max(lo, until))
            until = max(until, hi)
        ledger["charged_seconds"] = total
        temporary = path.with_name(f".acceptance-{uuid.uuid4().hex}.tmp")
        with temporary.open("x") as handle:
            json.dump(ledger, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a bounded acceptance command is required")
    started = time.time()
    result = subprocess.run(command)
    charge(args.label, started, time.time(), result.returncode)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
