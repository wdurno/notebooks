"""Read-only status and cooperative stop for Plan 17."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mnist_experiment.fisher_memory.control import request_stop, status
from .runtime import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "stop"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "stop":
        request_stop(args.root)
    result = status(args.root)
    if (args.root / "progress.json").exists():
        result["progress"] = read_json(args.root / "progress.json")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
