"""Read-only status and cooperative stop for Plan 16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mnist_experiment.fisher_memory.control import status, request_stop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "stop"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "stop":
        request_stop(args.root)
    result = status(args.root)
    if (args.root / "progress.json").exists():
        result["progress"] = json.loads((args.root / "progress.json").read_text())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
