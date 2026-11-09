"""Explicit Plan 19 commands. Importing this module cannot load a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def status(root: Path) -> dict:
    units = []
    for path in sorted(root.rglob("manifest.json")):
        if "transactions" in path.parts:
            continue
        manifest = json.loads(path.read_text())
        progress = path.parent / "progress.json"
        units.append({"path": str(path.parent.relative_to(root)), "status": manifest.get("status"),
            "completed": (path.parent / "COMPLETED").exists(),
            "progress": json.loads(progress.read_text()) if progress.exists() else {}})
    return {"root": str(root), "units": units,
            "completed": sum(unit["completed"] for unit in units), "total": len(units)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="accelerated tiny CPU acceptance, not scientific evidence")
    smoke.add_argument("--config", type=Path, default=Path("numerical_experiment/picar/configs/smoke.json"))
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--archive-root", type=Path)
    smoke.add_argument("--max-events", type=int, help="bounded events per incomplete unit, for recovery tests")
    inspect = commands.add_parser("status")
    inspect.add_argument("--output", type=Path, required=True)
    stop = commands.add_parser("stop", help="request a checkpoint-boundary stop")
    stop.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="artifact-only notebook rebuild")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--notebook", type=Path, default=Path("numerical_experiment/picar/plan19_results.ipynb"))
    args = parser.parse_args(argv)
    if args.command == "smoke":
        from .engine import run_smoke
        if args.resume:
            (args.output / "STOP").unlink(missing_ok=True)
        result = run_smoke(load_config(args.config), args.output, resume=args.resume,
            archive_root=args.archive_root, max_events=args.max_events)
    elif args.command == "status":
        result = status(args.output)
    elif args.command == "stop":
        if not args.output.is_dir():
            raise FileNotFoundError(args.output)
        (args.output / "STOP").touch()
        result = {"status": "stop_requested", "output": str(args.output)}
    else:
        from .report import build_notebook
        build_notebook(args.output, args.notebook)
        result = {"notebook": str(args.notebook)}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
