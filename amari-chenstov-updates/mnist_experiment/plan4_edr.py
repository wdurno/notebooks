"""Prepare, run, and analyze Plan 4 discounted-risk control."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from src.plan4_challenge import load_screen_bundle, status_rows
from src.plan4_edr import write_edr_analysis, write_edr_bundle, write_edr_screen
from src.plan4_edr_discovery import (
    write_discovery_analysis,
    write_discovery_bundle,
)
from src.plan4_edr_stress import write_stress_analysis, write_stress_bundle


REPO_ROOT = Path(__file__).parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "screen",
            "prepare",
            "status",
            "run",
            "analyze",
            "prepare-discovery",
            "analyze-discovery",
            "prepare-stress",
            "analyze-stress",
        ),
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--floor-bundle", type=Path)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--discovery-bundle", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve(path: Path | None, name: str) -> Path:
    if path is None:
        raise RuntimeError(f"{name} is required")
    return path if path.is_absolute() else REPO_ROOT / path


def _run(bundle: Path, *, resume: bool) -> None:
    bundle, _ = load_screen_bundle(bundle)
    rows = status_rows(bundle, REPO_ROOT)
    if any(row["run_state"] == "invalid" for row in rows):
        raise RuntimeError("EDR bundle contains an invalid run")
    if any(row["run_state"] == "incomplete" for row in rows) and not resume:
        raise RuntimeError("EDR bundle has incomplete runs; pass --resume")
    for row in rows:
        if row["run_state"] == "completed":
            print(f"skip completed: {row['run_id']}", flush=True)
            continue
        if row["replica_state"] != "completed":
            raise RuntimeError(f"paired replica is missing: {row['replica_bundle_id']}")
        command = [
            sys.executable,
            "-m",
            "mnist_experiment.run_hybrid",
            "--config",
            row["config_path"],
        ]
        if row["run_state"] == "incomplete":
            command.append("--resume")
        print("$", shlex.join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "screen":
        destination = write_edr_screen(
            _resolve(arguments.bundle, "--bundle"),
            REPO_ROOT,
        )
        summary = json.loads(
            (destination / "summary.json").read_text(encoding="utf-8")
        )
        value = {
            "analysis": str(destination.relative_to(REPO_ROOT)),
            "gate": summary["gate"],
        }
    elif arguments.command == "prepare":
        destination = write_edr_bundle(
            _resolve(arguments.floor_bundle, "--floor-bundle"),
            _resolve(arguments.screen, "--screen"),
            REPO_ROOT,
        )
        _, manifest = load_screen_bundle(destination)
        value = {
            "bundle": str(destination.relative_to(REPO_ROOT)),
            "entries": manifest["entry_count"],
            "new_runs": manifest["new_run_count"],
            "estimated_seconds": manifest["estimated_seconds"],
        }
    elif arguments.command == "prepare-discovery":
        destination = write_discovery_bundle(
            _resolve(arguments.bundle, "--bundle"), REPO_ROOT
        )
        _, manifest = load_screen_bundle(destination)
        value = {
            "bundle": str(destination.relative_to(REPO_ROOT)),
            "entries": manifest["entry_count"],
            "new_runs": manifest["new_run_count"],
            "estimated_seconds": manifest["estimated_seconds"],
        }
    elif arguments.command == "prepare-stress":
        destination = write_stress_bundle(
            _resolve(arguments.bundle, "--bundle"),
            _resolve(arguments.discovery_bundle, "--discovery-bundle"),
            REPO_ROOT,
        )
        _, manifest = load_screen_bundle(destination)
        value = {
            "bundle": str(destination.relative_to(REPO_ROOT)),
            "entries": manifest["entry_count"],
            "new_runs": manifest["new_run_count"],
            "estimated_seconds": manifest["estimated_seconds"],
        }
    elif arguments.command == "status":
        value = {
            "conditions": status_rows(
                _resolve(arguments.bundle, "--bundle"), REPO_ROOT
            )
        }
    elif arguments.command == "run":
        bundle = _resolve(arguments.bundle, "--bundle")
        _run(bundle, resume=arguments.resume)
        value = {"conditions": status_rows(bundle, REPO_ROOT)}
    elif arguments.command == "analyze":
        destination = write_edr_analysis(
            _resolve(arguments.bundle, "--bundle"), REPO_ROOT
        )
        summary = json.loads(
            (destination / "summary.json").read_text(encoding="utf-8")
        )
        value = {
            "analysis": str(destination.relative_to(REPO_ROOT)),
            "aggregate_descriptive_comparisons": summary[
                "aggregate_descriptive_comparisons"
            ],
        }
    elif arguments.command == "analyze-discovery":
        destination = write_discovery_analysis(
            _resolve(arguments.bundle, "--bundle"), REPO_ROOT
        )
        summary = json.loads(
            (destination / "summary.json").read_text(encoding="utf-8")
        )
        value = {
            "analysis": str(destination.relative_to(REPO_ROOT)),
            "interpretation_gate": summary["interpretation_gate"],
        }
    else:
        destination = write_stress_analysis(
            _resolve(arguments.bundle, "--bundle"), REPO_ROOT
        )
        summary = json.loads(
            (destination / "summary.json").read_text(encoding="utf-8")
        )
        value = {
            "analysis": str(destination.relative_to(REPO_ROOT)),
            "schedules": list(summary["schedules"]),
        }
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
