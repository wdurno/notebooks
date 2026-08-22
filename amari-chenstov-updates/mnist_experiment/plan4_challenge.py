"""Prepare, execute, and analyze the Plan 4 realized-actuation challenge."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from src.config import load_config
from src.initialization import derive_scheduled_replica_bundle
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.plan4_floor import write_floor_analysis, write_floor_bundle
from src.plan4_challenge import (
    load_screen_bundle,
    status_rows,
    write_actuation_analysis,
    write_screen_bundle,
)


REPO_ROOT = Path(__file__).parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "prepare-floor", "status", "run", "analyze", "analyze-floor"),
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--replica", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--baseline-bundle", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int)
    return parser.parse_args()


def _execute(command: list[str]) -> None:
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _bundle_path(arguments: argparse.Namespace) -> Path:
    if arguments.bundle is None:
        raise RuntimeError("this command requires --bundle")
    return arguments.bundle if arguments.bundle.is_absolute() else REPO_ROOT / arguments.bundle


def _run(bundle_path: Path, *, resume: bool, max_runs: int | None) -> None:
    bundle, _ = load_screen_bundle(bundle_path)
    rows = status_rows(bundle, REPO_ROOT)
    if any(row["run_state"] == "invalid" for row in rows):
        raise RuntimeError("challenge contains an invalid run")
    if any(row["run_state"] == "incomplete" for row in rows) and not resume:
        raise RuntimeError("challenge has incomplete runs; pass --resume")
    executed = 0
    initialized = set()
    train_targets = None
    for row in rows:
        bundle_id = row["replica_bundle_id"]
        if row["replica_state"] != "completed" and bundle_id not in initialized:
            if train_targets is None:
                train_dataset, _ = load_mnist_datasets(
                    REPO_ROOT / "cache" / "mnist_experiment" / "datasets",
                    download=False,
                )
                train_targets = dataset_targets(train_dataset)
            config = load_config(row["config_path"])
            derive_scheduled_replica_bundle(
                REPO_ROOT / row["source_replica_bundle"],
                REPO_ROOT / Path(config.cache_root).parent / "replicas",
                config,
                train_targets,
                repo_root=REPO_ROOT,
            )
            print(f"derived scheduled replica: {bundle_id}", flush=True)
            initialized.add(bundle_id)
        if row["run_state"] == "completed":
            print(f"skip completed: {row['run_id']}", flush=True)
            continue
        if max_runs is not None and executed >= max_runs:
            print(f"stopped after --max-runs={max_runs}", flush=True)
            break
        command = [
            sys.executable,
            "-m",
            "mnist_experiment.run_hybrid",
            "--config",
            row["config_path"],
        ]
        if row["run_state"] == "incomplete":
            command.append("--resume")
        _execute(command)
        executed += 1


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "prepare":
        destination = write_screen_bundle(
            REPO_ROOT,
            replica=arguments.replica,
            device=arguments.device,
        )
        _, manifest = load_screen_bundle(destination)
        value = {
            "bundle": str(destination.relative_to(REPO_ROOT)),
            "entries": manifest["entry_count"],
            "estimated_seconds": manifest["estimated_seconds"],
        }
    elif arguments.command == "prepare-floor":
        if arguments.baseline_bundle is None:
            raise RuntimeError("prepare-floor requires --baseline-bundle")
        baseline = _bundle_path(
            argparse.Namespace(bundle=arguments.baseline_bundle)
        )
        destination = write_floor_bundle(
            REPO_ROOT,
            baseline,
            device=arguments.device,
        )
        _, manifest = load_screen_bundle(destination)
        value = {
            "bundle": str(destination.relative_to(REPO_ROOT)),
            "entries": manifest["entry_count"],
            "new_runs": manifest["new_run_count"],
            "estimated_seconds": manifest["estimated_seconds"],
        }
    elif arguments.command == "status":
        value = {"conditions": status_rows(_bundle_path(arguments), REPO_ROOT)}
    elif arguments.command == "run":
        path = _bundle_path(arguments)
        _run(path, resume=arguments.resume, max_runs=arguments.max_runs)
        value = {"conditions": status_rows(path, REPO_ROOT)}
    elif arguments.command == "analyze":
        destination = write_actuation_analysis(_bundle_path(arguments), REPO_ROOT)
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        value = {
            "analysis": str(destination.relative_to(REPO_ROOT)),
            "decision": summary["decision"],
            "selected_schedule": summary["selected_schedule"],
        }
    else:
        destination = write_floor_analysis(_bundle_path(arguments), REPO_ROOT)
        value = {
            "analysis": str(destination.relative_to(REPO_ROOT)),
            "confirmatory": False,
        }
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
