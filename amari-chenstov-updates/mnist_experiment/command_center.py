"""Prepare, inspect, and execute immutable Phase 9 replica bundles."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from src.config import load_config
from src.phase9 import (
    Phase9Bundle,
    Phase9Error,
    build_phase9_bundle,
    load_phase9_bundle,
    load_phase9_spec,
    parse_replica_indices,
    phase9_status_rows,
    prepare_phase9_bundle,
)

REPO_ROOT = Path(__file__).parents[1]
DEFAULT_SPEC = Path(__file__).with_name("phase9_profiles.json")


def _profile_names(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names or len(names) != len(set(names)):
        raise Phase9Error("profiles must be a unique comma-separated list")
    return names


def _selection(arguments: argparse.Namespace):
    spec = load_phase9_spec(arguments.spec)
    replicas = parse_replica_indices(
        arguments.replicas,
        default_start=spec.default_replica_start,
        default_count=spec.default_replica_count,
    )
    return spec, _profile_names(arguments.profiles), replicas


def _summary(bundle: Phase9Bundle) -> dict[str, Any]:
    manifest = bundle.manifest
    return {
        "bundle_id": bundle.bundle_id,
        "path": str(bundle.path),
        "profiles": manifest["selection"]["profiles"],
        "replicas": manifest["selection"]["replica_indices"],
        "runs": manifest["entry_count"],
        "initializations": manifest["initialization_count"],
        "oracle_anchors": manifest["oracle_anchor_count"],
        "estimated_gpu_hours": manifest["estimated_seconds"] / 3600.0,
        "estimated_storage_gib": manifest["estimated_bytes"] / 2**30,
    }


def _print_status(rows: Sequence[dict[str, Any]]) -> None:
    columns = (
        "replica_id",
        "profile",
        "cell",
        "policy",
        "initialization_state",
        "run_state",
    )
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            "  ".join(
                str(row[column]).ljust(widths[column]) for column in columns
            )
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["run_state"]] = counts.get(row["run_state"], 0) + 1
    print("\nRun states:", json.dumps(counts, sort_keys=True))


def _run(command: list[str]) -> None:
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _ordered_entries(bundle: Phase9Bundle) -> list[dict[str, Any]]:
    entries = [dict(entry) for entry in bundle.entries]
    return sorted(
        entries,
        key=lambda entry: (
            entry["replica_index"],
            not entry["is_oracle_anchor"],
            entry["profile"],
            entry["kind"],
            entry["cell"],
        ),
    )


def _execute_bundle(
    bundle: Phase9Bundle,
    *,
    resume: bool,
    download: bool,
    max_runs: int | None,
) -> None:
    rows = phase9_status_rows(bundle, REPO_ROOT)
    invalid = [
        row
        for row in rows
        if row["run_state"] == "invalid"
        or row["initialization_state"] == "invalid"
    ]
    if invalid:
        raise Phase9Error(
            "bundle contains invalid artifacts: "
            + ", ".join(row["entry_id"] for row in invalid)
        )
    incomplete = [row for row in rows if row["run_state"] == "incomplete"]
    if incomplete and not resume:
        raise Phase9Error(
            "bundle contains incomplete runs; pass --resume: "
            + ", ".join(row["run_id"] for row in incomplete)
        )

    remaining_bytes = sum(
        int(row["estimated_run_bytes"])
        for row in rows
        if row["run_state"] != "completed"
    )
    free_bytes = shutil.disk_usage(REPO_ROOT).free
    if remaining_bytes and free_bytes < int(1.2 * remaining_bytes):
        raise Phase9Error(
            f"insufficient free disk: need roughly {1.2 * remaining_bytes / 2**30:.2f} GiB, "
            f"have {free_bytes / 2**30:.2f} GiB"
        )

    rows_by_entry = {row["entry_id"]: row for row in rows}
    initialized = set()
    executed = 0
    for entry in _ordered_entries(bundle):
        row = rows_by_entry[entry["entry_id"]]
        design = entry["replica_bundle_id"]
        if design not in initialized:
            if row["initialization_state"] != "completed":
                command = [
                    sys.executable,
                    "-m",
                    "mnist_experiment.run_initialization",
                    "--config",
                    row["config_path"],
                ]
                if download:
                    command.append("--download")
                _run(command)
            initialized.add(design)

        if row["run_state"] == "completed":
            print(f"skip completed: {entry['run_id']}", flush=True)
            continue
        if max_runs is not None and executed >= max_runs:
            print(f"stopped after --max-runs={max_runs}", flush=True)
            break
        command = [
            sys.executable,
            "-m",
            "mnist_experiment.run_controller",
            "--config",
            row["config_path"],
        ]
        if row["run_state"] == "incomplete":
            command.append("--resume")
        _run(command)
        executed += 1


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--profiles",
        help="comma-separated profile names; defaults to the spec selection",
    )
    parser.add_argument(
        "--replicas",
        help="indices such as 1-5 or 1,3,7; defaults to the spec selection",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 9 immutable experiment command center"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview")
    _add_selection_arguments(preview)

    prepare = subparsers.add_parser("prepare")
    _add_selection_arguments(prepare)

    status = subparsers.add_parser("status")
    status.add_argument("--bundle", required=True, type=Path)
    status.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--bundle", required=True, type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--download", action="store_true")
    run.add_argument("--max-runs", type=int)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "preview":
        spec, profiles, replicas = _selection(arguments)
        manifest, _ = build_phase9_bundle(
            spec,
            REPO_ROOT,
            profile_names=profiles,
            replica_indices=replicas,
        )
        print(json.dumps(_summary(Phase9Bundle(Path("<preview>"), manifest)), indent=2))
        return
    if arguments.command == "prepare":
        spec, profiles, replicas = _selection(arguments)
        bundle = prepare_phase9_bundle(
            spec,
            REPO_ROOT,
            profile_names=profiles,
            replica_indices=replicas,
        )
        print(json.dumps(_summary(bundle), indent=2))
        return
    bundle = load_phase9_bundle(arguments.bundle)
    if arguments.command == "status":
        rows = phase9_status_rows(bundle, REPO_ROOT)
        if arguments.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            _print_status(rows)
        return
    if arguments.max_runs is not None and arguments.max_runs < 1:
        raise Phase9Error("--max-runs must be positive")
    _execute_bundle(
        bundle,
        resume=arguments.resume,
        download=arguments.download,
        max_runs=arguments.max_runs,
    )


if __name__ == "__main__":
    main()
