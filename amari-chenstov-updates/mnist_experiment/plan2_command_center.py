"""Prepare, inspect, and execute immutable Plan 2 low-data bundles."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from src.phase9 import discover_phase9_bundles, parse_replica_indices
from src.classification_backfill import (
    build_classification_backfill,
    default_classification_backfill_root,
)
from src.plan2 import (
    Plan2Bundle,
    Plan2Error,
    build_plan2_bundle,
    load_plan2_bundle,
    load_plan2_spec,
    parse_samples_per_step,
    plan2_status_rows,
    prepare_plan2_bundle,
)
from src.plan2_analysis import PLAN2_M128_ANCHORS
from src.results_analysis import phase9_inventory_rows

REPO_ROOT = Path(__file__).parents[1]
DEFAULT_SPEC = Path(__file__).with_name("plan2_profiles.json")


def _selection(arguments: argparse.Namespace):
    spec = load_plan2_spec(arguments.spec)
    samples = parse_samples_per_step(
        arguments.samples_per_step, spec.default_samples_per_step
    )
    replicas = parse_replica_indices(
        arguments.replicas,
        default_start=spec.default_replica_start,
        default_count=spec.default_replica_count,
    )
    return spec, samples, replicas


def _summary(bundle: Plan2Bundle) -> dict[str, Any]:
    manifest = bundle.manifest
    return {
        "bundle_id": bundle.bundle_id,
        "path": str(bundle.path),
        "samples_per_step": manifest["selection"]["samples_per_step"],
        "replicas": manifest["selection"]["replica_indices"],
        "conditions": manifest["condition_count"],
        "runs": manifest["entry_count"],
        "derived_bundles": manifest["derived_bundle_count"],
        "new_initialization_fits": manifest["new_initialization_fit_count"],
        "estimated_wall_hours": manifest["estimated_seconds"] / 3600.0,
        "estimated_storage_gib": manifest["estimated_bytes"] / 2**30,
        "cost_basis": manifest["cost_model"]["basis"],
    }


def _print_status(rows: Sequence[dict[str, Any]]) -> None:
    columns = (
        "replica_id",
        "samples_per_step",
        "condition",
        "derived_bundle_state",
        "reference_state",
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


def _run(command: list[str]) -> None:
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _execute_bundle(
    bundle: Plan2Bundle,
    *,
    resume: bool,
    max_runs: int | None,
) -> None:
    rows = plan2_status_rows(bundle, REPO_ROOT)
    invalid = [
        row
        for row in rows
        if row["run_state"] == "invalid"
        or row["derived_bundle_state"] != "completed"
        or row["reference_state"] != "completed"
    ]
    if invalid:
        raise Plan2Error(
            "bundle has missing or invalid dependencies: "
            + ", ".join(row["entry_id"] for row in invalid)
        )
    incomplete = [row for row in rows if row["run_state"] == "incomplete"]
    if incomplete and not resume:
        raise Plan2Error(
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
        raise Plan2Error(
            f"insufficient free disk: need roughly "
            f"{1.2 * remaining_bytes / 2**30:.2f} GiB, "
            f"have {free_bytes / 2**30:.2f} GiB"
        )

    executed = 0
    for row in rows:
        if row["run_state"] == "completed":
            print(f"skip completed: {row['run_id']}", flush=True)
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


def _execute_classification_backfills(
    bundle: Plan2Bundle,
    *,
    device_name: str,
    resume: bool,
    max_runs: int | None,
    include_m128_anchors: bool,
) -> None:
    rows = plan2_status_rows(bundle, REPO_ROOT)
    incomplete = [row["run_id"] for row in rows if row["run_state"] != "completed"]
    if incomplete:
        raise Plan2Error(
            "classification backfill requires completed source runs: "
            + ", ".join(incomplete)
        )
    if include_m128_anchors:
        requested_replicas = set(bundle.manifest["selection"]["replica_indices"])
        anchor_pairs = {
            (profile, cell)
            for profile, cell, _ in PLAN2_M128_ANCHORS.values()
        }
        phase9_bundles = discover_phase9_bundles(
            REPO_ROOT / "cache" / "mnist_experiment" / "phase9" / "bundles"
        )
        anchors = [
            row
            for row in phase9_inventory_rows(phase9_bundles, REPO_ROOT)
            if int(row["replica_index"]) in requested_replicas
            and (row["profile"], row["cell"]) in anchor_pairs
        ]
        expected_anchor_count = len(requested_replicas) * len(anchor_pairs)
        if len(anchors) != expected_anchor_count or any(
            row["run_state"] != "completed" for row in anchors
        ):
            raise Plan2Error("the requested m=128 anchor runs are not complete")
        rows = [*rows, *anchors]
    rows = list({str(row["run_id"]): row for row in rows}.values())
    cache_root = REPO_ROOT / "cache" / "mnist_experiment"
    executed = 0
    for row in rows:
        if max_runs is not None and executed >= max_runs:
            print(f"stopped after --max-runs={max_runs}", flush=True)
            break
        destination = build_classification_backfill(
            row["run_path"],
            repo_root=REPO_ROOT,
            output_root=default_classification_backfill_root(REPO_ROOT),
            data_root=cache_root / "datasets",
            replica_root=cache_root / "replicas",
            device_name=device_name,
            resume=resume,
        )
        print(f"classification metrics ready: {destination}", flush=True)
        executed += 1


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--samples-per-step",
        help="explicit comma-separated m values; defaults to the spec",
    )
    parser.add_argument(
        "--replicas",
        help="indices such as 1-3 or 1,3; defaults to the spec",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan 2 low-data command center")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_selection(commands.add_parser("preview"))
    _add_selection(commands.add_parser("prepare"))
    status = commands.add_parser("status")
    status.add_argument("--bundle", required=True, type=Path)
    status.add_argument("--json", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--bundle", required=True, type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-runs", type=int)
    backfill = commands.add_parser("backfill-metrics")
    backfill.add_argument("--bundle", required=True, type=Path)
    backfill.add_argument("--device", default="auto")
    backfill.add_argument("--resume", action="store_true")
    backfill.add_argument("--max-runs", type=int)
    backfill.add_argument("--include-m128-anchors", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "preview":
        spec, samples, replicas = _selection(arguments)
        manifest, _ = build_plan2_bundle(
            spec,
            REPO_ROOT,
            samples_per_step=samples,
            replica_indices=replicas,
        )
        print(json.dumps(_summary(Plan2Bundle(Path("<preview>"), manifest)), indent=2))
        return
    if arguments.command == "prepare":
        spec, samples, replicas = _selection(arguments)
        bundle = prepare_plan2_bundle(
            spec,
            REPO_ROOT,
            samples_per_step=samples,
            replica_indices=replicas,
        )
        print(json.dumps(_summary(bundle), indent=2))
        return
    bundle = load_plan2_bundle(arguments.bundle)
    if arguments.command == "status":
        rows = plan2_status_rows(bundle, REPO_ROOT)
        if arguments.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            _print_status(rows)
        return
    if arguments.max_runs is not None and arguments.max_runs < 1:
        raise Plan2Error("--max-runs must be positive")
    if arguments.command == "backfill-metrics":
        _execute_classification_backfills(
            bundle,
            device_name=arguments.device,
            resume=arguments.resume,
            max_runs=arguments.max_runs,
            include_m128_anchors=arguments.include_m128_anchors,
        )
        return
    _execute_bundle(bundle, resume=arguments.resume, max_runs=arguments.max_runs)


if __name__ == "__main__":
    main()
