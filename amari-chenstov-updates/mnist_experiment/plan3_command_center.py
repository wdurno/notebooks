"""Preview Plan 3 and execute its gated immutable bundles."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from src.phase9 import parse_replica_indices
from src.plan3 import (
    audit_plan3_handoff,
    load_plan3_spec,
    load_plan3_phase1_bundle,
    load_plan3_phase2_bundle,
    load_plan3_phase3_bundle,
    load_plan3_phase4_bundle,
    load_plan3_phase5_bundle,
    load_plan3_phase6_bundle,
    parse_stage_names,
    plan3_phase1_status_rows,
    plan3_phase2_status_rows,
    plan3_phase3_status_rows,
    plan3_phase4_status_rows,
    plan3_phase5_status_rows,
    plan3_phase6_status_rows,
    prepare_plan3_phase1_bundle,
    prepare_plan3_phase2_bundle,
    prepare_plan3_phase3_bundle,
    prepare_plan3_phase4_bundle,
    prepare_plan3_phase5_bundle,
    prepare_plan3_phase6_bundle,
    preview_plan3,
)
from src.plan3_analysis import (
    write_phase2_analysis,
    write_phase4_analysis,
    write_phase5_analysis,
    write_phase6_analysis,
)


DEFAULT_SPEC = Path(__file__).with_name("plan3_profiles.json")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gated Plan 3 command center")
    parser.add_argument(
        "command",
        choices=(
            "preview",
            "audit-handoff",
            "prepare-phase1",
            "phase1-status",
            "run-phase1",
            "prepare-phase2",
            "phase2-status",
            "run-phase2",
            "analyze-phase2",
            "prepare-phase3",
            "phase3-status",
            "run-phase3",
            "prepare-phase4",
            "phase4-status",
            "run-phase4",
            "analyze-phase4",
            "prepare-phase5",
            "phase5-status",
            "run-phase5",
            "analyze-phase5",
            "prepare-phase6",
            "phase6-status",
            "run-phase6",
            "analyze-phase6",
        ),
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--stages", help="comma-separated stage names")
    parser.add_argument("--replicas", help="indices such as 6-10 or 6,8")
    parser.add_argument(
        "--details",
        action="store_true",
        help="include per-condition estimands and planning costs",
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int)
    return parser.parse_args()


def _run_phase1(bundle_path: Path, *, resume: bool, max_runs: int | None) -> None:
    bundle = load_plan3_phase1_bundle(bundle_path)
    rows = plan3_phase1_status_rows(bundle, Path(__file__).parents[1])
    invalid = [row["run_id"] for row in rows if row["run_state"] == "invalid"]
    if invalid:
        raise RuntimeError("invalid Phase 1 runs: " + ", ".join(invalid))
    incomplete = [row["run_id"] for row in rows if row["run_state"] == "incomplete"]
    if incomplete and not resume:
        raise RuntimeError(
            "Phase 1 bundle has incomplete runs; pass --resume: "
            + ", ".join(incomplete)
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
            "mnist_experiment.run_replay",
            "--config",
            row["config_path"],
        ]
        if row["run_state"] == "incomplete":
            command.append("--resume")
        print("$", shlex.join(command), flush=True)
        subprocess.run(command, cwd=Path(__file__).parents[1], check=True)
        executed += 1


def _run_phase2(bundle_path: Path, *, resume: bool, max_runs: int | None) -> None:
    bundle = load_plan3_phase2_bundle(bundle_path)
    rows = plan3_phase2_status_rows(bundle, Path(__file__).parents[1])
    invalid = [
        row["run_id"]
        for row in rows
        if row["run_state"] == "invalid"
        or row["current_control_state"] != "completed"
        or row["ewc_control_state"] != "completed"
    ]
    if invalid:
        raise RuntimeError("invalid Phase 2 dependencies: " + ", ".join(invalid))
    incomplete = [row["run_id"] for row in rows if row["run_state"] == "incomplete"]
    if incomplete and not resume:
        raise RuntimeError(
            "Phase 2 bundle has incomplete runs; pass --resume: "
            + ", ".join(incomplete)
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
            "mnist_experiment.run_replay",
            "--config",
            row["config_path"],
        ]
        if row["run_state"] == "incomplete":
            command.append("--resume")
        print("$", shlex.join(command), flush=True)
        subprocess.run(command, cwd=Path(__file__).parents[1], check=True)
        executed += 1


def _run_phase3(bundle_path: Path, *, resume: bool, max_runs: int | None) -> None:
    bundle = load_plan3_phase3_bundle(bundle_path)
    rows = plan3_phase3_status_rows(bundle, Path(__file__).parents[1])
    invalid = [
        row["run_id"]
        for row in rows
        if row["run_state"] == "invalid"
        or row["archive_source_state"] != "completed"
    ]
    if invalid:
        raise RuntimeError("invalid Phase 3 dependencies: " + ", ".join(invalid))
    incomplete = [row["run_id"] for row in rows if row["run_state"] == "incomplete"]
    if incomplete and not resume:
        raise RuntimeError(
            "Phase 3 bundle has incomplete runs; pass --resume: "
            + ", ".join(incomplete)
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
            "mnist_experiment.run_hybrid",
            "--config",
            row["config_path"],
        ]
        if row["run_state"] == "incomplete":
            command.append("--resume")
        print("$", shlex.join(command), flush=True)
        subprocess.run(command, cwd=Path(__file__).parents[1], check=True)
        executed += 1


def _run_phase4(bundle_path: Path, *, resume: bool, max_runs: int | None) -> None:
    bundle = load_plan3_phase4_bundle(bundle_path)
    rows = plan3_phase4_status_rows(bundle, Path(__file__).parents[1])
    invalid = [
        row["run_id"]
        for row in rows
        if row["run_state"] == "invalid"
        or row["reused_controls_state"] != "completed"
        or row["archive_source_state"] not in {"completed", "not-applicable"}
    ]
    if invalid:
        raise RuntimeError("invalid Phase 4 dependencies: " + ", ".join(invalid))
    incomplete = [row["run_id"] for row in rows if row["run_state"] == "incomplete"]
    if incomplete and not resume:
        raise RuntimeError(
            "Phase 4 bundle has incomplete runs; pass --resume: "
            + ", ".join(incomplete)
        )
    executed = 0
    for row in rows:
        if row["run_state"] == "completed":
            print(f"skip completed: {row['run_id']}", flush=True)
            continue
        if max_runs is not None and executed >= max_runs:
            print(f"stopped after --max-runs={max_runs}", flush=True)
            break
        module = (
            "mnist_experiment.run_replay"
            if row["runner"] == "replay"
            else "mnist_experiment.run_hybrid"
        )
        command = [
            sys.executable,
            "-m",
            module,
            "--config",
            row["config_path"],
        ]
        if row["run_state"] == "incomplete":
            command.append("--resume")
        print("$", shlex.join(command), flush=True)
        subprocess.run(command, cwd=Path(__file__).parents[1], check=True)
        executed += 1


def _run_phase5(bundle_path: Path, *, resume: bool, max_runs: int | None) -> None:
    bundle = load_plan3_phase5_bundle(bundle_path)
    rows = plan3_phase5_status_rows(bundle, Path(__file__).parents[1])
    invalid = [
        row["run_id"]
        for row in rows
        if row["run_state"] == "invalid"
        or row["reused_controls_state"] != "completed"
        or row["archive_source_state"] not in {"completed", "not-applicable"}
    ]
    if invalid:
        raise RuntimeError("invalid Phase 5 dependencies: " + ", ".join(invalid))
    incomplete = [row["run_id"] for row in rows if row["run_state"] == "incomplete"]
    if incomplete and not resume:
        raise RuntimeError(
            "Phase 5 bundle has incomplete runs; pass --resume: "
            + ", ".join(incomplete)
        )
    executed = 0
    for row in rows:
        if row["run_state"] == "completed":
            print(f"skip completed: {row['run_id']}", flush=True)
            continue
        if max_runs is not None and executed >= max_runs:
            print(f"stopped after --max-runs={max_runs}", flush=True)
            break
        module = (
            "mnist_experiment.run_controller"
            if row["runner"] == "controller"
            else "mnist_experiment.run_hybrid"
        )
        command = [sys.executable, "-m", module, "--config", row["config_path"]]
        if row["run_state"] == "incomplete":
            command.append("--resume")
        print("$", shlex.join(command), flush=True)
        subprocess.run(command, cwd=Path(__file__).parents[1], check=True)
        executed += 1


def _run_phase6(bundle_path: Path, *, resume: bool, max_runs: int | None) -> None:
    bundle = load_plan3_phase6_bundle(bundle_path)
    rows = plan3_phase6_status_rows(bundle, Path(__file__).parents[1])
    invalid = [
        row["run_id"]
        for row in rows
        if row["run_state"] == "invalid"
        or row["archive_source_state"] not in {"completed", "not-applicable"}
    ]
    if invalid:
        raise RuntimeError("invalid Phase 6 dependencies: " + ", ".join(invalid))
    incomplete = [row["run_id"] for row in rows if row["run_state"] == "incomplete"]
    if incomplete and not resume:
        raise RuntimeError(
            "Phase 6 bundle has incomplete runs; pass --resume: "
            + ", ".join(incomplete)
        )
    executed = 0
    for row in rows:
        if row["run_state"] == "completed":
            print(f"skip completed: {row['run_id']}", flush=True)
            continue
        if max_runs is not None and executed >= max_runs:
            print(f"stopped after --max-runs={max_runs}", flush=True)
            break
        module = (
            "mnist_experiment.run_replay"
            if row["runner"] == "replay"
            else "mnist_experiment.run_hybrid"
        )
        command = [sys.executable, "-m", module, "--config", row["config_path"]]
        if row["run_state"] == "incomplete":
            command.append("--resume")
        print("$", shlex.join(command), flush=True)
        subprocess.run(command, cwd=Path(__file__).parents[1], check=True)
        executed += 1


def main() -> None:
    arguments = parse_arguments()
    spec = load_plan3_spec(arguments.spec)
    if arguments.command == "audit-handoff":
        print(json.dumps(audit_plan3_handoff(spec, Path(__file__).parents[1]), indent=2))
        return
    if arguments.command == "prepare-phase1":
        bundle = prepare_plan3_phase1_bundle(spec, Path(__file__).parents[1])
        print(
            json.dumps(
                {
                    "bundle_id": bundle.bundle_id,
                    "path": str(bundle.path),
                    "runs": bundle.manifest["config_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "prepare-phase2":
        bundle = prepare_plan3_phase2_bundle(spec, Path(__file__).parents[1])
        print(
            json.dumps(
                {
                    "bundle_id": bundle.bundle_id,
                    "path": str(bundle.path),
                    "replay_runs": bundle.manifest["replay_entry_count"],
                    "new_runs": bundle.manifest["new_run_count"],
                    "reused_pilot": bundle.manifest["reused_pilot_count"],
                    "estimated_remaining_minutes": (
                        bundle.manifest["estimated_remaining_seconds"] / 60.0
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "prepare-phase3":
        bundle = prepare_plan3_phase3_bundle(spec, Path(__file__).parents[1])
        print(
            json.dumps(
                {
                    "bundle_id": bundle.bundle_id,
                    "path": str(bundle.path),
                    "runs": bundle.manifest["config_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "prepare-phase4":
        bundle = prepare_plan3_phase4_bundle(spec, Path(__file__).parents[1])
        print(
            json.dumps(
                {
                    "bundle_id": bundle.bundle_id,
                    "path": str(bundle.path),
                    "new_runs": bundle.manifest["new_entry_count"],
                    "reused_runs": bundle.manifest["reused_entry_count"],
                    "estimated_remaining_minutes": (
                        bundle.manifest["estimated_remaining_seconds"] / 60.0
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "prepare-phase5":
        bundle = prepare_plan3_phase5_bundle(spec, Path(__file__).parents[1])
        print(
            json.dumps(
                {
                    "bundle_id": bundle.bundle_id,
                    "path": str(bundle.path),
                    "new_runs": bundle.manifest["new_entry_count"],
                    "reused_runs": bundle.manifest["reused_entry_count"],
                    "estimated_remaining_minutes": (
                        bundle.manifest["estimated_remaining_seconds"] / 60.0
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "prepare-phase6":
        bundle = prepare_plan3_phase6_bundle(spec, Path(__file__).parents[1])
        print(
            json.dumps(
                {
                    "bundle_id": bundle.bundle_id,
                    "path": str(bundle.path),
                    "runs": bundle.manifest["entry_count"],
                    "estimated_minutes": bundle.manifest["estimated_seconds"]
                    / 60.0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command in {"phase1-status", "run-phase1"}:
        if arguments.bundle is None:
            raise RuntimeError(f"{arguments.command} requires --bundle")
        if arguments.max_runs is not None and arguments.max_runs < 1:
            raise RuntimeError("--max-runs must be positive")
        bundle = load_plan3_phase1_bundle(arguments.bundle)
        if arguments.command == "phase1-status":
            print(
                json.dumps(
                    plan3_phase1_status_rows(bundle, Path(__file__).parents[1]),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        _run_phase1(
            arguments.bundle,
            resume=arguments.resume,
            max_runs=arguments.max_runs,
        )
        return
    if arguments.command in {"phase3-status", "run-phase3"}:
        if arguments.bundle is None:
            raise RuntimeError(f"{arguments.command} requires --bundle")
        if arguments.max_runs is not None and arguments.max_runs < 1:
            raise RuntimeError("--max-runs must be positive")
        bundle = load_plan3_phase3_bundle(arguments.bundle)
        if arguments.command == "phase3-status":
            print(
                json.dumps(
                    plan3_phase3_status_rows(bundle, Path(__file__).parents[1]),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        _run_phase3(
            arguments.bundle,
            resume=arguments.resume,
            max_runs=arguments.max_runs,
        )
        return
    if arguments.command in {"phase2-status", "run-phase2", "analyze-phase2"}:
        if arguments.bundle is None:
            raise RuntimeError(f"{arguments.command} requires --bundle")
        if arguments.max_runs is not None and arguments.max_runs < 1:
            raise RuntimeError("--max-runs must be positive")
        bundle = load_plan3_phase2_bundle(arguments.bundle)
        if arguments.command == "phase2-status":
            print(
                json.dumps(
                    plan3_phase2_status_rows(bundle, Path(__file__).parents[1]),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if arguments.command == "analyze-phase2":
            destination = write_phase2_analysis(bundle, Path(__file__).parents[1])
            print(json.dumps({"path": str(destination)}, indent=2))
            return
        _run_phase2(
            arguments.bundle,
            resume=arguments.resume,
            max_runs=arguments.max_runs,
        )
        return
    if arguments.command in {"phase4-status", "run-phase4", "analyze-phase4"}:
        if arguments.bundle is None:
            raise RuntimeError(f"{arguments.command} requires --bundle")
        if arguments.max_runs is not None and arguments.max_runs < 1:
            raise RuntimeError("--max-runs must be positive")
        bundle = load_plan3_phase4_bundle(arguments.bundle)
        if arguments.command == "phase4-status":
            print(
                json.dumps(
                    plan3_phase4_status_rows(bundle, Path(__file__).parents[1]),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if arguments.command == "analyze-phase4":
            destination = write_phase4_analysis(bundle, Path(__file__).parents[1])
            print(json.dumps({"path": str(destination)}, indent=2))
            return
        _run_phase4(
            arguments.bundle,
            resume=arguments.resume,
            max_runs=arguments.max_runs,
        )
        return
    if arguments.command in {"phase5-status", "run-phase5", "analyze-phase5"}:
        if arguments.bundle is None:
            raise RuntimeError(f"{arguments.command} requires --bundle")
        if arguments.max_runs is not None and arguments.max_runs < 1:
            raise RuntimeError("--max-runs must be positive")
        bundle = load_plan3_phase5_bundle(arguments.bundle)
        if arguments.command == "phase5-status":
            print(
                json.dumps(
                    plan3_phase5_status_rows(bundle, Path(__file__).parents[1]),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if arguments.command == "analyze-phase5":
            destination = write_phase5_analysis(bundle, Path(__file__).parents[1])
            print(json.dumps({"path": str(destination)}, indent=2))
            return
        _run_phase5(
            arguments.bundle,
            resume=arguments.resume,
            max_runs=arguments.max_runs,
        )
        return
    if arguments.command in {"phase6-status", "run-phase6", "analyze-phase6"}:
        if arguments.bundle is None:
            raise RuntimeError(f"{arguments.command} requires --bundle")
        if arguments.max_runs is not None and arguments.max_runs < 1:
            raise RuntimeError("--max-runs must be positive")
        bundle = load_plan3_phase6_bundle(arguments.bundle)
        if arguments.command == "phase6-status":
            print(
                json.dumps(
                    plan3_phase6_status_rows(bundle, Path(__file__).parents[1]),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if arguments.command == "analyze-phase6":
            destination = write_phase6_analysis(bundle, Path(__file__).parents[1])
            print(json.dumps({"path": str(destination)}, indent=2))
            return
        _run_phase6(
            arguments.bundle,
            resume=arguments.resume,
            max_runs=arguments.max_runs,
        )
        return
    stages = parse_stage_names(arguments.stages, spec)
    replicas = parse_replica_indices(
        arguments.replicas,
        default_start=spec.replica_indices[0],
        default_count=len(spec.replica_indices),
    )
    preview = preview_plan3(
        spec,
        stage_names=stages,
        replica_indices=replicas,
    )
    if not arguments.details:
        preview.pop("conditions")
    print(json.dumps(preview, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
