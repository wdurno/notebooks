"""Preview and execute the paired Plan 4 infrastructure smoke run."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.config import ExperimentConfig, load_config
from src.initialization import replica_bundle_id, replica_design_hash
from src.plan4_analysis import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    write_phase2_analysis,
)
from src.schedules import resolve_schedule

REPO_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIGS = (
    Path(__file__).with_name("configs") / "plan4_phase1_fixed_smoke.json",
    Path(__file__).with_name("configs") / "plan4_phase1_adaptive_smoke.json",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan 4 command center")
    parser.add_argument(
        "command", choices=("preview", "status", "run", "analyze-phase2")
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_CONFIGS),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_configs(paths: list[Path]) -> list[tuple[Path, ExperimentConfig]]:
    loaded = [(path, load_config(_resolved_path(path))) for path in paths]
    if not loaded:
        raise RuntimeError("at least one configuration is required")
    designs = {replica_design_hash(config) for _, config in loaded}
    if len(designs) != 1:
        raise RuntimeError("paired Plan 4 configs do not share one replica design")
    schedule_hashes = {
        resolve_schedule(config.data).content_hash for _, config in loaded
    }
    if len(schedule_hashes) != 1:
        raise RuntimeError("paired Plan 4 configs do not share one schedule")
    if any(config.data.schedule is None for _, config in loaded):
        raise RuntimeError("Plan 4 command-center configs require explicit schedules")
    return loaded


def _run_state(config: ExperimentConfig) -> tuple[str, Path]:
    run_root = REPO_ROOT / config.cache_root
    final = run_root / config.run_id
    incomplete = run_root / ".incomplete" / config.run_id
    if (final / "COMPLETED").is_file():
        try:
            stored = _read_json(final / "config.json")
        except (OSError, json.JSONDecodeError):
            return "invalid", final
        return (
            ("completed", final)
            if stored == config.to_mapping()
            else ("invalid", final)
        )
    if final.exists():
        return "invalid", final
    if incomplete.exists():
        try:
            stored = _read_json(incomplete / "config.json")
        except (OSError, json.JSONDecodeError):
            return "invalid", incomplete
        return (
            ("incomplete", incomplete)
            if stored == config.to_mapping()
            else ("invalid", incomplete)
        )
    return "missing", final


def _replica_path(config: ExperimentConfig) -> Path:
    return (
        REPO_ROOT
        / Path(config.cache_root).parent
        / "replicas"
        / replica_bundle_id(config)
    )


def _replica_state(config: ExperimentConfig) -> str:
    path = _replica_path(config)
    if (path / "COMPLETED").is_file():
        try:
            metadata = _read_json(path / "metadata.json")
        except (OSError, json.JSONDecodeError):
            return "invalid"
        return (
            "completed"
            if metadata.get("replica_design_hash") == replica_design_hash(config)
            else "invalid"
        )
    return "invalid" if path.exists() else "missing"


def status_rows(
    loaded: list[tuple[Path, ExperimentConfig]],
) -> list[dict[str, Any]]:
    rows = []
    for path, config in loaded:
        state, run_path = _run_state(config)
        rows.append(
            {
                "config": str(path),
                "policy": config.controller.policy,
                "config_hash": config.config_hash,
                "run_id": config.run_id,
                "run_state": state,
                "run_path": str(run_path.relative_to(REPO_ROOT)),
                "replica_bundle_id": replica_bundle_id(config),
                "replica_state": _replica_state(config),
            }
        )
    return rows


def _audit_completed(
    rows: list[dict[str, Any]],
    *,
    expected_schedule_hash: str | None = None,
) -> dict[str, Any]:
    uniform_hashes = set()
    schedule_hashes = set()
    total_hvps = 0
    for row in rows:
        if row["run_state"] != "completed":
            raise RuntimeError(f"Plan 4 smoke run is not complete: {row['run_id']}")
        run_path = REPO_ROOT / row["run_path"]
        schedule_path = run_path / "schedule_trajectory.json"
        metrics_path = run_path / "phase8_metrics.json"
        if not schedule_path.is_file() or not metrics_path.is_file():
            raise RuntimeError(f"Plan 4 smoke artifacts are incomplete: {run_path}")
        schedule = _read_json(schedule_path)
        metrics = _read_json(metrics_path)
        uniform_hashes.add(schedule["uniform_stream_hash"])
        schedule_hashes.add(schedule["schedule_hash"])
        total_hvps += sum(
            int(step["hvp_count"]) for step in metrics["condition_steps"]
        )
        if not all(step["same_pi_consumed"] for step in metrics["condition_steps"]):
            raise RuntimeError("smoke run violated the unified-pi contract")
    if len(uniform_hashes) != 1 or len(schedule_hashes) != 1:
        raise RuntimeError("paired smoke runs do not share schedule randomness")
    if (
        expected_schedule_hash is not None
        and schedule_hashes != {expected_schedule_hash}
    ):
        raise RuntimeError("completed smoke schedule differs from configuration")
    if total_hvps != 0:
        raise RuntimeError("EMA-only Plan 4 smoke unexpectedly calculated HVPs")
    return {
        "completed_runs": len(rows),
        "schedule_hash": next(iter(schedule_hashes)),
        "uniform_stream_hash": next(iter(uniform_hashes)),
        "total_hvp_count": total_hvps,
    }


def _execute(command: list[str]) -> None:
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _run_smoke(
    loaded: list[tuple[Path, ExperimentConfig]], *, resume: bool
) -> dict[str, Any]:
    first_path, first = loaded[0]
    replica_state = _replica_state(first)
    if replica_state == "invalid":
        raise RuntimeError(f"invalid replica bundle: {_replica_path(first)}")
    if replica_state == "missing":
        _execute(
            [
                sys.executable,
                "-m",
                "mnist_experiment.run_initialization",
                "--config",
                str(_resolved_path(first_path)),
            ]
        )

    for path, config in loaded:
        state, _ = _run_state(config)
        if state == "invalid":
            raise RuntimeError(f"invalid run state: {config.run_id}")
        if state == "completed":
            print(f"skip completed: {config.run_id}", flush=True)
            continue
        if state == "incomplete" and not resume:
            raise RuntimeError(
                f"incomplete run requires --resume: {config.run_id}"
            )
        command = [
            sys.executable,
            "-m",
            "mnist_experiment.run_controller",
            "--config",
            str(_resolved_path(path)),
        ]
        if state == "incomplete":
            command.append("--resume")
        _execute(command)

    return _audit_completed(
        status_rows(loaded),
        expected_schedule_hash=resolve_schedule(first.data).content_hash,
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "analyze-phase2":
        destination = write_phase2_analysis(
            REPO_ROOT,
            bootstrap_replicates=arguments.bootstrap_replicates,
            bootstrap_seed=arguments.bootstrap_seed,
        )
        summary = _read_json(destination / "summary.json")
        print(
            json.dumps(
                {
                    "decision": summary["decision"],
                    "selected_schedule": summary["selected_schedule"],
                    "diagnostic_schedule": summary["diagnostic_schedule"],
                    "screening_reference_schedule": summary[
                        "screening_reference_schedule"
                    ],
                    "path": str(destination.relative_to(REPO_ROOT)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    loaded = _load_configs(arguments.configs)
    schedule = resolve_schedule(loaded[0][1].data)
    if arguments.command == "preview":
        value = {
            "replica_bundle_id": replica_bundle_id(loaded[0][1]),
            "schedule": schedule.to_mapping(),
            "schedule_hash": schedule.content_hash,
            "expected_nines": (
                loaded[0][1].data.samples_per_step * sum(schedule.p_values)
            ),
            "conditions": status_rows(loaded),
        }
    elif arguments.command == "status":
        value = {"conditions": status_rows(loaded)}
    else:
        value = _run_smoke(loaded, resume=arguments.resume)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
