"""Restartable Phase 0-2 Plan 11 work ledger and command line."""

from __future__ import annotations

import argparse
import fcntl
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ..artifacts import RotatedArtifactError, _read_json, _write_json
from .analysis import development_decision, linear_selection, sigmoid_policies
from .artifacts import UnitStore
from .config import ANCHORS, GAINS, Policy, Study
from .run import TRAJECTORY_REQUIRED, run_trajectory


DEFAULT_ROOT = Path("cache/mnist_experiment/rotated_mnist/plan11")
DEFAULT_DATA = Path("cache/mnist_experiment/datasets")


def _freeze(path: Path, value: dict[str, Any], *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise RotatedArtifactError(f"Plan 11 ledger exists; pass --resume: {path}")
        if _read_json(path) != value:
            raise RotatedArtifactError(f"Plan 11 ledger differs from frozen run: {path}")
        return
    _write_json(path, value)


def _phase_contract(store: UnitStore, phase: str, *, resume: bool) -> None:
    _freeze(
        store.root / "contracts" / f"{phase}.json",
        {"study": store.study.mapping(), "study_hash": store.study.config_hash, "source_hashes": store.sources},
        resume=resume,
    )


def _ledger(store: UnitStore, stage: str, phase: str, policies: tuple[Policy, ...], schedules: tuple[str, ...], replicas: int, *, resume: bool) -> list[dict[str, Any]]:
    units = [
        store.unit(phase, index, schedule, policy.mapping())
        for index in range(1, replicas + 1)
        for schedule in schedules
        for policy in policies
    ]
    _freeze(
        store.root / "ledgers" / f"{stage}.json",
        {"study_hash": store.study.config_hash, "source_hashes": store.sources, "units": units},
        resume=resume,
    )
    return units


def _run_units(store: UnitStore, units: list[dict[str, Any]], *, data_root: Path, resume: bool, deadline: float | None) -> bool:
    for ordinal, unit in enumerate(units, 1):
        if deadline is not None and time.monotonic() >= deadline:
            print(json.dumps({"status": "paused", "done_or_examined": ordinal - 1, "planned": len(units)}))
            return False
        policy = Policy(**unit["policy"])
        path = run_trajectory(
            store,
            unit["phase"],
            unit["replica_index"],
            unit["schedule"],
            policy,
            data_root=data_root,
            resume=resume,
        )
        print(json.dumps({"status": "completed", "ordinal": ordinal, "planned": len(units), "path": str(path)}), flush=True)
    return True


def _benchmark_summary(store: UnitStore) -> dict[str, Any]:
    rows = []
    trajectory_bytes = []
    for schedule in ("linear", "sigmoid"):
        for anchor in ANCHORS:
            for gain in GAINS:
                unit = store.unit("benchmark", 1, schedule, Policy(anchor, gain).mapping())
                path = store.completed(unit, TRAJECTORY_REQUIRED)
                if path is None:
                    raise RotatedArtifactError("Plan 11 benchmark is incomplete")
                rows.append(_read_json(path / "summary.json"))
                trajectory_bytes.append(sum(item.stat().st_size for item in path.iterdir() if item.is_file()))
    asset_unit = store.unit("benchmark", 1, None, None)
    asset_path = store.completed(asset_unit, ("assets.pt", "summary.json"))
    if asset_path is None:
        raise RotatedArtifactError("Plan 11 benchmark initializer is incomplete")
    asset_bytes = sum(item.stat().st_size for item in asset_path.iterdir() if item.is_file())
    mean_seconds = statistics.fmean(row["total_wall_time_seconds"] for row in rows)
    mean_bytes = statistics.fmean(trajectory_bytes)
    result = {
        "study_hash": store.study.config_hash,
        "trajectory_count": len(rows),
        "mean_seconds_per_trajectory": mean_seconds,
        "median_seconds_per_trajectory": statistics.median(row["total_wall_time_seconds"] for row in rows),
        "max_seconds_per_trajectory": max(row["total_wall_time_seconds"] for row in rows),
        "asset_bytes_per_replica": asset_bytes,
        "mean_bytes_per_trajectory": mean_bytes,
        "development_300_trajectory_hours": 300 * mean_seconds / 3600,
        "confirmation_2560_trajectory_hours": 2560 * mean_seconds / 3600,
        "development_projected_bytes": 12 * asset_bytes + 300 * mean_bytes,
        "confirmation_projected_bytes": 256 * asset_bytes + 2560 * mean_bytes,
    }
    path = store.root / "decisions" / "benchmark_summary.json"
    if path.exists():
        if _read_json(path) != result:
            raise RotatedArtifactError("Plan 11 benchmark summary changed")
    else:
        _write_json(path, result)
    return result


def execute(store: UnitStore, stage: str, *, data_root: Path, resume: bool, max_wall_seconds: float | None) -> dict[str, Any]:
    if stage not in {"smoke", "benchmark", "development"}:
        raise ValueError("unsupported Plan 11 stage")
    if stage != "smoke" and store.study.smoke:
        raise ValueError("smoke configuration cannot run full trajectories")
    if stage == "smoke" and not store.study.smoke:
        raise ValueError("smoke stage needs the tiny smoke configuration")
    deadline = None if max_wall_seconds is None else time.monotonic() + max_wall_seconds
    phase = "development" if stage == "development" else stage
    _phase_contract(store, phase, resume=resume)
    if stage == "smoke":
        policies = tuple(Policy(c, g) for c in ANCHORS for g in (0.0, 0.05, 1.0))
        units = _ledger(store, "smoke", phase, policies, ("linear",), 1, resume=resume)
        done = _run_units(store, units, data_root=data_root, resume=resume, deadline=deadline)
        return {"status": "complete" if done else "paused", "stage": stage}
    if stage == "benchmark":
        policies = tuple(Policy(c, g) for c in ANCHORS for g in GAINS)
        units = _ledger(store, "benchmark", phase, policies, ("linear", "sigmoid"), 1, resume=resume)
        done = _run_units(store, units, data_root=data_root, resume=resume, deadline=deadline)
        return {"status": "paused", "stage": stage} if not done else _benchmark_summary(store)
    benchmark_path = store.root / "decisions" / "benchmark_summary.json"
    if not benchmark_path.is_file():
        raise RotatedArtifactError("Phase 0 benchmark is required before development")
    benchmark = _read_json(benchmark_path)
    if benchmark["study_hash"] != store.study.config_hash:
        raise RotatedArtifactError("benchmark study hash differs")
    policies = tuple(Policy(c, g) for c in ANCHORS for g in GAINS)
    units = _ledger(store, "development_linear", phase, policies, ("linear",), 12, resume=resume)
    done = _run_units(store, units, data_root=data_root, resume=resume, deadline=deadline)
    if not done:
        return {"status": "paused", "stage": "development_linear"}
    selection = linear_selection(store)
    if not selection["finalists"]:
        return development_decision(store, benchmark["mean_seconds_per_trajectory"])
    units = _ledger(store, "development_sigmoid", phase, sigmoid_policies(selection), ("sigmoid",), 12, resume=resume)
    done = _run_units(store, units, data_root=data_root, resume=resume, deadline=deadline)
    if not done:
        return {"status": "paused", "stage": "development_sigmoid"}
    return development_decision(store, benchmark["mean_seconds_per_trajectory"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "benchmark", "development"), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float)
    args = parser.parse_args()
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")
    study = Study.from_path(args.config)
    repo_root = Path(__file__).parents[3]
    store = UnitStore(args.output_root, study, repo_root)
    store.root.mkdir(parents=True, exist_ok=True)
    with (store.root / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = execute(
            store,
            args.stage,
            data_root=args.data_root,
            resume=args.resume,
            max_wall_seconds=args.max_wall_seconds,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
