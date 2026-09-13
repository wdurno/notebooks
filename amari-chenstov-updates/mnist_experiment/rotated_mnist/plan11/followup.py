"""Frozen, resumable same-anchor .025 confirmation ledger."""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path
from typing import Any

from ..artifacts import RotatedArtifactError, _read_json, _write_json
from .artifacts import UnitStore, file_hash
from .config import Policy, Study
from .run import TRAJECTORY_REQUIRED, run_trajectory


PHASE = "followup025_v1"
SMOKE_PHASE = "followup025_smoke_v1"
REPLICAS = 352
SCHEDULES = ("linear", "sigmoid")
POLICIES = (Policy(0.025, 0.0), Policy(0.025, 0.025))
SOURCE = "mnist_experiment/rotated_mnist/plan11/followup.py"
DEFAULT_CONFIG = Path("mnist_experiment/rotated_mnist/plan11/configs/followup025.json")
DEFAULT_DATA = Path("cache/mnist_experiment/datasets")
DEFAULT_ROOT = Path("cache/mnist_experiment/rotated_mnist/plan11_followup025_v1")
SMOKE_ROOT = Path("cache/mnist_experiment/rotated_mnist/plan11_followup025_smoke_v1")


def make_store(root: Path, study: Study, repo_root: Path) -> UnitStore:
    store = UnitStore(root, study, repo_root)
    store.sources[SOURCE] = file_hash(repo_root / SOURCE)
    return store


def planned_units(store: UnitStore) -> list[dict[str, Any]]:
    phase = SMOKE_PHASE if store.study.smoke else PHASE
    count = 1 if store.study.smoke else REPLICAS
    return [
        store.unit(phase, index, schedule, policy.mapping())
        for index in range(1, count + 1)
        for schedule in SCHEDULES
        for policy in POLICIES
    ]


def contract(store: UnitStore) -> dict[str, Any]:
    return {
        "followup_schema_version": 1,
        "phase": SMOKE_PHASE if store.study.smoke else PHASE,
        "study": store.study.mapping(),
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "replicas_per_schedule": 1 if store.study.smoke else REPLICAS,
        "schedules": list(SCHEDULES),
        "policies": [policy.mapping() for policy in POLICIES],
        "endpoint": "environment_nll_auc",
        "contrast": "fixed_minus_blend",
        "two_sided_alpha_per_schedule": 0.05,
        "multiplicity_adjusted": False,
    }


def freeze(store: UnitStore, *, resume: bool) -> list[dict[str, Any]]:
    expected = (
        (store.root / "contract.json", contract(store)),
        (store.root / "ledger.json", {"units": planned_units(store)}),
    )
    for path, value in expected:
        if path.exists():
            if not resume:
                raise RotatedArtifactError(f"follow-up exists; pass --resume: {path}")
            if _read_json(path) != value:
                raise RotatedArtifactError(f"frozen follow-up differs: {path}")
        else:
            _write_json(path, value)
    return expected[1][1]["units"]


def execute(
    store: UnitStore,
    *,
    schedule: str,
    data_root: Path,
    resume: bool,
    max_pairs: int | None,
    max_wall_seconds: float | None,
) -> dict[str, Any]:
    units = freeze(store, resume=resume)
    selected = SCHEDULES if schedule == "both" else (schedule,)
    deadline = None if max_wall_seconds is None else time.monotonic() + max_wall_seconds
    completed_now = 0
    selected_count = 1 if store.study.smoke else REPLICAS
    for index in range(1, selected_count + 1):
        for kind in selected:
            pair = [unit for unit in units if unit["replica_index"] == index and unit["schedule"] == kind]
            if all(store.completed(unit, TRAJECTORY_REQUIRED) is not None for unit in pair):
                continue
            if max_pairs is not None and completed_now >= max_pairs:
                return {"status": "paused", "completed_pairs_this_invocation": completed_now}
            for unit in pair:
                if store.completed(unit, TRAJECTORY_REQUIRED) is not None:
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    return {"status": "paused", "completed_pairs_this_invocation": completed_now}
                run_trajectory(
                    store,
                    unit["phase"],
                    index,
                    kind,
                    Policy(**unit["policy"]),
                    data_root=data_root,
                    resume=resume,
                )
            completed_now += 1
            print(json.dumps({"completed_pair": index, "schedule": kind}), flush=True)
    return {"status": "requested_schedules_complete", "completed_pairs_this_invocation": completed_now}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--schedule", choices=(*SCHEDULES, "both"), default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    args = parser.parse_args()
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")
    repo_root = Path(__file__).parents[3]
    study = Study.from_path(args.config)
    root = args.output_root or (SMOKE_ROOT if study.smoke else DEFAULT_ROOT)
    store = make_store(root, study, repo_root)
    store.root.mkdir(parents=True, exist_ok=True)
    with (store.root / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = execute(
            store,
            schedule=args.schedule,
            data_root=args.data_root,
            resume=args.resume,
            max_pairs=args.max_pairs,
            max_wall_seconds=args.max_wall_seconds,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
