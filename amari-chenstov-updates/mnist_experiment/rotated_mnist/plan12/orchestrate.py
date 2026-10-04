"""Autonomous, bounded, and resumable execution of Plan 12."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ..artifacts import _read_json, _write_json
from .analysis import (
    PHASE1_ANALYSIS_REQUIRED,
    PHASE3_ANALYSIS_REQUIRED,
    TRAJECTORY_ANALYSIS_REQUIRED,
    run_phase1_analysis,
    run_phase3_analysis,
    run_trajectory_analysis,
)
from .anchors import ANCHOR_REQUIRED, ensure_anchor_path
from .artifacts import UnitStore, freeze_json
from .assets import ASSET_REQUIRED, ensure_replica_assets
from .audit import (
    AUDIT_REQUIRED,
    VALIDATION_REQUIRED,
    run_existing_artifact_audit,
    run_gauge_dense_validation,
)
from .benchmark import (
    LOCAL_BENCHMARK_REQUIRED,
    PROJECTION_REQUIRED,
    benchmark_condition,
    run_benchmark_trajectory,
    run_compute_projection,
    run_local_benchmark,
)
from .config import Plan12Study
from .local_response import (
    BRANCH_REQUIRED,
    REFERENCE_REQUIRED,
    TARGET_REQUIRED,
    LocalCondition,
    local_conditions,
    run_anchor_reference,
    run_local_branch,
    run_local_target,
)
from .mp_calibration import CALIBRATION_REQUIRED, run_calibration
from .refresh_notebook import DEFAULT_NOTEBOOK, refresh
from .trajectory import (
    TRAJECTORY_REQUIRED,
    TrajectoryCondition,
    phase2_conditions,
    run_trajectory,
)


DEFAULT_ROOT = Path("cache/mnist_experiment/rotated_mnist/plan12")
DEFAULT_DATA = Path("cache/mnist_experiment/datasets")
PHASE_ORDER = ("phase0", "phase1", "phase2", "phase3", "phase4")


@dataclasses.dataclass(frozen=True)
class WorkItem:
    action: str
    unit: dict[str, Any]
    required: tuple[str, ...]
    parameters: dict[str, Any] = dataclasses.field(default_factory=dict)

    def mapping(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "unit": self.unit,
            "required": list(self.required),
            "parameters": self.parameters,
        }


def _historical_roots(repo_root: Path) -> tuple[Path, ...]:
    base = repo_root / "cache/mnist_experiment/rotated_mnist"
    return (
        base / "plan11/development",
        base / "plan11_followup025_v1/followup025_v1",
    )


def _asset_item(store: UnitStore, phase: str, index: int) -> WorkItem:
    return WorkItem("assets", store.unit(phase, "assets", index), ASSET_REQUIRED, {"phase": phase, "index": index})


def _trajectory_item(
    store: UnitStore,
    phase: str,
    index: int,
    schedule: str,
    condition: TrajectoryCondition,
) -> WorkItem:
    unit = store.unit(
        phase,
        "trajectory",
        index,
        schedule=schedule,
        condition=condition.name,
        detail=condition.mapping(),
    )
    return WorkItem(
        "trajectory",
        unit,
        TRAJECTORY_REQUIRED,
        {"phase": phase, "index": index, "schedule": schedule, "condition": condition.mapping()},
    )


def _phase0_items(store: UnitStore) -> list[WorkItem]:
    roots = _historical_roots(store.repo_root)
    maximum = 4 if store.study.smoke else 48
    audit_unit = store.unit(
        "phase0",
        "existing_audit",
        1,
        detail={"roots": [str(path) for path in roots], "maximum_spectra": maximum},
    )
    validation_unit = store.unit("phase0", "gauge_dense_validation", 1)
    local_unit = store.unit(
        "phase0",
        "local_benchmark",
        1,
        detail={"conditions": [item.mapping() for item in local_conditions(store.study.ridge_scale_ratios)]},
    )
    projection_unit = store.unit("phase0", "compute_projection", 1)
    return [
        WorkItem("existing_audit", audit_unit, AUDIT_REQUIRED, {"roots": [str(path) for path in roots], "maximum_spectra": maximum}),
        _asset_item(store, "phase0", 1),
        WorkItem("gauge_validation", validation_unit, VALIDATION_REQUIRED),
        _asset_item(store, "phase0_benchmark", 1),
        _trajectory_item(store, "phase0_benchmark", 1, "linear", benchmark_condition()),
        WorkItem("local_benchmark", local_unit, LOCAL_BENCHMARK_REQUIRED),
        WorkItem("compute_projection", projection_unit, PROJECTION_REQUIRED),
    ]


def _phase1_items(store: UnitStore) -> list[WorkItem]:
    items = [_asset_item(store, "phase1", 1)]
    for schedule in ("linear", "sigmoid"):
        items.append(
            WorkItem(
                "anchors",
                store.unit("phase1", "anchors", 1, schedule=schedule),
                ANCHOR_REQUIRED,
                {"schedule": schedule},
            )
        )
    conditions = local_conditions(store.study.ridge_scale_ratios)
    anchor_count = 2 * store.study.anchors_per_schedule
    for anchor_index in range(1, anchor_count + 1):
        items.append(
            WorkItem(
                "reference",
                store.unit("phase1", "reference", anchor_index),
                REFERENCE_REQUIRED,
                {"anchor_index": anchor_index},
            )
        )
        for condition in conditions:
            items.append(
                WorkItem(
                    "target",
                    store.unit(
                        "phase1",
                        "target",
                        anchor_index,
                        condition=condition.name,
                        detail=condition.mapping(),
                    ),
                    TARGET_REQUIRED,
                    {"anchor_index": anchor_index, "condition": condition.mapping()},
                )
            )
        for batch_index in range(1, store.study.local_batches_per_anchor + 1):
            unit_index = (anchor_index - 1) * store.study.local_batches_per_anchor + batch_index
            for condition in conditions:
                items.append(
                    WorkItem(
                        "branch",
                        store.unit(
                            "phase1",
                            "branch",
                            unit_index,
                            condition=condition.name,
                            detail={"anchor_index": anchor_index, "batch_index": batch_index, **condition.mapping()},
                        ),
                        BRANCH_REQUIRED,
                        {"anchor_index": anchor_index, "batch_index": batch_index, "condition": condition.mapping()},
                    )
                )
    items.append(WorkItem("phase1_analysis", store.unit("phase1", "analysis", 1), PHASE1_ANALYSIS_REQUIRED))
    return items


def _phase1_selection(store: UnitStore) -> dict[str, Any]:
    path = store.completed(store.unit("phase1", "analysis", 1), PHASE1_ANALYSIS_REQUIRED)
    if path is None:
        raise RuntimeError("Phase 1 selection requested before analysis completed")
    return _read_json(path / "selection.json")


def _trajectory_phase_items(
    store: UnitStore,
    phase: str,
    conditions: tuple[TrajectoryCondition, ...],
    replicas: int,
) -> list[WorkItem]:
    items: list[WorkItem] = []
    for index in range(1, replicas + 1):
        items.append(_asset_item(store, phase, index))
        for schedule in ("linear", "sigmoid"):
            for condition in conditions:
                items.append(_trajectory_item(store, phase, index, schedule, condition))
    analysis_unit = store.unit(
        phase,
        "analysis",
        1,
        detail={"conditions": [item.mapping() for item in conditions], "replicas": replicas},
    )
    items.append(
        WorkItem(
            f"{phase}_analysis",
            analysis_unit,
            TRAJECTORY_ANALYSIS_REQUIRED,
            {"phase": phase, "conditions": [item.mapping() for item in conditions], "replicas": replicas},
        )
    )
    return items


def _phase2_items(store: UnitStore) -> list[WorkItem]:
    selection = _phase1_selection(store)
    conditions = phase2_conditions(
        float(selection["isotropic"]["selected_ratio"]),
        float(selection["tail"]["selected_ratio"]),
    )
    return _trajectory_phase_items(store, "phase2", conditions, store.study.phase2_replicas)


def _phase3_items(store: UnitStore) -> list[WorkItem]:
    items = [
        WorkItem(
            "calibration",
            store.unit("phase3", "calibration", anchor_index),
            CALIBRATION_REQUIRED,
            {"anchor_index": anchor_index},
        )
        for anchor_index in range(1, store.study.phase3_checkpoints + 1)
    ]
    items.append(WorkItem("phase3_analysis", store.unit("phase3", "analysis", 1), PHASE3_ANALYSIS_REQUIRED))
    return items


def _phase4_items(store: UnitStore) -> list[WorkItem]:
    local = _phase1_selection(store)
    phase3_path = store.completed(store.unit("phase3", "analysis", 1), PHASE3_ANALYSIS_REQUIRED)
    if phase3_path is None:
        raise RuntimeError("Phase 4 requested before Phase 3 analysis completed")
    spectral = _read_json(phase3_path / "selection.json")
    conditions = (
        TrajectoryCondition("gauge_no_ridge", chart=True),
        TrajectoryCondition(
            "isotropic_ridge",
            chart=True,
            ridge_geometry="isotropic",
            ridge_ratio=float(local["isotropic"]["selected_ratio"]),
        ),
        TrajectoryCondition(
            "tail_ridge",
            chart=True,
            ridge_geometry="tail",
            ridge_ratio=float(local["tail"]["selected_ratio"]),
        ),
        TrajectoryCondition(
            "spectral_selector",
            chart=True,
            ridge_geometry="isotropic",
            ridge_ratio=float(spectral["selected_scale_ratio"]),
        ),
    )
    return _trajectory_phase_items(store, "phase4", conditions, store.study.phase4_replicas)


def _items_for_phase(store: UnitStore, phase: str) -> list[WorkItem]:
    return {
        "phase0": _phase0_items,
        "phase1": _phase1_items,
        "phase2": _phase2_items,
        "phase3": _phase3_items,
        "phase4": _phase4_items,
    }[phase](store)


def _freeze_ledger(store: UnitStore, phase: str, items: list[WorkItem], *, resume: bool) -> None:
    freeze_json(
        store.root / "ledgers" / f"{phase}.json",
        {
            "phase": phase,
            "study": store.study.mapping(),
            "study_hash": store.study.config_hash,
            "source_hashes": store.sources,
            "items": [item.mapping() for item in items],
        },
        resume=resume,
    )


def _execute_item(store: UnitStore, item: WorkItem, data_root: Path) -> Path:
    p = item.parameters
    if item.action == "assets":
        return ensure_replica_assets(store, p["phase"], p["index"], data_root=data_root, resume=True)
    if item.action == "existing_audit":
        return run_existing_artifact_audit(
            store,
            tuple(Path(path) for path in p["roots"]),
            resume=True,
            maximum_spectra=p["maximum_spectra"],
        )
    if item.action == "gauge_validation":
        return run_gauge_dense_validation(store, data_root=data_root, resume=True)
    if item.action == "local_benchmark":
        return run_local_benchmark(store, data_root=data_root, resume=True)
    if item.action == "compute_projection":
        return run_compute_projection(store, resume=True)
    if item.action == "anchors":
        return ensure_anchor_path(store, p["schedule"], data_root=data_root, resume=True)
    if item.action == "reference":
        return run_anchor_reference(store, p["anchor_index"], data_root=data_root, resume=True)
    if item.action == "target":
        return run_local_target(
            store,
            p["anchor_index"],
            LocalCondition(**p["condition"]),
            data_root=data_root,
            resume=True,
        )
    if item.action == "branch":
        return run_local_branch(
            store,
            p["anchor_index"],
            p["batch_index"],
            LocalCondition(**p["condition"]),
            data_root=data_root,
            resume=True,
        )
    if item.action == "phase1_analysis":
        return run_phase1_analysis(store, resume=True)
    if item.action == "trajectory":
        return run_trajectory(
            store,
            p["phase"],
            p["index"],
            p["schedule"],
            TrajectoryCondition(**p["condition"]),
            data_root=data_root,
            resume=True,
        )
    if item.action in {"phase2_analysis", "phase4_analysis"}:
        return run_trajectory_analysis(
            store,
            p["phase"],
            tuple(TrajectoryCondition(**value) for value in p["conditions"]),
            p["replicas"],
            resume=True,
        )
    if item.action == "calibration":
        return run_calibration(store, p["anchor_index"], data_root=data_root, resume=True)
    if item.action == "phase3_analysis":
        return run_phase3_analysis(store, resume=True)
    raise ValueError(f"unknown Plan 12 action: {item.action}")


DEFAULT_SECONDS = {
    "assets": 30.0,
    "existing_audit": 30.0,
    "gauge_validation": 60.0,
    "trajectory": 90.0,
    "local_benchmark": 60.0,
    "compute_projection": 2.0,
    "anchors": 180.0,
    "reference": 45.0,
    "target": 20.0,
    "branch": 4.0,
    "phase1_analysis": 30.0,
    "phase2_analysis": 60.0,
    "calibration": 1800.0,
    "phase3_analysis": 10.0,
    "phase4_analysis": 60.0,
}


def _events(store: UnitStore) -> list[dict[str, Any]]:
    path = store.root / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _estimate_seconds(store: UnitStore, action: str) -> float:
    samples = [float(row["elapsed_seconds"]) for row in _events(store) if row.get("action") == action and row.get("status") == "completed"]
    if samples:
        return statistics.median(samples[-32:])
    if store.study.smoke and action == "calibration":
        return 30.0
    return DEFAULT_SECONDS[action]


def _append_event(store: UnitStore, value: dict[str, Any]) -> None:
    path = store.root / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _safe_refresh(store: UnitStore, notebook: Path) -> None:
    last_error: BaseException | None = None
    for attempt in range(2):
        try:
            refresh(store, notebook)
            return
        except BaseException as error:
            last_error = error
            _write_json(
                store.root / "reports" / "notebook_refresh_failure.json",
                {"attempt": attempt + 1, "error_type": type(error).__name__, "error": str(error), "time": time.time()},
            )
    assert last_error is not None
    print(json.dumps({"status": "notebook_refresh_failed", "error": str(last_error)}), flush=True)


def execute(
    store: UnitStore,
    *,
    data_root: Path,
    notebook: Path,
    resume: bool,
    max_units: int | None,
    max_wall_seconds: float | None,
    through_phase: str,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = None if max_wall_seconds is None else started + max_wall_seconds
    new_units = 0
    paused_reason = None
    through_index = PHASE_ORDER.index(through_phase)
    for phase in PHASE_ORDER[: through_index + 1]:
        items = _items_for_phase(store, phase)
        ledger_exists = (store.root / "ledgers" / f"{phase}.json").exists()
        _freeze_ledger(store, phase, items, resume=resume or ledger_exists)
        _safe_refresh(store, notebook)
        for ordinal, item in enumerate(items, 1):
            if store.completed(item.unit, item.required) is not None:
                continue
            if max_units is not None and new_units >= max_units:
                paused_reason = "max_units"
                break
            estimate = _estimate_seconds(store, item.action)
            if deadline is not None and time.monotonic() + 1.25 * estimate + 30.0 > deadline:
                paused_reason = "max_wall_seconds"
                break
            unit_started = time.perf_counter()
            try:
                path = _execute_item(store, item, data_root)
            except BaseException as error:
                store.record_failure(item.unit, error)
                _append_event(
                    store,
                    {"status": "failed", "phase": phase, "action": item.action, "ordinal": ordinal, "error": f"{type(error).__name__}: {error}"},
                )
                _safe_refresh(store, notebook)
                raise
            elapsed = time.perf_counter() - unit_started
            new_units += 1
            _append_event(
                store,
                {
                    "status": "completed",
                    "phase": phase,
                    "action": item.action,
                    "ordinal": ordinal,
                    "planned": len(items),
                    "elapsed_seconds": elapsed,
                    "path": str(path),
                },
            )
            print(
                json.dumps(
                    {"status": "completed", "phase": phase, "action": item.action, "ordinal": ordinal, "planned": len(items), "elapsed_seconds": elapsed, "new_units": new_units}
                ),
                flush=True,
            )
            if new_units % 16 == 0:
                _safe_refresh(store, notebook)
        _safe_refresh(store, notebook)
        if paused_reason is not None:
            break
    elapsed = time.monotonic() - started
    return {
        "status": "paused" if paused_reason else "complete",
        "paused_reason": paused_reason,
        "new_units": new_units,
        "elapsed_seconds": elapsed,
        "through_phase": through_phase,
        "resume_command": (
            f"/home/evan/.venv/bin/python -m mnist_experiment.rotated_mnist.plan12.orchestrate "
            f"--config {store.root / 'study-config.json'} --output-root {store.root} --resume"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--through-phase", choices=PHASE_ORDER, default="phase4")
    args = parser.parse_args()
    if args.max_units is not None and args.max_units <= 0:
        parser.error("--max-units must be positive")
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")
    study = Plan12Study.from_path(args.config)
    repo_root = Path(__file__).parents[3]
    store = UnitStore(args.output_root, study, repo_root)
    store.root.mkdir(parents=True, exist_ok=True)
    frozen_config = store.root / "study-config.json"
    freeze_json(frozen_config, _read_json(args.config), resume=args.resume)
    with (store.root / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = execute(
            store,
            data_root=args.data_root,
            notebook=args.notebook,
            resume=args.resume,
            max_units=args.max_units,
            max_wall_seconds=args.max_wall_seconds,
            through_phase=args.through_phase,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
