"""Post hoc MP q=.01 trajectory probe for Plan 12 Phase 5."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ..artifacts import _read_json, _write_json
from .artifacts import UnitStore, canonical_hash, file_hash, freeze_json
from .config import Plan12Study
from .trajectory import TRAJECTORY_REQUIRED, TrajectoryCondition, run_trajectory


DEFAULT_LEGACY_ROOT = Path("cache/mnist_experiment/rotated_mnist/plan12")
DEFAULT_OUTPUT_ROOT = Path("cache/mnist_experiment/rotated_mnist/plan12_mp_q01_v1")
DEFAULT_DATA_ROOT = Path("cache/mnist_experiment/datasets")
DEFAULT_NOTEBOOK = Path("mnist_experiment/rotated_mnist/ridge_estimator_health.ipynb")
DEFAULT_BASE_PROGRESS = DEFAULT_LEGACY_ROOT / "reports/progress.json"
PROBE_REPLICAS = 16
MP_Q01_RATIO = 11.818529434984821
MP_Q01_CONDITION = TrajectoryCondition(
    "mp_q01_isotropic",
    chart=True,
    ridge_geometry="isotropic",
    ridge_ratio=MP_Q01_RATIO,
)
CONTROL_CONDITIONS = ("gauge_no_ridge", "spectral_selector")
ANALYSIS_REQUIRED = ("summary.json", "checks.json")
T_CRITICAL_95_DF15 = 2.131449545559323


@dataclasses.dataclass(frozen=True)
class ExternalReference:
    action: str
    index: int
    schedule: str | None
    condition: str | None
    unit: dict[str, Any]
    path: Path
    required: tuple[str, ...]

    def descriptor(self, repo_root: Path) -> dict[str, Any]:
        resolved = self.path.resolve()
        try:
            relative_path = resolved.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            relative_path = resolved.as_posix()
        return {
            "action": self.action,
            "index": self.index,
            "schedule": self.schedule,
            "condition": self.condition,
            "run_id": self.path.name,
            "relative_path": relative_path,
            "config_hash": canonical_hash(self.unit),
            "integrity_hash": file_hash(self.path / "integrity.json"),
            "artifact_hashes": _read_json(self.path / "integrity.json"),
        }


@dataclasses.dataclass(frozen=True)
class ProbeItem:
    action: str
    unit: dict[str, Any]
    required: tuple[str, ...]
    index: int | None = None
    schedule: str | None = None

    def mapping(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "unit": self.unit,
            "required": list(self.required),
            "index": self.index,
            "schedule": self.schedule,
        }


def _phase4_items(legacy_root: Path) -> list[dict[str, Any]]:
    ledger = _read_json(legacy_root / "ledgers/phase4.json")
    if ledger.get("phase") != "phase4":
        raise RuntimeError("Plan 12 Phase 4 ledger has the wrong phase")
    return ledger["items"]


def _find_legacy_item(
    items: list[dict[str, Any]],
    *,
    action: str,
    index: int,
    schedule: str | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    matches = []
    for item in items:
        unit = item["unit"]
        if item["action"] != action or int(unit["index"]) != index:
            continue
        if schedule is not None and unit.get("schedule") != schedule:
            continue
        if condition is not None and unit.get("condition") != condition:
            continue
        matches.append(item)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Phase 4 reference for {action}/{index}/{schedule}/{condition}; "
            f"found {len(matches)}"
        )
    return matches[0]


def phase4_references(
    legacy_root: Path,
    study: Plan12Study,
    repo_root: Path,
    *,
    replicas: int = PROBE_REPLICAS,
) -> dict[tuple[str, int, str | None, str | None], ExternalReference]:
    ledger = _read_json(legacy_root / "ledgers/phase4.json")
    if ledger.get("study_hash") != study.config_hash:
        raise RuntimeError("Phase 4 study hash differs from the q=.01 probe")
    items = ledger["items"]
    legacy_store = UnitStore(legacy_root, study, repo_root)
    references: dict[tuple[str, int, str | None, str | None], ExternalReference] = {}
    specifications = []
    for index in range(1, replicas + 1):
        specifications.append(("assets", index, None, None))
        for schedule in ("linear", "sigmoid"):
            for condition in CONTROL_CONDITIONS:
                specifications.append(("trajectory", index, schedule, condition))
    for action, index, schedule, condition in specifications:
        item = _find_legacy_item(
            items,
            action=action,
            index=index,
            schedule=schedule,
            condition=condition,
        )
        unit = item["unit"]
        required = tuple(item["required"])
        path = legacy_store.completed(unit, required)
        if path is None:
            raise RuntimeError(f"required completed Phase 4 artifact is missing: {unit}")
        key = (action, index, schedule, condition)
        references[key] = ExternalReference(action, index, schedule, condition, unit, path, required)
    return references


def phase3_q01_ratios(
    legacy_root: Path,
    study: Plan12Study,
    repo_root: Path,
) -> list[float]:
    ledger = _read_json(legacy_root / "ledgers/phase3.json")
    if ledger.get("study_hash") != study.config_hash:
        raise RuntimeError("Phase 3 study hash differs from the q=.01 probe")
    legacy_store = UnitStore(legacy_root, study, repo_root)
    ratios = []
    for index in range(1, 9):
        item = _find_legacy_item(ledger["items"], action="calibration", index=index)
        path = legacy_store.completed(item["unit"], tuple(item["required"]))
        if path is None:
            raise RuntimeError(f"required completed Phase 3 calibration is missing: {index}")
        calibration = torch.load(
            path / "calibration.pt", map_location="cpu", weights_only=True
        )
        summary = _read_json(path / "summary.json")
        ratios.append(
            normalized_higher_quantile(
                calibration["full_maxima"],
                float(summary["mean_eigenvalue"]),
                quantile=0.01,
            )
        )
    median = statistics.median(ratios)
    if not math.isclose(median, MP_Q01_RATIO, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"archived q=.01 ratio {median} differs from frozen contract {MP_Q01_RATIO}"
        )
    return ratios


def normalized_higher_quantile(
    maxima: torch.Tensor,
    mean_eigenvalue: float,
    *,
    quantile: float,
) -> float:
    if maxima.ndim != 1 or maxima.numel() == 0:
        raise ValueError("bootstrap maxima must be a nonempty vector")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must lie in [0, 1]")
    if not math.isfinite(mean_eigenvalue) or mean_eigenvalue <= 0:
        raise ValueError("mean eigenvalue must be finite and positive")
    edge = torch.quantile(
        maxima.to(torch.float64),
        quantile,
        interpolation="higher",
    )
    return float(edge) / mean_eigenvalue


def trajectory_detail(asset: ExternalReference, repo_root: Path) -> dict[str, Any]:
    return {
        **MP_Q01_CONDITION.mapping(),
        "mp_quantile": 0.01,
        "quantile_method": "higher",
        "checkpoint_aggregation": "median_of_eight_normalized_checkpoint_quantiles",
        "protocol_phase": "phase4",
        "numerical_seed_phase": "phase4",
        "numerical_seed_condition": "spectral_selector",
        "external_asset": asset.descriptor(repo_root),
    }


def probe_items(
    store: UnitStore,
    references: dict[tuple[str, int, str | None, str | None], ExternalReference],
    *,
    legacy_root: Path = DEFAULT_LEGACY_ROOT,
    replicas: int = PROBE_REPLICAS,
) -> list[ProbeItem]:
    items: list[ProbeItem] = []
    for index in range(1, replicas + 1):
        asset = references[("assets", index, None, None)]
        detail = trajectory_detail(asset, store.repo_root)
        for schedule in ("linear", "sigmoid"):
            unit = store.unit(
                "phase5",
                "trajectory",
                index,
                schedule=schedule,
                condition=MP_Q01_CONDITION.name,
                detail=detail,
            )
            items.append(ProbeItem("trajectory", unit, TRAJECTORY_REQUIRED, index, schedule))
    analysis_detail = {
        "replicas": replicas,
        "condition": MP_Q01_CONDITION.mapping(),
        "controls": list(CONTROL_CONDITIONS),
        "phase4_ledger_hash": file_hash(legacy_root / "ledgers/phase4.json"),
    }
    items.append(
        ProbeItem(
            "analysis",
            store.unit("phase5", "analysis", 1, detail=analysis_detail),
            ANALYSIS_REQUIRED,
        )
    )
    return items


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot summarize an empty metric")
    return statistics.fmean(values)


def _effect(values: list[float]) -> dict[str, Any]:
    mean = _mean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = standard_deviation / len(values) ** 0.5
    critical = T_CRITICAL_95_DF15 if len(values) == PROBE_REPLICAS else 1.96
    return {
        "mean": mean,
        "median": statistics.median(values),
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
        "replicas": len(values),
        "favorable_count": sum(value > 0 for value in values),
    }


def _mean_where(
    rows: list[dict[str, Any]],
    key: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> float:
    return _mean([float(row[key]) for row in rows if predicate(row)])


SEGMENTS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "near_upright_0_to_5_degrees": lambda row: float(row["angle_degrees"]) <= 5.0,
    "far_rotation_25_to_30_degrees": lambda row: float(row["angle_degrees"]) >= 25.0,
    "first_ascent": lambda row: int(row["leg_id"]) == 0,
    "return_leg": lambda row: int(row["leg_id"]) == 1,
    "second_ascent": lambda row: int(row["leg_id"]) == 2,
}


def _trajectory_path(store: UnitStore, item: ProbeItem) -> Path:
    path = store.completed(item.unit, item.required)
    if path is None:
        raise RuntimeError(f"missing completed q=.01 trajectory: {item.unit}")
    return path


def run_analysis(
    store: UnitStore,
    items: list[ProbeItem],
    references: dict[tuple[str, int, str | None, str | None], ExternalReference],
    *,
    resume: bool,
    replicas: int = PROBE_REPLICAS,
) -> Path:
    item = next(value for value in items if value.action == "analysis")
    session = store.begin(item.unit, item.required, resume=resume)
    if session is None:
        completed = store.completed(item.unit, item.required)
        assert completed is not None
        return completed
    trajectory_items = {
        (value.index, value.schedule): value
        for value in items
        if value.action == "trajectory"
    }
    result: dict[str, Any] = {
        "phase": "phase5",
        "status": "complete_exploratory_probe",
        "replicas_per_schedule": replicas,
        "mp_quantile": 0.01,
        "quantile_method": "higher",
        "ridge_ratio": MP_Q01_RATIO,
        "condition": MP_Q01_CONDITION.mapping(),
        "controls": list(CONTROL_CONDITIONS),
        "schedule_results": [],
        "interpretation_contract": (
            "Post hoc descriptive probe. Confidence intervals describe paired uncertainty "
            "at the frozen 16-replica size and are not selector-calibration or confirmatory claims."
        ),
    }
    for schedule in ("linear", "sigmoid"):
        q_summaries: list[dict[str, Any]] = []
        q_metrics: list[list[dict[str, Any]]] = []
        q_paths: list[Path] = []
        controls: dict[str, dict[str, Any]] = {
            name: {"summaries": [], "metrics": [], "paths": []}
            for name in CONTROL_CONDITIONS
        }
        for index in range(1, replicas + 1):
            q_path = _trajectory_path(store, trajectory_items[(index, schedule)])
            q_paths.append(q_path)
            q_summaries.append(_read_json(q_path / "summary.json"))
            q_metrics.append(_read_json(q_path / "metrics.json"))
            for name in CONTROL_CONDITIONS:
                control_path = references[("trajectory", index, schedule, name)].path
                controls[name]["paths"].append(control_path)
                controls[name]["summaries"].append(_read_json(control_path / "summary.json"))
                controls[name]["metrics"].append(_read_json(control_path / "metrics.json"))

        q_movement = [
            float(
                torch.load(path / "trajectory.pt", map_location="cpu", weights_only=True)[
                    "displacements"
                ]
                .to(torch.float64)
                .square()
                .sum()
            )
            for path in q_paths
        ]
        comparisons = []
        for name in CONTROL_CONDITIONS:
            c_summaries = controls[name]["summaries"]
            c_metrics = controls[name]["metrics"]
            c_paths = controls[name]["paths"]
            c_movement = [
                float(
                    torch.load(path / "trajectory.pt", map_location="cpu", weights_only=True)[
                        "displacements"
                    ]
                    .to(torch.float64)
                    .square()
                    .sum()
                )
                for path in c_paths
            ]
            segment_results = {}
            for segment, predicate in SEGMENTS.items():
                nll = []
                accuracy = []
                for index in range(replicas):
                    nll.append(
                        _mean_where(c_metrics[index], "current_nll", predicate)
                        - _mean_where(q_metrics[index], "current_nll", predicate)
                    )
                    accuracy.append(
                        _mean_where(q_metrics[index], "current_accuracy", predicate)
                        - _mean_where(c_metrics[index], "current_accuracy", predicate)
                    )
                segment_results[segment] = {
                    "nll_gain": _effect(nll),
                    "accuracy_gain": _effect(accuracy),
                }
            comparisons.append(
                {
                    "control": name,
                    "nll_auc_gain": _effect(
                        [
                            float(c_summaries[i]["current_nll_auc"])
                            - float(q_summaries[i]["current_nll_auc"])
                            for i in range(replicas)
                        ]
                    ),
                    "accuracy_auc_gain": _effect(
                        [
                            float(q_summaries[i]["current_accuracy_auc"])
                            - float(c_summaries[i]["current_accuracy_auc"])
                            for i in range(replicas)
                        ]
                    ),
                    "brier_auc_gain": _effect(
                        [
                            float(c_summaries[i]["current_brier_auc"])
                            - float(q_summaries[i]["current_brier_auc"])
                            for i in range(replicas)
                        ]
                    ),
                    "worst_panel_nll_auc_gain": _effect(
                        [
                            float(c_summaries[i]["worst_panel_nll_auc"])
                            - float(q_summaries[i]["worst_panel_nll_auc"])
                            for i in range(replicas)
                        ]
                    ),
                    "cumulative_squared_displacement_ratio": _effect(
                        [q_movement[i] / c_movement[i] for i in range(replicas)]
                    ),
                    "segments": segment_results,
                }
            )

        proposal_rows = [
            row["proposal"]
            for replica in q_metrics
            for row in replica
            if row["proposal"] is not None
        ]
        fisher_rows = [
            row["fisher_update"]
            for replica in q_metrics
            for row in replica
            if row["fisher_update"] is not None
        ]
        ridge_rows = [
            row["ridge"]
            for replica in q_metrics
            for row in replica
            if row["ridge"] is not None
        ]
        result["schedule_results"].append(
            {
                "schedule": schedule,
                "q01_mean_current_nll_auc": _mean(
                    [float(value["current_nll_auc"]) for value in q_summaries]
                ),
                "q01_mean_current_accuracy_auc": _mean(
                    [float(value["current_accuracy_auc"]) for value in q_summaries]
                ),
                "q01_mean_cumulative_squared_displacement": _mean(q_movement),
                "optimizer_health": {
                    "mean_iterations": _mean(
                        [float(value["optimizer_iterations"]) for value in proposal_rows]
                    ),
                    "max_iteration_fraction": _mean(
                        [float(value["stopping_reason"] == "lbfgs_max_iterations") for value in proposal_rows]
                    ),
                    "mean_relative_final_gradient_norm": _mean(
                        [float(value["relative_final_gradient_norm"]) for value in proposal_rows]
                    ),
                    "mean_backtracking_rejections": _mean(
                        [float(value["backtracking_rejections"]) for value in proposal_rows]
                    ),
                },
                "fisher_health": {
                    "mean_compression_relative_frobenius_error": _mean(
                        [float(value["compression_relative_frobenius_error"]) for value in fisher_rows]
                    ),
                    "mean_archive_trace": _mean(
                        [float(row["archive_trace"]) for replica in q_metrics for row in replica]
                    ),
                    "mean_kappa": _mean([float(value["kappa"]) for value in ridge_rows]),
                    "mean_tau": _mean([float(value["tau"]) for value in ridge_rows]),
                },
                "comparisons": comparisons,
            }
        )
    checks = {
        "all_schedules_complete": len(result["schedule_results"]) == 2,
        "replicas_per_schedule": replicas,
        "ratio_matches_frozen_contract": MP_Q01_RATIO == MP_Q01_CONDITION.ridge_ratio,
        "post_hoc_label_present": "Post hoc" in result["interpretation_contract"],
    }
    if not (
        checks["all_schedules_complete"]
        and checks["replicas_per_schedule"] == replicas
        and checks["ratio_matches_frozen_contract"]
        and checks["post_hoc_label_present"]
    ):
        raise RuntimeError(f"Phase 5 analysis checks failed: {checks}")
    session.write_json("summary.json", result)
    session.write_json("checks.json", checks)
    return store.finish(session, item.required)


def _append_event(root: Path, value: dict[str, Any]) -> None:
    path = root / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _completed_event_times(root: Path, action: str) -> list[float]:
    path = root / "events.jsonl"
    if not path.is_file():
        return []
    return [
        float(row["elapsed_seconds"])
        for row in (_read_json_line(line) for line in path.read_text(encoding="utf-8").splitlines())
        if row.get("status") == "completed" and row.get("action") == action
    ]


def _read_json_line(line: str) -> dict[str, Any]:
    return json.loads(line)


def _safe_refresh(
    store: UnitStore,
    items: list[ProbeItem],
    references: dict[tuple[str, int, str | None, str | None], ExternalReference],
    *,
    base_progress: Path,
    notebook: Path,
) -> None:
    from .q01_notebook import refresh_q01_notebook

    try:
        refresh_q01_notebook(store, items, references, base_progress, notebook)
    except BaseException as error:
        _write_json(
            store.root / "reports/notebook_refresh_failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "recorded_at_unix": time.time(),
            },
        )
        print(json.dumps({"status": "notebook_refresh_failed", "error": str(error)}), flush=True)


def execute(
    store: UnitStore,
    items: list[ProbeItem],
    references: dict[tuple[str, int, str | None, str | None], ExternalReference],
    *,
    legacy_root: Path,
    data_root: Path,
    base_progress: Path,
    notebook: Path,
    resume: bool,
    max_units: int | None,
    max_wall_seconds: float | None,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = None if max_wall_seconds is None else started + max_wall_seconds
    new_units = 0
    paused_reason = None
    _safe_refresh(store, items, references, base_progress=base_progress, notebook=notebook)
    for ordinal, item in enumerate(items, 1):
        if store.completed(item.unit, item.required) is not None:
            continue
        if max_units is not None and new_units >= max_units:
            paused_reason = "max_units"
            break
        samples = _completed_event_times(store.root, item.action)
        estimate = statistics.median(samples[-16:]) if samples else (75.0 if item.action == "trajectory" else 30.0)
        if deadline is not None and time.monotonic() + 1.25 * estimate + 30.0 > deadline:
            paused_reason = "max_wall_seconds"
            break
        unit_started = time.perf_counter()
        try:
            if item.action == "trajectory":
                assert item.index is not None and item.schedule is not None
                asset = references[("assets", item.index, None, None)]
                path = run_trajectory(
                    store,
                    "phase5",
                    item.index,
                    item.schedule,
                    MP_Q01_CONDITION,
                    data_root=data_root,
                    resume=True,
                    asset_path=asset.path,
                    protocol_phase="phase4",
                    numerical_seed_phase="phase4",
                    numerical_seed_condition="spectral_selector",
                    unit_detail=item.unit["detail"],
                )
            elif item.action == "analysis":
                path = run_analysis(store, items, references, resume=True)
            else:
                raise RuntimeError(f"unknown Phase 5 action: {item.action}")
        except BaseException as error:
            store.record_failure(item.unit, error)
            _append_event(
                store.root,
                {
                    "status": "failed",
                    "phase": "phase5",
                    "action": item.action,
                    "ordinal": ordinal,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            _safe_refresh(store, items, references, base_progress=base_progress, notebook=notebook)
            raise
        elapsed = time.perf_counter() - unit_started
        new_units += 1
        _append_event(
            store.root,
            {
                "status": "completed",
                "phase": "phase5",
                "action": item.action,
                "ordinal": ordinal,
                "planned": len(items),
                "elapsed_seconds": elapsed,
                "path": str(path),
            },
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "phase": "phase5",
                    "action": item.action,
                    "ordinal": ordinal,
                    "planned": len(items),
                    "elapsed_seconds": elapsed,
                    "new_units": new_units,
                }
            ),
            flush=True,
        )
        if new_units % 4 == 0:
            _safe_refresh(store, items, references, base_progress=base_progress, notebook=notebook)
    _safe_refresh(store, items, references, base_progress=base_progress, notebook=notebook)
    return {
        "status": "paused" if paused_reason else "complete",
        "paused_reason": paused_reason,
        "new_units": new_units,
        "elapsed_seconds": time.monotonic() - started,
        "resume_command": (
            "/home/evan/.venv/bin/python -m "
            "mnist_experiment.rotated_mnist.plan12.q01_probe --resume"
        ),
        "legacy_root": str(legacy_root),
        "output_root": str(store.root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("mnist_experiment/rotated_mnist/plan12/configs/default.json"),
    )
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-progress", type=Path, default=DEFAULT_BASE_PROGRESS)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
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
    references = phase4_references(args.legacy_root, study, repo_root)
    checkpoint_ratios = phase3_q01_ratios(args.legacy_root, study, repo_root)
    items = probe_items(store, references, legacy_root=args.legacy_root)
    reference_manifest = {
        "phase": "phase5",
        "study_hash": study.config_hash,
        "source_hashes": store.sources,
        "mp_q01_ratio": MP_Q01_RATIO,
        "checkpoint_q01_ratios": checkpoint_ratios,
        "replicas_per_schedule": PROBE_REPLICAS,
        "phase4_ledger_hash": file_hash(args.legacy_root / "ledgers/phase4.json"),
        "external_references": [
            reference.descriptor(repo_root)
            for _, reference in sorted(references.items(), key=lambda value: str(value[0]))
        ],
        "items": [item.mapping() for item in items],
    }
    freeze_json(store.root / "ledgers/phase5_probe16.json", reference_manifest, resume=args.resume)
    with (store.root / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = execute(
            store,
            items,
            references,
            legacy_root=args.legacy_root,
            data_root=args.data_root,
            base_progress=args.base_progress,
            notebook=args.notebook,
            resume=args.resume,
            max_units=args.max_units,
            max_wall_seconds=args.max_wall_seconds,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
