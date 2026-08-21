"""Schema-normalized analysis for the Plan 3 replay-capacity screen."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .plan3 import (
    Plan3Error,
    Plan3Phase2Bundle,
    Plan3Phase4Bundle,
    Plan3Phase5Bundle,
    Plan3Phase6Bundle,
    Plan3Phase7Bundle,
    plan3_phase2_status_rows,
    plan3_phase4_status_rows,
    plan3_phase5_status_rows,
    plan3_phase6_status_rows,
    plan3_phase7_status_rows,
)


PLAN3_PHASE2_ANALYSIS_SCHEMA_VERSION = 1
PLAN3_PHASE4_ANALYSIS_SCHEMA_VERSION = 2
PLAN3_PHASE5_ANALYSIS_SCHEMA_VERSION = 1
PLAN3_PHASE6_ANALYSIS_SCHEMA_VERSION = 2
PLAN3_PHASE7_ANALYSIS_SCHEMA_VERSION = 1
PRIMARY_METRICS = (
    "environment_accuracy",
    "nine_ovr_accuracy",
    "nine_precision",
    "nine_recall",
    "non_nine_accuracy",
)
DIAGNOSTIC_METRICS = (
    "nine_false_positive_rate",
    "nll",
    "brier",
    "expected_calibration_error",
)
ALL_METRICS = (*PRIMARY_METRICS, *DIAGNOSTIC_METRICS)
LOWER_IS_BETTER = frozenset(
    {"nine_false_positive_rate", "nll", "brier", "expected_calibration_error"}
)
CONDITION_ORDER = (
    "current-only",
    "ewc-fixed005-no-lfu",
    "replay-b008",
    "replay-b032",
    "replay-b128",
    "replay-unbounded",
)
PHASE4_CONDITION_ORDER = (
    "ewc-fixed005-no-lfu",
    "replay-b008",
    "replay-memory-matched",
    "replay-selected",
    "hybrid-b008-no-lfu",
    "hybrid-memory-matched-no-lfu",
    "hybrid-selected-no-lfu",
    "replay-unbounded",
)
PHASE4_RESOURCE_METRICS = (
    "total_wall_seconds",
    "trajectory_wall_seconds",
    "learner_optimization_wall_seconds",
    "archive_consolidation_wall_seconds",
    "archive_score_fisher_wall_seconds",
    "archive_fisher_update_wall_seconds",
    "optimizer_iterations",
    "optimizer_function_evaluations",
    "optimizer_event_evaluations",
    "archive_score_gradient_count",
    "hvp_count",
    "logical_persistent_bytes",
    "logical_archive_persistent_bytes",
    "logical_replay_persistent_bytes",
    "physical_index_state_bytes",
    "serialized_archive_state_bytes",
    "serialized_replay_state_bytes",
)
PHASE5_CONDITION_ORDER = (
    "ewc-fixed005-no-lfu",
    "ewc-fixed005-ac-only",
    "ewc-fixed005-full-lfu",
    "hybrid-selected-no-lfu",
    "hybrid-selected-full-lfu",
)
PHASE6_CONDITION_ORDER = (
    "deployment-current-only",
    "deployment-ewc-fixed005",
    "deployment-hybrid-b032-fixed005",
    "deployment-ewc-adaptive-h020",
    "deployment-hybrid-b032-adaptive-h020",
    "deployment-replay-b032",
    "deployment-replay-unbounded",
)
PHASE7_CONDITION_ORDER = (
    "confirm-current-only",
    "confirm-ewc-fixed005",
    "confirm-hybrid-b032-fixed005",
    "confirm-replay-b032",
    "confirm-replay-unbounded",
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Plan3Error(f"could not read analysis dependency {path}: {exc}") from exc


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mean_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "standard_error": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {
            "count": 1,
            "mean": mean,
            "standard_deviation": None,
            "standard_error": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96
    )
    radius = critical * standard_error
    return {
        "count": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
    }


def _normalized_auc(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    points = [
        (float(row["p"]), float(row[metric]))
        for row in rows
        if 0.0 <= float(row["p"]) < 0.5 and row.get(metric) is not None
    ]
    if len(points) < 2:
        raise Plan3Error(f"insufficient points for {metric} AUC")
    if any(right[0] <= left[0] for left, right in zip(points, points[1:])):
        raise Plan3Error("analysis p grid must be strictly increasing")
    area = sum(
        (right_p - left_p) * (left_value + right_value) / 2.0
        for (left_p, left_value), (right_p, right_value) in zip(
            points, points[1:]
        )
    )
    return area / (points[-1][0] - points[0][0])


def _normalize_control_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    if metrics.get("phase8_metric_schema_version") != 8:
        raise Plan3Error("Phase 2 controls require schema-8 classification metrics")
    rows = metrics.get("condition_steps")
    if not isinstance(rows, list) or len(rows) != 100:
        raise Plan3Error("Phase 2 control trajectory must contain 100 rows")
    return [
        {
            "step": int(row["step"]),
            "p": float(row["p"]),
            **{metric: row.get(f"before_{metric}") for metric in ALL_METRICS},
        }
        for row in rows
    ]


def _normalize_replay_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    if metrics.get("plan3_replay_metric_schema_version") != 9:
        raise Plan3Error("Phase 2 replay requires schema-9 classification metrics")
    rows = metrics.get("condition_steps")
    if not isinstance(rows, list) or len(rows) != 100:
        raise Plan3Error("Phase 2 replay trajectory must contain 100 rows")
    normalized = []
    for row in rows:
        classification = row.get("classification")
        if not isinstance(classification, Mapping):
            raise Plan3Error("replay row is missing classification metrics")
        normalized.append(
            {
                "step": int(row["step"]),
                "p": float(row["p"]),
                **{metric: classification.get(metric) for metric in ALL_METRICS},
            }
        )
    return normalized


def _normalize_hybrid_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    if metrics.get("plan3_hybrid_metric_schema_version") not in {10, 11, 12}:
        raise Plan3Error(
            "Plan 3 hybrid requires schema-10, schema-11, or schema-12 metrics"
        )
    rows = metrics.get("condition_steps")
    if not isinstance(rows, list) or len(rows) != 100:
        raise Plan3Error("Phase 4 hybrid trajectory must contain 100 rows")
    normalized = []
    for row in rows:
        classification = row.get("classification")
        if not isinstance(classification, Mapping):
            raise Plan3Error("hybrid row is missing classification metrics")
        normalized.append(
            {
                "step": int(row["step"]),
                "p": float(row["p"]),
                **{metric: classification.get(metric) for metric in ALL_METRICS},
            }
        )
    return normalized


def _control_resource(metrics: Mapping[str, Any], *, uses_ewc: bool) -> dict[str, Any]:
    proposal_rows = [
        row["proposal"]
        for row in metrics["condition_steps"]
        if isinstance(row.get("proposal"), Mapping)
    ]
    function_evaluations = sum(
        int(row["optimizer_function_evaluations"]) for row in proposal_rows
    )
    return {
        "wall_time_comparable": False,
        "total_wall_seconds": None,
        "trajectory_wall_seconds": None,
        "optimization_wall_seconds": None,
        "optimizer_iterations": sum(
            int(row["optimizer_iterations"]) for row in proposal_rows
        ),
        "optimizer_function_evaluations": function_evaluations,
        "optimizer_event_evaluations": 8 * function_evaluations,
        "logical_persistent_bytes": 20_480 if uses_ewc else 0,
        "physical_index_state_bytes": 0,
        "serialized_replay_state_bytes": 0,
    }


def _replay_resource(metrics: Mapping[str, Any]) -> dict[str, Any]:
    ledger = metrics.get("resource_ledger")
    if not isinstance(ledger, Mapping):
        raise Plan3Error("replay artifact is missing its resource ledger")
    operations = ledger.get("operation_totals")
    if not isinstance(operations, Mapping) or "optimization" not in operations:
        raise Plan3Error("replay resource ledger lacks optimization timing")
    return {
        "wall_time_comparable": True,
        "total_wall_seconds": float(ledger["total_wall_seconds"]),
        "trajectory_wall_seconds": float(ledger["trajectory_wall_seconds"]),
        "optimization_wall_seconds": float(operations["optimization"]["wall_seconds"]),
        "optimizer_iterations": int(ledger["optimizer_iterations"]),
        "optimizer_function_evaluations": int(
            ledger["optimizer_function_evaluations"]
        ),
        "optimizer_event_evaluations": int(ledger["optimizer_event_evaluations"]),
        "logical_persistent_bytes": int(
            ledger["logical_replay_persistent_bytes_final"]
        ),
        "physical_index_state_bytes": int(
            ledger["physical_replay_index_state_bytes_final"]
        ),
        "serialized_replay_state_bytes": int(
            ledger["serialized_replay_state_bytes_final"]
        ),
    }


def _phase4_control_resource(metrics: Mapping[str, Any]) -> dict[str, Any]:
    base = _control_resource(metrics, uses_ewc=True)
    return {
        "wall_time_comparable": False,
        "total_wall_seconds": None,
        "trajectory_wall_seconds": None,
        "learner_optimization_wall_seconds": None,
        "archive_consolidation_wall_seconds": None,
        "archive_score_fisher_wall_seconds": None,
        "archive_fisher_update_wall_seconds": None,
        "optimizer_iterations": base["optimizer_iterations"],
        "optimizer_function_evaluations": base[
            "optimizer_function_evaluations"
        ],
        "optimizer_event_evaluations": base["optimizer_event_evaluations"],
        "archive_score_gradient_count": None,
        "hvp_count": None,
        "logical_persistent_bytes": 20_480,
        "logical_archive_persistent_bytes": 20_480,
        "logical_replay_persistent_bytes": 0,
        "physical_index_state_bytes": 0,
        "serialized_archive_state_bytes": None,
        "serialized_replay_state_bytes": 0,
    }


def _phase4_replay_resource(metrics: Mapping[str, Any]) -> dict[str, Any]:
    base = _replay_resource(metrics)
    ledger = metrics["resource_ledger"]
    operations = ledger["operation_totals"]
    return {
        "wall_time_comparable": True,
        "total_wall_seconds": base["total_wall_seconds"],
        "trajectory_wall_seconds": base["trajectory_wall_seconds"],
        "learner_optimization_wall_seconds": float(
            operations["optimization"]["wall_seconds"]
        ),
        "archive_consolidation_wall_seconds": 0.0,
        "archive_score_fisher_wall_seconds": 0.0,
        "archive_fisher_update_wall_seconds": 0.0,
        "optimizer_iterations": base["optimizer_iterations"],
        "optimizer_function_evaluations": base[
            "optimizer_function_evaluations"
        ],
        "optimizer_event_evaluations": base["optimizer_event_evaluations"],
        "archive_score_gradient_count": 0,
        "hvp_count": 0,
        "logical_persistent_bytes": base["logical_persistent_bytes"],
        "logical_archive_persistent_bytes": 0,
        "logical_replay_persistent_bytes": base["logical_persistent_bytes"],
        "physical_index_state_bytes": base["physical_index_state_bytes"],
        "serialized_archive_state_bytes": 0,
        "serialized_replay_state_bytes": base["serialized_replay_state_bytes"],
    }


def _operation_seconds(operations: Mapping[str, Any], name: str) -> float:
    value = operations.get(name)
    return 0.0 if not isinstance(value, Mapping) else float(value["wall_seconds"])


def _phase4_hybrid_resource(metrics: Mapping[str, Any]) -> dict[str, Any]:
    ledger = metrics.get("resource_ledger")
    accounting = metrics.get("archive_accounting")
    if not isinstance(ledger, Mapping) or not isinstance(accounting, Mapping):
        raise Plan3Error("hybrid artifact is missing its resource ledger")
    operations = ledger.get("operation_totals")
    if not isinstance(operations, Mapping):
        raise Plan3Error("hybrid resource ledger lacks operation timing")
    return {
        "wall_time_comparable": True,
        "total_wall_seconds": float(ledger["total_wall_seconds"]),
        "trajectory_wall_seconds": float(ledger["trajectory_wall_seconds"]),
        "learner_optimization_wall_seconds": _operation_seconds(
            operations, "learner_optimization"
        ),
        "archive_consolidation_wall_seconds": _operation_seconds(
            operations, "archive_consolidation"
        ),
        "archive_score_fisher_wall_seconds": _operation_seconds(
            operations, "archive_score_fisher"
        ),
        "archive_fisher_update_wall_seconds": _operation_seconds(
            operations, "archive_fisher_update"
        ),
        "optimizer_iterations": int(ledger["learner_optimizer_iterations"])
        + int(ledger["archive_optimizer_iterations"]),
        "optimizer_function_evaluations": int(
            ledger["learner_optimizer_function_evaluations"]
        )
        + int(ledger["archive_optimizer_function_evaluations"]),
        "optimizer_event_evaluations": int(
            ledger["learner_optimizer_event_evaluations"]
        )
        + int(ledger["archive_optimizer_event_evaluations"]),
        "archive_score_gradient_count": int(accounting["score_gradient_count"]),
        "hvp_count": int(accounting["hvp_count"]),
        "logical_persistent_bytes": int(
            ledger["logical_hybrid_persistent_bytes_final"]
        ),
        "logical_archive_persistent_bytes": int(
            ledger["logical_archive_persistent_bytes_final"]
        ),
        "logical_replay_persistent_bytes": int(
            ledger["logical_replay_persistent_bytes_final"]
        ),
        "physical_index_state_bytes": int(
            ledger["physical_replay_index_state_bytes_final"]
        ),
        "serialized_archive_state_bytes": int(
            ledger["serialized_archive_state_bytes_final"]
        ),
        "serialized_replay_state_bytes": int(
            ledger["serialized_replay_state_bytes_final"]
        ),
    }


def build_phase2_analysis(
    bundle: Plan3Phase2Bundle,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root)
    statuses = plan3_phase2_status_rows(bundle, root)
    incomplete = [row["run_id"] for row in statuses if row["run_state"] != "completed"]
    invalid_controls = [
        row["run_id"]
        for row in statuses
        if row["current_control_state"] != "completed"
        or row["ewc_control_state"] != "completed"
    ]
    if incomplete or invalid_controls:
        raise Plan3Error(
            "Phase 2 analysis requires complete replay and control artifacts: "
            + ", ".join(sorted(set((*incomplete, *invalid_controls))))
        )

    normalized: dict[tuple[int, str], list[dict[str, Any]]] = {}
    resources: dict[tuple[int, str], dict[str, Any]] = {}
    provenance: dict[tuple[int, str], dict[str, Any]] = {}
    for control in bundle.manifest["controls"]:
        path = root / control["cache_root"] / control["run_id"]
        metrics = _read_json(path / "phase8_metrics.json")
        key = (int(control["replica_index"]), str(control["condition"]))
        normalized[key] = _normalize_control_rows(metrics)
        resources[key] = _control_resource(
            metrics, uses_ewc=control["condition"] == "ewc-fixed005-no-lfu"
        )
        provenance[key] = {
            "run_id": control["run_id"],
            "config_hash": control["config_hash"],
            "stream_plan_hash": metrics["stream_plan_hash"],
            "replica_bundle_id": metrics["replica_bundle_id"],
            "initial_parameter_hash": metrics["pairing"]["initial_parameter_hash"],
            "source_schema": 8,
        }
    for entry in bundle.manifest["entries"]:
        path = root / entry["cache_root"] / entry["run_id"]
        metrics = _read_json(path / "plan3_replay_metrics.json")
        key = (int(entry["replica_index"]), str(entry["condition"]))
        if metrics.get("condition", {}).get("replay_capacity") != entry["capacity"]:
            raise Plan3Error(f"replay capacity mismatch for {entry['run_id']}")
        normalized[key] = _normalize_replay_rows(metrics)
        resources[key] = _replay_resource(metrics)
        provenance[key] = {
            "run_id": entry["run_id"],
            "config_hash": entry["config_hash"],
            "stream_plan_hash": metrics["stream_plan_hash"],
            "replica_bundle_id": metrics["replica_bundle_id"],
            "initial_parameter_hash": metrics["initial_parameter_hash"],
            "source_schema": 9,
        }

    replicas = tuple(bundle.manifest["replica_indices"])
    expected_keys = {
        (replica, condition)
        for replica in replicas
        for condition in CONDITION_ORDER
    }
    if set(normalized) != expected_keys:
        raise Plan3Error("Phase 2 analysis matrix is incomplete")
    for replica in replicas:
        replica_provenance = [
            provenance[(replica, condition)] for condition in CONDITION_ORDER
        ]
        if len({row["stream_plan_hash"] for row in replica_provenance}) != 1:
            raise Plan3Error(f"stream pairing failed for replica {replica}")
        if len({row["replica_bundle_id"] for row in replica_provenance}) != 1:
            raise Plan3Error(f"bundle pairing failed for replica {replica}")
        if len({row["initial_parameter_hash"] for row in replica_provenance}) != 1:
            raise Plan3Error(f"initialization pairing failed for replica {replica}")
        p_grids = {
            tuple(row["p"] for row in normalized[(replica, condition)])
            for condition in CONDITION_ORDER
        }
        if len(p_grids) != 1:
            raise Plan3Error(f"p-grid pairing failed for replica {replica}")

    expected_trajectories = []
    for condition in CONDITION_ORDER:
        p_values = [row["p"] for row in normalized[(replicas[0], condition)]]
        for step, p_value in enumerate(p_values):
            row: dict[str, Any] = {
                "condition": condition,
                "step": step,
                "p": p_value,
            }
            for metric in ALL_METRICS:
                values = [
                    normalized[(replica, condition)][step][metric]
                    for replica in replicas
                    if normalized[(replica, condition)][step][metric] is not None
                ]
                row[metric] = _mean_interval([float(value) for value in values])
            expected_trajectories.append(row)

    replica_auc = []
    for replica in replicas:
        for condition in CONDITION_ORDER:
            row = {"replica_index": replica, "condition": condition}
            for metric in ALL_METRICS:
                row[metric] = _normalized_auc(
                    normalized[(replica, condition)], metric
                )
            replica_auc.append(row)
    auc_by_key = {
        (row["replica_index"], row["condition"]): row for row in replica_auc
    }
    condition_auc = []
    for condition in CONDITION_ORDER:
        row = {"condition": condition}
        for metric in ALL_METRICS:
            row[metric] = _mean_interval(
                [auc_by_key[(replica, condition)][metric] for replica in replicas]
            )
            row[metric]["direction"] = (
                "lower" if metric in LOWER_IS_BETTER else "higher"
            )
        condition_auc.append(row)

    comparisons = []
    replay_conditions = CONDITION_ORDER[2:]
    comparison_pairs = [
        *((condition, "current-only") for condition in replay_conditions),
        *((condition, "ewc-fixed005-no-lfu") for condition in replay_conditions),
        ("replay-b032", "replay-b008"),
        ("replay-b128", "replay-b032"),
        ("replay-unbounded", "replay-b128"),
    ]
    for treatment, comparison in comparison_pairs:
        row = {"treatment": treatment, "comparison": comparison}
        for metric in ALL_METRICS:
            differences = [
                auc_by_key[(replica, treatment)][metric]
                - auc_by_key[(replica, comparison)][metric]
                for replica in replicas
            ]
            row[metric] = _mean_interval(differences)
            row[metric]["favorable_sign"] = (
                "negative" if metric in LOWER_IS_BETTER else "positive"
            )
        comparisons.append(row)

    resource_summary = []
    for condition in CONDITION_ORDER:
        rows = [resources[(replica, condition)] for replica in replicas]
        summary: dict[str, Any] = {
            "condition": condition,
            "wall_time_comparable": all(row["wall_time_comparable"] for row in rows),
        }
        for metric in (
            "total_wall_seconds",
            "trajectory_wall_seconds",
            "optimization_wall_seconds",
            "optimizer_iterations",
            "optimizer_function_evaluations",
            "optimizer_event_evaluations",
            "logical_persistent_bytes",
            "physical_index_state_bytes",
            "serialized_replay_state_bytes",
        ):
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            summary[metric] = _mean_interval(values)
        resource_summary.append(summary)

    bundle_sha = hashlib.sha256((bundle.path / "bundle.json").read_bytes()).hexdigest()
    return {
        "schema_version": PLAN3_PHASE2_ANALYSIS_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": bundle_sha,
        "replica_indices": list(replicas),
        "conditions": list(CONDITION_ORDER),
        "analysis_domain": {
            "auc": "normalized trapezoidal area over available 0 <= p < 0.5 points",
            "expected_trajectory": "mean within each exact p over paired replicas",
            "ci95": "two-sided Student-t interval across replicas",
            "primary_metrics": list(PRIMARY_METRICS),
            "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        },
        "pairing_validated": True,
        "provenance": [
            {
                "replica_index": replica,
                "condition": condition,
                **provenance[(replica, condition)],
            }
            for replica in replicas
            for condition in CONDITION_ORDER
        ],
        "expected_trajectories": expected_trajectories,
        "replica_auc": replica_auc,
        "condition_auc": condition_auc,
        "paired_auc_differences": comparisons,
        "resource_summary": resource_summary,
    }


def write_phase2_analysis(
    bundle: Plan3Phase2Bundle,
    repo_root: str | Path,
) -> Path:
    root = Path(repo_root)
    analysis = build_phase2_analysis(bundle, root)
    identity = {
        "schema_version": PLAN3_PHASE2_ANALYSIS_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": analysis["bundle_manifest_sha256"],
        "primary_metrics": list(PRIMARY_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "auc_domain": "0<=p<0.5",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "analysis"
        / f"phase2__{bundle.bundle_id}__{digest[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan3Error(f"incomplete Phase 2 analysis exists: {destination}")
        stored = _read_json(destination / "summary.json")
        if stored != analysis:
            raise Plan3Error("completed Phase 2 analysis has incompatible contents")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=incomplete))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(analysis, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(identity, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def build_phase4_analysis(
    bundle: Plan3Phase4Bundle,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root)
    statuses = plan3_phase4_status_rows(bundle, root)
    incomplete = [row["run_id"] for row in statuses if row["run_state"] != "completed"]
    invalid = [
        row["run_id"]
        for row in statuses
        if row["reused_controls_state"] != "completed"
        or row["archive_source_state"] not in {"completed", "not-applicable"}
    ]
    if incomplete or invalid:
        raise Plan3Error(
            "Phase 4 analysis requires complete treatments and dependencies: "
            + ", ".join(sorted(set((*incomplete, *invalid))))
        )

    normalized: dict[tuple[int, str], list[dict[str, Any]]] = {}
    resources: dict[tuple[int, str], dict[str, Any]] = {}
    provenance: dict[tuple[int, str], dict[str, Any]] = {}
    for control in bundle.manifest["controls"]:
        path = root / control["cache_root"] / control["run_id"]
        key = (int(control["replica_index"]), str(control["condition"]))
        if control["artifact_kind"] == "control":
            metrics = _read_json(path / "phase8_metrics.json")
            normalized[key] = _normalize_control_rows(metrics)
            resources[key] = _phase4_control_resource(metrics)
            initial_parameter_hash = metrics["pairing"]["initial_parameter_hash"]
            source_schema = 8
        elif control["artifact_kind"] == "replay":
            metrics = _read_json(path / "plan3_replay_metrics.json")
            normalized[key] = _normalize_replay_rows(metrics)
            resources[key] = _phase4_replay_resource(metrics)
            initial_parameter_hash = metrics["initial_parameter_hash"]
            source_schema = 9
        else:
            raise Plan3Error("Phase 4 reused artifact kind is unsupported")
        provenance[key] = {
            "run_id": control["run_id"],
            "config_hash": control["config_hash"],
            "stream_plan_hash": metrics["stream_plan_hash"],
            "replica_bundle_id": metrics["replica_bundle_id"],
            "initial_parameter_hash": initial_parameter_hash,
            "source_schema": source_schema,
            "source_condition": control["source_condition"],
        }

    for entry in bundle.manifest["entries"]:
        path = root / entry["cache_root"] / entry["run_id"]
        key = (int(entry["replica_index"]), str(entry["condition"]))
        if entry["runner"] == "replay":
            metrics = _read_json(path / "plan3_replay_metrics.json")
            normalized[key] = _normalize_replay_rows(metrics)
            resources[key] = _phase4_replay_resource(metrics)
            initial_parameter_hash = metrics["initial_parameter_hash"]
            source_schema = 9
        elif entry["runner"] == "hybrid":
            metrics = _read_json(path / "plan3_hybrid_metrics.json")
            normalized[key] = _normalize_hybrid_rows(metrics)
            resources[key] = _phase4_hybrid_resource(metrics)
            initial_parameter_hash = metrics["initial_parameter_hash"]
            source_schema = 10
            accounting = metrics["archive_accounting"]
            expected_archived = 8 * 99 - int(entry["capacity"])
            if accounting["archived_online_events"] != expected_archived:
                raise Plan3Error(
                    f"hybrid archive count mismatch for {entry['run_id']}"
                )
            if any(
                (row.get("identity_audit") or {}).get(
                    "active_archive_overlap_count", 0
                )
                != 0
                for row in metrics["condition_steps"]
            ):
                raise Plan3Error(
                    f"hybrid active/archive overlap for {entry['run_id']}"
                )
            update_rows = [
                row
                for row in metrics["condition_steps"]
                if row.get("learner_proposal") is not None
            ]
            if any(
                row["identity_audit"]["accounted_online_event_count"]
                != 8 * (int(row["step"]) + 1)
                for row in update_rows
            ):
                raise Plan3Error(
                    f"hybrid event accounting is incomplete for {entry['run_id']}"
                )
            if any(
                row["archive_before"] != row["archive_after"]
                for row in update_rows
                if row["replay_after"]["evicted_event_count"] == 0
            ):
                raise Plan3Error(
                    f"no-eviction archive changed for {entry['run_id']}"
                )
            if (
                int(accounting["score_gradient_count"]) != expected_archived
                or int(accounting["hvp_count"]) != 0
            ):
                raise Plan3Error(
                    f"hybrid derivative accounting mismatch for {entry['run_id']}"
                )
        else:
            raise Plan3Error("Phase 4 runner is unsupported")
        if metrics.get("condition", {}).get("replay_capacity") != entry["capacity"]:
            raise Plan3Error(f"Phase 4 capacity mismatch for {entry['run_id']}")
        provenance[key] = {
            "run_id": entry["run_id"],
            "config_hash": entry["config_hash"],
            "stream_plan_hash": metrics["stream_plan_hash"],
            "replica_bundle_id": metrics["replica_bundle_id"],
            "initial_parameter_hash": initial_parameter_hash,
            "source_schema": source_schema,
            "source_condition": None,
        }

    replicas = tuple(int(value) for value in bundle.manifest["replica_indices"])
    conditions = tuple(str(value) for value in bundle.manifest["condition_order"])
    if conditions != PHASE4_CONDITION_ORDER:
        raise Plan3Error("Phase 4 condition order is incompatible")
    expected_keys = {
        (replica, condition) for replica in replicas for condition in conditions
    }
    if set(normalized) != expected_keys:
        raise Plan3Error("Phase 4 analysis matrix is incomplete")
    for replica in replicas:
        replica_provenance = [
            provenance[(replica, condition)] for condition in conditions
        ]
        for field in (
            "stream_plan_hash",
            "replica_bundle_id",
            "initial_parameter_hash",
        ):
            if len({row[field] for row in replica_provenance}) != 1:
                raise Plan3Error(f"Phase 4 {field} pairing failed for {replica}")
        p_grids = {
            tuple(row["p"] for row in normalized[(replica, condition)])
            for condition in conditions
        }
        if len(p_grids) != 1:
            raise Plan3Error(f"Phase 4 p-grid pairing failed for replica {replica}")

    expected_trajectories = []
    for condition in conditions:
        p_values = [row["p"] for row in normalized[(replicas[0], condition)]]
        for step, p_value in enumerate(p_values):
            row: dict[str, Any] = {
                "condition": condition,
                "step": step,
                "p": p_value,
            }
            for metric in ALL_METRICS:
                values = [
                    normalized[(replica, condition)][step][metric]
                    for replica in replicas
                    if normalized[(replica, condition)][step][metric] is not None
                ]
                row[metric] = _mean_interval([float(value) for value in values])
            expected_trajectories.append(row)

    replica_auc = []
    for replica in replicas:
        for condition in conditions:
            row = {"replica_index": replica, "condition": condition}
            for metric in ALL_METRICS:
                row[metric] = _normalized_auc(
                    normalized[(replica, condition)], metric
                )
            replica_auc.append(row)
    auc_by_key = {
        (row["replica_index"], row["condition"]): row for row in replica_auc
    }
    condition_auc = []
    for condition in conditions:
        row = {"condition": condition}
        for metric in ALL_METRICS:
            row[metric] = _mean_interval(
                [auc_by_key[(replica, condition)][metric] for replica in replicas]
            )
            row[metric]["direction"] = (
                "lower" if metric in LOWER_IS_BETTER else "higher"
            )
        condition_auc.append(row)

    comparison_pairs = (
        ("replay-b008", "ewc-fixed005-no-lfu"),
        ("replay-memory-matched", "ewc-fixed005-no-lfu"),
        ("replay-selected", "ewc-fixed005-no-lfu"),
        ("hybrid-b008-no-lfu", "replay-b008"),
        ("hybrid-memory-matched-no-lfu", "replay-memory-matched"),
        ("hybrid-selected-no-lfu", "replay-selected"),
        ("hybrid-b008-no-lfu", "ewc-fixed005-no-lfu"),
        ("hybrid-memory-matched-no-lfu", "ewc-fixed005-no-lfu"),
        ("hybrid-selected-no-lfu", "ewc-fixed005-no-lfu"),
        ("replay-memory-matched", "replay-b008"),
        ("replay-selected", "replay-memory-matched"),
        ("hybrid-memory-matched-no-lfu", "hybrid-b008-no-lfu"),
        ("hybrid-selected-no-lfu", "hybrid-memory-matched-no-lfu"),
        ("replay-unbounded", "replay-selected"),
        ("hybrid-selected-no-lfu", "replay-unbounded"),
    )
    comparisons = []
    for treatment, comparison in comparison_pairs:
        row = {"treatment": treatment, "comparison": comparison}
        for metric in ALL_METRICS:
            differences = [
                auc_by_key[(replica, treatment)][metric]
                - auc_by_key[(replica, comparison)][metric]
                for replica in replicas
            ]
            row[metric] = _mean_interval(differences)
            row[metric]["favorable_sign"] = (
                "negative" if metric in LOWER_IS_BETTER else "positive"
            )
        comparisons.append(row)

    resource_summary = []
    for condition in conditions:
        rows = [resources[(replica, condition)] for replica in replicas]
        summary: dict[str, Any] = {
            "condition": condition,
            "wall_time_comparable": all(row["wall_time_comparable"] for row in rows),
        }
        for metric in PHASE4_RESOURCE_METRICS:
            values = [
                float(row[metric]) for row in rows if row.get(metric) is not None
            ]
            summary[metric] = _mean_interval(values)
        resource_summary.append(summary)

    bundle_sha = hashlib.sha256((bundle.path / "bundle.json").read_bytes()).hexdigest()
    return {
        "schema_version": PLAN3_PHASE4_ANALYSIS_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": bundle_sha,
        "replica_indices": list(replicas),
        "conditions": list(conditions),
        "analysis_domain": {
            "auc": "normalized trapezoidal area over available 0 <= p < 0.5 points",
            "expected_trajectory": "mean within each exact p over paired replicas",
            "ci95": "two-sided Student-t interval across replicas",
            "primary_metrics": list(PRIMARY_METRICS),
            "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
            "memory_matched_label": (
                "capacity 25 matches one EWC summary only for the replay component"
            ),
        },
        "pairing_validated": True,
        "provenance": [
            {
                "replica_index": replica,
                "condition": condition,
                **provenance[(replica, condition)],
            }
            for replica in replicas
            for condition in conditions
        ],
        "expected_trajectories": expected_trajectories,
        "replica_auc": replica_auc,
        "condition_auc": condition_auc,
        "paired_auc_differences": comparisons,
        "resource_summary": resource_summary,
    }


def write_phase4_analysis(
    bundle: Plan3Phase4Bundle,
    repo_root: str | Path,
) -> Path:
    root = Path(repo_root)
    analysis = build_phase4_analysis(bundle, root)
    identity = {
        "schema_version": PLAN3_PHASE4_ANALYSIS_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": analysis["bundle_manifest_sha256"],
        "primary_metrics": list(PRIMARY_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "auc_domain": "0<=p<0.5",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "analysis"
        / f"phase4__{bundle.bundle_id}__{digest[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan3Error(f"incomplete Phase 4 analysis exists: {destination}")
        stored = _read_json(destination / "summary.json")
        if stored != analysis:
            raise Plan3Error("completed Phase 4 analysis has incompatible contents")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=incomplete))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(analysis, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(identity, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _phase5_controller_resource(
    metrics: Mapping[str, Any],
    *,
    uses_lfu: bool,
) -> dict[str, Any]:
    rows = metrics["condition_steps"]
    proposals = [row["proposal"] for row in rows if isinstance(row.get("proposal"), Mapping)]
    evaluations = sum(int(row["optimizer_function_evaluations"]) for row in proposals)
    elapsed = metrics.get("condition_elapsed_seconds")
    elapsed_values = [] if not isinstance(elapsed, Mapping) else list(elapsed.values())
    parameter_count = int(metrics["parameter_count"])
    ridge_bytes = (2 * parameter_count**2 + parameter_count + 1) * 8 if uses_lfu else 0
    return {
        "total_wall_seconds": (
            None if not elapsed_values else float(elapsed_values[0])
        ),
        "derivative_wall_seconds": sum(
            float(row["derivative_elapsed_seconds"]) for row in rows
        ),
        "fisher_update_wall_seconds": sum(
            float(row["update_elapsed_seconds"]) for row in rows
        ),
        "optimizer_iterations": sum(
            int(row["optimizer_iterations"]) for row in proposals
        ),
        "optimizer_function_evaluations": evaluations,
        "optimizer_event_evaluations": 8 * evaluations,
        "score_gradient_count": sum(int(row["score_gradient_count"]) for row in rows),
        "hvp_count": sum(int(row["hvp_count"]) for row in rows),
        "logical_persistent_bytes": 20_480 + ridge_bytes,
        "logical_lfu_state_persistent_bytes": ridge_bytes,
    }


def _phase5_hybrid_resource(metrics: Mapping[str, Any]) -> dict[str, Any]:
    base = _phase4_hybrid_resource(metrics)
    ledger = metrics["resource_ledger"]
    operations = ledger["operation_totals"]
    return {
        "total_wall_seconds": base["total_wall_seconds"],
        "derivative_wall_seconds": _operation_seconds(
            operations, "archive_score_fisher"
        ),
        "fisher_update_wall_seconds": _operation_seconds(
            operations, "archive_fisher_update"
        ),
        "optimizer_iterations": base["optimizer_iterations"],
        "optimizer_function_evaluations": base[
            "optimizer_function_evaluations"
        ],
        "optimizer_event_evaluations": base["optimizer_event_evaluations"],
        "score_gradient_count": base["archive_score_gradient_count"],
        "hvp_count": base["hvp_count"],
        "logical_persistent_bytes": base["logical_persistent_bytes"],
        "logical_lfu_state_persistent_bytes": int(
            ledger.get("logical_lfu_state_persistent_bytes_final", 0)
        ),
    }


def _phase5_lfu_diagnostics(
    metrics: Mapping[str, Any],
    *,
    runner: str,
    method: str,
) -> dict[str, float]:
    if method == "ema":
        return {
            "applied_updates": 0.0,
            "mean_correction_fro": 0.0,
            "maximum_correction_fro": 0.0,
            "mean_relative_projection_distance": 0.0,
            "maximum_relative_projection_distance": 0.0,
            "direction_reset_fraction": 0.0,
        }
    if runner == "controller":
        rows = metrics["condition_steps"]
        correction_key = (
            "smoothed_ac_fro" if method == "ac_only" else "smoothed_full_fro"
        )
        corrections = [
            float(row["update_diagnostics"]["ridge"][correction_key])
            for row in rows
        ]
        resets = [
            bool(row["update_diagnostics"]["ridge"]["direction_reset"])
            for row in rows
        ]
        materially_negative = [
            int(
                row["update_diagnostics"].get(
                    "candidate_materially_negative_eigenvalue_count", 0
                )
            )
            for row in rows
        ]
        return {
            "applied_updates": float(len(rows) - 1),
            "mean_correction_fro": statistics.fmean(corrections[1:]),
            "maximum_correction_fro": max(corrections[1:]),
            "mean_relative_projection_distance": 0.0,
            "maximum_relative_projection_distance": 0.0,
            "direction_reset_fraction": sum(resets) / len(resets),
            "materially_indefinite_candidate_fraction": sum(
                value > 0 for value in materially_negative
            )
            / len(materially_negative),
        }
    consolidations = [
        row["archive_consolidation"]
        for row in metrics["condition_steps"]
        if isinstance(row.get("archive_consolidation"), Mapping)
    ]
    lfu_rows = [row["lfu"] for row in consolidations if isinstance(row.get("lfu"), Mapping)]
    corrections = [float(row["correction_fro"]) for row in lfu_rows]
    projections = [
        float(row["projection"]["relative_projection_distance"])
        for row in lfu_rows
    ]
    resets = [bool(row["ridge"]["direction_reset"]) for row in lfu_rows]
    return {
        "applied_updates": float(len(lfu_rows)),
        "mean_correction_fro": statistics.fmean(corrections),
        "maximum_correction_fro": max(corrections),
        "mean_relative_projection_distance": statistics.fmean(projections),
        "maximum_relative_projection_distance": max(projections),
        "direction_reset_fraction": sum(resets) / len(resets),
        "materially_indefinite_candidate_fraction": sum(value > 1e-10 for value in projections)
        / len(projections),
    }


def build_phase5_analysis(
    bundle: Plan3Phase5Bundle,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root)
    statuses = plan3_phase5_status_rows(bundle, root)
    invalid = [
        row["run_id"]
        for row in statuses
        if row["run_state"] != "completed"
        or row["reused_controls_state"] != "completed"
        or row["archive_source_state"] not in {"completed", "not-applicable"}
    ]
    if invalid:
        raise Plan3Error(
            "Phase 5 analysis requires complete treatments and dependencies: "
            + ", ".join(invalid)
        )

    normalized: dict[tuple[int, str], list[dict[str, Any]]] = {}
    resources: dict[tuple[int, str], dict[str, Any]] = {}
    diagnostics: dict[tuple[int, str], dict[str, float]] = {}
    provenance: dict[tuple[int, str], dict[str, Any]] = {}
    rows = [*bundle.manifest["controls"], *bundle.manifest["entries"]]
    for entry in rows:
        replica = int(entry["replica_index"])
        condition = str(entry["condition"])
        key = (replica, condition)
        runner = entry.get("runner")
        artifact_kind = entry.get("artifact_kind")
        is_hybrid = runner == "hybrid" or artifact_kind == "hybrid"
        method = str(entry.get("method", "ema"))
        path = root / entry["cache_root"] / entry["run_id"]
        if is_hybrid:
            metrics = _read_json(path / "plan3_hybrid_metrics.json")
            normalized[key] = _normalize_hybrid_rows(metrics)
            resources[key] = _phase5_hybrid_resource(metrics)
            initial_hash = metrics["initial_parameter_hash"]
            source_schema = int(metrics["plan3_hybrid_metric_schema_version"])
        else:
            metrics = _read_json(path / "phase8_metrics.json")
            normalized[key] = _normalize_control_rows(metrics)
            resources[key] = _phase5_controller_resource(
                metrics,
                uses_lfu=method != "ema",
            )
            initial_hash = metrics["pairing"]["initial_parameter_hash"]
            source_schema = int(metrics["phase8_metric_schema_version"])
        diagnostics[key] = _phase5_lfu_diagnostics(
            metrics,
            runner="hybrid" if is_hybrid else "controller",
            method=method,
        )
        provenance[key] = {
            "run_id": entry["run_id"],
            "config_hash": entry["config_hash"],
            "stream_plan_hash": metrics["stream_plan_hash"],
            "replica_bundle_id": metrics["replica_bundle_id"],
            "initial_parameter_hash": initial_hash,
            "source_schema": source_schema,
        }

    replicas = tuple(int(value) for value in bundle.manifest["replica_indices"])
    conditions = tuple(str(value) for value in bundle.manifest["condition_order"])
    if conditions != PHASE5_CONDITION_ORDER:
        raise Plan3Error("Phase 5 condition order is incompatible")
    expected = {(replica, condition) for replica in replicas for condition in conditions}
    if set(normalized) != expected:
        raise Plan3Error("Phase 5 analysis matrix is incomplete")
    for replica in replicas:
        for field in ("stream_plan_hash", "replica_bundle_id", "initial_parameter_hash"):
            if len({provenance[(replica, condition)][field] for condition in conditions}) != 1:
                raise Plan3Error(f"Phase 5 {field} pairing failed for {replica}")

    replica_auc = []
    expected_trajectories = []
    for condition in conditions:
        p_values = [row["p"] for row in normalized[(replicas[0], condition)]]
        for step, p_value in enumerate(p_values):
            summary: dict[str, Any] = {"condition": condition, "step": step, "p": p_value}
            for metric in ALL_METRICS:
                summary[metric] = _mean_interval(
                    [
                        float(normalized[(replica, condition)][step][metric])
                        for replica in replicas
                        if normalized[(replica, condition)][step][metric] is not None
                    ]
                )
            expected_trajectories.append(summary)
    for replica in replicas:
        for condition in conditions:
            row: dict[str, Any] = {"replica_index": replica, "condition": condition}
            for metric in ALL_METRICS:
                row[metric] = _normalized_auc(normalized[(replica, condition)], metric)
            replica_auc.append(row)
    auc_by_key = {(row["replica_index"], row["condition"]): row for row in replica_auc}
    condition_auc = []
    for condition in conditions:
        row = {"condition": condition}
        for metric in ALL_METRICS:
            row[metric] = _mean_interval(
                [auc_by_key[(replica, condition)][metric] for replica in replicas]
            )
            row[metric]["direction"] = "lower" if metric in LOWER_IS_BETTER else "higher"
        condition_auc.append(row)

    comparison_pairs = (
        ("ewc-fixed005-ac-only", "ewc-fixed005-no-lfu"),
        ("ewc-fixed005-full-lfu", "ewc-fixed005-no-lfu"),
        ("ewc-fixed005-full-lfu", "ewc-fixed005-ac-only"),
        ("hybrid-selected-full-lfu", "hybrid-selected-no-lfu"),
    )
    paired = []
    for treatment, comparison in comparison_pairs:
        row = {"treatment": treatment, "comparison": comparison}
        for metric in ALL_METRICS:
            row[metric] = _mean_interval(
                [
                    auc_by_key[(replica, treatment)][metric]
                    - auc_by_key[(replica, comparison)][metric]
                    for replica in replicas
                ]
            )
            row[metric]["favorable_sign"] = (
                "negative" if metric in LOWER_IS_BETTER else "positive"
            )
        paired.append(row)

    resource_summary = []
    diagnostic_summary = []
    for condition in conditions:
        resource_row: dict[str, Any] = {"condition": condition}
        for metric in next(iter(resources.values())):
            values = [
                float(resources[(replica, condition)][metric])
                for replica in replicas
                if resources[(replica, condition)][metric] is not None
            ]
            resource_row[metric] = _mean_interval(values)
        resource_summary.append(resource_row)
        diagnostic_row: dict[str, Any] = {"condition": condition}
        keys = set().union(*(diagnostics[(replica, condition)] for replica in replicas))
        for metric in sorted(keys):
            diagnostic_row[metric] = _mean_interval(
                [diagnostics[(replica, condition)][metric] for replica in replicas]
            )
        diagnostic_summary.append(diagnostic_row)

    bundle_sha = hashlib.sha256((bundle.path / "bundle.json").read_bytes()).hexdigest()
    return {
        "schema_version": PLAN3_PHASE5_ANALYSIS_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": bundle_sha,
        "replica_indices": list(replicas),
        "conditions": list(conditions),
        "analysis_domain": {
            "auc": "normalized trapezoidal area over available 0 <= p < 0.5 points",
            "ci95": "two-sided Student-t interval across paired replicas",
            "primary_metrics": list(PRIMARY_METRICS),
            "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
            "lfu_gate": "predictive effects are interpreted with HVP, ridge-memory, and pre-projection diagnostics",
        },
        "pairing_validated": True,
        "provenance": [
            {"replica_index": replica, "condition": condition, **provenance[(replica, condition)]}
            for replica in replicas
            for condition in conditions
        ],
        "expected_trajectories": expected_trajectories,
        "replica_auc": replica_auc,
        "condition_auc": condition_auc,
        "paired_auc_differences": paired,
        "resource_summary": resource_summary,
        "lfu_diagnostics": diagnostic_summary,
    }


def write_phase5_analysis(
    bundle: Plan3Phase5Bundle,
    repo_root: str | Path,
) -> Path:
    root = Path(repo_root)
    analysis = build_phase5_analysis(bundle, root)
    identity = {
        "schema_version": PLAN3_PHASE5_ANALYSIS_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": analysis["bundle_manifest_sha256"],
        "auc_domain": "0<=p<0.5",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "analysis"
        / f"phase5__{bundle.bundle_id}__{digest[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan3Error(f"incomplete Phase 5 analysis exists: {destination}")
        if _read_json(destination / "summary.json") != analysis:
            raise Plan3Error("completed Phase 5 analysis has incompatible contents")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=incomplete))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(analysis, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(identity, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _phase6_resource(
    metrics: Mapping[str, Any],
    *,
    runner: str,
) -> dict[str, float | int]:
    ledger = metrics.get("resource_ledger")
    if not isinstance(ledger, Mapping):
        raise Plan3Error("deployment artifact is missing its resource ledger")
    operations = ledger.get("operation_totals")
    if not isinstance(operations, Mapping):
        raise Plan3Error("deployment artifact is missing operation timing")
    evaluation_wall = _operation_seconds(operations, "evaluation")
    learner_wall = sum(
        _operation_seconds(operations, name)
        for name in operations
        if name != "evaluation"
    )
    artifact_bytes = sum(int(value) for value in metrics["artifact_files_bytes"].values())
    if runner == "replay":
        return {
            "total_wall_seconds": float(ledger["total_wall_seconds"]),
            "trajectory_wall_seconds": float(ledger["trajectory_wall_seconds"]),
            "learner_wall_seconds_excluding_evaluation": learner_wall,
            "offline_evaluation_wall_seconds": evaluation_wall,
            "optimizer_iterations": int(ledger["optimizer_iterations"]),
            "optimizer_function_evaluations": int(
                ledger["optimizer_function_evaluations"]
            ),
            "optimizer_event_evaluations": int(
                ledger["optimizer_event_evaluations"]
            ),
            "score_gradient_count": 0,
            "hvp_count": 0,
            "logical_persistent_bytes": int(
                ledger["logical_replay_persistent_bytes_final"]
            ),
            "logical_archive_persistent_bytes": 0,
            "logical_replay_persistent_bytes": int(
                ledger["logical_replay_persistent_bytes_final"]
            ),
            "logical_controller_persistent_bytes": 0,
            "peak_process_rss_bytes": int(ledger["peak_process_rss_bytes"]),
            "peak_cuda_memory_bytes": int(ledger["peak_cuda_memory_bytes"]),
            "artifact_files_bytes": artifact_bytes,
        }
    accounting = metrics.get("archive_accounting")
    if not isinstance(accounting, Mapping):
        raise Plan3Error("deployment hybrid lacks archive accounting")
    return {
        "total_wall_seconds": float(ledger["total_wall_seconds"]),
        "trajectory_wall_seconds": float(ledger["trajectory_wall_seconds"]),
        "learner_wall_seconds_excluding_evaluation": learner_wall,
        "offline_evaluation_wall_seconds": evaluation_wall,
        "optimizer_iterations": int(ledger["learner_optimizer_iterations"])
        + int(ledger["archive_optimizer_iterations"]),
        "optimizer_function_evaluations": int(
            ledger["learner_optimizer_function_evaluations"]
        )
        + int(ledger["archive_optimizer_function_evaluations"]),
        "optimizer_event_evaluations": int(
            ledger["learner_optimizer_event_evaluations"]
        )
        + int(ledger["archive_optimizer_event_evaluations"]),
        "score_gradient_count": int(accounting["score_gradient_count"]),
        "hvp_count": int(accounting["hvp_count"]),
        "logical_persistent_bytes": int(
            ledger["logical_hybrid_persistent_bytes_final"]
        ),
        "logical_archive_persistent_bytes": int(
            ledger["logical_archive_persistent_bytes_final"]
        ),
        "logical_replay_persistent_bytes": int(
            ledger["logical_replay_persistent_bytes_final"]
        ),
        "logical_controller_persistent_bytes": int(
            ledger["logical_controller_persistent_bytes_final"]
        ),
        "peak_process_rss_bytes": int(ledger["peak_process_rss_bytes"]),
        "peak_cuda_memory_bytes": int(ledger["peak_cuda_memory_bytes"]),
        "artifact_files_bytes": artifact_bytes,
    }


def build_phase6_analysis(
    bundle: Plan3Phase6Bundle,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root)
    statuses = plan3_phase6_status_rows(bundle, root)
    invalid = [
        row["run_id"]
        for row in statuses
        if row["run_state"] != "completed"
        or row["archive_source_state"] not in {"completed", "not-applicable"}
    ]
    if invalid:
        raise Plan3Error(
            "Phase 6 analysis requires complete deployment runs: "
            + ", ".join(invalid)
        )

    normalized: dict[tuple[int, str], list[dict[str, Any]]] = {}
    resources: dict[tuple[int, str], dict[str, float | int]] = {}
    provenance: dict[tuple[int, str], dict[str, Any]] = {}
    controller: dict[tuple[int, str], dict[str, Any] | None] = {}
    for entry in bundle.manifest["entries"]:
        replica = int(entry["replica_index"])
        condition = str(entry["condition"])
        key = (replica, condition)
        path = root / entry["cache_root"] / entry["run_id"]
        config = _read_json(path / "config.json")
        if config["controller"].get("reference_optimum_artifact") is not None:
            raise Plan3Error("Phase 6 deployment config retained an oracle artifact")
        if config["controller"].get("oracle_mode") != "none":
            raise Plan3Error("Phase 6 deployment config retained oracle mode")
        if entry["runner"] == "replay":
            metrics = _read_json(path / "plan3_replay_metrics.json")
            normalized[key] = _normalize_replay_rows(metrics)
            source_schema = int(metrics["plan3_replay_metric_schema_version"])
            controller[key] = None
        else:
            metrics = _read_json(path / "plan3_hybrid_metrics.json")
            if metrics.get("plan3_hybrid_metric_schema_version") != 12:
                raise Plan3Error("Phase 6 hybrid requires schema-12 metrics")
            if metrics.get("condition", {}).get("oracle_free") is not True:
                raise Plan3Error("Phase 6 hybrid is not marked oracle-free")
            normalized[key] = _normalize_hybrid_rows(metrics)
            source_schema = int(metrics["plan3_hybrid_metric_schema_version"])
            summary = metrics.get("controller_summary")
            if not isinstance(summary, Mapping):
                raise Plan3Error("Phase 6 hybrid lacks controller diagnostics")
            controller[key] = dict(summary)
        resources[key] = _phase6_resource(metrics, runner=entry["runner"])
        if resources[key]["hvp_count"] != 0:
            raise Plan3Error("Phase 6 no-LFU deployment unexpectedly used HVPs")
        provenance[key] = {
            "run_id": entry["run_id"],
            "config_hash": entry["config_hash"],
            "stream_plan_hash": metrics["stream_plan_hash"],
            "replica_bundle_id": metrics["replica_bundle_id"],
            "initial_parameter_hash": metrics["initial_parameter_hash"],
            "source_schema": source_schema,
        }

    replicas = tuple(int(value) for value in bundle.manifest["replica_indices"])
    conditions = tuple(str(value) for value in bundle.manifest["condition_order"])
    if conditions != PHASE6_CONDITION_ORDER:
        raise Plan3Error("Phase 6 condition order is incompatible")
    expected = {(replica, condition) for replica in replicas for condition in conditions}
    if set(normalized) != expected:
        raise Plan3Error("Phase 6 deployment matrix is incomplete")
    for replica in replicas:
        for field in ("stream_plan_hash", "replica_bundle_id", "initial_parameter_hash"):
            if len({provenance[(replica, condition)][field] for condition in conditions}) != 1:
                raise Plan3Error(f"Phase 6 {field} pairing failed for {replica}")

    expected_trajectories = []
    replica_auc = []
    for condition in conditions:
        p_values = [row["p"] for row in normalized[(replicas[0], condition)]]
        for step, p_value in enumerate(p_values):
            summary: dict[str, Any] = {
                "condition": condition,
                "step": step,
                "p": p_value,
            }
            for metric in ALL_METRICS:
                summary[metric] = _mean_interval(
                    [
                        float(normalized[(replica, condition)][step][metric])
                        for replica in replicas
                        if normalized[(replica, condition)][step][metric] is not None
                    ]
                )
            expected_trajectories.append(summary)
    for replica in replicas:
        for condition in conditions:
            row: dict[str, Any] = {"replica_index": replica, "condition": condition}
            for metric in ALL_METRICS:
                row[metric] = _normalized_auc(normalized[(replica, condition)], metric)
            replica_auc.append(row)
    auc_by_key = {
        (row["replica_index"], row["condition"]): row for row in replica_auc
    }
    condition_auc = []
    for condition in conditions:
        row = {"condition": condition}
        for metric in ALL_METRICS:
            row[metric] = _mean_interval(
                [auc_by_key[(replica, condition)][metric] for replica in replicas]
            )
            row[metric]["direction"] = (
                "lower" if metric in LOWER_IS_BETTER else "higher"
            )
        condition_auc.append(row)

    comparison_pairs = (
        ("deployment-ewc-fixed005", "deployment-current-only"),
        ("deployment-hybrid-b032-fixed005", "deployment-ewc-fixed005"),
        ("deployment-ewc-adaptive-h020", "deployment-ewc-fixed005"),
        (
            "deployment-hybrid-b032-adaptive-h020",
            "deployment-hybrid-b032-fixed005",
        ),
        ("deployment-replay-b032", "deployment-current-only"),
        ("deployment-hybrid-b032-fixed005", "deployment-replay-b032"),
        ("deployment-hybrid-b032-adaptive-h020", "deployment-replay-b032"),
        ("deployment-replay-unbounded", "deployment-replay-b032"),
        ("deployment-hybrid-b032-fixed005", "deployment-replay-unbounded"),
    )
    paired = []
    for treatment, comparison in comparison_pairs:
        row = {"treatment": treatment, "comparison": comparison}
        for metric in ALL_METRICS:
            row[metric] = _mean_interval(
                [
                    auc_by_key[(replica, treatment)][metric]
                    - auc_by_key[(replica, comparison)][metric]
                    for replica in replicas
                ]
            )
            row[metric]["favorable_sign"] = (
                "negative" if metric in LOWER_IS_BETTER else "positive"
            )
        paired.append(row)

    resource_summary = []
    for condition in conditions:
        row: dict[str, Any] = {"condition": condition}
        for metric in next(iter(resources.values())):
            row[metric] = _mean_interval(
                [float(resources[(replica, condition)][metric]) for replica in replicas]
            )
        resource_summary.append(row)

    controller_summary = []
    for condition in conditions:
        rows = [controller[(replica, condition)] for replica in replicas]
        available = [row for row in rows if row is not None]
        if not available:
            controller_summary.append({"condition": condition, "policy": "none"})
            continue
        row = {"condition": condition, "policy": available[0]["policy"]}
        for metric in (
            "applied_pi_min",
            "applied_pi_mean",
            "applied_pi_max",
            "lower_bound_fraction",
            "upper_bound_fraction",
        ):
            row[metric] = _mean_interval([float(value[metric]) for value in available])
        row["mean_actuation_range"] = _mean_interval(
            [
                float(value["applied_pi_max"]) - float(value["applied_pi_min"])
                for value in available
            ]
        )
        controller_summary.append(row)

    bundle_sha = hashlib.sha256((bundle.path / "bundle.json").read_bytes()).hexdigest()
    return {
        "schema_version": PLAN3_PHASE6_ANALYSIS_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": bundle_sha,
        "replica_indices": list(replicas),
        "conditions": list(conditions),
        "analysis_domain": {
            "auc": "normalized trapezoidal area over available 0 <= p < 0.5 points",
            "ci95": "two-sided Student-t interval across paired replicas",
            "learner_cost": "all timed learner operations excluding offline holdout evaluation",
            "fisher_update": "EMA with no LFU",
            "oracle_free": True,
        },
        "pairing_validated": True,
        "provenance": [
            {
                "replica_index": replica,
                "condition": condition,
                **provenance[(replica, condition)],
            }
            for replica in replicas
            for condition in conditions
        ],
        "expected_trajectories": expected_trajectories,
        "replica_auc": replica_auc,
        "condition_auc": condition_auc,
        "paired_auc_differences": paired,
        "resource_summary": resource_summary,
        "controller_summary": controller_summary,
    }


def write_phase6_analysis(
    bundle: Plan3Phase6Bundle,
    repo_root: str | Path,
) -> Path:
    root = Path(repo_root)
    analysis = build_phase6_analysis(bundle, root)
    identity = {
        "schema_version": PLAN3_PHASE6_ANALYSIS_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": analysis["bundle_manifest_sha256"],
        "analysis_sha256": _json_sha256(analysis),
        "auc_domain": "0<=p<0.5",
        "learner_cost_excludes_evaluation": True,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "analysis"
        / f"phase6__{bundle.bundle_id}__{digest[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan3Error(f"incomplete Phase 6 analysis exists: {destination}")
        if _read_json(destination / "summary.json") != analysis:
            raise Plan3Error("completed Phase 6 analysis has incompatible contents")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=incomplete))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(analysis, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(identity, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
}


def _phase7_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    if len(values) < 2:
        return _mean_interval(values)
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    critical = _T95.get(len(values) - 1, 1.96)
    radius = critical * standard_error
    return {
        "count": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
    }


def _interval_half_width(interval: Mapping[str, Any]) -> float | None:
    low = interval.get("ci95_low")
    high = interval.get("ci95_high")
    if low is None or high is None:
        return None
    return (float(high) - float(low)) / 2.0


def build_phase7_analysis(
    bundle: Plan3Phase7Bundle,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root)
    statuses = plan3_phase7_status_rows(bundle, root)
    by_replica: dict[int, list[dict[str, Any]]] = {}
    for row in statuses:
        by_replica.setdefault(int(row["replica_index"]), []).append(row)
    completed: list[int] = []
    for replica in bundle.manifest["replica_indices"]:
        rows = by_replica[int(replica)]
        if all(
            row["run_state"] == "completed"
            and row["replica_bundle_state"] == "completed"
            and row["archive_source_state"] in {"completed", "not-applicable"}
            for row in rows
        ):
            completed.append(int(replica))
        else:
            break
    block_size = int(bundle.manifest["block_size"])
    usable_count = len(completed) - len(completed) % block_size
    replicas = tuple(completed[:usable_count])
    if not replicas:
        raise Plan3Error("Phase 7 analysis requires one complete five-replica block")

    conditions = tuple(str(value) for value in bundle.manifest["condition_order"])
    if conditions != PHASE7_CONDITION_ORDER:
        raise Plan3Error("Phase 7 condition order is incompatible")
    selected_entries = [
        entry
        for entry in bundle.manifest["entries"]
        if int(entry["replica_index"]) in replicas
    ]
    normalized: dict[tuple[int, str], list[dict[str, Any]]] = {}
    resources: dict[tuple[int, str], dict[str, float | int]] = {}
    provenance: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in selected_entries:
        replica = int(entry["replica_index"])
        condition = str(entry["condition"])
        key = (replica, condition)
        path = root / entry["cache_root"] / entry["run_id"]
        config = _read_json(path / "config.json")
        if config["controller"].get("oracle_mode") != "none" or config[
            "controller"
        ].get("reference_optimum_artifact") is not None:
            raise Plan3Error("Phase 7 confirmation retained an oracle dependency")
        if entry["runner"] == "replay":
            metrics = _read_json(path / "plan3_replay_metrics.json")
            normalized[key] = _normalize_replay_rows(metrics)
            source_schema = int(metrics["plan3_replay_metric_schema_version"])
        else:
            metrics = _read_json(path / "plan3_hybrid_metrics.json")
            if metrics.get("plan3_hybrid_metric_schema_version") != 12:
                raise Plan3Error("Phase 7 hybrid requires schema-12 metrics")
            if metrics.get("condition", {}).get("oracle_free") is not True:
                raise Plan3Error("Phase 7 hybrid is not marked oracle-free")
            normalized[key] = _normalize_hybrid_rows(metrics)
            source_schema = int(metrics["plan3_hybrid_metric_schema_version"])
        resources[key] = _phase6_resource(metrics, runner=entry["runner"])
        if resources[key]["hvp_count"] != 0:
            raise Plan3Error("Phase 7 no-LFU confirmation unexpectedly used HVPs")
        archive_source = entry.get("archive_source_artifact")
        provenance[key] = {
            "run_id": entry["run_id"],
            "config_hash": entry["config_hash"],
            "stream_plan_hash": metrics["stream_plan_hash"],
            "replica_bundle_id": metrics["replica_bundle_id"],
            "initial_parameter_hash": metrics["initial_parameter_hash"],
            "source_schema": source_schema,
            "initial_archive_sha256": (
                None
                if archive_source is None
                else hashlib.sha256((root / archive_source).read_bytes()).hexdigest()
            ),
        }
    expected = {(replica, condition) for replica in replicas for condition in conditions}
    if set(normalized) != expected:
        raise Plan3Error("Phase 7 fresh confirmation matrix is incomplete")
    for replica in replicas:
        for field in ("stream_plan_hash", "replica_bundle_id", "initial_parameter_hash"):
            if len(
                {provenance[(replica, condition)][field] for condition in conditions}
            ) != 1:
                raise Plan3Error(f"Phase 7 {field} pairing failed for {replica}")

    expected_trajectories = []
    for condition in conditions:
        rows = [
            row
            for row in normalized[(replicas[0], condition)]
            if 0.0 <= float(row["p"]) < 0.5
        ]
        for row in rows:
            step = int(row["step"])
            summary: dict[str, Any] = {
                "condition": condition,
                "step": step,
                "p": float(row["p"]),
                "observations_before_evaluation": step * 8,
                "expected_nines_before_evaluation": (
                    8.0 * step * (step - 1) / (2.0 * 99.0)
                ),
            }
            for metric in ALL_METRICS:
                summary[metric] = _phase7_interval(
                    [
                        float(normalized[(replica, condition)][step][metric])
                        for replica in replicas
                        if normalized[(replica, condition)][step][metric] is not None
                    ]
                )
            expected_trajectories.append(summary)

    replica_auc = []
    for replica in replicas:
        for condition in conditions:
            row: dict[str, Any] = {"replica_index": replica, "condition": condition}
            for metric in ALL_METRICS:
                row[metric] = _normalized_auc(normalized[(replica, condition)], metric)
            replica_auc.append(row)
    auc_by_key = {
        (int(row["replica_index"]), str(row["condition"])): row
        for row in replica_auc
    }
    condition_auc = []
    for condition in conditions:
        row = {"condition": condition}
        for metric in ALL_METRICS:
            row[metric] = _phase7_interval(
                [auc_by_key[(replica, condition)][metric] for replica in replicas]
            )
            row[metric]["direction"] = (
                "lower" if metric in LOWER_IS_BETTER else "higher"
            )
        condition_auc.append(row)

    comparison_pairs = (
        ("confirm-ewc-fixed005", "confirm-current-only"),
        ("confirm-hybrid-b032-fixed005", "confirm-ewc-fixed005"),
        ("confirm-replay-b032", "confirm-current-only"),
        ("confirm-hybrid-b032-fixed005", "confirm-replay-b032"),
        ("confirm-replay-unbounded", "confirm-replay-b032"),
        ("confirm-hybrid-b032-fixed005", "confirm-replay-unbounded"),
    )
    paired_auc = []
    pointwise = []
    for treatment, comparison in comparison_pairs:
        auc_row: dict[str, Any] = {
            "treatment": treatment,
            "comparison": comparison,
        }
        for metric in ALL_METRICS:
            auc_row[metric] = _phase7_interval(
                [
                    auc_by_key[(replica, treatment)][metric]
                    - auc_by_key[(replica, comparison)][metric]
                    for replica in replicas
                ]
            )
            auc_row[metric]["favorable_sign"] = (
                "negative" if metric in LOWER_IS_BETTER else "positive"
            )
        paired_auc.append(auc_row)
        for step in range(50):
            row = {
                "treatment": treatment,
                "comparison": comparison,
                "step": step,
                "p": float(normalized[(replicas[0], treatment)][step]["p"]),
                "observations_before_evaluation": step * 8,
            }
            for metric in ALL_METRICS:
                row[metric] = _phase7_interval(
                    [
                        float(normalized[(replica, treatment)][step][metric])
                        - float(normalized[(replica, comparison)][step][metric])
                        for replica in replicas
                        if normalized[(replica, treatment)][step][metric] is not None
                        and normalized[(replica, comparison)][step][metric] is not None
                    ]
                )
            pointwise.append(row)

    resource_summary = []
    for condition in conditions:
        row: dict[str, Any] = {"condition": condition}
        for metric in next(iter(resources.values())):
            row[metric] = _phase7_interval(
                [float(resources[(replica, condition)][metric]) for replica in replicas]
            )
        resource_summary.append(row)

    anchor_by_replica = {
        int(row["replica_index"]): row for row in bundle.manifest["anchors"]
    }
    initial_archive_diagnostics = []
    for replica in replicas:
        anchor_entry = anchor_by_replica[replica]
        anchor_path = root / anchor_entry["cache_root"] / anchor_entry["run_id"]
        metrics = _read_json(anchor_path / "initial_archive_metrics.json")
        if metrics.get("oracle_free") is not True:
            raise Plan3Error("Phase 7 initial archive is not oracle-free")
        if metrics.get("replica_bundle_id") != anchor_entry["replica_bundle_id"]:
            raise Plan3Error("Phase 7 initial archive pairing failed")
        convergence = metrics["fisher"]["convergence"]
        dependence = metrics["fisher"]["dependence"]
        lanczos = metrics["lanczos"]
        initial_archive_diagnostics.append(
            {
                "replica_index": replica,
                "run_id": anchor_entry["run_id"],
                "config_hash": anchor_entry["config_hash"],
                "score_gradient_count": int(metrics["score_gradient_count"]),
                "converged": bool(convergence["converged"]),
                "stopping_reason": str(convergence["stopping_reason"]),
                "relative_confidence_radius": float(
                    convergence["relative_confidence_radius"]
                ),
                "relative_epsilon": float(convergence["relative_epsilon"]),
                "lag_one_frobenius_correlation": float(
                    dependence["lag_one_frobenius_correlation"]
                ),
                "draw_duplicate_fraction": float(
                    dependence["draw_duplicate_fraction"]
                ),
                "mean_adjacent_chunk_index_overlap": float(
                    dependence["mean_adjacent_chunk_index_overlap"]
                ),
                "realized_rank": int(lanczos["realized_rank"]),
                "lanczos_retry_count": int(lanczos["numerical_retry_count"]),
                "represented_diagonal_relative_error": float(
                    lanczos["represented_diagonal_relative_error"]
                ),
                "wall_time_seconds": float(metrics["wall_time_seconds"]),
            }
        )
    initial_archive_summary = {
        "replica_count": len(replicas),
        "converged_count": sum(
            int(row["converged"]) for row in initial_archive_diagnostics
        ),
        "maximum_budget_count": sum(
            int(row["stopping_reason"] == "maximum_budget")
            for row in initial_archive_diagnostics
        ),
        "relative_confidence_radius": _phase7_interval(
            [row["relative_confidence_radius"] for row in initial_archive_diagnostics]
        ),
        "lag_one_frobenius_correlation": _phase7_interval(
            [
                row["lag_one_frobenius_correlation"]
                for row in initial_archive_diagnostics
            ]
        ),
        "draw_duplicate_fraction": _phase7_interval(
            [row["draw_duplicate_fraction"] for row in initial_archive_diagnostics]
        ),
        "represented_diagonal_relative_error": _phase7_interval(
            [
                row["represented_diagonal_relative_error"]
                for row in initial_archive_diagnostics
            ]
        ),
        "wall_time_seconds": _phase7_interval(
            [row["wall_time_seconds"] for row in initial_archive_diagnostics]
        ),
    }

    primary = next(
        row
        for row in paired_auc
        if row["treatment"] == "confirm-hybrid-b032-fixed005"
        and row["comparison"] == "confirm-replay-b032"
    )
    primary_interval = primary["environment_accuracy"]
    nine_interval = primary["nine_ovr_accuracy"]
    primary_half_width = _interval_half_width(primary_interval)
    pointwise_half_widths = sorted(
        value
        for row in pointwise
        if row["treatment"] == "confirm-hybrid-b032-fixed005"
        and row["comparison"] == "confirm-replay-b032"
        for value in [_interval_half_width(row["environment_accuracy"])]
        if value is not None
    )
    median_pointwise_half_width = statistics.median(pointwise_half_widths)
    contract = bundle.manifest["analysis_contract"]
    precision_met = (
        len(replicas) >= int(bundle.manifest["initial_replica_target"])
        and primary_half_width is not None
        and primary_half_width
        <= float(contract["primary_auc_ci_half_width_target"])
        and median_pointwise_half_width
        <= float(contract["median_pointwise_ci_half_width_target"])
    )
    maximum_reached = len(replicas) >= int(bundle.manifest["maximum_replica_count"])
    precision_gate = {
        "completed_replica_count": len(replicas),
        "minimum_replica_count": int(bundle.manifest["initial_replica_target"]),
        "maximum_replica_count": int(bundle.manifest["maximum_replica_count"]),
        "primary_auc_ci_half_width": primary_half_width,
        "primary_auc_ci_half_width_target": float(
            contract["primary_auc_ci_half_width_target"]
        ),
        "median_pointwise_ci_half_width": median_pointwise_half_width,
        "median_pointwise_ci_half_width_target": float(
            contract["median_pointwise_ci_half_width_target"]
        ),
        "primary_effect_positive": (
            primary_interval["ci95_low"] is not None
            and float(primary_interval["ci95_low"]) > 0.0
        ),
        "nine_ovr_practically_equivalent": (
            nine_interval["ci95_low"] is not None
            and float(nine_interval["ci95_low"])
            >= -float(contract["nine_ovr_practical_equivalence_margin"])
            and float(nine_interval["ci95_high"])
            <= float(contract["nine_ovr_practical_equivalence_margin"])
        ),
        "precision_target_met": precision_met,
        "maximum_reached": maximum_reached,
        "recommendation": (
            "stop_precision_target_met"
            if precision_met
            else "stop_maximum_reached"
            if maximum_reached
            else "continue_one_predeclared_five_replica_block"
        ),
    }
    bundle_sha = hashlib.sha256((bundle.path / "bundle.json").read_bytes()).hexdigest()
    return {
        "schema_version": PLAN3_PHASE7_ANALYSIS_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": bundle_sha,
        "replica_indices": list(replicas),
        "conditions": list(conditions),
        "analysis_domain": {
            "trajectory": "only 0 <= p < 0.5 is summarized and plotted",
            "underlying_path": "unchanged 100-point linear p trajectory from 0 to 1",
            "auc": "normalized trapezoidal area over 0 <= p < 0.5",
            "ci95": "two-sided Student-t interval across fresh paired replicas",
            "evaluation_timing": (
                "step t metrics precede the step-t update; exposure is t*m"
            ),
            "oracle_free": True,
            "fisher_update": "EMA with no LFU",
        },
        "pairing_validated": True,
        "provenance": [
            {
                "replica_index": replica,
                "condition": condition,
                **provenance[(replica, condition)],
            }
            for replica in replicas
            for condition in conditions
        ],
        "expected_trajectories": expected_trajectories,
        "pointwise_paired_differences": pointwise,
        "replica_auc": replica_auc,
        "condition_auc": condition_auc,
        "paired_auc_differences": paired_auc,
        "resource_summary": resource_summary,
        "initial_archive_diagnostics": initial_archive_diagnostics,
        "initial_archive_summary": initial_archive_summary,
        "precision_gate": precision_gate,
    }


def write_phase7_analysis(
    bundle: Plan3Phase7Bundle,
    repo_root: str | Path,
) -> Path:
    root = Path(repo_root)
    analysis = build_phase7_analysis(bundle, root)
    identity = {
        "schema_version": PLAN3_PHASE7_ANALYSIS_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": analysis["bundle_manifest_sha256"],
        "replica_indices": analysis["replica_indices"],
        "analysis_sha256": _json_sha256(analysis),
        "auc_domain": "0<=p<0.5",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan3"
        / "analysis"
        / f"phase7__{bundle.bundle_id}__n{len(analysis['replica_indices']):02d}__{digest[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan3Error(f"incomplete Phase 7 analysis exists: {destination}")
        if _read_json(destination / "summary.json") != analysis:
            raise Plan3Error("completed Phase 7 analysis has incompatible contents")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=incomplete))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(analysis, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(identity, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
