"""Plan 4 exponentially discounted Fisher-risk screening and experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .controller import DiscountedRiskState, advance_discounted_risk
from .plan4_analysis import _event_mask
from .plan4_challenge import (
    SCHEDULES,
    Plan4ChallengeError,
    _read_json,
    _sha256,
    load_screen_bundle,
    status_rows,
)
from .schedules import resolve_schedule
from .config import ExperimentConfig, load_config
from .initialization import replica_bundle_id, replica_design_hash
from .plan4_floor import (
    FLOOR_CONDITIONS,
    PREDICTIVE_FIELDS,
    _condition_predictive_summary,
    _paired_predictive_summary,
)


PLAN4_EDR_SCREEN_SCHEMA_VERSION = 1
PLAN4_EDR_BUNDLE_SCHEMA_VERSION = 1
PLAN4_EDR_ANALYSIS_SCHEMA_VERSION = 2
HALF_LIVES = (2.0, 4.0, 8.0)
FLOORS = (0.01, 0.02, 0.03)
PRIMARY_HALF_LIFE = 4.0
PRIMARY_FLOOR = 0.01
COLD_START_PI = 0.025
PI_MAX = 0.95
RAW_CONDITION = "adaptive-fisher-pimin0025-h005"
EDR_CONDITION = "edr-fisher-pimin001-h04"
EDR_CONDITIONS = (*FLOOR_CONDITIONS, EDR_CONDITION)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _clip(value: float, floor: float) -> float:
    return min(PI_MAX, max(floor, value))


def filter_risk_coefficients(
    rows: Sequence[Mapping[str, Any]],
    event: np.ndarray,
    *,
    half_life_steps: float,
    pi_min: float,
) -> dict[str, Any]:
    """Filter stored predictable coefficients without claiming a rollout."""

    updates = [row for row in rows if isinstance(row.get("controller_decision"), Mapping)]
    if len(updates) != event.size:
        raise Plan4ChallengeError("EDR coefficient trace does not align with event mask")
    state = DiscountedRiskState()
    trajectory = []
    for index, row in enumerate(updates):
        decision = row["controller_decision"]
        old_risk = float(decision["signal_energy"]) + float(
            decision["old_covariance_risk"]
        )
        new_risk = float(decision["new_covariance_risk"])
        state, gain = advance_discounted_risk(
            state,
            old_risk,
            new_risk,
            half_life_steps=half_life_steps,
        )
        denominator = state.old_risk_moment + state.new_risk_moment
        zero_denominator = denominator == 0.0
        unclipped = (
            None if zero_denominator else state.old_risk_moment / denominator
        )
        cold_start = bool(decision["cold_start_active"])
        raw_action = (
            COLD_START_PI
            if cold_start or zero_denominator
            else float(unclipped)
        )
        applied = _clip(raw_action, pi_min)
        trajectory.append(
            {
                "step": int(row["step"]),
                "p": float(row["p"]),
                "in_event_window": bool(event[index]),
                "cold_start_active": cold_start,
                "instantaneous_old_risk": old_risk,
                "instantaneous_new_risk": new_risk,
                "old_risk_moment": state.old_risk_moment,
                "new_risk_moment": state.new_risk_moment,
                "unclipped_pi": unclipped,
                "applied_pi": applied,
                "gain": gain,
                "zero_denominator_fallback": zero_denominator,
            }
        )

    applied = np.asarray([row["applied_pi"] for row in trajectory], dtype=np.float64)
    unclipped = np.asarray(
        [math.nan if row["unclipped_pi"] is None else row["unclipped_pi"] for row in trajectory],
        dtype=np.float64,
    )
    cold = np.asarray([row["cold_start_active"] for row in trajectory], dtype=bool)
    old = np.asarray([row["instantaneous_old_risk"] for row in trajectory])
    new = np.asarray([row["instantaneous_new_risk"] for row in trajectory])
    risk = (1.0 - applied) ** 2 * old + applied**2 * new
    event_excess = np.maximum(applied[event] - COLD_START_PI, 0.0)
    post_cold = ~cold
    return {
        "half_life_steps": half_life_steps,
        "pi_min": pi_min,
        "applied_pi_min": float(applied.min()),
        "applied_pi_mean": float(applied.mean()),
        "applied_pi_max": float(applied.max()),
        "unclipped_pi_mean": (
            float(unclipped[np.isfinite(unclipped)].mean())
            if np.isfinite(unclipped).any()
            else None
        ),
        "total_variation": float(np.abs(np.diff(applied)).sum()),
        "event_excess_action_area": float(event_excess.sum()),
        "event_peak_step": int(trajectory[int(np.flatnonzero(event)[np.argmax(applied[event])])]["step"]),
        "post_cold_floor_fraction": (
            float(np.mean(applied[post_cold] <= pi_min + 1e-12))
            if post_cold.any()
            else None
        ),
        "same_state_risk": float(risk.sum()),
        "zero_denominator_count": int(
            sum(row["zero_denominator_fallback"] for row in trajectory)
        ),
        "trajectory": trajectory,
    }


def _raw_summary(rows: Sequence[Mapping[str, Any]], event: np.ndarray) -> dict[str, Any]:
    updates = [row for row in rows if isinstance(row.get("controller_decision"), Mapping)]
    applied = np.asarray(
        [float(row["controller_decision"]["applied_pi"]) for row in updates],
        dtype=np.float64,
    )
    return {
        "total_variation": float(np.abs(np.diff(applied)).sum()),
        "event_excess_action_area": float(
            np.maximum(applied[event] - COLD_START_PI, 0.0).sum()
        ),
        "event_peak_step": int(updates[int(np.flatnonzero(event)[np.argmax(applied[event])])]["step"]),
    }


def build_edr_screen(
    floor_bundle_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    bundle, manifest = load_screen_bundle(floor_bundle_path)
    if manifest.get("stage") != "exploratory-floor-sensitivity":
        raise Plan4ChallengeError("EDR screen requires the completed floor bundle")
    status = status_rows(bundle, repo_root)
    if any(row["run_state"] != "completed" for row in status):
        raise Plan4ChallengeError("EDR screen inputs must all be complete")

    schedules = {}
    provenance = []
    primary_rows = []
    for schedule_name in SCHEDULES:
        members = [
            row
            for row in status
            if row["schedule"] == schedule_name and row["condition"] == RAW_CONDITION
        ]
        if len(members) != 1:
            raise Plan4ChallengeError(f"missing raw EDR source for {schedule_name}")
        member = members[0]
        metric_path = Path(member["run_path"]) / "plan3_hybrid_metrics.json"
        metrics = _read_json(metric_path)
        rows = metrics["condition_steps"]
        resolved = resolve_schedule(load_config(member["config_path"]).data)
        event = _event_mask(resolved)
        raw = _raw_summary(rows, event)
        filters = {}
        for half_life in HALF_LIVES:
            for floor in FLOORS:
                key = f"h{int(half_life):02d}-floor{int(round(floor * 1000)):03d}"
                filtered = filter_risk_coefficients(
                    rows,
                    event,
                    half_life_steps=half_life,
                    pi_min=floor,
                )
                filtered["total_variation_ratio_vs_raw"] = (
                    filtered["total_variation"] / raw["total_variation"]
                    if raw["total_variation"] > 0.0
                    else 0.0
                )
                filtered["event_response_ratio_vs_raw"] = (
                    filtered["event_excess_action_area"]
                    / raw["event_excess_action_area"]
                    if raw["event_excess_action_area"] > 0.0
                    else None
                )
                filtered["event_peak_delay_steps_vs_raw"] = (
                    filtered["event_peak_step"] - raw["event_peak_step"]
                )
                filters[key] = filtered
                if half_life == PRIMARY_HALF_LIFE and floor == PRIMARY_FLOOR:
                    primary_rows.append(filtered)
        schedules[schedule_name] = {"raw": raw, "filters": filters}
        provenance.append(
            {
                "schedule": schedule_name,
                "run_id": member["run_id"],
                "metrics_sha256": _sha256(metric_path),
            }
        )

    nonlinear = primary_rows[1:]
    mean_variation_ratio = float(
        np.mean([row["total_variation_ratio_vs_raw"] for row in nonlinear])
    )
    response_ratios = [
        row["event_response_ratio_vs_raw"]
        for row in nonlinear
        if row["event_response_ratio_vs_raw"] is not None
    ]
    mean_response_ratio = (
        float(np.mean(response_ratios)) if response_ratios else None
    )
    floor_fractions = [
        row["post_cold_floor_fraction"]
        for row in primary_rows
        if row["post_cold_floor_fraction"] is not None
    ]
    gate = {
        "primary_half_life_steps": PRIMARY_HALF_LIFE,
        "primary_pi_min": PRIMARY_FLOOR,
        "mean_nonlinear_total_variation_ratio_vs_raw": mean_variation_ratio,
        "mean_nonlinear_event_response_ratio_vs_raw": mean_response_ratio,
        "maximum_post_cold_floor_fraction": max(floor_fractions),
    }
    gate["passes_assumption_screen"] = bool(
        mean_variation_ratio < 1.0
        and mean_response_ratio is not None
        and mean_response_ratio > 0.0
        and max(floor_fractions) < 0.9
    )
    return {
        "schema_version": PLAN4_EDR_SCREEN_SCHEMA_VERSION,
        "analysis_kind": "plan4_edr_fixed_state_screen",
        "counterfactual_predictive_claim": False,
        "source_bundle_id": manifest["bundle_id"],
        "source_manifest_sha256": _sha256(bundle / "manifest.json"),
        "input_provenance": provenance,
        "half_lives": list(HALF_LIVES),
        "floors": list(FLOORS),
        "gate": gate,
        "schedules": schedules,
    }


def write_edr_screen(floor_bundle_path: str | Path, repo_root: str | Path) -> Path:
    root = Path(repo_root)
    analysis = build_edr_screen(floor_bundle_path, root)
    identity = {
        "schema_version": analysis["schema_version"],
        "analysis_kind": analysis["analysis_kind"],
        "source_bundle_id": analysis["source_bundle_id"],
        "source_manifest_sha256": analysis["source_manifest_sha256"],
        "input_provenance": analysis["input_provenance"],
        "half_lives": analysis["half_lives"],
        "floors": analysis["floors"],
    }
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "edr"
        / "screen"
        / f"edr_screen__{_hash(identity)[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4ChallengeError(f"incomplete EDR screen: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _edr_config(
    source: Mapping[str, Any],
    *,
    schedule: str,
    cold_start_pi: float = COLD_START_PI,
    experiment_suffix: str | None = None,
) -> ExperimentConfig:
    mapping = json.loads(json.dumps(source))
    mapping.update(
        {
            "schema_version": 20,
            "artifact_schema_version": 11,
            "metric_schema_version": 15,
            "cache_root": "cache/mnist_experiment/plan4_edr_runs",
            "experiment": (
                f"mnist_plan4-phase6-edr_{schedule}"
                if experiment_suffix is None
                else f"mnist_plan4-phase6-{experiment_suffix}_{schedule}"
            ),
        }
    )
    mapping["controller"].update(
        {
            "policy": "discounted_risk",
            "fixed_pi": cold_start_pi,
            "pi_min": PRIMARY_FLOOR,
            "pi_max": PI_MAX,
            "trend_half_life_p": 0.05,
            "risk_metric": "fisher",
            "action_half_life_steps": PRIMARY_HALF_LIFE,
            "oracle_mode": "none",
            "reference_optimum_artifact": None,
        }
    )
    return ExperimentConfig.from_mapping(mapping)


def build_edr_bundle(
    floor_bundle_path: str | Path,
    screen_path: str | Path,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    floor_bundle, floor_manifest = load_screen_bundle(floor_bundle_path)
    if floor_manifest.get("stage") != "exploratory-floor-sensitivity":
        raise Plan4ChallengeError("EDR bundle requires the floor-sensitivity bundle")
    screen = Path(screen_path)
    screen_summary = _read_json(screen / "summary.json")
    if not (screen / "COMPLETED").is_file() or not screen_summary["gate"].get(
        "passes_assumption_screen"
    ):
        raise Plan4ChallengeError("EDR assumption screen did not pass")
    status = status_rows(floor_bundle, root)
    if any(row["run_state"] != "completed" for row in status):
        raise Plan4ChallengeError("EDR controls must be complete")

    entries = []
    configs: dict[str, dict[str, Any]] = {}
    for schedule in SCHEDULES:
        members = [row for row in status if row["schedule"] == schedule]
        if {row["condition"] for row in members} != set(FLOOR_CONDITIONS):
            raise Plan4ChallengeError(f"EDR controls differ for {schedule}")
        design_hashes = {row["replica_design_hash"] for row in members}
        bundle_ids = {row["replica_bundle_id"] for row in members}
        if len(design_hashes) != 1 or len(bundle_ids) != 1:
            raise Plan4ChallengeError("EDR controls are not paired")
        for member in members:
            relative = f"configs/{schedule}/{member['condition']}.json"
            configs[relative] = _read_json(Path(member["config_path"]))
            entries.append(
                {
                    **{
                        key: member[key]
                        for key in (
                            "schedule",
                            "condition",
                            "risk_metric",
                            "config_hash",
                            "run_id",
                            "cache_root",
                            "replica_bundle_id",
                            "replica_design_hash",
                            "source_archive",
                            "source_replica_bundle",
                        )
                    },
                    "config_file": relative,
                    "reused_completed_control": True,
                }
            )
        raw_member = next(row for row in members if row["condition"] == RAW_CONDITION)
        edr = _edr_config(_read_json(Path(raw_member["config_path"])), schedule=schedule)
        if (
            replica_design_hash(edr) != next(iter(design_hashes))
            or replica_bundle_id(edr) != next(iter(bundle_ids))
        ):
            raise Plan4ChallengeError("EDR treatment broke schedule pairing")
        relative = f"configs/{schedule}/{EDR_CONDITION}.json"
        configs[relative] = edr.to_mapping()
        entries.append(
            {
                "schedule": schedule,
                "condition": EDR_CONDITION,
                "risk_metric": edr.controller.risk_metric,
                "config_file": relative,
                "config_hash": edr.config_hash,
                "run_id": edr.run_id,
                "cache_root": edr.cache_root,
                "replica_bundle_id": replica_bundle_id(edr),
                "replica_design_hash": replica_design_hash(edr),
                "source_archive": raw_member["source_archive"],
                "source_replica_bundle": raw_member["source_replica_bundle"],
                "reused_completed_control": False,
            }
        )

    identity = {
        "schema_version": PLAN4_EDR_BUNDLE_SCHEMA_VERSION,
        "phase": 6,
        "stage": "edr-predictive-development",
        "source_bundle_id": floor_manifest["bundle_id"],
        "source_manifest_sha256": _sha256(floor_bundle / "manifest.json"),
        "screen_id": screen.name,
        "screen_manifest_sha256": _sha256(screen / "manifest.json"),
        "screen_summary_sha256": _sha256(screen / "summary.json"),
        "schedules": list(SCHEDULES),
        "conditions": list(EDR_CONDITIONS),
        "primary_half_life_steps": PRIMARY_HALF_LIFE,
        "primary_pi_min": PRIMARY_FLOOR,
        "config_hashes": [entry["config_hash"] for entry in entries],
        "new_run_count": len(SCHEDULES),
        "confirmatory": False,
    }
    return (
        {
            **identity,
            "bundle_id": f"plan4-edr__{_hash(identity)[:12]}",
            "entry_count": len(entries),
            "estimated_seconds": 67.0 * len(SCHEDULES),
            "entries": entries,
        },
        configs,
    )


def write_edr_bundle(
    floor_bundle_path: str | Path,
    screen_path: str | Path,
    repo_root: str | Path,
) -> Path:
    root = Path(repo_root)
    manifest, configs = build_edr_bundle(
        floor_bundle_path,
        screen_path,
        root,
    )
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "edr"
        / "bundles"
        / manifest["bundle_id"]
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4ChallengeError(f"incomplete EDR bundle: {destination}")
        if _read_json(destination / "manifest.json") != manifest:
            raise Plan4ChallengeError("completed EDR bundle differs")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent))
    try:
        for relative, mapping in configs.items():
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(mapping, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _predictive_trajectory(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    trajectory = []
    for row in rows:
        classification = row["classification"]
        trajectory.append(
            {
                "step": int(row["step"]),
                "p": float(row["p"]),
                **{
                    field: (
                        None
                        if classification.get(field) is None
                        else float(classification[field])
                    )
                    for field in PREDICTIVE_FIELDS
                },
            }
        )
    return trajectory


def _edr_controller_summary(
    rows: Sequence[Mapping[str, Any]],
    event: np.ndarray,
) -> dict[str, Any]:
    updates = [row for row in rows if isinstance(row.get("controller_decision"), Mapping)]
    if len(updates) != event.size:
        raise Plan4ChallengeError("EDR controller trace does not align with event")
    decisions = [row["controller_decision"] for row in updates]
    applied = np.asarray([float(row["applied_pi"]) for row in decisions])
    plugin = np.asarray([float(row["plugin_pi"]) for row in decisions])
    unclipped = np.asarray(
        [
            math.nan
            if row.get("edr_unclipped_pi") is None
            else float(row["edr_unclipped_pi"])
            for row in decisions
        ]
    )
    cold = np.asarray([bool(row["cold_start_active"]) for row in decisions])
    lower = np.asarray([bool(row["lower_bound_active"]) for row in decisions])
    zero = np.asarray(
        [bool(row["edr_zero_denominator_fallback"]) for row in decisions]
    )
    event_applied = applied[event]
    post_cold = ~cold
    return {
        "action_half_life_steps": PRIMARY_HALF_LIFE,
        "event_transition_count": int(event.sum()),
        "event_applied_pi_min": float(event_applied.min()),
        "event_applied_pi_mean": float(event_applied.mean()),
        "event_applied_pi_max": float(event_applied.max()),
        "event_action_area_above_fixed_0025": float(
            np.maximum(event_applied - COLD_START_PI, 0.0).sum()
        ),
        "full_applied_pi_min": float(applied.min()),
        "full_applied_pi_mean": float(applied.mean()),
        "full_applied_pi_max": float(applied.max()),
        "full_applied_pi_total_variation": float(np.abs(np.diff(applied)).sum()),
        "full_plugin_pi_total_variation": float(np.abs(np.diff(plugin)).sum()),
        "full_unclipped_pi_mean": (
            float(unclipped[np.isfinite(unclipped)].mean())
            if np.isfinite(unclipped).any()
            else None
        ),
        "post_cold_lower_bound_fraction": (
            float(lower[post_cold].mean()) if post_cold.any() else None
        ),
        "cold_start_transition_count": int(cold.sum()),
        "zero_denominator_fallback_count": int(zero.sum()),
        "trajectory": [
            {
                "step": int(row["step"]),
                "p": float(row["p"]),
                "in_event_window": bool(event[index]),
                "applied_pi": float(decision["applied_pi"]),
                "instantaneous_plugin_pi": float(decision["plugin_pi"]),
                "unclipped_edr_pi": (
                    None
                    if decision.get("edr_unclipped_pi") is None
                    else float(decision["edr_unclipped_pi"])
                ),
                "instantaneous_old_risk": float(
                    decision["edr_instantaneous_old_risk"]
                ),
                "instantaneous_new_risk": float(
                    decision["edr_instantaneous_new_risk"]
                ),
                "old_risk_moment": float(decision["edr_old_risk_moment"]),
                "new_risk_moment": float(decision["edr_new_risk_moment"]),
                "cold_start_active": bool(decision["cold_start_active"]),
                "lower_bound_active": bool(decision["lower_bound_active"]),
                "zero_denominator_fallback": bool(
                    decision["edr_zero_denominator_fallback"]
                ),
            }
            for index, (row, decision) in enumerate(zip(updates, decisions, strict=True))
        ],
    }


def _aggregate_comparisons(
    schedules: Mapping[str, Mapping[str, Any]],
    comparison: str,
) -> dict[str, Any]:
    def finite_mean(values: Sequence[float | None]) -> float | None:
        finite = [float(value) for value in values if value is not None and math.isfinite(value)]
        return float(np.mean(finite)) if finite else None

    result = {}
    for field in PREDICTIVE_FIELDS:
        full = [
            schedule["comparisons"][comparison][field]["full_mean_difference"]
            for schedule in schedules.values()
        ]
        event = [
            schedule["comparisons"][comparison][field]["event_mean_difference"]
            for schedule in schedules.values()
        ]
        endpoint = [
            schedule["comparisons"][comparison][field]["endpoint_difference"]
            for schedule in schedules.values()
        ]
        result[field] = {
            "schedule_mean_full_difference": finite_mean(full),
            "schedule_mean_event_difference": finite_mean(event),
            "schedule_mean_endpoint_difference": finite_mean(endpoint),
            "schedule_differences_are_descriptive": True,
        }
    return result


def build_edr_analysis(
    bundle_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    bundle, manifest = load_screen_bundle(bundle_path)
    if manifest.get("stage") != "edr-predictive-development":
        raise Plan4ChallengeError("not an EDR predictive bundle")
    status = status_rows(bundle, repo_root)
    if any(row["run_state"] != "completed" for row in status):
        raise Plan4ChallengeError("all EDR predictive runs must complete")

    schedules = {}
    initial_hashes = set()
    uniform_hashes = set()
    provenance = []
    for schedule in SCHEDULES:
        members = [row for row in status if row["schedule"] == schedule]
        if {row["condition"] for row in members} != set(EDR_CONDITIONS):
            raise Plan4ChallengeError(f"EDR conditions differ for {schedule}")
        loaded = {}
        for member in members:
            run_path = Path(member["run_path"])
            metric_path = run_path / "plan3_hybrid_metrics.json"
            schedule_path = run_path / "schedule_trajectory.json"
            metrics = _read_json(metric_path)
            schedule_artifact = _read_json(schedule_path)
            loaded[member["condition"]] = metrics
            initial_hashes.add(metrics["pairing"]["initial_parameter_hash"])
            uniform_hashes.add(schedule_artifact["uniform_stream_hash"])
            provenance.append(
                {
                    "run_id": member["run_id"],
                    "schedule": schedule,
                    "condition": member["condition"],
                    "metrics_sha256": _sha256(metric_path),
                    "schedule_sha256": _sha256(schedule_path),
                }
            )
        edr_member = next(
            row for row in members if row["condition"] == EDR_CONDITION
        )
        resolved = resolve_schedule(load_config(edr_member["config_path"]).data)
        event = _event_mask(resolved)
        rows = {name: value["condition_steps"] for name, value in loaded.items()}
        conditions = {
            name: {
                "predictive": _condition_predictive_summary(condition_rows),
                "predictive_trajectory": _predictive_trajectory(condition_rows),
                "resources": loaded[name]["resource_ledger"],
            }
            for name, condition_rows in rows.items()
        }
        conditions[EDR_CONDITION]["controller"] = _edr_controller_summary(
            rows[EDR_CONDITION], event
        )
        comparisons = {
            "edr_minus_fixed_0025": _paired_predictive_summary(
                rows[EDR_CONDITION], rows["fixed-pi0025"], event
            ),
            "edr_minus_fixed_005": _paired_predictive_summary(
                rows[EDR_CONDITION], rows["fixed-pi005"], event
            ),
            "edr_minus_raw_adaptive": _paired_predictive_summary(
                rows[EDR_CONDITION], rows[RAW_CONDITION], event
            ),
        }
        schedules[schedule] = {
            "event_transition_count": int(event.sum()),
            "conditions": conditions,
            "comparisons": comparisons,
        }

    if len(initial_hashes) != 1 or len(uniform_hashes) != 1:
        raise Plan4ChallengeError("EDR development comparison lost pairing")
    aggregate = {
        comparison: _aggregate_comparisons(schedules, comparison)
        for comparison in (
            "edr_minus_fixed_0025",
            "edr_minus_fixed_005",
            "edr_minus_raw_adaptive",
        )
    }
    primary_nll = aggregate["edr_minus_fixed_0025"]["nll"][
        "schedule_mean_full_difference"
    ]
    schedule_nll = {
        schedule: value["comparisons"]["edr_minus_fixed_0025"]["nll"][
            "full_mean_difference"
        ]
        for schedule, value in schedules.items()
    }
    floor_fractions = {
        schedule: value["conditions"][EDR_CONDITION]["controller"][
            "post_cold_lower_bound_fraction"
        ]
        for schedule, value in schedules.items()
    }
    if primary_nll is not None and primary_nll >= 0.0:
        gate_classification = "failure"
        gate_basis = "predictive_underperformance_vs_fixed_0025"
    else:
        gate_classification = "requires_secondary_review"
        gate_basis = "primary_nll_improvement_requires_regression_review"
    interpretation_gate = {
        "classification": gate_classification,
        "basis": gate_basis,
        "edr_minus_fixed_0025_schedule_mean_full_nll": primary_nll,
        "schedule_full_nll_differences": schedule_nll,
        "all_schedule_full_nll_differences_worse": all(
            value is not None and value > 0.0 for value in schedule_nll.values()
        ),
        "post_cold_floor_fractions": floor_fractions,
        "floor_driven": any(
            value is not None and value >= 0.9 for value in floor_fractions.values()
        ),
        "authorizes_fresh_replica_confirmation": False,
    }
    return {
        "schema_version": PLAN4_EDR_ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "plan4_edr_predictive_development",
        "confirmatory": False,
        "independent_outer_replica_count": 1,
        "schedule_contrasts_are_descriptive": True,
        "primary_predictive_metric": "nll",
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": _sha256(bundle / "manifest.json"),
        "common_initial_parameter_hash": next(iter(initial_hashes)),
        "common_uniform_stream_hash": next(iter(uniform_hashes)),
        "input_provenance": provenance,
        "schedules": schedules,
        "aggregate_descriptive_comparisons": aggregate,
        "interpretation_gate": interpretation_gate,
    }


def write_edr_analysis(bundle_path: str | Path, repo_root: str | Path) -> Path:
    root = Path(repo_root)
    analysis = build_edr_analysis(bundle_path, root)
    identity = {
        "schema_version": analysis["schema_version"],
        "analysis_kind": analysis["analysis_kind"],
        "bundle_id": analysis["bundle_id"],
        "bundle_manifest_sha256": analysis["bundle_manifest_sha256"],
        "input_provenance": analysis["input_provenance"],
    }
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "edr"
        / "analysis"
        / f"edr_predictive__{_hash(identity)[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4ChallengeError(f"incomplete EDR analysis: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
