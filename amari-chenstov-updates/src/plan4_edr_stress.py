"""Sigmoid stress diagnostics for Plan 4 discounted-risk control."""

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

from .config import load_config
from .initialization import replica_bundle_id, replica_design_hash
from .plan4_challenge import (
    SCHEDULES,
    Plan4ChallengeError,
    _read_json,
    _sha256,
    load_screen_bundle,
    status_rows,
)
from .plan4_edr import _edr_config, _predictive_trajectory
from .plan4_edr_discovery import DISCOVERY_CONDITION
from .plan4_floor import PREDICTIVE_FIELDS, _paired_predictive_summary
from .schedules import resolve_schedule


PLAN4_EDR_STRESS_BUNDLE_SCHEMA_VERSION = 1
PLAN4_EDR_STRESS_ANALYSIS_SCHEMA_VERSION = 2
STRESS_CONDITION = DISCOVERY_CONDITION
STRESS_CONDITIONS = ("fixed-pi005", "fixed-pi0025", STRESS_CONDITION)
PRIMARY_COMPARATOR = "fixed-pi005"
PI_BASELINE = 0.05
PI_MIN = 0.01
MAX_RESPONSE_LAG = 20
HYSTERESIS_GRID_SIZE = 64
PREQUENTIAL_HALF_LIFE_STEPS = 4.0
PREQUENTIAL_SENSITIVITY_HALF_LIVES = (2.0, 4.0, 8.0)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stress_config(source: Mapping[str, Any], *, schedule: str):
    return _edr_config(
        source,
        schedule=schedule,
        cold_start_pi=PI_BASELINE,
        experiment_suffix="edr-stress",
    )


def _member_fields(member: Mapping[str, Any]) -> dict[str, Any]:
    return {
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
    }


def build_stress_bundle(
    edr_bundle_path: str | Path,
    discovery_bundle_path: str | Path,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    edr_bundle, edr_manifest = load_screen_bundle(edr_bundle_path)
    discovery_bundle, discovery_manifest = load_screen_bundle(discovery_bundle_path)
    if edr_manifest.get("stage") != "edr-predictive-development":
        raise Plan4ChallengeError("stress amendment requires the Phase 6 EDR bundle")
    if discovery_manifest.get("stage") != "edr-cold-start-discovery":
        raise Plan4ChallengeError("stress amendment requires the discovery bundle")

    edr_status = status_rows(edr_bundle, root)
    discovery_status = status_rows(discovery_bundle, root)
    if any(row["run_state"] != "completed" for row in edr_status + discovery_status):
        raise Plan4ChallengeError("stress amendment inputs must be complete")

    entries = []
    configs: dict[str, dict[str, Any]] = {}
    for schedule in SCHEDULES:
        source_members = [row for row in edr_status if row["schedule"] == schedule]
        by_condition = {row["condition"]: row for row in source_members}
        if not {"fixed-pi005", "fixed-pi0025"}.issubset(by_condition):
            raise Plan4ChallengeError(f"stress controls are incomplete for {schedule}")

        for condition in ("fixed-pi005", "fixed-pi0025"):
            member = by_condition[condition]
            relative = f"configs/{schedule}/{condition}.json"
            configs[relative] = _read_json(Path(member["config_path"]))
            entries.append(
                {
                    **_member_fields(member),
                    "config_file": relative,
                    "reused_completed_control": True,
                }
            )

        if schedule == "linear":
            candidates = [
                row
                for row in discovery_status
                if row["schedule"] == schedule
                and row["condition"] == STRESS_CONDITION
            ]
            if len(candidates) != 1:
                raise Plan4ChallengeError("linear cold-.05 discovery run is missing")
            member = candidates[0]
            config = load_config(member["config_path"])
            reused = True
            source_archive = member["source_archive"]
            source_replica_bundle = member["source_replica_bundle"]
        else:
            source = next(
                row
                for row in source_members
                if row["condition"] == "edr-fisher-pimin001-h04"
            )
            config = _stress_config(
                _read_json(Path(source["config_path"])), schedule=schedule
            )
            reused = False
            source_archive = source["source_archive"]
            source_replica_bundle = source["source_replica_bundle"]

        design_hashes = {
            by_condition["fixed-pi005"]["replica_design_hash"],
            by_condition["fixed-pi0025"]["replica_design_hash"],
            replica_design_hash(config),
        }
        bundle_ids = {
            by_condition["fixed-pi005"]["replica_bundle_id"],
            by_condition["fixed-pi0025"]["replica_bundle_id"],
            replica_bundle_id(config),
        }
        if len(design_hashes) != 1 or len(bundle_ids) != 1:
            raise Plan4ChallengeError(f"stress treatment broke pairing for {schedule}")

        relative = f"configs/{schedule}/{STRESS_CONDITION}.json"
        configs[relative] = config.to_mapping()
        entries.append(
            {
                "schedule": schedule,
                "condition": STRESS_CONDITION,
                "risk_metric": config.controller.risk_metric,
                "config_file": relative,
                "config_hash": config.config_hash,
                "run_id": config.run_id,
                "cache_root": config.cache_root,
                "replica_bundle_id": replica_bundle_id(config),
                "replica_design_hash": replica_design_hash(config),
                "source_archive": source_archive,
                "source_replica_bundle": source_replica_bundle,
                "reused_completed_control": reused,
            }
        )

    identity = {
        "schema_version": PLAN4_EDR_STRESS_BUNDLE_SCHEMA_VERSION,
        "builder_code_sha256": _sha256(Path(__file__)),
        "phase": 6,
        "stage": "edr-sigmoid-stress-diagnostics",
        "source_bundle_id": edr_manifest["bundle_id"],
        "source_manifest_sha256": _sha256(edr_bundle / "manifest.json"),
        "discovery_bundle_id": discovery_manifest["bundle_id"],
        "discovery_manifest_sha256": _sha256(discovery_bundle / "manifest.json"),
        "schedules": list(SCHEDULES),
        "conditions": list(STRESS_CONDITIONS),
        "cold_start_pi": PI_BASELINE,
        "pi_min": PI_MIN,
        "action_half_life_steps": 4.0,
        "device": "cpu",
        "config_hashes": [entry["config_hash"] for entry in entries],
        "new_run_count": len(SCHEDULES) - 1,
        "confirmatory": False,
    }
    return (
        {
            **identity,
            "bundle_id": f"plan4-edr-stress__{_hash(identity)[:12]}",
            "entry_count": len(entries),
            "estimated_seconds": 67.0 * (len(SCHEDULES) - 1),
            "entries": entries,
        },
        configs,
    )


def write_stress_bundle(
    edr_bundle_path: str | Path,
    discovery_bundle_path: str | Path,
    repo_root: str | Path,
) -> Path:
    root = Path(repo_root)
    manifest, configs = build_stress_bundle(
        edr_bundle_path, discovery_bundle_path, root
    )
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "edr"
        / "stress"
        / "bundles"
        / manifest["bundle_id"]
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4ChallengeError(f"incomplete stress bundle: {destination}")
        if _read_json(destination / "manifest.json") != manifest:
            raise Plan4ChallengeError("completed stress bundle differs")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent)
    )
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


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 3 or float(np.ptp(left)) == 0.0 or float(np.ptp(right)) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _lag_profile(
    driver: np.ndarray,
    response: np.ndarray,
    *,
    maximum_lag: int = MAX_RESPONSE_LAG,
) -> dict[str, Any]:
    if driver.shape != response.shape:
        raise ValueError("lagged driver and response must align")
    rows = []
    for lag in range(min(maximum_lag, max(0, driver.size - 3)) + 1):
        left = driver if lag == 0 else driver[:-lag]
        right = response if lag == 0 else response[lag:]
        rows.append({"lag": lag, "correlation": _correlation(left, right)})
    finite = [row for row in rows if row["correlation"] is not None]
    best = max(finite, key=lambda row: row["correlation"]) if finite else None
    return {
        "maximum_lag": maximum_lag,
        "best_lag": None if best is None else best["lag"],
        "best_correlation": None if best is None else best["correlation"],
        "best_lag_hits_search_boundary": bool(
            best is not None and best["lag"] == maximum_lag
        ),
        "trajectory": rows,
    }


def _unavailable_lag_profile(reason: str) -> dict[str, Any]:
    return {
        "maximum_lag": MAX_RESPONSE_LAG,
        "best_lag": None,
        "best_correlation": None,
        "best_lag_hits_search_boundary": False,
        "unavailable_reason": reason,
        "trajectory": [],
    }


def _peak_delay(driver: np.ndarray, response: np.ndarray, steps: np.ndarray) -> int | None:
    if (
        driver.size == 0
        or driver.shape != response.shape
        or driver.shape != steps.shape
        or float(np.ptp(driver)) == 0.0
        or float(np.ptp(response)) == 0.0
    ):
        return None
    return int(steps[int(np.argmax(response))] - steps[int(np.argmax(driver))])


def _collapsed_branch(speed: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(speed)
    speed = speed[order]
    action = action[order]
    unique, inverse = np.unique(speed, return_inverse=True)
    sums = np.zeros(unique.size, dtype=np.float64)
    counts = np.zeros(unique.size, dtype=np.float64)
    np.add.at(sums, inverse, action)
    np.add.at(counts, inverse, 1.0)
    return unique, sums / counts


def _hysteresis(speed: np.ndarray, action: np.ndarray, live: np.ndarray) -> dict[str, Any]:
    if speed.shape != action.shape or speed.shape != live.shape:
        raise ValueError("hysteresis inputs must align")
    live_indices = np.flatnonzero(live)
    if live_indices.size < 5 or float(np.ptp(speed[live])) == 0.0:
        return {"available": False, "reason": "no_resolved_speed_bend"}
    peak = int(live_indices[int(np.argmax(speed[live]))])
    meaningful = max(1e-12, float(speed[live].max()) * 1e-6)
    entry = live & (np.arange(speed.size) <= peak) & (speed > meaningful)
    exit_ = live & (np.arange(speed.size) >= peak) & (speed > meaningful)
    if entry.sum() < 2 or exit_.sum() < 2:
        return {"available": False, "reason": "insufficient_branch_points"}
    entry_speed, entry_action = _collapsed_branch(speed[entry], action[entry])
    exit_speed, exit_action = _collapsed_branch(speed[exit_], action[exit_])
    lower = max(float(entry_speed.min()), float(exit_speed.min()))
    upper = min(float(entry_speed.max()), float(exit_speed.max()))
    if not upper > lower:
        return {"available": False, "reason": "no_common_speed_support"}
    grid = np.linspace(lower, upper, HYSTERESIS_GRID_SIZE)
    entry_grid = np.interp(grid, entry_speed, entry_action)
    exit_grid = np.interp(grid, exit_speed, exit_action)
    gap = exit_grid - entry_grid
    signed_area = float(np.trapezoid(gap, grid))
    absolute_area = float(np.trapezoid(np.abs(gap), grid))
    return {
        "available": True,
        "peak_transition": peak,
        "meaningful_speed_floor": meaningful,
        "entry_point_count": int(entry.sum()),
        "exit_point_count": int(exit_.sum()),
        "common_speed_min": lower,
        "common_speed_max": upper,
        "common_speed_span": upper - lower,
        "common_speed_fraction_of_meaningful_range": (
            (upper - lower) / (float(speed[live].max()) - meaningful)
        ),
        "grid_size": HYSTERESIS_GRID_SIZE,
        "signed_loop_area": signed_area,
        "absolute_loop_area": absolute_area,
        "mean_signed_branch_gap": signed_area / (upper - lower),
        "mean_absolute_branch_gap": absolute_area / (upper - lower),
        "trajectory": [
            {
                "speed": float(value),
                "entry_pi": float(entry_value),
                "exit_pi": float(exit_value),
                "exit_minus_entry_pi": float(gap_value),
            }
            for value, entry_value, exit_value, gap_value in zip(
                grid, entry_grid, exit_grid, gap, strict=True
            )
        ],
    }


def _tail_summary(action: np.ndarray, steps: np.ndarray, live: np.ndarray) -> dict[str, Any]:
    live_action = action[live]
    live_steps = steps[live]
    if live_action.size < 10:
        raise Plan4ChallengeError("stress trajectory has fewer than ten live actions")
    tail_action = live_action[-10:]
    tail_steps = live_steps[-10:]
    slope = float(np.polyfit(tail_steps, tail_action, 1)[0])
    return {
        "last_ten_pi_mean": float(tail_action.mean()),
        "last_ten_pi_standard_deviation": float(tail_action.std(ddof=1)),
        "last_ten_pi_slope_per_step": slope,
        "peak_live_pi": float(live_action.max()),
        "live_area_above_fixed_005": float(
            np.maximum(live_action - PI_BASELINE, 0.0).sum()
        ),
        "live_area_below_fixed_005": float(
            np.maximum(PI_BASELINE - live_action, 0.0).sum()
        ),
    }


def _prequential_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    half_life_steps: float = PREQUENTIAL_HALF_LIFE_STEPS,
) -> dict[str, Any]:
    """Compare each lagged Fisher-risk forecast with its realized residual energy."""

    if not math.isfinite(half_life_steps) or half_life_steps <= 0.0:
        raise ValueError("prequential half-life must be finite and positive")
    updates = [
        row
        for row in rows
        if isinstance(row.get("controller_decision"), Mapping)
    ]
    if not updates:
        raise Plan4ChallengeError("prequential calibration requires controller updates")

    gain = 1.0 - 2.0 ** (-1.0 / half_life_steps)
    observed_moment = 0.0
    predicted_moment = 0.0
    trajectory = []
    for row in updates:
        decision = row["controller_decision"]
        acceptance = row.get("controller_acceptance")
        if not isinstance(acceptance, Mapping):
            raise Plan4ChallengeError(
                "prequential calibration requires controller acceptances"
            )
        if decision.get("risk_metric") != "fisher":
            raise Plan4ChallengeError(
                "prequential calibration requires Fisher-risk artifacts"
            )
        observed = float(acceptance["residual_squared"])
        scale = float(acceptance["scale_observation"])
        lagged_scale = float(decision["trace_estimate"])
        predicted = scale * lagged_scale
        if (
            not all(math.isfinite(value) for value in (observed, scale, lagged_scale))
            or min(observed, scale, lagged_scale) < 0.0
        ):
            raise Plan4ChallengeError(
                "prequential calibration inputs must be finite and nonnegative"
            )

        observed_moment = (1.0 - gain) * observed_moment + gain * observed
        predicted_moment = (1.0 - gain) * predicted_moment + gain * predicted
        raw_ratio = observed / predicted if predicted > 0.0 else None
        ema_ratio = (
            observed_moment / predicted_moment
            if predicted_moment > 0.0
            else None
        )
        trajectory.append(
            {
                "step": int(row["step"]),
                "p": float(row["p"]),
                "cold_start_active": bool(decision["cold_start_active"]),
                "observed_residual_risk_energy": observed,
                "residual_scale_coefficient": scale,
                "lagged_uncertainty_scale_estimate": lagged_scale,
                "predicted_residual_risk_energy": predicted,
                "raw_observed_to_predicted_ratio": raw_ratio,
                "ema_observed_risk_energy": observed_moment,
                "ema_predicted_risk_energy": predicted_moment,
                "ema_observed_to_predicted_ratio": ema_ratio,
                "ema_log_calibration_ratio": (
                    math.log(ema_ratio)
                    if ema_ratio is not None and ema_ratio > 0.0
                    else None
                ),
            }
        )

    live = [row for row in trajectory if not row["cold_start_active"]]
    if len(live) < 10:
        raise Plan4ChallengeError(
            "prequential calibration has fewer than ten live transitions"
        )
    raw = np.asarray(
        [
            math.nan
            if row["raw_observed_to_predicted_ratio"] is None
            else row["raw_observed_to_predicted_ratio"]
            for row in live
        ],
        dtype=np.float64,
    )
    ema = np.asarray(
        [row["ema_observed_to_predicted_ratio"] for row in live],
        dtype=np.float64,
    )
    if not np.isfinite(ema).all() or np.any(ema <= 0.0):
        raise Plan4ChallengeError("live prequential calibration ratios must be positive")
    log_ema = np.log(ema)
    return {
        "strictly_prequential": True,
        "paired_comparator_used": False,
        "forecast_semantics": (
            "scale_observation_times_lagged_fisher_weighted_covariance_dimension"
        ),
        "observed_semantics": "accepted_residual_fisher_risk_energy",
        "ema_half_life_accepted_updates": half_life_steps,
        "ema_gain": gain,
        "cold_transition_count": len(trajectory) - len(live),
        "live_transition_count": len(live),
        "raw_ratio_median_live": float(np.nanmedian(raw)),
        "raw_ratio_90th_percentile_live": float(np.nanquantile(raw, 0.9)),
        "ema_ratio_mean_live": float(ema.mean()),
        "ema_ratio_minimum_live": float(ema.min()),
        "ema_ratio_maximum_live": float(ema.max()),
        "ema_ratio_last_ten_mean": float(ema[-10:].mean()),
        "mean_absolute_log_ema_ratio_live": float(np.abs(log_ema).mean()),
        "last_ten_absolute_log_ema_ratio_mean": float(
            np.abs(log_ema[-10:]).mean()
        ),
        "trajectory": trajectory,
    }


def _controller_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    delta_p: Sequence[float],
    *,
    speed_has_bend: bool = True,
) -> dict[str, Any]:
    updates = [row for row in rows if isinstance(row.get("controller_decision"), Mapping)]
    if len(updates) != len(delta_p):
        raise Plan4ChallengeError("stress controller trace does not align with schedule")
    decisions = [row["controller_decision"] for row in updates]
    steps = np.asarray([int(row["step"]) for row in updates], dtype=np.int64)
    p = np.asarray([float(row["p"]) for row in updates], dtype=np.float64)
    speed = np.abs(np.asarray(delta_p, dtype=np.float64))
    applied = np.asarray([float(row["applied_pi"]) for row in decisions])
    instantaneous = np.asarray([float(row["plugin_pi"]) for row in decisions])
    unclipped = np.asarray(
        [
            math.nan
            if row.get("edr_unclipped_pi") is None
            else float(row["edr_unclipped_pi"])
            for row in decisions
        ]
    )
    old = np.asarray([float(row["edr_old_risk_moment"]) for row in decisions])
    new = np.asarray([float(row["edr_new_risk_moment"]) for row in decisions])
    signal = np.asarray([float(row["signal_energy"]) for row in decisions])
    cold = np.asarray([bool(row["cold_start_active"]) for row in decisions])
    lower = np.asarray([bool(row["lower_bound_active"]) for row in decisions])
    live = ~cold
    curvature = old + new
    minimizer = np.divide(
        old,
        curvature,
        out=np.full_like(old, np.nan),
        where=curvature > 0.0,
    )
    baseline_risk = (1.0 - PI_BASELINE) ** 2 * old + PI_BASELINE**2 * new
    applied_risk = (1.0 - applied) ** 2 * old + applied**2 * new
    opportunity = baseline_risk - applied_risk
    if float(opportunity.min()) < -1e-12:
        raise Plan4ChallengeError("EDR action increased its discounted risk versus fixed .05")
    edr_tv = float(np.abs(np.diff(applied[live])).sum())
    instantaneous_tv = float(np.abs(np.diff(instantaneous[live])).sum())
    finite_unclipped = live & np.isfinite(unclipped)
    clipping_pressure = np.maximum(PI_MIN - unclipped[finite_unclipped], 0.0)

    speed_lag = (
        _lag_profile(speed[live], applied[live])
        if speed_has_bend
        else _unavailable_lag_profile("constant_speed_schedule")
    )
    signal_lag = _lag_profile(signal[live], applied[live])
    recommendation = np.isfinite(unclipped)
    recommendation_speed_lag = (
        _lag_profile(speed[recommendation], unclipped[recommendation])
        if speed_has_bend
        else _unavailable_lag_profile("constant_speed_schedule")
    )
    recommendation_signal_lag = _lag_profile(
        signal[recommendation], unclipped[recommendation]
    )
    result = {
        "cold_transition_count": int(cold.sum()),
        "live_transition_count": int(live.sum()),
        "risk": {
            "curvature_mean_live": float(curvature[live].mean()),
            "curvature_max_live": float(curvature[live].max()),
            "estimated_opportunity_sum_live": float(opportunity[live].sum()),
            "estimated_opportunity_mean_live": float(opportunity[live].mean()),
        },
        "noise_attenuation": {
            "edr_total_variation_live": edr_tv,
            "instantaneous_total_variation_live": instantaneous_tv,
            "total_variation_ratio": (
                edr_tv / instantaneous_tv if instantaneous_tv > 0.0 else None
            ),
            "mean_absolute_instantaneous_minus_edr_live": float(
                np.abs(instantaneous[live] - applied[live]).mean()
            ),
        },
        "response": {
            "speed": speed_lag,
            "signal_energy": signal_lag,
            "speed_peak_delay_steps": _peak_delay(
                speed[live], applied[live], steps[live]
            ) if speed_has_bend else None,
            "signal_peak_delay_steps": _peak_delay(
                signal[live], applied[live], steps[live]
            ),
            "unclipped_recommendation": {
                "speed": recommendation_speed_lag,
                "signal_energy": recommendation_signal_lag,
                "speed_peak_delay_steps": _peak_delay(
                    speed[recommendation],
                    unclipped[recommendation],
                    steps[recommendation],
                ) if speed_has_bend else None,
                "signal_peak_delay_steps": _peak_delay(
                    signal[recommendation],
                    unclipped[recommendation],
                    steps[recommendation],
                ),
            },
        },
        "hysteresis": {
            "applied_live_action": (
                _hysteresis(speed, applied, live)
                if speed_has_bend
                else {"available": False, "reason": "constant_speed_schedule"}
            ),
            "unclipped_recommendation": (
                _hysteresis(speed, unclipped, recommendation)
                if speed_has_bend
                else {"available": False, "reason": "constant_speed_schedule"}
            ),
        },
        "settling": _tail_summary(applied, steps, live),
        "boundary": {
            "post_cold_lower_bound_fraction": float(
                lower[live].mean()
            ),
            "mean_post_cold_clipping_pressure": (
                float(clipping_pressure.mean()) if clipping_pressure.size else None
            ),
            "maximum_post_cold_clipping_pressure": (
                float(clipping_pressure.max()) if clipping_pressure.size else None
            ),
        },
        "trajectory": [
            {
                "step": int(step),
                "p": float(p_value),
                "speed": float(speed_value),
                "cold_start_active": bool(cold_value),
                "instantaneous_pi": float(instantaneous_value),
                "edr_unclipped_pi": (
                    None if not math.isfinite(unclipped_value) else float(unclipped_value)
                ),
                "applied_pi": float(applied_value),
                "old_risk_moment": float(old_value),
                "new_risk_moment": float(new_value),
                "risk_curvature": float(curvature_value),
                "unconstrained_risk_minimizer": (
                    None if not math.isfinite(minimizer_value) else float(minimizer_value)
                ),
                "signal_energy": float(signal_value),
                "estimated_risk_opportunity_vs_fixed_005": float(opportunity_value),
            }
            for (
                step,
                p_value,
                speed_value,
                cold_value,
                instantaneous_value,
                unclipped_value,
                applied_value,
                old_value,
                new_value,
                curvature_value,
                minimizer_value,
                signal_value,
                opportunity_value,
            ) in zip(
                steps,
                p,
                speed,
                cold,
                instantaneous,
                unclipped,
                applied,
                old,
                new,
                curvature,
                minimizer,
                signal,
                opportunity,
                strict=True,
            )
        ],
    }
    return result


def _predictive_alignment(
    edr_rows: Sequence[Mapping[str, Any]],
    fixed_rows: Sequence[Mapping[str, Any]],
    controller: Mapping[str, Any],
) -> dict[str, Any]:
    if len(edr_rows) != len(fixed_rows) or len(edr_rows) != len(
        controller["trajectory"]
    ) + 1:
        raise Plan4ChallengeError("stress predictive trajectories do not align")
    opportunity = np.asarray(
        [
            row["estimated_risk_opportunity_vs_fixed_005"]
            for row in controller["trajectory"]
        ],
        dtype=np.float64,
    )
    fixed_nll = np.asarray(
        [float(row["classification"]["nll"]) for row in fixed_rows[1:]]
    )
    edr_nll = np.asarray(
        [float(row["classification"]["nll"]) for row in edr_rows[1:]]
    )
    nll_gain = fixed_nll - edr_nll
    if opportunity.shape != nll_gain.shape:
        raise Plan4ChallengeError("stress risk and predictive trajectories do not align")
    cumulative_risk = np.cumsum(opportunity)
    cumulative_nll = np.cumsum(nll_gain)
    return {
        "same_step_risk_nll_gain_correlation": _correlation(opportunity, nll_gain),
        "cumulative_risk_nll_gain_correlation": _correlation(
            cumulative_risk, cumulative_nll
        ),
        "estimated_risk_opportunity_sum": float(cumulative_risk[-1]),
        "realized_nll_gain_sum": float(cumulative_nll[-1]),
        "trajectory": [
            {
                "step": int(edr["step"]),
                "p": float(edr["p"]),
                "estimated_risk_opportunity": float(risk),
                "cumulative_estimated_risk_opportunity": float(cumulative_risk_value),
                "realized_nll_gain": float(gain),
                "cumulative_realized_nll_gain": float(cumulative_nll_value),
            }
            for edr, risk, cumulative_risk_value, gain, cumulative_nll_value in zip(
                edr_rows[1:],
                opportunity,
                cumulative_risk,
                nll_gain,
                cumulative_nll,
                strict=True,
            )
        ],
    }


def _prequential_validation(schedules: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for schedule in SCHEDULES:
        result = schedules[schedule]
        calibration = result["prequential_calibration"]
        hysteresis = result["controller"]["hysteresis"][
            "unclipped_recommendation"
        ]
        rows.append(
            {
                "schedule": schedule,
                "mean_absolute_log_ema_ratio_live": calibration[
                    "mean_absolute_log_ema_ratio_live"
                ],
                "last_ten_absolute_log_ema_ratio_mean": calibration[
                    "last_ten_absolute_log_ema_ratio_mean"
                ],
                "recommendation_mean_absolute_hysteresis_gap": (
                    hysteresis.get("mean_absolute_branch_gap")
                ),
                "mean_nll_regression_vs_fixed_005": result["comparisons"][
                    "edr_minus_fixed_005"
                ]["nll"]["full_mean_difference"],
            }
        )

    sigmoid = [row for row in rows if row["schedule"] != "linear"]
    calibration = np.asarray(
        [row["mean_absolute_log_ema_ratio_live"] for row in sigmoid]
    )
    tail = np.asarray(
        [row["last_ten_absolute_log_ema_ratio_mean"] for row in sigmoid]
    )
    hysteresis = np.asarray(
        [row["recommendation_mean_absolute_hysteresis_gap"] for row in sigmoid]
    )
    nll = np.asarray(
        [row["mean_nll_regression_vs_fixed_005"] for row in sigmoid]
    )
    return {
        "diagnostic_construction_uses_one_trajectory": True,
        "paired_fixed_005_used_only_for_retrospective_validation": True,
        "schedule_level_correlations_are_descriptive": True,
        "sigmoid_schedule_count": len(sigmoid),
        "sigmoid_mean_log_miscalibration_hysteresis_correlation": _correlation(
            calibration, hysteresis
        ),
        "sigmoid_mean_log_miscalibration_nll_regression_correlation": _correlation(
            calibration, nll
        ),
        "sigmoid_tail_log_miscalibration_nll_regression_correlation": _correlation(
            tail, nll
        ),
        "rows": rows,
    }


def build_stress_analysis(
    bundle_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    bundle, manifest = load_screen_bundle(bundle_path)
    if manifest.get("stage") != "edr-sigmoid-stress-diagnostics":
        raise Plan4ChallengeError("not an EDR sigmoid stress bundle")
    status = status_rows(bundle, repo_root)
    if any(row["run_state"] != "completed" for row in status):
        raise Plan4ChallengeError("all EDR stress conditions must complete")

    schedules = {}
    provenance = []
    common_initial_hashes = set()
    common_uniform_hashes = set()
    for schedule in SCHEDULES:
        members = [row for row in status if row["schedule"] == schedule]
        if {row["condition"] for row in members} != set(STRESS_CONDITIONS):
            raise Plan4ChallengeError(f"stress conditions differ for {schedule}")
        loaded = {}
        schedule_hashes = set()
        initial_hashes = set()
        uniform_hashes = set()
        for member in members:
            run_path = Path(member["run_path"])
            metric_path = run_path / "plan3_hybrid_metrics.json"
            schedule_path = run_path / "schedule_trajectory.json"
            metrics = _read_json(metric_path)
            schedule_artifact = _read_json(schedule_path)
            loaded[member["condition"]] = metrics
            initial_hashes.add(metrics["pairing"]["initial_parameter_hash"])
            uniform_hashes.add(schedule_artifact["uniform_stream_hash"])
            schedule_hashes.add(schedule_artifact["schedule_hash"])
            provenance.append(
                {
                    "run_id": member["run_id"],
                    "schedule": schedule,
                    "condition": member["condition"],
                    "metrics_sha256": _sha256(metric_path),
                    "schedule_sha256": _sha256(schedule_path),
                }
            )
        if len(initial_hashes) != 1 or len(uniform_hashes) != 1 or len(schedule_hashes) != 1:
            raise Plan4ChallengeError(f"stress comparison lost pairing for {schedule}")
        common_initial_hashes.update(initial_hashes)
        common_uniform_hashes.update(uniform_hashes)

        stress_member = next(
            row for row in members if row["condition"] == STRESS_CONDITION
        )
        resolved = resolve_schedule(load_config(stress_member["config_path"]).data)
        rows = {name: value["condition_steps"] for name, value in loaded.items()}
        edr_rows = rows[STRESS_CONDITION]
        fixed_rows = rows[PRIMARY_COMPARATOR]
        controller = _controller_diagnostics(
            edr_rows,
            resolved.delta_p_values[1:],
            speed_has_bend=resolved.kind != "linear",
        )
        calibration = _prequential_calibration(edr_rows)
        calibration_sensitivity = {}
        for half_life in PREQUENTIAL_SENSITIVITY_HALF_LIVES:
            value = (
                calibration
                if half_life == PREQUENTIAL_HALF_LIFE_STEPS
                else _prequential_calibration(
                    edr_rows, half_life_steps=half_life
                )
            )
            calibration_sensitivity[str(int(half_life))] = {
                key: item for key, item in value.items() if key != "trajectory"
            }
        cold_steps = [
            row["step"]
            for row in controller["trajectory"]
            if row["cold_start_active"]
        ]
        last_cold_outcome = max(cold_steps) + 1 if cold_steps else 0
        parameter_parity = all(
            left["parameter_hash"] == right["parameter_hash"]
            for left, right in zip(edr_rows, fixed_rows, strict=True)
            if int(left["step"]) <= last_cold_outcome
        )
        prediction_parity = all(
            left["classification"] == right["classification"]
            for left, right in zip(edr_rows, fixed_rows, strict=True)
            if int(left["step"]) <= last_cold_outcome
        )
        if not parameter_parity or not prediction_parity:
            raise Plan4ChallengeError(f"cold-prefix parity failed for {schedule}")

        event = np.ones(len(edr_rows) - 1, dtype=bool)
        schedules[schedule] = {
            "steepness": resolved.steepness,
            "max_speed_transition": resolved.max_speed_transition,
            "last_cold_outcome_step": last_cold_outcome,
            "cold_prefix_parameter_parity_with_fixed_005": parameter_parity,
            "cold_prefix_prediction_parity_with_fixed_005": prediction_parity,
            "controller": controller,
            "prequential_calibration": calibration,
            "prequential_calibration_half_life_sensitivity": (
                calibration_sensitivity
            ),
            "alignment": _predictive_alignment(edr_rows, fixed_rows, controller),
            "conditions": {
                name: {"predictive_trajectory": _predictive_trajectory(value)}
                for name, value in rows.items()
            },
            "comparisons": {
                "edr_minus_fixed_005": _paired_predictive_summary(
                    edr_rows, fixed_rows, event
                ),
                "edr_minus_fixed_0025_hindsight": _paired_predictive_summary(
                    edr_rows, rows["fixed-pi0025"], event
                ),
            },
        }

    if len(common_initial_hashes) != 1 or len(common_uniform_hashes) != 1:
        raise Plan4ChallengeError("stress schedule family lost shared provenance")
    return {
        "schema_version": PLAN4_EDR_STRESS_ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "plan4_edr_sigmoid_stress_diagnostics",
        "confirmatory": False,
        "independent_outer_replica_count": 1,
        "schedule_contrasts_are_descriptive": True,
        "binary_health_thresholds_predeclared": False,
        "primary_comparator": PRIMARY_COMPARATOR,
        "fixed_0025_is_hindsight_only": True,
        "primary_predictive_metric": "nll",
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": _sha256(bundle / "manifest.json"),
        "analysis_code_sha256": _sha256(Path(__file__)),
        "common_initial_parameter_hash": next(iter(common_initial_hashes)),
        "common_uniform_stream_hash": next(iter(common_uniform_hashes)),
        "input_provenance": provenance,
        "schedules": schedules,
        "prequential_validation": _prequential_validation(schedules),
    }


def write_stress_analysis(bundle_path: str | Path, repo_root: str | Path) -> Path:
    root = Path(repo_root)
    analysis = build_stress_analysis(bundle_path, root)
    identity = {
        "schema_version": analysis["schema_version"],
        "analysis_kind": analysis["analysis_kind"],
        "bundle_id": analysis["bundle_id"],
        "bundle_manifest_sha256": analysis["bundle_manifest_sha256"],
        "analysis_code_sha256": analysis["analysis_code_sha256"],
        "input_provenance": analysis["input_provenance"],
    }
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "edr"
        / "stress"
        / "analysis"
        / f"edr_stress__{_hash(identity)[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4ChallengeError(f"incomplete stress analysis: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent)
    )
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
