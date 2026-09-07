"""Pure coefficient attribution for the Plan 7 movement-premium audit."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any


def transition_key(
    schedule: str, current_angle: float, next_angle: float
) -> tuple[str, str, str]:
    values = (current_angle, next_angle)
    if not schedule or any(not math.isfinite(value) for value in values):
        raise ValueError("transition identity must be finite and named")
    return schedule, float(current_angle).hex(), float(next_angle).hex()


def marginal_pi(
    signal: float,
    q: float,
    old_covariance_shape: float,
    new_covariance_shape: float,
    batch_size: int,
) -> float:
    values = (signal, q, old_covariance_shape, new_covariance_shape)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("marginal-risk coefficients must be finite and nonnegative")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    numerator = signal + q * old_covariance_shape
    denominator = numerator + new_covariance_shape / batch_size
    if denominator <= 0.0:
        raise ValueError("marginal-risk denominator must be positive")
    return numerator / denominator


def movement_premium(
    signal: float, covariance_shape: float, batch_size: int
) -> float:
    if (
        not math.isfinite(signal)
        or signal < 0.0
        or not math.isfinite(covariance_shape)
        or covariance_shape <= 0.0
    ):
        raise ValueError("movement-premium coefficients are invalid")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return batch_size * signal / covariance_shape


def log_ratio(
    numerator: float, denominator: float, *, floor: float
) -> float | None:
    if not all(math.isfinite(value) for value in (numerator, denominator, floor)):
        raise ValueError("log-ratio inputs must be finite")
    if floor <= 0.0 or numerator < 0.0 or denominator < 0.0:
        raise ValueError("log-ratio inputs have invalid signs")
    if numerator <= floor or denominator <= floor:
        return None
    return math.log(numerator / denominator)


def _mean(rows: Sequence[dict[str, Any]], name: str) -> float:
    return statistics.fmean(float(row[name]) for row in rows)


def _mae(
    rows: Sequence[dict[str, Any]], name: str, target: str
) -> float:
    return statistics.fmean(
        abs(float(row[name]) - float(row[target])) for row in rows
    )


def _signed_error(
    rows: Sequence[dict[str, Any]], name: str, target: str
) -> float:
    return statistics.fmean(
        float(row[name]) - float(row[target]) for row in rows
    )


def _error_summary(
    rows: Sequence[dict[str, Any]], *, labels: dict[str, Any]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("error summary requires at least one row")
    return {
        **labels,
        "eligible_transition_count": len(rows),
        "applied_pi_signed_error": _signed_error(
            rows, "applied_pi", "population_marginal_pi"
        ),
        "applied_pi_mae": _mae(
            rows, "applied_pi", "population_marginal_pi"
        ),
        "online_instantaneous_pi_signed_error": _signed_error(
            rows, "online_instantaneous_pi", "population_marginal_pi"
        ),
        "online_instantaneous_pi_mae": _mae(
            rows, "online_instantaneous_pi", "population_marginal_pi"
        ),
        "mean_smoothing_shift": statistics.fmean(
            float(row["applied_pi"])
            - float(row["online_instantaneous_pi"])
            for row in rows
        ),
    }


def summarize_error_breakdowns(
    rows: Sequence[dict[str, Any]], *, reversal_window_steps: int
) -> dict[str, list[dict[str, Any]]]:
    if reversal_window_steps < 1:
        raise ValueError("reversal_window_steps must be positive")
    eligible = [row for row in rows if row["eligible"]]
    legs = []
    leg_keys = sorted(
        {
            (str(row["design"]), str(row["schedule"]), int(row["leg_id"]))
            for row in eligible
        }
    )
    for design, schedule, leg_id in leg_keys:
        selected = [
            row
            for row in eligible
            if row["design"] == design
            and row["schedule"] == schedule
            and int(row["leg_id"]) == leg_id
        ]
        legs.append(
            _error_summary(
                selected,
                labels={
                    "design": design,
                    "schedule": schedule,
                    "leg_id": leg_id,
                },
            )
        )

    reversals = []
    reversal_keys = sorted(
        {
            (
                str(row["design"]),
                str(row["schedule"]),
                int(row["reversal_step"]),
            )
            for row in eligible
            if row["reversal_window"]
        }
    )
    for design, schedule, reversal_step in reversal_keys:
        selected = [
            row
            for row in eligible
            if row["design"] == design
            and row["schedule"] == schedule
            and row["reversal_window"]
            and int(row["reversal_step"]) == reversal_step
        ]
        reversals.append(
            _error_summary(
                selected,
                labels={
                    "design": design,
                    "schedule": schedule,
                    "reversal_step": reversal_step,
                    "window_steps": reversal_window_steps,
                },
            )
        )
    return {"legs": legs, "reversal_windows": reversals}


def summarize_movement_rows(
    rows: Sequence[dict[str, Any]], *, attribution_tolerance: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows or attribution_tolerance <= 0.0:
        raise ValueError("movement summary requires rows and a positive tolerance")
    groups = sorted({(str(row["design"]), str(row["schedule"])) for row in rows})
    summaries = []
    for design, schedule in groups:
        selected = [
            row
            for row in rows
            if row["design"] == design and row["schedule"] == schedule
        ]
        eligible = [row for row in selected if row["eligible"]]
        if not eligible:
            raise ValueError("movement summary has no eligible post-cold rows")
        signal_logs = [
            float(row["signal_log_ratio"])
            for row in eligible
            if row["signal_log_ratio"] is not None
        ]
        scale_logs = [
            float(row["scale_log_ratio"])
            for row in eligible
            if row["scale_log_ratio"] is not None
        ]
        online_error = _mae(
            eligible, "online_instantaneous_pi", "population_marginal_pi"
        )
        replace_signal_error = _mae(
            eligible,
            "population_signal_online_scale_pi",
            "population_marginal_pi",
        )
        replace_scale_error = _mae(
            eligible,
            "online_signal_population_scale_pi",
            "population_marginal_pi",
        )
        summaries.append(
            {
                "design": design,
                "schedule": schedule,
                "transition_count": len(selected),
                "eligible_transition_count": len(eligible),
                "cold_start_count": sum(bool(row["cold_start"]) for row in selected),
                "fallback_count": sum(bool(row["fallback"]) for row in selected),
                "clipped_count": sum(bool(row["clipped"]) for row in selected),
                "precision_pass_fraction": statistics.fmean(
                    float(row["oracle_precision_pass"]) for row in eligible
                ),
                "mean_applied_pi": _mean(eligible, "applied_pi"),
                "mean_online_instantaneous_pi": _mean(
                    eligible, "online_instantaneous_pi"
                ),
                "mean_covariance_only_pi": _mean(
                    eligible, "tracked_q_covariance_pi"
                ),
                "mean_population_marginal_pi": _mean(
                    eligible, "population_marginal_pi"
                ),
                "mean_online_signal": _mean(eligible, "online_signal"),
                "mean_population_signal": _mean(eligible, "population_signal"),
                "median_signal_ratio": (
                    None if not signal_logs else math.exp(statistics.median(signal_logs))
                ),
                "mean_online_scale": _mean(eligible, "online_scale"),
                "mean_population_new_scale": _mean(
                    eligible, "population_new_scale"
                ),
                "median_scale_ratio": (
                    None if not scale_logs else math.exp(statistics.median(scale_logs))
                ),
                "mean_discounted_movement_premium": _mean(
                    eligible, "discounted_movement_premium"
                ),
                "mean_population_movement_premium": _mean(
                    eligible, "population_movement_premium"
                ),
                "applied_pi_mae": _mae(
                    eligible, "applied_pi", "population_marginal_pi"
                ),
                "applied_pi_signed_error": _signed_error(
                    eligible, "applied_pi", "population_marginal_pi"
                ),
                "online_instantaneous_pi_mae": online_error,
                "online_instantaneous_pi_signed_error": _signed_error(
                    eligible,
                    "online_instantaneous_pi",
                    "population_marginal_pi",
                ),
                "population_signal_online_scale_pi_mae": replace_signal_error,
                "online_signal_population_scale_pi_mae": replace_scale_error,
                "signal_replacement_error_reduction": online_error
                - replace_signal_error,
                "scale_replacement_error_reduction": online_error
                - replace_scale_error,
                "mean_smoothing_shift": statistics.fmean(
                    float(row["applied_pi"])
                    - float(row["online_instantaneous_pi"])
                    for row in eligible
                ),
            }
        )

    signal_repairs = all(
        row["population_signal_online_scale_pi_mae"] <= attribution_tolerance
        for row in summaries
    )
    scale_repairs = all(
        row["online_signal_population_scale_pi_mae"] <= attribution_tolerance
        for row in summaries
    )
    if signal_repairs and not scale_repairs:
        primary = "numerator_dominated"
    elif scale_repairs and not signal_repairs:
        primary = "scale_dominated"
    elif signal_repairs and scale_repairs:
        primary = "mixed_and_calibratable"
    else:
        primary = "structurally_underidentified"
    classification = {
        "primary": primary,
        "single_trajectory_feasibility": "structurally_underidentified",
        "trend_interpretation": "consistent_with_estimand_mismatch",
        "smoothing_conclusion": "does_not_remove_level_bias",
        "attribution_tolerance": attribution_tolerance,
    }
    return summaries, classification
