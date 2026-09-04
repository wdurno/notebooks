"""Lightweight, artifact-only analysis for the Plan 6 oracle comparison."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from .artifacts import RotatedArtifactError
from .phase5_single_lap_artifacts import LoadedSingleLapRun
from .phase6_artifacts import LoadedPhase6DebiasRun, LoadedPhase6OracleRun


def discounted_risk_actions(
    coefficients: Sequence[tuple[float, float]],
    *,
    half_life_steps: float,
    cold_start_steps: int,
    cold_start_pi: float,
    pi_min: float,
    pi_max: float,
) -> tuple[float, ...]:
    """Apply EDR's coefficient-wise EMA to an offline oracle sequence."""

    if not math.isfinite(half_life_steps) or half_life_steps <= 0.0:
        raise ValueError("half_life_steps must be finite and positive")
    if cold_start_steps < 0 or not 0.0 < pi_min < pi_max <= 1.0:
        raise ValueError("invalid cold-start or action bounds")
    gain = 1.0 - 0.5 ** (1.0 / half_life_steps)
    old_moment = 0.0
    new_moment = 0.0
    actions = []
    for step, (old_risk, new_risk) in enumerate(coefficients):
        if (
            not math.isfinite(old_risk)
            or not math.isfinite(new_risk)
            or old_risk < 0.0
            or new_risk < 0.0
        ):
            raise ValueError("risk coefficients must be finite and nonnegative")
        old_moment = (1.0 - gain) * old_moment + gain * old_risk
        new_moment = (1.0 - gain) * new_moment + gain * new_risk
        denominator = old_moment + new_moment
        raw = (
            cold_start_pi
            if step < cold_start_steps or denominator == 0.0
            else old_moment / denominator
        )
        actions.append(min(pi_max, max(pi_min, raw)))
    return tuple(actions)


def tracking_rows(
    source: LoadedSingleLapRun,
    debias: LoadedPhase6DebiasRun,
    oracle: LoadedPhase6OracleRun,
    *,
    sample_size: int = 2048,
    rank: int = 16,
    replicate_count: int = 64,
    variance_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Align source actions, online estimates, and oracle transition targets."""

    if (
        source.config.run_id != debias.config.source_run_id
        or source.config.run_id != oracle.config.source_run_id
    ):
        raise RotatedArtifactError("Plan 6 inputs do not share a source run")
    if debias.config.condition != oracle.config.condition:
        raise RotatedArtifactError("Plan 6 inputs do not share a condition")
    sample_key = str(sample_size)
    rank_key = str(rank)
    replicate_key = str(replicate_count)
    variance_key = f"variance_scale_{variance_scale:g}"
    controller = source.config.controller
    output = []
    for schedule in oracle.config.schedule_kinds:
        source_values = source.trajectory_metrics[schedule][oracle.config.condition]
        debias_values = debias.recommendations[schedule].get(variance_key)
        if debias_values is None:
            raise RotatedArtifactError(f"missing debias scale {variance_scale:g}")
        debias_by_step = {int(row["step"]): row for row in debias_values}
        oracle_values = oracle.oracle_estimates[schedule]
        steps = sorted(int(value) for value in oracle_values)
        expected_steps = list(range(len(source_values) - 1))
        if oracle.config.mode == "full" and steps != expected_steps:
            raise RotatedArtifactError(f"{schedule} oracle path is incomplete")
        coefficients = []
        details_by_step = {}
        for step in steps:
            try:
                details = oracle_values[str(step)][sample_key][rank_key][replicate_key]
            except KeyError as exc:
                raise RotatedArtifactError(
                    "requested oracle sample, rank, or replicate count is absent"
                ) from exc
            details_by_step[step] = details
            coefficients.append(
                (
                    float(details["signal"] + details["old_covariance_risk"]),
                    float(details["new_covariance_risk"]),
                )
            )
        smoothed = discounted_risk_actions(
            coefficients,
            half_life_steps=controller.action_half_life_steps,
            cold_start_steps=controller.cold_start_steps,
            cold_start_pi=controller.cold_start_pi,
            pi_min=controller.pi_min,
            pi_max=controller.pi_max,
        )
        for index, step in enumerate(steps):
            source_row = source_values[step]
            next_row = source_values[step + 1]
            online = debias_by_step.get(step)
            details = details_by_step[step]
            bootstrap = details["bootstrap"]
            decision = source_row["controller"]
            if decision is None:
                raise RotatedArtifactError("source transition lacks a controller decision")
            output.append(
                {
                    "schedule": schedule,
                    "step": step,
                    "angle_degrees": float(source_row["angle_degrees"]),
                    "next_angle_degrees": float(next_row["angle_degrees"]),
                    "angular_speed_degrees": abs(
                        float(next_row["angle_degrees"])
                        - float(source_row["angle_degrees"])
                    ),
                    "cumulative_angular_degrees": float(
                        source_row["cumulative_angular_degrees"]
                    ),
                    "leg_id": int(source_row["leg_id"]),
                    "cold_start": bool(decision["cold_start_active"]),
                    "realized_edr_pi": float(decision["applied_pi"]),
                    "debiased_instantaneous_pi": (
                        None
                        if online is None or online["debiased_clipped_pi"] is None
                        else float(online["debiased_clipped_pi"])
                    ),
                    "oracle_pi": float(details["pi"]),
                    "oracle_pi_bootstrap_mean": float(bootstrap["mean"]),
                    "oracle_pi_lower_95": float(bootstrap["lower_95"]),
                    "oracle_pi_upper_95": float(bootstrap["upper_95"]),
                    "oracle_smoothed_pi": smoothed[index],
                    "signal": float(details["signal"]),
                    "signal_raw": float(details["signal_raw"]),
                    "signal_noise_correction": float(
                        details["signal_noise_correction"]
                    ),
                    "old_covariance_risk": float(details["old_covariance_risk"]),
                    "new_covariance_risk": float(details["new_covariance_risk"]),
                    "risk_curvature": float(details["risk_curvature"]),
                    "q": float(details["q"]),
                    "six_standard_error_half_width": float(
                        details["six_standard_error_half_width"]
                    ),
                    "precision_pass": bool(details["precision_pass"]),
                    "sample_size": sample_size,
                    "rank": rank,
                    "replicate_count": replicate_count,
                }
            )
    return output


def tracking_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    schedules = sorted({str(row["schedule"]) for row in rows})
    for schedule in schedules:
        selected = [row for row in rows if row["schedule"] == schedule]
        live = [row for row in selected if not row["cold_start"]]
        debiased = [
            row for row in live if row["debiased_instantaneous_pi"] is not None
        ]
        edr_errors = [abs(row["realized_edr_pi"] - row["oracle_pi"]) for row in live]
        debiased_errors = [
            abs(row["debiased_instantaneous_pi"] - row["oracle_pi"])
            for row in debiased
        ]
        smoothed_errors = [
            abs(row["oracle_smoothed_pi"] - row["oracle_pi"]) for row in live
        ]
        edr_to_smoothed_errors = [
            abs(row["realized_edr_pi"] - row["oracle_smoothed_pi"])
            for row in live
        ]
        signal_lifts = [
            row["oracle_pi"]
            - row["old_covariance_risk"]
            / (row["old_covariance_risk"] + row["new_covariance_risk"])
            for row in live
        ]
        weighted = [
            row["risk_curvature"]
            * (row["realized_edr_pi"] - row["oracle_pi"]) ** 2
            for row in live
        ]
        output.append(
            {
                "schedule": schedule,
                "live_transitions": len(live),
                "edr_oracle_mae": statistics.fmean(edr_errors),
                "debiased_oracle_mae": statistics.fmean(debiased_errors),
                "oracle_smoothing_mae": statistics.fmean(smoothed_errors),
                "edr_oracle_smoothed_mae": statistics.fmean(
                    edr_to_smoothed_errors
                ),
                "consequence_weighted_error_sum": sum(weighted),
                "consequence_weighted_error_mean": statistics.fmean(weighted),
                "oracle_pi_mean": statistics.fmean(row["oracle_pi"] for row in live),
                "oracle_pi_min": min(row["oracle_pi"] for row in live),
                "oracle_pi_max": max(row["oracle_pi"] for row in live),
                "oracle_signal_lift_mean": statistics.fmean(signal_lifts),
                "oracle_signal_lift_max": max(signal_lifts),
                "edr_oracle_correlation": statistics.correlation(
                    [row["realized_edr_pi"] for row in live],
                    [row["oracle_pi"] for row in live],
                ),
                "precision_pass_fraction": statistics.fmean(
                    float(row["precision_pass"]) for row in selected
                ),
                "zero_signal_fraction": statistics.fmean(
                    float(row["signal"] == 0.0) for row in selected
                ),
            }
        )
    return output


def convergence_rows(run: LoadedPhase6OracleRun) -> list[dict[str, Any]]:
    output = []
    for schedule, schedule_values in run.oracle_estimates.items():
        for step, step_values in schedule_values.items():
            for sample_size, sample_values in step_values.items():
                for rank, rank_values in sample_values.items():
                    for count, details in rank_values.items():
                        output.append(
                            {
                                "schedule": schedule,
                                "step": int(step),
                                "sample_size": int(sample_size),
                                "rank": int(rank),
                                "replicate_count": int(count),
                                "oracle_pi": float(details["pi"]),
                                "six_standard_error_half_width": float(
                                    details["six_standard_error_half_width"]
                                ),
                                "precision_pass": bool(details["precision_pass"]),
                            }
                        )
    return output
