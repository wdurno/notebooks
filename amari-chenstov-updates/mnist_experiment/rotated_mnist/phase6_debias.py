"""Artifact-only trend-variance diagnostics for the Plan 6 EDR audit."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def reconstruct_debiased_recommendations(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    variance_scale: float,
    pi_min: float,
    pi_max: float,
) -> list[dict[str, Any]]:
    """Propagate approximate trend variance from persisted predictable states."""

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(variance_scale) or variance_scale <= 0.0:
        raise ValueError("variance_scale must be finite and positive")
    if not 0.0 < pi_min < pi_max <= 1.0:
        raise ValueError("pi bounds must satisfy 0 < min < max <= 1")

    trend_variance_energy = 0.0
    result: list[dict[str, Any]] = []
    for expected_step, row in enumerate(rows):
        if int(row["step"]) != expected_step:
            raise ValueError("trajectory steps must be contiguous and zero based")
        controller = row.get("controller")
        if controller is None:
            if expected_step != len(rows) - 1:
                raise ValueError("only the final trajectory point may omit a decision")
            continue

        signal = float(controller["signal_energy"])
        uncertainty = float(controller["uncertainty_scale_estimate"])
        effective_size = float(controller["effective_size"])
        values = (signal, uncertainty, effective_size, trend_variance_energy)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("controller risk values must be finite")
        if signal < 0.0 or uncertainty < 0.0 or effective_size <= 0.0:
            raise ValueError("controller risk values have invalid signs")

        q = 1.0 / effective_size
        correction = variance_scale * trend_variance_energy
        corrected_signal = max(0.0, signal - correction)
        old_covariance = q * uncertainty
        new_covariance = uncertainty / batch_size
        denominator = corrected_signal + old_covariance + new_covariance
        recommendation = None
        clipped = None
        if denominator > 0.0:
            recommendation = (corrected_signal + old_covariance) / denominator
            clipped = min(pi_max, max(pi_min, recommendation))
        result.append(
            {
                "step": expected_step,
                "angle_degrees": float(row["angle_degrees"]),
                "cumulative_angular_degrees": float(
                    row["cumulative_angular_degrees"]
                ),
                "leg_id": int(row["leg_id"]),
                "variance_scale": variance_scale,
                "trend_variance_energy_before": trend_variance_energy,
                "scaled_variance_correction": correction,
                "raw_signal_energy": signal,
                "debiased_signal_energy": corrected_signal,
                "correction_to_signal_ratio": (
                    correction / signal if signal > 0.0 else None
                ),
                "uncertainty_scale_estimate": uncertainty,
                "q_before": q,
                "old_covariance_risk": old_covariance,
                "new_covariance_risk": new_covariance,
                "debiased_unclipped_pi": recommendation,
                "debiased_clipped_pi": clipped,
                "realized_edr_pi": float(controller["applied_pi"]),
                "cold_start_active": bool(controller["cold_start_active"]),
            }
        )

        acceptance = row.get("controller_acceptance")
        if acceptance is None:
            raise ValueError("a transition decision requires acceptance diagnostics")
        gain = float(acceptance["trend_gain"])
        if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
            raise ValueError("trend gain must be in [0, 1]")
        innovation_energy = (q + 1.0 / batch_size) * uncertainty
        trend_variance_energy = (
            (1.0 - gain) ** 2 * trend_variance_energy
            + gain**2 * innovation_energy
        )
    return result


def summarize_debias_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    live_start_step: int,
) -> dict[str, Any]:
    live = [
        row
        for row in rows
        if int(row["step"]) >= live_start_step
        and row["debiased_unclipped_pi"] is not None
    ]
    if not live:
        raise ValueError("debias summary requires supported live decisions")
    corrected = [float(row["debiased_unclipped_pi"]) for row in live]
    realized = [float(row["realized_edr_pi"]) for row in live]
    ratios = [
        min(float(row["correction_to_signal_ratio"]), 1.0)
        for row in live
        if row["correction_to_signal_ratio"] is not None
    ]
    zeroed = sum(float(row["debiased_signal_energy"]) == 0.0 for row in live)
    return {
        "live_start_step": live_start_step,
        "live_decision_count": len(live),
        "debiased_pi_mean": statistics.fmean(corrected),
        "debiased_pi_minimum": min(corrected),
        "debiased_pi_maximum": max(corrected),
        "realized_edr_pi_mean": statistics.fmean(realized),
        "mean_absolute_debiased_to_realized_gap": statistics.fmean(
            abs(left - right) for left, right in zip(corrected, realized, strict=True)
        ),
        "mean_capped_correction_to_signal_ratio": statistics.fmean(ratios),
        "zeroed_signal_fraction": zeroed / len(live),
    }
