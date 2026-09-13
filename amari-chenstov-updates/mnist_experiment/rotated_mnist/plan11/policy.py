"""Predictable shrinkage of the Plan 8 decomposed risk recommendation."""

from __future__ import annotations

import dataclasses
import math

from src.controller import (
    ControllerDecision,
    ControllerState,
    DecomposedRiskDecision,
    DiscountedMovementState,
    decide_decomposed_risk_controller,
    decomposed_fixed_batch_pi,
    tracked_q_covariance_pi,
)
from src.representations import FisherRepresentation

from .config import Policy


def blend_action(anchor: float, gain: float, recommendation: float, *, cold: bool) -> tuple[float, float]:
    raw = anchor if cold else anchor + gain * (recommendation - anchor)
    if not math.isfinite(raw):
        raise ValueError("nonfinite Plan 11 action")
    return raw, min(0.95, max(0.01, raw))


def decide(
    controller: ControllerState,
    movement: DiscountedMovementState,
    fisher: FisherRepresentation,
    policy: Policy,
    *,
    batch_size: int,
) -> tuple[ControllerDecision, DecomposedRiskDecision, float]:
    decomposed = decide_decomposed_risk_controller(
        controller,
        movement,
        batch_size=batch_size,
        fisher=fisher,
        pi_min=0.01,
        pi_max=0.95,
        cold_start_pi=policy.anchor,
        cold_start_steps=8,
        movement_half_life_steps=8.0,
        trace_epsilon=1e-12,
    )
    recommendation = (
        tracked_q_covariance_pi(controller.q, batch_size)
        if decomposed.unsupported_scale_fallback
        else decomposed_fixed_batch_pi(
            decomposed.discounted_movement_premium, controller.q, batch_size
        )
    )
    raw, applied = blend_action(
        policy.anchor,
        policy.gain,
        recommendation,
        cold=decomposed.controller.cold_start_active,
    )
    decision = dataclasses.replace(
        decomposed.controller,
        policy="shrunk_decomposed_edr",
        raw_pi=raw,
        applied_pi=applied,
        lower_bound_active=raw < 0.01,
        upper_bound_active=raw > 0.95,
        ewc_odds=(1.0 - applied) / applied,
    )
    return decision, decomposed, recommendation
