"""Exogenous exposure-clock drift; evaluation, reset and sleep do not advance it."""

import math


def smoothstep5(progress: float) -> float:
    r = min(1., max(0., progress))
    return r ** 3 * (10 + r * (-15 + 6 * r))


def drift_alpha(exposure: float, total_intervals: int) -> float:
    if not math.isfinite(exposure) or exposure < 0 or total_intervals <= 0:
        raise ValueError("invalid drift exposure")
    fraction = exposure / total_intervals
    if fraction <= .1 or fraction >= .9:
        return 0.
    if fraction < .45:
        return smoothstep5((fraction - .1) / .35)
    if fraction <= .55:
        return 1.
    return 1. - smoothstep5((fraction - .55) / .35)


def steering_gain(alpha: float) -> float:
    return 1. - .35 * alpha
