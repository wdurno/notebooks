"""Exact scalar risk algebra for fixed-composition pace control."""

from __future__ import annotations

import dataclasses
import math


def _nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _fixed_pi(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("fixed_pi must be finite and strictly between zero and one")
    return result


def _batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("batch_size must be a positive integer")
    return value


def fixed_batch_risk(
    pi: float,
    *,
    movement_energy: float,
    q: float,
    old_covariance_shape: float,
    new_covariance_shape: float,
    batch_size: int,
) -> float:
    """Evaluate the frozen applied one-step Fisher-risk surrogate."""

    action = _fixed_pi(pi)
    signal = _nonnegative(movement_energy, "movement_energy")
    concentration = _nonnegative(q, "q")
    old_shape = _nonnegative(old_covariance_shape, "old_covariance_shape")
    new_shape = _nonnegative(new_covariance_shape, "new_covariance_shape")
    count = _batch_size(batch_size)
    return (1.0 - action) ** 2 * (signal + concentration * old_shape) + (
        action**2 * new_shape / count
    )


def optimal_fixed_batch_pi(
    *,
    movement_energy: float,
    q: float,
    old_covariance_shape: float,
    new_covariance_shape: float,
    batch_size: int,
) -> float:
    """Return the unique minimizer of the convex fixed-batch surrogate."""

    signal = _nonnegative(movement_energy, "movement_energy")
    concentration = _nonnegative(q, "q")
    old_shape = _nonnegative(old_covariance_shape, "old_covariance_shape")
    new_shape = _nonnegative(new_covariance_shape, "new_covariance_shape")
    count = _batch_size(batch_size)
    retained = signal + concentration * old_shape
    fresh = new_shape / count
    denominator = retained + fresh
    if denominator <= 0.0:
        raise ValueError("the risk surrogate has no unique minimizer")
    return retained / denominator


def movement_energy_target(
    fixed_pi: float,
    *,
    q: float,
    old_covariance_shape: float,
    new_covariance_shape: float,
    batch_size: int,
) -> float:
    """Invert the minimizer for the movement energy making ``fixed_pi`` optimal."""

    action = _fixed_pi(fixed_pi)
    concentration = _nonnegative(q, "q")
    old_shape = _nonnegative(old_covariance_shape, "old_covariance_shape")
    new_shape = _nonnegative(new_covariance_shape, "new_covariance_shape")
    count = _batch_size(batch_size)
    return action * new_shape / ((1.0 - action) * count) - concentration * old_shape


def fixed_pi_q_update(q: float, fixed_pi: float, batch_size: int) -> float:
    """Advance the exact squared-weight concentration recursion by one batch."""

    concentration = _nonnegative(q, "q")
    action = _fixed_pi(fixed_pi)
    count = _batch_size(batch_size)
    return (1.0 - action) ** 2 * concentration + action**2 / count


def fixed_pi_q_equilibrium(fixed_pi: float, batch_size: int) -> float:
    action = _fixed_pi(fixed_pi)
    count = _batch_size(batch_size)
    return action / (count * (2.0 - action))


def fixed_pi_q_at_step(
    initial_q: float, fixed_pi: float, batch_size: int, step: int
) -> float:
    """Evaluate the exact fixed-action concentration transient."""

    concentration = _nonnegative(initial_q, "initial_q")
    action = _fixed_pi(fixed_pi)
    count = _batch_size(batch_size)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    equilibrium = fixed_pi_q_equilibrium(action, count)
    return equilibrium + (1.0 - action) ** (2 * step) * (
        concentration - equilibrium
    )


def stationary_movement_target(
    fixed_pi: float,
    *,
    covariance_shape: float,
    batch_size: int,
) -> float:
    action = _fixed_pi(fixed_pi)
    shape = _nonnegative(covariance_shape, "covariance_shape")
    count = _batch_size(batch_size)
    return shape * action / (count * (1.0 - action) * (2.0 - action))


def centered_pace(movement_target: float, fisher_speed: float) -> float:
    """Solve ``a^2 J = S`` in the centered local model."""

    target = _nonnegative(movement_target, "movement_target")
    speed = _nonnegative(fisher_speed, "fisher_speed")
    if speed <= 0.0:
        raise ValueError("fisher_speed must be positive")
    return math.sqrt(target / speed)


@dataclasses.dataclass(frozen=True)
class PaceRoots:
    roots: tuple[float, ...]
    discriminant: float
    minimum_energy: float
    minimizing_pace: float


def pace_roots(
    movement_target: float,
    *,
    direction_energy: float,
    direction_trend_inner: float,
    trend_energy: float,
) -> PaceRoots:
    """Solve ``||a v - mu||_M^2 = S`` for nonnegative pace values."""

    target = _nonnegative(movement_target, "movement_target")
    direction = _nonnegative(direction_energy, "direction_energy")
    trend = _nonnegative(trend_energy, "trend_energy")
    cross = float(direction_trend_inner)
    if not math.isfinite(cross):
        raise ValueError("direction_trend_inner must be finite")
    if direction <= 0.0:
        raise ValueError("direction_energy must be positive")
    tolerance = 128.0 * math.ulp(max(1.0, abs(direction * trend), cross * cross))
    if cross * cross > direction * trend + tolerance:
        raise ValueError("quadratic terms violate Cauchy-Schwarz")
    discriminant = cross * cross - direction * (trend - target)
    minimizing_pace = max(0.0, cross / direction)
    minimum_energy = (
        trend - cross * cross / direction if cross > 0.0 else trend
    )
    if discriminant < -tolerance:
        roots: tuple[float, ...] = ()
    else:
        root = math.sqrt(max(0.0, discriminant))
        values = ((cross - root) / direction, (cross + root) / direction)
        roots = tuple(sorted({value for value in values if value >= 0.0}))
    return PaceRoots(
        roots=roots,
        discriminant=discriminant,
        minimum_energy=minimum_energy,
        minimizing_pace=minimizing_pace,
    )


def validate_pace_action_target(target: str) -> None:
    """Reject post-fit actuation, which changes effective composition."""

    if target != "next_distribution":
        raise ValueError("pace may act only on the next encountered distribution")
