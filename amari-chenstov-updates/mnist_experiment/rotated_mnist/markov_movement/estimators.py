"""Pure estimators for the Plan 9 anchor-cancelling construction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from src.representations import FisherRepresentation


def _finite_vector(value: Tensor, *, name: str) -> None:
    if value.ndim != 1 or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating vector")


def anchor_cancelled_observations(
    displacements: Tensor,
    actions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return normalized updates ``Y_t`` and lagged differences ``Z_t``.

    Row ``t`` of ``displacements`` is the accepted update from state ``t`` to
    ``t + 1`` and ``actions[t]`` is the predictable action used for that
    update. Row ``t - 1`` of the returned ``Z`` corresponds to ``Z_t`` for
    ``t >= 1``.
    """

    if (
        displacements.ndim != 2
        or not displacements.is_floating_point()
        or not torch.isfinite(displacements).all()
    ):
        raise ValueError("displacements must be a finite floating matrix")
    if (
        actions.ndim != 1
        or actions.shape[0] != displacements.shape[0]
        or not actions.is_floating_point()
        or not torch.isfinite(actions).all()
        or bool((actions <= 0.0).any())
        or bool((actions > 1.0).any())
    ):
        raise ValueError("actions must be finite, positive, and aligned")
    values = displacements / actions[:, None]
    if values.shape[0] < 2:
        return values, values.new_empty((0, values.shape[1]))
    cancelled = values[1:] - (1.0 - actions[:-1, None]) * values[:-1]
    return values, cancelled


def exponential_window_weights(
    coordinates: Tensor,
    *,
    half_life: float | None,
) -> Tensor:
    """Return normalized causal weights ending at the final coordinate."""

    _finite_vector(coordinates, name="coordinates")
    if coordinates.numel() == 0:
        raise ValueError("coordinates cannot be empty")
    if bool((coordinates[1:] < coordinates[:-1]).any()):
        raise ValueError("coordinates must be nondecreasing")
    if half_life is None:
        result = torch.zeros_like(coordinates)
        result[-1] = 1.0
        return result
    if not math.isfinite(half_life) or half_life <= 0.0:
        raise ValueError("half_life must be finite and positive")
    exponents = -(coordinates[-1] - coordinates) / half_life
    raw = torch.pow(torch.tensor(2.0, dtype=coordinates.dtype), exponents)
    return raw / raw.sum()


def innovation_coefficients(weights: Tensor) -> Tensor:
    """Map weights on ``xi_{k+1} - xi_k`` to coefficients on each ``xi``."""

    _finite_vector(weights, name="weights")
    if weights.numel() == 0:
        raise ValueError("weights cannot be empty")
    tolerance = 100.0 * torch.finfo(weights.dtype).eps
    if bool((weights < 0.0).any()) or not math.isclose(
        float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ValueError("weights must be nonnegative and sum to one")
    coefficients = weights.new_empty(weights.numel() + 1)
    coefficients[0] = -weights[0]
    if weights.numel() > 1:
        coefficients[1:-1] = weights[:-1] - weights[1:]
    coefficients[-1] = weights[-1]
    return coefficients


def innovation_noise_risk(weights: Tensor, innovation_risks: Tensor) -> float:
    """Return exact weighted risk for independent adjacent innovations.

    ``innovation_risks[j]`` is ``tr(G Cov(xi_j))`` in the common metric used
    to evaluate the filtered vector. The coefficient construction retains the
    negative covariance shared by adjacent differenced observations.
    """

    coefficients = innovation_coefficients(weights)
    _finite_vector(innovation_risks, name="innovation_risks")
    if innovation_risks.shape != coefficients.shape or bool(
        (innovation_risks < 0.0).any()
    ):
        raise ValueError("innovation risks must be aligned and nonnegative")
    return float(coefficients.square() @ innovation_risks)


def weighted_vector(vectors: Tensor, weights: Tensor) -> Tensor:
    if (
        vectors.ndim != 2
        or vectors.shape[0] != weights.numel()
        or not vectors.is_floating_point()
        or not torch.isfinite(vectors).all()
    ):
        raise ValueError("vectors and weights must be finite and aligned")
    innovation_coefficients(weights)
    return weights @ vectors


def weighted_cross_moment(
    vectors: Tensor,
    weights: Tensor,
    fisher: FisherRepresentation,
    *,
    lag_exclusion: int,
) -> tuple[float, dict[str, float | int]]:
    """Estimate squared local mean through distinct vector cross-products.

    Pair weights are products of deterministic observation weights. Each
    unordered pair is used once, and pairs separated by at most
    ``lag_exclusion`` rows are omitted.
    """

    if (
        vectors.ndim != 2
        or vectors.shape[0] != weights.numel()
        or vectors.shape[1] != fisher.shape[0]
        or not vectors.is_floating_point()
        or not torch.isfinite(vectors).all()
    ):
        raise ValueError("vectors, weights, and Fisher must be finite and aligned")
    innovation_coefficients(weights)
    if (
        not isinstance(lag_exclusion, int)
        or isinstance(lag_exclusion, bool)
        or lag_exclusion < 0
    ):
        raise ValueError("lag_exclusion must be a nonnegative integer")

    pair_products = []
    pair_weights = []
    converted = vectors.to(device=fisher.device, dtype=fisher.dtype)
    fisher_vectors = torch.stack([fisher.matvec(row) for row in converted])
    for left in range(vectors.shape[0]):
        for right in range(left + lag_exclusion + 1, vectors.shape[0]):
            pair_weights.append(weights[left] * weights[right])
            pair_products.append(converted[left] @ fisher_vectors[right])
    if not pair_weights:
        raise ValueError("cross-moment window has no admissible pairs")
    pair_weight = torch.stack(pair_weights)
    total = pair_weight.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        raise ValueError("cross-moment pair weight is invalid")
    normalized = pair_weight / total
    products = torch.stack(pair_products)
    estimate = float(normalized @ products)
    effective_pairs = float(normalized.square().sum().reciprocal())
    return estimate, {
        "pair_count": len(pair_weights),
        "effective_pair_count": effective_pairs,
        "maximum_pair_weight": float(normalized.max()),
    }


def quadratic_energy(vector: Tensor, fisher: FisherRepresentation) -> float:
    _finite_vector(vector, name="vector")
    if vector.numel() != fisher.shape[0]:
        raise ValueError("vector and Fisher dimensions differ")
    value = float(
        fisher.quadratic(vector.to(device=fisher.device, dtype=fisher.dtype))
    )
    tolerance = 500.0 * torch.finfo(fisher.dtype).eps * max(abs(value), 1.0)
    if value < -tolerance:
        raise ValueError("PSD Fisher produced a negative quadratic energy")
    return max(value, 0.0)


def marginal_action(
    signal: float,
    q: float,
    old_covariance_shape: float,
    new_covariance_shape: float,
    batch_size: int,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return the applied fixed-batch marginal-risk recommendation."""

    values = (signal, q, old_covariance_shape, new_covariance_shape, epsilon)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("marginal-action inputs must be finite")
    if signal < 0.0 or old_covariance_shape < 0.0 or new_covariance_shape < 0.0:
        raise ValueError("risk coefficients must be nonnegative")
    if q <= 0.0 or epsilon <= 0.0:
        raise ValueError("q and epsilon must be positive")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    old_risk = q * old_covariance_shape
    new_risk = new_covariance_shape / batch_size
    numerator = signal + old_risk
    return numerator / (numerator + new_risk + epsilon)


def causal_filter(
    vectors: Tensor,
    coordinates: Tensor,
    *,
    half_life: float | None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Filter every prefix and retain its exact realized weights."""

    if vectors.ndim != 2 or vectors.shape[0] != coordinates.numel():
        raise ValueError("vectors and coordinates must be aligned")
    filtered = []
    weights = []
    for stop in range(1, coordinates.numel() + 1):
        current = exponential_window_weights(
            coordinates[:stop], half_life=half_life
        )
        weights.append(current)
        filtered.append(weighted_vector(vectors[:stop], current))
    return torch.stack(filtered), tuple(weights)


def mean_absolute_error(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("MAE inputs must be nonempty and aligned")
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)
