"""Streaming confidence radii for means in finite-dimensional Hilbert spaces."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
from torch import Tensor


@dataclasses.dataclass
class HilbertMean:
    """Welford moments for tensor-valued observations.

    The scalar second moment is the sum of coordinate-wise second moments.
    For matrices this is Frobenius geometry; for vectors it is Euclidean
    geometry. No covariance matrix is materialized.
    """

    count: int = 0
    mean: Tensor | None = None
    centered_sum_squares: float = 0.0
    lagged_raw_inner_product: float = 0.0
    lagged_raw_left_squares: float = 0.0
    lagged_raw_right_squares: float = 0.0
    _first: Tensor | None = dataclasses.field(default=None, repr=False)
    _previous: Tensor | None = dataclasses.field(default=None, repr=False)

    def update(self, observation: Tensor) -> None:
        value = observation.detach()
        if not torch.isfinite(value).all():
            raise ValueError("Hilbert-mean observations must be finite")
        if self.mean is not None and value.shape != self.mean.shape:
            raise ValueError("Hilbert-mean observations must have equal shapes")

        self.count += 1
        if self.mean is None:
            self.mean = value.clone()
            self._first = value.clone()
        else:
            delta = value - self.mean
            self.mean.add_(delta / self.count)
            delta_after = value - self.mean
            self.centered_sum_squares += float(
                torch.sum(delta * delta_after)
            )

        if self._previous is not None:
            self.lagged_raw_inner_product += float(
                torch.sum(self._previous * value)
            )
            self.lagged_raw_left_squares += float(
                torch.sum(self._previous.square())
            )
            self.lagged_raw_right_squares += float(torch.sum(value.square()))
        self._previous = value.clone()

    @property
    def mean_norm(self) -> float:
        if self.mean is None:
            raise ValueError("the Hilbert mean has no observations")
        return float(torch.linalg.vector_norm(self.mean))

    @property
    def empirical_variance(self) -> float | None:
        if self.count < 2:
            return None
        return max(self.centered_sum_squares / (self.count - 1), 0.0)

    def confidence_radius(self, sigma: float) -> float | None:
        variance = self.empirical_variance
        if variance is None:
            return None
        return float(sigma) * math.sqrt(variance / self.count)

    @property
    def lag_one_correlation(self) -> float | None:
        if self.count < 2 or self.mean is None:
            return None
        if self._first is None or self._previous is None:
            raise RuntimeError("Hilbert lag state is incomplete")
        total = self.mean * self.count
        left_sum = total - self._previous
        right_sum = total - self._first
        mean_square = float(torch.sum(self.mean.square()))
        centered_inner = (
            self.lagged_raw_inner_product
            - float(torch.sum(left_sum * self.mean))
            - float(torch.sum(right_sum * self.mean))
            + (self.count - 1) * mean_square
        )
        centered_left_squares = (
            self.lagged_raw_left_squares
            - 2.0 * float(torch.sum(left_sum * self.mean))
            + (self.count - 1) * mean_square
        )
        centered_right_squares = (
            self.lagged_raw_right_squares
            - 2.0 * float(torch.sum(right_sum * self.mean))
            + (self.count - 1) * mean_square
        )
        denominator = math.sqrt(
            max(centered_left_squares, 0.0)
            * max(centered_right_squares, 0.0)
        )
        if denominator == 0.0:
            return None
        return centered_inner / denominator

    def diagnostics(
        self,
        *,
        sigma: float,
        relative_epsilon: float,
        absolute_epsilon: float,
        minimum_count: int,
        maximum_count: int,
    ) -> dict[str, Any]:
        if self.mean is None:
            raise ValueError("the Hilbert mean has no observations")
        radius = self.confidence_radius(sigma)
        threshold = absolute_epsilon + relative_epsilon * self.mean_norm
        converged = (
            self.count >= minimum_count
            and radius is not None
            and radius <= threshold
        )
        return {
            "count": self.count,
            "minimum_count": minimum_count,
            "maximum_count": maximum_count,
            "sigma": float(sigma),
            "relative_epsilon": float(relative_epsilon),
            "absolute_epsilon": float(absolute_epsilon),
            "mean_norm": self.mean_norm,
            "empirical_hilbert_variance": self.empirical_variance,
            "confidence_radius": radius,
            "relative_confidence_radius": (
                None if self.mean_norm == 0.0 or radius is None
                else radius / self.mean_norm
            ),
            "stopping_threshold": threshold,
            "converged": converged,
            "lag_one_hilbert_correlation": self.lag_one_correlation,
        }
