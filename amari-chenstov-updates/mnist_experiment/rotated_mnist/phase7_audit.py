"""Pure coefficient decompositions for the Plan 7 anchor audit."""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from src.representations import FisherRepresentation


def weight_concentration(weights: Tensor) -> float:
    if (
        weights.ndim != 1
        or not weights.is_floating_point()
        or not torch.isfinite(weights).all()
        or bool((weights < 0.0).any())
    ):
        raise ValueError("weights must be a finite nonnegative vector")
    tolerance = 100.0 * torch.finfo(weights.dtype).eps
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=tolerance):
        raise ValueError("weights must sum to one")
    return float(weights.square().sum())


def compose_observation_weights(weights: Tensor, pi: float, batch_size: int) -> Tensor:
    if not math.isfinite(pi) or not 0.0 <= pi <= 1.0:
        raise ValueError("pi must be in [0, 1]")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    weight_concentration(weights)
    new = torch.full(
        (batch_size,),
        pi / batch_size,
        dtype=weights.dtype,
        device=weights.device,
    )
    return torch.cat(((1.0 - pi) * weights, new))


def weight_concentration_update(q: float, pi: float, batch_size: int) -> float:
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("q must be finite and positive")
    if not math.isfinite(pi) or not 0.0 <= pi <= 1.0:
        raise ValueError("pi must be in [0, 1]")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return (1.0 - pi) ** 2 * q + pi**2 / batch_size


def stationary_weight_concentration(pi: float, batch_size: int) -> float:
    if not math.isfinite(pi) or not 0.0 < pi <= 1.0:
        raise ValueError("stationary pi must be in (0, 1]")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return pi / (batch_size * (2.0 - pi))


@dataclasses.dataclass(frozen=True)
class AnchorDecomposition:
    population_signal: float
    anchor_error_energy: float
    cross_term: float
    conditional_signal: float
    identity_error: float

    def mapping(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def anchor_decomposition(
    population_displacement: Tensor,
    anchor_error: Tensor,
    fisher: FisherRepresentation,
) -> AnchorDecomposition:
    vectors = (population_displacement, anchor_error)
    if any(
        value.ndim != 1
        or value.numel() != fisher.shape[0]
        or not torch.isfinite(value).all()
        for value in vectors
    ):
        raise ValueError("decomposition vectors are invalid")
    displacement = population_displacement.to(
        device=fisher.device, dtype=fisher.dtype
    )
    error = anchor_error.to(device=fisher.device, dtype=fisher.dtype)
    fisher_displacement = fisher.matvec(displacement)
    population = float(displacement @ fisher_displacement)
    anchor = float(fisher.quadratic(error))
    cross = float(-2.0 * (error @ fisher_displacement))
    conditional = float(fisher.quadratic(displacement - error))
    values = (population, anchor, conditional)
    scale = max(abs(population) + abs(anchor) + abs(cross), 1.0)
    tolerance = 500.0 * torch.finfo(fisher.dtype).eps * scale
    if any(value < -tolerance for value in values):
        raise ValueError("PSD Fisher produced a materially negative energy")
    population = max(population, 0.0)
    anchor = max(anchor, 0.0)
    conditional = max(conditional, 0.0)
    identity_error = conditional - (population + anchor + cross)
    if abs(identity_error) > tolerance:
        raise ValueError("anchor quadratic decomposition did not close")
    return AnchorDecomposition(
        population_signal=population,
        anchor_error_energy=anchor,
        cross_term=cross,
        conditional_signal=conditional,
        identity_error=identity_error,
    )


@dataclasses.dataclass(frozen=True)
class RiskRecommendations:
    covariance_only: float
    centered_marginal: float
    realized_conditional: float
    old_covariance_risk: float
    new_covariance_risk: float

    def mapping(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def risk_recommendations(
    *,
    q: float,
    old_covariance_shape: float,
    new_covariance_shape: float,
    deployed_batch_size: int,
    population_signal: float,
    conditional_signal: float,
) -> RiskRecommendations:
    values = (
        q,
        old_covariance_shape,
        new_covariance_shape,
        population_signal,
        conditional_signal,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("risk inputs must be finite")
    if q <= 0.0 or any(value < 0.0 for value in values[1:]):
        raise ValueError("risk inputs have invalid signs")
    if (
        not isinstance(deployed_batch_size, int)
        or isinstance(deployed_batch_size, bool)
        or deployed_batch_size < 1
    ):
        raise ValueError("deployed_batch_size must be positive")
    old = q * old_covariance_shape
    new = new_covariance_shape / deployed_batch_size

    def ratio(numerator: float, denominator_extra: float) -> float:
        denominator = numerator + denominator_extra
        if denominator <= 0.0:
            raise ValueError("risk recommendation has zero information")
        return numerator / denominator

    return RiskRecommendations(
        covariance_only=ratio(old, new),
        centered_marginal=ratio(population_signal + old, new),
        realized_conditional=ratio(conditional_signal, new),
        old_covariance_risk=old,
        new_covariance_risk=new,
    )


def mean_absolute_error(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("MAE inputs must be nonempty and equally sized")
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("MAE inputs must be finite")
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)


def summarize_coefficient_rows(
    rows: Sequence[dict[str, Any]],
    *,
    live_start_step: int,
) -> list[dict[str, Any]]:
    if live_start_step < 0 or not rows:
        raise ValueError("summary requires rows and a nonnegative live start")
    groups = sorted(
        {
            (str(row["schedule"]), str(row["condition"]), int(row["rank"]))
            for row in rows
        }
    )
    output = []
    for schedule, condition, rank in groups:
        selected = [
            row
            for row in rows
            if row["schedule"] == schedule
            and row["condition"] == condition
            and int(row["rank"]) == rank
            and int(row["step"]) >= live_start_step
        ]
        if not selected:
            raise ValueError("summary group has no live rows")

        def mean(name: str) -> float:
            return statistics.fmean(float(row[name]) for row in selected)

        online = [float(row["online_plugin_pi"]) for row in selected]
        marginal = [float(row["centered_marginal_pi"]) for row in selected]
        conditional = [float(row["realized_conditional_pi"]) for row in selected]
        efficient_conditional = [
            float(row["efficient_mle_conditional_pi"]) for row in selected
        ]
        covariance = [float(row["covariance_only_pi"]) for row in selected]
        applied = [float(row["applied_pi"]) for row in selected]
        debiased_values = [
            float(row["debiased_online_pi"])
            for row in selected
            if row["debiased_online_pi"] is not None
        ]
        result: dict[str, Any] = {
            "schedule": schedule,
            "condition": condition,
            "rank": rank,
            "live_transition_count": len(selected),
            "mean_applied_pi": statistics.fmean(applied),
            "mean_covariance_only_pi": statistics.fmean(covariance),
            "mean_centered_marginal_pi": statistics.fmean(marginal),
            "mean_realized_conditional_pi": statistics.fmean(conditional),
            "mean_efficient_mle_conditional_pi": statistics.fmean(
                efficient_conditional
            ),
            "mean_efficient_mle_centered_marginal_pi": mean(
                "efficient_mle_centered_marginal_pi"
            ),
            "mean_online_plugin_pi": statistics.fmean(online),
            "online_to_covariance_mae": mean_absolute_error(online, covariance),
            "online_to_marginal_mae": mean_absolute_error(online, marginal),
            "online_to_conditional_mae": mean_absolute_error(online, conditional),
            "online_to_efficient_mle_conditional_mae": mean_absolute_error(
                online, efficient_conditional
            ),
            "covariance_to_applied_half_mae": mean_absolute_error(
                covariance, [value / 2.0 for value in applied]
            ),
            "mean_population_signal": mean("population_signal_corrected"),
            "mean_anchor_error_energy": mean("anchor_error_energy"),
            "mean_cross_term": mean("cross_term"),
            "mean_conditional_signal": mean("conditional_signal_corrected"),
            "mean_online_signal": mean("online_signal"),
            "mean_old_covariance_shape_risk": mean(
                "old_covariance_shape_risk"
            ),
            "mean_new_covariance_shape_risk": mean(
                "new_covariance_shape_risk"
            ),
            "mean_old_shape_fraction_of_parameter_count": mean(
                "old_shape_fraction_of_parameter_count"
            ),
            "mean_new_shape_fraction_of_parameter_count": mean(
                "new_shape_fraction_of_parameter_count"
            ),
            "mean_online_signal_lift": statistics.fmean(
                left - right for left, right in zip(online, covariance, strict=True)
            ),
            "mean_population_signal_lift": statistics.fmean(
                left - right
                for left, right in zip(marginal, covariance, strict=True)
            ),
        }
        if debiased_values:
            corresponding = [
                row for row in selected if row["debiased_online_pi"] is not None
            ]
            result.update(
                {
                    "mean_debiased_online_pi": statistics.fmean(debiased_values),
                    "debiased_to_marginal_mae": mean_absolute_error(
                        debiased_values,
                        [float(row["centered_marginal_pi"]) for row in corresponding],
                    ),
                    "debiased_to_conditional_mae": mean_absolute_error(
                        debiased_values,
                        [
                            float(row["realized_conditional_pi"])
                            for row in corresponding
                        ],
                    ),
                    "debiased_to_efficient_mle_conditional_mae": (
                        mean_absolute_error(
                            debiased_values,
                            [
                                float(row["efficient_mle_conditional_pi"])
                                for row in corresponding
                            ],
                        )
                    ),
                }
            )
        else:
            result.update(
                {
                    "mean_debiased_online_pi": None,
                    "debiased_to_marginal_mae": None,
                    "debiased_to_conditional_mae": None,
                    "debiased_to_efficient_mle_conditional_mae": None,
                }
            )
        output.append(result)
    return output
