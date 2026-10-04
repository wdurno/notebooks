"""Deformed-MP and empirical unresolved-spectrum calibration for Plan 12."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
from torch import Tensor


TW1_99_QUANTILE = 2.023449


@dataclasses.dataclass(frozen=True)
class PowerLawFit:
    amplitude: float
    exponent: float
    start_rank: int
    stop_rank: int
    log_rmse: float
    r_squared: float

    def mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DeformedMPEdge:
    edge: float
    root_v: float
    second_derivative: float
    fluctuation_scale: float
    upper_quantile: float
    gamma: float
    nearest_pole_distance: float
    regular: bool

    def mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def fit_local_power_law(
    eigenvalues: Tensor,
    *,
    start_rank: int,
    stop_rank: int,
) -> PowerLawFit:
    if eigenvalues.ndim != 1 or start_rank < 1 or stop_rank > eigenvalues.numel() or stop_rank - start_rank < 2:
        raise ValueError("power-law fit needs at least three valid ranked eigenvalues")
    values = eigenvalues[start_rank - 1 : stop_rank].to(dtype=torch.float64)
    if not torch.isfinite(values).all() or bool((values <= 0).any()):
        raise ValueError("power-law eigenvalues must be finite and positive")
    ranks = torch.arange(start_rank, stop_rank + 1, dtype=torch.float64, device=values.device)
    x = torch.log(ranks)
    y = torch.log(values)
    centered = x - x.mean()
    slope = torch.sum(centered * (y - y.mean())) / torch.sum(centered.square())
    intercept = y.mean() - slope * x.mean()
    fitted = intercept + slope * x
    residual = y - fitted
    total = torch.sum((y - y.mean()).square())
    r_squared = 1.0 - float(torch.sum(residual.square()) / total) if float(total) > 0 else 1.0
    exponent = -float(slope)
    if exponent <= 0:
        raise ValueError("fitted spectrum does not decrease with rank")
    return PowerLawFit(
        amplitude=math.exp(float(intercept)),
        exponent=exponent,
        start_rank=start_rank,
        stop_rank=stop_rank,
        log_rmse=float(torch.sqrt(torch.mean(residual.square()))),
        r_squared=r_squared,
    )


def power_law_quadrature(
    fit: PowerLawFit,
    *,
    first_rank: int,
    parameter_count: int,
    nodes: int = 512,
) -> tuple[Tensor, Tensor]:
    if first_rank < 1 or first_rank > parameter_count or nodes < 16:
        raise ValueError("invalid power-law quadrature contract")
    # Uniform-rank midpoint quadrature integrates the induced spectral measure
    # without treating extrapolated ranks as observed eigenvalues.
    edges = torch.linspace(float(first_rank), float(parameter_count + 1), nodes + 1, dtype=torch.float64)
    ranks = (edges[:-1] + edges[1:]) / 2
    values = fit.amplitude * ranks.pow(-fit.exponent)
    weights = (edges[1:] - edges[:-1]) / (parameter_count - first_rank + 1)
    weights /= weights.sum()
    return values, weights


def effective_sample_size_ema(*, batch_size: int, gain: float) -> float:
    if batch_size <= 0 or not 0 < gain <= 1:
        raise ValueError("invalid EMA effective-sample inputs")
    return batch_size * (2.0 - gain) / gain


def _z_terms(v: float, values: Tensor, weights: Tensor, gamma: float) -> tuple[float, float, float]:
    denominator = 1 + values * v
    z = -1 / v + gamma * float(torch.sum(weights * values / denominator))
    first = 1 / v**2 - gamma * float(torch.sum(weights * values.square() / denominator.square()))
    second = -2 / v**3 + 2 * gamma * float(torch.sum(weights * values.pow(3) / denominator.pow(3)))
    return z, first, second


def solve_deformed_mp_edge(
    values: Tensor,
    weights: Tensor,
    *,
    gamma: float,
    effective_sample_size: float,
    tw_quantile: float = TW1_99_QUANTILE,
    iterations: int = 120,
) -> DeformedMPEdge:
    values = values.to(dtype=torch.float64, device="cpu")
    weights = weights.to(dtype=torch.float64, device="cpu")
    if values.ndim != 1 or weights.shape != values.shape or bool((values <= 0).any()):
        raise ValueError("spectral quadrature is invalid")
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("spectral weights must sum to one")
    if gamma <= 0 or effective_sample_size <= 0:
        raise ValueError("MP aspect ratio and effective sample size must be positive")
    maximum = float(values.max())
    left = -1.0 / maximum + max(1e-14, 1e-10 / maximum)
    right = -max(1e-14, 1e-10 / maximum)
    left_value = _z_terms(left, values, weights, gamma)[1]
    right_value = _z_terms(right, values, weights, gamma)[1]
    if not left_value < 0 < right_value:
        raise RuntimeError("could not bracket the regular right deformed-MP edge")
    for _ in range(iterations):
        midpoint = (left + right) / 2
        derivative = _z_terms(midpoint, values, weights, gamma)[1]
        if derivative > 0:
            right = midpoint
        else:
            left = midpoint
    root = (left + right) / 2
    edge, _, second = _z_terms(root, values, weights, gamma)
    scale = math.copysign(abs(second / 2) ** (1 / 3), second) * effective_sample_size ** (-2 / 3)
    poles = -torch.reciprocal(values)
    pole_distance = float(torch.min(torch.abs(poles - root)))
    regular = second > 0 and pole_distance > 1e-9 * max(1.0, abs(root))
    return DeformedMPEdge(
        edge=edge,
        root_v=root,
        second_derivative=second,
        fluctuation_scale=scale,
        upper_quantile=edge + scale * tw_quantile,
        gamma=gamma,
        nearest_pole_distance=pole_distance,
        regular=regular,
    )


def empirical_tail_maxima(
    scores: Tensor,
    resolved_basis: Tensor,
    *,
    sample_size: int,
    resamples: int,
    seed: int,
    power_iterations: int = 12,
    block_size: int = 16,
) -> Tensor:
    if scores.ndim != 2 or resolved_basis.ndim != 2 or scores.shape[1] != resolved_basis.shape[0]:
        raise ValueError("scores and resolved basis are incompatible")
    if sample_size <= 0 or resamples <= 0:
        raise ValueError("resampling sizes must be positive")
    device = scores.device
    dtype = scores.dtype
    projected = scores - (scores @ resolved_basis) @ resolved_basis.mT
    generator = torch.Generator(device=device).manual_seed(seed)
    maxima = []
    for start in range(0, resamples, block_size):
        count = min(block_size, resamples - start)
        indices = torch.randint(scores.shape[0], (count, sample_size), generator=generator, device=device)
        draws = projected[indices]
        vectors = torch.randn(count, projected.shape[1], generator=generator, device=device, dtype=dtype)
        vectors /= torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        for _ in range(power_iterations):
            products = torch.einsum("bnp,bp->bn", draws, vectors)
            vectors = torch.einsum("bnp,bn->bp", draws, products) / sample_size
            vectors /= torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        products = torch.einsum("bnp,bp->bn", draws, vectors)
        maxima.append(products.square().mean(dim=1).cpu())
    return torch.cat(maxima)


def empirical_quantile(values: Tensor, probability: float = 0.99) -> float:
    if values.ndim != 1 or values.numel() == 0 or not 0 < probability < 1:
        raise ValueError("invalid empirical quantile request")
    return float(torch.quantile(values.to(torch.float64), probability, interpolation="higher"))


def pit_value(maxima: Tensor, observed: float) -> float:
    if maxima.ndim != 1 or maxima.numel() == 0:
        raise ValueError("PIT calibration requires empirical maxima")
    return float((maxima <= observed).double().mean())


def shrunk_pit_probability(
    history: tuple[float, ...],
    *,
    target: float = 0.99,
    prior_weight: float = 20.0,
) -> float:
    if not history:
        return target
    empirical = float(torch.quantile(torch.tensor(history, dtype=torch.float64), target))
    weight = len(history) / (len(history) + prior_weight)
    return min(0.999, max(0.5, (1 - weight) * target + weight * empirical))
