"""Finite evidence weights and one-time episode consolidation in mean-NLL units."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def _sizes(batch_sizes: Sequence[int]) -> tuple[int, ...]:
    sizes = tuple(batch_sizes)
    if not sizes or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
                        for n in sizes):
        raise ValueError("batch_sizes must be a nonempty sequence of positive integers")
    return sizes


def weights(
    batch_sizes: Sequence[int], half_life: float, *, device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Normalized per-example weights, oldest arrival first, age in arrivals."""
    sizes = _sizes(batch_sizes)
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be finite and positive")
    if not dtype.is_floating_point:
        raise ValueError("weights require a floating dtype")
    ages = torch.arange(len(sizes) - 1, -1, -1, device=device, dtype=torch.float64)
    unnormalized = torch.repeat_interleave(
        torch.exp2(-ages / half_life), torch.tensor(sizes, device=device)
    )
    return (unnormalized / unnormalized.sum()).to(dtype=dtype)


def _normalized_weights(value: Tensor) -> Tensor:
    value = torch.as_tensor(value).detach().to(dtype=torch.float64)
    if value.ndim != 1 or value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError("weights must be a nonempty finite vector")
    if (value < 0).any() or not torch.isclose(value.sum(), value.new_tensor(1.),
                                           atol=1e-7, rtol=1e-7):
        raise ValueError("weights must be nonnegative and normalized")
    return value / value.sum()


def effective_sample_size(value: Tensor) -> float:
    """Actual finite-weight ESS, not the infinite-history approximation."""
    return float(1 / _normalized_weights(value).square().sum())


def gradient_noise_covariance(covariances: Tensor, value: Tensor) -> Tensor:
    """Independent-observation V = sum(a_i**2 Cov(g_i)), with g = -score.

    A single matrix assumes locally common covariance; a stack allows one
    covariance per observation. Temporal cross-covariances are not modeled.
    """
    a = _normalized_weights(value)
    covariances = torch.as_tensor(covariances, device=a.device, dtype=torch.float64)
    if not torch.isfinite(covariances).all():
        raise ValueError("covariances must be finite")
    if covariances.ndim == 2 and covariances.shape[0] == covariances.shape[1]:
        result = a.square().sum() * covariances
    elif (covariances.ndim == 3 and covariances.shape[0] == a.numel()
          and covariances.shape[1] == covariances.shape[2]):
        result = torch.einsum("n,nij->ij", a.square(), covariances)
    else:
        raise ValueError("expected one square covariance or one per observation")
    return (result + result.mT) / 2


def evidence_diagnostics(
    batch_sizes: Sequence[int], half_life: float, *,
    observation_ids: Sequence | None = None, labels: Tensor | None = None,
    expected_probabilities: Tensor | None = None,
) -> dict:
    """Evidence counts/ages; input IDs here may be repeated dataset identities."""
    sizes = _sizes(batch_sizes)
    a = weights(sizes, half_life)
    ages = torch.repeat_interleave(torch.arange(len(sizes) - 1, -1, -1),
                                   torch.tensor(sizes)).double()
    n = a.numel()
    if observation_ids is not None and len(observation_ids) != n:
        raise ValueError("observation_ids must have one entry per example")
    result = {
        "arrivals": len(sizes), "buffer_size": n,
        "unique_observations": None if observation_ids is None else len(set(observation_ids)),
        "ess": effective_sample_size(a), "sum_squared_weights": float(a.square().sum()),
        "half_life_arrivals": float(half_life),
        "mean_age_arrivals": float(a @ ages),
        "age_variance_arrivals": float(a @ (ages - a @ ages).square()),
        "max_age_arrivals": len(sizes) - 1,
        "arrival_weight_mass": [float(part.sum()) for part in a.split(sizes)],
        "realized_digit9_count": None, "expected_digit9_count": None,
    }
    if labels is not None:
        labels = torch.as_tensor(labels).cpu()
        if labels.shape != (n,):
            raise ValueError("labels must have one entry per example")
        result["realized_digit9_count"] = int((labels == 9).sum())
    if expected_probabilities is not None:
        p = torch.as_tensor(expected_probabilities, dtype=torch.float64).cpu()
        if p.shape != (n,) or not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("expected_probabilities must contain one probability per example")
        result["expected_digit9_count"] = float(p.sum())
    return result


def retention_precision(archive: Tensor, omega: float, ridge_ratio: float = 0.) -> tuple[Tensor, float]:
    """P = omega * (Omega + kappa I); ridge is never inserted into Omega."""
    if not math.isfinite(omega) or omega < 0 or not math.isfinite(ridge_ratio) or ridge_ratio < 0:
        raise ValueError("omega and ridge_ratio must be finite and nonnegative")
    archive = torch.as_tensor(archive, dtype=torch.float64).detach()
    if (archive.ndim != 2 or archive.shape[0] != archive.shape[1]
            or archive.shape[0] == 0 or not torch.isfinite(archive).all()):
        raise ValueError("archive must be a finite nonempty square matrix")
    archive = (archive + archive.mT) / 2
    kappa = ridge_ratio * float(archive.trace() / archive.shape[0])
    if kappa < 0:
        raise ValueError("archive has negative mean curvature")
    precision = omega * (archive + kappa * torch.eye(archive.shape[0],
                                                   device=archive.device, dtype=archive.dtype))
    return precision, kappa


def consolidate(
    archive: Tensor, theta: Tensor, new_fisher: Tensor, *, omega: float,
    arrivals: int, episode_id: str, observation_ids: Sequence[str | int],
    ledger: Mapping[str, Sequence[str | int]] | None = None,
    kind: str = "lr", rank: int = 8, layer_indices=None, ridge_ratio: float = 0.,
) -> dict:
    """Pure episode-end update; checkpoint the result before clearing the buffer.

    The ledger uses unique *exposure* IDs, not reusable MNIST dataset indices.
    Replaying an already consolidated episode or exposure raises rather than
    double-counting it. No input tensor or ledger is mutated. Persistence and
    episode-end scheduling are the caller's responsibility.
    """
    from .fisher import represent

    if isinstance(arrivals, bool) or not isinstance(arrivals, int) or arrivals <= 0:
        raise ValueError("arrivals must be a positive integer")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode_id must be a nonempty string")
    previous = {} if ledger is None else {key: list(ids) for key, ids in ledger.items()}
    if episode_id in previous:
        raise ValueError("episode already consolidated")
    ids = list(observation_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("consolidation requires unique exposure IDs")
    old_ids = {value for group in previous.values() for value in group}
    if old_ids.intersection(ids):
        raise ValueError("exposure already consolidated")
    archive = torch.as_tensor(archive, dtype=torch.float64).detach()
    new_fisher = torch.as_tensor(new_fisher, device=archive.device, dtype=torch.float64).detach()
    theta = torch.as_tensor(theta, device=archive.device).detach()
    if (archive.ndim != 2 or archive.shape != new_fisher.shape or theta.ndim != 1
            or archive.shape != (theta.numel(), theta.numel()) or not torch.isfinite(theta).all()):
        raise ValueError("archive, full-coordinate Fisher, and anchor dimensions must agree")
    retention_precision(archive, omega, ridge_ratio)
    pi = 1 / (1 + omega)
    eta = 1. if pi == 1 else -math.expm1(arrivals * math.log1p(-pi))
    updated = represent((1 - eta) * archive + eta * new_fisher,
                        kind=kind, rank=rank, layer_indices=layer_indices)
    precision, kappa = retention_precision(updated, omega, ridge_ratio)
    previous[episode_id] = ids
    return {"archive": updated, "anchor": theta.clone(), "precision": precision,
            "kappa": kappa, "eta": eta, "pi": pi, "ledger": previous,
            "episode_id": episode_id, "arrivals": arrivals}
