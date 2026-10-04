"""Penalized sandwich-covariance diagnostics for local Plan 12 fits."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.representations import FisherRepresentation


def dense_operator(fisher: Tensor | FisherRepresentation) -> Tensor:
    if isinstance(fisher, Tensor):
        return fisher
    if hasattr(fisher, "to_dense"):
        return fisher.to_dense()
    identity = torch.eye(fisher.shape[0], device=fisher.device, dtype=fisher.dtype)
    return fisher.matvec(identity)


def penalized_sandwich(
    score_or_loss_gradients: Tensor,
    penalty_fisher: Tensor | FisherRepresentation,
    *,
    beta: float,
    resolved_basis: Tensor,
) -> tuple[dict[str, Any], Tensor]:
    """Return covariance diagnostics using ``A^-1 B A^-1 / m``.

    Loss gradients are negative scores, so their covariance outer product is
    identical to the score covariance used by the sandwich formula.
    """

    if score_or_loss_gradients.ndim != 2 or score_or_loss_gradients.shape[1] != penalty_fisher.shape[0]:
        raise ValueError("gradient matrix and penalty Fisher are incompatible")
    gradients = score_or_loss_gradients.to(device=penalty_fisher.device, dtype=penalty_fisher.dtype)
    count = gradients.shape[0]
    meat = gradients.mT @ gradients / count
    penalty = dense_operator(penalty_fisher)
    bread = (meat + float(beta) * penalty)
    bread = (bread + bread.mT) / 2
    scale = float(torch.trace(bread) / bread.shape[0])
    damping = max(abs(scale) * 1e-8, torch.finfo(bread.dtype).eps * 100)
    identity = torch.eye(bread.shape[0], device=bread.device, dtype=bread.dtype)
    solved = torch.linalg.solve(bread + damping * identity, gradients.mT) / count
    total = float(solved.square().sum())
    basis = resolved_basis.to(device=solved.device, dtype=solved.dtype)
    resolved = float((basis.mT @ solved).square().sum())
    diagnostics = {
        "sample_count": count,
        "beta": float(beta),
        "numerical_damping": damping,
        "bread_trace": float(torch.trace(bread)),
        "meat_trace": float(torch.trace(meat)),
        "covariance_trace": total,
        "resolved_covariance_trace": resolved,
        "unresolved_covariance_trace": max(total - resolved, 0.0),
        "condition_number_with_numerical_damping": float(torch.linalg.cond(bread + damping * identity)),
    }
    return diagnostics, solved.detach()
