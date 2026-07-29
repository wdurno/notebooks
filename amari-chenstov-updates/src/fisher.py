"""Empirical Fisher and linearized Fisher update estimators."""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor


@dataclasses.dataclass(frozen=True)
class LFUBatchEstimate:
    """Dense statistics calculated from per-sample loss derivatives.

    Gradients use the negative-log-likelihood convention ``g = -s``.
    """

    fisher: Tensor
    amari_chentsov: Tensor
    residual: Tensor

    @property
    def full(self) -> Tensor:
        return self.amari_chentsov + self.residual


def _validate_samples(
    gradients: Tensor,
    *,
    hvps: Tensor | None = None,
    direction: Tensor | None = None,
) -> None:
    if gradients.ndim != 2 or gradients.shape[0] == 0:
        raise ValueError("gradients must have shape (nonempty_batch, parameters)")
    if hvps is not None and hvps.shape != gradients.shape:
        raise ValueError("hvps must have the same shape as gradients")
    if direction is not None and (
        direction.ndim != 1 or direction.shape[0] != gradients.shape[1]
    ):
        raise ValueError(
            f"direction must have shape ({gradients.shape[1]},), "
            f"got {tuple(direction.shape)}"
        )
    if hvps is not None and (
        hvps.device != gradients.device or hvps.dtype != gradients.dtype
    ):
        raise ValueError("gradients and hvps must share dtype and device")
    if direction is not None and (
        direction.device != gradients.device
        or direction.dtype != gradients.dtype
    ):
        raise ValueError("gradients and direction must share dtype and device")


def empirical_fisher(gradients: Tensor) -> Tensor:
    """Return ``mean(g g^T)``; this equals ``mean(s s^T)`` because ``g=-s``."""

    _validate_samples(gradients)
    return gradients.mT @ gradients / gradients.shape[0]


def dense_lfu_estimate(
    gradients: Tensor,
    hvps: Tensor,
    direction: Tensor,
) -> LFUBatchEstimate:
    """Calculate dense empirical Fisher, AC, and residual LFU statistics."""

    _validate_samples(gradients, hvps=hvps, direction=direction)
    batch_size = gradients.shape[0]
    directional_scores = gradients @ direction
    weighted_gradients = directional_scores.unsqueeze(1) * gradients

    amari_chentsov = -(gradients.mT @ weighted_gradients) / batch_size
    residual = (
        hvps.mT @ gradients + gradients.mT @ hvps
    ) / batch_size
    return LFUBatchEstimate(
        fisher=empirical_fisher(gradients),
        amari_chentsov=amari_chentsov,
        residual=residual,
    )


def sample_lfu_factorization(
    gradient: Tensor,
    hvp: Tensor,
    direction: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``U, B`` such that one sample's full LFU is ``U @ B @ U.T``."""

    if gradient.ndim != 1:
        raise ValueError("gradient must be a vector")
    if hvp.shape != gradient.shape or direction.shape != gradient.shape:
        raise ValueError("gradient, hvp, and direction must have equal shapes")
    if (
        hvp.device != gradient.device
        or direction.device != gradient.device
        or hvp.dtype != gradient.dtype
        or direction.dtype != gradient.dtype
    ):
        raise ValueError("gradient, hvp, and direction must share dtype and device")

    one = gradient.new_tensor(1.0)
    zero = gradient.new_tensor(0.0)
    coefficient = -(direction @ gradient)
    basis = torch.stack((gradient, hvp), dim=1)
    core = torch.stack(
        (
            torch.stack((coefficient, one)),
            torch.stack((one, zero)),
        )
    )
    return basis, core


def _validate_probes(gradients: Tensor, probes: Tensor) -> bool:
    if probes.ndim == 1:
        if probes.shape[0] != gradients.shape[1]:
            raise ValueError(
                f"probe must have shape ({gradients.shape[1]},), "
                f"got {tuple(probes.shape)}"
            )
        is_vector = True
    elif probes.ndim == 2:
        if probes.shape[0] != gradients.shape[1]:
            raise ValueError(
                f"probes must have shape ({gradients.shape[1]}, columns), "
                f"got {tuple(probes.shape)}"
            )
        is_vector = False
    else:
        raise ValueError("probes must be a vector or a column matrix")
    if probes.device != gradients.device or probes.dtype != gradients.dtype:
        raise ValueError("gradients and probes must share dtype and device")
    return is_vector


def _as_probe_matrix(probes: Tensor) -> Tensor:
    return probes.unsqueeze(1) if probes.ndim == 1 else probes


def _restore_probe_shape(result: Tensor, was_vector: bool) -> Tensor:
    return result.squeeze(1) if was_vector else result


def apply_empirical_fisher(gradients: Tensor, probes: Tensor) -> Tensor:
    """Apply the empirical Fisher without materializing its dense matrix."""

    _validate_samples(gradients)
    was_vector = _validate_probes(gradients, probes)
    probe_matrix = _as_probe_matrix(probes)
    result = gradients.mT @ (gradients @ probe_matrix) / gradients.shape[0]
    return _restore_probe_shape(result, was_vector)


def apply_amari_chentsov(
    gradients: Tensor,
    direction: Tensor,
    probes: Tensor,
) -> Tensor:
    """Apply ``-mean((u^T g) g g^T)`` without materializing it."""

    _validate_samples(gradients, direction=direction)
    was_vector = _validate_probes(gradients, probes)
    probe_matrix = _as_probe_matrix(probes)
    weights = (gradients @ direction).unsqueeze(1)
    result = (
        -gradients.mT @ (weights * (gradients @ probe_matrix))
        / gradients.shape[0]
    )
    return _restore_probe_shape(result, was_vector)


def apply_residual(
    gradients: Tensor,
    hvps: Tensor,
    probes: Tensor,
) -> Tensor:
    """Apply ``mean(H_u g^T + g H_u^T)`` without materializing it."""

    _validate_samples(gradients, hvps=hvps)
    was_vector = _validate_probes(gradients, probes)
    probe_matrix = _as_probe_matrix(probes)
    result = (
        hvps.mT @ (gradients @ probe_matrix)
        + gradients.mT @ (hvps @ probe_matrix)
    ) / gradients.shape[0]
    return _restore_probe_shape(result, was_vector)


def apply_full_lfu(
    gradients: Tensor,
    hvps: Tensor,
    direction: Tensor,
    probes: Tensor,
) -> Tensor:
    """Apply the full AC-plus-residual LFU without materializing it."""

    return apply_amari_chentsov(
        gradients,
        direction,
        probes,
    ) + apply_residual(gradients, hvps, probes)
