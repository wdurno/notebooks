"""Plan 15 likelihood-Fisher penalties, differentiated into both memory blocks.

All label expectations use the model probabilities, not observed-label gradient
outer products. Scores are gradients of log probability, or minus NLL gradients.
No function constructs a per-sample Hessian. Returned curvature values and dense
Fisher accumulators are float64; model derivatives retain theta's precision.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.func import jacrev, jvp, vmap

from src.parameters import ParameterLayout

from .model import FisherMemoryCNN, flat_logits, memory_indices


CURVATURE_SCHEMA_VERSION = 1


def _validate_inputs(layout: ParameterLayout, theta: Tensor, x: Tensor) -> None:
    if theta.ndim != 1 or theta.numel() != layout.total_numel:
        raise ValueError("theta must match the flat parameter layout")
    if theta.dtype not in (torch.float32, torch.float64):
        raise ValueError("theta must have float32 or float64 dtype")
    if x.ndim != 4 or x.shape[0] == 0 or x.shape[1] != 1:
        raise ValueError("inputs must be a nonempty batch of single-channel images")
    if x.dtype != theta.dtype or x.device != theta.device:
        raise ValueError("inputs and theta must share dtype and device")


def _stm_directions(layout: ParameterLayout, theta: Tensor, values: Tensor) -> Tensor:
    """Embed fixed STM vectors; reject accidentally supplied LTM motion."""
    ltm, stm = memory_indices(layout)
    values = torch.as_tensor(values, device=theta.device, dtype=theta.dtype).detach()
    if values.ndim != 2 or values.shape[0] == 0 or not torch.isfinite(values).all():
        raise ValueError("directions must be a finite nonempty (K, dimension) array")
    if values.shape[1] == stm.numel():
        result = theta.new_zeros(values.shape[0], layout.total_numel)
        return result.index_copy(1, stm.to(theta.device), values)
    if values.shape[1] != layout.total_numel:
        raise ValueError("directions must use STM-only or full coordinates")
    if torch.count_nonzero(values.index_select(1, ltm.to(theta.device))):
        raise ValueError("full-coordinate directions must be zero in LTM")
    return values


def _indices(indices: Tensor | None, theta: Tensor) -> Tensor:
    if indices is None:
        return torch.arange(theta.numel(), device=theta.device)
    selected = torch.as_tensor(indices, device=theta.device)
    if (selected.ndim != 1 or selected.numel() == 0
            or selected.dtype not in (torch.int32, torch.int64)):
        raise ValueError("indices must be a nonempty integer vector")
    if (selected.min() < 0 or selected.max() >= theta.numel()
            or selected.unique().numel() != selected.numel()):
        raise ValueError("indices must be unique global coordinates in range")
    return selected.long()


def _chunk_size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("chunk_size must be a positive integer")


def jvp_values(
    model: FisherMemoryCNN,
    layout: ParameterLayout,
    theta: Tensor,
    x: Tensor,
    probes: Tensor,
) -> Tensor:
    """Return exact directional Fisher values with shape (K, N).

    Probes are fixed (detached) rows of shape (K, d_S) or (K, d), with zero
    LTM entries in the latter case. No probe average or d_S normalization is
    applied here. Rademacher STM probes give an unbiased *trace* on averaging;
    an arbitrary supplied vector still gives its exact quadratic form.
    """
    _validate_inputs(layout, theta, x)
    directions = _stm_directions(layout, theta, probes)

    def one(direction: Tensor) -> Tensor:
        logits, tangent = jvp(
            lambda parameter: flat_logits(model, layout, parameter, x),
            (theta,), (direction,),
        )
        probabilities = logits.double().softmax(-1)
        tangent = tangent.double()
        centered = tangent - (probabilities * tangent).sum(-1, keepdim=True)
        return (probabilities * centered.square()).sum(-1)

    return vmap(one)(directions)


def kl_values(
    model: FisherMemoryCNN,
    layout: ParameterLayout,
    theta: Tensor,
    x: Tensor,
    probes: Tensor,
    h: float,
) -> Tensor:
    """Return symmetric KL stencil values (K, N) at theta +/- h*v/2.

    (KL(p+||p-) + KL(p-||p+))/h**2 approaches v.T F_SS v.
    Both predictive distributions remain differentiable into LTM and STM.
    The caller fixes h/probes throughout each solve and handles normalization.
    """
    _validate_inputs(layout, theta, x)
    if isinstance(h, bool) or not math.isfinite(float(h)) or h <= 0:
        raise ValueError("h must be a finite positive coordinate step")
    directions = _stm_directions(layout, theta, probes)

    def one(direction: Tensor) -> Tensor:
        plus = flat_logits(model, layout, theta + (h / 2) * direction, x)
        minus = flat_logits(model, layout, theta - (h / 2) * direction, x)
        lp = plus.double().log_softmax(-1)
        lm = minus.double().log_softmax(-1)
        # Jeffreys' identity avoids cancellation between two separate KL sums.
        return ((lp.exp() - lm.exp()) * (lp - lm)).sum(-1) / h**2

    return vmap(one)(directions)


def step_kl_values(
    model: FisherMemoryCNN,
    layout: ParameterLayout,
    theta: Tensor,
    x: Tensor,
    delta: Tensor,
    *,
    min_norm: float = 1e-12,
) -> Tensor:
    """Return 2*KL(p_theta || p_theta+delta)/||delta||^2 per input.

    Delta has shape (d_S,) or (d,), with zero LTM entries in full coordinates.
    It is detached internally, including its normalizer; *neither* predictive
    branch is detached. The caller bounds the proposal radius and records the
    unsupported flag when ||delta|| <= min_norm. Such directions return a
    differentiable zero rather than a regularized or unstable division.
    """
    _validate_inputs(layout, theta, x)
    if isinstance(min_norm, bool) or not math.isfinite(float(min_norm)) or min_norm < 0:
        raise ValueError("min_norm must be finite and nonnegative")
    delta = torch.as_tensor(delta, device=theta.device, dtype=theta.dtype)
    if delta.ndim != 1:
        raise ValueError("delta must be a vector")
    direction = _stm_directions(layout, theta, delta.unsqueeze(0))[0]
    squared_norm = direction.double().square().sum()
    if squared_norm <= min_norm**2:
        return theta.double().sum() * 0 + theta.new_zeros(x.shape[0], dtype=torch.float64)
    base = flat_logits(model, layout, theta, x).double().log_softmax(-1)
    moved = flat_logits(model, layout, theta + direction, x).double().log_softmax(-1)
    ratio = moved - base
    # Add sum(q-p)=0 to KL: p*(exp(r)-1-r) avoids first-order cancellation.
    # A local series is stable even when KL is far below machine epsilon;
    # the outer formula avoids overflow in exp(r) for saturated probabilities.
    small = ratio.abs() < 1e-3
    local = torch.where(small, ratio, torch.zeros_like(ratio))
    remainder = local.square() * (
        0.5 + local * (1 / 6 + local * (1 / 24 + local * (1 / 120 + local / 720)))
    )
    probabilities = base.exp()
    terms = torch.where(
        small, probabilities * remainder,
        moved.exp() - probabilities - probabilities * ratio,
    )
    return 2 * terms.sum(-1) / squared_norm


def _score_rows(
    model: FisherMemoryCNN, layout: ParameterLayout, theta: Tensor, x: Tensor
) -> tuple[Tensor, Tensor]:
    def log_probabilities(parameter: Tensor, image: Tensor) -> tuple[Tensor, Tensor]:
        logits = flat_logits(model, layout, parameter, image.unsqueeze(0))[0]
        logp = logits.double().log_softmax(-1)
        return logp, logp.exp()

    return vmap(jacrev(log_probabilities, argnums=0, has_aux=True), in_dims=(None, 0))(
        theta, x
    )


def reference_fisher(
    model: FisherMemoryCNN,
    layout: ParameterLayout,
    theta: Tensor,
    x: Tensor,
    *,
    indices: Tensor | None = None,
    weights: Tensor | None = None,
    chunk_size: int = 8,
) -> Tensor:
    """Detached float64 weighted-mean Fisher, integrating all ten model labels.

    Normalize the nonnegative input weights to unit total mass (uniform if
    omitted). Select a principal block after computing full-coordinate scores.
    Derivatives use theta's dtype; accumulation and label weights use float64.
    Supply a float64 theta and inputs for double-precision derivative audits.
    Scores are streamed in bounded chunks; no score bank or Hessian is retained.
    """
    _validate_inputs(layout, theta, x)
    _chunk_size(chunk_size)
    selected = _indices(indices, theta)
    mass = (
        torch.ones(x.shape[0], device=theta.device, dtype=torch.float64)
        if weights is None else torch.as_tensor(weights, device=theta.device).detach().double()
    )
    if (mass.shape != (x.shape[0],) or not torch.isfinite(mass).all()
            or (mass < 0).any() or not torch.isfinite(mass.sum()) or mass.sum() <= 0):
        raise ValueError("weights must be finite, nonnegative, with positive total mass")
    mass = mass / mass.sum()
    total = torch.zeros(selected.numel(), selected.numel(), device=theta.device, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            stop = min(start + chunk_size, x.shape[0])
            scores, probabilities = _score_rows(model, layout, theta.detach(), x[start:stop])
            scores = scores.index_select(-1, selected).double().reshape(-1, selected.numel())
            row_weights = (mass[start:stop, None] * probabilities).reshape(-1)
            total.add_(scores.mT @ (row_weights[:, None] * scores))
    if not torch.isfinite(total).all():
        raise ValueError("model generated a nonfinite reference Fisher")
    return (total + total.mT) / 2


def exact_trace(
    model: FisherMemoryCNN,
    layout: ParameterLayout,
    theta: Tensor,
    x: Tensor,
    indices: Tensor | None = None,
    *,
    chunk_size: int = 8,
) -> Tensor:
    """Differentiable per-input exact model-label Fisher traces, shape (N,).

    ``indices`` selects the traced principal block, not which parameters get
    gradients. Both the scores and label probabilities retain their derivatives
    with respect to the entire theta. No input averaging is performed.
    """
    _validate_inputs(layout, theta, x)
    _chunk_size(chunk_size)
    selected = _indices(indices, theta)
    parts = []
    for start in range(0, x.shape[0], chunk_size):
        scores, probabilities = _score_rows(model, layout, theta, x[start:start + chunk_size])
        squared = scores.index_select(-1, selected).double().square().sum(-1)
        parts.append((probabilities * squared).sum(-1))
    return torch.cat(parts)


def exact_directional_values(
    model: FisherMemoryCNN,
    layout: ParameterLayout,
    theta: Tensor,
    x: Tensor,
    probes: Tensor,
    *,
    chunk_size: int = 8,
) -> Tensor:
    """Differentiable score-Jacobian reference for STM quadratic forms (K, N).

    Unlike ``jvp_values``, this explicitly integrates all model-label score
    rows. It is an independent small-model numerical audit, not a cheap
    production implementation. No matrix Hessians or Fisher matrices are made.
    """
    _validate_inputs(layout, theta, x)
    _chunk_size(chunk_size)
    directions = _stm_directions(layout, theta, probes).double()
    parts = []
    for start in range(0, x.shape[0], chunk_size):
        scores, probabilities = _score_rows(model, layout, theta, x[start:start + chunk_size])
        products = torch.einsum("ncd,kd->knc", scores.double(), directions)
        parts.append((products.square() * probabilities[None]).sum(-1))
    return torch.cat(parts, dim=1)
