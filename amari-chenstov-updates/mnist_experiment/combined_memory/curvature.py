"""Float64 Gaussian diagonals and matched finite changes, without square matrices.

For loss gradients g = -s, observed scores are s = J^T(y-f)/4. The single
likelihood Fisher integrates model labels: diag(F) = sum_outputs J**2 / 4.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.func import grad, jacrev, jvp, vmap

from .model import INPUT_SHAPE, OUTPUT_DIM, flat_prediction, gaussian_nll, memory_indices


CURVATURE_SCHEMA_VERSION = 1
SIGMA2 = 4.


def _inputs(layout, theta, x, chunk_size, output_chunk_size=6):
    if (theta.ndim != 1 or theta.numel() != layout.total_numel
            or theta.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(theta).all())):
        raise ValueError("theta must be a finite floating vector matching the layout")
    if (x.ndim != 4 or tuple(x.shape[1:]) != INPUT_SHAPE or not len(x)
            or x.dtype not in (torch.float32, torch.float64) or x.device != theta.device):
        raise ValueError("nonempty floating (N,2,120,120) inputs must share theta's device")
    for name, value in (("chunk_size", chunk_size), ("output_chunk_size", output_chunk_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _indices(theta, indices):
    selected = (torch.arange(theta.numel(), device=theta.device) if indices is None
                else torch.as_tensor(indices, device=theta.device))
    if (selected.ndim != 1 or not len(selected) or selected.dtype not in (torch.int32, torch.int64)
            or bool(((selected < 0) | (selected >= theta.numel())).any())
            or selected.unique().numel() != selected.numel()):
        raise ValueError("indices must be nonempty unique integer coordinates in range")
    return selected.long()


def _jacobian(model, layout, fixed, images, selected, start, width):
    def one(value, image):
        full = fixed.index_copy(0, selected, value)
        return flat_prediction(model, layout, full, image.unsqueeze(0))[0, start:start + width]

    with torch.no_grad():
        rows = vmap(jacrev(one), in_dims=(None, 0))(fixed[selected], images)
    rows = rows.detach()
    if not bool(torch.isfinite(rows).all()):
        raise FloatingPointError("nonfinite Gaussian output Jacobians")
    return rows


def fisher_diagonal(model, layout, theta, x, indices=None, chunk_size=2,
                    output_chunk_size=6, check=lambda: None):
    """Detached float64 vector; sum all 54 outputs and average only grids.

    Float64 copies are used *before* differentiation, even for float32 training
    models. Input/output chunking bounds retained Jacobians, not the estimand.
    """
    _inputs(layout, theta, x, chunk_size, output_chunk_size)
    selected = _indices(theta, indices)
    fixed = theta.detach().double()
    total = torch.zeros(len(selected), dtype=torch.float64, device=theta.device)
    for images in x.split(chunk_size):
        images = images.detach().double()
        for start in range(0, OUTPUT_DIM, output_chunk_size):
            check()
            rows = _jacobian(model, layout, fixed, images, selected, start, output_chunk_size)
            total.add_(rows.square().sum((0, 1)))
    check()
    total.div_(SIGMA2 * len(x))
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("nonfinite Gaussian Fisher diagonal")
    return total


def paired_change(model, layout, before, after, x, indices=None, chunk_size=2,
                  output_chunk_size=6, check=lambda: None):
    """Return (finite change, endpoint diagonal, diagnostics) on matched inputs.

    Accumulate (Jplus-Jminus)*(Jplus+Jminus)/(4*N), never divide by a
    displacement norm or a stencil step. Do not project the signed change.
    """
    _inputs(layout, before, x, chunk_size, output_chunk_size)
    _inputs(layout, after, x, chunk_size, output_chunk_size)
    if before.device != after.device:
        raise ValueError("both accepted endpoints must share a device")
    selected = _indices(after, indices)
    minus, plus = before.detach().double(), after.detach().double()
    zero_move = torch.equal(minus, plus)
    delta = torch.zeros(len(selected), dtype=torch.float64, device=after.device)
    zplus, zminus = torch.zeros_like(delta), torch.zeros_like(delta)
    for images in x.split(chunk_size):
        images = images.detach().double()
        for start in range(0, OUTPUT_DIM, output_chunk_size):
            check()
            jp = _jacobian(model, layout, plus, images, selected, start, output_chunk_size)
            check()
            jm = jp if zero_move else _jacobian(
                model, layout, minus, images, selected, start, output_chunk_size)
            delta.add_(((jp - jm) * (jp + jm)).sum((0, 1)))
            zplus.add_(jp.square().sum((0, 1)))
            zminus.add_(jm.square().sum((0, 1)))
    check()
    for value in (delta, zplus, zminus):
        value.div_(SIGMA2 * len(x))
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("nonfinite paired Fisher change")
    error = (delta - (zplus - zminus)).norm()
    diagnostics = dict(zero_move=zero_move, correction_norm=float(delta.norm()),
                       fresh_norm=float(zplus.norm()), trace_before=float(zminus.sum()),
                       trace_after=float(zplus.sum()), negative_fraction=float((delta < 0).double().mean()),
                       direct_difference_error=float(error),
                       direct_difference_relative_error=float(error / delta.norm().clamp_min(1e-30)),
                       input_count=len(x), output_count=OUTPUT_DIM,
                       parameter_count=len(selected), derivative_dtype="float64")
    return delta, zplus, diagnostics


def observed_scores(model, layout, theta, x, y, indices=None, chunk_size=2,
                    check=lambda: None):
    """Per-grid *positive* scores, float64 (N,d), not a batch-mean outer product."""
    _inputs(layout, theta, x, chunk_size)
    if (y.shape != (len(x), OUTPUT_DIM) or not y.is_floating_point()
            or y.device != theta.device):
        raise ValueError("floating targets (N,54) must share theta's device")
    fixed = theta.detach().double()
    selected = _indices(theta, indices)

    def log_probability(value, image, target):
        full = fixed.index_copy(0, selected, value)
        pred = flat_prediction(model, layout, full, image.unsqueeze(0))
        return -gaussian_nll(pred, target.unsqueeze(0)).sum()

    pieces = []
    for start in range(0, len(x), chunk_size):
        check()
        with torch.no_grad():
            scores = vmap(grad(log_probability), in_dims=(None, 0, 0))(
                fixed[selected], x[start:start + chunk_size].detach().double(),
                y[start:start + chunk_size].detach().double())
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("nonfinite observed-label scores")
        pieces.append(scores.detach())
    check()
    return torch.cat(pieces)


def _directions(layout, theta, probes):
    ltm, stm = memory_indices(layout)
    values = torch.as_tensor(probes, device=theta.device, dtype=theta.dtype).detach()
    if (values.ndim != 2 or not len(values) or not stm.numel()
            or not bool(torch.isfinite(values).all())):
        raise ValueError("finite nonempty (K,108) STM or full-chart probes required")
    if values.shape[1] == stm.numel():
        return theta.new_zeros(len(values), theta.numel()).index_copy(1, stm.to(theta.device), values)
    if values.shape[1] != theta.numel() or bool(values[:, ltm.to(theta.device)].count_nonzero()):
        raise ValueError("full-chart probes must have zero LTM coordinates")
    return values


def fd_values(model, layout, theta, x, probes, step):
    """Differentiable symmetric Gaussian KL/h**2 (K,N), at theta +/- h*v/2.

    Both branches retain all parameter gradients. The learner supplies probe,
    age and STM-dimension averages; this function only sums output coordinates.
    Arithmetic follows theta, permitting float64 FD with gradients to float32
    optimizer variables through their differentiable dtype conversion.
    """
    _inputs(layout, theta, x, 1)
    if x.dtype != theta.dtype:
        raise ValueError("FD inputs and theta must share dtype")
    if isinstance(step, bool) or not math.isfinite(float(step)) or step <= 0:
        raise ValueError("FD step must be finite and positive")

    def one(direction):
        plus = flat_prediction(model, layout, theta + step / 2 * direction, x)
        minus = flat_prediction(model, layout, theta - step / 2 * direction, x)
        return ((plus - minus) / step).square().sum(-1) / SIGMA2

    return vmap(one)(_directions(layout, theta, probes))


def jvp_values(model, layout, theta, x, probes):
    """Test oracle: differentiable ||J_S v||**2/4 for each probe/grid (K,N)."""
    _inputs(layout, theta, x, 1)

    def one(direction):
        _, tangent = jvp(lambda value: flat_prediction(model, layout, value, x),
                         (theta,), (direction,))
        return tangent.square().sum(-1) / SIGMA2

    return vmap(one)(_directions(layout, theta, probes))
