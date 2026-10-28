"""Model-integrated Gaussian Fisher and differentiable STM KL in pixel units.

For physical means f, loss gradients g=J^T(f-y)/sigma2 are negative scores.
Integrating model targets gives J^T J/sigma2, not residual-weighted observed
gradient outer products. Coordinate contributions are summed, never averaged.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.func import jacrev, jvp, vmap

from mnist_experiment.fisher_memory.learner import _stop
from src.parameters import ParameterLayout

from .model import GAUSSIAN_SCALE_PIXELS, RegistrationMemoryCNN, flat_prediction, memory_indices


CURVATURE_SCHEMA_VERSION = 1
SIGMA2 = GAUSSIAN_SCALE_PIXELS**2


def gaussian_nll(prediction: Tensor, y: Tensor, scale: float = GAUSSIAN_SCALE_PIXELS) -> Tensor:
    """Physical-pixel Gaussian NLL (N,), summing all d coordinate NLLs."""
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("Gaussian scale must be finite and positive")
    if (prediction.ndim != 2 or prediction.shape[1] == 0 or prediction.shape != y.shape
            or prediction.device != y.device
            or not prediction.is_floating_point() or not y.is_floating_point()):
        raise ValueError("predictions and pixel targets must be matching (N, d) floating arrays on one device")
    return ((prediction.double() - y.double()).square().sum(-1) / (2 * scale**2)
            + prediction.shape[1] * math.log(scale * math.sqrt(2 * math.pi)))


def _validate_inputs(layout: ParameterLayout, theta: Tensor, x: Tensor) -> None:
    if theta.ndim != 1 or theta.numel() != layout.total_numel:
        raise ValueError("theta must match the flat parameter layout")
    if theta.dtype not in (torch.float32, torch.float64):
        raise ValueError("theta must have float32 or float64 dtype")
    if x.ndim != 4 or x.shape[0] == 0 or tuple(x.shape[1:]) != (2, 28, 28):
        raise ValueError("inputs must be a nonempty (N, 2, 28, 28) batch")
    if x.dtype != theta.dtype or x.device != theta.device:
        raise ValueError("inputs and theta must share dtype and device")


def _directions(layout: ParameterLayout, theta: Tensor, probes: Tensor) -> Tensor:
    ltm, stm = memory_indices(layout)
    values = torch.as_tensor(probes, dtype=theta.dtype, device=theta.device).detach()
    if values.ndim != 2 or not values.shape[0] or not torch.isfinite(values).all():
        raise ValueError("probes must be a nonempty finite (K, dimension) array")
    if values.shape[1] == stm.numel():
        return theta.new_zeros(len(values), theta.numel()).index_copy(1, stm.to(theta.device), values)
    if values.shape[1] != theta.numel():
        raise ValueError("probes must use STM-only or full coordinates")
    if values[:, ltm.to(theta.device)].count_nonzero():
        raise ValueError("full-coordinate probes must be zero in LTM")
    return values


def reference_fisher(model: RegistrationMemoryCNN, layout: ParameterLayout,
                     theta: Tensor, x: Tensor, *, indices: Tensor | None = None,
                     chunk_size: int = 16, check_stop: Callable | None = None) -> Tensor:
    """Detached float64 input mean of J_i^T J_i / sigma2, in bounded chunks.

    Jacobians have axes (input, output, parameter). Flattening the first two
    axes sums output contributions while normalizing only by input count.
    A coordinate subset selects a principal block of the same Fisher.
    """
    _stop(check_stop)
    _validate_inputs(layout, theta, x)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    selected = (torch.arange(theta.numel(), device=theta.device) if indices is None
                else torch.as_tensor(indices, device=theta.device))
    if (selected.ndim != 1 or not selected.numel()
            or selected.dtype not in (torch.int32, torch.int64)
            or selected.min() < 0 or selected.max() >= theta.numel()
            or selected.unique().numel() != selected.numel()):
        raise ValueError("indices must be unique integer global coordinates in range")
    selected = selected.long()
    total = torch.zeros(len(selected), len(selected), device=theta.device, dtype=torch.float64)

    def predict_one(parameter, image):
        return flat_prediction(model, layout, parameter, image.unsqueeze(0))[0]

    with torch.no_grad():
        for chunk in x.split(chunk_size):
            _stop(check_stop)
            rows = vmap(jacrev(predict_one), in_dims=(None, 0))(theta.detach(), chunk)
            rows = rows.index_select(-1, selected).double().reshape(-1, len(selected))
            total.add_(rows.mT @ rows / (len(x) * SIGMA2))
    _stop(check_stop)
    if not torch.isfinite(total).all():
        raise ValueError("model generated a nonfinite Gaussian Fisher")
    return (total + total.mT) / 2


def fd_values(model: RegistrationMemoryCNN, layout: ParameterLayout, theta: Tensor,
              x: Tensor, probes: Tensor, step: float) -> Tensor:
    """Sum of symmetric coordinate KLs / step**2, shape (K, N).

    Both theta +/- step*v/2 branches retain gradients into LTM and STM.
    Probe averaging and STM-dimension normalization belong to the objective.
    """
    _validate_inputs(layout, theta, x)
    if isinstance(step, bool) or not math.isfinite(float(step)) or step <= 0:
        raise ValueError("FD step must be finite and positive")
    directions = _directions(layout, theta, probes)

    def one(direction):
        plus = flat_prediction(model, layout, theta + step / 2 * direction, x)
        minus = flat_prediction(model, layout, theta - step / 2 * direction, x)
        return ((plus.double() - minus.double()) / step).square().sum(-1) / SIGMA2

    return vmap(one)(directions)


def jvp_values(model: RegistrationMemoryCNN, layout: ParameterLayout, theta: Tensor,
               x: Tensor, probes: Tensor) -> Tensor:
    """Differentiable exact ||J_S v||**2 / sigma2, shape (K, N)."""
    _validate_inputs(layout, theta, x)
    directions = _directions(layout, theta, probes)

    def one(direction):
        _, tangent = jvp(lambda parameter: flat_prediction(model, layout, parameter, x),
                         (theta,), (direction,))
        return tangent.double().square().sum(-1) / SIGMA2

    return vmap(one)(directions)
