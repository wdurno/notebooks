"""Exact Gaussian label integration and both-block STM finite-difference KL.

Loss gradients g=J.T(f-y)/4 are negative scores. Integrating model targets
gives F_i=J_i.T J_i/4, summing all 54 coordinates and averaging only grids.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import torch
from torch import Tensor
from torch.func import jacrev, jvp, vmap

from mnist_experiment.fisher_memory.learner import _stop
from mnist_experiment.fisher_memory_registration.curvature import gaussian_nll
from src.parameters import ParameterLayout

from .model import GAUSSIAN_SCALE_PIXELS, GridRegistrationMemoryCNN, flat_prediction, memory_indices


CURVATURE_SCHEMA_VERSION = 1
SIGMA2 = GAUSSIAN_SCALE_PIXELS**2


def _validate_inputs(layout: ParameterLayout, theta: Tensor, x: Tensor) -> None:
    if theta.ndim != 1 or theta.numel() != layout.total_numel:
        raise ValueError("theta must match the flat parameter layout")
    if theta.dtype not in (torch.float32, torch.float64):
        raise ValueError("theta must have float32 or float64 dtype")
    if x.ndim != 4 or not len(x) or tuple(x.shape[1:]) != (2, 84, 84):
        raise ValueError("inputs must be a nonempty (N, 2, 84, 84) batch")
    if x.dtype != theta.dtype or x.device != theta.device:
        raise ValueError("inputs and theta must share dtype and device")


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _selected_indices(theta: Tensor, indices: Tensor | None) -> Tensor:
    selected = (torch.arange(theta.numel(), device=theta.device) if indices is None
                else torch.as_tensor(indices, device=theta.device))
    if (selected.ndim != 1 or not selected.numel()
            or selected.dtype not in (torch.int32, torch.int64)
            or selected.min() < 0 or selected.max() >= theta.numel()
            or selected.unique().numel() != selected.numel()):
        raise ValueError("indices must be unique integer global coordinates in range")
    return selected.long()


def output_jacobians(model: GridRegistrationMemoryCNN, layout: ParameterLayout,
                     theta: Tensor, x: Tensor, *, indices: Tensor | None = None,
                     chunk_size: int = 2, output_chunk_size: int = 6,
                     check_stop: Callable | None = None) -> Iterator[Tensor]:
    """Yield detached float64 (input chunk, output chunk, parameter) rows.

    Traversal is input-major, then output-major. Optional indices select a
    principal block of the same Fisher before differentiation, not a new model.
    Both axes are bounded; no full per-grid output Jacobian is retained.
    """
    _stop(check_stop)
    _validate_inputs(layout, theta, x)
    _positive_integer(chunk_size, "chunk_size")
    _positive_integer(output_chunk_size, "output_chunk_size")
    selected = _selected_indices(theta, indices)
    fixed = theta.detach()
    for chunk in x.split(chunk_size):
        for start in range(0, model.output_dim, output_chunk_size):
            _stop(check_stop)

            def predict_one(parameter, image):
                full = fixed.index_copy(0, selected, parameter)
                return flat_prediction(model, layout, full, image.unsqueeze(0))[
                    0, start:start + output_chunk_size]

            with torch.no_grad():
                rows = vmap(jacrev(predict_one), in_dims=(None, 0))(fixed[selected], chunk)
            rows = rows.detach().double()
            if not torch.isfinite(rows).all():
                raise ValueError("model generated nonfinite Gaussian Jacobians")
            yield rows
    _stop(check_stop)


def reference_fisher(model: GridRegistrationMemoryCNN, layout: ParameterLayout,
                     theta: Tensor, x: Tensor, *, indices: Tensor | None = None,
                     chunk_size: int = 2, output_chunk_size: int = 6,
                     check_stop: Callable | None = None) -> Tensor:
    """Detached symmetric float64 mean of J_i.T J_i/4 in bounded chunks."""
    _stop(check_stop)
    _validate_inputs(layout, theta, x)
    selected = _selected_indices(theta, indices)
    total = torch.zeros(len(selected), len(selected), dtype=torch.float64, device=theta.device)
    for jacobian in output_jacobians(model, layout, theta, x, indices=selected,
                                    chunk_size=chunk_size, output_chunk_size=output_chunk_size,
                                    check_stop=check_stop):
        rows = jacobian.reshape(-1, len(selected))
        total.addmm_(rows.mT, rows, alpha=1 / (len(x) * SIGMA2))
    _stop(check_stop)
    if not torch.isfinite(total).all():
        raise ValueError("model generated a nonfinite Gaussian Fisher")
    return (total + total.mT) / 2


def _directions(layout: ParameterLayout, theta: Tensor, probes: Tensor) -> Tensor:
    ltm, stm = memory_indices(layout)
    values = torch.as_tensor(probes, dtype=theta.dtype, device=theta.device).detach()
    if values.ndim != 2 or not len(values) or not torch.isfinite(values).all():
        raise ValueError("probes must be a nonempty finite (K, dimension) array")
    if values.shape[1] == stm.numel():
        return theta.new_zeros(len(values), theta.numel()).index_copy(1, stm.to(theta.device), values)
    if values.shape[1] != theta.numel():
        raise ValueError("probes must use STM-only or full coordinates")
    if values[:, ltm.to(theta.device)].count_nonzero():
        raise ValueError("full-coordinate probes must be zero in LTM")
    return values


def fd_values(model: GridRegistrationMemoryCNN, layout: ParameterLayout, theta: Tensor,
              x: Tensor, probes: Tensor, step: float) -> Tensor:
    """Symmetric Gaussian KL/step**2 (K,N), summed over all 54 outputs.

    Both stencil branches retain LTM/STM gradients. The objective, not this
    quantity, applies probe averaging, squared ages and STM-dimension scaling.
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


def jvp_values(model: GridRegistrationMemoryCNN, layout: ParameterLayout, theta: Tensor,
               x: Tensor, probes: Tensor) -> Tensor:
    """Differentiable exact ||J_S v||**2/4, shape (K,N), without output averaging."""
    _validate_inputs(layout, theta, x)
    directions = _directions(layout, theta, probes)

    def one(direction):
        _, tangent = jvp(lambda parameter: flat_prediction(model, layout, parameter, x),
                         (theta,), (direction,))
        return tangent.double().square().sum(-1) / SIGMA2

    return vmap(one)(directions)
