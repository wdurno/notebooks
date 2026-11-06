"""448-parameter base CNN and the N(f_theta(x), 4 I_6) likelihood.

Observed-label scores are observer measurements, not Fisher observations.
The likelihood Fisher integrates model targets exactly as diag(J^T J)/4.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, jacrev, vmap

from src.parameters import ParameterLayout


MODEL_SCHEMA_VERSION = 1
GAUSSIAN_SCALE_PIXELS = 2.
PARAMETER_COUNT = 448


class RegistrationCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 6
        self.features = nn.Sequential(
            nn.Conv2d(2, 4, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
            nn.Conv2d(4, 6, 3, padding=1), nn.SiLU(), nn.AvgPool2d(7),
        )
        self.classifier = nn.Linear(24, 6)
        if sum(p.numel() for p in self.parameters()) != PARAMETER_COUNT:
            raise RuntimeError("registration base must have exactly 448 trainable parameters")

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (2, 28, 28):
            raise ValueError("registration inputs must have shape (N,2,28,28)")
        return self.classifier(self.features(inputs).flatten(1))


def build_model(seed: int, device: str | torch.device = "cpu",
                dtype: torch.dtype = torch.float32) -> tuple[RegistrationCNN, ParameterLayout]:
    """Seed the base on CPU without advancing global CPU or CUDA generators."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("registration supports float32 and float64")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = RegistrationCNN()
    model = model.to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


def predict(model: nn.Module, layout: ParameterLayout, theta: Tensor, x: Tensor) -> Tensor:
    """Functional unbounded pixel means, without changing the module or layout."""
    if theta.dtype != x.dtype or theta.device != x.device:
        raise ValueError("theta and inputs must share dtype and device")
    return functional_call(model, (layout.unflatten_named(theta), dict(model.named_buffers())),
                           (x,), strict=True)


flat_prediction = predict


def nll(outputs: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Sum six coordinate NLLs, then reduce observations (g = grad L = -s)."""
    if outputs.ndim != 2 or outputs.shape[-1] != 6 or targets.shape != outputs.shape:
        raise ValueError("outputs and targets must have matching shape (N,6)")
    if outputs.device != targets.device or outputs.dtype != targets.dtype:
        raise ValueError("outputs and targets must share dtype and device")
    values = (outputs - targets).square().sum(-1) / 8 + 3 * math.log(8 * math.pi)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        if not len(values):
            raise ValueError("a mean NLL requires observations")
        return values.mean()
    raise ValueError("reduction must be none, sum, or mean")


def _measurement_inputs(layout: ParameterLayout, theta: Tensor, x: Tensor, chunk_size: int) -> None:
    if theta.ndim != 1 or theta.numel() != layout.total_numel or not theta.is_floating_point():
        raise ValueError("theta must be a floating vector matching the parameter layout")
    if x.ndim != 4 or tuple(x.shape[1:]) != (2, 28, 28) or not len(x):
        raise ValueError("nonempty (N,2,28,28) inputs required")
    if not x.is_floating_point() or x.device != theta.device:
        raise ValueError("floating inputs must share theta's device")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")


def observed_scores(model: nn.Module, layout: ParameterLayout, theta: Tensor,
                    x: Tensor, y: Tensor, *, chunk_size: int = 8,
                    check: Callable[[], None] = lambda: None) -> Tensor:
    """Per-observation positive scores J^T(y-f)/4, returned as float64 (N,d)."""
    _measurement_inputs(layout, theta, x, chunk_size)
    if y.shape != (len(x), 6) or not y.is_floating_point() or y.device != theta.device:
        raise ValueError("floating targets of shape (N,6) must share theta's device")
    parameter = theta.detach().double()

    def log_probability(value: Tensor, image: Tensor, target: Tensor) -> Tensor:
        return -nll(predict(model, layout, value, image.unsqueeze(0)), target.unsqueeze(0))

    pieces = []
    for start in range(0, len(x), chunk_size):
        check()
        values = vmap(grad(log_probability), in_dims=(None, 0, 0))(
            parameter, x[start:start + chunk_size].double(), y[start:start + chunk_size].double())
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("nonfinite observed-label scores")
        pieces.append(values.detach())
    return torch.cat(pieces)


def observation(model: nn.Module, layout: ParameterLayout, theta: Tensor, x: Tensor, *,
                chunk_size: int = 8, check: Callable[[], None] = lambda: None) -> Tensor:
    """Mean model-target-integrated Fisher diagonal, accumulated in float64.

    jacrev's output chunk of one bounds the reverse-mode output work; vmap
    bounds simultaneous images. No d-by-d matrix or Hessian is constructed.
    """
    _measurement_inputs(layout, theta, x, chunk_size)
    parameter = theta.detach().double()
    total = torch.zeros_like(parameter)

    def mean(value: Tensor, image: Tensor) -> Tensor:
        return predict(model, layout, value, image.unsqueeze(0))[0]

    for chunk in x.split(chunk_size):
        check()
        jacobian = vmap(jacrev(mean, chunk_size=1), in_dims=(None, 0))(parameter, chunk.double())
        total += jacobian.detach().square().sum((0, 1)) / 4
    result = total / len(x)
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise FloatingPointError("invalid model-Fisher diagonal")
    return result


fisher_diagonal = observation
