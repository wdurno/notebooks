"""448-parameter padded registration and N(f_theta(x), 4 I_6) likelihood.

Scores from observed targets feed the observer. Fisher observations integrate
the model target analytically: diag(J^T J)/4, with no residual-square substitute.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.func import grad, jacrev, vmap

from src.parameters import ParameterLayout

from ..registration.model import GAUSSIAN_SCALE_PIXELS, PARAMETER_COUNT, nll, predict


MODEL_SCHEMA_VERSION = 1
CANVAS_SIZE = 40
flat_prediction = predict


class RegistrationCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 6
        self.features = nn.Sequential(
            nn.Conv2d(2, 4, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
            # Fixed 20x20 features make this equivalent to adaptive 2x2 pooling;
            # ordinary average pooling also supports deterministic CUDA backward.
            nn.Conv2d(4, 6, 3, padding=1), nn.SiLU(), nn.AvgPool2d(10),
        )
        self.classifier = nn.Linear(24, 6)
        if sum(p.numel() for p in self.parameters()) != PARAMETER_COUNT:
            raise RuntimeError("padded registration must have exactly 448 trainable parameters")

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (2, CANVAS_SIZE, CANVAS_SIZE):
            raise ValueError("registration inputs must have shape (N,2,40,40)")
        return self.classifier(self.features(inputs).flatten(1))


def build_model(seed: int, device: str | torch.device = "cpu",
                dtype: torch.dtype = torch.float32) -> tuple[RegistrationCNN, ParameterLayout]:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("registration supports float32 and float64")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = RegistrationCNN()
    model = model.to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


def _measurement_inputs(layout: ParameterLayout, theta: Tensor, x: Tensor, chunk_size: int) -> None:
    if theta.ndim != 1 or theta.numel() != layout.total_numel or not theta.is_floating_point():
        raise ValueError("theta must be a floating vector matching the parameter layout")
    if x.ndim != 4 or tuple(x.shape[1:]) != (2, CANVAS_SIZE, CANVAS_SIZE) or not len(x):
        raise ValueError("nonempty (N,2,40,40) inputs required")
    if not x.is_floating_point() or x.device != theta.device:
        raise ValueError("floating inputs must share theta's device")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")


def observed_scores(model: nn.Module, layout: ParameterLayout, theta: Tensor,
                    x: Tensor, y: Tensor, *, chunk_size: int = 8,
                    check: Callable[[], None] = lambda: None) -> Tensor:
    """Per-observation positive scores J^T(y-f)/4 in float64 (g=-s)."""
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
    """Mean model-target-integrated diagonal, accumulating output Jacobians only."""
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
per_sample_scores = observed_scores
