"""Label-integrated Fisher diagonals and their realized-path corrections.

Training loss gradients are negative scores. Fisher observations sum over all
model labels, rather than taking outer products of observed-label gradients.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.func import functional_call, jacrev, jvp, vmap

from src.parameters import ParameterLayout
from mnist_experiment.rotated_mnist.plan12.gauge import build_gauge_fixed_model


CONDITIONS = ("ema", "paired_fd", "analytic_lfu")


def build_model(seed: int, *, device="cpu", dtype=torch.float32):
    model, layout = build_gauge_fixed_model(seed, device=device, dtype=dtype)
    # 14x14 -> 2x2 is exactly fixed 7x7 pooling; its higher derivatives work on CUDA.
    model.features[-1] = torch.nn.AvgPool2d(7)
    return model, layout


def logits(model, layout: ParameterLayout, theta: Tensor, inputs: Tensor) -> Tensor:
    return functional_call(model, layout.unflatten_named(theta), (inputs,))


def diagonal_function(model, layout, theta: Tensor, inputs: Tensor,
                      *, chunk_size: int = 8, check=lambda: None) -> Tensor:
    """Differentiable exact ten-label mean; no p-by-p Fisher is materialized."""
    if theta.ndim != 1 or theta.numel() != layout.total_numel or not len(inputs):
        raise ValueError("nonempty inputs and a matching parameter vector required")
    total = theta * 0.

    def log_prob(parameter, image):
        return logits(model, layout, parameter, image.unsqueeze(0))[0].log_softmax(-1)

    for chunk in inputs.split(chunk_size):
        check()
        scores = vmap(jacrev(log_prob), in_dims=(None, 0))(theta, chunk)
        probabilities = vmap(log_prob, in_dims=(None, 0))(theta, chunk).exp()
        total = total + (probabilities.unsqueeze(-1) * scores.square()).sum((0, 1)) / len(inputs)
    return total


def observation(model, layout, theta, inputs, *, chunk_size=8, check=lambda: None):
    result = diagonal_function(model, layout, theta.double(), inputs.double(),
                               chunk_size=chunk_size, check=check).detach()
    if not torch.isfinite(result).all() or (result < 0).any():
        raise FloatingPointError("invalid model-Fisher diagonal")
    return result


def observe_update(model, layout, previous, current, old_diagonal, inputs,
                   *, condition, c, chunk_size=8, check=lambda: None):
    """Observe the previous realized displacement on one fresh common input batch."""
    if condition not in CONDITIONS or not 0 < c <= 1:
        raise ValueError("invalid condition or mixture weight")
    current, inputs = current.double(), inputs.double()
    fresh = observation(model, layout, current, inputs, chunk_size=chunk_size, check=check)
    delta = torch.zeros_like(fresh)
    if condition != "ema" and previous is not None:
        previous = previous.double()
        if condition == "paired_fd":
            delta = fresh - observation(model, layout, previous, inputs,
                                        chunk_size=chunk_size, check=check)
        else:
            def measured(parameter):
                return diagonal_function(model, layout, parameter, inputs,
                                         chunk_size=chunk_size, check=check)
            _, delta = jvp(measured, (previous,), (current - previous,))
            delta = delta.detach()
    raw = (1 - c) * (old_diagonal.double() + delta) + c * fresh
    if not torch.isfinite(raw).all():
        raise FloatingPointError("nonfinite raw diagonal update")
    diagnostics = {"negative_fraction": float((raw < 0).double().mean()),
                   "negative_mass": float((-raw.clamp_max(0)).sum()),
                   "raw_min": float(raw.min()), "fisher_trace": float(raw.clamp_min(0).sum()),
                   "correction_norm": float(delta.norm()), "fresh_norm": float(fresh.norm())}
    return raw.clamp_min(0), diagnostics
