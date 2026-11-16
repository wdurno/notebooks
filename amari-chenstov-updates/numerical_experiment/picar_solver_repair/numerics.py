"""Plan 19 Phase 3 Fix 2 diagonal solves, using live loss gradients, not scores.

The optimizer metric is numerical; EWC and ridge are statistical curvature.
No clipping, momentum, weight decay, or Fisher floor is added to these solves.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


BETA2 = .999
EPSILON = 1e-8
RECIPE_STEPS = {"proximal_1": 1, "lagged_1": 1, "proximal_3": 3}


def accumulation_dtype(value: Tensor) -> torch.dtype:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def _finite(value: Tensor, name: str, *, nonnegative=False, positive=False) -> None:
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite or nonfloating {name}")
    if (positive and bool((value <= 0).any())) or (nonnegative and bool((value < 0).any())):
        raise ValueError(f"invalid {name}")


def adaptive_metric(gradient: Tensor, second_moment: Tensor, evaluation: int,
                    learning_rate: float, *, validate=True) -> tuple[Tensor, Tensor]:
    """Return M_r and v_r; ``evaluation`` is the one-based gradient count.

    Only grad(live NLL + FD) enters v_r. Bias correction uses the total count
    across accepted events, including all inner iterations of proximal_3.
    """
    if type(evaluation) is not int or evaluation < 1:
        raise ValueError("evaluation must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if gradient.shape != second_moment.shape or gradient.device != second_moment.device:
        raise ValueError("gradient and second moment must match")
    if validate:
        _finite(gradient, "live gradient")
        _finite(second_moment, "second moment", nonnegative=True)
    gradient = gradient.detach().to(accumulation_dtype(gradient))
    moment = BETA2 * second_moment.detach().to(gradient) + (1 - BETA2) * gradient.square()
    correction = -math.expm1(evaluation * math.log(BETA2))
    metric = (torch.sqrt(moment / correction) + EPSILON) / learning_rate
    if validate:
        _finite(moment, "updated second moment", nonnegative=True)
        _finite(metric, "optimizer metric", positive=True)
    return metric, moment


def diagonal_step(x: Tensor, gradient: Tensor, metric: Tensor, center: Tensor,
                  ewc: Tensor, ridge: Tensor, *, recipe: str,
                  previous_center: Tensor | None = None, validate=True) -> Tensor:
    """Exact local diagonal displacement before rounding to actor dtype.

    EWC is zero on STM; ridge acts on both banks. For lagged_1 only EWC is
    explicit and lagged. ``center`` always remains the current event center.
    """
    if recipe not in RECIPE_STEPS:
        raise ValueError(f"unknown solver recipe: {recipe}")
    if any(value.shape != x.shape or value.device != x.device
           for value in (gradient, metric, center, ewc)):
        raise ValueError("diagonal solve coordinates must match")
    if validate:
        for name, value in (("iterate", x), ("gradient", gradient), ("center", center)):
            _finite(value, name)
        _finite(metric, "optimizer metric", positive=True)
        _finite(ewc, "EWC curvature", nonnegative=True)
        _finite(ridge, "ridge curvature", nonnegative=True)
    dtype = accumulation_dtype(x)
    x, gradient, metric, center, ewc, ridge = (
        value.to(dtype) for value in (x, gradient, metric, center, ewc, ridge))
    if recipe == "lagged_1":
        if previous_center is None or previous_center.shape != x.shape or previous_center.device != x.device:
            raise ValueError("lagged solve requires a matching previous center")
        if validate:
            _finite(previous_center, "previous center")
        force = gradient + ewc * (x - previous_center.to(dtype)) + ridge * (x - center)
        result = -force / (metric + ridge)
    else:
        curvature = ewc + ridge
        result = -(gradient + curvature * (x - center)) / (metric + curvature)
    if validate:
        _finite(result, "solver displacement")
    return result
