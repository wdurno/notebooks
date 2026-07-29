"""EWC-regularized parameter proposals for the original learning process."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor, nn

from .config import OptimizerConfig
from .parameters import ParameterLayout


@dataclasses.dataclass(frozen=True)
class EWCProposalResult:
    displacement: Tensor
    data_loss_before: float
    data_loss_after: float
    ewc_penalty_after: float
    objective_after: float
    displacement_norm: float
    fisher_weighted_displacement_norm: float
    inner_steps: int

    def metrics_mapping(self) -> dict[str, Any]:
        return {
            "data_loss_before": self.data_loss_before,
            "data_loss_after": self.data_loss_after,
            "ewc_penalty_after": self.ewc_penalty_after,
            "objective_after": self.objective_after,
            "displacement_norm": self.displacement_norm,
            "fisher_weighted_displacement_norm": (
                self.fisher_weighted_displacement_norm
            ),
            "inner_steps": self.inner_steps,
        }


def ewc_penalty(
    parameter_vector: Tensor,
    anchor: Tensor,
    fisher: Tensor,
    strength: float,
) -> Tensor:
    """Return ``strength / 2 * (theta-anchor)^T I (theta-anchor)``."""

    if parameter_vector.ndim != 1 or anchor.shape != parameter_vector.shape:
        raise ValueError("parameter vector and anchor must be equal-length vectors")
    if fisher.shape != (parameter_vector.numel(), parameter_vector.numel()):
        raise ValueError("Fisher shape does not match the parameter vector")
    if (
        anchor.device != parameter_vector.device
        or fisher.device != parameter_vector.device
        or anchor.dtype != parameter_vector.dtype
        or fisher.dtype != parameter_vector.dtype
    ):
        raise ValueError("parameter vector, anchor, and Fisher must share dtype/device")
    if not isinstance(strength, (int, float)) or not torch.isfinite(
        parameter_vector.new_tensor(strength)
    ):
        raise ValueError("EWC strength must be finite")
    if strength < 0:
        raise ValueError("EWC strength must be nonnegative")

    displacement = parameter_vector - anchor
    return 0.5 * float(strength) * (displacement @ (fisher @ displacement))


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Build the configured proposal optimizer."""

    config.validate()
    if config.name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return torch.optim.SGD(model.parameters(), lr=config.learning_rate)


def take_ewc_proposal(
    model: nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    fisher: Tensor,
    config: OptimizerConfig,
    optimizer: torch.optim.Optimizer,
) -> EWCProposalResult:
    """Optimize one batch around the current anchor and return its realized move."""

    config.validate()
    layout.validate_module(model)
    if inputs.shape[0] != targets.shape[0] or inputs.shape[0] == 0:
        raise ValueError("inputs and targets must have a nonempty shared batch")
    anchor = layout.flatten_module(model, detach=True)
    if fisher.shape != (layout.total_numel, layout.total_numel):
        raise ValueError("Fisher shape does not match the model")
    if fisher.device != anchor.device or fisher.dtype != anchor.dtype:
        raise ValueError("Fisher must share the model dtype and device")

    model.train()
    with torch.no_grad():
        data_loss_before = nn.functional.cross_entropy(model(inputs), targets)

    for _ in range(config.inner_steps):
        optimizer.zero_grad(set_to_none=True)
        data_loss = nn.functional.cross_entropy(model(inputs), targets)
        parameter_vector = layout.flatten_module(model)
        penalty = ewc_penalty(
            parameter_vector,
            anchor,
            fisher,
            config.ewc_strength,
        )
        (data_loss + penalty).backward()
        optimizer.step()

    final_vector = layout.flatten_module(model, detach=True)
    displacement = final_vector - anchor
    with torch.no_grad():
        data_loss_after = nn.functional.cross_entropy(model(inputs), targets)
        penalty_after = ewc_penalty(
            final_vector,
            anchor,
            fisher,
            config.ewc_strength,
        )
        quadratic = displacement @ (fisher @ displacement)
        numerical_floor = -100 * torch.finfo(quadratic.dtype).eps
        if float(quadratic) < numerical_floor:
            raise RuntimeError("PSD Fisher produced a materially negative quadratic")
        fisher_weighted_norm = torch.sqrt(quadratic.clamp_min(0))

    return EWCProposalResult(
        displacement=displacement.detach().cpu(),
        data_loss_before=float(data_loss_before),
        data_loss_after=float(data_loss_after),
        ewc_penalty_after=float(penalty_after),
        objective_after=float(data_loss_after + penalty_after),
        displacement_norm=float(torch.linalg.vector_norm(displacement)),
        fisher_weighted_displacement_norm=float(fisher_weighted_norm),
        inner_steps=config.inner_steps,
    )
