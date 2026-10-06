"""Fixed-budget preconditioned optimization for Plan 13."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor, nn

from src.parameters import ParameterLayout


class MatrixOperator(Protocol):
    def matvec(self, vector: Tensor) -> Tensor: ...


Preconditioner = Callable[[Tensor], Tensor]


@dataclasses.dataclass(frozen=True)
class FixedBudgetResult:
    displacement: Tensor
    initial_gradient: Tensor
    objective_before: float
    objective_after: float
    data_loss_before: float
    data_loss_after: float
    penalty_after: float
    initial_gradient_norm: float
    final_gradient_norm: float
    accepted_steps: int
    backtracking_rejections: int
    function_evaluations: int
    stopping_reason: str

    def mapping(self) -> dict[str, object]:
        value = dataclasses.asdict(self)
        value["displacement"] = None
        value["initial_gradient"] = None
        value["displacement_norm"] = float(torch.linalg.vector_norm(self.displacement))
        return value


def _flat_gradient(model: nn.Module) -> Tensor:
    pieces = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                pieces.append(torch.zeros_like(parameter).reshape(-1))
            else:
                pieces.append(parameter.grad.reshape(-1))
    if not pieces:
        raise RuntimeError("objective produced no trainable gradients")
    return torch.cat(pieces)


def _objective(
    model: nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    anchor: Tensor,
    fisher: MatrixOperator,
    *,
    strength: float,
    kappa: float,
    backward: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    if backward:
        model.zero_grad(set_to_none=True)
    data_loss = nn.functional.cross_entropy(model(inputs), targets)
    displacement = layout.flatten_module(model) - anchor
    fisher_product = fisher.matvec(displacement)
    quadratic = displacement @ fisher_product + float(kappa) * displacement.square().sum()
    penalty = 0.5 * float(strength) * quadratic
    objective = data_loss + penalty
    if not torch.isfinite(objective):
        raise RuntimeError("Plan 13 objective became nonfinite")
    if backward:
        objective.backward()
    return objective, data_loss, penalty


def fixed_budget_update(
    model: nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    fisher: MatrixOperator,
    precondition: Preconditioner,
    *,
    strength: float,
    kappa: float,
    inner_steps: int,
    learning_rate: float,
    max_backtracks: int,
    armijo: float = 1e-4,
) -> FixedBudgetResult:
    if inner_steps < 1 or max_backtracks < 1 or learning_rate <= 0:
        raise ValueError("optimizer budget and learning rate must be positive")
    anchor = layout.flatten_module(model, detach=True)
    objective, data_loss, _ = _objective(
        model,
        layout,
        inputs,
        targets,
        anchor,
        fisher,
        strength=strength,
        kappa=kappa,
        backward=True,
    )
    objective_before = float(objective.detach())
    data_loss_before = float(data_loss.detach())
    gradient = _flat_gradient(model).detach()
    initial_gradient = gradient.clone()
    initial_gradient_norm = float(torch.linalg.vector_norm(gradient))
    evaluations = 1
    accepted_steps = 0
    rejections = 0
    stopping_reason = "step_budget"
    tiny = torch.finfo(anchor.dtype).eps * max(layout.total_numel, 1)

    for _ in range(inner_steps):
        direction = -precondition(gradient)
        if direction.shape != gradient.shape or not torch.isfinite(direction).all():
            raise RuntimeError("preconditioner produced an invalid direction")
        directional_derivative = float(gradient @ direction)
        if directional_derivative >= -tiny:
            stopping_reason = "stationary_preconditioned_direction"
            break
        current = layout.flatten_module(model, detach=True)
        current_objective = float(objective.detach())
        step_size = float(learning_rate)
        accepted = False
        for _ in range(max_backtracks):
            candidate = current + step_size * direction
            layout.copy_vector_to_module(model, candidate)
            candidate_objective, _, _ = _objective(
                model,
                layout,
                inputs,
                targets,
                anchor,
                fisher,
                strength=strength,
                kappa=kappa,
                backward=False,
            )
            evaluations += 1
            if float(candidate_objective.detach()) <= current_objective + armijo * step_size * directional_derivative:
                accepted = True
                objective = candidate_objective
                accepted_steps += 1
                break
            rejections += 1
            step_size *= 0.5
        if not accepted:
            layout.copy_vector_to_module(model, current)
            stopping_reason = "line_search_exhausted"
            break
        objective, _, _ = _objective(
            model,
            layout,
            inputs,
            targets,
            anchor,
            fisher,
            strength=strength,
            kappa=kappa,
            backward=True,
        )
        evaluations += 1
        gradient = _flat_gradient(model).detach()

    final_objective, final_data, final_penalty = _objective(
        model,
        layout,
        inputs,
        targets,
        anchor,
        fisher,
        strength=strength,
        kappa=kappa,
        backward=True,
    )
    evaluations += 1
    final_gradient = _flat_gradient(model).detach()
    final_vector = layout.flatten_module(model, detach=True)
    if float(final_objective.detach()) > objective_before + 64 * torch.finfo(anchor.dtype).eps * max(abs(objective_before), 1.0):
        layout.copy_vector_to_module(model, anchor)
        raise RuntimeError("fixed-budget optimizer increased the objective")
    values = (
        objective_before,
        float(final_objective.detach()),
        data_loss_before,
        float(final_data.detach()),
        float(final_penalty.detach()),
        initial_gradient_norm,
        float(torch.linalg.vector_norm(final_gradient)),
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("fixed-budget optimizer produced nonfinite diagnostics")
    model.zero_grad(set_to_none=True)
    return FixedBudgetResult(
        displacement=(final_vector - anchor).detach(),
        initial_gradient=initial_gradient,
        objective_before=values[0],
        objective_after=values[1],
        data_loss_before=values[2],
        data_loss_after=values[3],
        penalty_after=values[4],
        initial_gradient_norm=values[5],
        final_gradient_norm=values[6],
        accepted_steps=accepted_steps,
        backtracking_rejections=rejections,
        function_evaluations=evaluations,
        stopping_reason=stopping_reason,
    )
