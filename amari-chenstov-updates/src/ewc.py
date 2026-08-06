"""EWC-regularized parameter proposals for the original learning process."""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any

import torch
from torch import Tensor, nn

from .config import OptimizerConfig
from .parameters import ParameterLayout
from .representations import FisherRepresentation

FisherLike = Tensor | FisherRepresentation

EWC_BACKTRACKING_FACTOR = 0.5
EWC_MAX_BACKTRACKS = 24


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
    optimizer_iterations: int
    optimizer_function_evaluations: int
    adaptation_weight: float | None
    effective_ewc_strength: float
    objective_normalization: str
    backtracking_rejections: int
    maximum_backtracks: int
    minimum_learning_rate: float
    optimization_guard: str
    initial_gradient_norm: float
    final_gradient_norm: float
    final_gradient_max_abs: float
    relative_final_gradient_norm: float
    final_gradient_rms: float
    objective_decrease: float
    stopping_reason: str

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
            "optimizer_iterations": self.optimizer_iterations,
            "optimizer_function_evaluations": self.optimizer_function_evaluations,
            "adaptation_weight": self.adaptation_weight,
            "effective_ewc_strength": self.effective_ewc_strength,
            "objective_normalization": self.objective_normalization,
            "backtracking_rejections": self.backtracking_rejections,
            "maximum_backtracks": self.maximum_backtracks,
            "minimum_learning_rate": self.minimum_learning_rate,
            "optimization_guard": self.optimization_guard,
            "initial_gradient_norm": self.initial_gradient_norm,
            "final_gradient_norm": self.final_gradient_norm,
            "final_gradient_max_abs": self.final_gradient_max_abs,
            "relative_final_gradient_norm": self.relative_final_gradient_norm,
            "final_gradient_rms": self.final_gradient_rms,
            "objective_decrease": self.objective_decrease,
            "stopping_reason": self.stopping_reason,
        }


def _flatten_gradients(model: nn.Module) -> Tensor:
    gradients = [
        parameter.grad.detach().reshape(-1)
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("EWC objective produced no parameter gradients")
    return torch.cat(gradients)


def mixture_ewc_strength(
    adaptation_weight: float,
    *,
    multiplier: float = 1.0,
) -> float:
    """Return the old-to-new evidence odds ``multiplier * (1-pi) / pi``."""

    if not isinstance(adaptation_weight, (int, float)) or not torch.isfinite(
        torch.tensor(adaptation_weight, dtype=torch.float64)
    ):
        raise ValueError("adaptation weight must be finite")
    if not 0.0 < float(adaptation_weight) <= 1.0:
        raise ValueError("adaptation weight must be in (0, 1]")
    if not isinstance(multiplier, (int, float)) or not torch.isfinite(
        torch.tensor(multiplier, dtype=torch.float64)
    ):
        raise ValueError("EWC multiplier must be finite")
    if float(multiplier) < 0.0:
        raise ValueError("EWC multiplier must be nonnegative")
    return float(multiplier) * (
        (1.0 - float(adaptation_weight)) / float(adaptation_weight)
    )


def ewc_penalty(
    parameter_vector: Tensor,
    anchor: Tensor,
    fisher: FisherLike,
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
    quadratic = (
        displacement @ (fisher @ displacement)
        if isinstance(fisher, Tensor)
        else fisher.quadratic(displacement)
    )
    return 0.5 * float(strength) * quadratic


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Build the configured proposal optimizer."""

    config.validate()
    if config.name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.name == "lbfgs":
        if (
            config.lbfgs_history_size is None
            or config.lbfgs_max_eval_factor is None
            or config.lbfgs_tolerance_grad is None
            or config.lbfgs_tolerance_change is None
            or config.lbfgs_line_search_fn is None
        ):
            raise RuntimeError("validated L-BFGS configuration is incomplete")
        return torch.optim.LBFGS(
            model.parameters(),
            lr=config.learning_rate,
            max_iter=config.inner_steps,
            max_eval=math.ceil(
                config.inner_steps * config.lbfgs_max_eval_factor
            ),
            tolerance_grad=config.lbfgs_tolerance_grad,
            tolerance_change=config.lbfgs_tolerance_change,
            history_size=config.lbfgs_history_size,
            line_search_fn=config.lbfgs_line_search_fn,
        )
    return torch.optim.SGD(model.parameters(), lr=config.learning_rate)


def _take_lbfgs_proposal(
    model: nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    fisher: FisherLike,
    config: OptimizerConfig,
    optimizer: torch.optim.Optimizer,
    *,
    adaptation_weight: float | None,
) -> EWCProposalResult:
    if not isinstance(optimizer, torch.optim.LBFGS):
        raise ValueError("optimizer must be L-BFGS when config.name is 'lbfgs'")
    layout.validate_module(model)
    if inputs.shape[0] != targets.shape[0] or inputs.shape[0] == 0:
        raise ValueError("inputs and targets must have a nonempty shared batch")
    anchor = layout.flatten_module(model, detach=True)
    if fisher.shape != (layout.total_numel, layout.total_numel):
        raise ValueError("Fisher shape does not match the model")
    if fisher.device != anchor.device or fisher.dtype != anchor.dtype:
        raise ValueError("Fisher must share the model dtype and device")
    effective_strength = (
        float(config.ewc_strength)
        if adaptation_weight is None
        else mixture_ewc_strength(
            adaptation_weight,
            multiplier=config.ewc_strength,
        )
    )
    objective_normalization = (
        "mean_new_loss_plus_direct_quadratic"
        if adaptation_weight is None
        else "mean_new_loss_plus_old_to_new_odds"
    )

    model.train()
    with torch.no_grad():
        data_loss_before = nn.functional.cross_entropy(model(inputs), targets)
    original_optimizer_state = copy.deepcopy(optimizer.state_dict())
    original_learning_rates = [
        float(group["lr"]) for group in optimizer.param_groups
    ]
    optimizer.state.clear()
    initial_gradient_norm: float | None = None

    def closure() -> Tensor:
        nonlocal initial_gradient_norm
        optimizer.zero_grad(set_to_none=True)
        data_loss = nn.functional.cross_entropy(model(inputs), targets)
        parameter_vector = layout.flatten_module(model)
        objective = data_loss + ewc_penalty(
            parameter_vector,
            anchor,
            fisher,
            effective_strength,
        )
        if not torch.isfinite(objective):
            raise RuntimeError("L-BFGS EWC objective became non-finite")
        objective.backward()
        gradient = _flatten_gradients(model)
        if not torch.isfinite(gradient).all():
            raise RuntimeError("L-BFGS EWC objective produced non-finite gradients")
        if initial_gradient_norm is None:
            initial_gradient_norm = float(torch.linalg.vector_norm(gradient))
        return objective

    try:
        objective_before = float(closure().detach())
        optimizer.step(closure)

        final_parameter_vector = layout.flatten_module(model)
        if not torch.isfinite(final_parameter_vector).all():
            raise RuntimeError("L-BFGS EWC proposal produced non-finite parameters")
        optimizer.zero_grad(set_to_none=True)
        data_loss_after_tensor = nn.functional.cross_entropy(
            model(inputs), targets
        )
        penalty_after_tensor = ewc_penalty(
            final_parameter_vector,
            anchor,
            fisher,
            effective_strength,
        )
        objective_after_tensor = data_loss_after_tensor + penalty_after_tensor
        if not torch.isfinite(objective_after_tensor):
            raise RuntimeError("final L-BFGS EWC objective is non-finite")
        tolerance = (
            64.0
            * torch.finfo(final_parameter_vector.dtype).eps
            * max(abs(objective_before), 1.0)
        )
        if float(objective_after_tensor.detach()) > objective_before + tolerance:
            raise RuntimeError("strong-Wolfe L-BFGS increased the EWC objective")
        objective_after_tensor.backward()
        final_gradient = _flatten_gradients(model)
        if not torch.isfinite(final_gradient).all():
            raise RuntimeError("final L-BFGS EWC gradient is non-finite")
    except BaseException:
        layout.copy_vector_to_module(model, anchor)
        optimizer.load_state_dict(original_optimizer_state)
        raise
    finally:
        for group, learning_rate in zip(
            optimizer.param_groups, original_learning_rates, strict=True
        ):
            group["lr"] = learning_rate

    if initial_gradient_norm is None:
        raise RuntimeError("L-BFGS proposal did not calculate an initial gradient")
    parameter = optimizer.param_groups[0]["params"][0]
    state = optimizer.state.get(parameter, {})
    iterations = int(state.get("n_iter", 0))
    evaluations = int(state.get("func_evals", 0))
    maximum_evaluations = math.ceil(
        config.inner_steps * float(config.lbfgs_max_eval_factor)
    )
    if iterations >= config.inner_steps:
        stopping_reason = "lbfgs_max_iterations"
    elif evaluations >= maximum_evaluations:
        stopping_reason = "lbfgs_max_evaluations"
    else:
        stopping_reason = "lbfgs_convergence_tolerance"

    final_gradient_norm = float(torch.linalg.vector_norm(final_gradient))
    final_gradient_max_abs = float(torch.max(torch.abs(final_gradient)))
    final_gradient_rms = final_gradient_norm / layout.total_numel**0.5
    relative_final_gradient_norm = final_gradient_norm / max(
        initial_gradient_norm,
        torch.finfo(final_gradient.dtype).tiny,
    )
    final_vector = final_parameter_vector.detach().clone()
    displacement = final_vector - anchor
    with torch.no_grad():
        quadratic = (
            displacement @ (fisher @ displacement)
            if isinstance(fisher, Tensor)
            else fisher.quadratic(displacement)
        )
        numerical_floor = -100 * torch.finfo(quadratic.dtype).eps
        if float(quadratic) < numerical_floor:
            raise RuntimeError("PSD Fisher produced a materially negative quadratic")
        fisher_weighted_norm = torch.sqrt(quadratic.clamp_min(0))

    result = EWCProposalResult(
        displacement=displacement.detach().cpu(),
        data_loss_before=float(data_loss_before),
        data_loss_after=float(data_loss_after_tensor.detach()),
        ewc_penalty_after=float(penalty_after_tensor.detach()),
        objective_after=float(objective_after_tensor.detach()),
        displacement_norm=float(torch.linalg.vector_norm(displacement)),
        fisher_weighted_displacement_norm=float(fisher_weighted_norm),
        inner_steps=config.inner_steps,
        optimizer_iterations=iterations,
        optimizer_function_evaluations=evaluations,
        adaptation_weight=(
            None if adaptation_weight is None else float(adaptation_weight)
        ),
        effective_ewc_strength=effective_strength,
        objective_normalization=objective_normalization,
        backtracking_rejections=0,
        maximum_backtracks=0,
        minimum_learning_rate=config.learning_rate,
        optimization_guard="strong_wolfe_line_search_transaction",
        initial_gradient_norm=initial_gradient_norm,
        final_gradient_norm=final_gradient_norm,
        final_gradient_max_abs=final_gradient_max_abs,
        relative_final_gradient_norm=relative_final_gradient_norm,
        final_gradient_rms=final_gradient_rms,
        objective_decrease=float(data_loss_before - objective_after_tensor.detach()),
        stopping_reason=stopping_reason,
    )
    optimizer.zero_grad(set_to_none=True)
    return result


def take_ewc_proposal(
    model: nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    fisher: FisherLike,
    config: OptimizerConfig,
    optimizer: torch.optim.Optimizer,
    *,
    adaptation_weight: float | None = None,
) -> EWCProposalResult:
    """Optimize one batch around the current anchor and return its realized move.

    When ``adaptation_weight`` is supplied, the mean new-data loss is used with
    the equivalent old-to-new odds coefficient
    ``config.ewc_strength * (1 - pi) / pi``. The optimizer's displacement is
    accepted directly; this function never applies a post-optimization
    multiplication by ``pi``.
    """

    config.validate()
    if config.name == "lbfgs":
        return _take_lbfgs_proposal(
            model,
            layout,
            inputs,
            targets,
            fisher,
            config,
            optimizer,
            adaptation_weight=adaptation_weight,
        )
    layout.validate_module(model)
    if inputs.shape[0] != targets.shape[0] or inputs.shape[0] == 0:
        raise ValueError("inputs and targets must have a nonempty shared batch")
    anchor = layout.flatten_module(model, detach=True)
    if fisher.shape != (layout.total_numel, layout.total_numel):
        raise ValueError("Fisher shape does not match the model")
    if fisher.device != anchor.device or fisher.dtype != anchor.dtype:
        raise ValueError("Fisher must share the model dtype and device")
    effective_strength = (
        float(config.ewc_strength)
        if adaptation_weight is None
        else mixture_ewc_strength(
            adaptation_weight,
            multiplier=config.ewc_strength,
        )
    )
    objective_normalization = (
        "mean_new_loss_plus_direct_quadratic"
        if adaptation_weight is None
        else "mean_new_loss_plus_old_to_new_odds"
    )

    model.train()
    with torch.no_grad():
        data_loss_before = nn.functional.cross_entropy(model(inputs), targets)

    original_optimizer_state = copy.deepcopy(optimizer.state_dict())
    original_learning_rates = [
        float(group["lr"]) for group in optimizer.param_groups
    ]
    if not original_learning_rates or any(
        not torch.isfinite(anchor.new_tensor(value)) or value <= 0.0
        for value in original_learning_rates
    ):
        raise ValueError("optimizer learning rates must be finite and positive")
    backtracking_rejections = 0
    maximum_backtracks = 0
    minimum_learning_rate = min(original_learning_rates)
    initial_gradient_norm: float | None = None
    optimizer_function_evaluations = 0

    try:
        for _ in range(config.inner_steps):
            for group, learning_rate in zip(
                optimizer.param_groups, original_learning_rates, strict=True
            ):
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            data_loss = nn.functional.cross_entropy(model(inputs), targets)
            parameter_vector = layout.flatten_module(model)
            penalty = ewc_penalty(
                parameter_vector,
                anchor,
                fisher,
                effective_strength,
            )
            objective = data_loss + penalty
            optimizer_function_evaluations += 1
            if not torch.isfinite(objective):
                raise RuntimeError("EWC objective became non-finite before a step")
            objective.backward()
            gradient_vector = _flatten_gradients(model)
            if not torch.isfinite(gradient_vector).all():
                raise RuntimeError("EWC objective produced non-finite gradients")
            if initial_gradient_norm is None:
                initial_gradient_norm = float(
                    torch.linalg.vector_norm(gradient_vector)
                )

            step_anchor = layout.flatten_module(model, detach=True)
            step_optimizer_state = copy.deepcopy(optimizer.state_dict())
            objective_before = float(objective.detach())
            tolerance = (
                64.0
                * torch.finfo(step_anchor.dtype).eps
                * max(abs(objective_before), 1.0)
            )
            accepted = False
            for backtracks in range(EWC_MAX_BACKTRACKS + 1):
                if backtracks:
                    layout.copy_vector_to_module(model, step_anchor)
                    optimizer.load_state_dict(step_optimizer_state)
                scale = EWC_BACKTRACKING_FACTOR**backtracks
                trial_learning_rates = [
                    value * scale for value in original_learning_rates
                ]
                for group, learning_rate in zip(
                    optimizer.param_groups,
                    trial_learning_rates,
                    strict=True,
                ):
                    group["lr"] = learning_rate
                optimizer.step()

                candidate_vector = layout.flatten_module(model, detach=True)
                if torch.isfinite(candidate_vector).all():
                    with torch.no_grad():
                        candidate_data_loss = nn.functional.cross_entropy(
                            model(inputs), targets
                        )
                        candidate_penalty = ewc_penalty(
                            candidate_vector,
                            anchor,
                            fisher,
                            effective_strength,
                        )
                        candidate_objective = (
                            candidate_data_loss + candidate_penalty
                        )
                    optimizer_function_evaluations += 1
                    accepted = bool(
                        torch.isfinite(candidate_objective)
                        and float(candidate_objective)
                        <= objective_before + tolerance
                    )
                if accepted:
                    backtracking_rejections += backtracks
                    maximum_backtracks = max(maximum_backtracks, backtracks)
                    minimum_learning_rate = min(
                        minimum_learning_rate,
                        min(trial_learning_rates),
                    )
                    break

            if not accepted:
                raise RuntimeError(
                    "EWC objective backtracking failed to find a finite "
                    f"descent step after {EWC_MAX_BACKTRACKS} reductions"
                )
    except BaseException:
        layout.copy_vector_to_module(model, anchor)
        optimizer.load_state_dict(original_optimizer_state)
        raise
    finally:
        for group, learning_rate in zip(
            optimizer.param_groups, original_learning_rates, strict=True
        ):
            group["lr"] = learning_rate

    try:
        optimizer.zero_grad(set_to_none=True)
        final_parameter_vector = layout.flatten_module(model)
        data_loss_after_tensor = nn.functional.cross_entropy(
            model(inputs), targets
        )
        penalty_after_tensor = ewc_penalty(
            final_parameter_vector,
            anchor,
            fisher,
            effective_strength,
        )
        objective_after_tensor = data_loss_after_tensor + penalty_after_tensor
        objective_after_tensor.backward()
        final_gradient = _flatten_gradients(model)
        if not torch.isfinite(final_gradient).all():
            raise RuntimeError("final EWC objective gradient is non-finite")
        final_gradient_norm = float(torch.linalg.vector_norm(final_gradient))
        final_gradient_max_abs = float(torch.max(torch.abs(final_gradient)))
        final_gradient_rms = final_gradient_norm / layout.total_numel**0.5
        if initial_gradient_norm is None:
            raise RuntimeError("EWC proposal did not calculate an initial gradient")
        relative_final_gradient_norm = final_gradient_norm / max(
            initial_gradient_norm,
            torch.finfo(final_gradient.dtype).tiny,
        )
    except BaseException:
        layout.copy_vector_to_module(model, anchor)
        optimizer.load_state_dict(original_optimizer_state)
        raise

    final_vector = final_parameter_vector.detach().clone()
    displacement = final_vector - anchor
    with torch.no_grad():
        data_loss_after = data_loss_after_tensor.detach()
        penalty_after = penalty_after_tensor.detach()
        objective_after = objective_after_tensor.detach()
        quadratic = (
            displacement @ (fisher @ displacement)
            if isinstance(fisher, Tensor)
            else fisher.quadratic(displacement)
        )
        numerical_floor = -100 * torch.finfo(quadratic.dtype).eps
        if float(quadratic) < numerical_floor:
            raise RuntimeError("PSD Fisher produced a materially negative quadratic")
        fisher_weighted_norm = torch.sqrt(quadratic.clamp_min(0))

    result = EWCProposalResult(
        displacement=displacement.detach().cpu(),
        data_loss_before=float(data_loss_before),
        data_loss_after=float(data_loss_after),
        ewc_penalty_after=float(penalty_after),
        objective_after=float(objective_after),
        displacement_norm=float(torch.linalg.vector_norm(displacement)),
        fisher_weighted_displacement_norm=float(fisher_weighted_norm),
        inner_steps=config.inner_steps,
        optimizer_iterations=config.inner_steps,
        optimizer_function_evaluations=optimizer_function_evaluations,
        adaptation_weight=(
            None if adaptation_weight is None else float(adaptation_weight)
        ),
        effective_ewc_strength=effective_strength,
        objective_normalization=objective_normalization,
        backtracking_rejections=backtracking_rejections,
        maximum_backtracks=maximum_backtracks,
        minimum_learning_rate=minimum_learning_rate,
        optimization_guard="monotone_objective_backtracking",
        initial_gradient_norm=initial_gradient_norm,
        final_gradient_norm=final_gradient_norm,
        final_gradient_max_abs=final_gradient_max_abs,
        relative_final_gradient_norm=relative_final_gradient_norm,
        final_gradient_rms=final_gradient_rms,
        objective_decrease=float(data_loss_before - objective_after),
        stopping_reason="fixed_inner_step_budget",
    )
    optimizer.zero_grad(set_to_none=True)
    return result
