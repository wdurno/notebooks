"""Fixed-anchor, weighted restricted fits with charged strong-Wolfe L-BFGS."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call

from src.parameters import ParameterLayout

from .budget import WorkBudget


class _FitStop(Exception):
    pass


def _precision_product(precision: Any, displacement: Tensor) -> Tensor:
    """Apply the full, symmetric episode precision, including cross blocks."""
    dtype = getattr(precision, "dtype", displacement.dtype)
    vector = displacement.to(dtype=dtype)
    if isinstance(precision, Tensor):
        if precision.ndim == 1 and precision.shape == vector.shape:
            product = precision * vector
        elif precision.shape == (vector.numel(), vector.numel()):
            product = precision @ vector
        else:
            raise ValueError("precision has incompatible dimensions")
    else:
        product = precision.matvec(vector)
    if product.shape != displacement.shape:
        raise ValueError("precision product has incompatible dimensions")
    return product.detach().to(dtype=torch.float64)


def fit_restricted(
    model: nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    weights: Tensor,
    anchor: Tensor,
    precision: Any,
    indices: Tensor,
    *,
    max_iter: int = 50,
    max_eval: int = 75,
    tolerance_grad: float = 1e-5,
    tolerance_change: float = 1e-9,
    residual_tolerance: float | None = None,
    budget: WorkBudget | None = None,
    charge_key: str | None = None,
) -> dict[str, Any]:
    """Fit ``sum(a_i CE_i) + (theta-anchor)' P (theta-anchor)/2``.

    Weights are normalized once. ``anchor`` is the fixed episode anchor,
    not the warm start. ``precision`` is a symmetric tensor (dense/diagonal)
    or a symmetric ``matvec`` operator and already includes all EWC scaling.
    Only selected coordinates have an autograd path; frozen layers and inputs
    are detached. Returned gradients use the loss-gradient convention g=-s,
    have shape ``(len(indices),)``, and follow ``gradient_indices`` ordering.

    Every new objective/gradient evaluation, including line-search trials,
    consumes one ``charge_key`` unit. Cached accepted gradients supply the
    exact restricted residual without a final diagnostic backward pass.
    Convergence requires its infinity norm <= ``residual_tolerance`` (defaults
    to ``tolerance_grad``). One unresolved fit retries with a fresh optimizer
    at twice both ceilings, warm-starting at the last accepted state.

    The model is only updated at completed accepted steps. The result is a
    safe checkpoint boundary; no optimizer history survives this local fit.
    A denial before any evaluation returns None for unavailable diagnostics.
    """
    layout.validate_module(model)
    before = layout.flatten_module(model, detach=True)
    if not torch.isfinite(before).all():
        raise ValueError("initial parameters must be finite")
    if anchor.shape != before.shape or not torch.isfinite(anchor).all():
        raise ValueError("anchor must be a finite full parameter vector")
    anchor = anchor.detach().to(device=before.device, dtype=torch.float64).clone()
    indices = torch.as_tensor(indices, device=before.device)
    if indices.ndim != 1 or indices.dtype != torch.long or not indices.numel():
        raise ValueError("indices must be a nonempty int64 vector")
    if (indices < 0).any() or (indices >= before.numel()).any() or indices.unique().numel() != indices.numel():
        raise ValueError("indices must be unique and in range")
    indices = indices.clone()
    for name, value in (("max_iter", max_iter), ("max_eval", max_eval)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    residual_tolerance = tolerance_grad if residual_tolerance is None else residual_tolerance
    for value in (tolerance_grad, tolerance_change, residual_tolerance):
        if not math.isfinite(value) or value < 0:
            raise ValueError("optimizer tolerances must be finite and nonnegative")
    if budget is not None:
        if charge_key is None:
            raise ValueError("budgeted fitting requires an explicit charge_key")
        budget.can_spend(charge_key, units=0)
    if isinstance(precision, Tensor):
        if precision.device != before.device or not precision.is_floating_point():
            raise ValueError("precision must be floating point on the model device")
        if precision.shape not in ((before.numel(),), (before.numel(), before.numel())):
            raise ValueError("precision has incompatible dimensions")
        if not torch.isfinite(precision).all():
            raise ValueError("precision must be finite")
        if precision.ndim == 2 and not torch.allclose(precision, precision.mT):
            raise ValueError("precision must be symmetric")
        precision = precision.detach().to(dtype=torch.float64).clone()
        if precision.ndim == 2:
            precision = (precision + precision.mT) / 2
    inputs, targets = inputs.detach(), targets.detach()
    weights = torch.as_tensor(weights, device=before.device, dtype=torch.float64).detach()
    if targets.ndim != 1 or inputs.shape[0] != targets.numel() or weights.shape != targets.shape:
        raise ValueError("inputs, targets, and weights must have the same observation count")
    if not weights.numel() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be nonempty, finite, and nonnegative")
    weight_sum = weights.sum()
    if not torch.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("weights must have a positive finite sum")
    weights = weights / weight_sum

    fixed_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    fixed_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    selected = []
    for spec in layout.specs:
        positions = torch.nonzero((indices >= spec.start) & (indices < spec.stop)).flatten()
        if positions.numel():
            selected.append((spec, positions, indices[positions] - spec.start))
    coordinates = nn.Parameter(before[indices].clone())
    accepted_coordinates = coordinates.detach().clone()
    accepted: dict[str, Any] | None = None
    initial: dict[str, Any] | None = None
    evaluations = gradient_evaluations = forward_evaluations = 0
    iterations = accepted_steps = cache_hits = 0
    attempts: list[dict[str, Any]] = []
    spent_before = 0.0 if budget is None else budget.spent
    stopping_reason = "max_iterations"
    budget_denied = False

    def evaluate() -> dict[str, Any]:
        nonlocal evaluations, gradient_evaluations, forward_evaluations
        if budget is not None and not budget.spend(charge_key):
            raise _FitStop("budget_denied")
        evaluations += 1
        if not torch.isfinite(coordinates).all():
            raise _FitStop("nonfinite_parameters")
        parameters = dict(fixed_parameters)
        for spec, positions, local_indices in selected:
            parameters[spec.name] = fixed_parameters[spec.name].reshape(-1).index_copy(
                0, local_indices, coordinates[positions]
            ).reshape(spec.shape)
        buffers = {name: value.clone() for name, value in fixed_buffers.items()}
        forward_evaluations += 1
        logits = functional_call(model, (parameters, buffers), (inputs,), strict=True)
        losses = nn.functional.cross_entropy(logits, targets, reduction="none")
        data_loss = weights @ losses.to(dtype=torch.float64)
        if data_loss.requires_grad:
            gradient_evaluations += 1
            data_gradient = torch.autograd.grad(data_loss, coordinates, allow_unused=True)[0]
        else:
            data_gradient = None
        if data_gradient is None:
            data_gradient = torch.zeros_like(coordinates)
        vector = before.index_copy(0, indices, coordinates.detach()).to(dtype=torch.float64)
        displacement = vector - anchor
        with torch.no_grad():
            penalty_product = _precision_product(precision, displacement)
            penalty = 0.5 * (displacement @ penalty_product)
            penalty_gradient = penalty_product[indices]
            retained_gradient = data_gradient.to(dtype=torch.float64) + penalty_gradient
            objective = data_loss.detach() + penalty
        if not all(bool(torch.isfinite(value).all()) for value in (
            objective, data_loss, penalty, data_gradient, penalty_gradient, retained_gradient,
        )):
            raise _FitStop("nonfinite_objective_or_gradient")
        return {
            "coordinates": coordinates.detach().clone(),
            "objective": float(objective),
            "data_loss": float(data_loss.detach()),
            "penalty": float(penalty),
            "data_gradient": data_gradient.detach().to(dtype=torch.float64).clone(),
            "penalty_gradient": penalty_gradient.clone(),
            "retained_gradient": retained_gradient.clone(),
        }

    def resolved(record: dict[str, Any] | None) -> bool:
        return record is not None and float(record["retained_gradient"].abs().max()) <= residual_tolerance

    for attempt in range(2):
        attempt_start = evaluations
        iteration_limit, evaluation_limit = max_iter * (attempt + 1), max_eval * (attempt + 1)
        optimizer = torch.optim.LBFGS(
            [coordinates], lr=1.0, max_iter=1, max_eval=evaluation_limit,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            history_size=20, line_search_fn="strong_wolfe",
        )
        attempt_iterations = 0
        try:
            for _ in range(iteration_limit):
                records = [] if accepted is None else [accepted]

                def closure() -> Tensor:
                    nonlocal accepted, initial, accepted_coordinates, cache_hits
                    record = next((r for r in records if torch.equal(r["coordinates"], coordinates.detach())), None)
                    if record is None:
                        if evaluations - attempt_start >= evaluation_limit:
                            raise _FitStop("max_evaluations")
                        record = evaluate()
                        records.append(record)
                        if accepted is None:
                            # The finite starting point is accepted, never a trial point.
                            accepted = initial = record
                            accepted_coordinates = record["coordinates"].clone()
                    else:
                        cache_hits += 1
                    coordinates.grad = record["retained_gradient"].to(coordinates).clone()
                    return torch.tensor(record["objective"], dtype=torch.float64, device=before.device)

                # One iteration per call exposes accepted boundaries; L-BFGS
                # history is retained within this attempt, not across local fits.
                optimizer.param_groups[0]["max_eval"] = max(2, evaluation_limit - (evaluations - attempt_start) + 1)
                previous_iterations = int(optimizer.state.get(coordinates, {}).get("n_iter", 0))
                try:
                    optimizer.step(closure)
                finally:
                    new_iterations = int(optimizer.state.get(coordinates, {}).get("n_iter", 0)) - previous_iterations
                    attempt_iterations += new_iterations
                    iterations += new_iterations
                candidate = next((r for r in records if torch.equal(r["coordinates"], coordinates.detach())), None)
                if candidate is None:
                    raise _FitStop("unevaluated_candidate")
                tolerance = 64 * torch.finfo(before.dtype).eps * max(abs(accepted["objective"]), 1.0)
                if candidate["objective"] > accepted["objective"] + tolerance:
                    raise _FitStop("objective_increase")
                moved = not torch.equal(accepted_coordinates, candidate["coordinates"])
                previous_objective = accepted["objective"]
                step_max = float((candidate["coordinates"] - accepted_coordinates).abs().max())
                accepted = candidate
                accepted_coordinates = candidate["coordinates"].clone()
                if moved:
                    accepted_steps += 1
                    layout.copy_vector_to_module(model, before.index_copy(0, indices, accepted_coordinates))
                if resolved(accepted):
                    stopping_reason = "restricted_residual"
                    break
                if not moved or step_max <= tolerance_change or abs(previous_objective - accepted["objective"]) < tolerance_change:
                    stopping_reason = "lbfgs_change_tolerance"
                    break
                stopping_reason = "max_iterations"
        except _FitStop as exc:
            stopping_reason = str(exc)
            budget_denied = stopping_reason == "budget_denied"
        finally:
            # A denied/nonfinite Wolfe trial is not an accepted optimizer step.
            with torch.no_grad():
                coordinates.copy_(accepted_coordinates)
            layout.copy_vector_to_module(model, before.index_copy(0, indices, accepted_coordinates))
        attempts.append({
            "max_iter": iteration_limit, "max_eval": evaluation_limit,
            "iterations": attempt_iterations, "function_evaluations": evaluations - attempt_start,
            "stopping_reason": stopping_reason,
        })
        if resolved(accepted) or budget_denied:
            break

    after = before.index_copy(0, indices, accepted_coordinates)
    converged = resolved(accepted)
    residual = None if accepted is None else accepted["retained_gradient"]
    result = {
        "status": "converged" if converged else ("budget_denied" if budget_denied else "restricted_fit_unresolved"),
        "converged": converged, "budget_denied": budget_denied,
        "restricted_fit_unresolved": not converged,
        "stopping_reason": stopping_reason,
        "before": before, "after": after, "displacement": after - before,
        "gradient_indices": indices, "active_dimension": indices.numel(),
        "initial_gradient": None if initial is None else initial["retained_gradient"],
        "data_gradient": None if accepted is None else accepted["data_gradient"],
        "penalty_gradient": None if accepted is None else accepted["penalty_gradient"],
        "retained_gradient": residual,
        "restricted_residual_norm": None if residual is None else float(torch.linalg.vector_norm(residual)),
        "restricted_residual_max": None if residual is None else float(residual.abs().max()),
        "restricted_residual_rms": None if residual is None else float(residual.square().mean().sqrt()),
        "residual_tolerance": residual_tolerance,
        "function_evaluations": evaluations, "gradient_evaluations": gradient_evaluations,
        "forward_evaluations": forward_evaluations,
        "iterations": iterations, "accepted_steps": accepted_steps, "cache_hits": cache_hits,
        "retries": len(attempts) - 1, "attempts": attempts,
        "charged_work": 0.0 if budget is None else budget.spent - spent_before,
    }
    for name in ("objective", "data_loss", "penalty"):
        result[f"{name}_before"] = None if initial is None else initial[name]
        result[f"{name}_after"] = None if accepted is None else accepted[name]
    return result
