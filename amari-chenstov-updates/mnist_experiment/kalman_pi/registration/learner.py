"""Physical Gaussian regression objectives and independent landmark metrics."""

from __future__ import annotations

import math
import time

import torch

from .model import predict


GAUSSIAN_CONSTANT = 3 * math.log(8 * math.pi)


def mean_nll(output, target):
    return (output - target).square().sum(-1).mean() / 8 + GAUSSIAN_CONSTANT


class _EvaluationLimit(RuntimeError):
    pass


def fit_batch(model, layout, start, diagonal, inputs, targets, *, pi, ridge,
              max_iter=50, max_eval=75, learning_rate=1., history_size=20,
              tolerance_grad=1e-7, tolerance_change=1e-9, check=lambda: None):
    if not 0 < pi < 1 or ridge < 0 or max_eval < max_iter:
        raise ValueError("invalid bounded regression solve")
    anchor = start.detach().double()
    kappa = ridge * float(diagonal.double().mean())
    precision = (1 - pi) / pi * (diagonal.double() + kappa)
    scale = (precision + diagonal.double()).clamp_min(1e-12).rsqrt().to(start.dtype)
    coordinate = torch.zeros_like(start).requires_grad_(True)
    optimizer = torch.optim.LBFGS([coordinate], lr=learning_rate, max_iter=1,
        max_eval=max_eval, history_size=history_size, tolerance_grad=0.,
        tolerance_change=tolerance_change, line_search_fn="strong_wolfe")
    calls, best = 0, None
    started = time.perf_counter()
    reason = "iteration_limit"
    for _ in range(max_iter):
        check()
        records = []

        def closure():
            nonlocal calls
            check()
            if calls >= max_eval:
                raise _EvaluationLimit()
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            theta = start + scale * coordinate
            data_loss = mean_nll(predict(model, layout, theta, inputs), targets)
            penalty = .5 * (precision * (theta.double() - anchor).square()).sum()
            loss = data_loss.double() + penalty
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(coordinate.grad).all():
                raise FloatingPointError("nonfinite Gaussian EWC objective/gradient")
            records.append({"theta": theta.detach().clone(), "coordinate": coordinate.detach().clone(),
                "objective": float(loss.detach()), "nll": float(data_loss.detach()),
                "penalty": float(penalty.detach()),
                "grad": float((coordinate.grad.detach() / scale).abs().max())})
            return loss

        try:
            optimizer.step(closure)
        except _EvaluationLimit:
            reason = "evaluation_limit"
        if not records:
            break
        previous = best
        best = min(records if best is None else [best, *records], key=lambda item: item["objective"])
        with torch.no_grad():
            reset = not torch.equal(coordinate.detach(), best["coordinate"])
            coordinate.copy_(best["coordinate"])
        if reset:
            optimizer.state.clear()
        if best["grad"] <= tolerance_grad:
            reason = "gradient_tolerance"
            break
        if reason == "evaluation_limit":
            break
        if previous is not None and abs(previous["objective"] - best["objective"]) <= tolerance_change:
            reason = "change_tolerance"
            break
    if best is None:
        raise FloatingPointError("no evaluated regression iterate")
    check()
    return best["theta"], {"closure_count": calls, "solver_seconds": time.perf_counter() - started,
        "objective": best["objective"], "batch_nll": best["nll"], "ewc_penalty": best["penalty"],
        "grad_norm": best["grad"], "solver_reason": reason,
        "solver_coordinates": "diagonal_scaled_physical_objective",
        "step_norm": float((best["theta"] - start).double().norm()), "kappa": kappa}


def evaluate(model, layout, theta, inputs, targets, digits, class_weights, *,
             chunk_size=256, check=lambda: None):
    risks, hits = [], []
    with torch.no_grad():
        for offset in range(0, len(targets), chunk_size):
            check()
            output = predict(model, layout, theta, inputs[offset:offset + chunk_size])
            squared = (output.double() - targets[offset:offset + chunk_size].double()).reshape(-1, 3, 2).square().sum(-1)
            risks.append(squared.mean(-1))
            hits.append((squared <= 1.).double().mean(-1))
    risk, accuracy = torch.cat(risks), torch.cat(hits)
    digits = digits.to(risk.device)
    per_risk, per_accuracy = [], []
    for digit in range(10):
        mask = digits == digit
        if not mask.any():
            raise ValueError("evaluation panel must contain all ten digits")
        per_risk.append(float(risk[mask].mean()))
        per_accuracy.append(float(accuracy[mask].mean()))
    weighted_risk = sum(float(class_weights[k]) * per_risk[k] for k in range(10))
    return {"risk": weighted_risk, "rms": math.sqrt(weighted_risk),
        "accuracy": sum(float(class_weights[k]) * per_accuracy[k] for k in range(10)),
        "nll": 3 * weighted_risk / 8 + GAUSSIAN_CONSTANT,
        "per_digit_risk": per_risk, "per_digit_accuracy": per_accuracy}
