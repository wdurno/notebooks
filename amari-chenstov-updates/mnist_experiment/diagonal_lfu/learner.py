"""Bounded diagonal-EWC local likelihood solves and holdout predictions."""

from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from .curvature import logits


class _EvaluationLimit(RuntimeError):
    pass


def fit_batch(model, layout, start, diagonal, inputs, labels, *, c, ridge_ratio,
              max_iter=50, max_eval=75, history_size=20, learning_rate=1.,
              tolerance_grad=1e-5, tolerance_change=1e-9, check=lambda: None):
    """Reset L-BFGS per arrival; keep only evaluated nonincreasing iterates."""
    anchor, precision = start.detach().double(), diagonal.detach().double()
    kappa = ridge_ratio * float(precision.mean())
    precision = (1 - c) / c * (precision + kappa)
    # Solve the same physical objective in coordinates scaled by its local diagonal.
    scale = (precision + diagonal.double()).clamp_min(1e-12).rsqrt().to(start.dtype)
    parameter = torch.zeros_like(start).requires_grad_(True)
    optimizer = torch.optim.LBFGS([parameter], lr=learning_rate, max_iter=1,
                                max_eval=max_eval, history_size=history_size,
                                tolerance_grad=0., tolerance_change=tolerance_change,
                                line_search_fn="strong_wolfe")
    calls, accepted_record = 0, None
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
            physical = start + scale * parameter
            nll = F.cross_entropy(logits(model, layout, physical, inputs), labels)
            value = nll.double() + .5 * (precision * (physical.double() - anchor).square()).sum()
            value.backward()
            if not torch.isfinite(value) or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("nonfinite local likelihood solve")
            records.append({"theta": physical.detach().clone(), "coordinate": parameter.detach().clone(),
                            "value": float(value.detach()),
                            "grad": float((parameter.grad.detach() / scale).abs().max())})
            return value

        try:
            optimizer.step(closure)
        except _EvaluationLimit:
            reason = "evaluation_limit"
        if not records:
            break
        candidates = records if accepted_record is None else [accepted_record, *records]
        best = min(candidates, key=lambda record: record["value"])
        old = accepted_record
        accepted_record = best
        with torch.no_grad():
            reset_history = not torch.equal(parameter.detach(), best["coordinate"])
            parameter.copy_(best["coordinate"])
        if reset_history:
            optimizer.state.clear()
        if best["grad"] <= tolerance_grad:
            reason = "gradient_tolerance"
            break
        if reason == "evaluation_limit":
            break
        if old is not None and abs(old["value"] - best["value"]) <= tolerance_change:
            reason = "change_tolerance"
            break
    if accepted_record is None:
        raise FloatingPointError("no evaluated local iterate")
    check()
    accepted = accepted_record["theta"]
    return accepted, {"closure_count": calls, "grad_norm": accepted_record["grad"],
                      "solver_coordinates": "diagonal_scaled_physical_objective",
                      "solver_reason": reason, "objective": accepted_record["value"],
                      "solver_seconds": time.perf_counter() - started,
                      "step_norm": float((accepted - start).double().norm()), "kappa": kappa}


def evaluate(model, layout, theta, inputs, labels, non9_weights, *, chunk_size=256,
             check=lambda: None):
    predictions, losses = [], []
    with torch.no_grad():
        for offset in range(0, len(labels), chunk_size):
            check()
            output = logits(model, layout, theta, inputs[offset:offset + chunk_size])
            predictions.append(output.argmax(-1))
            losses.append(F.cross_entropy(output, labels[offset:offset + chunk_size], reduction="none"))
    predicted, loss = torch.cat(predictions), torch.cat(losses).double()
    accuracy, false_positive, class_nll = [], [], []
    for k in range(10):
        mask = labels == k
        if not mask.any():
            raise ValueError("evaluation panel must contain every digit")
        accuracy.append(float((predicted[mask] == k).double().mean()))
        false_positive.append(float((predicted[mask] == 9).double().mean()))
        class_nll.append(float(loss[mask].mean()))
    fpr = sum(float(non9_weights[k]) * false_positive[k] for k in range(9))
    denom = .1 * accuracy[9] + .9 * fpr
    return {"nine_recall": accuracy[9], "old_accuracy": sum(accuracy[:9]) / 9,
            "nine_fpr": fpr, "nine_precision": .1 * accuracy[9] / denom if denom else None,
            "precision_undefined": denom == 0, "nll": sum(class_nll) / 10,
            "nine_nll": class_nll[9], "old_nll": sum(class_nll[:9]) / 9,
            "class_accuracy": accuracy}
