"""Batched holdout evaluation, with prevalence used only as evaluation weights."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from mnist_experiment.rotated_mnist.config import RotationConfig
from mnist_experiment.rotated_mnist.transform import rotate_mnist_batch


P_REFERENCE = 0.1


def _predictions(model: nn.Module, inputs: Tensor, targets: Tensor, *,
                 angle: float = 0.0, device=None, batch_size: int = 256):
    if inputs.ndim != 4 or inputs.shape[1:] != (1, 28, 28):
        raise ValueError("evaluation inputs must have shape (n, 1, 28, 28)")
    if (targets.ndim != 1 or not targets.numel() or len(inputs) != len(targets)
            or targets.dtype != torch.long or (targets < 0).any() or (targets > 9).any()):
        raise ValueError("evaluation targets must be a nonempty aligned int64 digit vector")
    if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
        raise ValueError("evaluation inputs must be finite floating-point tensors")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    parameter = next(model.parameters(), None)
    actual_device = inputs.device if parameter is None else parameter.device
    device = actual_device if device is None else torch.device(device)
    if device.type != actual_device.type or (device.index is not None and device.index != actual_device.index):
        raise ValueError("evaluation device must match the model device")
    dtype = inputs.dtype if parameter is None else parameter.dtype
    modes = [(module, module.training) for module in model.modules()]
    losses, predictions, probabilities = [], [], []
    rotation = RotationConfig((0.0, 30.0, 0.0, 30.0), 40, "bilinear", False, 0.0)
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(inputs), batch_size):
                batch = inputs[start:start + batch_size].detach().cpu()
                if angle != 0:
                    batch = rotate_mnist_batch(batch, angle, rotation)
                logits = model(batch.to(device=device, dtype=dtype))
                if logits.shape != (len(batch), 10) or not torch.isfinite(logits).all():
                    raise ValueError("model must produce ten finite logits per observation")
                labels = targets[start:start + batch_size].to(device)
                log_probs = logits.to(torch.float64).log_softmax(-1)
                losses.append(-log_probs.gather(1, labels[:, None]).squeeze(1).cpu())
                predictions.append(logits.argmax(-1).cpu())
                probabilities.append(log_probs[:, 9].exp().cpu())
    finally:
        for module, training in modes:
            module.training = training
    return torch.cat(losses), torch.cat(predictions), torch.cat(probabilities)


def source_quality(model: nn.Module, inputs: Tensor, targets: Tensor, *,
                   device=None, batch_size: int = 256) -> dict[str, float]:
    """Unweighted upright validation quality; valid for non-nine-only panels."""
    losses, predictions, _ = _predictions(model, inputs, targets, device=device,
                                         batch_size=batch_size)
    return {"nll": float(losses.mean()),
            "accuracy": float((predictions == targets.cpu()).double().mean())}


def _nine_metrics(targets: Tensor, predictions: Tensor, scores: Tensor,
                  *, return_pr_curve: bool) -> dict[str, Any]:
    positive = targets == 9
    predicted = predictions == 9
    tp = int((positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    fp = int((~positive & predicted).sum())
    tn = int((~positive & ~predicted).sum())
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    denominator = None if recall is None or fpr is None else (
        P_REFERENCE * recall + (1 - P_REFERENCE) * fpr)
    defined = denominator is not None and denominator > 0
    result = {
        "nine_precision": P_REFERENCE * recall / denominator if defined else None,
        "nine_precision_defined": defined,
        "nine_recall": recall, "nine_false_positive_rate": fpr,
        "nine_tp": tp, "nine_fn": fn, "nine_fp": fp, "nine_tn": tn,
        "nine_positive_count": tp + fn, "nine_negative_count": fp + tn,
        "nine_predicted_positive_count": tp + fp,
        "nine_reference_prevalence": P_REFERENCE,
        "nine_average_precision": None,
    }
    if recall is None or fpr is None:
        return result
    # Group tied scores before integration: AP must not depend on tie ordering.
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    labels = positive[order].double()
    endpoints = torch.cat((torch.nonzero(sorted_scores[:-1] != sorted_scores[1:]).flatten(),
                           torch.tensor([len(scores) - 1])))
    tpr = labels.cumsum(0)[endpoints] / (tp + fn)
    fprs = (1 - labels).cumsum(0)[endpoints] / (fp + tn)
    precision = P_REFERENCE * tpr / (P_REFERENCE * tpr + (1 - P_REFERENCE) * fprs)
    recall_increments = torch.diff(torch.cat((torch.zeros(1, dtype=torch.float64), tpr)))
    result["nine_average_precision"] = float((recall_increments * precision).sum())
    if return_pr_curve:
        result["nine_pr_curve"] = {
            "thresholds": torch.cat((torch.tensor([float("inf")]), sorted_scores[endpoints])),
            "precision": torch.cat((torch.tensor([float("nan")]), precision)),
            "precision_defined": torch.cat((torch.tensor([False]), torch.ones(len(tpr), dtype=torch.bool))),
            "recall": torch.cat((torch.zeros(1, dtype=torch.float64), tpr)),
            "false_positive_rate": torch.cat((torch.zeros(1, dtype=torch.float64), fprs)),
        }
    return result


def evaluate(model: nn.Module, inputs: Tensor, targets: Tensor, *, family: str,
             position: float, device=None, batch_size: int = 256,
             return_pr_curve: bool = False) -> dict[str, Any]:
    """Evaluate current, retention and fixed panels without caching rotated images.

    Rotation positions are degrees; mixture positions are prevalences. Mixture
    metrics average empirical conditional digit-9/non-9 risks with weights p and
    1-p, never reweight training or select thresholds. Ordinary argmax defines
    the digit-9 operating point; its precision is standardized to p_ref=.1.
    Optional ``nine_pr_curve`` CPU tensors belong in a torch artifact, not JSON.
    """
    if family not in {"rotation", "mixture"} or not math.isfinite(position):
        raise ValueError("family must be rotation/mixture and position must be finite")
    if family == "mixture" and not 0 <= position <= 1:
        raise ValueError("mixture position must be in [0, 1]")
    targets = targets.detach().cpu()
    angles = dict.fromkeys((float(position), 0.0, 30.0)) if family == "rotation" else {0.0: None}
    panels = {angle: _predictions(model, inputs, targets, angle=angle, device=device,
                                 batch_size=batch_size) for angle in angles}
    current = panels[float(position)] if family == "rotation" else panels[0.0]
    losses, predictions, scores = current
    correct = (predictions == targets).double()
    if family == "rotation":
        retention, fixed = panels[0.0], panels[30.0]
        result = {"current_nll": float(losses.mean()), "current_accuracy": float(correct.mean()),
                  "retention_nll": float(retention[0].mean()),
                  "retention_accuracy": float((retention[1] == targets).double().mean()),
                  "fixed_nll": float(fixed[0].mean()),
                  "fixed_accuracy": float((fixed[1] == targets).double().mean())}
    else:
        nine = targets == 9
        if not nine.any() or nine.all():
            raise ValueError("mixture evaluation requires both digit-9 and non-9 observations")
        nll0, nll9 = float(losses[~nine].mean()), float(losses[nine].mean())
        acc0, acc9 = float(correct[~nine].mean()), float(correct[nine].mean())
        result = {"current_nll": (1 - position) * nll0 + position * nll9,
                  "current_accuracy": (1 - position) * acc0 + position * acc9,
                  "retention_nll": nll0, "retention_accuracy": acc0,
                  "fixed_nll": (1 - P_REFERENCE) * nll0 + P_REFERENCE * nll9,
                  "fixed_accuracy": (1 - P_REFERENCE) * acc0 + P_REFERENCE * acc9,
                  "nine_nll": nll9, "non_nine_nll": nll0}
    result.update(_nine_metrics(targets, predictions, scores, return_pr_curve=return_pr_curve))
    return result
