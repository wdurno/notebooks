"""Shared materialization and one-pass metrics for Plan 5 Phase 4."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.classification_metrics import nine_environment_metrics

from .config import RotationConfig
from .transform import rotate_mnist_batch, tensor_content_hash


def materialize_base_panel(
    dataset: Dataset,
    indices: Sequence[int],
    *,
    num_workers: int,
) -> tuple[Tensor, Tensor]:
    loader = DataLoader(
        Subset(dataset, tuple(indices)),
        batch_size=1024,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    inputs = []
    targets = []
    for batch_inputs, batch_targets in loader:
        inputs.append(batch_inputs.to(device="cpu", dtype=torch.float32))
        targets.append(batch_targets.to(device="cpu", dtype=torch.long))
    return torch.cat(inputs).contiguous(), torch.cat(targets).contiguous()


def materialize_rotated_panel(
    base_inputs: Tensor,
    targets: Tensor,
    angle_degrees: float,
    rotation: RotationConfig,
) -> tuple[Tensor, Tensor, str]:
    inputs = rotate_mnist_batch(base_inputs, angle_degrees, rotation)
    digest = hashlib.sha256(
        (
            tensor_content_hash(inputs)
            + tensor_content_hash(targets)
            + f"{float(angle_degrees):.17g}"
        ).encode("ascii")
    ).hexdigest()
    return inputs, targets, digest


def evaluate_materialized_classifier(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    calibration_bins: int = 10,
    nine_prevalence: float,
) -> dict[str, Any]:
    if inputs.ndim != 4 or inputs.shape[1:] != (1, 28, 28):
        raise ValueError("materialized MNIST inputs have an invalid shape")
    if targets.shape != (inputs.shape[0],) or inputs.shape[0] == 0:
        raise ValueError("materialized targets do not match the inputs")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least two")
    if not 0.0 <= float(nine_prevalence) <= 1.0:
        raise ValueError("nine prevalence must be in [0, 1]")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    brier_total = 0.0
    group_loss = torch.zeros(2, dtype=torch.float64)
    group_brier = torch.zeros(2, dtype=torch.float64)
    calibration_count = torch.zeros(calibration_bins, dtype=torch.long)
    calibration_confidence = torch.zeros(calibration_bins, dtype=torch.float64)
    calibration_correct = torch.zeros(calibration_bins, dtype=torch.float64)
    boundaries = torch.linspace(
        0.0, 1.0, calibration_bins + 1, dtype=torch.float64
    )[1:-1]

    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch_inputs = inputs[start : start + batch_size].to(
                device=device, dtype=dtype
            )
            batch_targets = targets[start : start + batch_size].to(device=device)
            logits = model(batch_inputs)
            losses = nn.functional.cross_entropy(
                logits, batch_targets, reduction="none"
            )
            probabilities = torch.softmax(logits, dim=1).double()
            predictions = logits.argmax(dim=1)
            correct = predictions == batch_targets
            total_loss += float(losses.sum())
            encoded = batch_targets * 10 + predictions
            confusion.add_(
                torch.bincount(encoded, minlength=100).reshape(10, 10).cpu()
            )
            targets_one_hot = nn.functional.one_hot(
                batch_targets, num_classes=10
            ).double()
            brier = (probabilities - targets_one_hot).square().sum(dim=1)
            brier_total += float(brier.sum())
            is_nine = batch_targets == 9
            for group, mask in enumerate((~is_nine, is_nine)):
                group_loss[group] += float(losses[mask].sum())
                group_brier[group] += float(brier[mask].sum())

            confidence, calibrated_predictions = probabilities.max(dim=1)
            bins = torch.bucketize(confidence.cpu(), boundaries)
            calibration_count += torch.bincount(bins, minlength=calibration_bins)
            calibration_confidence.scatter_add_(0, bins, confidence.cpu())
            calibration_correct.scatter_add_(
                0, bins, (calibrated_predictions == batch_targets).double().cpu()
            )
    model.train(was_training)

    class_count = confusion.sum(dim=1)
    predicted_count = confusion.sum(dim=0)
    if bool((class_count == 0).any()):
        raise ValueError("evaluation panel must contain every MNIST class")
    class_correct = torch.diagonal(confusion)
    class_recall = class_correct.double() / class_count.double()
    class_precision = [
        None if int(count) == 0 else float(class_correct[index] / count)
        for index, count in enumerate(predicted_count)
    ]
    total_count = int(class_count.sum())
    nine_count = int(class_count[9])
    non_nine_count = total_count - nine_count
    nine_correct = int(class_correct[9])
    non_nine_correct = int(class_correct[:9].sum())
    nine_false_positive = int(confusion[:9, 9].sum())
    populated = calibration_count > 0
    mean_confidence = (
        calibration_confidence[populated]
        / calibration_count[populated].double()
    )
    mean_accuracy = (
        calibration_correct[populated] / calibration_count[populated].double()
    )
    weights = calibration_count[populated].double() / total_count
    nine_metrics = nine_environment_metrics(
        prevalence=float(nine_prevalence),
        recall=nine_correct / nine_count,
        false_positive_rate=nine_false_positive / non_nine_count,
        non_nine_accuracy=non_nine_correct / non_nine_count,
    )
    return {
        "sample_count": total_count,
        "nll": total_loss / total_count,
        "accuracy": int(class_correct.sum()) / total_count,
        "environment_accuracy": int(class_correct.sum()) / total_count,
        "non_nine_nll": float(group_loss[0]) / non_nine_count,
        "non_nine_accuracy": non_nine_correct / non_nine_count,
        "nine_nll": float(group_loss[1]) / nine_count,
        "nine_accuracy": nine_correct / nine_count,
        "nine_true_positive_count": nine_correct,
        "nine_false_positive_count": nine_false_positive,
        "nine_true_negative_count": non_nine_count - nine_false_positive,
        "nine_false_negative_count": nine_count - nine_correct,
        **{
            name: value
            for name, value in nine_metrics.items()
            if name != "environment_accuracy"
        },
        "nine_prevalence_adjusted_environment_accuracy": nine_metrics[
            "environment_accuracy"
        ],
        "brier": brier_total / total_count,
        "non_nine_brier": float(group_brier[0]) / non_nine_count,
        "nine_brier": float(group_brier[1]) / nine_count,
        "expected_calibration_error": float(
            (weights * (mean_confidence - mean_accuracy).abs()).sum()
        ),
        "calibration_bin_count": calibration_bins,
        "confusion_matrix": confusion.tolist(),
        "per_class_precision": class_precision,
        "per_class_recall": [float(value) for value in class_recall],
        "worst_class_recall": float(class_recall.min()),
        "worst_class_label": int(torch.argmin(class_recall)),
    }
