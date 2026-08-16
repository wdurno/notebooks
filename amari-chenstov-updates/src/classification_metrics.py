"""Digit-9 classification metrics at an explicit environmental prevalence."""

from __future__ import annotations

import math
from typing import Any


def nine_environment_metrics(
    *,
    prevalence: float,
    recall: float,
    false_positive_rate: float,
    non_nine_accuracy: float,
) -> dict[str, float | None]:
    """Reweight class-conditional holdout rates to environmental prevalence."""

    values = {
        "prevalence": prevalence,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "non_nine_accuracy": non_nine_accuracy,
    }
    if any(
        isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in values.values()
    ):
        raise ValueError("classification probabilities must be finite and in [0, 1]")

    p = float(prevalence)
    true_positive_rate = float(recall)
    false_positive = float(false_positive_rate)
    specificity = 1.0 - false_positive
    predicted_positive_probability = (
        p * true_positive_rate + (1.0 - p) * false_positive
    )
    return {
        "nine_metric_prevalence": p,
        "nine_recall": true_positive_rate,
        "nine_false_positive_rate": false_positive,
        "nine_specificity": specificity,
        "nine_ovr_accuracy": (
            p * true_positive_rate + (1.0 - p) * specificity
        ),
        "nine_precision": (
            None
            if predicted_positive_probability == 0.0
            else p * true_positive_rate / predicted_positive_probability
        ),
        "environment_accuracy": (
            p * true_positive_rate
            + (1.0 - p) * float(non_nine_accuracy)
        ),
    }


NINE_ENVIRONMENT_METRIC_NAMES = tuple(
    nine_environment_metrics(
        prevalence=0.5,
        recall=0.5,
        false_positive_rate=0.5,
        non_nine_accuracy=0.5,
    )
)


def select_nine_classification_metrics(
    evaluation: dict[str, Any],
) -> dict[str, float | int | None]:
    """Select the auditable confusion and environmental fields from evaluation."""

    names = (
        "nine_true_positive_count",
        "nine_false_positive_count",
        "nine_true_negative_count",
        "nine_false_negative_count",
        *NINE_ENVIRONMENT_METRIC_NAMES,
    )
    return {name: evaluation[name] for name in names}
