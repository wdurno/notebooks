import math

import numpy as np

from picar_kl.actions import ACTION_NAMES, action_count
from picar_kl.phase2.metrics import Phase2Breakout, action_metrics_from_arrays, merge_metric_sums


def test_action_metrics_reports_accuracy_probability_entropy_and_confusion():
    probabilities = np.zeros((1, 3, action_count()), dtype=np.float32)
    targets = np.zeros((1, 3, action_count()), dtype=np.float32)
    probabilities[0, 0, 0] = 0.9
    probabilities[0, 0, 1] = 0.1
    probabilities[0, 1, 2] = 0.4
    probabilities[0, 1, 3] = 0.6
    probabilities[0, 2, 4] = 1.0
    targets[0, 0, 0] = 1.0
    targets[0, 1, 2] = 1.0
    targets[0, 2, 4] = 1.0
    mask = np.array([[True, True, False]])

    metrics = action_metrics_from_arrays(probabilities=probabilities, targets=targets, mask=mask)

    assert metrics["valid_steps"] == 2
    assert metrics["top1_accuracy"] == 0.5
    assert math.isclose(metrics["target_action_probability"], 0.65, rel_tol=1e-6)
    assert metrics["entropy"] > 0.0
    assert metrics["confusion_matrix"][0][0] == 1
    assert metrics["confusion_matrix"][2][3] == 1
    assert metrics["action_names"] == list(ACTION_NAMES)


def test_merge_metric_sums_uses_valid_step_weighting():
    rows = [
        {
            "valid_steps": 1,
            "top1_accuracy": 1.0,
            "target_action_probability": 0.8,
            "entropy": 0.2,
            "confusion_matrix": np.eye(action_count(), dtype=int).tolist(),
        },
        {
            "valid_steps": 3,
            "top1_accuracy": 0.0,
            "target_action_probability": 0.4,
            "entropy": 0.6,
            "confusion_matrix": np.zeros((action_count(), action_count()), dtype=int).tolist(),
        },
    ]

    merged = merge_metric_sums(rows)

    assert merged["valid_steps"] == 4
    assert merged["top1_accuracy"] == 0.25
    assert math.isclose(merged["target_action_probability"], 0.5, rel_tol=1e-6)
    assert math.isclose(merged["entropy"], 0.5, rel_tol=1e-6)


def test_breakout_serializes_split_metadata():
    breakout = Phase2Breakout(
        fit_mode="window_sampling_fit",
        split_name="validation",
        split_strategy="random_window",
        run_ids=("run-b", "run-a"),
        context_steps=8,
        prediction_steps=4,
        window_stride=None,
        sampling_policy="shuffle_without_replacement",
        random_seed=11,
        metadata={"window_count": 5},
    )

    assert breakout.to_dict() == {
        "fit_mode": "window_sampling_fit",
        "split_name": "validation",
        "split_strategy": "random_window",
        "run_ids": ["run-b", "run-a"],
        "context_steps": 8,
        "prediction_steps": 4,
        "window_stride": None,
        "sampling_policy": "shuffle_without_replacement",
        "random_seed": 11,
        "metadata": {"window_count": 5},
    }
