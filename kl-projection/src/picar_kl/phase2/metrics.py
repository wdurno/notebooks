"""Metrics for phase 2 KL-projection fits."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from picar_kl.actions import ACTION_NAMES, action_count


@dataclass(frozen=True)
class Phase2Breakout:
    fit_mode: str
    split_name: str
    split_strategy: str
    run_ids: tuple[str, ...]
    context_steps: int
    prediction_steps: int
    window_stride: int | None
    sampling_policy: str
    random_seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_mode": self.fit_mode,
            "split_name": self.split_name,
            "split_strategy": self.split_strategy,
            "run_ids": list(self.run_ids),
            "context_steps": int(self.context_steps),
            "prediction_steps": int(self.prediction_steps),
            "window_stride": None if self.window_stride is None else int(self.window_stride),
            "sampling_policy": self.sampling_policy,
            "random_seed": self.random_seed,
            "metadata": dict(self.metadata),
        }


def action_metrics_from_arrays(
    *,
    probabilities: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if probs.shape != target.shape:
        raise ValueError("probabilities and targets must have matching shapes")
    if probs.ndim != 3:
        raise ValueError("probabilities must have shape [batch, steps, action_dim]")
    if valid.shape != probs.shape[:2]:
        raise ValueError("mask must have shape [batch, steps]")
    if probs.shape[-1] != action_count():
        raise ValueError(f"expected {action_count()} actions")
    if not valid.any():
        return _empty_metrics()

    flat_probs = probs[valid]
    flat_targets = target[valid]
    predicted_idx = flat_probs.argmax(axis=-1)
    target_idx = flat_targets.argmax(axis=-1)
    target_probs = flat_probs[np.arange(len(flat_probs)), target_idx]
    accuracy = float((predicted_idx == target_idx).mean())
    entropy = -np.sum(flat_probs * np.log(np.clip(flat_probs, 1e-12, 1.0)), axis=-1)
    confusion = np.zeros((action_count(), action_count()), dtype=np.int64)
    for true_idx, pred_idx in zip(target_idx, predicted_idx):
        confusion[int(true_idx), int(pred_idx)] += 1
    return {
        "valid_steps": int(len(flat_probs)),
        "top1_accuracy": accuracy,
        "target_action_probability": float(target_probs.mean()),
        "entropy": float(entropy.mean()),
        "confusion_matrix": confusion.tolist(),
        "action_names": list(ACTION_NAMES),
    }


def merge_metric_sums(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return _empty_metrics()
    total_steps = sum(int(row["valid_steps"]) for row in rows)
    if total_steps < 1:
        return _empty_metrics()
    confusion = np.zeros((action_count(), action_count()), dtype=np.int64)
    weighted = {"top1_accuracy": 0.0, "target_action_probability": 0.0, "entropy": 0.0}
    for row in rows:
        steps = int(row["valid_steps"])
        if steps < 1:
            continue
        for key in weighted:
            value = row.get(key)
            if value is not None and math.isfinite(float(value)):
                weighted[key] += float(value) * steps
        confusion += np.asarray(row["confusion_matrix"], dtype=np.int64)
    return {
        "valid_steps": int(total_steps),
        "top1_accuracy": weighted["top1_accuracy"] / total_steps,
        "target_action_probability": weighted["target_action_probability"] / total_steps,
        "entropy": weighted["entropy"] / total_steps,
        "confusion_matrix": confusion.tolist(),
        "action_names": list(ACTION_NAMES),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "valid_steps": 0,
        "top1_accuracy": None,
        "target_action_probability": None,
        "entropy": None,
        "confusion_matrix": np.zeros((action_count(), action_count()), dtype=np.int64).tolist(),
        "action_names": list(ACTION_NAMES),
    }
