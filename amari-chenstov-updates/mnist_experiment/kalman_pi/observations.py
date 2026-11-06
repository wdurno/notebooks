"""Observed-label scores for discrepancy measurements, not Fisher observations."""

from __future__ import annotations

import torch
from torch.func import grad, vmap

from mnist_experiment.diagonal_lfu.curvature import logits


def observed_scores(model, layout, theta, inputs, labels, *, chunk_size=8, check=lambda: None):
    """Return per-example positive scores; loss gradients have the opposite sign."""
    theta, inputs = theta.double(), inputs.double()

    def log_probability(parameter, image, label):
        logp = logits(model, layout, parameter, image.unsqueeze(0))[0].log_softmax(-1)
        return logp.gather(0, label.reshape(1)).squeeze(0)

    pieces = []
    for offset in range(0, len(labels), chunk_size):
        check()
        values = vmap(grad(log_probability), in_dims=(None, 0, 0))(
            theta, inputs[offset:offset + chunk_size], labels[offset:offset + chunk_size])
        pieces.append(values.detach())
    result = torch.cat(pieces)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite observed-label scores")
    return result


def score_moments(scores):
    if scores.ndim != 2 or len(scores) < 1 or not torch.isfinite(scores).all():
        raise ValueError("finite per-observation scores required")
    mean = scores.double().mean(0)
    variance = scores.double().var(0, unbiased=True) if len(scores) > 1 else None
    return mean, variance


def combine_moments(count, mean, m2, values):
    """Parallel Welford update; calibration observations are counted once."""
    values = values.double()
    n = len(values)
    batch_mean = values.mean(0)
    batch_m2 = (values - batch_mean).square().sum(0)
    if count == 0:
        return n, batch_mean, batch_m2
    delta = batch_mean - mean
    total = count + n
    return total, mean + delta * n / total, m2 + batch_m2 + delta.square() * count * n / total
