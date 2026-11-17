"""Fix 3 score-unit transforms and complete, physical-time return labels."""

from __future__ import annotations

import torch

from src.continual import picar_math as pm
from .config import CONDITIONS, power


def condition_power(condition):
    return power(condition)


def code_scale(condition):
    """Raw success-bonus ratio, not a fitted reward normalization."""
    return 1. if condition_power(condition) == 1 else 10.


def score_transform(value, condition):
    """F_1(x)=x, F_2(x)=x|x|, including negative terminal bonuses."""
    return value if condition_power(condition) == 1 else value * value.abs()


def episode_labels(elapsed, scores, bonuses, terminated, condition, edges,
                   gamma=.99, nominal_period=.5):
    """Return six detached [T] tensors for one or more complete episodes.

    Inputs are floating [T] tensors on one device, except boolean terminated.
    A terminal must carry exactly +10/-5/-1; all other bonuses must be zero.
    Each episode is at most 20 physical seconds. The caller owns terminal-event
    precedence and must not label an interrupted prefix as a task timeout.
    Edges are the fixed 162 equally spaced edges for 161 coded bins [-6,12].
    Output floats accumulate in elapsed's dtype (at least float32); bins are
    int64. Scores are Qwen rewards, not likelihood scores or NLL gradients.
    """
    exponent = condition_power(condition)
    # Reuse the unchanged score, duration and terminal-validation contract.
    linear = pm.dense_visual_rewards(elapsed, scores, bonuses, terminated)
    if (not isinstance(edges, torch.Tensor) or not edges.is_floating_point()
            or edges.ndim != 1 or edges.numel() != 162
            or edges.device != elapsed.device or not bool(torch.isfinite(edges).all())):
        raise ValueError("edges must be 162 finite floating edges on the elapsed device")
    expected = torch.linspace(-6., 12., 162, device=edges.device, dtype=edges.dtype)
    if edges[0] != -6 or edges[-1] != 12 or not torch.allclose(edges, expected, rtol=1e-6, atol=1e-6):
        raise ValueError("161 coded bins require fixed equally spaced support [-6,12]")
    discounts = pm.time_discounts(elapsed, gamma, nominal_period)
    if exponent == 1:
        raw = linear
    else:
        dt, q, b = (v.detach().to(linear.dtype) for v in (elapsed, scores, bonuses))
        raw = dt * (.01 * score_transform(q, condition) - .04) + score_transform(b, condition)
    raw_returns = pm.discounted_returns(raw, discounts, terminated)
    lower, upper = (-6., 12.) if exponent == 1 else (-26., 120.)
    if not bool(torch.isfinite(raw_returns).all()) or bool(((raw_returns < lower) | (raw_returns > upper)).any()):
        raise ValueError("raw returns outside the declared finite support; no clipping")
    scale = code_scale(condition)
    coded_returns = raw_returns / scale
    return {"raw_stage_rewards": raw, "coded_stage_rewards": raw / scale,
            "raw_returns": raw_returns, "coded_returns": coded_returns,
            "bins": pm.bin_returns(coded_returns, edges), "discounts": discounts}


def reward_arithmetic_audit(temperature=2.8375266913896):
    """CPU-only scripted audit, with raw/coded returns and bin log-weight gaps.

    Baseline zero is common to the illustrative paired approach trajectories;
    these deterministic sequences are diagnostics, not learning observations.
    """
    if not 0 < float(temperature) < float("inf"):
        raise ValueError("temperature must be finite and positive")
    edges = torch.linspace(-6., 12., 162, dtype=torch.float64)
    scenarios = {"immediate_success": ([0.], 10.),
                 "delayed_success": ([10.] * 40, 10.),
                 "loiter_timeout": ([10.] * 40, -1.),
                 "disappearance": ([10.] * 10 + [0.] * 30, -1.),
                 "collision": ([10.] * 40, -5.),
                 "timeout": ([0.] * 40, -1.),
                 "approach_low": ([1.] * 40, 10.),
                 "approach_high": ([1.5] * 40, 10.)}
    result = {}
    for condition in CONDITIONS:
        rows = {}
        for name, (q, bonus) in scenarios.items():
            dt = torch.full((len(q),), .5, dtype=torch.float64)
            b, terminal = torch.zeros_like(dt), torch.zeros(len(q), dtype=torch.bool)
            b[-1], terminal[-1] = bonus, True
            labels = episode_labels(dt, torch.tensor(q, dtype=dt.dtype), b, terminal, condition, edges)
            index = int(labels["bins"][0])
            rows[name] = {key: float(labels[key][0]) for key in ("raw_returns", "coded_returns")}
            rows[name].update(bin=index, bin_value=float(pm.return_bin_values(edges)[index]))
        low, high = rows["approach_low"], rows["approach_high"]
        result[condition] = {"scenarios": rows,
            "timely_success_beats_loitering": rows["immediate_success"]["raw_returns"] > rows["loiter_timeout"]["raw_returns"],
            "approach_bin_tie": low["bin"] == high["bin"],
            "approach_log_weight_difference": (high["coded_returns"] - low["coded_returns"]) / temperature,
            "approach_binned_log_weight_difference": (high["bin_value"] - low["bin_value"]) / temperature}
    return result
