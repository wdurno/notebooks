"""Artifact-only Plan 19 estimators. No Torch, model, storage, or simulator.

Uncertainty units are complete paired learning replicas. Replay draws and
evaluation frames never increase the replica count. Crossing analysis must
receive only the predeclared outward ramp/B hold (at most 55% of exposure).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math

import numpy as np


def _vector(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def _curve(samples, success):
    n, p = _vector(samples, "samples"), _vector(success, "success")
    if (len(n) < 2 or p.shape != n.shape or n[0] != 0 or not (np.diff(n) > 0).all()
            or ((p < 0) | (p > 1)).any()):
        raise ValueError("need increasing checkpoints from zero and matching success rates in [0,1]")
    return n, p


def budget_averaged_failure(samples, success) -> float:
    """Trapezoidal integral of failure / observed new-transition budget N."""
    n, p = _curve(samples, success)
    return float(np.trapezoid(1 - p, n) / n[-1])


def retention_summary(samples, success, *, initial_success: float) -> dict:
    n, p = _curve(samples, success)
    if not math.isfinite(initial_success) or not 0 <= initial_success <= 1:
        raise ValueError("initial_success must be in [0,1]")
    increase = initial_success - p
    return {"average_failure_increase": float(np.trapezoid(increase, n) / n[-1]),
            "endpoint_failure_increase": float(increase[-1])}


def paired_mean_ci(control: Mapping, fd: Mapping, *, draws: int = 10000,
                   seed: int = 190019, allow_incomplete: bool = False) -> dict:
    """Control-minus-FD mean and whole-pair percentile-bootstrap 95% interval.

    A single pair has no reported uncertainty interval. Unmatched replicas
    cause an error unless explicitly permitted; missing IDs are always shown.
    """
    left, right = set(control), set(fd)
    missing_control, missing_fd = sorted(right - left, key=str), sorted(left - right, key=str)
    if (missing_control or missing_fd) and not allow_incomplete:
        raise ValueError("incomplete replica pairs; explicitly opt in to reporting missing pairs")
    ids = sorted(left & right, key=str)
    values = _vector([float(control[i]) - float(fd[i]) for i in ids], "paired differences")
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    interval = None
    if len(values) >= 2:
        generator = np.random.default_rng(seed)
        means = values[generator.integers(len(values), size=(draws, len(values)))].mean(1)
        interval = [float(x) for x in np.quantile(means, [.025, .975])]
    return {"mean": float(values.mean()) if len(values) else None, "ci95": interval,
        "n_pairs": len(ids), "replica_ids": ids, "paired_differences": values.tolist(),
        "missing_control": missing_control, "missing_fd": missing_fd,
        "method": "paired_percentile_bootstrap", "draws": draws, "seed": seed,
        "interval_note": "unavailable with fewer than two independent pairs" if len(values) < 2
                         else "percentile interval; modest replica counts limit precision"}


@dataclass(frozen=True)
class Crossing:
    status: str
    measured_samples: float | None
    confirmation_samples: float | None
    lower_samples: float | None
    upper_samples: float | None
    interpolated_samples: float | None
    censored_at: float | None
    reason: str


def first_supported_crossing(samples, success, retention_increase, *, target=.7,
                             retention_ceiling=.05, max_samples=None) -> Crossing:
    """First of TWO consecutive measured checkpoints passing both gates.

    Measured attainment and an interpolated mean-curve crossing are distinct.
    Interpolation requires an observed capability bracket and acceptable
    retention at the interpolated point. A retention-only gate crossing is
    retained as measured attainment, not invented as equal-capability evidence.
    A last-checkpoint hit alone is right-censored, not a confirmed attainment.
    """
    n, p = _curve(samples, success)
    h = _vector(retention_increase, "retention increase")
    if h.shape != n.shape or not math.isfinite(target) or not 0 < target <= 1:
        raise ValueError("retention shape/target is invalid")
    if not math.isfinite(retention_ceiling) or retention_ceiling < 0:
        raise ValueError("retention ceiling must be finite and nonnegative")
    if max_samples is not None:
        if not math.isfinite(max_samples) or max_samples <= 0:
            raise ValueError("max_samples must be finite and positive")
        keep = n <= max_samples
        n, p, h = n[keep], p[keep], h[keep]
        if len(n) < 2:
            raise ValueError("crossing window must contain at least two measured checkpoints")
    meets = (p >= target) & (h <= retention_ceiling)
    for i in range(len(n) - 1):
        if not (meets[i] and meets[i + 1]):
            continue
        if i == 0:
            return Crossing("attained", 0., float(n[1]), 0., 0., 0., None,
                            "capability already attained at initialization")
        value = None
        reason = "measured attainment; no acceptable equal-capability interpolation bracket"
        if p[i - 1] < target <= p[i]:
            fraction = (target - p[i - 1]) / (p[i] - p[i - 1])
            retention = h[i - 1] + fraction * (h[i] - h[i - 1])
            if retention <= retention_ceiling:
                value = float(n[i - 1] + fraction * (n[i] - n[i - 1]))
                reason = "observed capability bracket; retention acceptable at interpolation"
        return Crossing("attained", float(n[i]), float(n[i + 1]), float(n[i - 1]),
                        float(n[i]), value, None, reason)
    return Crossing("right_censored", None, None, None, None, None, float(n[-1]),
                    "target/retention not confirmed at two consecutive measured checkpoints")


def efficiency_metrics(control_samples, fd_samples, control_seconds, fd_seconds) -> dict:
    """f=1-nFD/nC, c=TFD/TC, S=f/c; not a probability of success."""
    values = [float(x) for x in (control_samples, fd_samples, control_seconds, fd_seconds)]
    if not all(math.isfinite(x) for x in values) or min(values) < 0:
        raise ValueError("sample/cost values must be finite and nonnegative")
    nc, nf, tc, tf = values
    if nc <= 0 or min(tc, tf) <= 0:
        raise ValueError("control samples and both operational costs must be positive")
    f, c = 1 - nf / nc, tf / tc
    if not math.isfinite(c) or c <= 0:
        raise ValueError("efficiency ratio overflow or underflow")
    result = {"f": f, "c": c, "S": f / c}
    if not all(math.isfinite(v) for v in result.values()) or c <= 0:
        raise ValueError("efficiency ratio overflow or underflow")
    return result


@dataclass(frozen=True)
class LearningCurve:
    samples: Sequence[float]
    success: Sequence[float]
    retention_increase: Sequence[float]
    operational_seconds: Sequence[float]

    def arrays(self):
        n, p = _curve(self.samples, self.success)
        h = _vector(self.retention_increase, "retention increase")
        seconds = _vector(self.operational_seconds, "operational seconds")
        if (h.shape != n.shape or seconds.shape != n.shape or (seconds < 0).any()
                or (np.diff(seconds) < 0).any()):
            raise ValueError("retention/cumulative operational cost must align with measured checkpoints")
        return n, p, h, seconds


def sample_efficiency(control: LearningCurve, fd: LearningCurve, *, target=.7,
                      retention_ceiling=.05, max_samples=None) -> dict:
    curves = [curve.arrays() for curve in (control, fd)]
    crossings = [first_supported_crossing(n, p, h, target=target,
        retention_ceiling=retention_ceiling, max_samples=max_samples) for n, p, h, _ in curves]
    result = {"control": asdict(crossings[0]), "fd": asdict(crossings[1]),
        "status": "unavailable", "f": None, "c": None, "S": None,
        "cost_definition": "operational inference + learning + reward + Fisher + sleep"}
    if any(c.interpolated_samples is None for c in crossings):
        result["reason"] = "censored or unsupported mean-curve interpolation"
        return result
    nc, nf = (c.interpolated_samples for c in crossings)
    if nc == 0:
        result["reason"] = "control already attained target at zero new samples"
        return result
    costs = [float(np.interp(c.interpolated_samples, arrays[0], arrays[3]))
             for c, arrays in zip(crossings, curves)]
    if min(costs) <= 0:
        result["reason"] = "positive measured operational costs unavailable"
        return result
    result.update(efficiency_metrics(nc, nf, *costs), status="supported", reason=None,
                  control_seconds=costs[0], fd_seconds=costs[1])
    return result


def bootstrap_sample_efficiency(pairs: Sequence[tuple[LearningCurve, LearningCurve]], *,
                                draws=2000, seed=190020, target=.7,
                                retention_ceiling=.05, max_samples=None) -> dict:
    """Resample entire pairs; explicitly retain all failed-crossing draws.

    Conditional quantiles describe supported draws ONLY, never an unconditional
    95% interval when censoring exists. Input grids must match; callers must not
    silently replace missing checkpoints with fitted/extrapolated observations.
    """
    if not pairs or type(draws) is not int or draws < 1:
        raise ValueError("complete pairs and positive bootstrap draws are required")
    arrays = [[curve.arrays() for curve in pair] for pair in pairs]
    grid = arrays[0][0][0]
    if any(not np.array_equal(a[0], grid) for pair in arrays for a in pair):
        raise ValueError("mean-curve bootstrap requires a common measured checkpoint grid")
    data = np.asarray([[np.stack(a[1:]) for a in pair] for pair in arrays])
    options = dict(target=target, retention_ceiling=retention_ceiling, max_samples=max_samples)
    def evaluate(indices):
        means = data[indices].mean(0)
        return sample_efficiency(*(LearningCurve(grid, *a) for a in means), **options)
    point = evaluate(np.arange(len(pairs)))
    generator = np.random.default_rng(seed)
    supported, statuses = [], Counter()
    for _ in range(draws):
        result = evaluate(generator.integers(len(pairs), size=len(pairs)))
        statuses[result["status"] if result["status"] == "supported" else result["reason"]] += 1
        if result["status"] == "supported":
            supported.append([result[name] for name in ("f", "c", "S")])
    intervals = None
    if len(pairs) >= 2 and supported:
        bounds = np.quantile(np.asarray(supported), [.025, .975], axis=0)
        intervals = {key: bounds[:, i].tolist() for i, key in enumerate(("f", "c", "S"))}
    attained = {name: sum(first_supported_crossing(*arrays[i][j][:3], target=target,
        retention_ceiling=retention_ceiling, max_samples=max_samples).status == "attained"
        for i in range(len(pairs))) / len(pairs) for j, name in enumerate(("control", "fd"))}
    return {"point": point, "n_pairs": len(pairs), "draws": draws, "seed": seed,
        "supported_draws": len(supported), "supported_fraction": len(supported) / draws,
        "censored_or_unsupported_draws": draws - len(supported), "draw_status_counts": dict(statuses),
        "attainment_fraction": attained, "conditional_ci95": intervals,
        "interval_note": "conditional on supported draws; censoring retained in counts; "
                         "no interval with fewer than two independent pairs"}
