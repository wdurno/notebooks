"""Artifact-only design-oracle analysis for Plan 4."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import DataConfig, ScheduleConfig
from .schedules import ResolvedSchedule, resolve_schedule


PLAN4_PHASE2_ANALYSIS_SCHEMA_VERSION = 5
SOURCE_PATTERN = "mnist_lfu_plan2-low-data_m008_fixed-ewc-pi005__replica-*__*"
SOURCE_NAME = re.compile(r"__replica-(\d{4})__")
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 4_202
PI_MIN = 0.05
PI_MAX = 0.95
PI_SIGNAL_THRESHOLD = 0.07
MIN_SIGNAL_TRANSITIONS = 10
MIN_PI_RANGE = 0.05
MIN_RISK_REDUCTION = 0.05
SAMPLES_PER_STEP = 8
INITIAL_EFFECTIVE_SIZE = 30_000


class Plan4AnalysisError(RuntimeError):
    """Raised when a Phase 2 dependency or scientific contract is invalid."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Plan4AnalysisError(f"could not read {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _schedule_data(schedule: ScheduleConfig) -> DataConfig:
    return DataConfig(
        num_p_steps=100,
        samples_per_step=SAMPLES_PER_STEP,
        non_nine_sampling="empirical",
        initialization_size=INITIAL_EFFECTIVE_SIZE,
        online_pool_size=12_000,
        reference_pool_size=12_000,
        evaluation_size=10_000,
        schedule=schedule,
    )


def phase2_schedules() -> dict[str, ResolvedSchedule]:
    values = {
        "linear": ScheduleConfig("linear", 0.0, 0.2, None, None),
        **{
            f"logistic-k{kappa}": ScheduleConfig(
                "normalized_logistic", 0.0, 0.2, 0.5, float(kappa)
            )
            for kappa in (8, 16, 32, 64, 128, 256)
        },
    }
    return {
        name: resolve_schedule(_schedule_data(schedule))
        for name, schedule in values.items()
    }


def _validate_source_config(config: Mapping[str, Any], path: Path) -> None:
    data = config.get("data", {})
    controller = config.get("controller", {})
    estimator = config.get("estimator", {})
    expected = {
        "num_p_steps": 100,
        "samples_per_step": SAMPLES_PER_STEP,
        "initialization_size": INITIAL_EFFECTIVE_SIZE,
    }
    if any(data.get(key) != value for key, value in expected.items()):
        raise Plan4AnalysisError(f"invalid Phase 2 source data contract: {path}")
    if data.get("schedule") is not None:
        raise Plan4AnalysisError(f"Phase 2 source must use the legacy linear path: {path}")
    if (
        controller.get("policy") != "fixed_unified"
        or not math.isclose(float(controller.get("fixed_pi", -1.0)), 0.05)
        or not math.isclose(float(controller.get("trend_half_life_p", -1.0)), 0.2)
    ):
        raise Plan4AnalysisError(f"invalid Phase 2 source controller: {path}")
    if (
        estimator.get("method") != "ema"
        or estimator.get("representation") != "low_rank_diagonal"
        or estimator.get("low_rank") != 8
    ):
        raise Plan4AnalysisError(f"invalid Phase 2 source Fisher estimator: {path}")


def _load_sources(repo_root: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    run_root = repo_root / "cache" / "mnist_experiment" / "phase8_runs"
    paths = sorted(path for path in run_root.glob(SOURCE_PATTERN) if path.is_dir())
    if len(paths) != 10:
        raise Plan4AnalysisError(
            f"Phase 2 requires exactly 10 independent fixed-.05 sources, found {len(paths)}"
        )

    sources: list[dict[str, Any]] = []
    source_p: np.ndarray | None = None
    seen_replicas: set[int] = set()
    for path in paths:
        match = SOURCE_NAME.search(path.name)
        if match is None:
            raise Plan4AnalysisError(f"could not parse replica from {path.name}")
        replica = int(match.group(1))
        if replica in seen_replicas:
            raise Plan4AnalysisError(f"duplicate Phase 2 outer replica {replica}")
        seen_replicas.add(replica)
        if not (path / "COMPLETED").is_file():
            raise Plan4AnalysisError(f"incomplete Phase 2 source: {path}")

        config_path = path / "config.json"
        metrics_path = path / "phase8_metrics.json"
        reference_path = path / "phase8_reference_optimum.pt"
        config = _read_json(config_path)
        metrics = _read_json(metrics_path)
        _validate_source_config(config, path)
        if (
            metrics.get("phase8_metric_schema_version") != 8
            or metrics.get("policy") != "fixed_unified"
            or metrics.get("fisher_update_method") != "ema"
            or metrics.get("fisher_inverse_used") is not False
        ):
            raise Plan4AnalysisError(f"invalid Phase 2 source metrics: {path}")
        rows = metrics.get("condition_steps")
        if not isinstance(rows, list) or len(rows) != 100:
            raise Plan4AnalysisError(f"invalid Phase 2 source trajectory: {path}")
        p_values = np.asarray([float(row["p"]) for row in rows], dtype=np.float64)
        if source_p is None:
            source_p = p_values
        elif not np.array_equal(source_p, p_values):
            raise Plan4AnalysisError("Phase 2 source p grids differ")

        plugin_numerator = []
        plugin_denominator = []
        oracle_numerator = []
        oracle_denominator = []
        for row in rows:
            state = row.get("controller_state_pre")
            if not isinstance(state, Mapping):
                raise Plan4AnalysisError(f"source row lacks controller moments: {path}")
            plugin_numerator.append(float(state["residual_moment"]))
            plugin_denominator.append(float(state["scale_moment"]))
            oracle_numerator.append(float(state["oracle_residual_moment"]))
            oracle_denominator.append(float(state["oracle_scale_moment"]))

        try:
            reference = torch.load(reference_path, map_location="cpu", weights_only=True)
            optimum = reference["path"]
            repeated = optimum["replicate_parameters"].to(torch.float64).numpy()
            reference_p = np.asarray(optimum["p_values"], dtype=np.float64)
            diagnostics = optimum["displacement_diagnostics"]
        except (OSError, KeyError, TypeError, RuntimeError) as exc:
            raise Plan4AnalysisError(f"invalid reference path {reference_path}: {exc}") from exc
        if repeated.shape != (32, 100, 512):
            raise Plan4AnalysisError(f"unexpected repeated-fit shape at {reference_path}")
        if not np.array_equal(reference_p, p_values):
            raise Plan4AnalysisError("reference and trace p grids differ")
        interior_diagnostics = [
            row for row in diagnostics if float(row["p"]) <= 0.2 + 1e-12
        ]
        sources.append(
            {
                "replica_index": replica,
                "run_id": path.name,
                "path": path,
                "plugin_numerator": np.asarray(plugin_numerator),
                "plugin_denominator": np.asarray(plugin_denominator),
                "oracle_numerator": np.asarray(oracle_numerator),
                "oracle_denominator": np.asarray(oracle_denominator),
                "repeated_parameters": repeated,
                "reference_converged_points_to_p020": sum(
                    bool(row["converged"]) for row in interior_diagnostics
                ),
                "reference_points_to_p020": len(interior_diagnostics),
                "provenance": {
                    "replica_index": replica,
                    "run_id": path.name,
                    "config_sha256": _sha256(config_path),
                    "metrics_sha256": _sha256(metrics_path),
                    "reference_optimum_sha256": _sha256(reference_path),
                    "reference_path_hash": optimum["content_hash"],
                    "reference_fit_count": int(repeated.shape[0]),
                },
            }
        )
    assert source_p is not None
    if seen_replicas != set(range(1, 11)):
        raise Plan4AnalysisError("Phase 2 requires outer replicas 1 through 10")
    return sources, source_p


def interpolate_paths(
    source_p: np.ndarray,
    paths: np.ndarray,
    target_p: np.ndarray,
    *,
    stride: int = 1,
) -> np.ndarray:
    """Linearly interpolate complete fit paths without breaking endpoint pairing."""

    if source_p.ndim != 1 or paths.ndim != 3 or paths.shape[1] != source_p.size:
        raise ValueError("path interpolation shapes are incompatible")
    if target_p.ndim != 1 or np.any(np.diff(target_p) < 0.0):
        raise ValueError("target p values must be one-dimensional and nondecreasing")
    if stride < 1:
        raise ValueError("interpolation stride must be positive")
    indices = np.arange(0, source_p.size, stride, dtype=np.int64)
    if indices[-1] != source_p.size - 1:
        indices = np.append(indices, source_p.size - 1)
    x = source_p[indices]
    values = paths[:, indices, :]
    if target_p[0] < x[0] or target_p[-1] > x[-1]:
        raise ValueError("target p values require extrapolation")
    right = np.searchsorted(x, target_p, side="right")
    right = np.clip(right, 1, x.size - 1)
    left = right - 1
    weight = (target_p - x[left]) / (x[right] - x[left])
    return (
        values[:, left, :] * (1.0 - weight)[None, :, None]
        + values[:, right, :] * weight[None, :, None]
    )


def corrected_displacement_moments(
    interpolated_paths: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Estimate mean-path movement and its paired repeated-fit noise floor."""

    displacements = np.diff(interpolated_paths, axis=1)
    fit_count = displacements.shape[0]
    if fit_count < 2 or bootstrap_counts.shape[1] != fit_count:
        raise ValueError("invalid repeated-fit bootstrap dimensions")
    by_transition = np.swapaxes(displacements, 0, 1)
    gram = by_transition @ np.swapaxes(by_transition, 1, 2)
    diagonal = np.diagonal(gram, axis1=1, axis2=2)

    central_raw = gram.sum(axis=(1, 2)) / fit_count**2
    central_sum_norms = diagonal.sum(axis=1)
    central_noise = np.maximum(
        0.0,
        (central_sum_norms - fit_count * central_raw)
        / (fit_count * (fit_count - 1)),
    )
    central_corrected = np.maximum(0.0, central_raw - central_noise)

    counts = bootstrap_counts.astype(np.float64, copy=False)
    bootstrap_raw = np.einsum(
        "br,trs,bs->bt", counts, gram, counts, optimize=True
    ) / fit_count**2
    bootstrap_sum_norms = counts @ diagonal.T
    bootstrap_noise = np.maximum(
        0.0,
        (bootstrap_sum_norms - fit_count * bootstrap_raw)
        / (fit_count * (fit_count - 1)),
    )
    return {
        "central_raw": central_raw,
        "central_noise": central_noise,
        "central_corrected": central_corrected,
        "bootstrap_raw": bootstrap_raw,
        "bootstrap_corrected": np.maximum(0.0, bootstrap_raw - bootstrap_noise),
    }


def _fill_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    if numerator.shape != denominator.shape:
        raise ValueError("trace moment shapes differ")
    curves = np.atleast_2d(numerator)
    scales = np.atleast_2d(denominator)
    result = np.empty_like(curves, dtype=np.float64)
    positions = np.arange(curves.shape[1], dtype=np.float64)
    for index, (values, weights) in enumerate(zip(curves, scales, strict=True)):
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(valid):
            raise Plan4AnalysisError("trace estimator has no positive scale moments")
        ratio = values[valid] / weights[valid]
        result[index] = np.interp(positions, positions[valid], ratio)
    return result if numerator.ndim == 2 else result[0]


def interpolate_curves(
    source_p: np.ndarray,
    curves: np.ndarray,
    target_p: np.ndarray,
    *,
    stride: int = 1,
) -> np.ndarray:
    values = np.atleast_2d(curves)
    indices = np.arange(0, source_p.size, stride, dtype=np.int64)
    if indices[-1] != source_p.size - 1:
        indices = np.append(indices, source_p.size - 1)
    result = np.vstack(
        [np.interp(target_p, source_p[indices], row[indices]) for row in values]
    )
    return result if curves.ndim == 2 else result[0]


def _trace_curves(
    sources: Sequence[Mapping[str, Any]],
    source_p: np.ndarray,
    target_p: np.ndarray,
    outer_counts: np.ndarray,
    *,
    kind: str,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    numerators = np.stack([source[f"{kind}_numerator"] for source in sources])
    denominators = np.stack([source[f"{kind}_denominator"] for source in sources])
    central = _fill_ratio(numerators.mean(axis=0), denominators.mean(axis=0))
    sample_size = len(sources)
    bootstrap_numerator = outer_counts @ numerators / sample_size
    bootstrap_denominator = outer_counts @ denominators / sample_size
    bootstrap = _fill_ratio(bootstrap_numerator, bootstrap_denominator)
    return (
        interpolate_curves(source_p, central, target_p, stride=stride),
        interpolate_curves(source_p, bootstrap, target_p, stride=stride),
    )


def _event_mask(schedule: ResolvedSchedule) -> np.ndarray:
    speeds = np.diff(np.asarray(schedule.p_values, dtype=np.float64))
    high = np.flatnonzero(speeds >= 0.5 * speeds.max())
    start = max(0, int(high[0]) - 5)
    stop = min(speeds.size, int(high[-1]) + 6)
    mask = np.zeros(speeds.size, dtype=bool)
    mask[start:stop] = True
    return mask


def _adaptive_rollout(
    displacement_squared: np.ndarray,
    trace: np.ndarray,
    *,
    bounded: bool,
) -> dict[str, np.ndarray]:
    d2 = np.atleast_2d(displacement_squared)
    fisher_trace = np.atleast_2d(trace)
    if d2.shape != fisher_trace.shape:
        raise ValueError("risk coefficient shapes differ")
    q = np.full(d2.shape[0], 1.0 / INITIAL_EFFECTIVE_SIZE)
    q_rows = []
    raw_pi_rows = []
    pi_rows = []
    risk_rows = []
    for step in range(d2.shape[1]):
        q_rows.append(q.copy())
        old = d2[:, step] + fisher_trace[:, step] * q
        new = fisher_trace[:, step] / SAMPLES_PER_STEP
        raw_pi = old / np.maximum(old + new, 1e-15)
        pi = np.clip(raw_pi, PI_MIN, PI_MAX) if bounded else raw_pi
        risk = (1.0 - pi) ** 2 * old + pi**2 * new
        raw_pi_rows.append(raw_pi)
        pi_rows.append(pi)
        risk_rows.append(risk)
        q = (1.0 - pi) ** 2 * q + pi**2 / SAMPLES_PER_STEP
    return {
        "q": np.stack(q_rows, axis=1),
        "raw_pi": np.stack(raw_pi_rows, axis=1),
        "pi": np.stack(pi_rows, axis=1),
        "risk": np.stack(risk_rows, axis=1),
    }


def _fixed_risk_grid(
    displacement_squared: np.ndarray,
    trace: np.ndarray,
    event_mask: np.ndarray,
    *,
    bounded: bool,
) -> dict[str, np.ndarray]:
    d2 = np.atleast_2d(displacement_squared)
    fisher_trace = np.atleast_2d(trace)
    if bounded:
        grid = np.linspace(PI_MIN, PI_MAX, 361, dtype=np.float64)
    else:
        grid = np.unique(
            np.concatenate(
                (
                    np.asarray([0.0]),
                    np.geomspace(1e-8, 0.05, 600),
                    np.linspace(0.05, 1.0, 381),
                )
            )
        )
    q = np.full((d2.shape[0], grid.size), 1.0 / INITIAL_EFFECTIVE_SIZE)
    event_risk = np.zeros_like(q)
    full_risk = np.zeros_like(q)
    for step in range(d2.shape[1]):
        old = d2[:, step, None] + fisher_trace[:, step, None] * q
        new = fisher_trace[:, step, None] / SAMPLES_PER_STEP
        risk = (1.0 - grid[None, :]) ** 2 * old + grid[None, :] ** 2 * new
        full_risk += risk
        if event_mask[step]:
            event_risk += risk
        q = (1.0 - grid[None, :]) ** 2 * q + grid[None, :] ** 2 / SAMPLES_PER_STEP
    event_index = np.argmin(event_risk, axis=1)
    full_index = np.argmin(full_risk, axis=1)
    rows = np.arange(d2.shape[0])
    return {
        "event_risk": event_risk[rows, event_index],
        "event_pi": grid[event_index],
        "full_risk": full_risk[rows, full_index],
        "full_pi": grid[full_index],
    }


def policy_opportunity(
    displacement_squared: np.ndarray,
    trace: np.ndarray,
    event_mask: np.ndarray,
    *,
    bounded: bool = True,
) -> dict[str, np.ndarray]:
    rollout = _adaptive_rollout(displacement_squared, trace, bounded=bounded)
    fixed = _fixed_risk_grid(
        displacement_squared, trace, event_mask, bounded=bounded
    )
    event_risk = rollout["risk"][:, event_mask].sum(axis=1)
    full_risk = rollout["risk"].sum(axis=1)
    event_absolute = fixed["event_risk"] - event_risk
    full_absolute = fixed["full_risk"] - full_risk
    event_pi = rollout["pi"][:, event_mask]
    return {
        **rollout,
        "signal_transition_count": (event_pi > PI_SIGNAL_THRESHOLD).sum(axis=1),
        "event_pi_range": np.ptp(event_pi, axis=1),
        "event_adaptive_risk": event_risk,
        "event_best_constant_risk": fixed["event_risk"],
        "event_best_constant_pi": fixed["event_pi"],
        "event_absolute_risk_reduction": event_absolute,
        "event_relative_risk_reduction": event_absolute
        / np.maximum(fixed["event_risk"], 1e-15),
        "full_adaptive_risk": full_risk,
        "full_best_constant_risk": fixed["full_risk"],
        "full_best_constant_pi": fixed["full_pi"],
        "full_absolute_risk_reduction": full_absolute,
        "full_relative_risk_reduction": full_absolute
        / np.maximum(fixed["full_risk"], 1e-15),
        "lower_bound_fraction": np.mean(rollout["pi"] <= PI_MIN + 1e-12, axis=1),
        "upper_bound_fraction": np.mean(rollout["pi"] >= PI_MAX - 1e-12, axis=1),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q975": float(np.quantile(array, 0.975)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _central_metrics(opportunity: Mapping[str, np.ndarray]) -> dict[str, float | int]:
    return {
        "signal_transition_count": int(opportunity["signal_transition_count"][0]),
        "event_pi_range": float(opportunity["event_pi_range"][0]),
        "event_adaptive_risk": float(opportunity["event_adaptive_risk"][0]),
        "event_best_constant_risk": float(
            opportunity["event_best_constant_risk"][0]
        ),
        "event_best_constant_pi": float(opportunity["event_best_constant_pi"][0]),
        "event_absolute_risk_reduction": float(
            opportunity["event_absolute_risk_reduction"][0]
        ),
        "event_relative_risk_reduction": float(
            opportunity["event_relative_risk_reduction"][0]
        ),
        "full_relative_risk_reduction": float(
            opportunity["full_relative_risk_reduction"][0]
        ),
        "full_best_constant_pi": float(opportunity["full_best_constant_pi"][0]),
        "lower_bound_fraction": float(opportunity["lower_bound_fraction"][0]),
        "upper_bound_fraction": float(opportunity["upper_bound_fraction"][0]),
    }


def _opportunity_summary(
    central: Mapping[str, np.ndarray],
    bootstrap: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    metrics = (
        "signal_transition_count",
        "event_pi_range",
        "event_relative_risk_reduction",
        "full_relative_risk_reduction",
        "event_best_constant_pi",
    )
    intervals = {metric: _distribution(bootstrap[metric]) for metric in metrics}
    robust_pass = (
        intervals["signal_transition_count"]["q025"] >= MIN_SIGNAL_TRANSITIONS
        and intervals["event_pi_range"]["q025"] >= MIN_PI_RANGE
        and intervals["event_relative_risk_reduction"]["q025"]
        >= MIN_RISK_REDUCTION
    )
    central_values = _central_metrics(central)
    central_pass = (
        central_values["signal_transition_count"] >= MIN_SIGNAL_TRANSITIONS
        and central_values["event_pi_range"] >= MIN_PI_RANGE
        and central_values["event_relative_risk_reduction"] >= MIN_RISK_REDUCTION
    )
    return {
        "central": central_values,
        "bootstrap": intervals,
        "central_pass": central_pass,
        "robust_pass": robust_pass,
    }


def _aggregate_displacements(
    source_values: Sequence[Mapping[str, np.ndarray]],
    outer_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    outer_size = len(source_values)
    central = {
        key: np.mean(np.stack([value[key] for value in source_values]), axis=0)
        for key in ("central_raw", "central_noise", "central_corrected")
    }
    bootstrap = {}
    for key in ("bootstrap_raw", "bootstrap_corrected"):
        values = np.stack([value[key] for value in source_values])
        bootstrap[key] = np.einsum(
            "bo,obt->bt", outer_counts, values, optimize=True
        ) / outer_size
    return {**central, **bootstrap}


def _trajectory_rows(
    schedule: ResolvedSchedule,
    event_mask: np.ndarray,
    displacement: Mapping[str, np.ndarray],
    coarse: Mapping[str, np.ndarray],
    oracle_trace: np.ndarray,
    plugin_trace: np.ndarray,
    opportunity: Mapping[str, np.ndarray],
    unbounded_opportunity: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    p_values = np.asarray(schedule.p_values)
    speeds = np.diff(p_values)
    rows = []
    for step in range(speeds.size):
        rows.append(
            {
                "transition": step,
                "p": float(p_values[step]),
                "next_p": float(p_values[step + 1]),
                "delta_p": float(speeds[step]),
                "in_event_window": bool(event_mask[step]),
                "displacement_squared_raw": float(displacement["central_raw"][step]),
                "displacement_noise_floor": float(
                    displacement["central_noise"][step]
                ),
                "displacement_squared_corrected": float(
                    displacement["central_corrected"][step]
                ),
                "coarse_displacement_squared_corrected": float(
                    coarse["central_corrected"][step]
                ),
                "oracle_trace_estimate": float(oracle_trace[step]),
                "plugin_trace_estimate": float(plugin_trace[step]),
                "bounded_oracle_pi": float(opportunity["pi"][0, step]),
                "bounded_state_raw_oracle_pi": float(
                    opportunity["raw_pi"][0, step]
                ),
                "unbounded_raw_oracle_pi": float(
                    unbounded_opportunity["pi"][0, step]
                ),
                "bounded_oracle_q": float(opportunity["q"][0, step]),
                "bounded_oracle_risk": float(opportunity["risk"][0, step]),
            }
        )
    return rows


def _analyze_schedule(
    name: str,
    schedule: ResolvedSchedule,
    sources: Sequence[Mapping[str, Any]],
    source_p: np.ndarray,
    outer_counts: np.ndarray,
    inner_counts: Sequence[np.ndarray],
) -> dict[str, Any]:
    target_p = np.asarray(schedule.p_values, dtype=np.float64)
    event = _event_mask(schedule)
    full_sources = []
    coarse_sources = []
    for source, counts in zip(sources, inner_counts, strict=True):
        repeated = source["repeated_parameters"]
        full_sources.append(
            corrected_displacement_moments(
                interpolate_paths(source_p, repeated, target_p), counts
            )
        )
        coarse_sources.append(
            corrected_displacement_moments(
                interpolate_paths(source_p, repeated, target_p, stride=2), counts
            )
        )
    full = _aggregate_displacements(full_sources, outer_counts)
    coarse = _aggregate_displacements(coarse_sources, outer_counts)
    oracle_trace, oracle_trace_bootstrap = _trace_curves(
        sources, source_p, target_p, outer_counts, kind="oracle"
    )
    plugin_trace, plugin_trace_bootstrap = _trace_curves(
        sources, source_p, target_p, outer_counts, kind="plugin"
    )
    oracle_trace_coarse, oracle_trace_coarse_bootstrap = _trace_curves(
        sources, source_p, target_p, outer_counts, kind="oracle", stride=2
    )

    scenarios: dict[str, dict[str, Any]] = {}
    scenario_inputs = {
        "corrected-oracle-full-grid": (
            full["central_corrected"],
            full["bootstrap_corrected"],
            oracle_trace[:-1],
            oracle_trace_bootstrap[:, :-1],
        ),
        "corrected-oracle-coarse-grid": (
            coarse["central_corrected"],
            coarse["bootstrap_corrected"],
            oracle_trace_coarse[:-1],
            oracle_trace_coarse_bootstrap[:, :-1],
        ),
        "corrected-plugin-full-grid": (
            full["central_corrected"],
            full["bootstrap_corrected"],
            plugin_trace[:-1],
            plugin_trace_bootstrap[:, :-1],
        ),
        "raw-oracle-full-grid": (
            full["central_raw"],
            full["bootstrap_raw"],
            oracle_trace[:-1],
            oracle_trace_bootstrap[:, :-1],
        ),
    }
    principal: dict[str, np.ndarray] | None = None
    for scenario_name, (central_d2, bootstrap_d2, central_trace, bootstrap_trace) in (
        scenario_inputs.items()
    ):
        central = policy_opportunity(
            central_d2[None, :], central_trace[None, :], event
        )
        bootstrap = policy_opportunity(bootstrap_d2, bootstrap_trace, event)
        scenarios[scenario_name] = _opportunity_summary(central, bootstrap)
        if scenario_name == "corrected-oracle-full-grid":
            principal = central
    assert principal is not None

    unbounded = policy_opportunity(
        full["central_raw"][None, :], oracle_trace[None, :-1], event, bounded=False
    )
    robust_scenarios = (
        "corrected-oracle-full-grid",
        "corrected-oracle-coarse-grid",
        "corrected-plugin-full-grid",
    )
    robust_pass = all(scenarios[key]["robust_pass"] for key in robust_scenarios)
    central_diagnostic = scenarios["raw-oracle-full-grid"]["central_pass"] or (
        _central_metrics(unbounded)["signal_transition_count"]
        >= MIN_SIGNAL_TRANSITIONS
        and _central_metrics(unbounded)["event_pi_range"] >= MIN_PI_RANGE
        and _central_metrics(unbounded)["event_relative_risk_reduction"]
        >= MIN_RISK_REDUCTION
    )
    corrected_norm = np.sqrt(np.maximum(full["central_corrected"], 0.0))
    smoothness = float(
        np.abs(np.diff(corrected_norm)).sum()
        / max(corrected_norm.sum(), 1e-15)
    )
    event_indices = np.flatnonzero(event)
    return {
        "name": name,
        "schedule": schedule.to_mapping(),
        "schedule_hash": schedule.content_hash,
        "event_window": {
            "first_transition": int(event_indices[0]),
            "last_transition": int(event_indices[-1]),
            "transition_count": int(event.sum()),
            "maximum_speed_transition": schedule.max_speed_transition - 1,
        },
        "numerical_plateau_transitions": int(
            np.sum(np.diff(np.asarray(schedule.p_values)) == 0.0)
        ),
        "interpolation": {
            "source_points": int(source_p.size),
            "full_grid_stride": 1,
            "coarse_grid_stride": 2,
            "corrected_zero_fraction": float(
                np.mean(full["central_corrected"] <= 0.0)
            ),
            "median_noise_fraction_of_raw": float(
                np.median(
                    full["central_noise"]
                    / np.maximum(full["central_raw"], 1e-15)
                )
            ),
            "median_coarse_relative_change": float(
                np.median(
                    np.abs(coarse["central_corrected"] - full["central_corrected"])
                    / np.maximum(full["central_corrected"], 1e-15)
                )
            ),
            "displacement_norm_total_variation_ratio": smoothness,
        },
        "scenarios": scenarios,
        "unbounded_raw_diagnostic": _central_metrics(unbounded),
        "robust_pass": robust_pass,
        "central_diagnostic_signal": bool(central_diagnostic),
        "trajectory": _trajectory_rows(
            schedule,
            event,
            full,
            coarse,
            oracle_trace,
            plugin_trace,
            principal,
            unbounded,
        ),
    }


def _fixed_bracket(schedule: Mapping[str, Any]) -> dict[str, Any]:
    values = np.asarray(
        [row["bounded_oracle_pi"] for row in schedule["trajectory"]][1:-1]
    )
    q90 = float(np.quantile(values, 0.9))
    rounded = math.floor(q90 / 0.05 + 0.5) * 0.05
    third = min(0.5, max(0.1, rounded))
    return {
        "oracle_q90": q90,
        "rounded_third_control": third,
        "controls": sorted({0.05, 0.1, third}),
        "rule": "interior bounded-oracle q90, nearest .05, clipped to [.10,.50]",
    }


def build_phase2_analysis(
    repo_root: str | Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    root = Path(repo_root)
    if bootstrap_replicates < 100:
        raise ValueError("Phase 2 requires at least 100 bootstrap replicates")
    sources, source_p = _load_sources(root)
    rng = np.random.default_rng(bootstrap_seed)
    outer_counts = rng.multinomial(
        len(sources), np.full(len(sources), 1.0 / len(sources)), bootstrap_replicates
    )
    inner_counts = [
        rng.multinomial(32, np.full(32, 1.0 / 32), bootstrap_replicates)
        for _ in sources
    ]
    results = {
        name: _analyze_schedule(
            name,
            schedule,
            sources,
            source_p,
            outer_counts,
            inner_counts,
        )
        for name, schedule in phase2_schedules().items()
    }
    candidates = [
        results[f"logistic-k{kappa}"]
        for kappa in (8, 16, 32, 64, 128, 256)
    ]
    passing = [result for result in candidates if result["robust_pass"]]
    diagnostic = [
        result for result in candidates if result["central_diagnostic_signal"]
    ]
    if passing:
        decision = "promote"
        selected = passing[0]
    elif diagnostic:
        decision = "diagnostic-only"
        selected = max(
            diagnostic,
            key=lambda result: result["scenarios"][
                "raw-oracle-full-grid"
            ]["central"]["event_relative_risk_reduction"],
        )
    else:
        decision = "stop"
        selected = max(
            candidates,
            key=lambda result: max(
                row["bounded_state_raw_oracle_pi"]
                for row in result["trajectory"]
                if row["in_event_window"]
            ),
        )

    provenance = [source["provenance"] for source in sources]
    return {
        "schema_version": PLAN4_PHASE2_ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "plan4_design_oracle_feasibility",
        "analysis_code_sha256": _sha256(Path(__file__)),
        "decision": decision,
        "selected_schedule": selected["name"] if decision == "promote" else None,
        "diagnostic_schedule": (
            selected["name"] if decision == "diagnostic-only" else None
        ),
        "screening_reference_schedule": selected["name"],
        "fixed_policy_bracket": {
            **_fixed_bracket(selected),
            "activated": decision != "stop",
        },
        "gate": {
            "pi_signal_threshold": PI_SIGNAL_THRESHOLD,
            "minimum_signal_transitions": MIN_SIGNAL_TRANSITIONS,
            "minimum_pi_range": MIN_PI_RANGE,
            "minimum_relative_risk_reduction": MIN_RISK_REDUCTION,
            "robust_scenarios": [
                "corrected-oracle-full-grid",
                "corrected-oracle-coarse-grid",
                "corrected-plugin-full-grid",
            ],
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "outer_unit": "complete independent learner/reference replica",
            "inner_unit": "complete repeated-fit parameter trajectory",
            "interval": "joint percentile interval over nested resamples",
        },
        "trace_contract": {
            "estimand": "trace(I(theta*(p))^-1)",
            "primary": "pooled oracle-detrended residual moments / pooled scale moments",
            "sensitivity": "pooled plug-in residual moments / pooled scale moments",
            "source_policy": "m=8 fixed pi=.05 no-LFU EMA",
            "ratio_of_pooled_moments": True,
            "fisher_inverse_used": False,
            "left_boundary_handling": "nearest positive-scale trace estimate",
        },
        "reference_contract": {
            "outer_replicas": len(sources),
            "fits_per_outer_replica": 32,
            "paired_endpoint_dependence_retained": True,
            "noise_correction": "trace sample covariance of repeated-fit displacement divided by 32",
            "strictly_converged_points_to_p020": sum(
                source["reference_converged_points_to_p020"] for source in sources
            ),
            "total_reference_points_to_p020": sum(
                source["reference_points_to_p020"] for source in sources
            ),
        },
        "exclusions": {
            "adaptive_source_trajectories": "numerical duplicates; not independent replicas",
            "historical_hybrid_traces": (
                "controller batch size included replay observations and differs from "
                "the Plan 4 m=8 covariance contract"
            ),
            "m128_trace_family": "different learner/noise regime; not pooled into the estimand",
        },
        "assumptions": [
            "covariance calibration transfers the residual trace curve from the linear path by p",
            "the repeated-fit mean targets the local reference optimum",
            "piecewise-linear interpolation is adequate when the gate survives factor-two coarsening",
            "Phase 2 screens ideal controller opportunity and does not validate predictability",
        ],
        "source_provenance": provenance,
        "source_provenance_hash": _canonical_hash(provenance),
        "schedules": results,
    }


def write_phase2_analysis(
    repo_root: str | Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> Path:
    root = Path(repo_root)
    analysis = build_phase2_analysis(
        root,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    identity = {
        "schema_version": PLAN4_PHASE2_ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": analysis["analysis_kind"],
        "analysis_code_sha256": analysis["analysis_code_sha256"],
        "source_provenance_hash": analysis["source_provenance_hash"],
        "bootstrap": analysis["bootstrap"],
        "gate": analysis["gate"],
        "schedule_hashes": {
            name: value["schedule_hash"]
            for name, value in analysis["schedules"].items()
        },
    }
    digest = _canonical_hash(identity)
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "analysis"
        / f"phase2__{digest[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4AnalysisError(f"incomplete Phase 2 analysis: {destination}")
        if _read_json(destination / "manifest.json") != identity:
            raise Plan4AnalysisError("completed Phase 2 analysis identity differs")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=incomplete))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(analysis, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(identity, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
