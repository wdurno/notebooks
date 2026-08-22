"""Score-only Fisher-risk diagnostic replay for Plan 4."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from mnist_experiment.run_experiment import (
    _calculate_online_statistics,
    _clone_at_vector,
    _materialize_batch,
)
from src.config import (
    ControllerConfig,
    ExperimentConfig,
    ScheduleConfig,
    load_config,
)
from src.controller import (
    ControllerState,
    OracleControllerInput,
    decide_controller,
    effective_size_update,
    fixed_batch_optimal_pi,
    half_life_gain,
)
from src.initialization import load_replica_bundle_for_config
from src.lanczos_wrapper import approximate_low_rank_diagonal
from src.mnist_data import (
    dataset_targets,
    generate_mixture_stream,
    load_mnist_datasets,
)
from src.mnist_model import configure_torch_runtime, resolve_device
from src.plan4_analysis import (
    INITIAL_EFFECTIVE_SIZE,
    MIN_PI_RANGE,
    MIN_RISK_REDUCTION,
    MIN_SIGNAL_TRANSITIONS,
    PI_MAX,
    PI_MIN,
    PI_SIGNAL_THRESHOLD,
    SAMPLES_PER_STEP,
    _event_mask,
    phase2_schedules,
    policy_opportunity,
)
from src.representations import DenseFisher, FisherRepresentation
from src.seeding import derive_component_seed, derive_seed_map


PLAN4_FISHER_ANALYSIS_SCHEMA_VERSION = 1
SOURCE_PATTERN = "mnist_lfu_plan2-low-data_m008_fixed-ewc-pi005__replica-*__*"
PRIMARY_HALF_LIFE = 0.05
SENSITIVITY_HALF_LIFE = 0.10
HALF_LIVES = (PRIMARY_HALF_LIFE, SENSITIVITY_HALF_LIFE)
SCHEDULE_NAMES = (
    "linear",
    "logistic-k32",
    "logistic-k64",
    "logistic-k128",
    "logistic-k256",
)


class Plan4FisherAnalysisError(RuntimeError):
    """Raised when a Fisher diagnostic dependency or invariant fails."""


@dataclasses.dataclass(frozen=True)
class _DirectEmaUpdate:
    blend_gain: float


class _DirectEmaTracker:
    """Rank-8-plus-diagonal EMA for an already-PSD score Fisher stream."""

    def __init__(self, initial_fisher: Tensor, *, rank: int) -> None:
        self.initial_fisher = initial_fisher
        self.rank = rank
        self.previous: FisherRepresentation | None = None
        self.next_step = 0

    def update(
        self,
        step: int,
        observed_fisher: Tensor,
        *,
        lanczos_seed: int,
        blend_gain: float,
    ) -> _DirectEmaUpdate:
        if step != self.next_step:
            raise ValueError(
                f"expected Fisher step {self.next_step}, received {step}"
            )
        if not 0.0 <= blend_gain <= 1.0:
            raise ValueError("blend_gain must be in [0, 1]")
        if step == 0:
            candidate = self.initial_fisher
        else:
            assert self.previous is not None
            candidate = (
                (1.0 - blend_gain) * self.previous.to_dense()
                + blend_gain * observed_fisher
            )
        candidate = (candidate + candidate.mT) / 2
        approximation = approximate_low_rank_diagonal(
            lambda vector: candidate @ vector,
            torch.diagonal(candidate),
            rank=self.rank,
            seed=lanczos_seed,
        )
        self.previous = approximation.representation
        self.next_step += 1
        return _DirectEmaUpdate(blend_gain=blend_gain)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Plan4FisherAnalysisError(f"could not read {path}: {exc}") from exc


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
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _interpolate_parameter_path(
    source_p: Tensor,
    source_parameters: Tensor,
    target_p: Tensor,
) -> Tensor:
    if source_p.ndim != 1 or target_p.ndim != 1:
        raise ValueError("p grids must be vectors")
    if (
        source_parameters.ndim != 2
        or source_parameters.shape[0] != source_p.numel()
    ):
        raise ValueError("parameter path does not align with its p grid")
    if float(target_p[0]) < float(source_p[0]) or float(target_p[-1]) > float(
        source_p[-1]
    ):
        raise ValueError("target path requires extrapolation")
    right = torch.searchsorted(source_p, target_p, right=True).clamp(
        1, source_p.numel() - 1
    )
    left = right - 1
    weight = (target_p - source_p[left]) / (source_p[right] - source_p[left])
    return (
        source_parameters[left] * (1.0 - weight).unsqueeze(1)
        + source_parameters[right] * weight.unsqueeze(1)
    )


def _predictable_fisher(
    tracker: _DirectEmaTracker,
    initial_fisher: Tensor,
) -> FisherRepresentation:
    return (
        DenseFisher(initial_fisher)
        if tracker.previous is None
        else tracker.previous
    )


def _quadratic(fisher: FisherRepresentation, vector: Tensor) -> float:
    metric_vector = vector.to(device=fisher.device, dtype=fisher.dtype)
    value = float(fisher.quadratic(metric_vector))
    if not math.isfinite(value) or value < -1e-10:
        raise Plan4FisherAnalysisError("invalid Fisher quadratic")
    return max(value, 0.0)


def expected_controller_update(
    state: ControllerState,
    applied_pi: float,
    displacement: Tensor,
    fisher: FisherRepresentation,
    covariance_dimension: float,
    *,
    batch_size: int,
    delta_p: float,
    half_life_p: float,
) -> tuple[ControllerState, dict[str, float]]:
    """Advance expected plug-in moments without drawing synthetic parameters."""

    if covariance_dimension < 0.0 or not math.isfinite(covariance_dimension):
        raise ValueError("covariance_dimension must be finite and nonnegative")
    gain = half_life_gain(delta_p, half_life_p)
    trend_error = displacement - state.trend
    trend_error_energy = _quadratic(fisher, trend_error)
    scale = applied_pi**2 * (state.q + 1.0 / batch_size)
    residual_energy = applied_pi**2 * (
        trend_error_energy
        + (state.q + 1.0 / batch_size) * covariance_dimension
    )
    oracle_residual_energy = scale * covariance_dimension
    trend = (1.0 - gain) * state.trend + gain * displacement
    next_state = ControllerState(
        trend=trend,
        q=effective_size_update(state.q, applied_pi, batch_size),
        residual_moment=(
            (1.0 - gain) * state.residual_moment + gain * residual_energy
        ),
        scale_moment=(1.0 - gain) * state.scale_moment + gain * scale,
        environment_distance=state.environment_distance + delta_p,
        previous_pi=applied_pi,
        accepted_steps=state.accepted_steps + 1,
        oracle_residual_moment=(
            (1.0 - gain) * state.oracle_residual_moment
            + gain * oracle_residual_energy
        ),
        oracle_scale_moment=(
            (1.0 - gain) * state.oracle_scale_moment + gain * scale
        ),
    )
    return next_state, {
        "gain": gain,
        "scale_observation": scale,
        "expected_residual_energy": residual_energy,
        "expected_oracle_residual_energy": oracle_residual_energy,
        "trend_error_energy": trend_error_energy,
    }


def _controller_config(source: ExperimentConfig, half_life: float) -> ControllerConfig:
    return dataclasses.replace(
        source.controller,
        policy="optimal_plugin",
        fixed_pi=0.5,
        pi_min=PI_MIN,
        pi_max=PI_MAX,
        trend_half_life_p=half_life,
        oracle_mode="diagnostic",
        reference_optimum_artifact=None,
        risk_metric="fisher",
    )


def _source_paths(repo_root: Path, source_count: int) -> list[Path]:
    paths = sorted(
        path
        for path in (
            repo_root / "cache" / "mnist_experiment" / "phase8_runs"
        ).glob(SOURCE_PATTERN)
        if path.is_dir() and (path / "COMPLETED").is_file()
    )
    if not 1 <= source_count <= len(paths):
        raise Plan4FisherAnalysisError(
            f"requested {source_count} sources but found {len(paths)}"
        )
    return paths[:source_count]


def _source_inputs(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    metrics_path = path / "phase8_metrics.json"
    reference_path = path / "phase8_reference_optimum.pt"
    checkpoint_path = path / "phase8_checkpoints.pt"
    controller_path = path / "phase8_controller_states.pt"
    config = load_config(config_path)
    metrics = _read_json(metrics_path)
    method = "low_rank_diagonal_r8"
    if (
        config.data.samples_per_step != SAMPLES_PER_STEP
        or config.data.initialization_size != INITIAL_EFFECTIVE_SIZE
        or config.controller.policy != "fixed_unified"
        or not math.isclose(config.controller.fixed_pi, PI_MIN)
        or config.estimator.method != "ema"
        or config.estimator.low_rank != 8
    ):
        raise Plan4FisherAnalysisError(f"invalid source treatment: {path}")
    rows = [
        row for row in metrics["condition_steps"] if row["method"] == method
    ]
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    checkpoints = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )["conditions"][method]
    controller = torch.load(
        controller_path, map_location="cpu", weights_only=False
    )["conditions"][method]
    residuals = []
    scales = []
    valid_p = []
    euclidean_trace_p = []
    euclidean_trace = []
    for row in rows:
        state = row.get("controller_state_pre")
        if state is None:
            continue
        value = state.get("oracle_trace_estimate")
        if value is not None and math.isfinite(float(value)) and float(value) >= 0.0:
            euclidean_trace_p.append(float(row["p"]))
            euclidean_trace.append(float(value))
    for row in rows[:-1]:
        step = str(row["step"])
        residual = controller[step]["oracle_residual"]
        acceptance = row["acceptance"]
        if residual is None or acceptance is None:
            continue
        scale = float(acceptance["scale_observation"])
        if scale <= 0.0:
            continue
        valid_p.append(float(row["p"]))
        residuals.append(residual.to(torch.float64))
        scales.append(scale)
    if not residuals:
        raise Plan4FisherAnalysisError(f"source has no residual vectors: {path}")
    if not euclidean_trace:
        raise Plan4FisherAnalysisError(f"source has no Euclidean trace estimates: {path}")
    optimum = reference["path"]
    return {
        "path": path,
        "config": config,
        "source_p": torch.as_tensor(optimum["p_values"], dtype=torch.float64),
        "parameters": optimum["parameters"].to(torch.float64),
        "initial_fisher": checkpoints["0"]["reference"].to(torch.float64),
        "residual_p": torch.tensor(valid_p, dtype=torch.float64),
        "residuals": torch.stack(residuals),
        "residual_scales": torch.tensor(scales, dtype=torch.float64),
        "euclidean_trace_p": torch.tensor(euclidean_trace_p, dtype=torch.float64),
        "euclidean_trace": torch.tensor(euclidean_trace, dtype=torch.float64),
        "provenance": {
            "run_id": path.name,
            "config_sha256": _sha256(config_path),
            "metrics_sha256": _sha256(metrics_path),
            "reference_sha256": _sha256(reference_path),
            "checkpoints_sha256": _sha256(checkpoint_path),
            "controller_states_sha256": _sha256(controller_path),
        },
    }


def _nearest_residual(source: dict[str, Any], p_value: float) -> tuple[Tensor, float]:
    distances = (source["residual_p"] - p_value).abs()
    index = int(torch.argmin(distances))
    return source["residuals"][index], float(source["residual_scales"][index])


def _nearest_euclidean_trace(source: dict[str, Any], p_value: float) -> float:
    distances = (source["euclidean_trace_p"] - p_value).abs()
    return float(source["euclidean_trace"][int(torch.argmin(distances))])


def _schedule_stream(
    source: dict[str, Any],
    schedule_name: str,
    train_targets: Tensor,
):
    schedule = phase2_schedules()[schedule_name]
    # Source runs predate explicit schedules, so construct the accepted config
    # from the resolved family rather than relying on the historical field.
    if schedule.name == "linear":
        specification = ScheduleConfig("linear", 0.0, 0.2, None, None)
    else:
        specification = ScheduleConfig(
            "normalized_logistic",
            0.0,
            0.2,
            0.5,
            float(schedule_name.removeprefix("logistic-k")),
        )
    data = dataclasses.replace(source["config"].data, schedule=specification)
    stream = generate_mixture_stream(
        train_targets,
        source["loaded"].partitions,
        data,
        seed=derive_seed_map(source["config"].replica_seed)["online_stream"],
    )
    if stream.p_values != schedule.p_values:
        raise Plan4FisherAnalysisError("generated stream has the wrong schedule")
    return schedule, stream


def _new_tracker(source: dict[str, Any]) -> _DirectEmaTracker:
    return _DirectEmaTracker(source["initial_fisher"], rank=8)


def _fixed_policy_risk(
    signal: np.ndarray,
    covariance_dimension: np.ndarray,
    event: np.ndarray,
    pi: float,
) -> tuple[float, float]:
    q = 1.0 / INITIAL_EFFECTIVE_SIZE
    event_risk = 0.0
    full_risk = 0.0
    for step, (movement, dimension) in enumerate(
        zip(signal, covariance_dimension, strict=True)
    ):
        risk = (1.0 - pi) ** 2 * (movement + dimension * q)
        risk += pi**2 * dimension / SAMPLES_PER_STEP
        full_risk += risk
        if event[step]:
            event_risk += risk
        q = effective_size_update(q, pi, SAMPLES_PER_STEP)
    return event_risk, full_risk


def _operational_policy_risk(
    trajectory: list[dict[str, Any]],
    event: np.ndarray,
) -> tuple[float, float]:
    event_risk = 0.0
    full_risk = 0.0
    for step, row in enumerate(trajectory[:-1]):
        pi = float(row["operational_pi"])
        movement = float(row["current_displacement_energy"])
        dimension = float(row["instantaneous_covariance_dimension"])
        q = float(row["q"])
        risk = (1.0 - pi) ** 2 * (movement + dimension * q)
        risk += pi**2 * dimension / SAMPLES_PER_STEP
        full_risk += risk
        if event[step]:
            event_risk += risk
    return event_risk, full_risk


def _relative_reduction(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(reference, 1e-15)


def _event_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2 or float(np.ptp(first)) == 0.0 or float(np.ptp(second)) == 0.0:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if math.isfinite(value) else None


def _analyze_source_schedule(
    source: dict[str, Any],
    schedule_name: str,
    train_dataset,
    train_targets: Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    schedule, stream = _schedule_stream(source, schedule_name, train_targets)
    target_p = torch.tensor(schedule.p_values, dtype=torch.float64)
    parameters = _interpolate_parameter_path(
        source["source_p"], source["parameters"], target_p
    )
    displacements = parameters[1:] - parameters[:-1]
    zero_displacement = torch.zeros(parameters.shape[1], dtype=torch.float64)
    trackers = {half_life: _new_tracker(source) for half_life in HALF_LIVES}
    states = {
        half_life: ControllerState.initialize(
            parameters.shape[1], INITIAL_EFFECTIVE_SIZE, dtype=torch.float64
        )
        for half_life in HALF_LIVES
    }
    configs = {
        half_life: _controller_config(source["config"], half_life)
        for half_life in HALF_LIVES
    }
    rows = {half_life: [] for half_life in HALF_LIVES}
    started = time.perf_counter()
    for step, p_value in enumerate(schedule.p_values):
        model, layout = _clone_at_vector(
            source["loaded"].model,
            parameters[step],
            device=device,
            dtype=torch.float64,
        )
        inputs, targets = _materialize_batch(
            train_dataset,
            stream.observation_indices[step],
            device=device,
            dtype=torch.float64,
        )
        statistics = _calculate_online_statistics(
            model,
            layout,
            inputs,
            targets,
            zero_displacement.to(device=device),
            matrix_dtype=torch.float64,
            include_hvp=False,
            device=device,
        )
        displacement = (
            displacements[step]
            if step < displacements.shape[0]
            else zero_displacement
        )
        source_residual, source_scale = _nearest_residual(source, p_value)
        euclidean_trace = _nearest_euclidean_trace(source, p_value)
        contemporaneous_fisher = DenseFisher(statistics.estimate.fisher)
        for half_life in HALF_LIVES:
            tracker = trackers[half_life]
            state = states[half_life]
            fisher = _predictable_fisher(tracker, source["initial_fisher"])
            covariance_dimension = _quadratic(fisher, source_residual) / source_scale
            decision = decide_controller(
                state,
                configs[half_life],
                batch_size=SAMPLES_PER_STEP,
                oracle=OracleControllerInput(displacement),
                fisher=fisher,
            )
            current_signal = _quadratic(fisher, displacement)
            euclidean_signal = float(displacement @ displacement)
            contemporaneous_signal = _quadratic(
                contemporaneous_fisher, displacement
            )
            contemporaneous_dimension = (
                _quadratic(contemporaneous_fisher, source_residual)
                / source_scale
            )
            contemporaneous_pi = fixed_batch_optimal_pi(
                contemporaneous_signal,
                contemporaneous_dimension,
                state.q,
                SAMPLES_PER_STEP,
                epsilon=float(configs[half_life].trace_epsilon),
            )
            euclidean_oracle_pi = fixed_batch_optimal_pi(
                euclidean_signal,
                euclidean_trace,
                state.q,
                SAMPLES_PER_STEP,
                epsilon=float(configs[half_life].trace_epsilon),
            )
            update = tracker.update(
                step,
                statistics.estimate.fisher,
                lanczos_seed=derive_component_seed(
                    source["config"].replica_seed,
                    f"plan4_fisher:{schedule_name}:step={step}",
                ),
                blend_gain=decision.applied_pi,
            )
            row = {
                "step": step,
                "p": float(p_value),
                "delta_p": (
                    0.0
                    if step + 1 == len(schedule.p_values)
                    else float(schedule.p_values[step + 1] - p_value)
                ),
                "risk_metric": "fisher",
                "half_life_p": half_life,
                "cold_start_active": decision.cold_start_active,
                "operational_raw_pi": decision.raw_pi,
                "operational_pi": decision.applied_pi,
                "ungated_plugin_pi": decision.plugin_pi,
                "predictable_oracle_pi": decision.oracle_pi,
                "contemporaneous_pi": contemporaneous_pi,
                "predictable_signal_energy": decision.signal_squared,
                "current_displacement_energy": current_signal,
                "euclidean_displacement_energy": euclidean_signal,
                "euclidean_uncertainty_scale": euclidean_trace,
                "euclidean_contemporaneous_oracle_pi": euclidean_oracle_pi,
                "contemporaneous_displacement_energy": contemporaneous_signal,
                "contemporaneous_covariance_dimension": contemporaneous_dimension,
                "uncertainty_scale_estimate": decision.trace_estimate,
                "instantaneous_covariance_dimension": covariance_dimension,
                "old_covariance_risk": decision.old_covariance_trace,
                "new_covariance_risk": decision.new_covariance_trace,
                "q": state.q,
                "trend_norm": float(torch.linalg.vector_norm(state.trend)),
                "displacement_norm": float(torch.linalg.vector_norm(displacement)),
                "fisher_kind": (
                    "dense" if isinstance(fisher, DenseFisher) else "low_rank_diagonal"
                ),
                "fisher_rank": 0 if isinstance(fisher, DenseFisher) else fisher.rank,
                "fisher_blend_pi": update.blend_gain,
                "score_gradient_count": statistics.score_gradient_count,
                "hvp_count": statistics.hvp_count,
                "zero_information_fallback": decision.zero_information_fallback,
            }
            if step < displacements.shape[0]:
                next_state, moment = expected_controller_update(
                    state,
                    decision.applied_pi,
                    displacement,
                    fisher,
                    covariance_dimension,
                    batch_size=SAMPLES_PER_STEP,
                    delta_p=float(schedule.p_values[step + 1] - p_value),
                    half_life_p=half_life,
                )
                states[half_life] = next_state
                row.update(moment)
            rows[half_life].append(row)

    event = _event_mask(schedule)
    values: dict[str, Any] = {}
    for half_life in HALF_LIVES:
        trajectory = rows[half_life]
        signal = np.asarray(
            [row["current_displacement_energy"] for row in trajectory[:-1]],
            dtype=np.float64,
        )
        dimension = np.asarray(
            [row["instantaneous_covariance_dimension"] for row in trajectory[:-1]],
            dtype=np.float64,
        )
        opportunity = policy_opportunity(signal[None, :], dimension[None, :], event)
        euclidean_signal = np.asarray(
            [row["euclidean_displacement_energy"] for row in trajectory[:-1]],
            dtype=np.float64,
        )
        euclidean_trace = np.asarray(
            [row["euclidean_uncertainty_scale"] for row in trajectory[:-1]],
            dtype=np.float64,
        )
        euclidean_opportunity = policy_opportunity(
            euclidean_signal[None, :], euclidean_trace[None, :], event
        )
        fixed_event_risk, fixed_full_risk = _fixed_policy_risk(
            signal, dimension, event, PI_MIN
        )
        operational_event_risk, operational_full_risk = _operational_policy_risk(
            trajectory, event
        )
        event_rows = [row for row, keep in zip(trajectory[:-1], event, strict=True) if keep]
        operational = np.asarray([row["operational_pi"] for row in event_rows])
        ungated = np.asarray([row["ungated_plugin_pi"] for row in event_rows])
        contemporaneous = np.asarray([row["contemporaneous_pi"] for row in event_rows])
        event_signal = np.asarray(
            [row["current_displacement_energy"] for row in event_rows]
        )
        operational_correlation = _event_correlation(event_signal, operational)
        ungated_correlation = _event_correlation(event_signal, ungated)
        summary = {
            "operational_signal_transition_count": int(
                np.sum(operational > PI_SIGNAL_THRESHOLD)
            ),
            "operational_event_pi_range": float(np.ptp(operational)),
            "ungated_signal_transition_count": int(
                np.sum(ungated > PI_SIGNAL_THRESHOLD)
            ),
            "ungated_event_pi_range": float(np.ptp(ungated)),
            "contemporaneous_signal_transition_count": int(
                np.sum(contemporaneous > PI_SIGNAL_THRESHOLD)
            ),
            "contemporaneous_event_pi_range": float(np.ptp(contemporaneous)),
            "event_cold_start_fraction": float(
                np.mean([row["cold_start_active"] for row in event_rows])
            ),
            "event_relative_oracle_risk_reduction": float(
                opportunity["event_relative_risk_reduction"][0]
            ),
            "event_operational_risk": operational_event_risk,
            "event_fixed_005_risk": fixed_event_risk,
            "event_best_fixed_risk": float(
                opportunity["event_best_constant_risk"][0]
            ),
            "event_relative_operational_risk_reduction_vs_fixed_005": (
                _relative_reduction(fixed_event_risk, operational_event_risk)
            ),
            "event_relative_operational_risk_reduction_vs_best_fixed": (
                _relative_reduction(
                    float(opportunity["event_best_constant_risk"][0]),
                    operational_event_risk,
                )
            ),
            "full_operational_risk": operational_full_risk,
            "full_fixed_005_risk": fixed_full_risk,
            "full_relative_operational_risk_reduction_vs_fixed_005": (
                _relative_reduction(fixed_full_risk, operational_full_risk)
            ),
            "event_best_constant_pi": float(
                opportunity["event_best_constant_pi"][0]
            ),
            "event_euclidean_oracle_pi_range": float(
                euclidean_opportunity["event_pi_range"][0]
            ),
            "event_euclidean_oracle_maximum_pi": float(
                np.max(euclidean_opportunity["pi"][0, event])
            ),
            "event_euclidean_oracle_risk_reduction": float(
                euclidean_opportunity["event_relative_risk_reduction"][0]
            ),
            "event_operational_signal_correlation": operational_correlation,
            "event_ungated_signal_correlation": ungated_correlation,
            "event_zero_information_fallback_count": int(
                np.sum([row["zero_information_fallback"] for row in event_rows])
            ),
            "lower_bound_fraction": float(
                np.mean([row["operational_pi"] <= PI_MIN + 1e-12 for row in trajectory])
            ),
            "maximum_operational_pi": float(
                max(row["operational_pi"] for row in trajectory)
            ),
            "maximum_ungated_plugin_pi": float(
                max(row["ungated_plugin_pi"] for row in trajectory)
            ),
            "maximum_contemporaneous_pi": float(
                max(row["contemporaneous_pi"] for row in trajectory)
            ),
        }
        summary["passes_signal_gate"] = bool(
            summary["operational_signal_transition_count"]
            >= MIN_SIGNAL_TRANSITIONS
            and summary["operational_event_pi_range"] >= MIN_PI_RANGE
            and summary["event_relative_operational_risk_reduction_vs_fixed_005"]
            >= MIN_RISK_REDUCTION
            and operational_correlation is not None
            and operational_correlation > 0.0
            and summary["event_zero_information_fallback_count"] == 0
        )
        summary["ungated_diagnostic_signal"] = bool(
            summary["ungated_signal_transition_count"]
            >= MIN_SIGNAL_TRANSITIONS
            and summary["ungated_event_pi_range"] >= MIN_PI_RANGE
            and ungated_correlation is not None
            and ungated_correlation > 0.0
        )
        values[f"h={half_life:.2f}"] = {
            "summary": summary,
            "trajectory": trajectory,
        }
    return {
        "schedule": schedule_name,
        "schedule_hash": schedule.content_hash,
        "event_window": {
            "first_transition": int(np.flatnonzero(event)[0]),
            "last_transition": int(np.flatnonzero(event)[-1]),
            "transition_count": int(event.sum()),
            "maximum_speed_transition": schedule.max_speed_transition - 1,
        },
        "uniform_stream_hash": stream.uniform_stream_hash,
        "elapsed_seconds": time.perf_counter() - started,
        "half_lives": values,
    }


def _aggregate(source_results: list[dict[str, Any]]) -> dict[str, Any]:
    schedules: dict[str, Any] = {}
    primary_passes = []
    primary_diagnostics = []
    for schedule_name in SCHEDULE_NAMES:
        members = [source["schedules"][schedule_name] for source in source_results]
        half_lives = {}
        for half_life in HALF_LIVES:
            key = f"h={half_life:.2f}"
            summaries = [member["half_lives"][key]["summary"] for member in members]
            numeric_keys = [
                name
                for name, value in summaries[0].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            aggregate = {
                name: {
                    "mean": float(np.mean([row[name] for row in summaries])),
                    "minimum": float(np.min([row[name] for row in summaries])),
                    "maximum": float(np.max([row[name] for row in summaries])),
                }
                for name in numeric_keys
            }
            all_pass = all(row["passes_signal_gate"] for row in summaries)
            any_diagnostic = any(row["ungated_diagnostic_signal"] for row in summaries)
            half_lives[key] = {
                "all_sources_pass_signal_gate": all_pass,
                "any_source_has_ungated_diagnostic_signal": any_diagnostic,
                "metrics": aggregate,
                "source_summaries": summaries,
                "mean_trajectory": [
                    {
                        name: (
                            float(np.mean([member["half_lives"][key]["trajectory"][step][name] for member in members]))
                            if isinstance(members[0]["half_lives"][key]["trajectory"][step][name], (int, float, bool))
                            else members[0]["half_lives"][key]["trajectory"][step][name]
                        )
                        for name in members[0]["half_lives"][key]["trajectory"][step]
                    }
                    for step in range(len(members[0]["half_lives"][key]["trajectory"]))
                ],
            }
            if half_life == PRIMARY_HALF_LIFE and schedule_name != "linear":
                primary_passes.append(all_pass)
                primary_diagnostics.append(any_diagnostic)
        schedules[schedule_name] = {
            "schedule_hash": members[0]["schedule_hash"],
            "event_window": members[0]["event_window"],
            "uniform_stream_hashes": sorted(
                {member["uniform_stream_hash"] for member in members}
            ),
            "half_lives": half_lives,
        }
    if any(primary_passes):
        decision = "promote"
    elif any(primary_diagnostics):
        decision = "diagnostic-only"
    else:
        decision = "stop"
    ranking = sorted(
        (name for name in SCHEDULE_NAMES if name != "linear"),
        key=lambda name: schedules[name]["half_lives"]["h=0.05"]["metrics"]
        ["maximum_operational_pi"]["mean"],
        reverse=True,
    )
    return {
        "decision": decision,
        "screening_schedule": ranking[0],
        "schedules": schedules,
    }


def build_fisher_diagnostic(
    repo_root: str | Path,
    *,
    source_count: int = 3,
    device_name: str = "auto",
) -> dict[str, Any]:
    root = Path(repo_root)
    device = resolve_device(device_name)
    configure_torch_runtime(
        deterministic_algorithms=True,
        warn_only=device.type == "cuda",
    )
    train_dataset, _ = load_mnist_datasets(
        root / "cache" / "mnist_experiment" / "datasets", download=False
    )
    train_targets = dataset_targets(train_dataset)
    paths = _source_paths(root, source_count)
    source_results = []
    provenance = []
    for source_index, path in enumerate(paths, start=1):
        source = _source_inputs(path)
        source["loaded"] = load_replica_bundle_for_config(
            root / "cache" / "mnist_experiment" / "replicas",
            source["config"],
            device="cpu",
        )
        schedules = {}
        for schedule_name in SCHEDULE_NAMES:
            print(
                f"Fisher diagnostic source {source_index}/{source_count}: {schedule_name}",
                flush=True,
            )
            schedules[schedule_name] = _analyze_source_schedule(
                source,
                schedule_name,
                train_dataset,
                train_targets,
                device=device,
            )
        source_results.append(
            {"run_id": path.name, "schedules": schedules}
        )
        provenance.append(source["provenance"])
    aggregate = _aggregate(source_results)
    return {
        "schema_version": PLAN4_FISHER_ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "plan4_fisher_risk_score_only_diagnostic",
        "analysis_code_sha256": _sha256(Path(__file__)),
        "device": str(device),
        "source_count": source_count,
        "source_provenance": provenance,
        "source_provenance_hash": _canonical_hash(provenance),
        "half_lives": list(HALF_LIVES),
        "schedule_names": list(SCHEDULE_NAMES),
        "contracts": {
            "parameter_path": "interpolated high-sample reference optimum",
            "fisher_update": "rank-8-plus-diagonal direct EMA with operational pi",
            "fisher_projection": (
                "fixed-rank Lanczos without redundant dense PSD "
                "eigendecomposition"
            ),
            "metric_timing": "pre-decision previous auxiliary summary",
            "uncertainty": "source oracle-residual Fisher energy divided by source covariance scale",
            "trend": "predictable EMA of prior reference-path displacements",
            "optimization_performed": False,
            "hvp_performed": False,
            "fisher_inverse_used": False,
            "risk_comparison": "conditional on each diagnostic Fisher trajectory with policy-specific q",
            "primary_half_life": PRIMARY_HALF_LIFE,
            "sensitivity_half_life": SENSITIVITY_HALF_LIFE,
        },
        "gate": {
            "pi_signal_threshold": PI_SIGNAL_THRESHOLD,
            "minimum_signal_transitions": MIN_SIGNAL_TRANSITIONS,
            "minimum_pi_range": MIN_PI_RANGE,
            "minimum_relative_risk_reduction": MIN_RISK_REDUCTION,
            "promotion_requires_every_source": True,
        },
        "decision": aggregate["decision"],
        "screening_schedule": aggregate["screening_schedule"],
        "schedules": aggregate["schedules"],
        "source_results": source_results,
    }


def write_fisher_diagnostic(
    repo_root: str | Path,
    *,
    source_count: int = 3,
    device_name: str = "auto",
) -> Path:
    root = Path(repo_root)
    analysis = build_fisher_diagnostic(
        root, source_count=source_count, device_name=device_name
    )
    identity = {
        "schema_version": PLAN4_FISHER_ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": analysis["analysis_kind"],
        "analysis_code_sha256": analysis["analysis_code_sha256"],
        "source_provenance_hash": analysis["source_provenance_hash"],
        "source_count": source_count,
        "half_lives": analysis["half_lives"],
        "schedule_names": analysis["schedule_names"],
        "gate": analysis["gate"],
        "device": analysis["device"],
    }
    digest = _canonical_hash(identity)
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "fisher_analysis"
        / f"phase4__{digest[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4FisherAnalysisError(
                f"incomplete Fisher analysis: {destination}"
            )
        if _read_json(destination / "manifest.json") != identity:
            raise Plan4FisherAnalysisError("completed Fisher analysis identity differs")
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
