"""Run source-backed decomposed-EDR trajectories from a shared initial state."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.config import ControllerConfig
from src.controller import (
    ControllerState,
    DecomposedRiskDecision,
    DiscountedMovementState,
    accept_controller_step,
    decide_controller,
    decide_decomposed_risk_controller,
    decide_tracked_q_covariance_controller,
)
from src.ewc import build_optimizer, take_ewc_proposal
from src.hybrid import blend_archive_fisher
from src.initialization import state_dict_hash
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.representations import (
    DiagonalFisher,
    LowRankDiagonalFisher,
    representation_from_artifact,
)
from src.seeding import derive_component_seed

from .data import RotatedPartitions, RotatedStreamPlan
from .phase4_artifacts import LoadedRotatedPhase4Run, load_completed_phase4_run
from .phase4_metrics import (
    evaluate_materialized_classifier,
    materialize_base_panel,
    materialize_rotated_panel,
)
from .phase5_double_lap_artifacts import (
    LoadedDoubleLapRun,
    load_completed_double_lap_run,
)
from .phase5_single_lap_artifacts import (
    LoadedSingleLapRun,
    load_completed_single_lap_run,
)
from .phase8_artifacts import PHASE8_REQUIRED_ARTIFACTS, Phase8RunStore
from .phase8_config import Phase8Config, load_phase8_config
from .run import CALIBRATION_BINS, _state_dict_cpu
from .run_phase3 import _fresh_fisher
from .run_phase5_double_lap import _learner_optimizer_config
from .transform import tensor_content_hash


FIXED_PANEL_ANGLES = {
    "panel_000": 0.0,
    "panel_015": 15.0,
    "panel_030": 30.0,
}


@dataclasses.dataclass(frozen=True)
class _SourceBundle:
    path: Path
    loaded: LoadedSingleLapRun | LoadedDoubleLapRun | LoadedRotatedPhase4Run
    source_config: Any
    partitions: RotatedPartitions
    plans: dict[str, RotatedStreamPlan]
    streams: dict[str, dict[str, Tensor]]
    source_metrics: dict[str, dict[str, tuple[dict[str, Any], ...]]]
    initial_state: dict[str, Tensor]
    initial_fisher: LowRankDiagonalFisher
    source_fixed_condition: str
    source_legacy_condition: str | None


@dataclasses.dataclass
class _LearnerState:
    schedule_kind: str
    condition: str
    model: nn.Module
    layout: ParameterLayout
    optimizer: torch.optim.Optimizer
    fisher: LowRankDiagonalFisher
    controller: ControllerState
    movement: DiscountedMovementState | None
    rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    parameters: list[Tensor] = dataclasses.field(default_factory=list)
    displacements: list[Tensor] = dataclasses.field(default_factory=list)
    optimizer_iterations: int = 0
    optimizer_function_evaluations: int = 0
    optimizer_event_evaluations: int = 0
    score_gradient_count: int = 0
    evaluation_wall_seconds: float = 0.0
    learner_wall_seconds: float = 0.0
    fisher_wall_seconds: float = 0.0
    lanczos_diagnostics: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("cache/mnist_experiment/datasets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/phase8/runs"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _extract_initial_state(model_states: dict[str, Any]) -> dict[str, Tensor]:
    states = []
    for outer in model_states.values():
        if "initial" in outer:
            states.append(outer["initial"])
        else:
            states.extend(value["initial"] for value in outer.values())
    if not states:
        raise ValueError("source contains no initial model state")
    hashes = {state_dict_hash(state) for state in states}
    if len(hashes) != 1:
        raise ValueError("source conditions do not share one initial model state")
    return states[0]


def _load_source(config: Phase8Config, repo_root: Path) -> _SourceBundle:
    source_path = Path(config.source_artifact)
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    if config.source_kind == "single_lap":
        loaded = load_completed_single_lap_run(source_path)
        plans = dict(loaded.stream_plans)
        source_metrics = {
            schedule: dict(values)
            for schedule, values in loaded.trajectory_metrics.items()
        }
        fixed_condition = "fixed_pi005"
        legacy_condition = "edr_slowtrend_slowaction"
    elif config.source_kind == "double_lap":
        loaded = load_completed_double_lap_run(source_path)
        plans = dict(loaded.stream_plans)
        source_metrics = {
            schedule: dict(values)
            for schedule, values in loaded.trajectory_metrics.items()
        }
        fixed_condition = "fixed_pi005"
        legacy_condition = "edr_fasttrend_slowaction"
    else:
        loaded = load_completed_phase4_run(source_path)
        plans = {"canonical": loaded.stream_plan}
        source_metrics = {"canonical": dict(loaded.trajectory_metrics)}
        fixed_condition = "ewc_fixed_pi005"
        legacy_condition = None

    source_config = loaded.config
    if (
        config.runtime.dtype != source_config.runtime.dtype
        or config.runtime.deterministic_algorithms
        != source_config.runtime.deterministic_algorithms
    ):
        raise ValueError("Plan 8 dtype and determinism must match the source runtime")
    stream_value = torch.load(
        source_path / "stream_tensors.pt", map_location="cpu", weights_only=True
    )
    if config.source_kind == "canonical":
        streams = {"canonical": stream_value}
    else:
        streams = stream_value
    model_states = torch.load(
        source_path / "model_states.pt", map_location="cpu", weights_only=True
    )
    initial_state = _extract_initial_state(model_states)
    fisher_value = torch.load(
        source_path / "initial_fisher.pt", map_location="cpu", weights_only=True
    )
    representation = representation_from_artifact(fisher_value["representation"])
    if not isinstance(representation, LowRankDiagonalFisher) or representation.rank != 8:
        raise ValueError("Plan 8 requires a rank-8-plus-diagonal initial Fisher")
    for schedule, plan in plans.items():
        stream = streams[schedule]
        if stream["inputs"].shape[0] != plan.schedule.num_points:
            raise ValueError("source stream and schedule lengths differ")
        if stream["targets"].shape[:2] != stream["inputs"].shape[:2]:
            raise ValueError("source stream input and target shapes differ")
    return _SourceBundle(
        path=source_path,
        loaded=loaded,
        source_config=source_config,
        partitions=loaded.partitions,
        plans=plans,
        streams=streams,
        source_metrics=source_metrics,
        initial_state=initial_state,
        initial_fisher=representation,
        source_fixed_condition=fixed_condition,
        source_legacy_condition=legacy_condition,
    )


def _source_contract(source: _SourceBundle, config: Phase8Config) -> dict[str, Any]:
    critical = (
        "config.json",
        "manifest.json",
        "partitions.json",
        "stream_tensors.pt",
        "initial_fisher.pt",
        "model_states.pt",
        "evaluation_panel.json",
        "trajectory_metrics.json",
    )
    plan_names = (
        ("stream_plan.json",)
        if config.source_kind == "canonical"
        else ("stream_plans.json",)
    )
    files = critical + plan_names
    return {
        "source_kind": config.source_kind,
        "source_path": str(source.path),
        "source_run_id": source.loaded.manifest["run_id"],
        "source_config_hash": source.loaded.manifest["config_hash"],
        "source_manifest_status": source.loaded.manifest["status"],
        "source_file_sha256": {
            name: _file_hash(source.path / name) for name in files
        },
        "initial_state_hash": state_dict_hash(source.initial_state),
        "initial_fisher_kind": "low_rank_diagonal",
        "initial_fisher_rank": source.initial_fisher.rank,
        "parameter_count": source.initial_fisher.shape[0],
        "partition_hash": source.partitions.content_hash,
        "schedule_hashes": {
            name: plan.schedule.content_hash for name, plan in source.plans.items()
        },
        "stream_plan_hashes": {
            name: plan.content_hash for name, plan in source.plans.items()
        },
        "stream_tensor_hashes": {
            name: {
                field: tensor_content_hash(tensor)
                for field, tensor in stream.items()
            }
            for name, stream in source.streams.items()
        },
        "source_fixed_condition": source.source_fixed_condition,
        "source_legacy_condition": source.source_legacy_condition,
        "endogenous_source_state_reused": False,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, operation):
    _synchronize(device)
    started = time.perf_counter()
    value = operation()
    _synchronize(device)
    return value, time.perf_counter() - started


def _make_states(
    source: _SourceBundle,
    config: Phase8Config,
    schedule_kind: str,
    *,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> dict[str, _LearnerState]:
    states = {}
    for condition in config.conditions:
        if condition == "hybrid_b032_decomposed_edr":
            raise NotImplementedError("adaptive Hybrid is gated behind Phase 4")
        model, _ = build_canonical_model(
            config.replica_seed, device=device, dtype=training_dtype
        )
        model.load_state_dict(source.initial_state)
        layout = ParameterLayout.from_module(model)
        states[condition] = _LearnerState(
            schedule_kind=schedule_kind,
            condition=condition,
            model=model,
            layout=layout,
            optimizer=build_optimizer(
                model, _learner_optimizer_config(source.source_config)
            ),
            fisher=source.initial_fisher.to(device=device, dtype=matrix_dtype),
            controller=ControllerState.initialize(
                layout.total_numel,
                source.source_config.data.initialization_size,
                dtype=matrix_dtype,
                device="cpu",
            ),
            movement=(
                DiscountedMovementState()
                if condition == "decomposed_edr"
                else None
            ),
        )
    return states


def _choose_decision(
    state: _LearnerState,
    config: Phase8Config,
    *,
    batch_size: int,
) -> tuple[Any, DecomposedRiskDecision | None]:
    if state.condition == "fixed_pi005_sentinel":
        fixed = 0.05
    elif state.condition == "fixed_pi0025":
        fixed = 0.025
    else:
        fixed = None
    if fixed is not None:
        controller_config = ControllerConfig(
            policy="fixed_unified",
            fixed_pi=fixed,
            pi_min=config.controller.pi_min,
            pi_max=config.controller.pi_max,
            trend_half_life_p=config.controller.trend_half_life_degrees,
            trace_epsilon=config.controller.trace_epsilon,
            oracle_mode="none",
            reference_optimum_artifact=None,
            risk_metric="fisher",
            action_half_life_steps=None,
            damping=None,
            epsilon=None,
        )
        return (
            decide_controller(
                state.controller,
                controller_config,
                batch_size=batch_size,
                fisher=state.fisher,
            ),
            None,
        )
    if state.condition == "tracked_q_covariance":
        return (
            decide_tracked_q_covariance_controller(
                state.controller,
                batch_size=batch_size,
                pi_min=config.controller.pi_min,
                pi_max=config.controller.pi_max,
                cold_start_pi=config.controller.cold_start_pi,
                cold_start_steps=config.controller.cold_start_steps,
            ),
            None,
        )
    if state.condition != "decomposed_edr" or state.movement is None:
        raise ValueError(f"unsupported Plan 8 condition: {state.condition}")
    decomposed = decide_decomposed_risk_controller(
        state.controller,
        state.movement,
        batch_size=batch_size,
        fisher=state.fisher,
        pi_min=config.controller.pi_min,
        pi_max=config.controller.pi_max,
        cold_start_pi=config.controller.cold_start_pi,
        cold_start_steps=config.controller.cold_start_steps,
        movement_half_life_steps=config.controller.movement_half_life_steps,
        trace_epsilon=config.controller.trace_epsilon,
    )
    return decomposed.controller, decomposed


def _evaluate(
    state: _LearnerState,
    panels: dict[float, tuple[Tensor, Tensor, str]],
    current_angle: float,
    source_config: Any,
    *,
    nine_prevalence: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    requested = {"current": current_angle, **FIXED_PANEL_ANGLES}
    by_angle = {}
    started = time.perf_counter()
    for angle in dict.fromkeys(requested.values()):
        inputs, targets, _ = panels[angle]
        by_angle[angle] = evaluate_materialized_classifier(
            state.model,
            inputs,
            targets,
            batch_size=source_config.initialization.batch_size,
            device=device,
            dtype=dtype,
            calibration_bins=CALIBRATION_BINS,
            nine_prevalence=nine_prevalence,
        )
    _synchronize(device)
    state.evaluation_wall_seconds += time.perf_counter() - started
    return {
        f"{name}_{field}": value
        for name, angle in requested.items()
        for field, value in by_angle[angle].items()
    }


def _apply_update(
    state: _LearnerState,
    inputs: Tensor,
    targets: Tensor,
    config: Phase8Config,
    source: _SourceBundle,
    *,
    step: int,
    parameter_before: Tensor,
    delta_degrees: float,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> dict[str, Any]:
    decision, decomposed = _choose_decision(
        state, config, batch_size=inputs.shape[0]
    )
    predictable_fisher = state.fisher
    fresh, fisher_seconds = _timed(
        device,
        lambda: _fresh_fisher(
            state.model,
            state.layout,
            inputs,
            targets,
            matrix_dtype=matrix_dtype,
        ),
    )
    state.fisher_wall_seconds += fisher_seconds
    state.score_gradient_count += inputs.shape[0]
    if step == 0:
        fisher_mapping = {
            "update": "initial_summary_only",
            "blend_gain": decision.applied_pi,
            "previous_trace": float(state.fisher.diagonal_vector().sum()),
            "fresh_trace": float(torch.trace(fresh)),
            "candidate_trace": float(state.fisher.diagonal_vector().sum()),
            "lanczos": None,
        }
    else:
        if state.condition == "fixed_pi005_sentinel":
            if config.source_kind == "single_lap":
                seed_label = (
                    "plan5_single_lap_update_lanczos:"
                    f"schedule={state.schedule_kind}:condition=fixed_pi005:step={step}"
                )
            elif config.source_kind == "double_lap":
                seed_label = (
                    "plan5_double_lap_update_lanczos:"
                    f"schedule={state.schedule_kind}:condition=fixed_pi005:step={step}"
                )
            else:
                seed_label = f"plan5_phase4_ewc_lanczos:step={step}"
        else:
            seed_label = (
                "plan8_update_lanczos:"
                f"source={config.source_kind}:schedule={state.schedule_kind}:"
                f"condition={state.condition}:step={step}"
            )
        update, update_seconds = _timed(
            device,
            lambda: blend_archive_fisher(
                state.fisher,
                fresh,
                blend_gain=decision.applied_pi,
                rank=source.source_config.fisher.rank,
                lanczos_seed=derive_component_seed(
                    config.replica_seed,
                    seed_label,
                ),
            ),
        )
        state.fisher_wall_seconds += update_seconds
        state.fisher = update.representation
        diagnostics = update.lanczos.mapping()
        state.lanczos_diagnostics.append(diagnostics)
        fisher_mapping = {
            "update": "direct_ema",
            "blend_gain": update.blend_gain,
            "previous_trace": update.previous_trace,
            "fresh_trace": update.fresh_trace,
            "candidate_trace": update.candidate_trace,
            "lanczos": diagnostics,
        }

    proposal, learner_seconds = _timed(
        device,
        lambda: take_ewc_proposal(
            state.model,
            state.layout,
            inputs,
            targets,
            state.fisher.to(dtype=training_dtype),
            _learner_optimizer_config(source.source_config),
            state.optimizer,
            adaptation_weight=decision.applied_pi,
            penalty_anchor=parameter_before,
        ),
    )
    state.learner_wall_seconds += learner_seconds
    state.optimizer_iterations += proposal.optimizer_iterations
    state.optimizer_function_evaluations += proposal.optimizer_function_evaluations
    state.optimizer_event_evaluations += (
        inputs.shape[0] * proposal.optimizer_function_evaluations
    )
    displacement = (
        state.layout.flatten_module(state.model, detach=True) - parameter_before
    ).cpu()
    if not torch.isfinite(displacement).all():
        raise RuntimeError("Plan 8 displacement is nonfinite")
    state.displacements.append(displacement)
    acceptance = accept_controller_step(
        state.controller,
        decision,
        displacement.to(dtype=matrix_dtype),
        batch_size=inputs.shape[0],
        delta_p=delta_degrees,
        half_life_p=config.controller.trend_half_life_degrees,
        fisher=predictable_fisher,
    )
    state.controller = acceptance.state
    if decomposed is not None:
        state.movement = decomposed.state
    controller_mapping = {
        **decision.mapping(extended=True),
        "decision_uses_current_batch": False,
        "cold_start_semantics": "accepted_updates",
        "cold_start_steps": config.controller.cold_start_steps,
    }
    if decomposed is not None:
        controller_mapping.update(decomposed.mapping())
    elif state.condition == "tracked_q_covariance":
        controller_mapping.update(
            {
                "covariance_only_pi": decision.plugin_pi,
                "instantaneous_movement_premium": None,
                "discounted_movement_premium": 0.0,
                "movement_updates": 0,
                "unsupported_scale_fallback": False,
            }
        )
    return {
        "proposal": proposal.metrics_mapping(),
        "fisher_update": fisher_mapping,
        "controller": controller_mapping,
        "controller_acceptance": {
            "trend_gain": acceptance.gain,
            "residual_risk_energy": acceptance.residual_squared,
            "residual_euclidean_squared": acceptance.residual_euclidean_squared,
            "scale_observation": acceptance.scale_observation,
            "state_after": state.controller.scalar_mapping(
                config.controller.trace_epsilon, risk_metric="fisher"
            ),
        },
    }


def _auc(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...], field: str) -> float:
    x = torch.tensor(
        [row["observations_before_evaluation"] for row in rows],
        dtype=torch.float64,
    )
    y = torch.tensor([row[field] for row in rows], dtype=torch.float64)
    width = float(x[-1] - x[0])
    if width <= 0.0:
        return float(y.mean())
    return float(torch.trapezoid(y, x=x) / width)


def _summary(state: _LearnerState, transitions_per_arrow: int) -> dict[str, Any]:
    rows = state.rows
    transition_count = len(rows) - 1
    complete_legs = transition_count // transitions_per_arrow
    leg_aucs = []
    for leg in range(complete_legs):
        start = leg * transitions_per_arrow
        stop = (leg + 1) * transitions_per_arrow
        leg_aucs.append(_auc(rows[start : stop + 1], "current_environment_accuracy"))
    controller_rows = [row for row in rows if row["controller"] is not None]
    post_cold = [
        row for row in controller_rows if not row["controller"]["cold_start_active"]
    ]
    action_rows = post_cold or controller_rows
    actions = [float(row["controller"]["applied_pi"]) for row in action_rows]
    speeds = [
        float(row["lagged_angular_speed_degrees_per_update"])
        for row in action_rows
        if row["lagged_angular_speed_degrees_per_update"] is not None
    ]
    speed_actions = [
        float(row["controller"]["applied_pi"])
        for row in action_rows
        if row["lagged_angular_speed_degrees_per_update"] is not None
    ]
    correlation = None
    if len(speeds) >= 2 and statistics.pstdev(speeds) > 0 and statistics.pstdev(speed_actions) > 0:
        correlation = statistics.correlation(speeds, speed_actions)
    mean_speed = statistics.fmean(speeds) if speeds else 0.0
    fast = [a for a, speed in zip(speed_actions, speeds, strict=True) if speed > mean_speed]
    slow = [a for a, speed in zip(speed_actions, speeds, strict=True) if speed <= mean_speed]
    return {
        "schedule_kind": state.schedule_kind,
        "condition": state.condition,
        "environment_accuracy_auc": _auc(rows, "current_environment_accuracy"),
        "environment_nll_auc": _auc(rows, "current_nll"),
        "expected_calibration_error_auc": _auc(
            rows, "current_expected_calibration_error"
        ),
        "upright_accuracy_auc": _auc(rows, "panel_000_environment_accuracy"),
        "leg_environment_accuracy_aucs": leg_aucs,
        "final_current_environment_accuracy": rows[-1][
            "current_environment_accuracy"
        ],
        "final_current_nll": rows[-1]["current_nll"],
        "final_upright_environment_accuracy": rows[-1][
            "panel_000_environment_accuracy"
        ],
        "final_worst_class_recall": rows[-1]["current_worst_class_recall"],
        "optimizer_iterations": state.optimizer_iterations,
        "optimizer_function_evaluations": state.optimizer_function_evaluations,
        "optimizer_event_evaluations": state.optimizer_event_evaluations,
        "score_gradient_count": state.score_gradient_count,
        "evaluation_wall_time_seconds": state.evaluation_wall_seconds,
        "learner_optimization_wall_time_seconds": state.learner_wall_seconds,
        "fisher_update_wall_time_seconds": state.fisher_wall_seconds,
        "lanczos_update_count": len(state.lanczos_diagnostics),
        "lanczos_minimum_realized_rank": (
            None
            if not state.lanczos_diagnostics
            else min(item["realized_rank"] for item in state.lanczos_diagnostics)
        ),
        "cold_start_transition_count": len(controller_rows) - len(post_cold),
        "action_min": min(actions),
        "action_mean": statistics.fmean(actions),
        "action_max": max(actions),
        "action_span": max(actions) - min(actions),
        "action_total_variation": sum(
            abs(right - left) for left, right in zip(actions, actions[1:])
        ),
        "action_floor_fraction": statistics.fmean(
            float(row["controller"]["lower_bound_active"]) for row in action_rows
        ),
        "unsupported_scale_fallback_fraction": statistics.fmean(
            float(row["controller"].get("unsupported_scale_fallback", False))
            for row in action_rows
        ),
        "lagged_speed_action_correlation": correlation,
        "fast_region_action_mean": None if not fast else statistics.fmean(fast),
        "slow_region_action_mean": None if not slow else statistics.fmean(slow),
        "fast_minus_slow_action": (
            None if not fast or not slow else statistics.fmean(fast) - statistics.fmean(slow)
        ),
    }


def _source_summary(
    rows: tuple[dict[str, Any], ...], transitions_per_arrow: int
) -> dict[str, float]:
    leg_count = (len(rows) - 1) // transitions_per_arrow
    return {
        "environment_accuracy_auc": _auc(rows, "current_environment_accuracy"),
        "environment_nll_auc": _auc(rows, "current_nll"),
        "final_current_environment_accuracy": rows[-1][
            "current_environment_accuracy"
        ],
        "final_upright_environment_accuracy": rows[-1][
            "panel_000_environment_accuracy"
        ],
        "final_worst_class_recall": rows[-1]["current_worst_class_recall"],
        **{
            f"leg_{leg}_environment_accuracy_auc": _auc(
                rows[
                    leg * transitions_per_arrow : (leg + 1) * transitions_per_arrow + 1
                ],
                "current_environment_accuracy",
            )
            for leg in range(leg_count)
        },
    }


def _compatibility(
    summaries: dict[str, dict[str, dict[str, Any]]],
    source: _SourceBundle,
    config: Phase8Config,
    full_trajectory: bool,
) -> dict[str, Any]:
    if "fixed_pi005_sentinel" not in config.conditions or not full_trajectory:
        return {"evaluated": False, "passed": None, "schedules": {}}
    values = {}
    passed = True
    for schedule, by_condition in summaries.items():
        observed = by_condition["fixed_pi005_sentinel"]
        expected = _source_summary(
            source.source_metrics[schedule][source.source_fixed_condition],
            source.plans[schedule].schedule.transitions_per_arrow,
        )
        differences = {
            "environment_accuracy_auc": observed["environment_accuracy_auc"]
            - expected["environment_accuracy_auc"],
            "environment_nll_auc": observed["environment_nll_auc"]
            - expected["environment_nll_auc"],
            "final_current_environment_accuracy": observed[
                "final_current_environment_accuracy"
            ]
            - expected["final_current_environment_accuracy"],
        }
        schedule_passed = (
            abs(differences["environment_accuracy_auc"])
            <= config.compatibility.accuracy_auc_tolerance
            and abs(differences["environment_nll_auc"])
            <= config.compatibility.nll_auc_tolerance
            and abs(differences["final_current_environment_accuracy"])
            <= config.compatibility.final_accuracy_tolerance
        )
        values[schedule] = {
            "observed": observed,
            "expected": expected,
            "differences": differences,
            "passed": schedule_passed,
        }
        passed = passed and schedule_passed
    return {"evaluated": True, "passed": passed, "schedules": values}


def run_phase8(
    config: Phase8Config,
    *,
    data_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    download: bool = False,
    resume: bool = False,
) -> Path:
    config.validate()
    repo_path = Path(repo_root)
    source = _load_source(config, repo_path)
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.dtype)
    matrix_dtype = resolve_dtype(source.source_config.fisher.matrix_dtype)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    session = Phase8RunStore(output_root).begin(config, repo_path, resume=resume)
    total_started = time.perf_counter()
    session.write_json("source_contract.json", _source_contract(source, config))

    _, test_dataset = load_mnist_datasets(data_root, download=download)
    test_targets = dataset_targets(test_dataset)
    nine_prevalence = float((test_targets == 9).double().mean())
    base_inputs, base_targets = materialize_base_panel(
        test_dataset,
        source.partitions.evaluation,
        num_workers=config.runtime.num_workers,
    )
    fixed_panels = {
        angle: materialize_rotated_panel(
            base_inputs, base_targets, angle, source.source_config.rotation
        )
        for angle in set(FIXED_PANEL_ANGLES.values())
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    all_states = {}
    total_points = sum(
        min(
            plan.schedule.num_transitions,
            config.transition_limit or plan.schedule.num_transitions,
        )
        + 1
        for plan in source.plans.values()
    )
    progress = tqdm(total=total_points, desc="Plan 8 rechallenge", unit="point")
    for schedule_kind, plan in source.plans.items():
        schedule = plan.schedule
        transition_count = min(
            schedule.num_transitions,
            config.transition_limit or schedule.num_transitions,
        )
        states = _make_states(
            source,
            config,
            schedule_kind,
            device=device,
            training_dtype=training_dtype,
            matrix_dtype=matrix_dtype,
        )
        panels = dict(fixed_panels)
        stream = source.streams[schedule_kind]
        for step in range(transition_count + 1):
            angle = schedule.angles_degrees[step]
            if angle not in panels:
                panels[angle] = materialize_rotated_panel(
                    base_inputs, base_targets, angle, source.source_config.rotation
                )
            lagged_speed = (
                None
                if step == 0
                else abs(angle - schedule.angles_degrees[step - 1])
            )
            next_speed = (
                None
                if step == transition_count
                else abs(schedule.angles_degrees[step + 1] - angle)
            )
            for state in states.values():
                parameter_before = state.layout.flatten_module(
                    state.model, detach=True
                )
                state.parameters.append(parameter_before.cpu())
                row = {
                    "step": step,
                    "schedule_kind": schedule_kind,
                    "condition": state.condition,
                    "angle_degrees": angle,
                    "leg_id": schedule.leg_ids[step],
                    "direction_to_next": schedule.directions_to_next[step],
                    "knot": schedule.knot_flags[step],
                    "cumulative_angular_degrees": schedule.cumulative_degrees[step],
                    "lagged_angular_speed_degrees_per_update": lagged_speed,
                    "next_angular_speed_degrees_per_update": next_speed,
                    "observations_before_evaluation": (
                        step * source.source_config.data.samples_per_step
                    ),
                    "parameter_hash": tensor_content_hash(parameter_before.cpu()),
                    "proposal": None,
                    "fisher_update": None,
                    "controller": None,
                    "controller_acceptance": None,
                    **_evaluate(
                        state,
                        panels,
                        angle,
                        source.source_config,
                        nine_prevalence=nine_prevalence,
                        device=device,
                        dtype=training_dtype,
                    ),
                }
                if step < transition_count:
                    inputs = stream["inputs"][step].to(
                        device=device, dtype=training_dtype
                    )
                    targets = stream["targets"][step].to(device=device)
                    row.update(
                        _apply_update(
                            state,
                            inputs,
                            targets,
                            config,
                            source,
                            step=step,
                            parameter_before=parameter_before,
                            delta_degrees=float(next_speed),
                            device=device,
                            training_dtype=training_dtype,
                            matrix_dtype=matrix_dtype,
                        )
                    )
                state.rows.append(row)
            if angle not in FIXED_PANEL_ANGLES.values():
                del panels[angle]
            progress.update(1)
        all_states[schedule_kind] = states
    progress.close()

    metrics = {
        schedule: {condition: state.rows for condition, state in states.items()}
        for schedule, states in all_states.items()
    }
    trajectories = {}
    model_states = {}
    controller_states = {}
    summaries = {}
    for schedule, states in all_states.items():
        trajectories[schedule] = {}
        model_states[schedule] = {}
        controller_states[schedule] = {}
        summaries[schedule] = {}
        transitions_per_arrow = source.plans[
            schedule
        ].schedule.transitions_per_arrow
        for condition, state in states.items():
            parameters = torch.stack(state.parameters)
            displacements = torch.stack(state.displacements)
            if not torch.equal(displacements, parameters[1:] - parameters[:-1]):
                raise RuntimeError("Plan 8 displacement identity failed")
            trajectories[schedule][condition] = {
                "parameters": parameters,
                "displacements": displacements,
            }
            model_states[schedule][condition] = {
                "initial": source.initial_state,
                "final": _state_dict_cpu(state.model),
            }
            controller_states[schedule][condition] = {
                "controller": state.controller,
                "movement": state.movement,
            }
            summaries[schedule][condition] = _summary(
                state, transitions_per_arrow
            )

    full_trajectory = all(
        len(metrics[schedule][condition])
        == source.plans[schedule].schedule.num_points
        for schedule in metrics
        for condition in config.conditions
    )
    compatibility = _compatibility(
        summaries, source, config, full_trajectory=full_trajectory
    )
    controller_rows = [
        row
        for values in metrics.values()
        for rows in values.values()
        for row in rows
        if row["controller"] is not None
    ]
    reconstruction_errors = []
    q_errors = []
    for row in controller_rows:
        controller = row["controller"]
        if controller["cold_start_active"]:
            reconstructed = config.controller.cold_start_pi
        elif controller["policy"] == "tracked_q_covariance":
            reconstructed = controller["covariance_only_pi"]
        elif controller["policy"] == "decomposed_edr":
            reconstructed = (
                controller["covariance_only_pi"]
                if controller["unsupported_scale_fallback"]
                else (
                    controller["discounted_movement_premium"]
                    + source.source_config.data.samples_per_step
                    / controller["effective_size"]
                )
                / (
                    controller["discounted_movement_premium"]
                    + source.source_config.data.samples_per_step
                    / controller["effective_size"]
                    + 1.0
                )
            )
        else:
            reconstructed = controller["raw_pi"]
        reconstruction_errors.append(abs(controller["raw_pi"] - reconstructed))
        after = row["controller_acceptance"]["state_after"]
        expected_q = (
            (1.0 - controller["applied_pi"]) ** 2
            / controller["effective_size"]
            + controller["applied_pi"] ** 2
            / source.source_config.data.samples_per_step
        )
        q_errors.append(abs(after["q"] - expected_q))
    all_lanczos = [
        item
        for states in all_states.values()
        for state in states.values()
        for item in state.lanczos_diagnostics
    ]
    checks = {
        "all_finite": _finite_tree(metrics) and _finite_tree(summaries),
        "source_completed": source.loaded.manifest["status"] == "completed",
        "shared_initial_parameter_hash": len(
            {
                rows[0]["parameter_hash"]
                for values in metrics.values()
                for rows in values.values()
            }
        )
        == 1,
        "all_decisions_predictable": all(
            row["controller"]["decision_uses_current_batch"] is False
            for row in controller_rows
        ),
        "all_trajectories_complete": full_trajectory,
        "rank8_resolved": all(item["realized_rank"] == 8 for item in all_lanczos),
        "maximum_q_recursion_error": max(q_errors, default=0.0),
        "maximum_action_reconstruction_error": max(
            reconstruction_errors, default=0.0
        ),
        "compatibility_sentinel": compatibility,
        "endogenous_source_state_reused": False,
    }
    checks["gate_recommendation"] = (
        "go"
        if all(
            (
                checks["all_finite"],
                checks["source_completed"],
                checks["shared_initial_parameter_hash"],
                checks["all_decisions_predictable"],
                checks["rank8_resolved"],
                checks["maximum_q_recursion_error"] <= 1e-15,
                compatibility["passed"] is not False,
            )
        )
        else "no_go"
    )
    peak_cuda = (
        0 if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
    )
    session.write_json("trajectory_metrics.json", metrics)
    session.write_torch("trajectories.pt", trajectories)
    session.write_torch("model_states.pt", model_states)
    session.write_torch("controller_states.pt", controller_states)
    session.write_json("operational_checks.json", checks)
    session.write_json(
        "run_summary.json",
        {
            "config_hash": config.config_hash,
            "source_run_id": source.loaded.manifest["run_id"],
            "source_kind": config.source_kind,
            "conditions": list(config.conditions),
            "schedules": list(source.plans),
            "samples_per_step": source.source_config.data.samples_per_step,
            "condition_summaries": summaries,
            "total_wall_time_seconds": time.perf_counter() - total_started,
            "peak_process_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            * 1024,
            "peak_cuda_memory_bytes": peak_cuda,
        },
    )
    return session.complete(PHASE8_REQUIRED_ARTIFACTS)


def main() -> None:
    arguments = parse_arguments()
    config = load_phase8_config(arguments.config)
    repo_root = Path(__file__).parents[2]
    path = run_phase8(
        config,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        repo_root=repo_root,
        download=arguments.download,
        resume=arguments.resume,
    )
    print(json.dumps({"run_id": config.run_id, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
