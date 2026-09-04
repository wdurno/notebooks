"""Run the paired linear-versus-sigmoid double-lap EDR challenge."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import resource
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.config import ControllerConfig
from src.controller import (
    ControllerState,
    DiscountedRiskState,
    accept_controller_step,
    decide_controller,
    decide_discounted_risk_controller,
)
from src.ewc import build_optimizer, take_ewc_proposal
from src.hybrid import blend_archive_fisher
from src.initialization import state_dict_hash
from src.lanczos_wrapper import approximate_low_rank_diagonal
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.representations import DiagonalFisher, LowRankDiagonalFisher
from src.seeding import derive_component_seed

from .data import generate_rotated_stream, partition_all_digit_mnist
from .phase4_metrics import (
    evaluate_materialized_classifier,
    materialize_base_panel,
    materialize_rotated_panel,
)
from .phase5_double_lap_artifacts import (
    DOUBLE_LAP_REQUIRED_ARTIFACTS,
    RotatedDoubleLapRunStore,
)
from .phase5_double_lap_config import (
    DOUBLE_LAP_CONDITIONS,
    DOUBLE_LAP_FIXED_PI,
    DOUBLE_LAP_SCHEDULES,
    RotatedDoubleLapConfig,
    load_double_lap_config,
)
from .phase5_single_lap_config import RotatedSlowSingleLapConfig
from .run import (
    CALIBRATION_BINS,
    _fit_upright_initializer,
    _learner_optimizer_config,
    _state_dict_cpu,
)
from .run_phase3 import _estimate_initial_fisher, _fresh_fisher
from .schedule import resolve_shaped_rotation_schedule
from .transform import tensor_content_hash


FIXED_PANEL_ANGLES = {
    "panel_000": 0.0,
    "panel_015": 15.0,
    "panel_030": 30.0,
}
EDR_CONDITION = "edr_fasttrend_slowaction"
ClosedLoopConfig = RotatedDoubleLapConfig | RotatedSlowSingleLapConfig
FIXED_PI_BY_CONDITION = dict(DOUBLE_LAP_FIXED_PI)


def _is_edr_condition(condition: str) -> bool:
    return condition.startswith("edr_")


def _edr_condition(config: ClosedLoopConfig) -> str:
    matches = tuple(item for item in config.conditions if _is_edr_condition(item))
    if len(matches) != 1:
        raise ValueError("closed-loop protocol requires exactly one EDR condition")
    return matches[0]


@dataclasses.dataclass
class _LearnerState:
    schedule_kind: str
    condition: str
    model: nn.Module
    layout: ParameterLayout
    optimizer: torch.optim.Optimizer
    fisher: LowRankDiagonalFisher | None
    controller: ControllerState | None
    discounted: DiscountedRiskState | None
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
        default=Path("cache/mnist_experiment/rotated_mnist/phase5/double_lap"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, operation):
    _synchronize(device)
    started = time.perf_counter()
    value = operation()
    _synchronize(device)
    return value, time.perf_counter() - started


def _prefix(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


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


def _controller_config(
    config: ClosedLoopConfig,
    *,
    policy: str,
    fixed_pi: float,
) -> ControllerConfig:
    value = ControllerConfig(
        policy=policy,
        fixed_pi=fixed_pi,
        pi_min=config.controller.pi_min,
        pi_max=config.controller.pi_max,
        trend_half_life_p=config.controller.trend_half_life_degrees,
        trace_epsilon=config.controller.trace_epsilon,
        oracle_mode="none",
        reference_optimum_artifact=None,
        risk_metric="fisher",
        action_half_life_steps=(
            config.controller.action_half_life_steps
            if policy == "discounted_risk"
            else None
        ),
        damping=None,
        epsilon=None,
    )
    value.validate()
    return value


def _normalized_auc(rows: list[dict[str, Any]], field: str) -> float:
    x = torch.tensor(
        [row["observations_before_evaluation"] for row in rows],
        dtype=torch.float64,
    )
    y = torch.tensor([row[field] for row in rows], dtype=torch.float64)
    width = float(x[-1] - x[0])
    if width <= 0.0:
        raise ValueError("trajectory AUC requires increasing exposure")
    return float(torch.trapezoid(y, x=x) / width)


def _window_auc(
    rows: list[dict[str, Any]], field: str, start: int, stop: int
) -> float:
    values = rows[start : stop + 1]
    x = torch.tensor(
        [row["observations_before_evaluation"] for row in values],
        dtype=torch.float64,
    )
    y = torch.tensor([row[field] for row in values], dtype=torch.float64)
    return float(torch.trapezoid(y, x=x) / float(x[-1] - x[0]))


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs must have equal nontrivial length")
    left_centered = torch.tensor(left, dtype=torch.float64) - statistics.fmean(left)
    right_centered = torch.tensor(right, dtype=torch.float64) - statistics.fmean(right)
    denominator = float(
        torch.linalg.vector_norm(left_centered)
        * torch.linalg.vector_norm(right_centered)
    )
    if denominator == 0.0:
        return None
    return float((left_centered @ right_centered) / denominator)


def _evaluate_state(
    state: _LearnerState,
    panels: dict[float, tuple[Tensor, Tensor, str]],
    current_angle: float,
    config: ClosedLoopConfig,
    *,
    nine_prevalence: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    requested = {"current": current_angle, **FIXED_PANEL_ANGLES}
    by_angle: dict[float, dict[str, Any]] = {}
    started = time.perf_counter()
    for angle in dict.fromkeys(requested.values()):
        inputs, targets, _ = panels[angle]
        by_angle[angle] = evaluate_materialized_classifier(
            state.model,
            inputs,
            targets,
            batch_size=config.initialization.batch_size,
            device=device,
            dtype=dtype,
            calibration_bins=CALIBRATION_BINS,
            nine_prevalence=nine_prevalence,
        )
    _synchronize(device)
    state.evaluation_wall_seconds += time.perf_counter() - started
    output = {}
    for name, angle in requested.items():
        output.update(_prefix(name, by_angle[angle]))
    return output


def _make_states(
    initializer: nn.Module,
    initial_fisher: LowRankDiagonalFisher,
    config: ClosedLoopConfig,
    *,
    schedule_kind: str,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> dict[str, _LearnerState]:
    states = {}
    for condition in config.conditions:
        model = copy.deepcopy(initializer).to(device=device, dtype=training_dtype)
        layout = ParameterLayout.from_module(model)
        uses_ewc = condition != "current_only"
        states[condition] = _LearnerState(
            schedule_kind=schedule_kind,
            condition=condition,
            model=model,
            layout=layout,
            optimizer=build_optimizer(model, _learner_optimizer_config(config)),
            fisher=(
                initial_fisher.to(device=device, dtype=matrix_dtype)
                if uses_ewc
                else None
            ),
            controller=(
                ControllerState.initialize(
                    layout.total_numel,
                    config.data.initialization_size,
                    dtype=matrix_dtype,
                    device="cpu",
                )
                if uses_ewc
                else None
            ),
            discounted=(
                DiscountedRiskState() if _is_edr_condition(condition) else None
            ),
        )
    return states


def _apply_update(
    state: _LearnerState,
    inputs: Tensor,
    targets: Tensor,
    config: ClosedLoopConfig,
    *,
    step: int,
    parameter_before: Tensor,
    delta_degrees: float,
    lagged_speed: float | None,
    zero_fisher: DiagonalFisher,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
    seed_namespace: str,
) -> dict[str, Any]:
    if state.condition == "current_only":
        decision = None
        edr = None
        predictable_fisher = None
        proposal_fisher = zero_fisher
        applied_pi = 1.0
        fisher_mapping = None
    else:
        assert state.fisher is not None and state.controller is not None
        predictable_fisher = state.fisher
        if _is_edr_condition(state.condition):
            assert state.discounted is not None
            controller_config = _controller_config(
                config,
                policy="discounted_risk",
                fixed_pi=config.controller.cold_start_pi,
            )
            edr = decide_discounted_risk_controller(
                state.controller,
                state.discounted,
                controller_config,
                batch_size=config.data.samples_per_step,
                fisher=predictable_fisher,
                cold_start_steps=config.controller.cold_start_steps,
            )
            decision = edr.controller
        else:
            edr = None
            fixed_pi = FIXED_PI_BY_CONDITION[state.condition]
            controller_config = _controller_config(
                config, policy="fixed_unified", fixed_pi=fixed_pi
            )
            decision = decide_controller(
                state.controller,
                controller_config,
                batch_size=config.data.samples_per_step,
                fisher=predictable_fisher,
            )
        applied_pi = decision.applied_pi
        (fresh, fisher_seconds) = _timed(
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
                "blend_gain": applied_pi,
                "previous_trace": float(state.fisher.diagonal_vector().sum()),
                "fresh_trace": float(torch.trace(fresh)),
                "candidate_trace": float(state.fisher.diagonal_vector().sum()),
                "lanczos": None,
            }
        else:
            (update, update_seconds) = _timed(
                device,
                lambda: blend_archive_fisher(
                    state.fisher,
                    fresh,
                    blend_gain=applied_pi,
                    rank=config.fisher.rank,
                    lanczos_seed=derive_component_seed(
                        config.replica_seed,
                        f"{seed_namespace}_update_lanczos:"
                        f"schedule={state.schedule_kind}:"
                        f"condition={state.condition}:step={step}",
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
        proposal_fisher = state.fisher.to(dtype=training_dtype)

    (proposal, learner_seconds) = _timed(
        device,
        lambda: take_ewc_proposal(
            state.model,
            state.layout,
            inputs,
            targets,
            proposal_fisher,
            _learner_optimizer_config(config),
            state.optimizer,
            adaptation_weight=applied_pi,
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
        raise RuntimeError("closed-loop learner displacement is nonfinite")
    state.displacements.append(displacement)

    controller_mapping = None
    acceptance_mapping = None
    if decision is not None:
        assert state.controller is not None and predictable_fisher is not None
        acceptance = accept_controller_step(
            state.controller,
            decision,
            displacement.to(dtype=matrix_dtype),
            batch_size=config.data.samples_per_step,
            delta_p=delta_degrees,
            half_life_p=config.controller.trend_half_life_degrees,
            fisher=predictable_fisher,
        )
        state.controller = acceptance.state
        if edr is not None:
            state.discounted = edr.state
        controller_mapping = {
            **decision.mapping(extended=True),
            "decision_uses_current_batch": False,
            "cold_start_semantics": (
                "accepted_updates"
                if _is_edr_condition(state.condition)
                else "fixed"
            ),
            "cold_start_steps": (
                config.controller.cold_start_steps
                if _is_edr_condition(state.condition)
                else None
            ),
            "lagged_angular_speed_degrees_per_update": lagged_speed,
        }
        if edr is not None:
            controller_mapping.update(edr.mapping())
        acceptance_mapping = {
            "trend_gain": acceptance.gain,
            "residual_risk_energy": acceptance.residual_squared,
            "residual_euclidean_squared": acceptance.residual_euclidean_squared,
            "scale_observation": acceptance.scale_observation,
            "state_after": state.controller.scalar_mapping(
                config.controller.trace_epsilon, risk_metric="fisher"
            ),
        }
    return {
        "proposal": proposal.metrics_mapping(),
        "fisher_update": fisher_mapping,
        "controller": controller_mapping,
        "controller_acceptance": acceptance_mapping,
    }


def _condition_summary(
    state: _LearnerState,
    config: ClosedLoopConfig,
) -> dict[str, Any]:
    rows = state.rows
    transitions = config.rotation.transitions_per_arrow
    leg_count = len(config.rotation.knots_degrees) - 1
    leg_aucs = [
        _window_auc(
            rows,
            "current_environment_accuracy",
            leg * transitions,
            (leg + 1) * transitions,
        )
        for leg in range(leg_count)
    ]
    summary = {
        "schedule_kind": state.schedule_kind,
        "condition": state.condition,
        "environment_accuracy_auc": _normalized_auc(
            rows, "current_environment_accuracy"
        ),
        "environment_nll_auc": _normalized_auc(rows, "current_nll"),
        "expected_calibration_error_auc": _normalized_auc(
            rows, "current_expected_calibration_error"
        ),
        "upright_accuracy_auc": _normalized_auc(
            rows, "panel_000_environment_accuracy"
        ),
        "leg_environment_accuracy_aucs": leg_aucs,
        "first_ascent_environment_accuracy_auc": leg_aucs[0],
        "return_environment_accuracy_auc": leg_aucs[1],
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
        "lanczos_total_retry_count": sum(
            item["numerical_retry_count"] for item in state.lanczos_diagnostics
        ),
    }
    if leg_count >= 3:
        summary["second_ascent_environment_accuracy_auc"] = leg_aucs[2]
    if _is_edr_condition(state.condition):
        controller_rows = [row for row in rows if row["controller"] is not None]
        post_cold = [
            row
            for row in controller_rows
            if not row["controller"]["cold_start_active"]
        ]
        action_rows = post_cold if post_cold else controller_rows
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
        mean_speed = 30.0 / transitions
        fast = [
            action
            for action, speed in zip(speed_actions, speeds, strict=True)
            if speed > mean_speed
        ]
        slow = [
            action
            for action, speed in zip(speed_actions, speeds, strict=True)
            if speed <= mean_speed
        ]
        summary.update(
            {
                "cold_start_transition_count": len(controller_rows) - len(post_cold),
                "action_statistics_include_cold_start": not bool(post_cold),
                "action_min": min(actions),
                "action_mean": statistics.fmean(actions),
                "action_max": max(actions),
                "action_span": max(actions) - min(actions),
                "action_floor_fraction": statistics.fmean(
                    float(row["controller"]["lower_bound_active"])
                    for row in action_rows
                ),
                "lagged_speed_action_correlation": _pearson(speeds, speed_actions),
                "fast_region_action_mean": (
                    None if not fast else statistics.fmean(fast)
                ),
                "slow_region_action_mean": (
                    None if not slow else statistics.fmean(slow)
                ),
                "fast_minus_slow_action": (
                    None
                    if not fast or not slow
                    else statistics.fmean(fast) - statistics.fmean(slow)
                ),
            }
        )
    return summary


def _classify_retry(
    summaries: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    fixed_names = tuple(DOUBLE_LAP_FIXED_PI)
    comparisons = {}
    for schedule in DOUBLE_LAP_SCHEDULES:
        values = summaries[schedule]
        best_fixed = max(
            fixed_names,
            key=lambda name: values[name]["environment_accuracy_auc"],
        )
        edr = values[EDR_CONDITION]
        fixed05 = values["fixed_pi005"]
        comparisons[schedule] = {
            "best_fixed_condition": best_fixed,
            "best_fixed_environment_accuracy_auc": values[best_fixed][
                "environment_accuracy_auc"
            ],
            "edr_environment_accuracy_auc": edr["environment_accuracy_auc"],
            "edr_minus_fixed005_environment_accuracy_auc": (
                edr["environment_accuracy_auc"]
                - fixed05["environment_accuracy_auc"]
            ),
            "edr_gap_to_best_fixed_environment_accuracy_auc": (
                edr["environment_accuracy_auc"]
                - values[best_fixed]["environment_accuracy_auc"]
            ),
            "edr_minus_fixed005_environment_nll_auc": (
                edr["environment_nll_auc"] - fixed05["environment_nll_auc"]
            ),
            "edr_minus_fixed005_final_upright_accuracy": (
                edr["final_upright_environment_accuracy"]
                - fixed05["final_upright_environment_accuracy"]
            ),
            "edr_minus_fixed005_final_worst_class_recall": (
                edr["final_worst_class_recall"]
                - fixed05["final_worst_class_recall"]
            ),
        }
    interaction = (
        comparisons["sigmoid"]["edr_minus_fixed005_environment_accuracy_auc"]
        - comparisons["linear"]["edr_minus_fixed005_environment_accuracy_auc"]
    )
    sigmoid = summaries["sigmoid"][EDR_CONDITION]
    no_secondary_regression = (
        comparisons["sigmoid"]["edr_minus_fixed005_environment_nll_auc"] <= 0.10
        and comparisons["sigmoid"]["edr_minus_fixed005_final_upright_accuracy"]
        >= -0.02
        and comparisons["sigmoid"][
            "edr_minus_fixed005_final_worst_class_recall"
        ]
        >= -0.02
    )
    dynamic = (
        sigmoid["fast_minus_slow_action"] is not None
        and sigmoid["fast_minus_slow_action"] >= 0.005
        and comparisons["sigmoid"][
            "edr_minus_fixed005_environment_accuracy_auc"
        ]
        > 0.0
        and interaction > 0.0
        and no_secondary_regression
    )
    automatic = all(
        comparisons[schedule][
            "edr_gap_to_best_fixed_environment_accuracy_auc"
        ]
        >= -0.01
        for schedule in DOUBLE_LAP_SCHEDULES
    ) and no_secondary_regression
    diagnostic = (
        sigmoid["fast_minus_slow_action"] is not None
        and sigmoid["fast_minus_slow_action"] >= 0.005
        and sigmoid["lagged_speed_action_correlation"] is not None
        and sigmoid["lagged_speed_action_correlation"] > 0.0
        and no_secondary_regression
    )
    classification = (
        "dynamic_value"
        if dynamic
        else "automatic_selection_value"
        if automatic
        else "diagnostic_only"
        if diagnostic
        else "stop"
    )
    return {
        "classification": classification,
        "comparisons": comparisons,
        "schedule_by_policy_interaction": interaction,
        "no_material_secondary_regression": no_secondary_regression,
        "development_replica_only": True,
    }


def run_double_lap(
    config: ClosedLoopConfig,
    *,
    data_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    download: bool = False,
    resume: bool = False,
    run_store: RotatedDoubleLapRunStore | None = None,
    required_artifacts: tuple[str, ...] = DOUBLE_LAP_REQUIRED_ARTIFACTS,
    classifier: Callable[
        [dict[str, dict[str, dict[str, Any]]]], dict[str, Any]
    ] = _classify_retry,
    progress_description: str = "Phase 5 double lap",
    seed_namespace: str = "plan5_double_lap",
) -> Path:
    config.validate()
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.dtype)
    matrix_dtype = resolve_dtype(config.fisher.matrix_dtype)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    store = run_store or RotatedDoubleLapRunStore(output_root)
    session = store.begin(config, repo_root, resume=resume)
    total_started = time.perf_counter()
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=download)
    train_targets = dataset_targets(train_dataset)
    test_targets = dataset_targets(test_dataset)
    nine_prevalence = float((test_targets == 9).double().mean())
    partitions = partition_all_digit_mnist(
        train_targets,
        test_targets,
        config.data,
        replica_seed=config.replica_seed,
    )
    schedules = {
        kind: resolve_shaped_rotation_schedule(
            config.rotation,
            kind=kind,
            sigmoid_kappa=config.sigmoid_kappa,
        )
        for kind in config.schedule_kinds
    }
    streams = {
        kind: generate_rotated_stream(
            train_dataset,
            train_targets,
            partitions,
            schedules[kind],
            config.data,
            config.rotation,
            replica_seed=config.replica_seed,
        )
        for kind in config.schedule_kinds
    }
    linear_plan = streams["linear"][0]
    sigmoid_plan = streams["sigmoid"][0]
    stream_identity_paired = (
        linear_plan.observation_indices == sigmoid_plan.observation_indices
        and linear_plan.class_labels == sigmoid_plan.class_labels
    )
    if not stream_identity_paired:
        raise RuntimeError("closed-loop schedule streams are not canonically paired")
    session.write_json("partitions.json", partitions.to_mapping())
    session.write_json(
        "stream_plans.json",
        {kind: streams[kind][0].to_mapping() for kind in config.schedule_kinds},
    )
    session.write_torch(
        "stream_tensors.pt",
        {
            kind: {"inputs": streams[kind][1], "targets": streams[kind][2]}
            for kind in config.schedule_kinds
        },
    )

    initializer, layout = build_canonical_model(
        derive_component_seed(
            config.replica_seed, "plan5_model_initialization"
        ),
        device=device,
        dtype=training_dtype,
    )
    initialization = _fit_upright_initializer(
        initializer,
        train_dataset,
        test_dataset,
        partitions.initialization,
        partitions.evaluation,
        config,
        nine_prevalence=nine_prevalence,
        device=device,
        dtype=training_dtype,
    )
    initial_state = _state_dict_cpu(initializer)
    initial_state_hash = state_dict_hash(initial_state)
    session.write_json("initialization_metrics.json", initialization)
    dense_fisher, fisher_metrics = _estimate_initial_fisher(
        initializer,
        layout,
        train_dataset,
        partitions.reference,
        config,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    initial_approximation = approximate_low_rank_diagonal(
        lambda vector: dense_fisher @ vector,
        torch.diagonal(dense_fisher),
        rank=config.fisher.rank,
        seed=derive_component_seed(
            config.replica_seed, f"{seed_namespace}_initial_lanczos"
        ),
    )
    fisher_metrics["lanczos"] = initial_approximation.diagnostics.mapping()
    fisher_metrics["representation_storage_bytes"] = (
        initial_approximation.representation.storage_bytes()
    )
    session.write_torch(
        "initial_fisher.pt",
        {
            "dense": dense_fisher.cpu(),
            "representation": initial_approximation.representation.artifact_mapping(),
            "sample_indices": torch.as_tensor(
                partitions.reference[: config.fisher.initial_sample_size],
                dtype=torch.long,
            ),
        },
    )
    del dense_fisher

    base_inputs, base_targets = materialize_base_panel(
        test_dataset,
        partitions.evaluation,
        num_workers=config.runtime.num_workers,
    )
    fixed_panels = {
        angle: materialize_rotated_panel(
            base_inputs, base_targets, angle, config.rotation
        )
        for angle in set(FIXED_PANEL_ANGLES.values())
    }
    session.write_json(
        "evaluation_panel.json",
        {
            "evaluation_indices_hash": tensor_content_hash(
                torch.as_tensor(partitions.evaluation, dtype=torch.long)
            ),
            "sample_count": base_targets.numel(),
            "base_inputs_hash": tensor_content_hash(base_inputs),
            "base_targets_hash": tensor_content_hash(base_targets),
            "fixed_angle_panel_hashes": {
                f"{angle:.1f}": fixed_panels[angle][2]
                for angle in sorted(fixed_panels)
            },
        },
    )

    all_states = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_points = sum(schedule.num_points for schedule in schedules.values())
    progress = tqdm(total=total_points, desc=progress_description, unit="point")
    for schedule_kind in config.schedule_kinds:
        schedule = schedules[schedule_kind]
        plan, stream_inputs, stream_targets = streams[schedule_kind]
        states = _make_states(
            initializer,
            initial_approximation.representation,
            config,
            schedule_kind=schedule_kind,
            device=device,
            training_dtype=training_dtype,
            matrix_dtype=matrix_dtype,
        )
        panels = dict(fixed_panels)
        zero_fisher = DiagonalFisher(
            torch.zeros(layout.total_numel, device=device, dtype=training_dtype)
        )
        for step in range(schedule.num_points):
            angle = schedule.angles_degrees[step]
            if angle not in panels:
                panels[angle] = materialize_rotated_panel(
                    base_inputs, base_targets, angle, config.rotation
                )
            lagged_speed = (
                None
                if step == 0
                else abs(
                    schedule.angles_degrees[step]
                    - schedule.angles_degrees[step - 1]
                )
            )
            next_speed = (
                None
                if step == schedule.num_transitions
                else abs(
                    schedule.angles_degrees[step + 1]
                    - schedule.angles_degrees[step]
                )
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
                    "observations_before_evaluation": step
                    * config.data.samples_per_step,
                    "parameter_hash": tensor_content_hash(parameter_before.cpu()),
                    "proposal": None,
                    "fisher_update": None,
                    "controller": None,
                    "controller_acceptance": None,
                    **_evaluate_state(
                        state,
                        panels,
                        angle,
                        config,
                        nine_prevalence=nine_prevalence,
                        device=device,
                        dtype=training_dtype,
                    ),
                }
                if step < schedule.num_transitions:
                    inputs = stream_inputs[step].to(
                        device=device, dtype=training_dtype
                    )
                    targets = stream_targets[step].to(device=device)
                    row.update(
                        _apply_update(
                            state,
                            inputs,
                            targets,
                            config,
                            step=step,
                            parameter_before=parameter_before,
                            delta_degrees=float(next_speed),
                            lagged_speed=lagged_speed,
                            zero_fisher=zero_fisher,
                            device=device,
                            training_dtype=training_dtype,
                            matrix_dtype=matrix_dtype,
                            seed_namespace=seed_namespace,
                        )
                    )
                state.rows.append(row)
            if angle not in FIXED_PANEL_ANGLES.values():
                del panels[angle]
            progress.update(1)
        all_states[schedule_kind] = states
    progress.close()

    all_metrics = {
        schedule: {
            condition: all_states[schedule][condition].rows
            for condition in config.conditions
        }
        for schedule in config.schedule_kinds
    }
    trajectories = {}
    model_states = {}
    controller_states = {}
    summaries = {}
    for schedule in config.schedule_kinds:
        trajectories[schedule] = {}
        model_states[schedule] = {}
        controller_states[schedule] = {}
        summaries[schedule] = {}
        for condition, state in all_states[schedule].items():
            parameters = torch.stack(state.parameters)
            displacements = torch.stack(state.displacements)
            if not torch.equal(displacements, parameters[1:] - parameters[:-1]):
                raise RuntimeError("closed-loop displacement identity failed")
            trajectories[schedule][condition] = {
                "parameters": parameters,
                "displacements": displacements,
            }
            model_states[schedule][condition] = {
                "initial": initial_state,
                "final": _state_dict_cpu(state.model),
            }
            controller_states[schedule][condition] = {
                "controller": state.controller,
                "discounted": state.discounted,
            }
            summaries[schedule][condition] = _condition_summary(state, config)

    first_hashes = {
        all_metrics[schedule][condition][0]["parameter_hash"]
        for schedule in config.schedule_kinds
        for condition in config.conditions
    }
    first_metrics = {
        (
            all_metrics[schedule][condition][0]["current_environment_accuracy"],
            all_metrics[schedule][condition][0]["current_nll"],
        )
        for schedule in config.schedule_kinds
        for condition in config.conditions
    }
    all_lanczos = [
        fisher_metrics["lanczos"],
        *(
            item
            for schedule in config.schedule_kinds
            for state in all_states[schedule].values()
            for item in state.lanczos_diagnostics
        ),
    ]
    all_finite = _finite_tree(all_metrics) and _finite_tree(summaries)
    checks = {
        "all_finite": all_finite,
        "stream_identity_paired": stream_identity_paired,
        "shared_initial_parameter_hash": len(first_hashes) == 1,
        "shared_initial_metrics": len(first_metrics) == 1,
        "initial_model_state_hash": initial_state_hash,
        "all_trajectories_complete": all(
            len(all_metrics[schedule][condition]) == schedules[schedule].num_points
            for schedule in config.schedule_kinds
            for condition in config.conditions
        ),
        "all_decisions_predictable": all(
            row["controller"] is None
            or row["controller"]["decision_uses_current_batch"] is False
            for schedule in config.schedule_kinds
            for condition in config.conditions
            for row in all_metrics[schedule][condition]
        ),
        "edr_cold_start_counts": {
            schedule: summaries[schedule][_edr_condition(config)][
                "cold_start_transition_count"
            ]
            for schedule in config.schedule_kinds
        },
        "rank8_resolved": all(item["realized_rank"] == 8 for item in all_lanczos),
    }
    checks["gate_recommendation"] = (
        "go"
        if all(
            (
                checks["all_finite"],
                checks["stream_identity_paired"],
                checks["shared_initial_parameter_hash"],
                checks["shared_initial_metrics"],
                checks["all_trajectories_complete"],
                checks["all_decisions_predictable"],
                checks["rank8_resolved"],
                all(
                    count
                    == min(config.controller.cold_start_steps, schedules[name].num_transitions)
                    for name, count in checks["edr_cold_start_counts"].items()
                ),
            )
        )
        else "no_go"
    )
    classification = classifier(summaries)
    peak_cuda = (
        0
        if device.type != "cuda"
        else int(torch.cuda.max_memory_allocated(device))
    )
    session.write_json("trajectory_metrics.json", all_metrics)
    session.write_torch("trajectories.pt", trajectories)
    session.write_torch("model_states.pt", model_states)
    session.write_torch("controller_states.pt", controller_states)
    session.write_json("operational_checks.json", checks)
    session.write_json(
        "run_summary.json",
        {
            "config_hash": config.config_hash,
            "conditions": list(config.conditions),
            "schedule_kinds": list(config.schedule_kinds),
            "schedule_hashes": {
                name: schedules[name].content_hash for name in config.schedule_kinds
            },
            "stream_plan_hashes": {
                name: streams[name][0].content_hash
                for name in config.schedule_kinds
            },
            "partition_hash": partitions.content_hash,
            "parameter_count": layout.total_numel,
            "num_points": schedules["linear"].num_points,
            "num_transitions": schedules["linear"].num_transitions,
            "samples_per_step": config.data.samples_per_step,
            "total_online_observations": (
                schedules["linear"].num_transitions
                * config.data.samples_per_step
            ),
            "initial_fisher": fisher_metrics,
            "condition_summaries": summaries,
            "retry_classification": classification,
            "total_wall_time_seconds": time.perf_counter() - total_started,
            "peak_process_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            * 1024,
            "peak_cuda_memory_bytes": peak_cuda,
        },
    )
    return session.complete(required_artifacts)


def main() -> None:
    arguments = parse_arguments()
    config = load_double_lap_config(arguments.config)
    repo_root = Path(__file__).parents[2]
    path = run_double_lap(
        config,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        repo_root=repo_root,
        download=arguments.download,
        resume=arguments.resume,
    )
    print(
        json.dumps(
            {"run_id": config.run_id, "path": str(path), "status": "completed"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
