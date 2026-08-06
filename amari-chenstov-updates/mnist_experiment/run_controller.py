"""Run the Phase 8 unified adaptation-controller experiment."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from mnist_experiment.run_coupled import (
    _online_metrics,
    _path_divergence_rows,
    _prefix_mapping,
    _tensor_hash,
    _trajectory_hash,
)
from mnist_experiment.run_experiment import (
    _ReferenceOracle,
    _calculate_online_statistics,
    _clone_at_vector,
    _cpu_tree,
    _materialize_batch,
    _statistics_to_device,
    _synchronize,
)
from mnist_experiment.run_representations import _fixed_probes
from mnist_experiment.run_structured_coupled import _structured_tracking_metrics
from src.artifacts import RunStore
from src.config import ExperimentConfig, load_config
from src.controller import (
    ControllerState,
    OracleControllerInput,
    accept_controller_step,
    decide_controller,
)
from src.coupled_trajectory import DenseFisherTracker, dense_fisher_metrics
from src.ewc import build_optimizer, mixture_ewc_strength, take_ewc_proposal
from src.initialization import evaluate_classifier, load_replica_bundle_for_config
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import configure_torch_runtime, resolve_device, resolve_dtype
from src.parameters import ParameterLayout
from src.reference import ReferenceFisherStore
from src.reference_optimum import (
    ReferenceOptimumPath,
    build_reference_optimum_path,
)
from src.representations import DiagonalFisher, LowRankDiagonalFisher
from src.seeding import derive_component_seed
from src.structured_trajectory import (
    DiagonalFisherTracker,
    LowRankDiagonalFisherTracker,
)

PHASE8_METRIC_SCHEMA_VERSION = 6
PHASE8_ARTIFACT_SCHEMA_VERSION = 4


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_config(config: ExperimentConfig) -> int:
    if config.schema_version not in {9, 10}:
        raise ValueError(
            "corrected Phase 8 controller runs require schema version 9 or 10"
        )
    expected_metric_schema = 6 if config.schema_version >= 10 else 5
    if config.metric_schema_version != expected_metric_schema:
        raise ValueError(
            "corrected Phase 8 metric schema must be version "
            f"{expected_metric_schema}"
        )
    if config.artifact_schema_version != PHASE8_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("corrected Phase 8 artifact schema must be version 4")
    if config.estimator.ema_gain is not None:
        raise ValueError("Phase 8 forbids an independent estimator.ema_gain")
    if config.estimator.representation != "low_rank_diagonal":
        raise ValueError("Phase 8 requires the selected low-rank representation")
    if (
        config.estimator.low_rank_grid is None
        or config.estimator.low_rank_grid != [0, config.estimator.low_rank]
        or config.estimator.low_rank is None
        or config.estimator.low_rank < 1
    ):
        raise ValueError("Phase 8 low_rank_grid must contain zero and one rank")
    if config.estimator.fresh_fisher_cadence is None:
        raise ValueError("Phase 8 requires fresh_fisher_cadence")
    if config.schema_version >= 10 and config.estimator.controller_methods is None:
        raise ValueError(
            "schema-v10 Phase 8 requires explicit estimator.controller_methods"
        )
    if config.estimator.ridge_half_life_steps is None:
        raise ValueError("Phase 8 requires directional-ridge settings")
    if config.controller.oracle_mode not in {"reference_path", "diagnostic"}:
        raise ValueError("Phase 8 requires reference-path oracle diagnostics")
    return config.estimator.low_rank


def _cosine(left: Tensor, right: Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float((left @ right) / denominator)


def _relative_scalar_error(estimate: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return abs(estimate - reference) / abs(reference)


@dataclasses.dataclass(frozen=True)
class _ConditionRun:
    method: str
    trajectory_hash: str
    parameters: Tensor
    displacements: Tensor
    rows: tuple[dict[str, Any], ...]
    references: tuple[dict[str, Any], ...]
    checkpoints: dict[str, Any]
    controller_states: dict[str, Any]
    optimizer_state: dict[str, Any]
    dependence: dict[str, Any]
    elapsed_seconds: float


def _phase8_methods(
    config: ExperimentConfig, selected_rank: int
) -> tuple[str, ...]:
    requested = config.estimator.controller_methods or [
        "dense",
        "diagonal",
        "low_rank_diagonal",
    ]
    names = {
        "dense": "dense_ridge_full",
        "diagonal": "diagonal_ridge_full",
        "low_rank_diagonal": f"low_rank_diagonal_r{selected_rank}",
    }
    return tuple(names[method] for method in requested)


def _make_tracker(
    method: str,
    selected_rank: int,
    initial_fisher: Tensor,
    config: ExperimentConfig,
):
    arguments = {
        "initial_fisher": initial_fisher,
        "ema_gain": None,
        "ridge_half_life_steps": config.estimator.ridge_half_life_steps,
        "ridge_amplitude_epsilon": config.estimator.ridge_amplitude_epsilon,
        "ridge_coherence_threshold": config.estimator.ridge_coherence_threshold,
    }
    if method == "dense_ridge_full":
        return DenseFisherTracker(
            "ridge_full_lfu",
            fresh_fisher_cadence=config.estimator.fresh_fisher_cadence,
            **arguments,
        )
    if method == "diagonal_ridge_full":
        return DiagonalFisherTracker(**arguments)
    if method == f"low_rank_diagonal_r{selected_rank}":
        return LowRankDiagonalFisherTracker(rank=selected_rank, **arguments)
    raise ValueError(f"unsupported Phase 8 method: {method}")


def _oracle_inputs(
    path: ReferenceOptimumPath,
    *,
    matrix_dtype: torch.dtype,
) -> tuple[tuple[OracleControllerInput, ...], tuple[dict[str, Any], ...]]:
    values = []
    rows = []
    for step, (p_value, parameter) in enumerate(
        zip(path.p_values, path.parameters, strict=True)
    ):
        displacement = (
            path.displacements[step]
            if step < path.displacements.shape[0]
            else torch.zeros_like(parameter)
        )
        values.append(
            OracleControllerInput(
                displacement=displacement.to(dtype=matrix_dtype),
            )
        )
        rows.append(
            {
                "step": step,
                "p": p_value,
                "displacement_norm": float(torch.linalg.vector_norm(displacement)),
            }
        )
    return tuple(values), tuple(rows)


def _run_condition(
    method: str,
    selected_rank: int,
    config: ExperimentConfig,
    initial_model: nn.Module,
    initial_template: nn.Module,
    train_dataset: Dataset,
    evaluation_dataset: Dataset,
    stream_plan,
    oracle: _ReferenceOracle,
    oracle_path: ReferenceOptimumPath,
    oracle_inputs: tuple[OracleControllerInput, ...],
    probes: Tensor,
    *,
    device: torch.device,
    training_dtype: torch.dtype,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> _ConditionRun:
    model = copy.deepcopy(initial_model).to(device=device, dtype=training_dtype)
    layout = ParameterLayout.from_module(model)
    optimizer = build_optimizer(model, config.optimizer)
    initial_vector = layout.flatten_module(model, detach=True)
    initial_record = oracle.estimate(0, stream_plan.p_values[0], initial_vector)
    initial_fisher = initial_record.matrix.to(device=device, dtype=matrix_dtype)
    tracker = _make_tracker(method, selected_rank, initial_fisher, config)
    controller_state = ControllerState.initialize(
        layout.total_numel,
        config.data.initialization_size,
        dtype=matrix_dtype,
        device="cpu",
    )
    checkpoint_indices = {
        0,
        len(stream_plan.p_values) // 2,
        len(stream_plan.p_values) - 1,
    }
    parameters: list[Tensor] = []
    displacements: list[Tensor] = []
    rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    controller_states: dict[str, Any] = {}
    lagged_displacement = torch.zeros_like(initial_vector).cpu()
    previous_reference = None
    started = time.perf_counter()

    progress = tqdm(
        stream_plan.p_values,
        desc=f"{method} trajectory",
        unit="step",
        leave=False,
    )
    for step, p_value in enumerate(progress):
        state_before = controller_state
        decision = decide_controller(
            state_before,
            config.controller,
            batch_size=config.data.samples_per_step,
            oracle=oracle_inputs[step],
        )
        parameter_before = layout.flatten_module(model, detach=True)
        parameters.append(parameter_before.cpu())
        parameter_hash = _tensor_hash(parameter_before)
        record = oracle.estimate(step, p_value, parameter_before)
        reference = record.matrix.to(device=device, dtype=matrix_dtype)

        derivative_model, derivative_layout = _clone_at_vector(
            initial_template,
            parameter_before,
            device=device,
            dtype=derivative_dtype,
        )
        derivative_inputs, derivative_targets = _materialize_batch(
            train_dataset,
            stream_plan.observation_indices[step],
            device=device,
            dtype=derivative_dtype,
        )
        expected_targets = torch.as_tensor(
            stream_plan.class_labels[step], dtype=torch.long, device=device
        )
        if not torch.equal(derivative_targets, expected_targets):
            raise RuntimeError("paired stream labels do not match the dataset")
        statistics_cpu = _calculate_online_statistics(
            derivative_model,
            derivative_layout,
            derivative_inputs,
            derivative_targets,
            lagged_displacement.to(device=device, dtype=derivative_dtype),
            matrix_dtype=matrix_dtype,
            include_hvp=step > 0,
            device=device,
        )
        statistics = _statistics_to_device(
            [statistics_cpu], device, matrix_dtype
        )[0]
        direction = lagged_displacement.to(device=device, dtype=matrix_dtype)
        update_started = time.perf_counter()
        if method == "dense_ridge_full":
            update = tracker.update(
                step,
                statistics,
                direction,
                reference,
                blend_gain=decision.applied_pi,
            )
            fisher = update.estimate
            tracking = dense_fisher_metrics(update, reference, direction)
            representation_mapping = {"kind": "dense", "matrix": update.estimate}
            update_diagnostics = {
                "projection": dataclasses.asdict(update.projection),
                "ridge": update.ridge_metrics,
            }
            ridge_state = update.ridge_state
        elif method == "diagonal_ridge_full":
            update = tracker.update(
                step,
                statistics,
                direction,
                blend_gain=decision.applied_pi,
            )
            fisher = update.representation
            tracking = _structured_tracking_metrics(
                fisher, reference, probes, direction
            )
            representation_mapping = fisher.artifact_mapping()
            update_diagnostics = {
                "projection": dataclasses.asdict(update.projection),
                "ridge": update.ridge_metrics,
            }
            ridge_state = update.ridge_state
        else:
            update = tracker.update(
                step,
                statistics,
                direction,
                lanczos_seed=derive_component_seed(
                    config.replica_seed,
                    f"phase8_controller_lanczos:{method}:step={step}",
                ),
                blend_gain=decision.applied_pi,
            )
            fisher = update.representation
            tracking = _structured_tracking_metrics(
                fisher, reference, probes, direction
            )
            representation_mapping = fisher.artifact_mapping()
            update_diagnostics = {
                "candidate_minimum_eigenvalue": (
                    update.candidate_minimum_eigenvalue
                ),
                "candidate_negative_eigenvalue_count": (
                    update.candidate_negative_eigenvalue_count
                ),
                "lanczos": update.lanczos.mapping(),
                "ridge": update.ridge_metrics,
            }
            ridge_state = update.ridge_state
        if update.blend_gain != decision.applied_pi:
            raise RuntimeError("Fisher tracker consumed the wrong pi")
        _synchronize(device)
        update_elapsed = time.perf_counter() - update_started

        before_evaluation = evaluate_classifier(
            model,
            evaluation_dataset,
            batch_size=config.initialization.batch_size,
            device=device,
            dtype=training_dtype,
            num_workers=config.initialization.num_workers,
        )
        proposal_mapping = None
        acceptance_mapping = None
        after_evaluation = before_evaluation
        accepted_displacement = None
        state_after = state_before
        if step + 1 < len(stream_plan.p_values):
            if decision.applied_pi == 0.0:
                accepted_displacement = torch.zeros_like(parameter_before).cpu()
                proposal_mapping = {
                    "inner_steps": config.optimizer.inner_steps,
                    "optimizer_iterations": 0,
                    "optimizer_function_evaluations": 0,
                    "stopping_reason": "hard_freeze",
                    "adaptation_weight": 0.0,
                    "effective_ewc_strength": None,
                    "post_optimization_scaling_applied": False,
                    "hard_freeze": True,
                    "accepted_displacement_norm": 0.0,
                }
            else:
                training_inputs, training_targets = _materialize_batch(
                    train_dataset,
                    stream_plan.observation_indices[step],
                    device=device,
                    dtype=training_dtype,
                )
                proposal_fisher = fisher.to(dtype=training_dtype)
                proposal = take_ewc_proposal(
                    model,
                    layout,
                    training_inputs,
                    training_targets,
                    proposal_fisher,
                    config.optimizer,
                    optimizer,
                    adaptation_weight=decision.applied_pi,
                )
                accepted_displacement = (
                    layout.flatten_module(model, detach=True) - parameter_before
                ).cpu()
                agreement_error = float(
                    torch.linalg.vector_norm(
                        accepted_displacement - proposal.displacement
                    )
                )
                tolerance = 10.0 * torch.finfo(accepted_displacement.dtype).eps * max(
                    float(torch.linalg.vector_norm(accepted_displacement)), 1.0
                )
                if agreement_error > tolerance:
                    raise RuntimeError(
                        "proposal displacement was altered after optimization"
                    )
                expected_strength = mixture_ewc_strength(
                    decision.applied_pi,
                    multiplier=config.optimizer.ewc_strength,
                )
                if proposal.effective_ewc_strength != expected_strength:
                    raise RuntimeError("EWC proposal consumed the wrong pi")
                proposal_mapping = {
                    **proposal.metrics_mapping(),
                    "accepted_displacement_norm": proposal.displacement_norm,
                    "accepted_displacement_agreement_error": agreement_error,
                    "accepted_displacement_agreement_tolerance": tolerance,
                    "post_optimization_scaling_applied": False,
                    "hard_freeze": False,
                }
            displacements.append(accepted_displacement)
            lagged_displacement = accepted_displacement
            delta_p = stream_plan.p_values[step + 1] - p_value
            acceptance = accept_controller_step(
                state_before,
                decision,
                accepted_displacement.to(dtype=matrix_dtype),
                batch_size=config.data.samples_per_step,
                delta_p=delta_p,
                half_life_p=config.controller.trend_half_life_p,
                oracle_displacement=oracle_inputs[step].displacement,
            )
            controller_state = acceptance.state
            state_after = controller_state
            acceptance_mapping = {
                "gain": acceptance.gain,
                "residual_norm": math.sqrt(acceptance.residual_squared),
                "residual_squared": acceptance.residual_squared,
                "scale_observation": acceptance.scale_observation,
                "residual_to_scale_ratio": (
                    acceptance.residual_squared / acceptance.scale_observation
                    if acceptance.scale_observation > 0.0
                    else None
                ),
                "normalized_displacement_norm": (
                    None
                    if acceptance.normalized_displacement is None
                    else float(
                        torch.linalg.vector_norm(
                            acceptance.normalized_displacement
                        )
                    )
                ),
                "oracle_residual_norm": (
                    None
                    if acceptance.oracle_residual is None
                    else float(torch.linalg.vector_norm(acceptance.oracle_residual))
                ),
                "oracle_residual_to_scale_ratio": (
                    float(acceptance.oracle_residual.square().sum())
                    / acceptance.scale_observation
                    if acceptance.scale_observation > 0.0
                    and acceptance.oracle_residual is not None
                    else None
                ),
            }
            after_evaluation = evaluate_classifier(
                model,
                evaluation_dataset,
                batch_size=config.initialization.batch_size,
                device=device,
                dtype=training_dtype,
                num_workers=config.initialization.num_workers,
            )

        oracle_displacement = oracle_inputs[step].displacement
        trend = state_before.trend
        trend_error = trend - oracle_displacement
        parameter_error = parameter_before.cpu().to(dtype=matrix_dtype) - (
            oracle_path.parameters[step].to(dtype=matrix_dtype)
        )
        controller_states[str(step)] = {
            "pre": {
                **state_before.scalar_mapping(config.controller.trace_epsilon),
                "trend": state_before.trend,
            },
            "decision": decision.mapping(),
            "post": {
                **state_after.scalar_mapping(config.controller.trace_epsilon),
                "trend": state_after.trend,
            },
            "accepted_displacement": accepted_displacement,
            "normalized_displacement": (
                None
                if acceptance_mapping is None
                else acceptance.normalized_displacement
            ),
            "plugin_residual": (
                None if acceptance_mapping is None else acceptance.residual
            ),
            "oracle_residual": (
                None
                if acceptance_mapping is None
                else acceptance.oracle_residual
            ),
            "oracle_displacement": oracle_displacement,
            "parameter_error_to_oracle": parameter_error,
        }
        online = _online_metrics(
            statistics, reference, direction, previous_reference
        )
        rows.append(
            {
                "method": method,
                "step": step,
                "p": p_value,
                "representation": representation_mapping["kind"],
                "requested_rank": (
                    selected_rank
                    if method.startswith("low_rank_diagonal")
                    else 0 if method == "diagonal_ridge_full" else None
                ),
                "parameter_hash": parameter_hash,
                "parameter_norm": float(torch.linalg.vector_norm(parameter_before)),
                "reference_cache_digest": record.cache_digest,
                "reference_plan_hash": record.plan.content_hash,
                "controller": decision.mapping(),
                "controller_state_pre": state_before.scalar_mapping(
                    config.controller.trace_epsilon
                ),
                "controller_state_post": state_after.scalar_mapping(
                    config.controller.trace_epsilon
                ),
                "acceptance": acceptance_mapping,
                "proposal": proposal_mapping,
                "fisher_blend_pi": update.blend_gain,
                "ewc_pi": (
                    None if proposal_mapping is None else decision.applied_pi
                ),
                "same_pi_consumed": (
                    proposal_mapping is None
                    or update.blend_gain == decision.applied_pi
                ),
                "trend_oracle_norm_error": float(
                    torch.linalg.vector_norm(trend_error)
                ),
                "trend_oracle_cosine": _cosine(trend, oracle_displacement),
                "oracle_displacement_norm": float(
                    torch.linalg.vector_norm(oracle_displacement)
                ),
                "parameter_squared_error_to_oracle": float(
                    parameter_error @ parameter_error
                ),
                "trace_relative_error_to_oracle_residual": (
                    None
                    if decision.oracle_trace_estimate is None
                    else _relative_scalar_error(
                        decision.trace_estimate,
                        decision.oracle_trace_estimate,
                    )
                ),
                "oracle_residual_trace_estimate": (
                    decision.oracle_trace_estimate
                ),
                "update_elapsed_seconds": update_elapsed,
                "update_diagnostics": update_diagnostics,
                **_prefix_mapping("before", before_evaluation),
                **_prefix_mapping("after", after_evaluation),
                **online,
                **tracking,
            }
        )
        reference_rows.append(
            {
                "method": method,
                "step": step,
                "p": p_value,
                "parameter_hash": parameter_hash,
                **record.metrics_mapping(step),
            }
        )
        if step in checkpoint_indices:
            checkpoints[str(step)] = {
                "parameter": parameter_before,
                "reference": reference,
                "representation": representation_mapping,
                "direct_fisher": statistics.estimate.fisher,
                "amari_chentsov": statistics.estimate.amari_chentsov,
                "residual": statistics.estimate.residual,
                "lagged_direction": direction,
                "accepted_displacement": accepted_displacement,
                "ridge_state": ridge_state,
                "update_diagnostics": update_diagnostics,
                "controller_decision": decision.mapping(),
                "parameter_hash": parameter_hash,
            }
        previous_reference = reference

    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    if not torch.equal(
        displacement_tensor, parameter_tensor[1:] - parameter_tensor[:-1]
    ):
        raise RuntimeError("Phase 8 displacements are inconsistent")
    content_hash = _trajectory_hash(
        method, parameter_tensor, displacement_tensor, stream_plan.content_hash
    )
    dependence = _residual_dependence_diagnostics(
        rows,
        controller_states,
        p_values=stream_plan.p_values,
        half_life_p=config.controller.trend_half_life_p,
    )
    return _ConditionRun(
        method=method,
        trajectory_hash=content_hash,
        parameters=parameter_tensor,
        displacements=displacement_tensor,
        rows=tuple(rows),
        references=tuple(reference_rows),
        checkpoints=checkpoints,
        controller_states=controller_states,
        optimizer_state=optimizer.state_dict(),
        dependence=dependence,
        elapsed_seconds=time.perf_counter() - started,
    )


def _oracle_path_checks(path: ReferenceOptimumPath) -> dict[str, Any]:
    turning = []
    for index in range(1, path.displacements.shape[0]):
        previous = path.displacements[index - 1]
        current = path.displacements[index]
        cosine = _cosine(previous, current)
        turning.append(
            {
                "step": index,
                "cosine": cosine,
                "angle_radians": None if cosine is None else math.acos(
                    min(1.0, max(-1.0, cosine))
                ),
                "second_difference_norm": float(
                    torch.linalg.vector_norm(current - previous)
                ),
            }
        )
    return {
        "turning": turning,
        "optimization": list(path.rows),
        "independent_fits": list(path.fit_rows),
        "displacement_convergence": list(path.displacement_diagnostics),
    }


def _lag_one_scalar_correlation(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    mean = sum(values) / len(values)
    left = [value - mean for value in values[:-1]]
    right = [value - mean for value in values[1:]]
    denominator = math.sqrt(
        sum(value * value for value in left)
        * sum(value * value for value in right)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def _vector_lag_one_correlation(values: list[Tensor]) -> float | None:
    if len(values) < 3:
        return None
    stacked = torch.stack(values)
    centered = stacked - stacked.mean(dim=0)
    left = centered[:-1]
    right = centered[1:]
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.sum(left * right) / denominator)


def _coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean == 0.0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / abs(mean)


def _residual_dependence_diagnostics(
    rows: list[dict[str, Any]],
    controller_states: dict[str, Any],
    *,
    p_values,
    half_life_p: float,
) -> dict[str, Any]:
    plugin_residuals: list[Tensor] = []
    oracle_residuals: list[Tensor] = []
    plugin_ratios: list[float] = []
    oracle_ratios: list[float] = []
    for step, row in enumerate(rows[:-1]):
        state = controller_states[str(step)]
        accepted = state["accepted_displacement"]
        if accepted is None:
            continue
        pi = float(state["decision"]["applied_pi"])
        if pi == 0.0:
            continue
        plugin_residuals.append(accepted - pi * state["pre"]["trend"])
        oracle_residuals.append(accepted - pi * state["oracle_displacement"])
        acceptance = row["acceptance"]
        if acceptance["residual_to_scale_ratio"] is not None:
            plugin_ratios.append(acceptance["residual_to_scale_ratio"])
        if acceptance["oracle_residual_to_scale_ratio"] is not None:
            oracle_ratios.append(acceptance["oracle_residual_to_scale_ratio"])

    increments = [
        float(right - left)
        for left, right in zip(p_values[:-1], p_values[1:], strict=True)
    ]
    mean_increment = sum(increments) / len(increments) if increments else 1.0
    window_steps = max(2, math.ceil(half_life_p / mean_increment))

    def ratio_windows(values: list[float]) -> list[dict[str, Any]]:
        return [
            {
                "start": start,
                "stop": start + window_steps,
                "mean": sum(values[start : start + window_steps]) / window_steps,
                "coefficient_of_variation": _coefficient_of_variation(
                    values[start : start + window_steps]
                ),
            }
            for start in range(0, len(values) - window_steps + 1)
        ]

    return {
        "trend_removed": True,
        "residual_semantics": "u_minus_pi_times_predictable_drift",
        "residual_count": len(plugin_residuals),
        "half_life_window_steps": window_steps,
        "plugin_vector_lag_one_correlation": _vector_lag_one_correlation(
            plugin_residuals
        ),
        "oracle_vector_lag_one_correlation": _vector_lag_one_correlation(
            oracle_residuals
        ),
        "plugin_squared_norm_lag_one_correlation": _lag_one_scalar_correlation(
            [float(value.square().sum()) for value in plugin_residuals]
        ),
        "oracle_squared_norm_lag_one_correlation": _lag_one_scalar_correlation(
            [float(value.square().sum()) for value in oracle_residuals]
        ),
        "plugin_ratio_coefficient_of_variation": _coefficient_of_variation(
            plugin_ratios
        ),
        "oracle_ratio_coefficient_of_variation": _coefficient_of_variation(
            oracle_ratios
        ),
        "plugin_half_life_windows": ratio_windows(plugin_ratios),
        "oracle_half_life_windows": ratio_windows(oracle_ratios),
    }


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    selected_rank = _validate_config(config)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=(
            config.runtime.deterministic_algorithms
            and config.runtime.device in {"cuda", "auto"}
        ),
    )
    cache_parent = Path(config.cache_root).parent
    data_root = arguments.data_root or cache_parent / "datasets"
    replica_root = arguments.replica_root or cache_parent / "replicas"
    reference_root = arguments.reference_root or cache_parent / "references"
    output_root = arguments.output_root or Path(config.cache_root)
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.training_dtype)
    derivative_dtype = resolve_dtype(config.reference.derivative_dtype)
    matrix_dtype = resolve_dtype(config.runtime.matrix_dtype)
    session = RunStore(output_root).begin(
        config, Path(__file__).parents[1], resume=arguments.resume
    )
    artifact_version = config.artifact_schema_version
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=False)
    train_targets = dataset_targets(train_dataset)
    loaded = load_replica_bundle_for_config(replica_root, config, device=device)
    evaluation_dataset = Subset(test_dataset, loaded.partitions.evaluation)
    initial_template = copy.deepcopy(loaded.model).cpu()
    oracle = _ReferenceOracle(
        config,
        initial_template,
        train_dataset,
        train_targets,
        loaded.partitions,
        ReferenceFisherStore(reference_root),
        device=device,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
    )
    oracle_path_file = session.path / "phase8_reference_optimum.pt"
    oracle_path_provenance: dict[str, Any]
    if oracle_path_file.is_file():
        stored_oracle = torch.load(
            oracle_path_file,
            map_location="cpu",
            weights_only=False,
        )
        if stored_oracle.get("schema_version") != artifact_version:
            raise RuntimeError("checkpointed oracle path schema is incompatible")
        oracle_path = ReferenceOptimumPath.from_artifact_mapping(
            stored_oracle["path"]
        )
        if oracle_path.p_values != loaded.stream_plan.p_values:
            raise RuntimeError("checkpointed oracle path uses a different p grid")
        oracle_path_provenance = dict(stored_oracle["provenance"])
        oracle_inputs, oracle_reference_rows = _oracle_inputs(
            oracle_path, matrix_dtype=matrix_dtype
        )
        print("phase8 reference-optimum path resumed from checkpoint", flush=True)
    elif config.controller.reference_optimum_artifact is not None:
        source_path = Path(config.controller.reference_optimum_artifact)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"reference-optimum artifact does not exist: {source_path}"
            )
        stored_source = torch.load(
            source_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(stored_source, dict) or "path" not in stored_source:
            raise RuntimeError("reference-optimum source has an invalid envelope")
        oracle_path = ReferenceOptimumPath.from_artifact_mapping(
            stored_source["path"]
        )
        if oracle_path.p_values != loaded.stream_plan.p_values:
            raise RuntimeError("external oracle path uses a different p grid")
        if oracle_path.parameters.shape[1] != loaded.layout.total_numel:
            raise RuntimeError("external oracle path uses a different model layout")
        expected_partition_hash = loaded.partitions.content_hash
        if any(
            plan.partition_hash != expected_partition_hash
            for plan in oracle_path.sample_plans
        ):
            raise RuntimeError("external oracle path uses different data partitions")
        oracle_inputs, oracle_reference_rows = _oracle_inputs(
            oracle_path, matrix_dtype=matrix_dtype
        )
        oracle_path_provenance = {
            "mode": "external_artifact",
            "source_path": str(source_path),
            "source_envelope_schema_version": stored_source.get("schema_version"),
            "content_hash": oracle_path.content_hash,
        }
        session.write_torch(
            "phase8_reference_optimum.pt",
            _cpu_tree(
                {
                    "schema_version": artifact_version,
                    "path": oracle_path.artifact_mapping(),
                    "path_displacements": oracle_reference_rows,
                    "provenance": oracle_path_provenance,
                }
            ),
        )
        print(
            "phase8 reference-optimum path copied from validated artifact",
            flush=True,
        )
    else:
        print("phase8 reference-optimum path started", flush=True)
        oracle_path = build_reference_optimum_path(
            loaded.model,
            train_dataset,
            train_targets,
            loaded.partitions,
            loaded.stream_plan.p_values,
            config,
            device=device,
            dtype=training_dtype,
        )
        oracle_inputs, oracle_reference_rows = _oracle_inputs(
            oracle_path, matrix_dtype=matrix_dtype
        )
        oracle_path_provenance = {
            "mode": "built_in_run",
            "source_path": None,
            "source_envelope_schema_version": None,
            "content_hash": oracle_path.content_hash,
        }
        session.write_torch(
            "phase8_reference_optimum.pt",
            _cpu_tree(
                {
                    "schema_version": artifact_version,
                    "path": oracle_path.artifact_mapping(),
                    "path_displacements": oracle_reference_rows,
                    "provenance": oracle_path_provenance,
                }
            ),
        )
        print(
            "phase8 reference-optimum path completed and checkpointed",
            flush=True,
        )
    probes = _fixed_probes(
        loaded.layout.total_numel,
        count=8,
        seed=derive_component_seed(
            config.replica_seed, "phase8_controller_fixed_probes"
        ),
        device=device,
        dtype=matrix_dtype,
    )
    methods = _phase8_methods(config, selected_rank)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    conditions = []
    for method in methods:
        print(f"phase8 controller condition started: {method}", flush=True)
        condition = _run_condition(
            method,
            selected_rank,
            config,
            loaded.model,
            initial_template,
            train_dataset,
            evaluation_dataset,
            loaded.stream_plan,
            oracle,
            oracle_path,
            oracle_inputs,
            probes,
            device=device,
            training_dtype=training_dtype,
            derivative_dtype=derivative_dtype,
            matrix_dtype=matrix_dtype,
        )
        conditions.append(condition)
        print(
            f"phase8 controller condition completed: {method} "
            f"({condition.elapsed_seconds:.3f}s)",
            flush=True,
        )
    total_elapsed = time.perf_counter() - started
    path_divergence = _path_divergence_rows(
        conditions, loaded.stream_plan.p_values
    )
    initial_hashes = {_tensor_hash(row.parameters[0]) for row in conditions}
    if len(initial_hashes) != 1:
        raise RuntimeError("paired Phase 8 conditions did not share initialization")

    trajectory_artifact = {
        "schema_version": artifact_version,
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "p_values": loaded.stream_plan.p_values,
        "observation_indices": loaded.stream_plan.observation_indices,
        "class_labels": loaded.stream_plan.class_labels,
        "parameter_layout": loaded.layout.metadata(),
        "conditions": {
            row.method: {
                "content_hash": row.trajectory_hash,
                "parameters": row.parameters,
                "displacements": row.displacements,
                "optimizer_state": row.optimizer_state,
            }
            for row in conditions
        },
    }
    controller_artifact = {
        "schema_version": artifact_version,
        "policy": config.controller.policy,
        "conditions": {
            row.method: row.controller_states for row in conditions
        },
    }
    checkpoint_artifact = {
        "schema_version": artifact_version,
        "checkpoint_policy": "initial, midpoint, and final indices",
        "conditions": {row.method: row.checkpoints for row in conditions},
    }
    metrics = {
        "phase8_metric_schema_version": config.metric_schema_version,
        "run_kind": "unified_controller",
        "replica_bundle_id": loaded.metadata["bundle_id"],
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "parameter_count": loaded.layout.total_numel,
        "methods": list(methods),
        "policy": config.controller.policy,
        "controller_config": dataclasses.asdict(config.controller),
        "unified_pi_contract": True,
        "post_optimization_scaling": False,
        "fisher_inverse_used": False,
        "trace_estimator": "accepted_displacement_residual_moments",
        "controller_displacement_law": "u_approx_pi_times_drift_plus_noise",
        "trend_estimand": "environmental_parameter_displacement",
        "trend_observation": "accepted_displacement_divided_by_pi",
        "covariance_residual": "u_minus_pi_times_predictable_drift",
        "oracle_covariance_residual": "u_minus_pi_times_oracle_drift",
        "ewc_optimality_diagnostic": "final_objective_gradient",
        "optimizer_budget_role": "fixed_compute_budget_not_convergence_claim",
        "optimizer_accounting": "structured_iterations_and_function_evaluations",
        "oracle_path_hash": oracle_path.content_hash,
        "oracle_path_provenance": oracle_path_provenance,
        "condition_steps": [
            item for row in conditions for item in row.rows
        ],
        "references": [
            item for row in conditions for item in row.references
        ],
        "oracle_path_checks": _oracle_path_checks(oracle_path),
        "residual_dependence": {
            row.method: row.dependence for row in conditions
        },
        "oracle_convergence_contract": {
            "minimum_fisher_chunks": config.reference.convergence_min_chunks,
            "minimum_independent_fits": config.reference.calibration_min_fits,
            "maximum_independent_fits": config.reference.calibration_max_fits,
            "sigma": config.reference.convergence_sigma,
            "relative_epsilon": (
                config.reference.convergence_relative_epsilon
            ),
            "absolute_epsilon": (
                config.reference.convergence_absolute_epsilon
            ),
            "fisher_geometry": "frobenius",
            "displacement_geometry": "euclidean",
        },
        "path_divergence": path_divergence,
        "path_hashes": {
            row.method: row.trajectory_hash for row in conditions
        },
        "pairing": {
            "shared_initialization": True,
            "shared_observation_stream": True,
            "shared_oracle_path": True,
            "initial_parameter_hash": next(iter(initial_hashes)),
        },
        "condition_elapsed_seconds": {
            row.method: row.elapsed_seconds for row in conditions
        },
        "total_elapsed_seconds": total_elapsed,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }
    trajectory_path = session.write_torch(
        "phase8_trajectories.pt", _cpu_tree(trajectory_artifact)
    )
    controller_path = session.write_torch(
        "phase8_controller_states.pt", _cpu_tree(controller_artifact)
    )
    checkpoint_path = session.write_torch(
        "phase8_checkpoints.pt", _cpu_tree(checkpoint_artifact)
    )
    if not oracle_path_file.is_file():
        raise RuntimeError("reference-optimum checkpoint disappeared")
    metrics["artifact_files_bytes"] = {
        "phase8_trajectories.pt": trajectory_path.stat().st_size,
        "phase8_controller_states.pt": controller_path.stat().st_size,
        "phase8_checkpoints.pt": checkpoint_path.stat().st_size,
        "phase8_reference_optimum.pt": oracle_path_file.stat().st_size,
    }
    session.write_json("phase8_metrics.json", metrics)
    destination = session.complete(
        [
            "phase8_metrics.json",
            "phase8_trajectories.pt",
            "phase8_controller_states.pt",
            "phase8_checkpoints.pt",
            "phase8_reference_optimum.pt",
        ]
    )
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "path": str(destination),
                "policy": config.controller.policy,
                "methods": list(methods),
                "steps": len(loaded.stream_plan.p_values),
                "oracle_path_hash": oracle_path.content_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
