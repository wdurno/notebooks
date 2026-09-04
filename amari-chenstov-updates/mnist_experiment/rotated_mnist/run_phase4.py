"""Execute the paired repeated-path memory screen from Plan 5 Phase 4."""

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
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

from src.derivatives import per_sample_derivatives
from src.ewc import build_optimizer, take_ewc_proposal
from src.fisher import empirical_fisher
from src.hybrid import (
    HybridArchiveState,
    blend_archive_fisher,
    stage_hybrid_replay_transition,
)
from src.initialization import evaluate_classifier, state_dict_hash
from src.lanczos_wrapper import approximate_low_rank_diagonal
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    mnist_nll,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.replay import FifoReplayBuffer, ReplayEvent, stream_events
from src.representations import DiagonalFisher, LowRankDiagonalFisher
from src.seeding import derive_component_seed

from .data import generate_rotated_stream, partition_all_digit_mnist
from .phase3_config import Phase3FisherConfig
from .phase4_artifacts import PHASE4_REQUIRED_ARTIFACTS, RotatedPhase4RunStore
from .phase4_config import (
    PHASE4_CONDITIONS,
    RotatedPhase4Config,
    load_phase4_config,
)
from .phase4_metrics import (
    evaluate_materialized_classifier,
    materialize_base_panel,
    materialize_rotated_panel,
)
from .run import (
    CALIBRATION_BINS,
    _fit_upright_initializer,
    _learner_optimizer_config,
    _state_dict_cpu,
)
from .run_phase3 import _confusion_matrix, _estimate_initial_fisher, _fresh_fisher
from .schedule import resolve_rotation_schedule
from .transform import tensor_content_hash


FIXED_PANEL_ANGLES = {
    "panel_000": 0.0,
    "panel_015": 15.0,
    "panel_030": 30.0,
}
LOGICAL_ROTATED_EVENT_BYTES = 28 * 28 * 4 + 8 + 8 + 5 * 8


@dataclasses.dataclass
class ConditionState:
    name: str
    model: nn.Module
    layout: ParameterLayout
    optimizer: torch.optim.Optimizer
    fisher: LowRankDiagonalFisher | None = None
    replay: FifoReplayBuffer | None = None
    archive: HybridArchiveState | None = None
    archived_event_ids: frozenset[int] = frozenset()
    rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    parameters: list[Tensor] = dataclasses.field(default_factory=list)
    displacements: list[Tensor] = dataclasses.field(default_factory=list)
    optimizer_iterations: int = 0
    optimizer_function_evaluations: int = 0
    optimizer_event_evaluations: int = 0
    archive_optimizer_function_evaluations: int = 0
    archive_optimizer_event_evaluations: int = 0
    score_gradient_count: int = 0
    evaluation_wall_seconds: float = 0.0
    learner_wall_seconds: float = 0.0
    fisher_wall_seconds: float = 0.0
    archive_wall_seconds: float = 0.0
    materialization_wall_seconds: float = 0.0
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
        default=Path("cache/mnist_experiment/rotated_mnist/phase4"),
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


def _normalized_auc(rows: list[dict[str, Any]], field: str) -> float:
    x = torch.tensor(
        [row["observations_before_evaluation"] for row in rows],
        dtype=torch.float64,
    )
    y = torch.tensor([row[field] for row in rows], dtype=torch.float64)
    width = float(x[-1] - x[0])
    if width <= 0.0:
        raise ValueError("trajectory AUC requires increasing observation exposure")
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


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _event_batch(
    events: tuple[ReplayEvent, ...],
    stream_inputs: Tensor,
    stream_targets: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if not events:
        return (
            stream_inputs.new_empty((0, 1, 28, 28)).to(device=device, dtype=dtype),
            stream_targets.new_empty((0,), dtype=torch.long).to(device=device),
        )
    inputs = torch.stack(
        [stream_inputs[event.stream_step, event.within_step] for event in events]
    ).to(device=device, dtype=dtype)
    targets = torch.stack(
        [stream_targets[event.stream_step, event.within_step] for event in events]
    ).to(device=device)
    expected = torch.as_tensor(
        [event.class_label for event in events], dtype=torch.long, device=device
    )
    if not torch.equal(targets, expected):
        raise RuntimeError("replay event labels differ from the arrival stream")
    return inputs, targets


def _audit_arrivals(
    events: tuple[ReplayEvent, ...],
    stream_inputs: Tensor,
    stream_plan,
) -> tuple[list[dict[str, Any]], int]:
    records = []
    violations = 0
    for event in events:
        expected_hash = stream_plan.transformed_hashes[event.stream_step][
            event.within_step
        ]
        actual_hash = tensor_content_hash(
            stream_inputs[event.stream_step, event.within_step]
        )
        angle = stream_plan.schedule.angles_degrees[event.stream_step]
        if expected_hash != actual_hash:
            violations += 1
        records.append(
            {
                "event_id": event.event_id,
                "arrival_step": event.stream_step,
                "arrival_angle_degrees": angle,
                "transformed_hash": actual_hash,
            }
        )
    return records, violations


def _logical_replay_bytes(replay: FifoReplayBuffer | None) -> int:
    if replay is None:
        return 0
    return 24 + len(replay.events) * LOGICAL_ROTATED_EVENT_BYTES


def _make_states(
    initializer: nn.Module,
    initial_fisher: LowRankDiagonalFisher,
    config: RotatedPhase4Config,
    *,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> dict[str, ConditionState]:
    optimizer_config = _learner_optimizer_config(config)
    states: dict[str, ConditionState] = {}
    for name in PHASE4_CONDITIONS:
        model = copy.deepcopy(initializer).to(device=device, dtype=training_dtype)
        layout = ParameterLayout.from_module(model)
        state = ConditionState(
            name=name,
            model=model,
            layout=layout,
            optimizer=build_optimizer(model, optimizer_config),
        )
        if name == "ewc_fixed_pi005":
            state.fisher = initial_fisher.to(device=device, dtype=matrix_dtype)
        elif name == "replay_b032":
            state.replay = FifoReplayBuffer(config.replay.bounded_capacity)
        elif name == "replay_unbounded":
            state.replay = FifoReplayBuffer(None)
        elif name == "hybrid_b032_fixed_pi005":
            state.replay = FifoReplayBuffer(config.replay.bounded_capacity)
            state.archive = HybridArchiveState(
                anchor=layout.flatten_module(model, detach=True),
                fisher=initial_fisher.to(device=device, dtype=matrix_dtype),
                initial_anchor_observations=config.data.initialization_size,
                initial_fisher_score_observations=config.fisher.initial_sample_size,
            )
            state.archive.validate()
        states[name] = state
    return states


def _legacy_metric_equivalence(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    config: RotatedPhase4Config,
    *,
    nine_prevalence: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    count = min(512, inputs.shape[0])
    while count < inputs.shape[0] and set(targets[:count].tolist()) != set(range(10)):
        count += 1
    subset_inputs = inputs[:count]
    subset_targets = targets[:count]
    current = evaluate_materialized_classifier(
        model,
        subset_inputs,
        subset_targets,
        batch_size=config.initialization.batch_size,
        device=device,
        dtype=dtype,
        calibration_bins=CALIBRATION_BINS,
        nine_prevalence=nine_prevalence,
    )
    dataset = TensorDataset(subset_inputs, subset_targets)
    legacy = evaluate_classifier(
        model,
        dataset,
        batch_size=config.initialization.batch_size,
        device=device,
        dtype=dtype,
        num_workers=0,
        calibration_bins=CALIBRATION_BINS,
        nine_prevalence=nine_prevalence,
    )
    confusion = _confusion_matrix(
        model,
        dataset,
        batch_size=config.initialization.batch_size,
        device=device,
        dtype=dtype,
        num_workers=0,
    )
    common = (
        "sample_count",
        "nll",
        "accuracy",
        "non_nine_nll",
        "non_nine_accuracy",
        "nine_nll",
        "nine_accuracy",
        "nine_true_positive_count",
        "nine_false_positive_count",
        "nine_true_negative_count",
        "nine_false_negative_count",
        "nine_metric_prevalence",
        "nine_recall",
        "nine_false_positive_rate",
        "nine_specificity",
        "nine_ovr_accuracy",
        "nine_precision",
        "brier",
        "non_nine_brier",
        "nine_brier",
        "expected_calibration_error",
        "calibration_bin_count",
    )
    differences = {}
    for name in common:
        if current[name] is None or legacy[name] is None:
            differences[name] = 0.0 if current[name] is legacy[name] else math.inf
        else:
            differences[name] = abs(float(current[name]) - float(legacy[name]))
    return {
        "sample_count": count,
        "maximum_absolute_scalar_difference": max(differences.values()),
        "scalar_differences": differences,
        "confusion_exact": current["confusion_matrix"] == confusion.tolist(),
    }


def _evaluate_state(
    state: ConditionState,
    panels: dict[float, tuple[Tensor, Tensor, str]],
    current_angle: float,
    config: RotatedPhase4Config,
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
    output: dict[str, Any] = {}
    for name, angle in requested.items():
        output.update(_prefix(name, by_angle[angle]))
    return output


def _apply_simple_update(
    state: ConditionState,
    current_events: tuple[ReplayEvent, ...],
    stream_inputs: Tensor,
    stream_targets: Tensor,
    config: RotatedPhase4Config,
    *,
    step: int,
    parameter_before: Tensor,
    zero_fisher: DiagonalFisher,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> dict[str, Any]:
    assert state.name in {"current_only", "ewc_fixed_pi005", "replay_b032", "replay_unbounded"}
    replay_before = () if state.replay is None else state.replay.events
    active_events = (*current_events, *replay_before)
    (batch, materialization_seconds) = _timed(
        device,
        lambda: _event_batch(
            active_events,
            stream_inputs,
            stream_targets,
            device=device,
            dtype=training_dtype,
        ),
    )
    state.materialization_wall_seconds += materialization_seconds
    inputs, targets = batch
    fisher_mapping = None
    if state.name == "ewc_fixed_pi005":
        assert state.fisher is not None
        (fresh, fisher_seconds) = _timed(
            device,
            lambda: _fresh_fisher(
                state.model,
                state.layout,
                inputs[: len(current_events)],
                targets[: len(current_events)],
                matrix_dtype=matrix_dtype,
            ),
        )
        state.fisher_wall_seconds += fisher_seconds
        state.score_gradient_count += len(current_events)
        if step == 0:
            fisher_mapping = {
                "update": "initial_summary_only",
                "blend_gain": config.fisher.fixed_pi,
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
                    blend_gain=config.fisher.fixed_pi,
                    rank=config.fisher.rank,
                    lanczos_seed=derive_component_seed(
                        config.replica_seed,
                        f"plan5_phase4_ewc_lanczos:step={step}",
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
        adaptation_weight = config.fisher.fixed_pi
    else:
        proposal_fisher = zero_fisher
        adaptation_weight = 1.0
    (proposal, optimization_seconds) = _timed(
        device,
        lambda: take_ewc_proposal(
            state.model,
            state.layout,
            inputs,
            targets,
            proposal_fisher,
            _learner_optimizer_config(config),
            state.optimizer,
            adaptation_weight=adaptation_weight,
            penalty_anchor=parameter_before,
        ),
    )
    state.learner_wall_seconds += optimization_seconds
    state.optimizer_iterations += proposal.optimizer_iterations
    state.optimizer_function_evaluations += proposal.optimizer_function_evaluations
    state.optimizer_event_evaluations += (
        len(active_events) * proposal.optimizer_function_evaluations
    )
    replay_after = None
    if state.replay is not None:
        evicted = state.replay.insert(current_events)
        replay_after = {
            "event_count": len(state.replay.events),
            "event_ids": [event.event_id for event in state.replay.events],
            "inserted_event_count": len(current_events),
            "evicted_event_count": len(evicted),
            "total_insertions": state.replay.total_insertions,
            "total_evictions": state.replay.total_evictions,
            "logical_persistent_bytes": _logical_replay_bytes(state.replay),
            "physical_index_state_bytes": state.replay.physical_index_state_bytes,
        }
    return {
        "active_block": {
            "current_event_count": len(current_events),
            "replay_event_count": len(replay_before),
            "presented_event_count": len(active_events),
            "unique_event_count": len({event.event_id for event in active_events}),
            "current_event_already_in_replay_count": len(
                {event.event_id for event in current_events}.intersection(
                    event.event_id for event in replay_before
                )
            ),
        },
        "proposal": proposal.metrics_mapping(),
        "fisher_update": fisher_mapping,
        "replay_before": {
            "event_count": len(replay_before),
            "event_ids": [event.event_id for event in replay_before],
        },
        "replay_after": replay_after,
        "archive_consolidation": None,
    }


def _apply_hybrid_update(
    state: ConditionState,
    current_events: tuple[ReplayEvent, ...],
    stream_inputs: Tensor,
    stream_targets: Tensor,
    initializer: nn.Module,
    config: RotatedPhase4Config,
    *,
    step: int,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> dict[str, Any]:
    assert state.replay is not None and state.archive is not None
    replay_before = state.replay.events
    transition = stage_hybrid_replay_transition(
        state.replay, current_events, state.archived_event_ids
    )
    active_events = (*current_events, *replay_before)
    (batch, materialization_seconds) = _timed(
        device,
        lambda: _event_batch(
            active_events,
            stream_inputs,
            stream_targets,
            device=device,
            dtype=training_dtype,
        ),
    )
    state.materialization_wall_seconds += materialization_seconds
    inputs, targets = batch
    (proposal, learner_seconds) = _timed(
        device,
        lambda: take_ewc_proposal(
            state.model,
            state.layout,
            inputs,
            targets,
            state.archive.fisher.to(dtype=training_dtype),
            _learner_optimizer_config(config),
            state.optimizer,
            adaptation_weight=config.fisher.fixed_pi,
            penalty_anchor=state.archive.anchor,
        ),
    )
    state.learner_wall_seconds += learner_seconds
    state.optimizer_iterations += proposal.optimizer_iterations
    state.optimizer_function_evaluations += proposal.optimizer_function_evaluations
    state.optimizer_event_evaluations += (
        len(active_events) * proposal.optimizer_function_evaluations
    )
    consolidation = None
    if transition.evicted:
        archive_started = time.perf_counter()
        archive_inputs, archive_targets = _event_batch(
            transition.evicted,
            stream_inputs,
            stream_targets,
            device=device,
            dtype=training_dtype,
        )
        archive_model = copy.deepcopy(initializer).to(
            device=device, dtype=training_dtype
        )
        archive_layout = ParameterLayout.from_module(archive_model)
        archive_layout.copy_vector_to_module(archive_model, state.archive.anchor)
        archive_optimizer = build_optimizer(
            archive_model, _learner_optimizer_config(config)
        )
        archive_proposal = take_ewc_proposal(
            archive_model,
            archive_layout,
            archive_inputs,
            archive_targets,
            state.archive.fisher.to(dtype=training_dtype),
            _learner_optimizer_config(config),
            archive_optimizer,
            adaptation_weight=config.fisher.fixed_pi,
        )
        derivatives = per_sample_derivatives(
            archive_model,
            archive_inputs,
            archive_targets,
            mnist_nll,
            archive_layout,
            strategy="vmap",
        )
        fresh = empirical_fisher(derivatives.gradients.to(dtype=matrix_dtype))
        update = blend_archive_fisher(
            state.archive.fisher,
            fresh,
            blend_gain=config.fisher.fixed_pi,
            rank=config.fisher.rank,
            lanczos_seed=derive_component_seed(
                config.replica_seed,
                f"plan5_phase4_hybrid_archive_lanczos:step={step}",
            ),
        )
        diagnostics = update.lanczos.mapping()
        state.lanczos_diagnostics.append(diagnostics)
        state.archive = HybridArchiveState(
            anchor=archive_layout.flatten_module(archive_model, detach=True),
            fisher=update.representation,
            initial_anchor_observations=state.archive.initial_anchor_observations,
            initial_fisher_score_observations=(
                state.archive.initial_fisher_score_observations
            ),
            archived_online_events=(
                state.archive.archived_online_events + len(transition.evicted)
            ),
            consolidation_steps=state.archive.consolidation_steps + 1,
        )
        state.archive.validate()
        state.archive_optimizer_function_evaluations += (
            archive_proposal.optimizer_function_evaluations
        )
        state.archive_optimizer_event_evaluations += (
            len(transition.evicted)
            * archive_proposal.optimizer_function_evaluations
        )
        state.score_gradient_count += len(transition.evicted)
        _synchronize(device)
        state.archive_wall_seconds += time.perf_counter() - archive_started
        consolidation = {
            "evicted_event_ids": [event.event_id for event in transition.evicted],
            "proposal": archive_proposal.metrics_mapping(),
            "score_gradient_count": len(transition.evicted),
            "fisher_blend_gain": update.blend_gain,
            "previous_fisher_trace": update.previous_trace,
            "fresh_fisher_trace": update.fresh_trace,
            "candidate_fisher_trace": update.candidate_trace,
            "lanczos": diagnostics,
        }
    state.replay = transition.replay
    state.archived_event_ids = transition.archived_event_ids
    overlap = state.archived_event_ids.intersection(
        event.event_id for event in state.replay.events
    )
    if overlap:
        raise RuntimeError("hybrid archive and replay identities overlap")
    return {
        "active_block": {
            "current_event_count": len(current_events),
            "replay_event_count": len(replay_before),
            "presented_event_count": len(active_events),
            "unique_event_count": len({event.event_id for event in active_events}),
            "current_event_already_in_replay_count": len(
                {event.event_id for event in current_events}.intersection(
                    event.event_id for event in replay_before
                )
            ),
        },
        "proposal": proposal.metrics_mapping(),
        "fisher_update": None,
        "replay_before": {
            "event_count": len(replay_before),
            "event_ids": [event.event_id for event in replay_before],
        },
        "replay_after": {
            "event_count": len(state.replay.events),
            "event_ids": [event.event_id for event in state.replay.events],
            "inserted_event_count": len(current_events),
            "evicted_event_count": len(transition.evicted),
            "total_insertions": state.replay.total_insertions,
            "total_evictions": state.replay.total_evictions,
            "logical_persistent_bytes": _logical_replay_bytes(state.replay),
            "physical_index_state_bytes": state.replay.physical_index_state_bytes,
        },
        "archive_consolidation": consolidation,
        "identity_audit": {
            "archived_event_count": len(state.archived_event_ids),
            "active_archive_overlap_count": len(overlap),
            "accounted_online_event_count": (
                len(state.archived_event_ids) + len(state.replay.events)
            ),
        },
    }


def _condition_summary(
    state: ConditionState,
    config: RotatedPhase4Config,
) -> dict[str, Any]:
    rows = state.rows
    transitions = config.rotation.transitions_per_arrow
    summary = {
        "condition": state.name,
        "environment_accuracy_auc": _normalized_auc(
            rows, "current_environment_accuracy"
        ),
        "environment_nll_auc": _normalized_auc(rows, "current_nll"),
        "first_ascent_environment_accuracy_auc": _window_auc(
            rows, "current_environment_accuracy", 0, 2 * transitions
        ),
        "first_ascent_environment_nll_auc": _window_auc(
            rows, "current_nll", 0, 2 * transitions
        ),
        "return_environment_accuracy_auc": _window_auc(
            rows, "current_environment_accuracy", 2 * transitions, 3 * transitions
        ),
        "second_ascent_environment_accuracy_auc": _window_auc(
            rows, "current_environment_accuracy", 3 * transitions, 5 * transitions
        ),
        "second_ascent_environment_nll_auc": _window_auc(
            rows, "current_nll", 3 * transitions, 5 * transitions
        ),
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
        "archive_optimizer_function_evaluations": (
            state.archive_optimizer_function_evaluations
        ),
        "archive_optimizer_event_evaluations": (
            state.archive_optimizer_event_evaluations
        ),
        "score_gradient_count": state.score_gradient_count,
        "evaluation_wall_time_seconds": state.evaluation_wall_seconds,
        "learner_optimization_wall_time_seconds": state.learner_wall_seconds,
        "fisher_update_wall_time_seconds": state.fisher_wall_seconds,
        "archive_consolidation_wall_time_seconds": state.archive_wall_seconds,
        "training_materialization_wall_time_seconds": (
            state.materialization_wall_seconds
        ),
        "replay_event_count_final": (
            0 if state.replay is None else len(state.replay.events)
        ),
        "replay_total_evictions": (
            0 if state.replay is None else state.replay.total_evictions
        ),
        "logical_replay_persistent_bytes_final": _logical_replay_bytes(
            state.replay
        ),
        "physical_replay_index_state_bytes_final": (
            0 if state.replay is None else state.replay.physical_index_state_bytes
        ),
        "fisher_summary_bytes_final": (
            state.fisher.storage_bytes() if state.fisher is not None else 0
        ),
        "archive_summary_bytes_final": (
            0
            if state.archive is None
            else state.archive.canonical_persistent_bytes(4)
        ),
        "archive_consolidation_steps": (
            0 if state.archive is None else state.archive.consolidation_steps
        ),
        "archive_online_events": (
            0 if state.archive is None else state.archive.archived_online_events
        ),
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
    return summary


def _smoke_projection(
    states: dict[str, ConditionState],
    config: RotatedPhase4Config,
    *,
    shared_panel_materialization_seconds: float,
) -> float | None:
    if "smoke" not in config.experiment:
        return None
    transition_ratio = 100 / config.rotation.transitions_per_arrow / 5
    evaluation_ratio = 101 / config.rotation.transitions_per_arrow / 5
    inner_ratio = 50 / config.learner.inner_steps
    projected = 60.0 + shared_panel_materialization_seconds * evaluation_ratio
    for name, state in states.items():
        projected += state.evaluation_wall_seconds * evaluation_ratio
        active_ratio = 1.0
        if name in {"replay_b032", "hybrid_b032_fixed_pi005"}:
            active_ratio = 40.0 / max(
                8.0,
                sum(
                    (row.get("active_block") or {}).get(
                        "presented_event_count", 0
                    )
                    for row in state.rows
                )
                / max(1, len(state.rows) - 1),
            )
        elif name == "replay_unbounded":
            active_ratio = 404.0 / max(
                8.0,
                sum(
                    (row.get("active_block") or {}).get(
                        "presented_event_count", 0
                    )
                    for row in state.rows
                )
                / max(1, len(state.rows) - 1),
            )
        projected += (
            state.learner_wall_seconds
            * transition_ratio
            * inner_ratio
            * active_ratio
        )
        projected += state.fisher_wall_seconds * transition_ratio
        if state.archive is not None and state.archive.consolidation_steps:
            projected += (
                state.archive_wall_seconds
                / state.archive.consolidation_steps
                * 96
                * inner_ratio
            )
    return projected


def run_phase4(
    config: RotatedPhase4Config,
    *,
    data_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    download: bool = False,
    resume: bool = False,
) -> Path:
    config.validate()
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.dtype)
    matrix_dtype = resolve_dtype(config.fisher.matrix_dtype)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    session = RotatedPhase4RunStore(output_root).begin(
        config, repo_root, resume=resume
    )
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
    schedule = resolve_rotation_schedule(config.rotation)
    stream_plan, stream_inputs, stream_targets = generate_rotated_stream(
        train_dataset,
        train_targets,
        partitions,
        schedule,
        config.data,
        config.rotation,
        replica_seed=config.replica_seed,
    )
    session.write_json("partitions.json", partitions.to_mapping())
    session.write_json("stream_plan.json", stream_plan.to_mapping())
    session.write_torch(
        "stream_tensors.pt", {"inputs": stream_inputs, "targets": stream_targets}
    )

    model_seed = derive_component_seed(
        config.replica_seed, "plan5_model_initialization"
    )
    initializer, layout = build_canonical_model(
        model_seed, device=device, dtype=training_dtype
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
            config.replica_seed, "plan5_phase4_initial_lanczos"
        ),
    )
    fisher_metrics["lanczos"] = initial_approximation.diagnostics.mapping()
    fisher_metrics["representation_storage_bytes"] = (
        initial_approximation.representation.storage_bytes()
    )
    selected_reference = tuple(
        partitions.reference[: config.fisher.initial_sample_size]
    )
    session.write_torch(
        "initial_fisher.pt",
        {
            "dense": dense_fisher.cpu(),
            "representation": initial_approximation.representation.artifact_mapping(),
            "sample_indices": torch.as_tensor(selected_reference, dtype=torch.long),
        },
    )
    del dense_fisher

    base_inputs, base_targets = materialize_base_panel(
        test_dataset,
        partitions.evaluation,
        num_workers=config.runtime.num_workers,
    )
    panels: dict[float, tuple[Tensor, Tensor, str]] = {}
    panel_started = time.perf_counter()
    for angle in set(FIXED_PANEL_ANGLES.values()):
        panels[angle] = materialize_rotated_panel(
            base_inputs, base_targets, angle, config.rotation
        )
    shared_panel_materialization_seconds = time.perf_counter() - panel_started
    evaluation_indices_hash = tensor_content_hash(
        torch.as_tensor(partitions.evaluation, dtype=torch.long)
    )
    session.write_json(
        "evaluation_panel.json",
        {
            "evaluation_indices_hash": evaluation_indices_hash,
            "base_inputs_hash": tensor_content_hash(base_inputs),
            "base_targets_hash": tensor_content_hash(base_targets),
            "fixed_angle_panel_hashes": {
                f"{angle:.1f}": panels[angle][2] for angle in sorted(panels)
            },
            "sample_count": base_targets.numel(),
            "materialization_wall_time_seconds": (
                shared_panel_materialization_seconds
            ),
        },
    )
    equivalence = _legacy_metric_equivalence(
        initializer,
        panels[0.0][0],
        panels[0.0][1],
        config,
        nine_prevalence=nine_prevalence,
        device=device,
        dtype=training_dtype,
    )

    states = _make_states(
        initializer,
        initial_approximation.representation,
        config,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    initial_hashes = {
        name: state_dict_hash(_state_dict_cpu(state.model))
        for name, state in states.items()
    }
    if set(initial_hashes.values()) != {initial_state_hash}:
        raise RuntimeError("Phase 4 treatments do not share their initializer")
    zero_fisher = DiagonalFisher(
        torch.zeros(layout.total_numel, device=device, dtype=training_dtype)
    )
    arrival_records: list[dict[str, Any]] = []
    arrival_violations = 0
    disjoint_violations = 0
    duplicate_exposure_violations = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress = tqdm(range(schedule.num_points), desc="Phase 4 paired path", unit="step")
    for step in progress:
        angle = schedule.angles_degrees[step]
        if angle not in panels:
            materialized, seconds = _timed(
                device,
                lambda: materialize_rotated_panel(
                    base_inputs, base_targets, angle, config.rotation
                ),
            )
            panels[angle] = materialized
            shared_panel_materialization_seconds += seconds
        for state in states.values():
            parameter_before = state.layout.flatten_module(state.model, detach=True)
            state.parameters.append(parameter_before.cpu())
            evaluations = _evaluate_state(
                state,
                panels,
                angle,
                config,
                nine_prevalence=nine_prevalence,
                device=device,
                dtype=training_dtype,
            )
            row = {
                "step": step,
                "condition": state.name,
                "angle_degrees": angle,
                "leg_id": schedule.leg_ids[step],
                "direction_to_next": schedule.directions_to_next[step],
                "knot": schedule.knot_flags[step],
                "cumulative_angular_degrees": schedule.cumulative_degrees[step],
                "observations_before_evaluation": step
                * config.data.samples_per_step,
                "parameter_hash": tensor_content_hash(parameter_before.cpu()),
                "proposal": None,
                "fisher_update": None,
                "active_block": None,
                "replay_before": None,
                "replay_after": None,
                "archive_consolidation": None,
                **evaluations,
            }
            if step < schedule.num_transitions:
                current_events = stream_events(
                    step,
                    stream_plan.observation_indices[step],
                    stream_plan.class_labels[step],
                    samples_per_step=config.data.samples_per_step,
                )
                if state.name == PHASE4_CONDITIONS[0]:
                    records, violations = _audit_arrivals(
                        current_events, stream_inputs, stream_plan
                    )
                    arrival_records.extend(records)
                    arrival_violations += violations
                if state.name == "hybrid_b032_fixed_pi005":
                    update_mapping = _apply_hybrid_update(
                        state,
                        current_events,
                        stream_inputs,
                        stream_targets,
                        initializer,
                        config,
                        step=step,
                        device=device,
                        training_dtype=training_dtype,
                        matrix_dtype=matrix_dtype,
                    )
                else:
                    update_mapping = _apply_simple_update(
                        state,
                        current_events,
                        stream_inputs,
                        stream_targets,
                        config,
                        step=step,
                        parameter_before=parameter_before,
                        zero_fisher=zero_fisher,
                        device=device,
                        training_dtype=training_dtype,
                        matrix_dtype=matrix_dtype,
                    )
                row.update(update_mapping)
                active = row["active_block"]
                if active["unique_event_count"] != active["presented_event_count"]:
                    duplicate_exposure_violations += 1
                if active["current_event_already_in_replay_count"]:
                    duplicate_exposure_violations += 1
                identity = row.get("identity_audit")
                if identity and identity["active_archive_overlap_count"]:
                    disjoint_violations += 1
                displacement = (
                    state.layout.flatten_module(state.model, detach=True)
                    - parameter_before
                ).cpu()
                if not torch.isfinite(displacement).all():
                    raise RuntimeError("Phase 4 learner displacement is nonfinite")
                state.displacements.append(displacement)
            state.rows.append(row)
        if angle not in FIXED_PANEL_ANGLES.values():
            del panels[angle]

    trajectories = {}
    model_states = {}
    condition_summaries = {}
    for name, state in states.items():
        parameter_tensor = torch.stack(state.parameters)
        displacement_tensor = torch.stack(state.displacements)
        if not torch.equal(
            displacement_tensor, parameter_tensor[1:] - parameter_tensor[:-1]
        ):
            raise RuntimeError(f"Phase 4 displacement identity failed for {name}")
        trajectories[name] = {
            "parameters": parameter_tensor,
            "displacements": displacement_tensor,
        }
        model_states[name] = {
            "initial": initial_state,
            "final": _state_dict_cpu(state.model),
        }
        condition_summaries[name] = _condition_summary(state, config)

    first_rows = [states[name].rows[0] for name in PHASE4_CONDITIONS]
    pairing_fields = (
        "parameter_hash",
        "current_environment_accuracy",
        "current_nll",
        "current_confusion_matrix",
    )
    pairing_exact = all(
        all(row[field] == first_rows[0][field] for row in first_rows[1:])
        for field in pairing_fields
    )
    all_metrics = {
        name: state.rows for name, state in states.items()
    }
    all_finite = _finite_tree(all_metrics) and _finite_tree(condition_summaries)
    lanczos_rows = [
        fisher_metrics["lanczos"],
        *(item for state in states.values() for item in state.lanczos_diagnostics),
    ]
    rank8_resolved = all(item["realized_rank"] == 8 for item in lanczos_rows)
    hybrid = states["hybrid_b032_fixed_pi005"]
    bounded_replay = states["replay_b032"]
    evictions_exercised = (
        bounded_replay.replay is not None
        and bounded_replay.replay.total_evictions > 0
        and hybrid.archive is not None
        and hybrid.archive.consolidation_steps > 0
    )
    projected_seconds = _smoke_projection(
        states,
        config,
        shared_panel_materialization_seconds=shared_panel_materialization_seconds,
    )
    peak_cuda = (
        0
        if device.type != "cuda"
        else int(torch.cuda.max_memory_allocated(device))
    )
    total_cuda = (
        0
        if device.type != "cuda"
        else int(torch.cuda.get_device_properties(device).total_memory)
    )
    gpu_fraction = 0.0 if total_cuda == 0 else peak_cuda / total_cuda
    smoke_checks = {
        "all_conditions_complete": all(
            len(state.rows) == schedule.num_points for state in states.values()
        ),
        "pairing_exact": pairing_exact,
        "arrival_transform_violations": arrival_violations,
        "arrival_records": len(arrival_records),
        "bounded_eviction_and_hybrid_consolidation_exercised": evictions_exercised,
        "archive_replay_disjoint_violations": disjoint_violations,
        "duplicate_exposure_violations": duplicate_exposure_violations,
        "metric_equivalence": equivalence,
        "all_finite": all_finite,
        "rank8_resolved": rank8_resolved,
        "peak_cuda_memory_bytes": peak_cuda,
        "cuda_capacity_bytes": total_cuda,
        "peak_cuda_capacity_fraction": gpu_fraction,
        "projected_full_wall_time_seconds": projected_seconds,
    }
    if "smoke" in config.experiment:
        gate_components = {
            "all_conditions_complete": smoke_checks["all_conditions_complete"],
            "pairing_exact": pairing_exact,
            "arrival_semantics": arrival_violations == 0,
            "eviction_and_consolidation": evictions_exercised,
            "archive_replay_disjoint": disjoint_violations == 0,
            "no_duplicate_exposure": duplicate_exposure_violations == 0,
            "metric_equivalence": (
                equivalence["maximum_absolute_scalar_difference"] <= 1e-10
                and equivalence["confusion_exact"]
            ),
            "all_finite": all_finite,
            "rank8_resolved": rank8_resolved,
            "gpu_capacity": gpu_fraction < 0.80,
            "projected_runtime": (
                projected_seconds is not None and projected_seconds <= 3600.0
            ),
        }
        smoke_checks["gate_components"] = gate_components
        smoke_checks["gate_recommendation"] = (
            "go" if all(gate_components.values()) else "no_go"
        )

    session.write_json("trajectory_metrics.json", all_metrics)
    session.write_torch("trajectories.pt", trajectories)
    session.write_torch("model_states.pt", model_states)
    session.write_json("operational_checks.json", smoke_checks)
    run_summary = {
        "artifact_schema_version": config.artifact_schema_version,
        "metric_schema_version": config.metric_schema_version,
        "conditions": list(PHASE4_CONDITIONS),
        "schedule_hash": schedule.content_hash,
        "stream_plan_hash": stream_plan.content_hash,
        "partition_hash": partitions.content_hash,
        "evaluation_indices_hash": evaluation_indices_hash,
        "initial_model_state_hash": initial_state_hash,
        "condition_initial_model_state_hashes": initial_hashes,
        "parameter_count": layout.total_numel,
        "num_points": schedule.num_points,
        "num_transitions": schedule.num_transitions,
        "samples_per_step": config.data.samples_per_step,
        "nine_prevalence": nine_prevalence,
        "logical_rotated_replay_event_bytes": LOGICAL_ROTATED_EVENT_BYTES,
        "shared_panel_materialization_wall_time_seconds": (
            shared_panel_materialization_seconds
        ),
        "initial_fisher": fisher_metrics,
        "condition_summaries": condition_summaries,
        "total_wall_time_seconds": time.perf_counter() - total_started,
        "peak_process_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        * 1024,
        "peak_cuda_memory_bytes": peak_cuda,
    }
    session.write_json("run_summary.json", run_summary)
    return session.complete(PHASE4_REQUIRED_ARTIFACTS)


def main() -> None:
    arguments = parse_arguments()
    config = load_phase4_config(arguments.config)
    repo_root = Path(__file__).parents[2]
    path = run_phase4(
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
