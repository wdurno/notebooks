"""Run immutable Plan 3 replay-plus-EWC hybrid smoke trajectories."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import io
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Subset
from tqdm.auto import tqdm

from mnist_experiment.run_experiment import _materialize_batch
from mnist_experiment.run_replay import (
    CALIBRATION_BINS,
    _active_indices,
    _cpu_tree,
    _operation_totals,
    _state_dict_cpu,
    _tensor_hash,
    _tensor_tree_bytes,
    _timed,
)
from src.artifacts import RunStore
from src.config import ExperimentConfig, load_config
from src.controller import (
    ControllerAcceptance,
    ControllerDecision,
    ControllerState,
    DiscountedRiskDecision,
    DiscountedRiskState,
    accept_controller_step,
    decide_controller,
    decide_discounted_risk_controller,
)
from src.derivatives import per_sample_derivatives
from src.directional_ridge import DirectionalRidgeLFUState
from src.ewc import build_optimizer, mixture_ewc_strength, take_ewc_proposal
from src.fisher import LFUBatchEstimate, dense_lfu_estimate, empirical_fisher
from src.hybrid import (
    HybridArchiveState,
    blend_archive_fisher,
    load_initial_archive_source,
    stage_hybrid_replay_transition,
    update_archive_fisher_lfu,
)
from src.initialization import evaluate_classifier, load_replica_bundle_for_config
from src.mnist_data import load_mnist_datasets
from src.mnist_model import (
    configure_torch_runtime,
    mnist_nll,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.replay import FifoReplayBuffer, ReplayEvent, stream_events
from src.seeding import derive_component_seed
from src.schedules import resolve_schedule, schedule_trajectory_mapping


PLAN3_HYBRID_ARTIFACT_SCHEMA_VERSION = 6
PLAN3_HYBRID_METRIC_SCHEMA_VERSION = 10
PLAN3_HYBRID_LFU_ARTIFACT_SCHEMA_VERSION = 7
PLAN3_HYBRID_LFU_METRIC_SCHEMA_VERSION = 11
PLAN3_DEPLOYMENT_ARTIFACT_SCHEMA_VERSION = 8
PLAN3_DEPLOYMENT_METRIC_SCHEMA_VERSION = 12
PLAN4_FISHER_HYBRID_ARTIFACT_SCHEMA_VERSION = 10
PLAN4_FISHER_HYBRID_METRIC_SCHEMA_VERSION = 14
PLAN4_EDR_HYBRID_ARTIFACT_SCHEMA_VERSION = 11
PLAN4_EDR_HYBRID_METRIC_SCHEMA_VERSION = 15
CANONICAL_SCALAR_BYTES = 4


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_config(config: ExperimentConfig) -> None:
    expected = {
        14: (
            PLAN3_HYBRID_ARTIFACT_SCHEMA_VERSION,
            PLAN3_HYBRID_METRIC_SCHEMA_VERSION,
            {"ema"},
        ),
        15: (
            PLAN3_HYBRID_LFU_ARTIFACT_SCHEMA_VERSION,
            PLAN3_HYBRID_LFU_METRIC_SCHEMA_VERSION,
            {"ac_only", "full_lfu"},
        ),
        16: (
            PLAN3_DEPLOYMENT_ARTIFACT_SCHEMA_VERSION,
            PLAN3_DEPLOYMENT_METRIC_SCHEMA_VERSION,
            {"ema"},
        ),
        18: (
            PLAN4_FISHER_HYBRID_ARTIFACT_SCHEMA_VERSION,
            PLAN4_FISHER_HYBRID_METRIC_SCHEMA_VERSION,
            {"ema"},
        ),
        19: (
            PLAN4_FISHER_HYBRID_ARTIFACT_SCHEMA_VERSION,
            PLAN4_FISHER_HYBRID_METRIC_SCHEMA_VERSION,
            {"ema"},
        ),
        20: (
            PLAN4_EDR_HYBRID_ARTIFACT_SCHEMA_VERSION,
            PLAN4_EDR_HYBRID_METRIC_SCHEMA_VERSION,
            {"ema"},
        ),
    }.get(config.schema_version)
    if expected is None:
        raise ValueError(
            "hybrid runs require schema version 14, 15, 16, 18, 19, or 20"
        )
    artifact_schema, metric_schema, methods = expected
    if config.artifact_schema_version != artifact_schema:
        raise ValueError(
            f"Plan 3 hybrid artifact schema must be version {artifact_schema}"
        )
    if config.metric_schema_version != metric_schema:
        raise ValueError(
            f"Plan 3 hybrid metric schema must be version {metric_schema}"
        )
    if (
        config.replay is None
        or config.replay.policy != "fifo"
        or config.replay.mode != "hybrid"
        or config.replay.archive_initialization_artifact is None
    ):
        raise ValueError("Plan 3 hybrid runs require a FIFO hybrid configuration")
    if config.schema_version < 16:
        if config.controller.policy != "fixed_unified":
            raise ValueError("Plan 3 hybrid runs require a fixed unified controller")
        if not (
            config.controller.fixed_pi == 0.05
            and config.controller.pi_min == 0.05
            and config.controller.pi_max >= 0.05
        ):
            raise ValueError("Plan 3 hybrid runs require fixed pi=.05")
    elif config.schema_version == 20:
        if (
            config.controller.policy != "discounted_risk"
            or config.controller.fixed_pi not in {0.025, 0.05}
            or config.controller.pi_min != 0.01
            or config.controller.pi_max != 0.95
            or config.controller.trend_half_life_p != 0.05
            or config.controller.action_half_life_steps != 4.0
            or config.controller.risk_metric != "fisher"
        ):
            raise ValueError(
                "schema-v20 requires the frozen H=4 Fisher EDR treatment "
                "with cold-start pi=.025 or .05"
            )
        if (
            config.controller.oracle_mode != "none"
            or config.controller.reference_optimum_artifact is not None
        ):
            raise ValueError("EDR hybrid must not depend on an oracle path")
    else:
        if config.controller.policy not in {"fixed_unified", "optimal_plugin"}:
            raise ValueError("deployment hybrid requires fixed or plug-in control")
        if (
            config.controller.oracle_mode != "none"
            or config.controller.reference_optimum_artifact is not None
        ):
            raise ValueError("deployment hybrid must not depend on an oracle path")
        expected_pi_min = 0.025 if config.schema_version >= 19 else 0.05
        if config.controller.pi_min != expected_pi_min:
            raise ValueError(
                f"schema-v{config.schema_version} hybrid requires "
                f"pi_min={expected_pi_min}"
            )
        if config.controller.policy == "fixed_unified" and not (
            config.controller.fixed_pi == expected_pi_min
            and config.controller.pi_max == expected_pi_min
        ):
            raise ValueError(
                f"fixed schema-v{config.schema_version} hybrid requires "
                f"pi={expected_pi_min}"
            )
        if config.controller.policy == "optimal_plugin":
            expected_half_life = (
                0.05
                if config.schema_version >= 18
                and config.controller.risk_metric == "fisher"
                else 0.2
            )
            if not (
                config.controller.trend_half_life_p == expected_half_life
                and config.controller.pi_max == 0.95
            ):
                raise ValueError(
                    "adaptive hybrid uses the predeclared half-life and pi_max=.95"
                )
    if config.estimator.method not in methods:
        raise ValueError(
            f"schema-v{config.schema_version} does not support hybrid "
            f"estimator method {config.estimator.method}"
        )
    if config.schema_version == 15 and any(
        value is None
        for value in (
            config.estimator.ridge_half_life_steps,
            config.estimator.ridge_amplitude_epsilon,
            config.estimator.ridge_coherence_threshold,
        )
    ):
        raise ValueError("hybrid LFUs require directional-ridge settings")
    if (
        config.estimator.representation != "low_rank_diagonal"
        or config.estimator.low_rank != 8
        or config.estimator.controller_methods != ["low_rank_diagonal"]
    ):
        raise ValueError("Phase 3 requires the selected rank-8 representation")


def _serialized_bytes(value: Any) -> int:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.tell()


def _archive_hashes(state: HybridArchiveState) -> dict[str, Any]:
    return {
        "anchor_hash": _tensor_hash(state.anchor),
        "factor_hash": _tensor_hash(state.fisher.factor),
        "residual_diagonal_hash": _tensor_hash(state.fisher.residual_diagonal),
        "archived_online_events": state.archived_online_events,
        "consolidation_steps": state.consolidation_steps,
        "fisher_rank": state.rank,
        "fisher_trace": float(state.fisher.diagonal_vector().sum()),
    }


def _controller_state_mapping(state: ControllerState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        **dataclasses.asdict(state),
        "trend": state.trend.detach().cpu(),
    }


def _controller_state_from_mapping(
    mapping: dict[str, Any],
    *,
    dtype: torch.dtype,
) -> ControllerState:
    values = dict(mapping)
    trend = values.get("trend")
    if not isinstance(trend, Tensor):
        raise RuntimeError("deployment checkpoint controller trend is invalid")
    values["trend"] = trend.to(device="cpu", dtype=dtype)
    return ControllerState(**values)


def _discounted_risk_state_mapping(
    state: DiscountedRiskState | None,
) -> dict[str, Any] | None:
    return None if state is None else dataclasses.asdict(state)


def _discounted_risk_state_from_mapping(
    mapping: dict[str, Any],
) -> DiscountedRiskState:
    state = DiscountedRiskState(**mapping)
    state.validate()
    return state


def _controller_acceptance_mapping(
    acceptance: ControllerAcceptance,
) -> dict[str, Any]:
    return {
        "gain": acceptance.gain,
        "residual_squared": acceptance.residual_squared,
        "residual_euclidean_squared": acceptance.residual_euclidean_squared,
        "scale_observation": acceptance.scale_observation,
        "normalized_displacement_norm": (
            None
            if acceptance.normalized_displacement is None
            else float(torch.linalg.vector_norm(acceptance.normalized_displacement))
        ),
    }


def _checkpoint_mapping(
    *,
    next_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: FifoReplayBuffer,
    archive: HybridArchiveState,
    archived_event_ids: frozenset[int],
    rows: list[dict[str, Any]],
    parameters: list[Tensor],
    displacements: list[Tensor],
    archive_anchors: list[Tensor],
    observed_source_indices: set[int],
    learner_optimizer_event_evaluations: int,
    archive_optimizer_event_evaluations: int,
    archive_score_gradient_count: int,
    archive_source_sha256: str,
    directional_ridge: DirectionalRidgeLFUState | None,
    controller_state: ControllerState | None,
    discounted_risk_state: DiscountedRiskState | None,
    artifact_schema_version: int,
) -> dict[str, Any]:
    mapping = {
        "schema_version": artifact_schema_version,
        "next_step": next_step,
        "model_state": _state_dict_cpu(model),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "replay_state": replay.to_mapping(),
        "archive_state": archive.to_mapping(),
        "archived_event_ids": sorted(archived_event_ids),
        "rows": rows,
        "parameters": parameters,
        "displacements": displacements,
        "archive_anchors": archive_anchors,
        "observed_source_indices": sorted(observed_source_indices),
        "learner_optimizer_event_evaluations": (
            learner_optimizer_event_evaluations
        ),
        "archive_optimizer_event_evaluations": (
            archive_optimizer_event_evaluations
        ),
        "archive_score_gradient_count": archive_score_gradient_count,
        "archive_source_sha256": archive_source_sha256,
    }
    if artifact_schema_version >= PLAN3_HYBRID_LFU_ARTIFACT_SCHEMA_VERSION:
        mapping["directional_ridge_state"] = (
            None
            if directional_ridge is None
            else directional_ridge.state_mapping()
        )
    if artifact_schema_version >= PLAN3_DEPLOYMENT_ARTIFACT_SCHEMA_VERSION:
        mapping["controller_state"] = _controller_state_mapping(controller_state)
    if artifact_schema_version >= PLAN4_EDR_HYBRID_ARTIFACT_SCHEMA_VERSION:
        mapping["discounted_risk_state"] = _discounted_risk_state_mapping(
            discounted_risk_state
        )
    return mapping


def _labels(events: tuple[ReplayEvent, ...], device: torch.device) -> Tensor:
    return torch.as_tensor(
        [event.class_label for event in events],
        dtype=torch.long,
        device=device,
    )


def main() -> None:
    process_started = time.perf_counter()
    arguments = parse_arguments()
    config = load_config(arguments.config)
    _validate_config(config)
    assert config.replay is not None
    assert config.replay.archive_initialization_artifact is not None
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=(
            config.runtime.deterministic_algorithms
            and config.runtime.device in {"cuda", "auto"}
        ),
    )
    repo_root = Path(__file__).parents[1]
    cache_parent = Path(config.cache_root).parent
    data_root = arguments.data_root or cache_parent / "datasets"
    replica_root = arguments.replica_root or cache_parent / "replicas"
    output_root = arguments.output_root or Path(config.cache_root)
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.training_dtype)
    matrix_dtype = resolve_dtype(config.runtime.matrix_dtype)
    session = RunStore(output_root).begin(
        config,
        repo_root,
        resume=arguments.resume,
    )
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=False)
    loaded = load_replica_bundle_for_config(replica_root, config, device=device)
    model = copy.deepcopy(loaded.model).to(device=device, dtype=training_dtype)
    layout = ParameterLayout.from_module(model)
    optimizer = build_optimizer(model, config.optimizer)
    source_path = Path(config.replay.archive_initialization_artifact)
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    initial_source = load_initial_archive_source(
        source_path,
        layout.flatten_module(model, detach=True),
        expected_rank=config.estimator.low_rank,
        initial_anchor_observations=config.data.initialization_size,
        initial_fisher_score_observations=config.reference.sample_size,
        device=device,
        anchor_dtype=training_dtype,
        fisher_dtype=matrix_dtype,
    )
    archive = initial_source.state
    controller_state = (
        ControllerState.initialize(
            layout.total_numel,
            config.data.initialization_size,
            dtype=matrix_dtype,
            device="cpu",
        )
        if config.controller.policy in {"optimal_plugin", "discounted_risk"}
        else None
    )
    discounted_risk_state = (
        DiscountedRiskState()
        if config.controller.policy == "discounted_risk"
        else None
    )
    directional_ridge = (
        None
        if config.estimator.method == "ema"
        else DirectionalRidgeLFUState(
            half_life_steps=config.estimator.ridge_half_life_steps,
            amplitude_epsilon=config.estimator.ridge_amplitude_epsilon,
            coherence_threshold=config.estimator.ridge_coherence_threshold,
        )
    )
    capacity = (
        None if config.replay.capacity == "unbounded" else config.replay.capacity
    )
    replay = FifoReplayBuffer(capacity=capacity)
    archived_event_ids: frozenset[int] = frozenset()
    evaluation_dataset = Subset(test_dataset, loaded.partitions.evaluation)
    effective_steps = config.replay.max_steps or config.data.num_p_steps
    p_values = loaded.stream_plan.p_values[:effective_steps]
    rows: list[dict[str, Any]] = []
    parameters: list[Tensor] = []
    displacements: list[Tensor] = []
    archive_anchors: list[Tensor] = []
    observed_source_indices: set[int] = set()
    learner_optimizer_event_evaluations = 0
    archive_optimizer_event_evaluations = 0
    archive_score_gradient_count = 0
    start_step = 0
    checkpoint_path = session.path / "plan3_hybrid_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != config.artifact_schema_version:
            raise RuntimeError("hybrid checkpoint schema is incompatible")
        if checkpoint.get("archive_source_sha256") != initial_source.artifact_sha256:
            raise RuntimeError("hybrid checkpoint archive source changed")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        replay = FifoReplayBuffer.from_mapping(checkpoint["replay_state"])
        archive = HybridArchiveState.from_mapping(
            checkpoint["archive_state"],
            device=device,
            anchor_dtype=training_dtype,
            fisher_dtype=matrix_dtype,
        )
        archived_event_ids = frozenset(checkpoint["archived_event_ids"])
        rows = list(checkpoint["rows"])
        parameters = list(checkpoint["parameters"])
        displacements = list(checkpoint["displacements"])
        archive_anchors = list(checkpoint["archive_anchors"])
        observed_source_indices = set(checkpoint["observed_source_indices"])
        learner_optimizer_event_evaluations = int(
            checkpoint["learner_optimizer_event_evaluations"]
        )
        archive_optimizer_event_evaluations = int(
            checkpoint["archive_optimizer_event_evaluations"]
        )
        archive_score_gradient_count = int(
            checkpoint["archive_score_gradient_count"]
        )
        ridge_mapping = checkpoint.get("directional_ridge_state")
        if directional_ridge is None:
            if ridge_mapping is not None:
                raise RuntimeError("no-LFU checkpoint unexpectedly contains ridge state")
        elif not isinstance(ridge_mapping, dict):
            raise RuntimeError("LFU checkpoint is missing directional-ridge state")
        else:
            directional_ridge = DirectionalRidgeLFUState.from_state_mapping(
                ridge_mapping,
                device=device,
                dtype=matrix_dtype,
            )
        controller_mapping = checkpoint.get("controller_state")
        if controller_state is None:
            if controller_mapping is not None:
                raise RuntimeError(
                    "fixed hybrid checkpoint unexpectedly contains controller state"
                )
        elif not isinstance(controller_mapping, dict):
            raise RuntimeError(
                "adaptive deployment checkpoint is missing controller state"
            )
        else:
            controller_state = _controller_state_from_mapping(
                controller_mapping,
                dtype=matrix_dtype,
            )
        discounted_mapping = checkpoint.get("discounted_risk_state")
        if discounted_risk_state is None:
            if discounted_mapping is not None:
                raise RuntimeError(
                    "non-EDR checkpoint unexpectedly contains discounted risk state"
                )
        elif not isinstance(discounted_mapping, dict):
            raise RuntimeError("EDR checkpoint is missing discounted risk state")
        else:
            discounted_risk_state = _discounted_risk_state_from_mapping(
                discounted_mapping
            )
        start_step = int(checkpoint["next_step"])
        print(f"resumed Plan 3 hybrid at step {start_step}", flush=True)

    trajectory_started = time.perf_counter()
    progress = tqdm(
        range(start_step, effective_steps),
        desc=f"hybrid {config.replay.capacity}",
        unit="step",
    )
    for step in progress:
        p_value = p_values[step]
        parameter_before = layout.flatten_module(model, detach=True)
        parameters.append(parameter_before.cpu())
        archive_anchors.append(archive.anchor.detach().cpu())
        classification, evaluation_timing = _timed(
            device,
            lambda: evaluate_classifier(
                model,
                evaluation_dataset,
                batch_size=config.initialization.batch_size,
                device=device,
                dtype=training_dtype,
                num_workers=config.initialization.num_workers,
                calibration_bins=CALIBRATION_BINS,
                nine_prevalence=p_value,
            ),
        )
        operations = {"evaluation": evaluation_timing}
        row: dict[str, Any] = {
            "step": step,
            "p": p_value,
            "parameter_hash": _tensor_hash(parameter_before),
            "classification": classification,
            "operations": operations,
            "learner_proposal": None,
            "archive_consolidation": None,
            "archive_before": _archive_hashes(archive),
            "replay_before": {
                "event_count": len(replay.events),
                "event_ids": [event.event_id for event in replay.events],
            },
        }
        if config.schema_version >= 16:
            row["controller_state_before"] = (
                None
                if controller_state is None
                else controller_state.scalar_mapping(
                    config.controller.trace_epsilon,
                    risk_metric=config.controller.risk_metric,
                )
            )
            row["controller_decision"] = None
            row["controller_acceptance"] = None
        if config.schema_version >= 20:
            row["discounted_risk_state_before"] = _discounted_risk_state_mapping(
                discounted_risk_state
            )
            row["discounted_risk_state_after"] = None
        if step + 1 < effective_steps:
            current_events = stream_events(
                step,
                loaded.stream_plan.observation_indices[step],
                loaded.stream_plan.class_labels[step],
                samples_per_step=config.data.samples_per_step,
            )
            pre_update_replay = replay.events
            transition = stage_hybrid_replay_transition(
                replay,
                current_events,
                archived_event_ids,
            )
            active_indices = _active_indices(current_events, pre_update_replay)
            observed_source_indices.update(
                event.observation_index for event in current_events
            )
            (training_inputs, training_targets), materialization_timing = _timed(
                device,
                lambda: _materialize_batch(
                    train_dataset,
                    active_indices,
                    device=device,
                    dtype=training_dtype,
                ),
            )
            operations["learner_materialization"] = materialization_timing
            expected_targets = _labels(
                (*current_events, *pre_update_replay),
                device,
            )
            if not torch.equal(training_targets, expected_targets):
                raise RuntimeError("hybrid active labels do not match the dataset")

            controller_decision: ControllerDecision | None = None
            controller_fisher = (
                archive.fisher
                if config.controller.risk_metric == "fisher"
                else None
            )
            adaptation_weight = float(config.controller.fixed_pi)
            discounted_decision: DiscountedRiskDecision | None = None
            if controller_state is not None:
                if discounted_risk_state is not None:
                    assert controller_fisher is not None
                    discounted_decision = decide_discounted_risk_controller(
                        controller_state,
                        discounted_risk_state,
                        config.controller,
                        batch_size=len(active_indices),
                        fisher=controller_fisher,
                    )
                    controller_decision = discounted_decision.controller
                else:
                    controller_decision = decide_controller(
                        controller_state,
                        config.controller,
                        batch_size=len(active_indices),
                        fisher=controller_fisher,
                    )
                adaptation_weight = controller_decision.applied_pi
            if config.schema_version >= 16:
                row["controller_decision"] = (
                    {
                        "policy": "fixed_unified",
                        "raw_pi": adaptation_weight,
                        "applied_pi": adaptation_weight,
                        "lower_bound_active": False,
                        "upper_bound_active": False,
                        "cold_start_active": False,
                        "risk_metric": config.controller.risk_metric,
                    }
                    if controller_decision is None
                    else {
                        **controller_decision.mapping(
                            extended=config.schema_version >= 18
                        ),
                        **(
                            {}
                            if discounted_decision is None
                            else discounted_decision.mapping()
                        ),
                    }
                )

            learner_state_before = _state_dict_cpu(model)
            optimizer_state_before = copy.deepcopy(optimizer.state_dict())
            candidate_archive = archive
            candidate_ridge = directional_ridge
            archive_proposal = None
            fisher_update = None
            try:
                learner_proposal, learner_timing = _timed(
                    device,
                    lambda: take_ewc_proposal(
                        model,
                        layout,
                        training_inputs,
                        training_targets,
                        archive.fisher.to(dtype=training_dtype),
                        config.optimizer,
                        optimizer,
                        adaptation_weight=adaptation_weight,
                        penalty_anchor=archive.anchor,
                    ),
                )
                operations["learner_optimization"] = learner_timing
                if transition.evicted:
                    evicted_indices = tuple(
                        event.observation_index for event in transition.evicted
                    )
                    (archive_inputs, archive_targets), archive_materialization = _timed(
                        device,
                        lambda: _materialize_batch(
                            train_dataset,
                            evicted_indices,
                            device=device,
                            dtype=training_dtype,
                        ),
                    )
                    operations["archive_materialization"] = archive_materialization
                    if not torch.equal(
                        archive_targets,
                        _labels(transition.evicted, device),
                    ):
                        raise RuntimeError("evicted labels do not match the dataset")
                    archive_model = copy.deepcopy(loaded.model).to(
                        device=device,
                        dtype=training_dtype,
                    )
                    archive_layout = ParameterLayout.from_module(archive_model)
                    archive_layout.copy_vector_to_module(archive_model, archive.anchor)
                    archive_optimizer = build_optimizer(archive_model, config.optimizer)
                    archive_proposal, archive_timing = _timed(
                        device,
                        lambda: take_ewc_proposal(
                            archive_model,
                            archive_layout,
                            archive_inputs,
                            archive_targets,
                            archive.fisher.to(dtype=training_dtype),
                            config.optimizer,
                            archive_optimizer,
                            adaptation_weight=adaptation_weight,
                        ),
                    )
                    operations["archive_consolidation"] = archive_timing

                    def estimate_fisher() -> LFUBatchEstimate:
                        direction = archive_proposal.displacement.to(
                            device=device,
                            dtype=training_dtype,
                        )
                        derivatives = per_sample_derivatives(
                            archive_model,
                            archive_inputs,
                            archive_targets,
                            mnist_nll,
                            archive_layout,
                            direction=(
                                None
                                if config.estimator.method == "ema"
                                else direction
                            ),
                            strategy="vmap",
                        )
                        gradients = derivatives.gradients.to(dtype=matrix_dtype)
                        if config.estimator.method == "ema":
                            zero = gradients.new_zeros(
                                (archive_layout.total_numel,) * 2
                            )
                            return LFUBatchEstimate(
                                fisher=empirical_fisher(gradients),
                                amari_chentsov=zero,
                                residual=zero.clone(),
                            )
                        return dense_lfu_estimate(
                            gradients,
                            derivatives.hvps.to(dtype=matrix_dtype),
                            direction.to(dtype=matrix_dtype),
                        )

                    statistics, score_timing = _timed(device, estimate_fisher)
                    operations["archive_score_fisher"] = score_timing
                    lanczos_seed = derive_component_seed(
                        config.replica_seed,
                        f"plan3_hybrid_archive_lanczos:step={step}",
                    )
                    if config.estimator.method == "ema":
                        fisher_update, fisher_timing = _timed(
                            device,
                            lambda: blend_archive_fisher(
                                archive.fisher,
                                statistics.fisher,
                                blend_gain=adaptation_weight,
                                rank=config.estimator.low_rank,
                                lanczos_seed=lanczos_seed,
                            ),
                        )
                    else:
                        assert directional_ridge is not None
                        candidate_ridge = (
                            DirectionalRidgeLFUState.from_state_mapping(
                                directional_ridge.state_mapping(),
                                device=device,
                                dtype=matrix_dtype,
                            )
                        )
                        archive_direction = archive_proposal.displacement.to(
                            device=device,
                            dtype=matrix_dtype,
                        )
                        fisher_update, fisher_timing = _timed(
                            device,
                            lambda: update_archive_fisher_lfu(
                                archive.fisher,
                                statistics,
                                archive_direction,
                                candidate_ridge,
                                correction_method=config.estimator.method,
                                blend_gain=adaptation_weight,
                                rank=config.estimator.low_rank,
                                lanczos_seed=lanczos_seed,
                            ),
                        )
                    operations["archive_fisher_update"] = fisher_timing
                    candidate_archive = HybridArchiveState(
                        anchor=archive_layout.flatten_module(
                            archive_model,
                            detach=True,
                        ),
                        fisher=fisher_update.representation,
                        initial_anchor_observations=(
                            archive.initial_anchor_observations
                        ),
                        initial_fisher_score_observations=(
                            archive.initial_fisher_score_observations
                        ),
                        archived_online_events=(
                            archive.archived_online_events + len(transition.evicted)
                        ),
                        consolidation_steps=archive.consolidation_steps + 1,
                    )
                    candidate_archive.validate()
            except BaseException:
                model.load_state_dict(learner_state_before)
                optimizer.load_state_dict(optimizer_state_before)
                raise

            replay, archived_event_ids, archive = (
                transition.replay,
                transition.archived_event_ids,
                candidate_archive,
            )
            directional_ridge = candidate_ridge
            displacement = (
                layout.flatten_module(model, detach=True) - parameter_before
            ).cpu()
            if not torch.equal(displacement, learner_proposal.displacement):
                raise RuntimeError("hybrid proposal does not equal the realized move")
            expected_strength = mixture_ewc_strength(
                adaptation_weight,
                multiplier=config.optimizer.ewc_strength,
            )
            if learner_proposal.effective_ewc_strength != expected_strength:
                raise RuntimeError("hybrid learner consumed the wrong pi")
            if (
                archive_proposal is not None
                and archive_proposal.effective_ewc_strength != expected_strength
            ):
                raise RuntimeError("hybrid archive consumed the wrong pi")
            if fisher_update is not None and fisher_update.blend_gain != adaptation_weight:
                raise RuntimeError("hybrid Fisher blend consumed the wrong pi")
            if controller_decision is not None:
                delta_p = p_values[step + 1] - p_value
                acceptance = accept_controller_step(
                    controller_state,
                    controller_decision,
                    displacement.to(dtype=matrix_dtype),
                    batch_size=len(active_indices),
                    delta_p=delta_p,
                    half_life_p=config.controller.trend_half_life_p,
                    fisher=controller_fisher,
                )
                controller_state = acceptance.state
                row["controller_acceptance"] = _controller_acceptance_mapping(
                    acceptance
                )
                if discounted_decision is not None:
                    discounted_risk_state = discounted_decision.state
            if config.schema_version >= 16:
                row["controller_state_after"] = (
                    None
                    if controller_state is None
                    else controller_state.scalar_mapping(
                        config.controller.trace_epsilon,
                        risk_metric=config.controller.risk_metric,
                    )
                )
            if config.schema_version >= 20:
                row["discounted_risk_state_after"] = (
                    _discounted_risk_state_mapping(discounted_risk_state)
                )
            displacements.append(displacement)
            learner_evaluations = learner_proposal.optimizer_function_evaluations
            learner_event_evaluations = len(active_indices) * learner_evaluations
            learner_optimizer_event_evaluations += learner_event_evaluations
            archive_event_evaluations = 0
            if archive_proposal is not None:
                archive_event_evaluations = (
                    len(transition.evicted)
                    * archive_proposal.optimizer_function_evaluations
                )
                archive_optimizer_event_evaluations += archive_event_evaluations
                archive_score_gradient_count += len(transition.evicted)
            archive_hvp_count = (
                len(transition.evicted)
                if archive_proposal is not None
                and config.estimator.method != "ema"
                else 0
            )
            row.update(
                {
                    "active_block": {
                        "current_event_count": len(current_events),
                        "replay_event_count": len(pre_update_replay),
                        "presented_event_count": len(active_indices),
                        "unique_event_count": len(
                            {
                                event.event_id
                                for event in (*current_events, *pre_update_replay)
                            }
                        ),
                        "current_event_already_in_replay_count": 0,
                    },
                    "learner_proposal": learner_proposal.metrics_mapping(),
                    "learner_optimizer_event_evaluations": (
                        learner_event_evaluations
                    ),
                    "archive_optimizer_event_evaluations": (
                        archive_event_evaluations
                    ),
                    "archive_consolidation": (
                        None
                        if archive_proposal is None
                        else {
                            "evicted_event_ids": [
                                event.event_id for event in transition.evicted
                            ],
                            "proposal": archive_proposal.metrics_mapping(),
                            "score_gradient_count": len(transition.evicted),
                            "hvp_count": archive_hvp_count,
                            "fisher_blend_gain": fisher_update.blend_gain,
                            "previous_fisher_trace": fisher_update.previous_trace,
                            "fresh_fisher_trace": fisher_update.fresh_trace,
                            "candidate_fisher_trace": fisher_update.candidate_trace,
                            "lanczos": fisher_update.lanczos.mapping(),
                            "lfu": (
                                None
                                if config.estimator.method == "ema"
                                else {
                                    "correction_method": (
                                        fisher_update.correction_method
                                    ),
                                    "prediction_trace": (
                                        fisher_update.prediction_trace
                                    ),
                                    "correction_fro": (
                                        fisher_update.correction_fro
                                    ),
                                    "prediction_fro": (
                                        fisher_update.prediction_fro
                                    ),
                                    "candidate_fro": fisher_update.candidate_fro,
                                    "projection_backend": (
                                        fisher_update.projection_backend
                                    ),
                                    "projection": dataclasses.asdict(
                                        fisher_update.projection
                                    ),
                                    "ridge": fisher_update.ridge.metrics_mapping(),
                                }
                            ),
                        }
                    ),
                    "archive_after": _archive_hashes(archive),
                    "replay_after": {
                        "event_count": len(replay.events),
                        "event_ids": [event.event_id for event in replay.events],
                        "inserted_event_count": len(current_events),
                        "evicted_event_count": len(transition.evicted),
                        "total_insertions": replay.total_insertions,
                        "total_evictions": replay.total_evictions,
                    },
                    "identity_audit": {
                        "archived_event_count": len(archived_event_ids),
                        "active_archive_overlap_count": len(
                            archived_event_ids.intersection(
                                event.event_id for event in replay.events
                            )
                        ),
                        "accounted_online_event_count": (
                            len(archived_event_ids) + len(replay.events)
                        ),
                    },
                }
            )

        rows.append(row)
        session.write_torch(
            "plan3_hybrid_checkpoint.pt",
            _checkpoint_mapping(
                next_step=step + 1,
                model=model,
                optimizer=optimizer,
                replay=replay,
                archive=archive,
                archived_event_ids=archived_event_ids,
                rows=rows,
                parameters=parameters,
                displacements=displacements,
                archive_anchors=archive_anchors,
                observed_source_indices=observed_source_indices,
                learner_optimizer_event_evaluations=(
                    learner_optimizer_event_evaluations
                ),
                archive_optimizer_event_evaluations=(
                    archive_optimizer_event_evaluations
                ),
                archive_score_gradient_count=archive_score_gradient_count,
                archive_source_sha256=initial_source.artifact_sha256,
                directional_ridge=directional_ridge,
                controller_state=controller_state,
                discounted_risk_state=discounted_risk_state,
                artifact_schema_version=config.artifact_schema_version,
            ),
        )

    trajectory_elapsed = time.perf_counter() - trajectory_started
    total_elapsed = time.perf_counter() - process_started
    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    archive_anchor_tensor = torch.stack(archive_anchors)
    if not torch.equal(
        displacement_tensor,
        parameter_tensor[1:] - parameter_tensor[:-1],
    ):
        raise RuntimeError("hybrid displacements do not match parameter states")
    if archived_event_ids.intersection(event.event_id for event in replay.events):
        raise RuntimeError("final replay and archive identities overlap")
    trajectory_artifact = {
        "schema_version": config.artifact_schema_version,
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "p_values": p_values,
        "observation_indices": loaded.stream_plan.observation_indices[:effective_steps],
        "class_labels": loaded.stream_plan.class_labels[:effective_steps],
        "parameter_layout": loaded.layout.metadata(),
        "parameters": parameter_tensor,
        "displacements": displacement_tensor,
        "archive_anchors": archive_anchor_tensor,
        "final_replay_state": replay.to_mapping(),
        "final_archive_state": archive.to_mapping(),
        "archived_event_ids_audit": sorted(archived_event_ids),
    }
    if config.schema_version >= 16:
        trajectory_artifact["final_controller_state"] = _controller_state_mapping(
            controller_state
        )
    if config.schema_version >= 20:
        trajectory_artifact["final_discounted_risk_state"] = (
            _discounted_risk_state_mapping(discounted_risk_state)
        )
    schedule_path = None
    if config.data.schedule is not None:
        schedule = resolve_schedule(config.data)
        if schedule.p_values != p_values:
            raise RuntimeError("resolved schedule differs from the immutable stream")
        if schedule.content_hash != loaded.stream_plan.schedule_hash:
            raise RuntimeError("resolved schedule hash differs from the stream")
        schedule_path = session.write_json(
            "schedule_trajectory.json",
            schedule_trajectory_mapping(
                schedule,
                loaded.stream_plan.class_labels,
                samples_per_step=config.data.samples_per_step,
                stream_plan_hash=loaded.stream_plan.content_hash,
                uniform_stream_hash=loaded.stream_plan.uniform_stream_hash,
            ),
        )

    trajectory_path = session.write_torch(
        "plan3_hybrid_trajectory.pt",
        trajectory_artifact,
    )
    checkpoint_bytes = checkpoint_path.stat().st_size
    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    logical_archive_bytes = archive.canonical_persistent_bytes(
        CANONICAL_SCALAR_BYTES
    )
    logical_lfu_state_bytes = (
        0 if directional_ridge is None else directional_ridge.tensor_bytes()
    )
    logical_controller_bytes = (
        0
        if controller_state is None
        else (layout.total_numel + 8) * CANONICAL_SCALAR_BYTES
    )
    logical_discounted_risk_bytes = (
        0 if discounted_risk_state is None else 3 * CANONICAL_SCALAR_BYTES
    )
    archive_mapping = archive.to_mapping()
    operation_totals = _operation_totals(rows)
    metrics = {
        "plan3_hybrid_metric_schema_version": config.metric_schema_version,
        "run_kind": "plan3_hybrid_replay_ewc",
        "replica_bundle_id": loaded.metadata["bundle_id"],
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "condition": {
            "data_mode": "hybrid",
            "replay_policy": "fifo",
            "replay_capacity": config.replay.capacity,
            "uses_ewc": True,
            "adaptation_weight": config.controller.fixed_pi,
            "ewc_old_to_active_odds": mixture_ewc_strength(
                config.controller.fixed_pi,
                multiplier=config.optimizer.ewc_strength,
            ),
            "fisher_update": (
                "ema_no_lfu"
                if config.estimator.method == "ema"
                else config.estimator.method
            ),
            "archive_anchor": "separate_archive_only_parameter_estimate",
            "archive_transition": "current_to_exact_replay_to_compressed_archive",
            "transaction": "learner_fifo_archive_all_or_none",
        },
        "steps": effective_steps,
        "optimizer_steps": effective_steps - 1,
        "parameter_count": layout.total_numel,
        "initial_parameter_hash": _tensor_hash(parameter_tensor[0]),
        "condition_steps": rows,
        "archive_accounting": {
            "initial_anchor_observations": archive.initial_anchor_observations,
            "initial_fisher_score_observations": (
                archive.initial_fisher_score_observations
            ),
            "archived_online_events": archive.archived_online_events,
            "consolidation_steps": archive.consolidation_steps,
            "archive_identity_audit_count": len(archived_event_ids),
            "score_gradient_count": archive_score_gradient_count,
            "hvp_count": sum(
                int(row["archive_consolidation"]["hvp_count"])
                for row in rows
                if row.get("archive_consolidation") is not None
            ),
            "fisher_blend_gain": config.controller.fixed_pi,
            "ridge_half_life_steps": (
                None
                if directional_ridge is None
                else directional_ridge.half_life_steps
            ),
        },
        "resource_ledger": {
            "operation_totals": operation_totals,
            "total_wall_seconds": total_elapsed,
            "trajectory_wall_seconds": trajectory_elapsed,
            "learner_optimizer_iterations": sum(
                int(row["learner_proposal"]["optimizer_iterations"])
                for row in rows
                if row["learner_proposal"] is not None
            ),
            "learner_optimizer_function_evaluations": sum(
                int(row["learner_proposal"]["optimizer_function_evaluations"])
                for row in rows
                if row["learner_proposal"] is not None
            ),
            "learner_optimizer_event_evaluations": (
                learner_optimizer_event_evaluations
            ),
            "archive_optimizer_iterations": sum(
                int(row["archive_consolidation"]["proposal"]["optimizer_iterations"])
                for row in rows
                if row["archive_consolidation"] is not None
            ),
            "archive_optimizer_function_evaluations": sum(
                int(
                    row["archive_consolidation"]["proposal"][
                        "optimizer_function_evaluations"
                    ]
                )
                for row in rows
                if row["archive_consolidation"] is not None
            ),
            "archive_optimizer_event_evaluations": (
                archive_optimizer_event_evaluations
            ),
            "logical_archive_persistent_bytes_final": logical_archive_bytes,
            "logical_replay_persistent_bytes_final": (
                replay.logical_persistent_bytes
            ),
            "logical_hybrid_persistent_bytes_final": (
                logical_archive_bytes
                + replay.logical_persistent_bytes
                + logical_lfu_state_bytes
                + logical_controller_bytes
            ),
            "logical_lfu_state_persistent_bytes_final": logical_lfu_state_bytes,
            "measured_archive_tensor_bytes_final": archive.measured_tensor_bytes(),
            "physical_replay_index_state_bytes_final": (
                replay.physical_index_state_bytes
            ),
            "serialized_archive_state_bytes_final": _serialized_bytes(
                archive_mapping
            ),
            "serialized_replay_state_bytes_final": replay.serialized_state_bytes,
            "serialized_lfu_state_bytes_final": (
                0
                if directional_ridge is None
                else _serialized_bytes(directional_ridge.state_mapping())
            ),
            "identity_audit_bytes_excluded_from_learner_state": (
                len(archived_event_ids) * 8
            ),
            "common_model_parameter_bytes": model_parameter_bytes,
            "common_optimizer_tensor_bytes_final": _tensor_tree_bytes(
                optimizer.state_dict()
            ),
            "peak_process_rss_bytes": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "peak_cuda_memory_bytes": (
                None
                if device.type != "cuda"
                else max(
                    int(item["peak_cuda_allocated_bytes"])
                    for row in rows
                    for item in row["operations"].values()
                    if item["peak_cuda_allocated_bytes"] is not None
                )
            ),
            "checkpoint_file_bytes": checkpoint_bytes,
        },
        "pairing": {
            "shared_initialization": True,
            "shared_observation_stream": True,
            "replica_bundle_id": loaded.metadata["bundle_id"],
            "initial_parameter_hash": _tensor_hash(parameter_tensor[0]),
        },
        "archive_source": {
            "path": config.replay.archive_initialization_artifact,
            "sha256": initial_source.artifact_sha256,
            "condition": initial_source.condition,
            "checkpoint_step": initial_source.checkpoint_step,
        },
        "artifact_files_bytes": {
            "plan3_hybrid_trajectory.pt": trajectory_path.stat().st_size,
            "plan3_hybrid_checkpoint.pt": checkpoint_bytes,
        },
    }
    if schedule_path is not None:
        metrics["artifact_files_bytes"]["schedule_trajectory.json"] = (
            schedule_path.stat().st_size
        )
    if config.schema_version >= 16:
        applied_pis = [
            float(row["controller_decision"]["applied_pi"])
            for row in rows
            if isinstance(row.get("controller_decision"), dict)
        ]
        evaluation_wall = float(
            operation_totals.get("evaluation", {}).get("wall_seconds", 0.0)
        )
        learner_wall = sum(
            float(value["wall_seconds"])
            for name, value in operation_totals.items()
            if name != "evaluation"
        )
        metrics["condition"].update(
            {
                "controller_policy": config.controller.policy,
                "adaptation_weight": (
                    config.controller.fixed_pi
                    if config.controller.policy == "fixed_unified"
                    else None
                ),
                "ewc_old_to_active_odds": (
                    mixture_ewc_strength(
                        config.controller.fixed_pi,
                        multiplier=config.optimizer.ewc_strength,
                    )
                    if config.controller.policy == "fixed_unified"
                    else None
                ),
                "oracle_free": True,
            }
        )
        metrics["archive_accounting"]["fisher_blend_gain"] = (
            config.controller.fixed_pi
            if config.controller.policy == "fixed_unified"
            else None
        )
        metrics["controller_summary"] = {
            "policy": config.controller.policy,
            "risk_metric": config.controller.risk_metric,
            "pi_min": config.controller.pi_min,
            "pi_max": config.controller.pi_max,
            "trend_half_life_p": config.controller.trend_half_life_p,
            "action_half_life_steps": config.controller.action_half_life_steps,
            "applied_pi_min": min(applied_pis),
            "applied_pi_mean": sum(applied_pis) / len(applied_pis),
            "applied_pi_max": max(applied_pis),
            "raw_pi_max": max(
                float(row["controller_decision"]["raw_pi"])
                for row in rows
                if isinstance(row.get("controller_decision"), dict)
            ),
            "lower_bound_fraction": sum(
                bool(row["controller_decision"]["lower_bound_active"])
                for row in rows
                if isinstance(row.get("controller_decision"), dict)
            )
            / len(applied_pis),
            "upper_bound_fraction": sum(
                bool(row["controller_decision"]["upper_bound_active"])
                for row in rows
                if isinstance(row.get("controller_decision"), dict)
            )
            / len(applied_pis),
        }
        if config.schema_version >= 20:
            edr_rows = [
                row["controller_decision"]
                for row in rows
                if isinstance(row.get("controller_decision"), dict)
            ]
            finite_unclipped = [
                float(row["edr_unclipped_pi"])
                for row in edr_rows
                if row.get("edr_unclipped_pi") is not None
            ]
            metrics["controller_summary"].update(
                {
                    "edr_unclipped_pi_min": (
                        min(finite_unclipped) if finite_unclipped else None
                    ),
                    "edr_unclipped_pi_mean": (
                        sum(finite_unclipped) / len(finite_unclipped)
                        if finite_unclipped
                        else None
                    ),
                    "edr_unclipped_pi_max": (
                        max(finite_unclipped) if finite_unclipped else None
                    ),
                    "edr_zero_denominator_count": sum(
                        bool(row["edr_zero_denominator_fallback"])
                        for row in edr_rows
                    ),
                }
            )
        metrics["resource_ledger"].update(
            {
                "learner_wall_seconds_excluding_evaluation": learner_wall,
                "offline_evaluation_wall_seconds": evaluation_wall,
                "logical_controller_persistent_bytes_final": (
                    logical_controller_bytes + logical_discounted_risk_bytes
                ),
                "serialized_controller_state_bytes_final": (
                    0
                    if controller_state is None
                    else _serialized_bytes(_controller_state_mapping(controller_state))
                ),
                "serialized_discounted_risk_state_bytes_final": (
                    0
                    if discounted_risk_state is None
                    else _serialized_bytes(
                        _discounted_risk_state_mapping(discounted_risk_state)
                    )
                ),
            }
        )
    session.write_json("plan3_hybrid_metrics.json", metrics)
    required_artifacts = [
            "plan3_hybrid_metrics.json",
            "plan3_hybrid_trajectory.pt",
            "plan3_hybrid_checkpoint.pt",
        ]
    if schedule_path is not None:
        required_artifacts.append("schedule_trajectory.json")
    destination = session.complete(required_artifacts)
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "path": str(destination),
                "capacity": config.replay.capacity,
                "steps": effective_steps,
                "archive_events": archive.archived_online_events,
                "wall_seconds": total_elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
