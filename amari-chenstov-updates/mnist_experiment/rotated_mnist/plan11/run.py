"""Fresh paired assets and one restartable Plan 11 EWC trajectory."""

from __future__ import annotations

import math
import resource
import time
from pathlib import Path
from typing import Any

import torch

from src.controller import ControllerState, DiscountedMovementState, accept_controller_step
from src.ewc import build_optimizer, take_ewc_proposal
from src.hybrid import blend_archive_fisher
from src.initialization import state_dict_hash
from src.lanczos_wrapper import approximate_low_rank_diagonal
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import build_canonical_model, configure_torch_runtime, resolve_device, resolve_dtype
from src.parameters import ParameterLayout
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed

from ..data import RotatedStreamPlan, generate_rotated_stream, partition_all_digit_mnist
from ..phase4_metrics import materialize_base_panel, materialize_rotated_panel
from ..run import CALIBRATION_BINS, _fit_upright_initializer, _state_dict_cpu
from ..run_phase3 import _estimate_initial_fisher, _fresh_fisher
from ..run_phase5_double_lap import _learner_optimizer_config
from ..run_phase8 import FIXED_PANEL_ANGLES, _LearnerState, _auc, _evaluate, _finite_tree, _summary, _timed
from ..schedule import resolve_shaped_rotation_schedule
from ..transform import tensor_content_hash
from .artifacts import UnitStore
from .config import Policy, Study
from .policy import decide


ASSET_REQUIRED = ("assets.pt", "summary.json")
TRAJECTORY_REQUIRED = ("metrics.json", "summary.json", "checks.json", "final_state.pt")


def _runtime(study: Study) -> tuple[torch.device, torch.dtype, torch.dtype]:
    config = study.protocol
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.dtype)
    matrix_dtype = resolve_dtype(config.fisher.matrix_dtype)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    return device, training_dtype, matrix_dtype


def ensure_assets(
    store: UnitStore,
    phase: str,
    index: int,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    unit = store.unit(phase, index, None, None)
    session = store.begin(unit, ASSET_REQUIRED, resume=resume)
    if session is None:
        return store.paths(unit)[0]
    study = store.study
    config = study.protocol_for_replica(phase, index)
    device, training_dtype, matrix_dtype = _runtime(study)
    started = time.perf_counter()
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=False)
    train_targets, test_targets = dataset_targets(train_dataset), dataset_targets(test_dataset)
    nine_prevalence = float((test_targets == 9).double().mean())
    partitions = partition_all_digit_mnist(
        train_targets, test_targets, config.data, replica_seed=config.replica_seed
    )
    schedules = {
        kind: resolve_shaped_rotation_schedule(
            config.rotation, kind=kind, sigmoid_kappa=config.sigmoid_kappa
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
    linear, sigmoid = streams["linear"][0], streams["sigmoid"][0]
    if linear.observation_indices != sigmoid.observation_indices or linear.class_labels != sigmoid.class_labels:
        raise RuntimeError("linear/sigmoid stream identities differ")
    initializer, layout = build_canonical_model(
        derive_component_seed(config.replica_seed, "plan5_model_initialization"),
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
    approximation = approximate_low_rank_diagonal(
        lambda vector: dense_fisher @ vector,
        torch.diagonal(dense_fisher),
        rank=config.fisher.rank,
        seed=derive_component_seed(config.replica_seed, "plan11_initial_lanczos"),
    )
    base_inputs, base_targets = materialize_base_panel(
        test_dataset, partitions.evaluation, num_workers=config.runtime.num_workers
    )
    assets = {
        "partitions": partitions.to_mapping(),
        "stream_plans": {kind: streams[kind][0].to_mapping() for kind in streams},
        "streams": {
            kind: {"inputs": streams[kind][1], "targets": streams[kind][2]}
            for kind in streams
        },
        "initial_state": initial_state,
        "initial_fisher": approximation.representation.artifact_mapping(),
        "base_inputs": base_inputs,
        "base_targets": base_targets,
        "nine_prevalence": nine_prevalence,
    }
    summary = {
        "replica_seed": config.replica_seed,
        "initialization": initialization,
        "initial_fisher": {**fisher_metrics, "lanczos": approximation.diagnostics.mapping()},
        "initial_model_state_hash": state_dict_hash(initial_state),
        "evaluation_inputs_hash": tensor_content_hash(base_inputs),
        "evaluation_targets_hash": tensor_content_hash(base_targets),
        "stream_identity_paired": True,
        "stream_plan_hashes": {
            kind: tensor_content_hash(streams[kind][1]) for kind in streams
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    session.write_torch("assets.pt", assets)
    session.write_json("summary.json", summary)
    return store.finish(session, ASSET_REQUIRED)


def _apply_update(
    state: _LearnerState,
    policy: Policy,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    phase: str,
    replica_seed: int,
    step: int,
    parameter_before: torch.Tensor,
    delta_degrees: float,
    study: Study,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> dict[str, Any]:
    if state.movement is None:
        raise RuntimeError("Plan 11 movement state is absent")
    decision, decomposed, recommendation = decide(
        state.controller,
        state.movement,
        state.fisher,
        policy,
        batch_size=inputs.shape[0],
    )
    predictable_fisher = state.fisher
    fresh, elapsed = _timed(
        device,
        lambda: _fresh_fisher(
            state.model, state.layout, inputs, targets, matrix_dtype=matrix_dtype
        ),
    )
    state.fisher_wall_seconds += elapsed
    state.score_gradient_count += inputs.shape[0]
    if step == 0:
        fisher_mapping = {
            "update": "initial_summary_only",
            "blend_gain": decision.applied_pi,
            "previous_trace": float(state.fisher.diagonal_vector().sum()),
            "fresh_trace": float(torch.trace(fresh)),
            "lanczos": None,
        }
    else:
        update, elapsed = _timed(
            device,
            lambda: blend_archive_fisher(
                state.fisher,
                fresh,
                blend_gain=decision.applied_pi,
                rank=study.protocol.fisher.rank,
                lanczos_seed=derive_component_seed(
                    replica_seed,
                    f"plan11_update_lanczos:{phase}:{state.schedule_kind}:{policy.name}:step={step}",
                ),
            ),
        )
        state.fisher_wall_seconds += elapsed
        state.fisher = update.representation
        state.lanczos_diagnostics.append(update.lanczos.mapping())
        fisher_mapping = {
            "update": "direct_ema",
            "blend_gain": update.blend_gain,
            "previous_trace": update.previous_trace,
            "fresh_trace": update.fresh_trace,
            "candidate_trace": update.candidate_trace,
            "lanczos": update.lanczos.mapping(),
        }
    proposal, elapsed = _timed(
        device,
        lambda: take_ewc_proposal(
            state.model,
            state.layout,
            inputs,
            targets,
            state.fisher.to(dtype=training_dtype),
            _learner_optimizer_config(study.protocol),
            state.optimizer,
            adaptation_weight=decision.applied_pi,
            penalty_anchor=parameter_before,
        ),
    )
    state.learner_wall_seconds += elapsed
    state.optimizer_iterations += proposal.optimizer_iterations
    state.optimizer_function_evaluations += proposal.optimizer_function_evaluations
    state.optimizer_event_evaluations += inputs.shape[0] * proposal.optimizer_function_evaluations
    displacement = (state.layout.flatten_module(state.model, detach=True) - parameter_before).cpu()
    if not torch.isfinite(displacement).all():
        raise RuntimeError("nonfinite Plan 11 displacement")
    state.displacements.append(displacement)
    acceptance = accept_controller_step(
        state.controller,
        decision,
        displacement.to(dtype=matrix_dtype),
        batch_size=inputs.shape[0],
        delta_p=delta_degrees,
        half_life_p=1.875,
        fisher=predictable_fisher,
    )
    state.controller = acceptance.state
    state.movement = decomposed.state
    controller_mapping = {
        **decision.mapping(extended=True),
        **decomposed.mapping(),
        "anchor": policy.anchor,
        "gain": policy.gain,
        "recommendation_pi": recommendation,
        "decision_uses_current_batch": False,
        "cold_start_semantics": "accepted_updates",
        "cold_start_steps": 8,
    }
    return {
        "proposal": proposal.metrics_mapping(),
        "fisher_update": fisher_mapping,
        "controller": controller_mapping,
        "controller_acceptance": {
            "trend_gain": acceptance.gain,
            "residual_risk_energy": acceptance.residual_squared,
            "residual_euclidean_squared": acceptance.residual_euclidean_squared,
            "scale_observation": acceptance.scale_observation,
            "state_after": state.controller.scalar_mapping(1e-12, risk_metric="fisher"),
        },
    }


def run_trajectory(
    store: UnitStore,
    phase: str,
    index: int,
    schedule_kind: str,
    policy: Policy,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    if schedule_kind not in {"linear", "sigmoid"}:
        raise ValueError("unsupported Plan 11 schedule")
    asset_path = ensure_assets(store, phase, index, data_root=data_root, resume=True)
    unit = store.unit(phase, index, schedule_kind, policy.mapping())
    session = store.begin(unit, TRAJECTORY_REQUIRED, resume=resume)
    if session is None:
        return store.paths(unit)[0]
    started = time.perf_counter()
    study = store.study
    config = study.protocol_for_replica(phase, index)
    device, training_dtype, matrix_dtype = _runtime(study)
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    plan = RotatedStreamPlan.from_mapping(assets["stream_plans"][schedule_kind])
    schedule = plan.schedule
    stream = assets["streams"][schedule_kind]
    model, layout = build_canonical_model(config.replica_seed, device=device, dtype=training_dtype)
    model.load_state_dict(assets["initial_state"])
    initial_fisher = representation_from_artifact(assets["initial_fisher"], device=device)
    if not isinstance(initial_fisher, LowRankDiagonalFisher):
        raise RuntimeError("Plan 11 initial Fisher is not rank-plus-diagonal")
    state = _LearnerState(
        schedule_kind=schedule_kind,
        condition=policy.name,
        model=model,
        layout=layout,
        optimizer=build_optimizer(model, _learner_optimizer_config(config)),
        fisher=initial_fisher.to(dtype=matrix_dtype),
        controller=ControllerState.initialize(
            layout.total_numel,
            config.data.initialization_size,
            dtype=matrix_dtype,
            device="cpu",
        ),
        movement=DiscountedMovementState(),
    )
    base_inputs, base_targets = assets["base_inputs"], assets["base_targets"]
    panels = {
        angle: materialize_rotated_panel(base_inputs, base_targets, angle, config.rotation)
        for angle in set(FIXED_PANEL_ANGLES.values())
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(schedule.num_points):
        angle = schedule.angles_degrees[step]
        if angle not in panels:
            panels[angle] = materialize_rotated_panel(base_inputs, base_targets, angle, config.rotation)
        lagged_speed = None if step == 0 else abs(angle - schedule.angles_degrees[step - 1])
        next_speed = None if step == schedule.num_transitions else abs(schedule.angles_degrees[step + 1] - angle)
        parameter_before = state.layout.flatten_module(state.model, detach=True)
        state.parameters.append(parameter_before.cpu())
        row = {
            "step": step,
            "schedule_kind": schedule_kind,
            "condition": policy.name,
            "angle_degrees": angle,
            "leg_id": schedule.leg_ids[step],
            "direction_to_next": schedule.directions_to_next[step],
            "knot": schedule.knot_flags[step],
            "cumulative_angular_degrees": schedule.cumulative_degrees[step],
            "lagged_angular_speed_degrees_per_update": lagged_speed,
            "next_angular_speed_degrees_per_update": next_speed,
            "observations_before_evaluation": step * config.data.samples_per_step,
            "parameter_hash": tensor_content_hash(parameter_before.cpu()),
            "proposal": None,
            "fisher_update": None,
            "controller": None,
            "controller_acceptance": None,
            **_evaluate(
                state,
                panels,
                angle,
                config,
                nine_prevalence=assets["nine_prevalence"],
                device=device,
                dtype=training_dtype,
            ),
        }
        if step < schedule.num_transitions:
            inputs = stream["inputs"][step].to(device=device, dtype=training_dtype)
            targets = stream["targets"][step].to(device=device)
            row.update(
                _apply_update(
                    state,
                    policy,
                    inputs,
                    targets,
                    phase=phase,
                    replica_seed=config.replica_seed,
                    step=step,
                    parameter_before=parameter_before,
                    delta_degrees=float(next_speed),
                    study=study,
                    device=device,
                    training_dtype=training_dtype,
                    matrix_dtype=matrix_dtype,
                )
            )
        state.rows.append(row)
        if angle not in FIXED_PANEL_ANGLES.values():
            del panels[angle]
    parameters = torch.stack(state.parameters)
    displacements = torch.stack(state.displacements)
    if not torch.equal(displacements, parameters[1:] - parameters[:-1]):
        raise RuntimeError("Plan 11 displacement identity failed")
    summary = _summary(state, schedule.transitions_per_arrow)
    summary.update(
        {
            "anchor": policy.anchor,
            "gain": policy.gain,
            "leg_environment_nll_aucs": [
                _auc(state.rows[start : start + schedule.transitions_per_arrow + 1], "current_nll")
                for start in range(0, schedule.num_transitions, schedule.transitions_per_arrow)
            ],
            "total_wall_time_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "peak_cuda_memory_bytes": 0 if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device)),
        }
    )
    checks = {
        "all_finite": _finite_tree(state.rows) and _finite_tree(summary),
        "shared_initial_model": state.rows[0]["parameter_hash"] == tensor_content_hash(parameters[0]),
        "all_decisions_predictable": all(row["controller"]["decision_uses_current_batch"] is False for row in state.rows[:-1]),
        "maximum_q_recursion_error": max(
            abs(
                row["controller_acceptance"]["state_after"]["q"]
                - ((1 - row["controller"]["applied_pi"]) ** 2 / row["controller"]["effective_size"]
                   + row["controller"]["applied_pi"] ** 2 / config.data.samples_per_step)
            )
            for row in state.rows[:-1]
        ),
        "every_action_matches_blend": all(
            math.isclose(
                row["controller"]["applied_pi"],
                min(0.95, max(0.01, policy.anchor if row["controller"]["cold_start_active"] else policy.anchor + policy.gain * (row["controller"]["recommendation_pi"] - policy.anchor))),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for row in state.rows[:-1]
        ),
    }
    if not all((checks["all_finite"], checks["all_decisions_predictable"], checks["every_action_matches_blend"], checks["maximum_q_recursion_error"] <= 1e-14)):
        raise RuntimeError(f"Plan 11 trajectory checks failed: {checks}")
    session.write_json("metrics.json", state.rows)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    session.write_torch("final_state.pt", {"model": _state_dict_cpu(state.model), "fisher": state.fisher.artifact_mapping(), "controller": state.controller, "movement": state.movement})
    return store.finish(session, TRAJECTORY_REQUIRED)
