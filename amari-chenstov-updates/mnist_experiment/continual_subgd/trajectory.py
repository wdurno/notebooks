"""Causal rotation trajectories for Plan 13."""

from __future__ import annotations

import dataclasses
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.derivatives import per_sample_derivatives
from src.ewc import mixture_ewc_strength
from src.hybrid import blend_archive_fisher
from src.mnist_model import mnist_nll
from src.parameters import ParameterLayout
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed

from mnist_experiment.rotated_mnist.data import RotatedStreamPlan
from mnist_experiment.rotated_mnist.phase4_metrics import materialize_rotated_panel
from mnist_experiment.rotated_mnist.plan12.gauge import build_gauge_fixed_model
from mnist_experiment.rotated_mnist.plan12.ridge import fisher_scale
from mnist_experiment.rotated_mnist.plan12.trajectory import _evaluate_state
from mnist_experiment.rotated_mnist.run import _state_dict_cpu
from mnist_experiment.rotated_mnist.run_phase8 import FIXED_PANEL_ANGLES, _auc, _finite_tree
from mnist_experiment.rotated_mnist.transform import tensor_content_hash

from .artifacts import UnitStore
from .conditions import Condition
from .controller import InnovationControllerState
from .geometry import (
    AdaptationGeometry,
    half_life_gain,
    projector_distance,
    random_geometry,
)
from .optimizer import FixedBudgetResult, fixed_budget_update
from .rotation import ensure_rotation_assets, runtime


BURN_IN_REQUIRED = ("burn_in.pt", "metrics.json", "summary.json", "checks.json")
TRAJECTORY_REQUIRED = (
    "trajectory.pt",
    "metrics.json",
    "summary.json",
    "checks.json",
    "final_state.pt",
)
ROTATION_PI = 0.025
ROTATION_RIDGE_RATIO = 0.1


def _load_model_fisher(
    assets: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> tuple[torch.nn.Module, ParameterLayout, LowRankDiagonalFisher]:
    model, layout = build_gauge_fixed_model(seed, device=device, dtype=training_dtype)
    model.load_state_dict(assets["chart_initial_state"])
    layout.assert_metadata(assets["chart_parameter_layout"])
    fisher = representation_from_artifact(
        assets["chart_initial_fisher"],
        device=device,
    )
    if not isinstance(fisher, LowRankDiagonalFisher) or fisher.rank != 8:
        raise RuntimeError("Plan 13 requires a rank-eight chart Fisher")
    return model, layout, fisher.to(device=device, dtype=matrix_dtype)


def _panels(
    assets: dict[str, Any],
    config: Any,
) -> dict[float, tuple[Tensor, Tensor, str]]:
    base_inputs = assets["base_inputs"]
    base_targets = assets["base_targets"]
    return {
        angle: materialize_rotated_panel(
            base_inputs,
            base_targets,
            angle,
            config.rotation,
        )
        for angle in set(FIXED_PANEL_ANGLES.values())
    }


def _fresh_fisher(
    model: torch.nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    *,
    matrix_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    gradients = per_sample_derivatives(
        model,
        inputs,
        targets,
        mnist_nll,
        layout,
        strategy="vmap",
    ).gradients.to(dtype=matrix_dtype)
    return (gradients.mT @ gradients) / gradients.shape[0], gradients


def _head_mask(layout: ParameterLayout, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    mask = torch.zeros(layout.total_numel, device=device, dtype=dtype)
    for spec in layout.specs:
        if spec.name.startswith("classifier."):
            mask[spec.start : spec.stop] = 1
    if int(mask.sum()) != 225:
        raise RuntimeError("gauge-fixed classifier head must contain 225 coordinates")
    return mask


def _optimizer_update(
    store: UnitStore,
    model: torch.nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    fisher: LowRankDiagonalFisher,
    precondition: Any,
) -> FixedBudgetResult:
    study = store.study
    training_dtype = next(model.parameters()).dtype
    training_fisher = fisher.to(device=inputs.device, dtype=training_dtype)
    return fixed_budget_update(
        model,
        layout,
        inputs,
        targets,
        training_fisher,
        precondition,
        strength=mixture_ewc_strength(ROTATION_PI),
        kappa=ROTATION_RIDGE_RATIO * fisher_scale(fisher),
        inner_steps=study.inner_steps,
        learning_rate=study.learning_rate,
        max_backtracks=study.max_backtracks,
    )


def _evaluation_row(
    model: torch.nn.Module,
    layout: ParameterLayout,
    fisher: LowRankDiagonalFisher,
    panels: dict[float, tuple[Tensor, Tensor, str]],
    angle: float,
    config: Any,
    assets: dict[str, Any],
    *,
    device: torch.device,
    training_dtype: torch.dtype,
) -> dict[str, Any]:
    if angle not in panels:
        panels[angle] = materialize_rotated_panel(
            assets["base_inputs"],
            assets["base_targets"],
            angle,
            config.rotation,
        )
    evaluation, _ = _evaluate_state(
        model,
        layout,
        fisher,
        panels,
        angle,
        config,
        nine_prevalence=assets["nine_prevalence"],
        device=device,
        dtype=training_dtype,
    )
    return evaluation


def ensure_burn_in(
    store: UnitStore,
    phase: str,
    index: int,
    schedule_kind: str,
    burn_in_steps: int,
    *,
    data_root: Path,
    resume: bool,
    asset_path: Path | None = None,
) -> Path:
    detail = {"burn_in_steps": int(burn_in_steps)}
    unit = store.unit(
        phase,
        "burn_in",
        index,
        schedule=schedule_kind,
        detail=detail,
    )
    session = store.begin(unit, BURN_IN_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, BURN_IN_REQUIRED)
        assert completed is not None
        return completed
    if asset_path is None:
        asset_path = ensure_rotation_assets(
            store,
            phase,
            index,
            data_root=data_root,
            resume=True,
        )
    study = store.study
    config = study.protocol_for_replica(phase, index)
    device, training_dtype, matrix_dtype = runtime(study)
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    plan = RotatedStreamPlan.from_mapping(assets["stream_plans"][schedule_kind])
    if not 0 < burn_in_steps < plan.schedule.num_transitions:
        raise ValueError("burn-in must leave a scored transition")
    stream = assets["streams"][schedule_kind]
    model, layout, fisher = _load_model_fisher(
        assets,
        seed=config.replica_seed,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    panels = _panels(assets, config)
    rows = []
    observations = []
    gradient_proxies = []
    parameters = [layout.flatten_module(model, detach=True).cpu()]
    learner_seconds = 0.0
    fisher_seconds = 0.0
    started = time.perf_counter()
    for step in range(burn_in_steps):
        angle = plan.schedule.angles_degrees[step]
        evaluation = _evaluation_row(
            model,
            layout,
            fisher,
            panels,
            angle,
            config,
            assets,
            device=device,
            training_dtype=training_dtype,
        )
        inputs = stream["inputs"][step].to(device=device, dtype=training_dtype)
        targets = stream["targets"][step].to(device=device)
        fisher_started = time.perf_counter()
        fresh, _ = _fresh_fisher(
            model,
            layout,
            inputs,
            targets,
            matrix_dtype=matrix_dtype,
        )
        fisher_seconds += time.perf_counter() - fisher_started
        learner_started = time.perf_counter()
        result = _optimizer_update(
            store,
            model,
            layout,
            inputs,
            targets,
            fisher,
            lambda gradient: gradient,
        )
        learner_seconds += time.perf_counter() - learner_started
        observation = result.displacement.to(dtype=matrix_dtype)
        observations.append(observation.cpu())
        gradient_proxies.append((-study.learning_rate * result.initial_gradient).to(dtype=matrix_dtype).cpu())
        parameters.append(layout.flatten_module(model, detach=True).cpu())
        update_started = time.perf_counter()
        update = blend_archive_fisher(
            fisher,
            fresh,
            blend_gain=ROTATION_PI,
            rank=config.fisher.rank,
            lanczos_seed=derive_component_seed(
                config.replica_seed,
                f"plan13:update_lanczos:{phase}:{schedule_kind}:step={step}",
            ),
        )
        fisher = update.representation
        fisher_seconds += time.perf_counter() - update_started
        rows.append(
            {
                "step": step,
                "schedule_kind": schedule_kind,
                "condition": "shared_full_space_burn_in",
                "angle_degrees": angle,
                "leg_id": plan.schedule.leg_ids[step],
                "knot": plan.schedule.knot_flags[step],
                "observations_before_evaluation": step * config.data.samples_per_step,
                "parameter_hash": tensor_content_hash(parameters[-2]),
                "archive_trace": float(fisher.diagonal_vector().sum()),
                "optimizer": result.mapping(),
                "shadow_displacement_norm": float(torch.linalg.vector_norm(observation)),
                **evaluation,
            }
        )
    observation_tensor = torch.stack(observations)
    parameter_tensor = torch.stack(parameters)
    if not torch.equal(parameter_tensor[1:] - parameter_tensor[:-1], observation_tensor.to(parameter_tensor.dtype)):
        raise RuntimeError("burn-in displacement identity failed")
    summary = {
        "phase": phase,
        "replica_index": index,
        "schedule_kind": schedule_kind,
        "burn_in_steps": burn_in_steps,
        "observations_used": burn_in_steps * config.data.samples_per_step,
        "learner_wall_seconds": learner_seconds,
        "fisher_wall_seconds": fisher_seconds,
        "total_wall_time_seconds": time.perf_counter() - started,
    }
    checks = {
        "all_finite": _finite_tree(rows) and _finite_tree(summary),
        "displacement_identity": True,
        "parameter_count": layout.total_numel,
        "shared_treatment_state": True,
    }
    session.write_torch(
        "burn_in.pt",
        {
            "model": _state_dict_cpu(model),
            "fisher": fisher.artifact_mapping(),
            "parameters": parameter_tensor,
            "shadow_displacements": observation_tensor,
            "gradient_proxies": torch.stack(gradient_proxies),
            "parameter_layout": layout.metadata(),
        },
    )
    session.write_json("metrics.json", rows)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, BURN_IN_REQUIRED)


def _geometry_preconditioner(
    geometry: AdaptationGeometry,
    *,
    alpha: float,
    epsilon: float,
    projector_only: bool,
) -> Any:
    def apply(gradient: Tensor) -> Tensor:
        result = geometry.precondition(
            gradient.to(dtype=geometry.basis.dtype),
            alpha=alpha,
            epsilon=epsilon,
            projector_only=projector_only,
        )
        return result.to(dtype=gradient.dtype)

    return apply


def run_rotation_trajectory(
    store: UnitStore,
    phase: str,
    index: int,
    schedule_kind: str,
    condition: Condition,
    *,
    burn_in_steps: int,
    rank: int,
    data_root: Path,
    resume: bool,
    asset_path: Path | None = None,
    burn_in_path: Path | None = None,
) -> Path:
    detail = {
        "burn_in_steps": burn_in_steps,
        "adaptation_rank": rank,
        "condition": condition.mapping(),
    }
    unit = store.unit(
        phase,
        "trajectory",
        index,
        schedule=schedule_kind,
        condition=condition.name,
        detail=detail,
    )
    session = store.begin(unit, TRAJECTORY_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, TRAJECTORY_REQUIRED)
        assert completed is not None
        return completed
    if asset_path is None:
        asset_path = ensure_rotation_assets(store, phase, index, data_root=data_root, resume=True)
    if burn_in_path is None:
        burn_in_path = ensure_burn_in(
            store,
            phase,
            index,
            schedule_kind,
            burn_in_steps,
            data_root=data_root,
            resume=True,
            asset_path=asset_path,
        )
    study = store.study
    config = study.protocol_for_replica(phase, index)
    device, training_dtype, matrix_dtype = runtime(study)
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    burn_in = torch.load(burn_in_path / "burn_in.pt", map_location="cpu", weights_only=False)
    burn_in_rows = __import__("json").loads((burn_in_path / "metrics.json").read_text(encoding="utf-8"))
    plan = RotatedStreamPlan.from_mapping(assets["stream_plans"][schedule_kind])
    schedule = plan.schedule
    stream = assets["streams"][schedule_kind]
    model, layout = build_gauge_fixed_model(config.replica_seed, device=device, dtype=training_dtype)
    model.load_state_dict(burn_in["model"])
    layout.assert_metadata(burn_in["parameter_layout"])
    fisher = representation_from_artifact(burn_in["fisher"], device=device)
    if not isinstance(fisher, LowRankDiagonalFisher):
        raise RuntimeError("burn-in Fisher is not rank plus diagonal")
    fisher = fisher.to(device=device, dtype=matrix_dtype)
    observations = burn_in["shadow_displacements"].to(device=device, dtype=matrix_dtype)
    geometry = AdaptationGeometry.from_observations(observations, rank=rank)
    if condition.kind == "random_rank_matched":
        geometry = random_geometry(
            layout.total_numel,
            geometry.eigenvalues,
            seed=store.study.seed(f"random_basis:{phase}:{schedule_kind}", index),
        )
    panels = _panels(assets, config)
    controller_state = InnovationControllerState()
    head_mask = _head_mask(layout, device=device, dtype=training_dtype)
    rows = []
    parameters = []
    displacements = []
    shadow_displacements = []
    gradient_proxies = []
    bases = []
    eigenvalues = []
    learner_seconds = 0.0
    shadow_seconds = 0.0
    fisher_seconds = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(burn_in_steps, schedule.num_points):
        angle = schedule.angles_degrees[step]
        parameter_before = layout.flatten_module(model, detach=True)
        parameters.append(parameter_before.cpu())
        bases.append(geometry.basis.detach().cpu())
        eigenvalues.append(geometry.eigenvalues.detach().cpu())
        evaluation = _evaluation_row(
            model,
            layout,
            fisher,
            panels,
            angle,
            config,
            assets,
            device=device,
            training_dtype=training_dtype,
        )
        row: dict[str, Any] = {
            "step": step,
            "post_burn_in_step": step - burn_in_steps,
            "schedule_kind": schedule_kind,
            "condition": condition.name,
            "angle_degrees": angle,
            "leg_id": schedule.leg_ids[step],
            "direction_to_next": schedule.directions_to_next[step],
            "knot": schedule.knot_flags[step],
            "observations_before_evaluation": step * config.data.samples_per_step,
            "post_burn_in_observations": (step - burn_in_steps) * config.data.samples_per_step,
            "parameter_hash": tensor_content_hash(parameter_before.cpu()),
            "archive_trace": float(fisher.diagonal_vector().sum()),
            "shadow": None,
            "optimizer": None,
            "geometry": None,
            "fisher_update": None,
            **evaluation,
        }
        if step < schedule.num_transitions:
            inputs = stream["inputs"][step].to(device=device, dtype=training_dtype)
            targets = stream["targets"][step].to(device=device)
            fisher_started = time.perf_counter()
            fresh, _ = _fresh_fisher(
                model,
                layout,
                inputs,
                targets,
                matrix_dtype=matrix_dtype,
            )
            fisher_seconds += time.perf_counter() - fisher_started
            innovation = None
            alpha = 1.0
            beta = None
            if condition.kind == "no_update":
                shadow = torch.zeros_like(parameter_before, dtype=matrix_dtype)
                gradient_proxy = torch.zeros_like(shadow)
                actual = torch.zeros_like(parameter_before)
                optimizer_mapping = {"stopping_reason": "no_update", "displacement_norm": 0.0}
            else:
                if condition.kind == "full_space":
                    learner_started = time.perf_counter()
                    result = _optimizer_update(
                        store,
                        model,
                        layout,
                        inputs,
                        targets,
                        fisher,
                        lambda gradient: gradient,
                    )
                    learner_seconds += time.perf_counter() - learner_started
                    shadow = result.displacement.to(dtype=matrix_dtype)
                    gradient_proxy = (-study.learning_rate * result.initial_gradient).to(dtype=matrix_dtype)
                    actual = result.displacement
                    optimizer_mapping = result.mapping()
                else:
                    shadow_started = time.perf_counter()
                    shadow_result = _optimizer_update(
                        store,
                        model,
                        layout,
                        inputs,
                        targets,
                        fisher,
                        lambda gradient: gradient,
                    )
                    shadow_seconds += time.perf_counter() - shadow_started
                    shadow = shadow_result.displacement.to(dtype=matrix_dtype)
                    gradient_proxy = (-study.learning_rate * shadow_result.initial_gradient).to(dtype=matrix_dtype)
                    layout.copy_vector_to_module(model, parameter_before)
                    if condition.kind == "head_only":
                        precondition = lambda gradient: head_mask * gradient
                    elif condition.kind in {
                        "random_rank_matched",
                        "static_projector",
                        "static_subgd",
                        "online_subgd",
                        "adaptive_subgd",
                    }:
                        innovation = geometry.innovation(shadow)
                        if condition.kind == "adaptive_subgd":
                            assert condition.controller is not None
                            decision, controller_state = controller_state.decide(
                                innovation,
                                condition.controller,
                            )
                            alpha = decision.alpha
                            beta = decision.beta
                        precondition = _geometry_preconditioner(
                            geometry,
                            alpha=alpha,
                            epsilon=condition.epsilon,
                            projector_only=condition.kind == "static_projector",
                        )
                    else:
                        raise RuntimeError(f"unhandled condition: {condition.kind}")
                    learner_started = time.perf_counter()
                    result = _optimizer_update(
                        store,
                        model,
                        layout,
                        inputs,
                        targets,
                        fisher,
                        precondition,
                    )
                    learner_seconds += time.perf_counter() - learner_started
                    actual = result.displacement
                    optimizer_mapping = result.mapping()
            displacements.append(actual.detach().cpu())
            shadow_displacements.append(shadow.detach().cpu())
            gradient_proxies.append(gradient_proxy.detach().cpu())
            distance = 0.0
            if condition.kind == "online_subgd":
                assert condition.covariance_half_life is not None
                updated = geometry.update(shadow, half_life_gain(condition.covariance_half_life))
                distance = projector_distance(geometry.basis, updated.basis)
                geometry = updated
                beta = half_life_gain(condition.covariance_half_life)
            elif condition.kind == "adaptive_subgd":
                assert beta is not None
                updated = geometry.update(shadow, beta)
                distance = projector_distance(geometry.basis, updated.basis)
                geometry = updated
            update_started = time.perf_counter()
            update = blend_archive_fisher(
                fisher,
                fresh,
                blend_gain=ROTATION_PI,
                rank=config.fisher.rank,
                lanczos_seed=derive_component_seed(
                    config.replica_seed,
                    f"plan13:update_lanczos:{phase}:{schedule_kind}:step={step}",
                ),
            )
            fisher = update.representation
            fisher_seconds += time.perf_counter() - update_started
            row["shadow"] = {
                "displacement_norm": float(torch.linalg.vector_norm(shadow)),
                "objective": None if condition.kind in {"no_update", "full_space"} else shadow_result.mapping(),
            }
            row["optimizer"] = optimizer_mapping
            row["geometry"] = {
                "rank": geometry.rank,
                "innovation": innovation,
                "smoothed_innovation": controller_state.smoothed_innovation,
                "alpha": alpha,
                "beta": beta,
                "epsilon": condition.epsilon,
                "orthogonal_gain": (1 - alpha) + alpha * condition.epsilon,
                "projector_distance": distance,
                "eigenvalues": [float(value) for value in geometry.eigenvalues],
                "accepted_shadow_alignment": (
                    None
                    if float(torch.linalg.vector_norm(actual)) == 0 or float(torch.linalg.vector_norm(shadow)) == 0
                    else float(
                        (actual.to(matrix_dtype) @ shadow)
                        / (
                            torch.linalg.vector_norm(actual.to(matrix_dtype))
                            * torch.linalg.vector_norm(shadow)
                        )
                    )
                ),
            }
            row["fisher_update"] = {
                "blend_gain": update.blend_gain,
                "previous_trace": update.previous_trace,
                "fresh_trace": update.fresh_trace,
                "candidate_trace": update.candidate_trace,
                "lanczos": update.lanczos.mapping(),
            }
        rows.append(row)

    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    shadow_tensor = torch.stack(shadow_displacements)
    gradient_proxy_tensor = torch.stack(gradient_proxies)
    if not torch.equal(parameter_tensor[1:] - parameter_tensor[:-1], displacement_tensor):
        raise RuntimeError("Plan 13 displacement identity failed")
    end_to_end_rows = [*burn_in_rows, *rows]
    summary = {
        "phase": phase,
        "replica_index": index,
        "schedule_kind": schedule_kind,
        "condition": condition.mapping(),
        "burn_in_steps": burn_in_steps,
        "adaptation_rank": rank,
        "post_burn_in_current_nll_auc": _auc(rows, "current_nll"),
        "post_burn_in_current_accuracy_auc": _auc(rows, "current_accuracy"),
        "post_burn_in_worst_panel_nll_auc": max(
            _auc(rows, f"{name}_nll") for name in FIXED_PANEL_ANGLES
        ),
        "end_to_end_current_nll_auc": _auc(end_to_end_rows, "current_nll"),
        "end_to_end_current_accuracy_auc": _auc(end_to_end_rows, "current_accuracy"),
        "learner_wall_seconds": learner_seconds,
        "shadow_wall_seconds": shadow_seconds,
        "fisher_wall_seconds": fisher_seconds,
        "total_wall_time_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "peak_cuda_memory_bytes": 0 if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device)),
    }
    checks = {
        "all_finite": _finite_tree(rows) and _finite_tree(summary),
        "displacement_identity": True,
        "parameter_count": layout.total_numel,
        "burn_in_hash": tensor_content_hash(burn_in["parameters"][-1]),
        "first_parameter_hash": tensor_content_hash(parameter_tensor[0]),
        "burn_in_pairing": torch.equal(burn_in["parameters"][-1], parameter_tensor[0]),
        "causal_geometry_order": True,
    }
    if not checks["all_finite"] or not checks["burn_in_pairing"]:
        raise RuntimeError(f"Plan 13 trajectory checks failed: {checks}")
    session.write_torch(
        "trajectory.pt",
        {
            "parameters": parameter_tensor,
            "displacements": displacement_tensor,
            "shadow_displacements": shadow_tensor,
            "gradient_proxies": gradient_proxy_tensor,
            "bases": torch.stack(bases),
            "eigenvalues": torch.stack(eigenvalues),
        },
    )
    session.write_json("metrics.json", rows)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    session.write_torch(
        "final_state.pt",
        {
            "model": _state_dict_cpu(model),
            "fisher": fisher.artifact_mapping(),
            "geometry_basis": geometry.basis.detach().cpu(),
            "geometry_eigenvalues": geometry.eigenvalues.detach().cpu(),
            "controller_state": dataclasses.asdict(controller_state),
        },
    )
    return store.finish(session, TRAJECTORY_REQUIRED)
