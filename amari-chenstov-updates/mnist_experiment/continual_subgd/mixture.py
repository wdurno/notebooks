"""Causal Plan 13 trajectories on the historical digit-9 mixture path."""

from __future__ import annotations

import dataclasses
import resource
import time
from pathlib import Path
from typing import Any

import torch

from src.ewc import mixture_ewc_strength
from src.hybrid import blend_archive_fisher
from src.mnist_data import MixtureStreamPlan
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed

from mnist_experiment.rotated_mnist.plan12.gauge import build_gauge_fixed_model
from mnist_experiment.rotated_mnist.run import _state_dict_cpu
from mnist_experiment.rotated_mnist.run_phase8 import _auc, _finite_tree
from mnist_experiment.rotated_mnist.transform import tensor_content_hash

from .artifacts import UnitStore
from .conditions import Condition
from .controller import InnovationControllerState
from .environments.digit9_mixture import (
    MIXTURE_ASSET_REQUIRED,
    MIXTURE_PI,
    ensure_mixture_assets,
    evaluate_mixture,
    mixture_asset_unit,
)
from .geometry import AdaptationGeometry, half_life_gain, projector_distance, random_geometry
from .optimizer import FixedBudgetResult, fixed_budget_update
from .rotation import runtime
from .trajectory import _fresh_fisher, _geometry_preconditioner, _head_mask


MIXTURE_BURN_REQUIRED = ("burn_in.pt", "metrics.json", "summary.json", "checks.json")
MIXTURE_TRAJECTORY_REQUIRED = (
    "trajectory.pt",
    "metrics.json",
    "summary.json",
    "checks.json",
    "final_state.pt",
)


def mixture_burn_unit(store: UnitStore, index: int, burn_in_steps: int) -> dict[str, Any]:
    return store.unit(
        "phase5",
        "mixture_burn_in",
        index,
        environment="digit9_mixture",
        detail={"burn_in_steps": burn_in_steps},
    )


def mixture_trajectory_unit(
    store: UnitStore,
    index: int,
    condition: Condition,
    *,
    burn_in_steps: int,
    rank: int,
) -> dict[str, Any]:
    return store.unit(
        "phase5",
        "mixture_trajectory",
        index,
        environment="digit9_mixture",
        condition=condition.name,
        detail={
            "burn_in_steps": burn_in_steps,
            "adaptation_rank": rank,
            "condition": condition.mapping(),
        },
    )


def _load_initial_state(
    store: UnitStore,
    assets: dict[str, Any],
    index: int,
) -> tuple[torch.nn.Module, Any, LowRankDiagonalFisher]:
    device, training_dtype, matrix_dtype = runtime(store.study)
    model, layout = build_gauge_fixed_model(
        store.study.seed("phase5:load_model", index),
        device=device,
        dtype=training_dtype,
    )
    model.load_state_dict(assets["model"])
    layout.assert_metadata(assets["parameter_layout"])
    fisher = representation_from_artifact(assets["initial_fisher"], device=device)
    if not isinstance(fisher, LowRankDiagonalFisher) or fisher.rank != 8:
        raise RuntimeError("mixture initial Fisher must be rank-eight plus diagonal")
    return model, layout, fisher.to(device=device, dtype=matrix_dtype)


def _mixture_optimizer(
    store: UnitStore,
    model: torch.nn.Module,
    layout: Any,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    fisher: LowRankDiagonalFisher,
    precondition: Any,
) -> FixedBudgetResult:
    training_fisher = fisher.to(device=inputs.device, dtype=inputs.dtype)
    return fixed_budget_update(
        model,
        layout,
        inputs,
        targets,
        training_fisher,
        precondition,
        strength=mixture_ewc_strength(MIXTURE_PI),
        kappa=0.0,
        inner_steps=store.study.inner_steps,
        learning_rate=store.study.learning_rate,
        max_backtracks=store.study.max_backtracks,
    )


def _evaluation_assets(
    assets: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        assets["evaluation_inputs"].to(device=device, dtype=dtype),
        assets["evaluation_targets"].to(device=device),
    )


def ensure_mixture_burn_in(
    store: UnitStore,
    index: int,
    burn_in_steps: int,
    *,
    data_root: Path,
    resume: bool,
    asset_path: Path | None = None,
) -> Path:
    unit = mixture_burn_unit(store, index, burn_in_steps)
    session = store.begin(unit, MIXTURE_BURN_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, MIXTURE_BURN_REQUIRED)
        assert completed is not None
        return completed
    if asset_path is None:
        asset_path = ensure_mixture_assets(store, index, data_root=data_root, resume=True)
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    plan = MixtureStreamPlan.from_mapping(assets["stream_plan"])
    if not 0 < burn_in_steps < len(plan.p_values):
        raise ValueError("mixture burn-in must leave at least one scored path point")
    device, training_dtype, matrix_dtype = runtime(store.study)
    model, layout, fisher = _load_initial_state(store, assets, index)
    evaluation_inputs, evaluation_targets = _evaluation_assets(
        assets, device=device, dtype=training_dtype
    )
    rows = []
    displacements = []
    gradient_proxies = []
    parameters = [layout.flatten_module(model, detach=True).cpu()]
    learner_seconds = 0.0
    fisher_seconds = 0.0
    started = time.perf_counter()
    for step in range(burn_in_steps):
        p = float(plan.p_values[step])
        evaluation = evaluate_mixture(model, evaluation_inputs, evaluation_targets, p)
        inputs = assets["stream_inputs"][step].to(device=device, dtype=training_dtype)
        targets = assets["stream_targets"][step].to(device=device)
        fisher_started = time.perf_counter()
        fresh, _ = _fresh_fisher(model, layout, inputs, targets, matrix_dtype=matrix_dtype)
        fisher_seconds += time.perf_counter() - fisher_started
        learner_started = time.perf_counter()
        result = _mixture_optimizer(
            store, model, layout, inputs, targets, fisher, lambda gradient: gradient
        )
        learner_seconds += time.perf_counter() - learner_started
        displacement = result.displacement.to(dtype=matrix_dtype)
        displacements.append(displacement.cpu())
        gradient_proxies.append(
            (-store.study.learning_rate * result.initial_gradient).to(dtype=matrix_dtype).cpu()
        )
        parameters.append(layout.flatten_module(model, detach=True).cpu())
        update_started = time.perf_counter()
        update = blend_archive_fisher(
            fisher,
            fresh,
            blend_gain=MIXTURE_PI,
            rank=8,
            lanczos_seed=derive_component_seed(
                store.study.seed("phase5:mixture_replica", index),
                f"burn_update_lanczos:{step}",
            ),
        )
        fisher = update.representation
        fisher_seconds += time.perf_counter() - update_started
        rows.append(
            {
                "step": step,
                "p": p,
                "condition": "shared_full_space_burn_in",
                "observations_before_evaluation": step * plan.samples_per_step,
                "parameter_hash": tensor_content_hash(parameters[-2]),
                "archive_trace": float(fisher.diagonal_vector().sum()),
                "optimizer": result.mapping(),
                **evaluation,
            }
        )
    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    if not torch.equal(
        parameter_tensor[1:] - parameter_tensor[:-1],
        displacement_tensor.to(parameter_tensor.dtype),
    ):
        raise RuntimeError("mixture burn-in displacement identity failed")
    summary = {
        "phase": "phase5",
        "environment": "digit9_mixture",
        "replica_index": index,
        "burn_in_steps": burn_in_steps,
        "observations_used": burn_in_steps * plan.samples_per_step,
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
            "shadow_displacements": displacement_tensor,
            "gradient_proxies": torch.stack(gradient_proxies),
            "parameter_layout": layout.metadata(),
        },
    )
    session.write_json("metrics.json", rows)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, MIXTURE_BURN_REQUIRED)


def run_mixture_trajectory(
    store: UnitStore,
    index: int,
    condition: Condition,
    *,
    burn_in_steps: int,
    rank: int,
    data_root: Path,
    resume: bool,
) -> Path:
    unit = mixture_trajectory_unit(
        store, index, condition, burn_in_steps=burn_in_steps, rank=rank
    )
    session = store.begin(unit, MIXTURE_TRAJECTORY_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, MIXTURE_TRAJECTORY_REQUIRED)
        assert completed is not None
        return completed
    asset_path = ensure_mixture_assets(store, index, data_root=data_root, resume=True)
    burn_path = ensure_mixture_burn_in(
        store,
        index,
        burn_in_steps,
        data_root=data_root,
        resume=True,
        asset_path=asset_path,
    )
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    burn = torch.load(burn_path / "burn_in.pt", map_location="cpu", weights_only=False)
    plan = MixtureStreamPlan.from_mapping(assets["stream_plan"])
    device, training_dtype, matrix_dtype = runtime(store.study)
    model, layout = build_gauge_fixed_model(
        store.study.seed("phase5:trajectory_model", index),
        device=device,
        dtype=training_dtype,
    )
    model.load_state_dict(burn["model"])
    layout.assert_metadata(burn["parameter_layout"])
    fisher = representation_from_artifact(burn["fisher"], device=device)
    if not isinstance(fisher, LowRankDiagonalFisher):
        raise RuntimeError("mixture burn-in Fisher is not rank plus diagonal")
    fisher = fisher.to(device=device, dtype=matrix_dtype)
    geometry = AdaptationGeometry.from_observations(
        burn["shadow_displacements"].to(device=device, dtype=matrix_dtype),
        rank=rank,
    )
    if condition.kind == "random_rank_matched":
        geometry = random_geometry(
            layout.total_numel,
            geometry.eigenvalues,
            seed=store.study.seed("phase5:random_basis", index),
        )
    controller_state = InnovationControllerState()
    head_mask = _head_mask(layout, device=device, dtype=training_dtype)
    evaluation_inputs, evaluation_targets = _evaluation_assets(
        assets, device=device, dtype=training_dtype
    )
    rows = []
    parameters = []
    displacements = []
    shadow_displacements = []
    bases = []
    eigenvalues = []
    learner_seconds = shadow_seconds = fisher_seconds = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(burn_in_steps, len(plan.p_values) - 1):
        p = float(plan.p_values[step])
        before = layout.flatten_module(model, detach=True)
        parameters.append(before.cpu())
        bases.append(geometry.basis.detach().cpu())
        eigenvalues.append(geometry.eigenvalues.detach().cpu())
        evaluation = evaluate_mixture(model, evaluation_inputs, evaluation_targets, p)
        inputs = assets["stream_inputs"][step].to(device=device, dtype=training_dtype)
        targets = assets["stream_targets"][step].to(device=device)
        fisher_started = time.perf_counter()
        fresh, _ = _fresh_fisher(model, layout, inputs, targets, matrix_dtype=matrix_dtype)
        fisher_seconds += time.perf_counter() - fisher_started
        innovation = None
        alpha = 1.0
        beta = None
        if condition.kind == "full_space":
            learner_started = time.perf_counter()
            result = _mixture_optimizer(
                store, model, layout, inputs, targets, fisher, lambda gradient: gradient
            )
            learner_seconds += time.perf_counter() - learner_started
            shadow = result.displacement.to(dtype=matrix_dtype)
            actual = result.displacement
        else:
            shadow_started = time.perf_counter()
            shadow_result = _mixture_optimizer(
                store, model, layout, inputs, targets, fisher, lambda gradient: gradient
            )
            shadow_seconds += time.perf_counter() - shadow_started
            shadow = shadow_result.displacement.to(dtype=matrix_dtype)
            layout.copy_vector_to_module(model, before)
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
                        innovation, condition.controller
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
                raise RuntimeError(f"unsupported mixture condition: {condition.kind}")
            learner_started = time.perf_counter()
            result = _mixture_optimizer(
                store, model, layout, inputs, targets, fisher, precondition
            )
            learner_seconds += time.perf_counter() - learner_started
            actual = result.displacement
        displacements.append(actual.detach().cpu())
        shadow_displacements.append(shadow.detach().cpu())
        distance = 0.0
        if condition.kind == "online_subgd":
            assert condition.covariance_half_life is not None
            beta = half_life_gain(condition.covariance_half_life)
            updated = geometry.update(shadow, beta)
            distance = projector_distance(geometry.basis, updated.basis)
            geometry = updated
        elif condition.kind == "adaptive_subgd":
            assert beta is not None
            updated = geometry.update(shadow, beta)
            distance = projector_distance(geometry.basis, updated.basis)
            geometry = updated
        update_started = time.perf_counter()
        update = blend_archive_fisher(
            fisher,
            fresh,
            blend_gain=MIXTURE_PI,
            rank=8,
            lanczos_seed=derive_component_seed(
                store.study.seed("phase5:mixture_replica", index),
                f"trajectory_update_lanczos:{step}",
            ),
        )
        fisher = update.representation
        fisher_seconds += time.perf_counter() - update_started
        rows.append(
            {
                "step": step,
                "post_burn_in_step": step - burn_in_steps,
                "post_burn_in_observations": (step - burn_in_steps) * plan.samples_per_step,
                "observations_before_evaluation": step * plan.samples_per_step,
                "p": p,
                "schedule_kind": "digit9_mixture",
                "condition": condition.name,
                "parameter_hash": tensor_content_hash(before.cpu()),
                "archive_trace": float(fisher.diagonal_vector().sum()),
                "optimizer": result.mapping(),
                "shadow": {"displacement_norm": float(torch.linalg.vector_norm(shadow))},
                "geometry": {
                    "rank": geometry.rank,
                    "innovation": innovation,
                    "smoothed_innovation": controller_state.smoothed_innovation,
                    "alpha": alpha,
                    "beta": beta,
                    "epsilon": condition.epsilon,
                    "orthogonal_gain": (1 - alpha) + alpha * condition.epsilon,
                    "projector_distance": distance,
                    "eigenvalues": [float(value) for value in geometry.eigenvalues],
                },
                **evaluation,
            }
        )
    step = len(plan.p_values) - 1
    p = float(plan.p_values[step])
    before = layout.flatten_module(model, detach=True)
    parameters.append(before.cpu())
    bases.append(geometry.basis.detach().cpu())
    eigenvalues.append(geometry.eigenvalues.detach().cpu())
    rows.append(
        {
            "step": step,
            "post_burn_in_step": step - burn_in_steps,
            "post_burn_in_observations": (step - burn_in_steps) * plan.samples_per_step,
            "observations_before_evaluation": step * plan.samples_per_step,
            "p": p,
            "schedule_kind": "digit9_mixture",
            "condition": condition.name,
            "parameter_hash": tensor_content_hash(before.cpu()),
            "archive_trace": float(fisher.diagonal_vector().sum()),
            "optimizer": None,
            "shadow": None,
            "geometry": None,
            **evaluate_mixture(model, evaluation_inputs, evaluation_targets, p),
        }
    )
    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    if not torch.equal(parameter_tensor[1:] - parameter_tensor[:-1], displacement_tensor):
        raise RuntimeError("mixture trajectory displacement identity failed")
    summary = {
        "phase": "phase5",
        "environment": "digit9_mixture",
        "replica_index": index,
        "schedule_kind": "digit9_mixture",
        "condition": condition.mapping(),
        "burn_in_steps": burn_in_steps,
        "adaptation_rank": rank,
        "post_burn_in_current_nll_auc": _auc(rows, "current_nll"),
        "post_burn_in_current_accuracy_auc": _auc(rows, "current_accuracy"),
        "post_burn_in_retention_nll_auc": _auc(rows, "p0_nll"),
        "post_burn_in_worst_panel_nll_auc": _auc(rows, "worst_panel_nll"),
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
        "burn_in_pairing": torch.equal(burn["parameters"][-1], parameter_tensor[0]),
        "coordinate_name": "p_t",
        "fisher_pi": MIXTURE_PI,
        "extra_ridge_kappa": 0.0,
    }
    if not checks["all_finite"] or not checks["burn_in_pairing"]:
        raise RuntimeError(f"mixture trajectory checks failed: {checks}")
    session.write_torch(
        "trajectory.pt",
        {
            "parameters": parameter_tensor,
            "displacements": displacement_tensor,
            "shadow_displacements": torch.stack(shadow_displacements),
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
    return store.finish(session, MIXTURE_TRAJECTORY_REQUIRED)


__all__ = [
    "MIXTURE_BURN_REQUIRED",
    "MIXTURE_TRAJECTORY_REQUIRED",
    "ensure_mixture_burn_in",
    "mixture_burn_unit",
    "mixture_trajectory_unit",
    "run_mixture_trajectory",
]
