"""Fixed-pi estimator-health trajectories for Plan 12."""

from __future__ import annotations

import dataclasses
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.controller import ControllerState, DiscountedMovementState, accept_controller_step
from src.derivatives import per_sample_derivatives
from src.ewc import build_optimizer, mixture_ewc_strength, take_ewc_proposal
from src.hybrid import blend_archive_fisher
from src.mnist_model import build_canonical_model, mnist_nll
from src.parameters import ParameterLayout
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed

from ..data import RotatedStreamPlan
from ..phase4_metrics import materialize_rotated_panel
from ..run import _learner_optimizer_config, _state_dict_cpu
from ..run_phase3 import _fresh_fisher
from ..run_phase8 import FIXED_PANEL_ANGLES, _LearnerState, _auc, _evaluate, _finite_tree, _timed
from ..transform import tensor_content_hash
from .artifacts import UnitStore
from .assets import ASSET_REQUIRED, ensure_replica_assets, runtime
from .config import PI
from .gauge import build_gauge_fixed_model
from .ridge import RidgeFisher, displacement_components, fisher_scale
from .sandwich import penalized_sandwich
from ..plan11.config import Policy
from ..plan11.policy import decide as decide_shadow_pi


TRAJECTORY_REQUIRED = ("metrics.json", "summary.json", "checks.json", "trajectory.pt", "final_state.pt")


@dataclasses.dataclass(frozen=True)
class TrajectoryCondition:
    name: str
    chart: bool
    current_only: bool = False
    ridge_geometry: str | None = None
    ridge_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.ridge_geometry not in {None, "isotropic", "tail"}:
            raise ValueError("unsupported ridge geometry")
        if self.ridge_ratio < 0:
            raise ValueError("ridge ratio must be nonnegative")
        if self.ridge_geometry is None and self.ridge_ratio != 0:
            raise ValueError("nonzero ridge ratio requires a geometry")
        if self.current_only and self.ridge_geometry is not None:
            raise ValueError("current-only control cannot use ridge")

    def mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def phase2_conditions(isotropic_ratio: float, tail_ratio: float) -> tuple[TrajectoryCondition, ...]:
    return (
        TrajectoryCondition("current_only", chart=True, current_only=True),
        TrajectoryCondition("gauge_no_ridge", chart=True),
        TrajectoryCondition("isotropic_ridge", chart=True, ridge_geometry="isotropic", ridge_ratio=isotropic_ratio),
        TrajectoryCondition("tail_ridge", chart=True, ridge_geometry="tail", ridge_ratio=tail_ratio),
        TrajectoryCondition("legacy_raw", chart=False),
    )


def _load_model_and_fishers(
    assets: dict[str, Any],
    condition: TrajectoryCondition,
    *,
    seed: int,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> tuple[nn.Module, ParameterLayout, LowRankDiagonalFisher, Tensor]:
    if condition.chart:
        model, layout = build_gauge_fixed_model(seed, device=device, dtype=training_dtype)
        model.load_state_dict(assets["chart_initial_state"])
        compressed = representation_from_artifact(assets["chart_initial_fisher"], device=device)
        dense = assets["chart_initial_dense_fisher"].to(device=device, dtype=matrix_dtype)
    else:
        model, layout = build_canonical_model(seed, device=device, dtype=training_dtype)
        model.load_state_dict(assets["raw_initial_state"])
        compressed = representation_from_artifact(assets["raw_initial_fisher"], device=device)
        dense = assets["raw_initial_dense_fisher"].to(device=device, dtype=matrix_dtype)
    if not isinstance(compressed, LowRankDiagonalFisher):
        raise RuntimeError("Plan 12 trajectory Fisher is not rank-plus-diagonal")
    layout.assert_metadata(assets["chart_parameter_layout"] if condition.chart else assets["raw_parameter_layout"])
    return model, layout, compressed.to(dtype=matrix_dtype), dense


def _resolved_basis(fisher: LowRankDiagonalFisher) -> Tensor:
    if fisher.rank == 0:
        return torch.empty(fisher.shape[0], 0, device=fisher.device, dtype=fisher.dtype)
    basis, triangular = torch.linalg.qr(fisher.factor, mode="reduced")
    diagonal = torch.abs(torch.diagonal(triangular))
    keep = diagonal > torch.finfo(fisher.dtype).eps * max(fisher.shape)
    return basis[:, keep]


def _penalty_operator(
    fisher: LowRankDiagonalFisher,
    condition: TrajectoryCondition,
) -> tuple[LowRankDiagonalFisher | RidgeFisher, float, Tensor]:
    basis = _resolved_basis(fisher)
    kappa = condition.ridge_ratio * fisher_scale(fisher)
    if condition.ridge_geometry is None:
        return fisher, 0.0, basis
    return RidgeFisher(fisher, kappa, condition.ridge_geometry, basis if condition.ridge_geometry == "tail" else None), kappa, basis


def _evaluate_state(
    model: nn.Module,
    layout: ParameterLayout,
    fisher: LowRankDiagonalFisher,
    panels: dict[float, tuple[Tensor, Tensor, str]],
    angle: float,
    config: Any,
    *,
    nine_prevalence: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], float]:
    state = _LearnerState(
        schedule_kind="plan12",
        condition="plan12",
        model=model,
        layout=layout,
        optimizer=None,
        fisher=fisher,
        controller=None,
        movement=None,
    )
    metrics = _evaluate(
        state,
        panels,
        angle,
        config,
        nine_prevalence=nine_prevalence,
        device=device,
        dtype=dtype,
    )
    return metrics, state.evaluation_wall_seconds


def run_trajectory(
    store: UnitStore,
    phase: str,
    index: int,
    schedule_kind: str,
    condition: TrajectoryCondition,
    *,
    data_root: Path,
    resume: bool,
    asset_path: Path | None = None,
    protocol_phase: str | None = None,
    numerical_seed_phase: str | None = None,
    numerical_seed_condition: str | None = None,
    unit_detail: dict[str, Any] | None = None,
) -> Path:
    if asset_path is None:
        asset_path = ensure_replica_assets(store, phase, index, data_root=data_root, resume=True)
    unit = store.unit(
        phase,
        "trajectory",
        index,
        schedule=schedule_kind,
        condition=condition.name,
        detail=condition.mapping() if unit_detail is None else unit_detail,
    )
    session = store.begin(unit, TRAJECTORY_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, TRAJECTORY_REQUIRED)
        assert completed is not None
        return completed
    started = time.perf_counter()
    study = store.study
    config = study.protocol_for_replica(protocol_phase or phase, index)
    device, training_dtype, matrix_dtype = runtime(study)
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    plan = RotatedStreamPlan.from_mapping(assets["stream_plans"][schedule_kind])
    schedule = plan.schedule
    stream = assets["streams"][schedule_kind]
    model, layout, fisher, dense_archive = _load_model_and_fishers(
        assets,
        condition,
        seed=config.replica_seed,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    optimizer = build_optimizer(model, _learner_optimizer_config(config))
    base_inputs, base_targets = assets["base_inputs"], assets["base_targets"]
    panels = {
        angle: materialize_rotated_panel(base_inputs, base_targets, angle, config.rotation)
        for angle in set(FIXED_PANEL_ANGLES.values())
    }
    rows: list[dict[str, Any]] = []
    parameters: list[Tensor] = []
    displacements: list[Tensor] = []
    fisher_wall_seconds = 0.0
    learner_wall_seconds = 0.0
    evaluation_wall_seconds = 0.0
    lanczos_diagnostics: list[dict[str, Any]] = []
    controller = ControllerState.initialize(
        layout.total_numel,
        config.data.initialization_size,
        dtype=matrix_dtype,
        device="cpu",
    )
    movement = DiscountedMovementState()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    beta = mixture_ewc_strength(PI)

    for step in range(schedule.num_points):
        angle = schedule.angles_degrees[step]
        if angle not in panels:
            panels[angle] = materialize_rotated_panel(base_inputs, base_targets, angle, config.rotation)
        parameter_before = layout.flatten_module(model, detach=True)
        parameters.append(parameter_before.cpu())
        evaluation, elapsed = _evaluate_state(
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
        evaluation_wall_seconds += elapsed
        row: dict[str, Any] = {
            "step": step,
            "schedule_kind": schedule_kind,
            "condition": condition.name,
            "angle_degrees": angle,
            "leg_id": schedule.leg_ids[step],
            "direction_to_next": schedule.directions_to_next[step],
            "knot": schedule.knot_flags[step],
            "cumulative_angular_degrees": schedule.cumulative_degrees[step],
            "observations_before_evaluation": step * config.data.samples_per_step,
            "parameter_hash": tensor_content_hash(parameter_before.cpu()),
            "archive_trace": float(fisher.diagonal_vector().sum()),
            "dense_archive_trace": float(torch.trace(dense_archive)),
            "proposal": None,
            "ridge": None,
            "shadow_pi": None,
            "sandwich": None,
            "fisher_update": None,
            **evaluation,
        }
        if step < schedule.num_transitions:
            inputs = stream["inputs"][step].to(device=device, dtype=training_dtype)
            targets = stream["targets"][step].to(device=device)
            predictable_fisher = fisher
            shadow_decision, shadow_decomposed, recommendation = decide_shadow_pi(
                controller,
                movement,
                predictable_fisher,
                Policy(PI, 0.0),
                batch_size=inputs.shape[0],
            )
            row["shadow_pi"] = {
                **shadow_decision.mapping(extended=True),
                **shadow_decomposed.mapping(),
                "recommendation_pi": recommendation,
                "actuated": False,
            }
            penalty, kappa, basis = _penalty_operator(fisher, condition)
            adaptation_weight = 1.0 if condition.current_only else PI
            gradients = per_sample_derivatives(
                model,
                inputs,
                targets,
                mnist_nll,
                layout,
                strategy="vmap",
            ).gradients.to(dtype=matrix_dtype)
            fresh, elapsed = _timed(
                device,
                lambda: (gradients.mT @ gradients) / gradients.shape[0],
            )
            fisher_wall_seconds += elapsed
            if schedule.knot_flags[step]:
                sandwich, _ = penalized_sandwich(
                    gradients,
                    penalty,
                    beta=0.0 if condition.current_only else beta,
                    resolved_basis=basis,
                )
                row["sandwich"] = sandwich
            proposal, elapsed = _timed(
                device,
                lambda: take_ewc_proposal(
                    model,
                    layout,
                    inputs,
                    targets,
                    penalty.to(dtype=training_dtype),
                    _learner_optimizer_config(config),
                    optimizer,
                    adaptation_weight=adaptation_weight,
                    penalty_anchor=parameter_before,
                ),
            )
            learner_wall_seconds += elapsed
            displacement = layout.flatten_module(model, detach=True) - parameter_before
            displacements.append(displacement.cpu())
            next_angle = schedule.angles_degrees[step + 1]
            acceptance = accept_controller_step(
                controller,
                shadow_decision,
                displacement.detach().cpu().to(dtype=matrix_dtype),
                batch_size=inputs.shape[0],
                delta_p=abs(float(next_angle - angle)),
                half_life_p=1.875,
                fisher=predictable_fisher,
            )
            controller = acceptance.state
            movement = shadow_decomposed.state
            row["shadow_pi"]["acceptance"] = {
                "residual_risk_energy": acceptance.residual_squared,
                "residual_euclidean_squared": acceptance.residual_euclidean_squared,
                "controller_after": controller.scalar_mapping(1e-12, risk_metric="fisher"),
            }
            components = displacement_components(displacement.to(matrix_dtype), basis)
            row["proposal"] = proposal.metrics_mapping()
            row["ridge"] = {
                "geometry": condition.ridge_geometry,
                "scale_ratio": condition.ridge_ratio,
                "fisher_mean_eigenvalue": fisher_scale(fisher),
                "kappa": kappa,
                "beta": beta,
                "tau": beta * kappa,
                "resolved_rank": basis.shape[1],
                **components,
            }
            dense_archive = (1 - PI) * dense_archive + PI * fresh
            dense_archive = (dense_archive + dense_archive.mT) / 2
            update, elapsed = _timed(
                device,
                lambda: blend_archive_fisher(
                    fisher,
                    fresh,
                    blend_gain=PI,
                    rank=config.fisher.rank,
                    lanczos_seed=derive_component_seed(
                        config.replica_seed,
                        "plan12_update_lanczos:"
                        f"{numerical_seed_phase or phase}:{schedule_kind}:"
                        f"{numerical_seed_condition or condition.name}:step={step}",
                    ),
                ),
            )
            fisher_wall_seconds += elapsed
            fisher = update.representation
            lanczos_diagnostics.append(update.lanczos.mapping())
            row["fisher_update"] = {
                "blend_gain": PI,
                "previous_trace": update.previous_trace,
                "fresh_trace": update.fresh_trace,
                "candidate_trace": update.candidate_trace,
                "dense_trace": float(torch.trace(dense_archive)),
                "compression_relative_frobenius_error": float(
                    torch.linalg.matrix_norm(fisher.to_dense() - dense_archive, ord="fro")
                    / torch.linalg.matrix_norm(dense_archive, ord="fro").clamp_min(torch.finfo(matrix_dtype).tiny)
                ),
                "lanczos": update.lanczos.mapping(),
            }
        rows.append(row)
        if angle not in FIXED_PANEL_ANGLES.values():
            del panels[angle]

    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    if not torch.equal(displacement_tensor, parameter_tensor[1:] - parameter_tensor[:-1]):
        raise RuntimeError("Plan 12 displacement identity failed")
    summary = {
        "schedule_kind": schedule_kind,
        "condition": condition.mapping(),
        "protocol_phase": protocol_phase or phase,
        "numerical_seed_phase": numerical_seed_phase or phase,
        "numerical_seed_condition": numerical_seed_condition or condition.name,
        "fixed_pi": PI,
        "current_nll_auc": _auc(rows, "current_nll"),
        "current_accuracy_auc": _auc(rows, "current_accuracy"),
        "current_brier_auc": _auc(rows, "current_brier"),
        "worst_panel_nll_auc": max(_auc(rows, f"{name}_nll") for name in FIXED_PANEL_ANGLES),
        "fisher_wall_seconds": fisher_wall_seconds,
        "learner_wall_seconds": learner_wall_seconds,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "total_wall_time_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "peak_cuda_memory_bytes": 0 if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device)),
    }
    checks = {
        "all_finite": _finite_tree(rows) and _finite_tree(summary),
        "parameter_count": layout.total_numel,
        "displacement_identity": True,
        "fixed_pi": PI,
        "all_tau_identities": all(
            row["ridge"] is None
            or abs(row["ridge"]["tau"] - row["ridge"]["beta"] * row["ridge"]["kappa"]) <= 1e-15
            for row in rows
        ),
    }
    if not checks["all_finite"] or not checks["all_tau_identities"]:
        raise RuntimeError(f"Plan 12 trajectory checks failed: {checks}")
    session.write_json("metrics.json", rows)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    session.write_torch(
        "trajectory.pt",
        {"parameters": parameter_tensor, "displacements": displacement_tensor},
    )
    session.write_torch(
        "final_state.pt",
        {
            "model": _state_dict_cpu(model),
            "fisher": fisher.artifact_mapping(),
            "dense_archive": dense_archive.detach().cpu(),
            "lanczos_diagnostics": lanczos_diagnostics,
            "controller": controller,
            "movement": movement,
        },
    )
    return store.finish(session, TRAJECTORY_REQUIRED)
