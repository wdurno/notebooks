"""Repaired digit-9-mixture transport study for Plan 13 Phase 5R."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import resource
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from scipy.stats import t as student_t

from src.config import OptimizerConfig
from src.ewc import (
    build_optimizer,
    mixture_ewc_strength,
    take_ewc_proposal,
)
from src.hybrid import blend_archive_fisher
from src.mnist_data import MixtureStreamPlan
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed

from mnist_experiment.rotated_mnist.artifacts import _read_json
from mnist_experiment.rotated_mnist.plan12.gauge import build_gauge_fixed_model
from mnist_experiment.rotated_mnist.run import _state_dict_cpu
from mnist_experiment.rotated_mnist.run_phase8 import _auc, _finite_tree
from mnist_experiment.rotated_mnist.transform import tensor_content_hash

from .artifacts import UnitStore, file_hash, freeze_json
from .conditions import Condition, condition_from_mapping
from .config import Plan13Study, canonical_hash
from .controller import InnovationControllerState
from .coordinate_lbfgs import (
    CoordinateFactor,
    coordinate_lbfgs_update,
    geometry_factor,
    identity_factor,
    selection_factor,
)
from .environments.digit9_mixture import MIXTURE_PI, evaluate_mixture
from .geometry import AdaptationGeometry, projector_distance, random_geometry
from .mixture import _evaluation_assets, _load_initial_state
from .refresh_notebook import DEFAULT_NOTEBOOK, refresh
from .rotation import runtime
from .trajectory import _fresh_fisher, _head_mask


DEFAULT_CONFIG = Path("mnist_experiment/continual_subgd/configs/default.json")
DEFAULT_ROOT = Path("cache/mnist_experiment/continual_subgd/default")
PHASE = "phase5r"
SMOKE_PHASE = "phase5r_smoke"
SCHEMA_VERSION = "plan13-phase5r-v1"
BURN_REQUIRED = ("burn_in.pt", "metrics.json", "summary.json", "checks.json")
GATE_REQUIRED = ("summary.json", "checks.json")
TRAJECTORY_REQUIRED = (
    "trajectory.pt",
    "metrics.json",
    "summary.json",
    "checks.json",
    "final_state.pt",
)
ANALYSIS_REQUIRED = ("summary.json", "checks.json")
SMOKE_PROBE_REQUIRED = ("summary.json", "checks.json")
GATE_THRESHOLDS = {
    "mean_nine_ovr_recall": 0.60,
    "mean_p0_accuracy": 0.65,
    "mean_nine_ovr_balanced_accuracy": 0.70,
}


@dataclasses.dataclass(frozen=True)
class Phase5RContract:
    ledger_path: Path
    ledger_sha256: str
    asset_items: dict[int, dict[str, Any]]
    asset_paths: dict[int, Path]
    asset_digests: dict[int, str]
    asset_unit_hashes: dict[int, str]
    conditions: tuple[Condition, ...]
    burn_in_steps: int
    rank: int


@dataclasses.dataclass(frozen=True)
class _OptimizerResult:
    displacement: torch.Tensor
    metrics: dict[str, Any]

    def mapping(self) -> dict[str, Any]:
        return self.metrics


def _load_phase5_contract(store: UnitStore) -> Phase5RContract:
    ledger_path = store.root / "ledgers" / "phase5.json"
    ledger = _read_json(ledger_path)
    if ledger.get("study_hash") != store.study.config_hash or ledger.get("phase") != "phase5":
        raise RuntimeError("the frozen Phase 5 ledger is incompatible with this study")
    asset_items: dict[int, dict[str, Any]] = {}
    conditions: dict[str, Condition] = {}
    burn_values = set()
    rank_values = set()
    for item in ledger["items"]:
        action = item["action"]
        parameters = item["parameters"]
        if action == "mixture_assets":
            asset_items[int(parameters["index"])] = item
        elif action == "mixture_burn_in":
            burn_values.add(int(parameters["burn_in_steps"]))
        elif action == "mixture_trajectory":
            condition = condition_from_mapping(parameters["condition"])
            conditions[condition.name] = condition
            burn_values.add(int(parameters["burn_in_steps"]))
            rank_values.add(int(parameters["rank"]))
    expected = {
        "full_space",
        "head_only",
        "random_rank_matched",
        "static_subgd",
        "adaptive_floor_0.1",
    }
    if set(asset_items) != set(range(1, store.study.phase5_replicas + 1)):
        raise RuntimeError("frozen Phase 5 ledger does not contain all source assets")
    if set(conditions) != expected or len(burn_values) != 1 or len(rank_values) != 1:
        raise RuntimeError("frozen Phase 5 treatment contract is not the reviewed design")
    ordered = tuple(conditions[name] for name in (
        "full_space",
        "head_only",
        "random_rank_matched",
        "static_subgd",
        "adaptive_floor_0.1",
    ))
    asset_paths = {}
    asset_digests = {}
    asset_unit_hashes = {}
    for index, item in asset_items.items():
        path = store.completed(item["unit"], tuple(item["required"]))
        if path is None:
            raise RuntimeError(f"missing immutable source asset for replica {index}")
        digest = file_hash(path / "assets.pt")
        integrity = _read_json(path / "integrity.json")
        if integrity.get("assets.pt") != digest:
            raise RuntimeError(f"source asset integrity failed for replica {index}")
        asset_paths[index] = path
        asset_digests[index] = digest
        asset_unit_hashes[index] = canonical_hash(item["unit"])
    return Phase5RContract(
        ledger_path=ledger_path,
        ledger_sha256=file_hash(ledger_path),
        asset_items=asset_items,
        asset_paths=asset_paths,
        asset_digests=asset_digests,
        asset_unit_hashes=asset_unit_hashes,
        conditions=ordered,
        burn_in_steps=next(iter(burn_values)),
        rank=next(iter(rank_values)),
    )


def _source_asset(
    store: UnitStore,
    contract: Phase5RContract,
    index: int,
) -> tuple[Path, str, str]:
    del store
    return (
        contract.asset_paths[index],
        contract.asset_digests[index],
        contract.asset_unit_hashes[index],
    )


def _optimizer_contract(store: UnitStore) -> dict[str, Any]:
    learner = store.study.protocol.learner
    maximum_evaluations = math.ceil(
        learner.inner_steps * float(learner.lbfgs_max_eval_factor)
    )
    return {
        "name": learner.optimizer,
        "learning_rate": learner.learning_rate,
        "inner_steps": learner.inner_steps,
        "max_eval": maximum_evaluations,
        "history_size": learner.lbfgs_history_size,
        "line_search_fn": learner.lbfgs_line_search_fn,
        "tolerance_grad": learner.lbfgs_tolerance_grad,
        "tolerance_change": learner.lbfgs_tolerance_change,
    }


def burn_unit(
    store: UnitStore,
    contract: Phase5RContract,
    index: int,
    *,
    phase: str = PHASE,
    burn_in_steps: int | None = None,
) -> dict[str, Any]:
    _, asset_digest, asset_unit_hash = _source_asset(store, contract, index)
    steps = contract.burn_in_steps if burn_in_steps is None else int(burn_in_steps)
    return store.unit(
        phase,
        "mixture_burn_in",
        index,
        environment="digit9_mixture",
        detail={
            "schema_version": SCHEMA_VERSION,
            "burn_in_steps": steps,
            "source_phase5_ledger_sha256": contract.ledger_sha256,
            "source_asset_sha256": asset_digest,
            "source_asset_unit_hash": asset_unit_hash,
            "proposal_fisher_timing": "prior_archive",
            "optimizer": _optimizer_contract(store),
        },
    )


def gate_unit(
    store: UnitStore,
    contract: Phase5RContract,
    replicas: tuple[int, ...],
) -> dict[str, Any]:
    return store.unit(
        PHASE,
        "viability_gate",
        0,
        environment="digit9_mixture",
        detail={
            "schema_version": SCHEMA_VERSION,
            "replicas": list(replicas),
            "burn_unit_hashes": [
                canonical_hash(burn_unit(store, contract, index)) for index in replicas
            ],
            "thresholds": GATE_THRESHOLDS,
        },
    )


def trajectory_unit(
    store: UnitStore,
    contract: Phase5RContract,
    index: int,
    condition: Condition,
) -> dict[str, Any]:
    _, asset_digest, asset_unit_hash = _source_asset(store, contract, index)
    return store.unit(
        PHASE,
        "mixture_trajectory",
        index,
        environment="digit9_mixture",
        condition=condition.name,
        detail={
            "schema_version": SCHEMA_VERSION,
            "burn_in_steps": contract.burn_in_steps,
            "adaptation_rank": contract.rank,
            "condition": condition.mapping(),
            "source_phase5_ledger_sha256": contract.ledger_sha256,
            "source_asset_sha256": asset_digest,
            "source_asset_unit_hash": asset_unit_hash,
            "burn_unit_hash": canonical_hash(burn_unit(store, contract, index)),
            "gate_unit_hash": canonical_hash(
                gate_unit(
                    store,
                    contract,
                    tuple(range(1, store.study.phase5_replicas + 1)),
                )
            ),
            "optimizer": _optimizer_contract(store),
        },
    )


def analysis_unit(
    store: UnitStore,
    contract: Phase5RContract,
    replicas: tuple[int, ...],
) -> dict[str, Any]:
    return store.unit(
        PHASE,
        "analysis",
        0,
        environment="digit9_mixture",
        detail={
            "schema_version": SCHEMA_VERSION,
            "replicas": list(replicas),
            "conditions": [condition.mapping() for condition in contract.conditions],
            "trajectory_unit_hashes": [
                canonical_hash(trajectory_unit(store, contract, index, condition))
                for index in replicas
                for condition in contract.conditions
            ],
        },
    )


def smoke_probe_unit(
    store: UnitStore,
    contract: Phase5RContract,
) -> dict[str, Any]:
    _, asset_digest, asset_unit_hash = _source_asset(store, contract, 1)
    return store.unit(
        SMOKE_PHASE,
        "coordinate_probe",
        1,
        environment="digit9_mixture",
        detail={
            "schema_version": SCHEMA_VERSION,
            "step": contract.burn_in_steps,
            "conditions": [condition.mapping() for condition in contract.conditions[1:]],
            "source_asset_sha256": asset_digest,
            "source_asset_unit_hash": asset_unit_hash,
            "smoke_burn_unit_hash": canonical_hash(
                burn_unit(store, contract, 1, phase=SMOKE_PHASE)
            ),
        },
    )


def _optimize(
    store: UnitStore,
    model: torch.nn.Module,
    layout: Any,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    fisher: LowRankDiagonalFisher,
    factor: CoordinateFactor,
) -> _OptimizerResult:
    optimizer = _optimizer_contract(store)
    training_fisher = fisher.to(device=inputs.device, dtype=inputs.dtype)
    if factor.kind == "identity":
        config = OptimizerConfig(
            name="lbfgs",
            learning_rate=float(optimizer["learning_rate"]),
            inner_steps=int(optimizer["inner_steps"]),
            ewc_strength=1.0,
            lbfgs_history_size=int(optimizer["history_size"]),
            lbfgs_max_eval_factor=(
                int(optimizer["max_eval"]) / int(optimizer["inner_steps"])
            ),
            lbfgs_tolerance_grad=float(optimizer["tolerance_grad"]),
            lbfgs_tolerance_change=float(optimizer["tolerance_change"]),
            lbfgs_line_search_fn=str(optimizer["line_search_fn"]),
        )
        result = take_ewc_proposal(
            model,
            layout,
            inputs,
            targets,
            training_fisher,
            config,
            build_optimizer(model, config),
            adaptation_weight=MIXTURE_PI,
        )
        return _OptimizerResult(
            result.displacement,
            {
                **result.metrics_mapping(),
                "solver_parameterization": "direct_full_space",
                "factor": factor.mapping(),
            },
        )
    result = coordinate_lbfgs_update(
        model,
        layout,
        inputs,
        targets,
        training_fisher,
        factor,
        strength=mixture_ewc_strength(MIXTURE_PI),
        learning_rate=float(optimizer["learning_rate"]),
        inner_steps=int(optimizer["inner_steps"]),
        max_eval=int(optimizer["max_eval"]),
        history_size=int(optimizer["history_size"]),
        tolerance_grad=float(optimizer["tolerance_grad"]),
        tolerance_change=float(optimizer["tolerance_change"]),
    )
    return _OptimizerResult(
        result.displacement,
        {
            **result.mapping(),
            "solver_parameterization": "affine_coordinate",
        },
    )


def _training_geometry(geometry: AdaptationGeometry, dtype: torch.dtype) -> AdaptationGeometry:
    return AdaptationGeometry(
        geometry.basis.to(dtype=dtype),
        geometry.eigenvalues.to(dtype=dtype),
    )


def run_burn_in(
    store: UnitStore,
    contract: Phase5RContract,
    index: int,
    *,
    resume: bool,
    phase: str = PHASE,
    burn_in_steps: int | None = None,
) -> Path:
    steps = contract.burn_in_steps if burn_in_steps is None else int(burn_in_steps)
    unit = burn_unit(store, contract, index, phase=phase, burn_in_steps=steps)
    session = store.begin(unit, BURN_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, BURN_REQUIRED)
        assert completed is not None
        return completed
    asset_path, asset_digest, _ = _source_asset(store, contract, index)
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    plan = MixtureStreamPlan.from_mapping(assets["stream_plan"])
    if not 0 < steps < len(plan.p_values):
        raise ValueError("Phase 5R burn-in must leave an endpoint for evaluation")
    device, training_dtype, matrix_dtype = runtime(store.study)
    model, layout, fisher = _load_initial_state(store, assets, index)
    evaluation_inputs, evaluation_targets = _evaluation_assets(
        assets,
        device=device,
        dtype=training_dtype,
    )
    factor = identity_factor(
        layout.total_numel,
        device=device,
        dtype=training_dtype,
    )
    rows = []
    parameters = [layout.flatten_module(model, detach=True).cpu()]
    displacements = []
    learner_seconds = fisher_seconds = 0.0
    started = time.perf_counter()
    for step in range(steps):
        p = float(plan.p_values[step])
        evaluation = evaluate_mixture(model, evaluation_inputs, evaluation_targets, p)
        inputs = assets["stream_inputs"][step].to(device=device, dtype=training_dtype)
        targets = assets["stream_targets"][step].to(device=device)
        fisher_started = time.perf_counter()
        fresh, _ = _fresh_fisher(
            model,
            layout,
            inputs,
            targets,
            matrix_dtype=matrix_dtype,
        )
        fisher_seconds += time.perf_counter() - fisher_started
        archive_trace_before = float(fisher.diagonal_vector().sum())
        learner_started = time.perf_counter()
        result = _optimize(
            store,
            model,
            layout,
            inputs,
            targets,
            fisher,
            factor,
        )
        learner_seconds += time.perf_counter() - learner_started
        displacements.append(result.displacement)
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
                "observations_before_evaluation": step * plan.samples_per_step,
                "condition": "shared_full_space_lbfgs_burn_in",
                "parameter_hash": tensor_content_hash(parameters[-2]),
                "archive_trace_before": archive_trace_before,
                "archive_trace_after": float(fisher.diagonal_vector().sum()),
                "optimizer": result.mapping(),
                **evaluation,
            }
        )
    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    if not torch.equal(parameter_tensor[1:] - parameter_tensor[:-1], displacement_tensor):
        raise RuntimeError("Phase 5R burn-in displacement identity failed")
    endpoint_p = float(plan.p_values[steps])
    endpoint = {
        "step": steps,
        "p": endpoint_p,
        "observations_before_evaluation": steps * plan.samples_per_step,
        **evaluate_mixture(model, evaluation_inputs, evaluation_targets, endpoint_p),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "environment": "digit9_mixture",
        "replica_index": index,
        "burn_in_steps": steps,
        "observations_used": steps * plan.samples_per_step,
        "source_asset_sha256": asset_digest,
        "proposal_fisher_timing": "prior_archive",
        "optimizer": _optimizer_contract(store),
        "endpoint": endpoint,
        "learner_wall_seconds": learner_seconds,
        "fisher_wall_seconds": fisher_seconds,
        "total_wall_time_seconds": time.perf_counter() - started,
    }
    checks = {
        "all_finite": _finite_tree(rows) and _finite_tree(summary),
        "displacement_identity": True,
        "parameter_count": layout.total_numel,
        "source_asset_integrity_verified": True,
        "original_phase5_burn_in_loaded": False,
        "original_phase5_basis_loaded": False,
        "extra_ridge_kappa": 0.0,
        "fisher_pi": MIXTURE_PI,
        "proposal_fisher_timing": "prior_archive",
    }
    if not checks["all_finite"]:
        raise RuntimeError(f"Phase 5R burn-in checks failed: {checks}")
    session.write_torch(
        "burn_in.pt",
        {
            "model": _state_dict_cpu(model),
            "fisher": fisher.artifact_mapping(),
            "parameters": parameter_tensor,
            "shadow_displacements": displacement_tensor,
            "parameter_layout": layout.metadata(),
        },
    )
    session.write_json("metrics.json", rows)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, BURN_REQUIRED)


def run_gate(
    store: UnitStore,
    contract: Phase5RContract,
    replicas: tuple[int, ...],
    *,
    resume: bool,
) -> Path:
    if replicas != tuple(range(1, store.study.phase5_replicas + 1)):
        raise ValueError("production viability gate requires all frozen replicas")
    unit = gate_unit(store, contract, replicas)
    session = store.begin(unit, GATE_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, GATE_REQUIRED)
        assert completed is not None
        return completed
    endpoints = []
    burn_hashes = []
    for index in replicas:
        path = store.completed(burn_unit(store, contract, index), BURN_REQUIRED)
        if path is None:
            raise RuntimeError(f"missing repaired burn-in for replica {index}")
        endpoints.append(_read_json(path / "summary.json")["endpoint"])
        burn_hashes.append(file_hash(path / "summary.json"))
    observed, criteria, passed = evaluate_gate(endpoints)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "kind": "viability_gate",
        "replicas": list(replicas),
        "endpoint_p": contract.burn_in_steps / 99,
        "thresholds": GATE_THRESHOLDS,
        "observed": observed,
        "criteria": criteria,
        "passed": passed,
        "burn_summary_hashes": burn_hashes,
        "interpretation": (
            "The common full-space learner is viable; treatment execution is unlocked."
            if passed
            else "The common learner failed the reviewed validity gate; treatments remain locked."
        ),
    }
    checks = {
        "all_finite": _finite_tree(summary),
        "replica_count": len(endpoints),
        "all_burns_complete": len(endpoints) == store.study.phase5_replicas,
        "thresholds_frozen": GATE_THRESHOLDS,
    }
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, GATE_REQUIRED)


def evaluate_gate(
    endpoints: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, bool], bool]:
    if not endpoints:
        raise ValueError("viability gate requires endpoint measurements")
    observed = {
        "mean_nine_ovr_recall": statistics.fmean(
            float(row["nine_ovr_recall"]) for row in endpoints
        ),
        "mean_p0_accuracy": statistics.fmean(float(row["p0_accuracy"]) for row in endpoints),
        "mean_nine_ovr_balanced_accuracy": statistics.fmean(
            float(row["nine_ovr_balanced_accuracy"]) for row in endpoints
        ),
    }
    criteria = {
        name: observed[name] >= threshold for name, threshold in GATE_THRESHOLDS.items()
    }
    return observed, criteria, all(criteria.values())


def _gate_passed(store: UnitStore, contract: Phase5RContract) -> bool:
    replicas = tuple(range(1, store.study.phase5_replicas + 1))
    path = store.completed(gate_unit(store, contract, replicas), GATE_REQUIRED)
    return path is not None and bool(_read_json(path / "summary.json")["passed"])


def _factor_for_condition(
    condition: Condition,
    geometry: AdaptationGeometry,
    head_mask: torch.Tensor,
    *,
    alpha: float,
    training_dtype: torch.dtype,
) -> CoordinateFactor:
    if condition.kind == "full_space":
        return identity_factor(
            geometry.parameter_count,
            device=geometry.basis.device,
            dtype=training_dtype,
        )
    if condition.kind == "head_only":
        return selection_factor(head_mask)
    if condition.kind not in {
        "random_rank_matched",
        "static_projector",
        "static_subgd",
        "online_subgd",
        "adaptive_subgd",
    }:
        raise RuntimeError(f"unsupported Phase 5R condition: {condition.kind}")
    training_geometry = _training_geometry(geometry, training_dtype)
    return geometry_factor(
        training_geometry,
        alpha=alpha,
        epsilon=condition.epsilon,
        projector_only=condition.kind == "static_projector",
    )


def run_trajectory(
    store: UnitStore,
    contract: Phase5RContract,
    index: int,
    condition: Condition,
    *,
    resume: bool,
) -> Path:
    if not _gate_passed(store, contract):
        raise RuntimeError("Phase 5R treatment is locked by the viability gate")
    unit = trajectory_unit(store, contract, index, condition)
    session = store.begin(unit, TRAJECTORY_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, TRAJECTORY_REQUIRED)
        assert completed is not None
        return completed
    asset_path, asset_digest, _ = _source_asset(store, contract, index)
    burn_path = store.completed(burn_unit(store, contract, index), BURN_REQUIRED)
    if burn_path is None:
        raise RuntimeError(f"missing Phase 5R burn-in for replica {index}")
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    burn = torch.load(burn_path / "burn_in.pt", map_location="cpu", weights_only=False)
    plan = MixtureStreamPlan.from_mapping(assets["stream_plan"])
    device, training_dtype, matrix_dtype = runtime(store.study)
    model, layout = build_gauge_fixed_model(
        store.study.seed("phase5r:trajectory_model", index),
        device=device,
        dtype=training_dtype,
    )
    model.load_state_dict(burn["model"])
    layout.assert_metadata(burn["parameter_layout"])
    fisher = representation_from_artifact(burn["fisher"], device=device)
    if not isinstance(fisher, LowRankDiagonalFisher) or fisher.rank != 8:
        raise RuntimeError("Phase 5R burn-in Fisher is not rank eight plus diagonal")
    fisher = fisher.to(device=device, dtype=matrix_dtype)
    geometry = AdaptationGeometry.from_observations(
        burn["shadow_displacements"].to(device=device, dtype=matrix_dtype),
        rank=contract.rank,
    )
    if condition.kind == "random_rank_matched":
        geometry = random_geometry(
            layout.total_numel,
            geometry.eigenvalues,
            seed=store.study.seed("phase5r:random_basis", index),
        )
    controller_state = InnovationControllerState()
    head_mask = _head_mask(layout, device=device, dtype=training_dtype)
    evaluation_inputs, evaluation_targets = _evaluation_assets(
        assets,
        device=device,
        dtype=training_dtype,
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
    for step in range(contract.burn_in_steps, len(plan.p_values) - 1):
        p = float(plan.p_values[step])
        before = layout.flatten_module(model, detach=True)
        parameters.append(before.cpu())
        bases.append(geometry.basis.detach().cpu())
        eigenvalues.append(geometry.eigenvalues.detach().cpu())
        evaluation = evaluate_mixture(model, evaluation_inputs, evaluation_targets, p)
        inputs = assets["stream_inputs"][step].to(device=device, dtype=training_dtype)
        targets = assets["stream_targets"][step].to(device=device)
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
        if condition.kind == "full_space":
            factor = _factor_for_condition(
                condition,
                geometry,
                head_mask,
                alpha=alpha,
                training_dtype=training_dtype,
            )
            learner_started = time.perf_counter()
            result = _optimize(
                store,
                model,
                layout,
                inputs,
                targets,
                fisher,
                factor,
            )
            learner_seconds += time.perf_counter() - learner_started
            shadow = result.displacement.to(device=device, dtype=matrix_dtype)
        else:
            shadow_factor = identity_factor(
                layout.total_numel,
                device=device,
                dtype=training_dtype,
            )
            shadow_started = time.perf_counter()
            shadow_result = _optimize(
                store,
                model,
                layout,
                inputs,
                targets,
                fisher,
                shadow_factor,
            )
            shadow_seconds += time.perf_counter() - shadow_started
            shadow = shadow_result.displacement.to(device=device, dtype=matrix_dtype)
            layout.copy_vector_to_module(model, before)
            if condition.kind not in {"head_only", "random_rank_matched", "static_subgd"}:
                innovation = geometry.innovation(shadow)
            if condition.kind == "adaptive_subgd":
                assert condition.controller is not None and innovation is not None
                decision, controller_state = controller_state.decide(
                    innovation,
                    condition.controller,
                )
                alpha = decision.alpha
                beta = decision.beta
            factor = _factor_for_condition(
                condition,
                geometry,
                head_mask,
                alpha=alpha,
                training_dtype=training_dtype,
            )
            learner_started = time.perf_counter()
            result = _optimize(
                store,
                model,
                layout,
                inputs,
                targets,
                fisher,
                factor,
            )
            learner_seconds += time.perf_counter() - learner_started
        actual = result.displacement
        displacements.append(actual)
        shadow_displacements.append(shadow.cpu())
        distance = 0.0
        if condition.kind == "online_subgd":
            raise RuntimeError("the frozen Phase 5R conditions contain no fixed-rate online method")
        if condition.kind == "adaptive_subgd":
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
                "post_burn_in_step": step - contract.burn_in_steps,
                "post_burn_in_observations": (
                    step - contract.burn_in_steps
                ) * plan.samples_per_step,
                "observations_before_evaluation": step * plan.samples_per_step,
                "p": p,
                "schedule_kind": "digit9_mixture",
                "condition": condition.name,
                "parameter_hash": tensor_content_hash(before.cpu()),
                "archive_trace": float(fisher.diagonal_vector().sum()),
                "optimizer": result.mapping(),
                "shadow": {
                    "displacement_norm": float(torch.linalg.vector_norm(shadow)),
                },
                "geometry": {
                    "rank": geometry.rank,
                    "innovation": innovation,
                    "smoothed_innovation": controller_state.smoothed_innovation,
                    "alpha": alpha,
                    "beta": beta,
                    "epsilon": condition.epsilon,
                    "orthogonal_gain": (1 - alpha) + alpha * condition.epsilon,
                    "projector_distance": distance,
                    "factor": factor.mapping(),
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
            "post_burn_in_step": step - contract.burn_in_steps,
            "post_burn_in_observations": (
                step - contract.burn_in_steps
            ) * plan.samples_per_step,
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
        raise RuntimeError("Phase 5R trajectory displacement identity failed")
    auc_fields = (
        "current_nll",
        "current_accuracy",
        "p0_nll",
        "p0_accuracy",
        "worst_panel_nll",
        "nine_ovr_accuracy",
        "nine_ovr_balanced_accuracy",
        "nine_ovr_recall",
        "nine_ovr_specificity",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "environment": "digit9_mixture",
        "replica_index": index,
        "schedule_kind": "digit9_mixture",
        "condition": condition.mapping(),
        "burn_in_steps": contract.burn_in_steps,
        "adaptation_rank": contract.rank,
        "source_asset_sha256": asset_digest,
        **{f"post_burn_in_{field}_auc": _auc(rows, field) for field in auc_fields},
        "learner_wall_seconds": learner_seconds,
        "shadow_wall_seconds": shadow_seconds,
        "fisher_wall_seconds": fisher_seconds,
        "total_wall_time_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "peak_cuda_memory_bytes": (
            0 if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
        ),
    }
    checks = {
        "all_finite": _finite_tree(rows) and _finite_tree(summary),
        "displacement_identity": True,
        "parameter_count": layout.total_numel,
        "burn_in_pairing": torch.equal(burn["parameters"][-1], parameter_tensor[0]),
        "viability_gate_passed": True,
        "source_asset_integrity_verified": True,
        "phase5r_burn_in_loaded": True,
        "original_phase5_burn_in_loaded": False,
        "original_phase5_basis_loaded": False,
        "coordinate_name": "p_t",
        "fisher_pi": MIXTURE_PI,
        "extra_ridge_kappa": 0.0,
        "proposal_fisher_timing": "prior_archive",
    }
    if not checks["all_finite"] or not checks["burn_in_pairing"]:
        raise RuntimeError(f"Phase 5R trajectory checks failed: {checks}")
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
    return store.finish(session, TRAJECTORY_REQUIRED)


def _paired_effect(
    method: list[float],
    comparator: list[float],
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    if len(method) != len(comparator) or not method:
        raise ValueError("paired effects require equal nonempty samples")
    gains = [
        (treatment - reference) if higher_is_better else (reference - treatment)
        for treatment, reference in zip(method, comparator)
    ]
    mean = statistics.fmean(gains)
    standard_error = 0.0 if len(gains) < 2 else statistics.stdev(gains) / math.sqrt(len(gains))
    critical = 0.0 if len(gains) < 2 else float(student_t.ppf(0.975, len(gains) - 1))
    standard_deviation = 0.0 if len(gains) < 2 else statistics.stdev(gains)
    return {
        "mean_gain": mean,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
        "median_gain": statistics.median(gains),
        "standardized_effect": (
            None if standard_deviation == 0 else mean / standard_deviation
        ),
        "favorable_replicas": sum(value > 0 for value in gains),
        "replicas": len(gains),
        "gains": gains,
    }


def run_analysis(
    store: UnitStore,
    contract: Phase5RContract,
    replicas: tuple[int, ...],
    *,
    resume: bool,
) -> Path:
    if replicas != tuple(range(1, store.study.phase5_replicas + 1)):
        raise ValueError("Phase 5R analysis requires the frozen 16-replica cohort")
    unit = analysis_unit(store, contract, replicas)
    session = store.begin(unit, ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_hashes = []
    for index in replicas:
        for condition in contract.conditions:
            path = store.completed(
                trajectory_unit(store, contract, index, condition),
                TRAJECTORY_REQUIRED,
            )
            if path is None:
                raise RuntimeError(f"missing Phase 5R trajectory {index}/{condition.name}")
            by_condition[condition.name].append(_read_json(path / "summary.json"))
            trajectory_hashes.append(file_hash(path / "summary.json"))
    fields = {
        "current_nll": False,
        "current_accuracy": True,
        "p0_nll": False,
        "p0_accuracy": True,
        "nine_ovr_accuracy": True,
        "nine_ovr_balanced_accuracy": True,
        "nine_ovr_recall": True,
        "nine_ovr_specificity": True,
    }
    aggregate = []
    for condition in contract.conditions:
        values = by_condition[condition.name]
        aggregate.append(
            {
                "condition": condition.name,
                "replicas": len(values),
                **{
                    f"mean_{field}_auc": statistics.fmean(
                        float(value[f"post_burn_in_{field}_auc"]) for value in values
                    )
                    for field in fields
                },
                "mean_wall_seconds": statistics.fmean(
                    float(value["total_wall_time_seconds"]) for value in values
                ),
            }
        )
    comparisons = []
    full = by_condition["full_space"]
    for condition in contract.conditions:
        if condition.name == "full_space":
            continue
        method = by_condition[condition.name]
        comparisons.append(
            {
                "condition": condition.name,
                **{
                    f"{field}_vs_full": _paired_effect(
                        [float(value[f"post_burn_in_{field}_auc"]) for value in method],
                        [float(value[f"post_burn_in_{field}_auc"]) for value in full],
                        higher_is_better=higher,
                    )
                    for field, higher in fields.items()
                },
            }
        )
    gate_path = store.completed(gate_unit(store, contract, replicas), GATE_REQUIRED)
    assert gate_path is not None
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "environment": "digit9_mixture",
        "classification": "post_diagnostic_development_not_confirmation",
        "gate": _read_json(gate_path / "summary.json"),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "trajectory_summary_hashes": trajectory_hashes,
        "rotation_and_mixture_evidence_pooled": False,
        "original_phase5_valid_for_subgd_efficacy": False,
    }
    checks = {
        "all_finite": _finite_tree(summary),
        "replica_count": len(replicas),
        "condition_count": len(contract.conditions),
        "trajectory_count": len(trajectory_hashes),
        "gate_passed": bool(summary["gate"]["passed"]),
        "analysis_hash": canonical_hash(summary),
    }
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, ANALYSIS_REQUIRED)


def run_coordinate_smoke(
    store: UnitStore,
    contract: Phase5RContract,
    *,
    resume: bool,
) -> Path:
    unit = smoke_probe_unit(store, contract)
    session = store.begin(unit, SMOKE_PROBE_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, SMOKE_PROBE_REQUIRED)
        assert completed is not None
        return completed
    asset_path, _, _ = _source_asset(store, contract, 1)
    burn_path = store.completed(
        burn_unit(store, contract, 1, phase=SMOKE_PHASE),
        BURN_REQUIRED,
    )
    if burn_path is None:
        raise RuntimeError("coordinate smoke requires the completed CUDA burn smoke")
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    burn = torch.load(burn_path / "burn_in.pt", map_location="cpu", weights_only=False)
    device, training_dtype, matrix_dtype = runtime(store.study)
    step = contract.burn_in_steps
    inputs = assets["stream_inputs"][step].to(device=device, dtype=training_dtype)
    targets = assets["stream_targets"][step].to(device=device)

    shadow_model, shadow_layout = build_gauge_fixed_model(
        store.study.seed("phase5r:smoke_shadow", 1),
        device=device,
        dtype=training_dtype,
    )
    shadow_model.load_state_dict(burn["model"])
    shadow_layout.assert_metadata(burn["parameter_layout"])
    fisher = representation_from_artifact(burn["fisher"], device=device)
    if not isinstance(fisher, LowRankDiagonalFisher):
        raise RuntimeError("coordinate smoke Fisher is not rank plus diagonal")
    fisher = fisher.to(device=device, dtype=matrix_dtype)
    shadow = _optimize(
        store,
        shadow_model,
        shadow_layout,
        inputs,
        targets,
        fisher,
        identity_factor(
            shadow_layout.total_numel,
            device=device,
            dtype=training_dtype,
        ),
    ).displacement.to(device=device, dtype=matrix_dtype)
    learned = AdaptationGeometry.from_observations(
        burn["shadow_displacements"].to(device=device, dtype=matrix_dtype),
        rank=contract.rank,
    )
    rows = []
    for condition in contract.conditions[1:]:
        model, layout = build_gauge_fixed_model(
            store.study.seed(f"phase5r:smoke:{condition.name}", 1),
            device=device,
            dtype=training_dtype,
        )
        model.load_state_dict(burn["model"])
        layout.assert_metadata(burn["parameter_layout"])
        geometry = learned
        if condition.kind == "random_rank_matched":
            geometry = random_geometry(
                layout.total_numel,
                learned.eigenvalues,
                seed=store.study.seed("phase5r:random_basis", 1),
            )
        alpha = 1.0
        beta = None
        innovation = None
        if condition.kind == "adaptive_subgd":
            assert condition.controller is not None
            innovation = geometry.innovation(shadow)
            decision, _ = InnovationControllerState().decide(
                innovation,
                condition.controller,
            )
            alpha = decision.alpha
            beta = decision.beta
        factor = _factor_for_condition(
            condition,
            geometry,
            _head_mask(layout, device=device, dtype=training_dtype),
            alpha=alpha,
            training_dtype=training_dtype,
        )
        result = _optimize(
            store,
            model,
            layout,
            inputs,
            targets,
            fisher,
            factor,
        )
        displacement = result.displacement.to(device=device, dtype=matrix_dtype)
        if factor.kind == "selection":
            assert factor.selected_indices is not None
            mask = torch.zeros(layout.total_numel, device=device, dtype=torch.bool)
            mask[factor.selected_indices] = True
            confinement_error = float(torch.linalg.vector_norm(displacement[~mask]))
        elif factor.kind == "low_rank":
            assert factor.basis is not None
            basis = factor.basis.to(dtype=matrix_dtype)
            residual = displacement - basis @ (basis.mT @ displacement)
            confinement_error = float(torch.linalg.vector_norm(residual))
        else:
            confinement_error = 0.0
        rows.append(
            {
                "condition": condition.name,
                "factor": factor.mapping(),
                "innovation": innovation,
                "alpha": alpha,
                "beta": beta,
                "confinement_error": confinement_error,
                "optimizer": result.mapping(),
            }
        )
    tolerance = 128 * torch.finfo(training_dtype).eps
    checks = {
        "all_finite": _finite_tree(rows),
        "all_objectives_decreased": all(
            float(row["optimizer"]["objective_decrease"]) >= 0 for row in rows
        ),
        "singular_factors_confined": all(
            row["factor"]["kind"] not in {"selection", "low_rank"}
            or float(row["confinement_error"]) <= tolerance
            for row in rows
        ),
        "condition_count": len(rows),
        "cuda_exercised": device.type == "cuda",
    }
    if not all(
        value for key, value in checks.items() if key != "condition_count"
    ):
        raise RuntimeError(f"Phase 5R coordinate CUDA smoke failed: {checks}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": SMOKE_PHASE,
        "kind": "coordinate_probe",
        "step": step,
        "rows": rows,
    }
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, SMOKE_PROBE_REQUIRED)


def build_ledger(
    store: UnitStore,
    contract: Phase5RContract,
    replicas: tuple[int, ...],
) -> dict[str, Any]:
    items = []
    for index in replicas:
        items.append(
            {
                "action": "phase5r_burn_in",
                "unit": burn_unit(store, contract, index),
                "required": list(BURN_REQUIRED),
                "parameters": {"index": index},
            }
        )
    items.append(
        {
            "action": "phase5r_gate",
            "unit": gate_unit(store, contract, replicas),
            "required": list(GATE_REQUIRED),
            "parameters": {},
        }
    )
    for index in replicas:
        for condition in contract.conditions:
            items.append(
                {
                    "action": "phase5r_trajectory",
                    "unit": trajectory_unit(store, contract, index, condition),
                    "required": list(TRAJECTORY_REQUIRED),
                    "parameters": {
                        "index": index,
                        "condition": condition.mapping(),
                    },
                }
            )
    items.append(
        {
            "action": "phase5r_analysis",
            "unit": analysis_unit(store, contract, replicas),
            "required": list(ANALYSIS_REQUIRED),
            "parameters": {},
        }
    )
    return {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "source_phase5_ledger_sha256": contract.ledger_sha256,
        "items": items,
    }


def _execute_item(
    store: UnitStore,
    contract: Phase5RContract,
    replicas: tuple[int, ...],
    item: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    action = item["action"]
    if action == "phase5r_burn_in":
        return run_burn_in(
            store,
            contract,
            int(item["parameters"]["index"]),
            resume=resume,
        )
    if action == "phase5r_gate":
        return run_gate(store, contract, replicas, resume=resume)
    if action == "phase5r_trajectory":
        return run_trajectory(
            store,
            contract,
            int(item["parameters"]["index"]),
            condition_from_mapping(item["parameters"]["condition"]),
            resume=resume,
        )
    if action == "phase5r_analysis":
        return run_analysis(store, contract, replicas, resume=resume)
    raise RuntimeError(f"unsupported Phase 5R ledger action: {action}")


def run_smoke(
    store: UnitStore,
    contract: Phase5RContract,
    *,
    resume: bool,
) -> Path:
    burn = burn_unit(store, contract, 1, phase=SMOKE_PHASE)
    probe = smoke_probe_unit(store, contract)
    ledger = {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": SMOKE_PHASE,
        "schema_version": SCHEMA_VERSION,
        "items": [
            {
                "action": "phase5r_smoke_burn_in",
                "unit": burn,
                "required": list(BURN_REQUIRED),
                "parameters": {"index": 1},
            },
            {
                "action": "phase5r_smoke_coordinate_probe",
                "unit": probe,
                "required": list(SMOKE_PROBE_REQUIRED),
                "parameters": {"index": 1},
            },
        ],
    }
    smoke_id = canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "source_hashes": store.sources,
            "source_phase5_ledger_sha256": contract.ledger_sha256,
        }
    )[:12]
    freeze_json(
        store.root / "ledgers" / f"{SMOKE_PHASE}_{smoke_id}.json",
        ledger,
        resume=resume,
    )
    run_burn_in(store, contract, 1, resume=resume, phase=SMOKE_PHASE)
    return run_coordinate_smoke(store, contract, resume=resume)


def run_production(
    store: UnitStore,
    contract: Phase5RContract,
    *,
    notebook: Path,
    resume: bool,
    max_units: int | None,
    max_wall_seconds: float | None,
) -> tuple[int, bool]:
    replicas = tuple(range(1, store.study.phase5_replicas + 1))
    ledger = build_ledger(store, contract, replicas)
    freeze_json(store.root / "ledgers" / f"{PHASE}.json", ledger, resume=resume)
    started = time.monotonic()
    last_refresh = started
    completed_now = 0
    exhausted = False
    for item in ledger["items"]:
        required = tuple(item["required"])
        if store.completed(item["unit"], required) is not None:
            continue
        if max_units is not None and completed_now >= max_units:
            exhausted = True
            break
        if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
            exhausted = True
            break
        if item["action"] == "phase5r_trajectory" and not _gate_passed(store, contract):
            gate_path = store.completed(gate_unit(store, contract, replicas), GATE_REQUIRED)
            if gate_path is None:
                raise RuntimeError("Phase 5R treatment reached before gate completion")
            refresh(store, notebook)
            return completed_now, True
        try:
            path = _execute_item(store, contract, replicas, item, resume=resume)
        except BaseException as error:
            store.record_failure(item["unit"], error)
            refresh(store, notebook)
            raise
        completed_now += 1
        print(f"[phase5r] completed {item['action']}: {path}")
        if item["action"] in {"phase5r_gate", "phase5r_analysis"} or time.monotonic() - last_refresh >= 300:
            refresh(store, notebook)
            last_refresh = time.monotonic()
    refresh(store, notebook)
    return completed_now, exhausted


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.max_units is not None and arguments.max_units < 1:
        raise ValueError("--max-units must be positive")
    if arguments.max_wall_seconds is not None and arguments.max_wall_seconds <= 0:
        raise ValueError("--max-wall-seconds must be positive")
    repo_root = Path(__file__).parents[2]
    study = Plan13Study.from_path(arguments.config)
    store = UnitStore(arguments.output_root, study, repo_root)
    contract = _load_phase5_contract(store)
    if arguments.cuda_smoke:
        path = run_smoke(store, contract, resume=arguments.resume)
        refresh(store, arguments.notebook)
        print(f"[phase5r] CUDA smoke: {path}")
        return
    completed, exhausted = run_production(
        store,
        contract,
        notebook=arguments.notebook,
        resume=arguments.resume,
        max_units=arguments.max_units,
        max_wall_seconds=arguments.max_wall_seconds,
    )
    print(f"[phase5r] completed_now={completed} exhausted={exhausted}")


if __name__ == "__main__":
    main()


__all__ = [
    "ANALYSIS_REQUIRED",
    "BURN_REQUIRED",
    "GATE_REQUIRED",
    "PHASE",
    "SCHEMA_VERSION",
    "TRAJECTORY_REQUIRED",
    "Phase5RContract",
    "analysis_unit",
    "build_ledger",
    "burn_unit",
    "evaluate_gate",
    "gate_unit",
    "run_analysis",
    "run_burn_in",
    "run_coordinate_smoke",
    "run_gate",
    "run_trajectory",
    "trajectory_unit",
]
