"""Predictable structured-gain audits for Plan 9 E9.12--E9.14."""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm.auto import tqdm

from src.derivatives import per_sample_derivatives
from src.fisher import empirical_fisher
from src.hybrid import blend_archive_fisher
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    mnist_nll,
    resolve_device,
    resolve_dtype,
)
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed

from ..phase8_artifacts import load_completed_phase8_run
from ..run_phase8 import _load_source as load_phase8_source
from ..transform import tensor_content_hash
from .artifacts import file_sha256, read_json, validate_completed
from .attribution import _SOURCE_REQUIRED as E9_1_REQUIRED
from .config import AttributionConfig, RetrospectiveConfig
from .gain_config import (
    GainCalibrationConfig,
    GainContractConfig,
    GainEstimatorConfig,
)
from .retrospective import _fisher, _load_inputs, _risk_inner


E9_9_REQUIRED = (
    "source_contract.json",
    "fit_rows.json",
    "attribution_rows.json",
    "optimizer_sensitivity_rows.json",
    "vectors.pt",
    "summary.json",
)
E9_12_REQUIRED = ("source_contract.json", "audit_rows.json", "curvatures.pt", "summary.json")
E9_13_REQUIRED = ("source_contract.json", "solver_rows.json", "predictions.pt", "summary.json")


def ema_gain(half_life_steps: float) -> float:
    """Return an EMA gain whose observation weight halves after the given steps."""

    if not math.isfinite(half_life_steps) or half_life_steps <= 0.0:
        raise ValueError("half-life must be finite and positive")
    return 1.0 - math.exp(math.log(0.5) / half_life_steps)


def stationary_effective_observations(half_life_steps: float, batch_size: int) -> float:
    gain = ema_gain(half_life_steps)
    return float(batch_size) * (2.0 - gain) / gain


def combine_fishers(
    old: LowRankDiagonalFisher,
    new: LowRankDiagonalFisher,
    pi: float,
) -> LowRankDiagonalFisher:
    """Represent ``(1-pi) old + pi new`` without materializing it."""

    if old.shape != new.shape or old.device != new.device or old.dtype != new.dtype:
        raise ValueError("gain curvatures must share shape, dtype, and device")
    if not 0.0 < float(pi) <= 1.0:
        raise ValueError("pi must lie in (0, 1]")
    factors = []
    if old.rank:
        factors.append(math.sqrt(1.0 - float(pi)) * old.factor)
    if new.rank:
        factors.append(math.sqrt(float(pi)) * new.factor)
    factor = (
        torch.cat(factors, dim=1)
        if factors
        else old.factor.new_zeros((old.shape[0], 0))
    )
    diagonal = (
        (1.0 - float(pi)) * old.residual_diagonal
        + float(pi) * new.residual_diagonal
    )
    return LowRankDiagonalFisher(factor, diagonal)


def structured_gain_action(
    old: LowRankDiagonalFisher,
    new: LowRankDiagonalFisher,
    fresh_displacement: Tensor,
    *,
    pi: float,
    relative_damping: float,
) -> tuple[Tensor, dict[str, float | int]]:
    """Apply the local EWC gain through a diagonal-plus-low-rank solve."""

    if fresh_displacement.dtype != old.dtype or fresh_displacement.device != old.device:
        raise ValueError("displacement must share the curvature dtype and device")
    combined = combine_fishers(old, new, pi)
    right_hand_side = float(pi) * new.matvec(fresh_displacement)
    scale = max(float(combined.diagonal_vector().mean()), torch.finfo(old.dtype).tiny)
    damping = float(relative_damping) * scale
    action = combined.damped_solve(right_hand_side, damping)
    residual = combined.matvec(action) + damping * action - right_hand_side
    denominator = max(float(torch.linalg.vector_norm(right_hand_side)), torch.finfo(old.dtype).tiny)
    normalized_residual = float(torch.linalg.vector_norm(residual)) / denominator
    return action, {
        "combined_rank": combined.rank,
        "combined_storage_bytes": combined.storage_bytes(),
        "damping": damping,
        "normalized_residual": normalized_residual,
        "right_hand_side_norm": float(torch.linalg.vector_norm(right_hand_side)),
        "action_norm": float(torch.linalg.vector_norm(action)),
    }


def _relative(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _load_gain_sources(config: GainContractConfig, repo_root: Path) -> dict[str, Any]:
    e9_1_path = repo_root / config.e9_1_run_path
    e9_9_path = repo_root / config.e9_9_run_path
    validate_completed(e9_1_path, required=E9_1_REQUIRED)
    validate_completed(e9_9_path, required=E9_9_REQUIRED)
    if e9_1_path.name != config.e9_1_run_id or e9_9_path.name != config.e9_9_run_id:
        raise ValueError("gain source identities differ")
    e9_1_config = RetrospectiveConfig.from_mapping(read_json(e9_1_path / "config.json"))
    e9_9_config = AttributionConfig.from_mapping(read_json(e9_9_path / "config.json"))
    if e9_9_config.e9_1_run_id != e9_1_config.run_id:
        raise ValueError("E9.9 does not descend from the selected E9.1 artifact")
    retrospective = _load_inputs(e9_1_config, repo_root)
    phase8_sources = {
        design: load_phase8_source(run.config, repo_root)
        for design, run in retrospective["runs"].items()
    }
    return {
        "paths": {"e9_1": e9_1_path, "e9_9": e9_9_path},
        "e9_1_config": e9_1_config,
        "e9_9_config": e9_9_config,
        "retrospective": retrospective,
        "phase8_sources": phase8_sources,
        "e9_1_vectors": torch.load(e9_1_path / "vectors.pt", map_location="cpu", weights_only=True),
        "e9_1_rows": read_json(e9_1_path / "audit_rows.json"),
        "e9_9_vectors": torch.load(e9_9_path / "vectors.pt", map_location="cpu", weights_only=True),
        "e9_9_fit_rows": read_json(e9_9_path / "fit_rows.json"),
    }


def run_gain_contract(
    config: GainContractConfig,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay EWC curvature and freeze compact inputs for gain estimation."""

    started = time.perf_counter()
    inputs = _load_gain_sources(config, repo_root)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=True,
    )
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.dtype)
    matrix_dtype = torch.float64
    fit_lookup = {
        (row["design"], row["schedule"], int(row["step"])): row
        for row in inputs["e9_9_fit_rows"]
        if row["primary_budget"]
    }
    reversal_lookup = {
        (row["design"], row["schedule"], int(row["step"])): bool(row["reversal_window"])
        for row in inputs["e9_1_rows"]
        if row["condition"] == config.condition
    }
    total = sum(
        inputs["e9_1_vectors"][design][schedule][config.condition]["actions"].numel()
        for design in ("single_lap", "double_lap")
        for schedule in inputs["e9_1_config"].schedule_kinds
    )
    progress = tqdm(total=total, desc="E9.12 Fisher replay", unit="transition")
    audit_rows: list[dict[str, Any]] = []
    curvature_artifact: dict[str, Any] = {}
    maximum_trace_error = 0.0
    maximum_displacement_error = 0.0
    for design in ("single_lap", "double_lap"):
        curvature_artifact[design] = {}
        source = inputs["phase8_sources"][design]
        run = inputs["retrospective"]["runs"][design]
        for schedule in inputs["e9_1_config"].schedule_kinds:
            stored = inputs["e9_1_vectors"][design][schedule][config.condition]
            parameters = stored["parameters"].to(torch.float64)
            actions = stored["actions"].to(torch.float64)
            updates = parameters[1:] - parameters[:-1]
            affinities = inputs["e9_9_vectors"][design][schedule][
                str(inputs["e9_9_config"].primary_optimizer_step_budget)
            ]["affinities"].to(torch.float64)
            fresh_displacements = (updates - affinities) / actions[:, None]
            model, layout = build_canonical_model(
                config.replica_seed,
                device=device,
                dtype=training_dtype,
            )
            fisher = source.initial_fisher.to(device=device, dtype=matrix_dtype)
            score_gradients = []
            old_fishers = []
            penalty_fishers = []
            scalar_oracles = []
            scalar_predictions = {
                str(half_life): [] for half_life in config.scalar_half_lives_steps
            }
            scalar_states = {
                half_life: float(actions[0]) for half_life in config.scalar_half_lives_steps
            }
            for step in range(actions.numel()):
                old_fisher = fisher
                old_fishers.append(old_fisher.artifact_mapping())
                for half_life in config.scalar_half_lives_steps:
                    scalar_predictions[str(half_life)].append(scalar_states[half_life])
                layout.copy_vector_to_module(
                    model, parameters[step].to(device=device, dtype=training_dtype)
                )
                stream = source.streams[schedule]
                batch_inputs = stream["inputs"][step].to(device=device, dtype=training_dtype)
                batch_targets = stream["targets"][step].to(device=device)
                gradients = per_sample_derivatives(
                    model,
                    batch_inputs,
                    batch_targets,
                    mnist_nll,
                    layout,
                    strategy="vmap",
                ).gradients.to(dtype=matrix_dtype)
                fresh = empirical_fisher(gradients)
                score_gradients.append(gradients.cpu())
                pi = float(actions[step])
                if step == 0:
                    penalty_fisher = old_fisher
                    reconstructed_trace = float(old_fisher.diagonal_vector().sum())
                    lanczos = None
                else:
                    update = blend_archive_fisher(
                        old_fisher,
                        fresh,
                        blend_gain=pi,
                        rank=config.fisher_rank,
                        lanczos_seed=derive_component_seed(
                            run.config.replica_seed,
                            "plan8_update_lanczos:"
                            f"source={run.config.source_kind}:schedule={schedule}:"
                            f"condition={config.condition}:step={step}",
                        ),
                    )
                    penalty_fisher = update.representation
                    reconstructed_trace = update.candidate_trace
                    lanczos = update.lanczos.mapping()
                fisher = penalty_fisher
                penalty_fishers.append(penalty_fisher.artifact_mapping())
                source_update = run.trajectory_metrics[schedule][config.condition][step][
                    "fisher_update"
                ]
                source_trace = float(source_update["candidate_trace"])
                trace_error = abs(reconstructed_trace - source_trace) / max(abs(source_trace), 1e-12)
                maximum_trace_error = max(maximum_trace_error, trace_error)
                denominator = float(old_fisher.quadratic(fresh_displacements[step].to(device)))
                numerator = float(
                    updates[step].to(device) @ old_fisher.matvec(fresh_displacements[step].to(device))
                )
                scalar_oracle = pi if denominator <= 1e-15 else numerator / denominator
                scalar_oracles.append(scalar_oracle)
                for half_life in config.scalar_half_lives_steps:
                    gain = ema_gain(half_life)
                    scalar_states[half_life] = (
                        (1.0 - gain) * scalar_states[half_life] + gain * scalar_oracle
                    )
                fit_row = fit_lookup[(design, schedule, step)]
                norm_error = abs(
                    float(torch.linalg.vector_norm(fresh_displacements[step]))
                    - float(fit_row["fresh_displacement_norm"])
                ) / max(float(fit_row["fresh_displacement_norm"]), 1e-12)
                maximum_displacement_error = max(maximum_displacement_error, norm_error)
                audit_rows.append(
                    {
                        "design": design,
                        "schedule": schedule,
                        "step": step,
                        "angle_degrees": float(run.trajectory_metrics[schedule][config.condition][step]["angle_degrees"]),
                        "leg_id": int(run.trajectory_metrics[schedule][config.condition][step]["leg_id"]),
                        "applied_pi": pi,
                        "cold_start": bool(run.trajectory_metrics[schedule][config.condition][step]["controller"]["cold_start_active"]),
                        "reversal_window": reversal_lookup.get((design, schedule, step), False),
                        "source_candidate_trace": source_trace,
                        "reconstructed_candidate_trace": reconstructed_trace,
                        "candidate_trace_relative_error": trace_error,
                        "fresh_displacement_norm_relative_error": norm_error,
                        "scalar_oracle_gain": scalar_oracle,
                        "old_fisher_storage_bytes": old_fisher.storage_bytes(),
                        "penalty_fisher_storage_bytes": penalty_fisher.storage_bytes(),
                        "score_parameter_hash": tensor_content_hash(parameters[step].to(training_dtype)),
                        "lanczos": lanczos,
                    }
                )
                progress.update(1)
            curvature_artifact[design][schedule] = {
                "actions": actions,
                "updates": updates,
                "fresh_displacements": fresh_displacements,
                "score_gradients": torch.stack(score_gradients),
                "old_fishers": old_fishers,
                "penalty_fishers": penalty_fishers,
                "scalar_oracle_gains": torch.tensor(scalar_oracles, dtype=torch.float64),
                "scalar_predictions": {
                    key: torch.tensor(value, dtype=torch.float64)
                    for key, value in scalar_predictions.items()
                },
            }
    progress.close()
    checks = {
        "all_required_vectors_present": total == 400,
        "maximum_candidate_trace_relative_error": maximum_trace_error,
        "maximum_fresh_displacement_norm_relative_error": maximum_displacement_error,
        "fisher_trace_replay_passed": maximum_trace_error <= config.trace_relative_tolerance,
        "fresh_displacement_reconstruction_passed": (
            maximum_displacement_error <= config.displacement_relative_tolerance
        ),
        "current_batch_excluded_from_old_fisher": True,
        "learner_retrained": False,
    }
    status = "supported" if all(
        checks[name]
        for name in (
            "all_required_vectors_present",
            "fisher_trace_replay_passed",
            "fresh_displacement_reconstruction_passed",
            "current_batch_excluded_from_old_fisher",
        )
    ) else "rejected"
    summary = {
        "study": "E9.12",
        "status": status,
        "transition_count": total,
        "checks": checks,
        "scalar_half_lives_steps": list(config.scalar_half_lives_steps),
        "primary_half_life_steps": config.primary_half_life_steps,
        "wall_time_seconds": time.perf_counter() - started,
    }
    source_contract = {
        "source_run_ids": {"e9_1": config.e9_1_run_id, "e9_9": config.e9_9_run_id},
        "source_paths": {name: _relative(repo_root, path) for name, path in inputs["paths"].items()},
        "source_sha256": {
            name: {
                artifact: file_sha256(path / artifact)
                for artifact in (
                    (E9_1_REQUIRED if name == "e9_1" else E9_9_REQUIRED)
                    + ("config.json", "manifest.json")
                )
            }
            for name, path in inputs["paths"].items()
        },
        "score_convention": "loss gradients g=-s; outer products are sign invariant",
        "parameter_chart": "canonical flattened PyTorch parameter order, p=512",
        "old_fisher_timing": "state before observing transition batch",
        "penalty_fisher_timing": "state used by realized EWC fit after the pipeline Fisher update",
        "fresh_curvature_timing": "batch t evaluated at retained pre-update parameter theta_hat_t",
        "curvature_representation": "rank-8 plus nonnegative residual diagonal",
        "source_artifacts_mutated": False,
    }
    return audit_rows, curvature_artifact, summary, source_contract


def _load_e9_12(config: GainEstimatorConfig, repo_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = repo_root / config.e9_12_run_path
    validate_completed(path, required=E9_12_REQUIRED)
    if path.name != config.e9_12_run_id:
        raise ValueError("E9.13 source identity differs")
    summary = read_json(path / "summary.json")
    if summary.get("status") != "supported":
        raise ValueError("E9.12 gate did not pass")
    vectors = torch.load(path / "curvatures.pt", map_location="cpu", weights_only=True)
    return path, summary, vectors


def run_gain_estimator(
    config: GainEstimatorConfig,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Apply causal fast-Fisher gains to every retained fresh displacement."""

    started = time.perf_counter()
    source_path, source_summary, source = _load_e9_12(config, repo_root)
    rows: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}
    total = sum(
        group["actions"].numel()
        for by_schedule in source.values()
        for group in by_schedule.values()
    ) * len(config.fast_half_lives_steps)
    progress = tqdm(total=total, desc="E9.13 structured gains", unit="solve")
    maximum_primary_residual = 0.0
    all_finite = True
    all_ranks_valid = True
    for design, by_schedule in source.items():
        predictions[design] = {}
        for schedule, group in by_schedule.items():
            actions = group["actions"].to(torch.float64)
            fresh_displacements = group["fresh_displacements"].to(torch.float64)
            score_gradients = group["score_gradients"].to(torch.float64)
            structured: dict[str, Tensor] = {}
            fast_update_diagnostics: dict[str, list[dict[str, Any] | None]] = {}
            for half_life in config.fast_half_lives_steps:
                key = str(half_life)
                update_gain = ema_gain(half_life)
                fast_fisher = representation_from_artifact(group["old_fishers"][0]).to(dtype=torch.float64)
                if not isinstance(fast_fisher, LowRankDiagonalFisher):
                    raise ValueError("E9.13 requires low-rank-plus-diagonal Fisher states")
                values = []
                update_rows = []
                for step in range(actions.numel()):
                    old_fisher = representation_from_artifact(group["old_fishers"][step]).to(dtype=torch.float64)
                    if not isinstance(old_fisher, LowRankDiagonalFisher):
                        raise ValueError("E9.12 old Fisher representation is incompatible")
                    action, diagnostics = structured_gain_action(
                        old_fisher,
                        fast_fisher,
                        fresh_displacements[step],
                        pi=float(actions[step]),
                        relative_damping=config.relative_damping,
                    )
                    values.append(action)
                    finite = bool(torch.isfinite(action).all()) and math.isfinite(
                        float(diagnostics["normalized_residual"])
                    )
                    all_finite = all_finite and finite
                    if half_life == config.primary_half_life_steps:
                        maximum_primary_residual = max(
                            maximum_primary_residual,
                            float(diagnostics["normalized_residual"]),
                        )
                    fresh = empirical_fisher(score_gradients[step])
                    update = blend_archive_fisher(
                        fast_fisher,
                        fresh,
                        blend_gain=update_gain,
                        rank=config.fisher_rank,
                        lanczos_seed=derive_component_seed(
                            config.replica_seed,
                            f"e9_13_fast_fisher:{design}:{schedule}:{half_life}:{step}",
                        ),
                    )
                    fast_fisher = update.representation
                    all_ranks_valid = all_ranks_valid and fast_fisher.rank <= config.fisher_rank
                    update_rows.append(update.lanczos.mapping())
                    rows.append(
                        {
                            "design": design,
                            "schedule": schedule,
                            "step": step,
                            "half_life_steps": half_life,
                            "primary_half_life": half_life == config.primary_half_life_steps,
                            "applied_pi": float(actions[step]),
                            "normalized_solver_residual": diagnostics["normalized_residual"],
                            "solver_damping": diagnostics["damping"],
                            "combined_rank": diagnostics["combined_rank"],
                            "combined_storage_bytes": diagnostics["combined_storage_bytes"],
                            "fast_fisher_rank_after": fast_fisher.rank,
                            "fast_fisher_storage_bytes_after": fast_fisher.storage_bytes(),
                            "fast_fisher_update_gain": update_gain,
                            "effective_observation_support": stationary_effective_observations(
                                half_life, score_gradients.shape[1]
                            ),
                            "prediction_uses_current_batch": False,
                            "current_batch_updates_next_state": True,
                        }
                    )
                    progress.update(1)
                structured[key] = torch.stack(values)
                fast_update_diagnostics[key] = update_rows
            predictions[design][schedule] = {
                "actions": actions,
                "updates": group["updates"].to(torch.float64),
                "fresh_displacements": fresh_displacements,
                "scalar_oracle_gains": group["scalar_oracle_gains"].to(torch.float64),
                "scalar_predictions": {
                    key: value.to(torch.float64) for key, value in group["scalar_predictions"].items()
                },
                "structured_predictions": structured,
                "fast_fisher_update_diagnostics": fast_update_diagnostics,
            }
    progress.close()
    checks = {
        "all_predictions_finite": all_finite,
        "all_fast_fisher_ranks_valid": all_ranks_valid,
        "maximum_primary_normalized_solver_residual": maximum_primary_residual,
        "primary_solver_residual_passed": (
            maximum_primary_residual <= config.solver_residual_tolerance
        ),
        "all_predictions_strictly_lagged": all(
            not row["prediction_uses_current_batch"] for row in rows
        ),
    }
    status = "supported" if all(
        checks[name]
        for name in (
            "all_predictions_finite",
            "all_fast_fisher_ranks_valid",
            "primary_solver_residual_passed",
            "all_predictions_strictly_lagged",
        )
    ) else "rejected"
    summary = {
        "study": "E9.13",
        "status": status,
        "solve_count": len(rows),
        "checks": checks,
        "fast_half_lives_steps": list(config.fast_half_lives_steps),
        "primary_half_life_steps": config.primary_half_life_steps,
        "effective_observation_support": {
            str(value): stationary_effective_observations(value, 4)
            for value in config.fast_half_lives_steps
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    source_contract = {
        "source_run_id": config.e9_12_run_id,
        "source_path": _relative(repo_root, source_path),
        "source_sha256": {
            name: file_sha256(source_path / name)
            for name in E9_12_REQUIRED + ("config.json", "manifest.json")
        },
        "source_status": source_summary["status"],
        "solver": "Woodbury diagonal-plus-low-rank directional solve",
        "matrix_inverse_materialized": False,
        "current_batch_prediction_leakage": False,
        "source_artifacts_mutated": False,
    }
    return rows, predictions, summary, source_contract


def _metric(
    update: Tensor,
    prediction: Tensor,
    fisher: Any,
    baseline_energy: float,
) -> dict[str, float | None]:
    residual = update - prediction
    residual_energy = float(fisher.quadratic(residual))
    update_energy = float(fisher.quadratic(update))
    prediction_energy = float(fisher.quadratic(prediction))
    cross = _risk_inner(update, prediction, fisher)
    cosine = None
    if update_energy > 0.0 and prediction_energy > 0.0:
        cosine = cross / math.sqrt(update_energy * prediction_energy)
    return {
        "residual_energy": residual_energy,
        "residual_energy_ratio": residual_energy / max(baseline_energy, 1e-15),
        "fisher_cosine": cosine,
        "norm_calibration": math.sqrt(prediction_energy / max(update_energy, 1e-15)),
    }


def run_gain_calibration(
    config: GainCalibrationConfig,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Score structured-gain predictions and apply the frozen pivot gate."""

    started = time.perf_counter()
    source_path = repo_root / config.e9_13_run_path
    validate_completed(source_path, required=E9_13_REQUIRED)
    if source_path.name != config.e9_13_run_id:
        raise ValueError("E9.14 source identity differs")
    estimator_config = GainEstimatorConfig.from_mapping(read_json(source_path / "config.json"))
    estimator_summary = read_json(source_path / "summary.json")
    if estimator_summary.get("status") != "supported":
        raise ValueError("E9.13 numerical gate did not pass")
    predictions = torch.load(source_path / "predictions.pt", map_location="cpu", weights_only=True)
    e9_12_path = repo_root / estimator_config.e9_12_run_path
    contract_config = GainContractConfig.from_mapping(read_json(e9_12_path / "config.json"))
    e9_1_path = repo_root / contract_config.e9_1_run_path
    retrospective_config = RetrospectiveConfig.from_mapping(read_json(e9_1_path / "config.json"))
    retrospective = _load_inputs(retrospective_config, repo_root)
    e9_12_rows = read_json(e9_12_path / "audit_rows.json")
    row_lookup = {
        (row["design"], row["schedule"], int(row["step"])): row for row in e9_12_rows
    }
    rows: list[dict[str, Any]] = []
    half_lives = estimator_config.fast_half_lives_steps
    for design, by_schedule in predictions.items():
        for schedule, group in by_schedule.items():
            updates = group["updates"].to(torch.float64)
            fresh = group["fresh_displacements"].to(torch.float64)
            actions = group["actions"].to(torch.float64)
            for step in range(actions.numel()):
                meta = row_lookup[(design, schedule, step)]
                fisher = _fisher(
                    retrospective,
                    float(meta["angle_degrees"]),
                    retrospective_config.oracle_rank,
                )
                baseline = float(actions[step]) * fresh[step]
                baseline_energy = float(fisher.quadratic(updates[step] - baseline))
                oracle = float(group["scalar_oracle_gains"][step]) * fresh[step]
                for half_life in half_lives:
                    key = str(half_life)
                    comparators = {
                        "scalar_affine": baseline,
                        "causal_scalar_ema": float(group["scalar_predictions"][key][step]) * fresh[step],
                        "predictable_structured_gain": group["structured_predictions"][key][step],
                        "contemporaneous_scalar_oracle": oracle,
                    }
                    for method, prediction in comparators.items():
                        rows.append(
                            {
                                "design": design,
                                "schedule": schedule,
                                "step": step,
                                "angle_degrees": meta["angle_degrees"],
                                "leg_id": meta["leg_id"],
                                "cold_start": meta["cold_start"],
                                "reversal_window": meta["reversal_window"],
                                "half_life_steps": half_life,
                                "primary_half_life": half_life == estimator_config.primary_half_life_steps,
                                "method": method,
                                **_metric(updates[step], prediction, fisher, baseline_energy),
                            }
                        )

    groups = []
    for design in ("single_lap", "double_lap"):
        for schedule in retrospective_config.schedule_kinds:
            for half_life in half_lives:
                selected = [
                    row for row in rows
                    if row["design"] == design
                    and row["schedule"] == schedule
                    and row["half_life_steps"] == half_life
                ]
                by_method = {}
                for method in (
                    "scalar_affine",
                    "causal_scalar_ema",
                    "predictable_structured_gain",
                    "contemporaneous_scalar_oracle",
                ):
                    values = [row for row in selected if row["method"] == method]
                    baseline_total = sum(
                        row["residual_energy"]
                        for row in selected
                        if row["method"] == "scalar_affine"
                    )
                    reversal = [row for row in values if row["reversal_window"]]
                    reversal_baseline = sum(
                        row["residual_energy"]
                        for row in selected
                        if row["method"] == "scalar_affine" and row["reversal_window"]
                    )
                    by_method[method] = {
                        "aggregate_energy_ratio": sum(row["residual_energy"] for row in values)
                        / max(baseline_total, 1e-15),
                        "median_step_energy_ratio": statistics.median(
                            row["residual_energy_ratio"] for row in values
                        ),
                        "mean_fisher_cosine": statistics.fmean(
                            row["fisher_cosine"] for row in values if row["fisher_cosine"] is not None
                        ),
                        "mean_norm_calibration": statistics.fmean(
                            row["norm_calibration"] for row in values
                        ),
                        "reversal_aggregate_energy_ratio": (
                            None
                            if not reversal
                            else sum(row["residual_energy"] for row in reversal)
                            / max(reversal_baseline, 1e-15)
                        ),
                    }
                groups.append(
                    {
                        "design": design,
                        "schedule": schedule,
                        "half_life_steps": half_life,
                        "primary_half_life": half_life == estimator_config.primary_half_life_steps,
                        "methods": by_method,
                    }
                )
    primary = [row for row in groups if row["primary_half_life"]]
    reduction_limit = 1.0 - config.mean_energy_reduction_threshold
    reduction_count = sum(
        row["methods"]["predictable_structured_gain"]["aggregate_energy_ratio"] <= reduction_limit
        for row in primary
    )
    scalar_win_count = sum(
        row["methods"]["predictable_structured_gain"]["aggregate_energy_ratio"]
        < row["methods"]["causal_scalar_ema"]["aggregate_energy_ratio"]
        for row in primary
    )
    sensitivity_counts = {
        str(half_life): sum(
            row["methods"]["predictable_structured_gain"]["aggregate_energy_ratio"] < 1.0
            for row in groups
            if row["half_life_steps"] == half_life
        )
        for half_life in half_lives
    }
    reversal_values = [
        row["methods"]["predictable_structured_gain"]["reversal_aggregate_energy_ratio"]
        for row in primary
        if row["methods"]["predictable_structured_gain"]["reversal_aggregate_energy_ratio"] is not None
    ]
    checks = {
        "structured_reduction_group_count": reduction_count,
        "structured_reduction_gate_passed": reduction_count >= config.minimum_passing_groups,
        "scalar_ema_win_group_count": scalar_win_count,
        "scalar_ema_gate_passed": scalar_win_count >= config.minimum_passing_groups,
        "neighboring_half_life_improvement_counts": sensitivity_counts,
        "sensitivity_gate_passed": min(sensitivity_counts.values()) >= config.minimum_passing_groups,
        "maximum_reversal_energy_ratio": max(reversal_values, default=0.0),
        "reversal_gate_passed": all(value <= config.reversal_ratio_limit for value in reversal_values),
        "numerical_gate_passed": estimator_summary["status"] == "supported",
    }
    if checks["structured_reduction_gate_passed"] and all(
        checks[name]
        for name in (
            "scalar_ema_gate_passed",
            "sensitivity_gate_passed",
            "reversal_gate_passed",
            "numerical_gate_passed",
        )
    ):
        status = "supported"
        decision = "authorize_reviewed_gain-aware observer amendment"
    elif checks["structured_reduction_gate_passed"]:
        status = "inconclusive"
        decision = "do not promote; ordinary-stretch evidence lacks stable calibration"
    else:
        status = "rejected"
        decision = "close structured-gain family and release Plan 10 hypothesis"
    summary = {
        "study": "E9.14",
        "status": status,
        "decision": decision,
        "checks": checks,
        "group_summaries": groups,
        "predictive_accuracy_or_nll_evaluated": False,
        "wall_time_seconds": time.perf_counter() - started,
    }
    source_contract = {
        "source_run_id": config.e9_13_run_id,
        "source_path": _relative(repo_root, source_path),
        "source_sha256": {
            name: file_sha256(source_path / name)
            for name in E9_13_REQUIRED + ("config.json", "manifest.json")
        },
        "evaluation_fisher": "Plan 6 high-sample rank-16 population reference",
        "predictive_metrics_sealed": True,
        "source_artifacts_mutated": False,
    }
    return rows, summary, source_contract
