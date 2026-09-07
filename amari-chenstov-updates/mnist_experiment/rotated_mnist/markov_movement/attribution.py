"""Operational EWC-affinity attribution for Plan 9 E9.9."""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from src.ewc import build_optimizer, take_ewc_proposal
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    resolve_device,
    resolve_dtype,
)
from src.representations import DiagonalFisher

from ..run import _learner_optimizer_config
from ..run_phase8 import _load_source as load_phase8_source
from .artifacts import file_sha256, read_json, validate_completed
from .config import AttributionConfig, RetrospectiveConfig
from .estimators import quadratic_energy
from .retrospective import _fisher, _load_inputs, _risk_inner


_SOURCE_REQUIRED = (
    "source_contract.json",
    "audit_rows.json",
    "vectors.pt",
    "summary.json",
)


def _selected_steps(count: int, requested: int) -> set[int]:
    if requested >= count:
        return set(range(count))
    return {
        round(index * (count - 1) / (requested - 1)) for index in range(requested)
    }


def _fit_fresh_proposal(
    current: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    source_config: Any,
    *,
    step_budget: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    model, layout = build_canonical_model(seed, device=device, dtype=dtype)
    layout.copy_vector_to_module(model, current.to(device=device, dtype=dtype))
    optimizer_config = dataclasses.replace(
        _learner_optimizer_config(source_config), inner_steps=step_budget
    )
    optimizer = build_optimizer(model, optimizer_config)
    identity = DiagonalFisher(
        torch.ones(layout.total_numel, device=device, dtype=dtype)
    )
    anchor = layout.flatten_module(model, detach=True)
    proposal = take_ewc_proposal(
        model,
        layout,
        inputs.to(device=device, dtype=dtype),
        targets.to(device=device),
        identity,
        optimizer_config,
        optimizer,
        adaptation_weight=1.0,
        penalty_anchor=anchor,
    )
    fitted = layout.flatten_module(model, detach=True).cpu().to(torch.float64)
    return fitted, proposal.metrics_mapping()


def run_affinity_attribution(
    config: AttributionConfig,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    source_path = repo_root / config.e9_1_run_path
    validate_completed(source_path, required=_SOURCE_REQUIRED)
    if source_path.name != config.e9_1_run_id:
        raise ValueError("E9.9 source identity differs")
    source_config = RetrospectiveConfig.from_mapping(read_json(source_path / "config.json"))
    source_vectors = torch.load(
        source_path / "vectors.pt", map_location="cpu", weights_only=True
    )
    retrospective = _load_inputs(source_config, repo_root)
    phase8_sources = {
        design: load_phase8_source(run.config, repo_root)
        for design, run in retrospective["runs"].items()
    }

    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=True,
    )
    device = resolve_device(config.runtime.device)
    dtype = resolve_dtype(config.runtime.dtype)
    total_primary = sum(
        source_vectors[design][schedule][config.condition]["actions"].numel()
        for design in ("single_lap", "double_lap")
        for schedule in source_config.schedule_kinds
    )
    total_sensitivity = (
        config.sensitivity_transitions_per_group
        * len(source_config.schedule_kinds)
        * 2
        * (len(config.optimizer_step_budgets) - 1)
    )
    progress = tqdm(
        total=total_primary + total_sensitivity,
        desc="E9.9 fresh-batch fits",
        unit="fit",
    )
    fit_rows = []
    sensitivity_rows = []
    vector_artifact: dict[str, Any] = {}
    primary_affinities: dict[tuple[str, str], torch.Tensor] = {}
    for design in ("single_lap", "double_lap"):
        vector_artifact[design] = {}
        phase8_source = phase8_sources[design]
        for schedule in source_config.schedule_kinds:
            stored = source_vectors[design][schedule][config.condition]
            parameters = stored["parameters"].to(torch.float64)
            actions = stored["actions"].to(torch.float64)
            stream = phase8_source.streams[schedule]
            sensitivity = _selected_steps(
                actions.numel(), config.sensitivity_transitions_per_group
            )
            by_budget: dict[int, torch.Tensor] = {}
            for budget in config.optimizer_step_budgets:
                selected = (
                    range(actions.numel())
                    if budget == config.primary_optimizer_step_budget
                    else sorted(sensitivity)
                )
                affinities = []
                selected_steps = []
                for step in selected:
                    fitted, metrics = _fit_fresh_proposal(
                        parameters[step],
                        stream["inputs"][step],
                        stream["targets"][step],
                        phase8_source.source_config,
                        step_budget=budget,
                        seed=config.replica_seed + step,
                        device=device,
                        dtype=dtype,
                    )
                    pi = float(actions[step])
                    affine = (1.0 - pi) * parameters[step] + pi * fitted
                    affinity = parameters[step + 1] - affine
                    affinities.append(affinity)
                    selected_steps.append(step)
                    fit_rows.append(
                        {
                            "design": design,
                            "schedule": schedule,
                            "step": step,
                            "optimizer_step_budget": budget,
                            "primary_budget": budget
                            == config.primary_optimizer_step_budget,
                            "applied_pi": pi,
                            "fresh_displacement_norm": float(
                                torch.linalg.vector_norm(fitted - parameters[step])
                            ),
                            "affinity_norm": float(torch.linalg.vector_norm(affinity)),
                            "optimizer_iterations": metrics["optimizer_iterations"],
                            "optimizer_function_evaluations": metrics[
                                "optimizer_function_evaluations"
                            ],
                            "stopping_reason": metrics["stopping_reason"],
                            "final_gradient_norm": metrics["final_gradient_norm"],
                        }
                    )
                    progress.update(1)
                by_budget[budget] = torch.stack(affinities)
                vector_artifact[design].setdefault(schedule, {})[str(budget)] = {
                    "steps": torch.tensor(selected_steps, dtype=torch.int64),
                    "affinities": by_budget[budget],
                }
            primary_affinities[(design, schedule)] = by_budget[
                config.primary_optimizer_step_budget
            ]
            for budget in config.optimizer_step_budgets:
                if budget == config.primary_optimizer_step_budget:
                    continue
                sensitivity_steps = sorted(sensitivity)
                for position, step in enumerate(sensitivity_steps):
                    primary = by_budget[config.primary_optimizer_step_budget][step]
                    alternative = by_budget[budget][position]
                    fisher = _fisher(
                        retrospective,
                        float(
                            retrospective["runs"][design]
                            .trajectory_metrics[schedule][config.condition][step][
                                "angle_degrees"
                            ]
                        ),
                        source_config.oracle_rank,
                    )
                    primary_energy = quadratic_energy(primary, fisher)
                    difference_energy = quadratic_energy(alternative - primary, fisher)
                    sensitivity_rows.append(
                        {
                            "design": design,
                            "schedule": schedule,
                            "step": step,
                            "primary_optimizer_step_budget": (
                                config.primary_optimizer_step_budget
                            ),
                            "alternative_optimizer_step_budget": budget,
                            "primary_affinity_energy": primary_energy,
                            "affinity_difference_energy": difference_energy,
                            "relative_affinity_difference": difference_energy
                            / max(primary_energy, 1e-12),
                        }
                    )
    progress.close()

    attribution_rows = []
    for design in ("single_lap", "double_lap"):
        run = retrospective["runs"][design]
        for schedule in source_config.schedule_kinds:
            stored = source_vectors[design][schedule][config.condition]
            actions = stored["actions"].to(torch.float64)
            affinities = primary_affinities[(design, schedule)]
            metrics = run.trajectory_metrics[schedule][config.condition]
            for step in range(1, actions.numel()):
                left_row = metrics[step]
                fisher = _fisher(
                    retrospective,
                    float(left_row["angle_degrees"]),
                    source_config.oracle_rank,
                )
                residual = (
                    stored["z"][step - 1].to(torch.float64)
                    - stored["population_displacements"][step].to(torch.float64)
                )
                contribution = (
                    affinities[step] / actions[step]
                    - affinities[step - 1] / actions[step - 1]
                )
                post = residual - contribution
                residual_energy = quadratic_energy(residual, fisher)
                post_energy = quadratic_energy(post, fisher)
                contribution_energy = quadratic_energy(contribution, fisher)
                cross = _risk_inner(residual, contribution, fisher)
                cosine = (
                    None
                    if residual_energy <= 0.0 or contribution_energy <= 0.0
                    else cross / math.sqrt(residual_energy * contribution_energy)
                )
                controller = left_row["controller"]
                attribution_rows.append(
                    {
                        "design": design,
                        "schedule": schedule,
                        "step": step,
                        "leg_id": int(left_row["leg_id"]),
                        "angle_degrees": float(left_row["angle_degrees"]),
                        "cumulative_angular_degrees": float(
                            left_row["cumulative_angular_degrees"]
                        ),
                        "cold_start": bool(controller["cold_start_active"]),
                        "residual_energy": residual_energy,
                        "affinity_contribution_energy": contribution_energy,
                        "post_affinity_energy": post_energy,
                        "energy_reduction_fraction": (
                            1.0 - post_energy / residual_energy
                            if residual_energy > 0.0
                            else 0.0
                        ),
                        "residual_affinity_cosine": cosine,
                    }
                )
    group_summaries = []
    for design, schedule in sorted(
        {(row["design"], row["schedule"]) for row in attribution_rows}
    ):
        selected = [
            row
            for row in attribution_rows
            if row["design"] == design
            and row["schedule"] == schedule
            and not row["cold_start"]
        ]
        group_summaries.append(
            {
                "design": design,
                "schedule": schedule,
                "transition_count": len(selected),
                "mean_residual_energy": statistics.fmean(
                    row["residual_energy"] for row in selected
                ),
                "mean_affinity_contribution_energy": statistics.fmean(
                    row["affinity_contribution_energy"] for row in selected
                ),
                "mean_post_affinity_energy": statistics.fmean(
                    row["post_affinity_energy"] for row in selected
                ),
                "median_energy_reduction_fraction": statistics.median(
                    row["energy_reduction_fraction"] for row in selected
                ),
                "mean_residual_affinity_cosine": statistics.fmean(
                    row["residual_affinity_cosine"]
                    for row in selected
                    if row["residual_affinity_cosine"] is not None
                ),
            }
        )
    affinity_dominant = sum(
        row["mean_post_affinity_energy"] <= 0.5 * row["mean_residual_energy"]
        for row in group_summaries
    )
    sensitivity_summaries = []
    for design, schedule in sorted(
        {(row["design"], row["schedule"]) for row in sensitivity_rows}
    ):
        selected = [
            row
            for row in sensitivity_rows
            if row["design"] == design and row["schedule"] == schedule
        ]
        sensitivity_summaries.append(
            {
                "design": design,
                "schedule": schedule,
                "transition_count": len(selected),
                "median_relative_affinity_difference": statistics.median(
                    row["relative_affinity_difference"] for row in selected
                ),
                "mean_relative_affinity_difference": statistics.fmean(
                    row["relative_affinity_difference"] for row in selected
                ),
            }
        )
    stable_groups = sum(
        row["median_relative_affinity_difference"] <= 0.25
        for row in sensitivity_summaries
    )
    if affinity_dominant >= 3 and stable_groups >= 3:
        attribution = "affinity_remainder_dominant"
    elif affinity_dominant >= 3:
        attribution = "affinity_remainder_dominant_but_procedure_sensitive"
    else:
        attribution = "affinity_remainder_not_dominant"
    summary = {
        "study": "E9.9",
        "status": "supported",
        "config_hash": config.config_hash,
        "attribution": attribution,
        "groups_with_half_residual_energy_after_affinity_removal": affinity_dominant,
        "groups_with_stable_optimizer_sensitivity": stable_groups,
        "group_count": len(group_summaries),
        "group_summaries": group_summaries,
        "optimizer_sensitivity_summaries": sensitivity_summaries,
        "fit_count": len(fit_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    contract = {
        "source_run_id": config.e9_1_run_id,
        "source_path": config.e9_1_run_path,
        "source_sha256": {
            name: file_sha256(source_path / name)
            for name in (*_SOURCE_REQUIRED, "config.json", "manifest.json")
        },
        "operational_fresh_estimator": {
            "initialization": "exact_pre_transition_parameter",
            "data": "exact_stored_fresh_batch",
            "objective": "unregularized_batch_mean_nll",
            "optimizer": "fresh_strong_wolfe_lbfgs",
            "step_budgets": list(config.optimizer_step_budgets),
        },
        "source_artifacts_mutated": False,
    }
    return (
        fit_rows,
        attribution_rows,
        sensitivity_rows,
        vector_artifact,
        summary,
        contract,
    )
