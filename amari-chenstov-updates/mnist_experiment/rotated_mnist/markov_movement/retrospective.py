"""Artifact-only Plan 9 anchor-cancellation and smooth-drift studies."""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.representations import FisherRepresentation, representation_from_artifact

from ..phase6_artifacts import load_completed_phase6_oracle
from ..phase6_oracle import angle_key, covariance_risk
from ..phase7_movement_audit import transition_key
from ..phase8_artifacts import load_completed_phase8_run
from .artifacts import file_sha256
from .config import RetrospectiveConfig
from .estimators import (
    anchor_cancelled_observations,
    exponential_window_weights,
    innovation_noise_risk,
    marginal_action,
    quadratic_energy,
    weighted_vector,
)


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _selected_oracle(config: RetrospectiveConfig, oracle, schedule: str, step: int):
    try:
        return oracle.oracle_estimates[schedule][str(step)][
            str(config.oracle_sample_size)
        ][str(config.oracle_rank)][str(config.oracle_replicate_count)]
    except KeyError as exc:
        raise ValueError("selected Plan 6 oracle estimate is absent") from exc


def _source_transition_lookup(config: RetrospectiveConfig, single, oracle):
    output: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
    for schedule in config.schedule_kinds:
        rows = single.trajectory_metrics[schedule][config.condition]
        for step, (left, right) in enumerate(zip(rows, rows[1:])):
            key = transition_key(
                schedule,
                float(left["angle_degrees"]),
                float(right["angle_degrees"]),
            )
            if key in output:
                raise ValueError("population transition identity is not unique")
            output[key] = (step, _selected_oracle(config, oracle, schedule, step))
    return output


def _load_inputs(config: RetrospectiveConfig, repo_root: Path) -> dict[str, Any]:
    paths = {
        "oracle": _resolve(repo_root, config.oracle_run_path),
        "single_lap": _resolve(repo_root, config.single_lap_run_path),
        "double_lap": _resolve(repo_root, config.double_lap_run_path),
    }
    oracle = load_completed_phase6_oracle(paths["oracle"])
    runs = {
        "single_lap": load_completed_phase8_run(paths["single_lap"]),
        "double_lap": load_completed_phase8_run(paths["double_lap"]),
    }
    if oracle.config.run_id != config.oracle_run_id or any(
        run.config.run_id != getattr(config, f"{design}_run_id")
        for design, run in runs.items()
    ):
        raise ValueError("Plan 9 source run identity differs from configuration")
    partition_hashes = {
        oracle.source_contract["source_partition_hash"],
        *(run.source_contract["partition_hash"] for run in runs.values()),
    }
    parameter_counts = {
        int(oracle.reference_contract["parameter_count"]),
        *(int(run.source_contract["parameter_count"]) for run in runs.values()),
    }
    initial_hashes = {run.source_contract["initial_state_hash"] for run in runs.values()}
    if len(partition_hashes) != 1 or parameter_counts != {512} or len(initial_hashes) != 1:
        raise ValueError("Plan 9 source ancestry or parameter layout differs")
    if runs["single_lap"].source_contract["schedule_hashes"] != oracle.source_contract[
        "schedule_hashes"
    ]:
        raise ValueError("single-lap and population schedules differ")
    for run in runs.values():
        if run.run_summary["samples_per_step"] != config.deployed_batch_size:
            raise ValueError("Plan 9 deployed batch size differs from source")

    trajectories = {
        design: torch.load(
            paths[design] / "trajectories.pt", map_location="cpu", weights_only=True
        )
        for design in runs
    }
    references = torch.load(
        paths["oracle"] / "reference_states.pt", map_location="cpu", weights_only=True
    )
    fishers = torch.load(
        paths["oracle"] / "reference_fishers.pt", map_location="cpu", weights_only=True
    )
    local_parameters = torch.load(
        paths["oracle"] / "local_mle_parameters.pt",
        map_location="cpu",
        weights_only=True,
    )
    lookup = _source_transition_lookup(config, runs["single_lap"], oracle)
    return {
        "paths": paths,
        "oracle": oracle,
        "runs": runs,
        "trajectories": trajectories,
        "references": references,
        "fishers": fishers,
        "local_parameters": local_parameters,
        "population_lookup": lookup,
        "partition_hash": partition_hashes.pop(),
        "initial_state_hash": initial_hashes.pop(),
    }


def _fisher(inputs: dict[str, Any], angle: float, rank: int) -> FisherRepresentation:
    artifact = inputs["fishers"][angle_key(angle)][str(rank)]["representation"]
    return representation_from_artifact(artifact, device="cpu").to(dtype=torch.float64)


def _local_values(
    inputs: dict[str, Any], angle: float, sample_size: int
) -> Tensor:
    return inputs["local_parameters"][angle_key(angle)][str(sample_size)].to(
        dtype=torch.float64
    )


def _population_vector(inputs: dict[str, Any], left: float, right: float) -> Tensor:
    parameters = inputs["references"]["parameters"]
    return (
        parameters[angle_key(right)] - parameters[angle_key(left)]
    ).to(dtype=torch.float64)


def _risk_inner(left: Tensor, right: Tensor, fisher: FisherRepresentation) -> float:
    value = left.to(dtype=fisher.dtype, device=fisher.device)
    other = right.to(dtype=fisher.dtype, device=fisher.device)
    return float(value @ fisher.matvec(other))


def _directions(rows: tuple[dict[str, Any], ...]) -> tuple[list[int | None], list[int | None]]:
    reversal_steps: list[int | None] = []
    since: list[int | None] = []
    previous = None
    reversal = None
    for row in rows[:-1]:
        direction = int(row["direction_to_next"])
        if previous is not None and direction and direction != previous:
            reversal = int(row["step"])
        if direction:
            previous = direction
        reversal_steps.append(reversal)
        since.append(None if reversal is None else int(row["step"]) - reversal)
    return reversal_steps, since


def build_anchor_audit(
    config: RetrospectiveConfig,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    inputs = _load_inputs(config, repo_root)
    output: list[dict[str, Any]] = []
    vectors: dict[str, Any] = {}
    maximum_population_error = 0.0
    maximum_displacement_error = 0.0
    transition_uses: defaultdict[tuple[str, str, str], int] = defaultdict(int)

    for design, run in inputs["runs"].items():
        vectors[design] = {}
        available = set(inputs["trajectories"][design][config.schedule_kinds[0]])
        conditions = [
            name
            for name in (config.condition, "tracked_q_covariance", "fixed_pi005_sentinel")
            if name in available
        ]
        for schedule in config.schedule_kinds:
            vectors[design][schedule] = {}
            for condition in conditions:
                rows = run.trajectory_metrics[schedule][condition]
                stored = inputs["trajectories"][design][schedule][condition]
                parameters = stored["parameters"].to(dtype=torch.float64)
                displacements = stored["displacements"].to(dtype=torch.float64)
                maximum_displacement_error = max(
                    maximum_displacement_error,
                    float((displacements - (parameters[1:] - parameters[:-1])).abs().max()),
                )
                controllers = [row["controller"] for row in rows[:-1]]
                if any(item is None for item in controllers):
                    raise ValueError("Plan 8 source lacks controller decisions")
                actions = torch.tensor(
                    [float(item["applied_pi"]) for item in controllers],
                    dtype=torch.float64,
                )
                y_values, z_values = anchor_cancelled_observations(displacements, actions)
                population = []
                reversal_steps, steps_since = _directions(rows)
                vector_rows = []
                for step, (left_row, right_row) in enumerate(
                    zip(rows, rows[1:])
                ):
                    left = float(left_row["angle_degrees"])
                    right = float(right_row["angle_degrees"])
                    key = transition_key(schedule, left, right)
                    try:
                        source_step, details = inputs["population_lookup"][key]
                    except KeyError as exc:
                        raise ValueError(f"unmatched Plan 9 transition: {key}") from exc
                    transition_uses[key] += 1
                    movement = _population_vector(inputs, left, right)
                    population.append(movement)
                    if step == 0:
                        continue
                    fisher = _fisher(inputs, left, config.oracle_rank)
                    old_shape, _ = covariance_risk(
                        _local_values(inputs, left, config.oracle_sample_size),
                        fisher,
                        local_sample_size=config.oracle_sample_size,
                    )
                    new_shape, _ = covariance_risk(
                        _local_values(inputs, right, config.oracle_sample_size),
                        fisher,
                        local_sample_size=config.oracle_sample_size,
                    )
                    maximum_population_error = max(
                        maximum_population_error,
                        abs(old_shape - float(details["old_covariance_shape_risk"])),
                        abs(new_shape - float(details["new_covariance_shape_risk"])),
                    )
                    z = z_values[step - 1]
                    target_signal = float(details["signal"])
                    raw_energy = quadratic_energy(z, fisher)
                    correction = (old_shape + new_shape) / config.deployed_batch_size
                    signed_energy = raw_energy - correction
                    clipped_energy = max(0.0, signed_energy)
                    q = 1.0 / float(controllers[step]["effective_size"])
                    target_action = marginal_action(
                        target_signal,
                        q,
                        old_shape,
                        new_shape,
                        config.deployed_batch_size,
                    )
                    z_action = marginal_action(
                        clipped_energy,
                        q,
                        old_shape,
                        new_shape,
                        config.deployed_batch_size,
                    )
                    target_raw = quadratic_energy(movement, fisher)
                    cross = _risk_inner(z, movement, fisher)
                    cosine = (
                        None
                        if raw_energy <= 0.0 or target_raw <= 0.0
                        else cross / math.sqrt(raw_energy * target_raw)
                    )
                    historical_action = float(controllers[step]["plugin_pi"])
                    cold = bool(controllers[step]["cold_start_active"])
                    reversal_window = (
                        steps_since[step] is not None
                        and steps_since[step] < config.reversal_window_steps
                    )
                    row = {
                        "design": design,
                        "schedule": schedule,
                        "condition": condition,
                        "step": step,
                        "oracle_source_step": source_step,
                        "angle_degrees": left,
                        "next_angle_degrees": right,
                        "leg_id": int(left_row["leg_id"]),
                        "cumulative_angular_degrees": float(
                            left_row["cumulative_angular_degrees"]
                        ),
                        "reversal_step": reversal_steps[step],
                        "steps_since_reversal": steps_since[step],
                        "reversal_window": reversal_window,
                        "cold_start": cold,
                        "eligible": not cold and bool(details["precision_pass"]),
                        "applied_pi": float(controllers[step]["applied_pi"]),
                        "historical_trend_pi": historical_action,
                        "population_pi": target_action,
                        "z_pi": z_action,
                        "q": q,
                        "population_signal": target_signal,
                        "population_signal_raw": target_raw,
                        "z_signal_raw": raw_energy,
                        "z_noise_correction": correction,
                        "z_signal_signed": signed_energy,
                        "z_signal_clipped": clipped_energy,
                        "z_clipped": signed_energy < 0.0,
                        "old_covariance_shape": old_shape,
                        "new_covariance_shape": new_shape,
                        "fisher_squared_error": quadratic_energy(z - movement, fisher),
                        "fisher_cosine": cosine,
                        "oracle_precision_pass": bool(details["precision_pass"]),
                    }
                    output.append(row)
                    vector_rows.append(step)
                vectors[design][schedule][condition] = {
                    "parameters": parameters,
                    "actions": actions,
                    "y": y_values,
                    "z": z_values,
                    "population_displacements": torch.stack(population),
                    "audit_steps": torch.tensor(vector_rows, dtype=torch.int64),
                }

    diagnostics = {
        "all_finite": all(
            not isinstance(value, float) or math.isfinite(value)
            for row in output
            for value in row.values()
        ),
        "maximum_displacement_identity_error": maximum_displacement_error,
        "maximum_population_coefficient_error": maximum_population_error,
        "unique_population_transition_count": len(transition_uses),
        "reused_population_transition_count": sum(
            count - 1 for count in transition_uses.values()
        ),
        "row_count": len(output),
    }
    return output, vectors, {"inputs": inputs, "diagnostics": diagnostics}


def _group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted(
        {
            (row["design"], row["schedule"])
            for row in rows
            if row["condition"] == "decomposed_edr"
        }
    )
    output = []
    for design, schedule in groups:
        selected = [
            row
            for row in rows
            if row["design"] == design
            and row["schedule"] == schedule
            and row["condition"] == "decomposed_edr"
            and row["eligible"]
        ]
        if not selected:
            raise ValueError("Plan 9 primary group has no eligible transitions")
        historical_mae = statistics.fmean(
            abs(row["historical_trend_pi"] - row["population_pi"]) for row in selected
        )
        z_mae = statistics.fmean(abs(row["z_pi"] - row["population_pi"]) for row in selected)
        output.append(
            {
                "design": design,
                "schedule": schedule,
                "eligible_transition_count": len(selected),
                "historical_trend_action_mae": historical_mae,
                "z_action_mae": z_mae,
                "relative_action_mae_reduction": (
                    (historical_mae - z_mae) / historical_mae
                    if historical_mae > 0.0
                    else 0.0
                ),
                "z_signal_mae": statistics.fmean(
                    abs(row["z_signal_clipped"] - row["population_signal"])
                    for row in selected
                ),
                "z_signal_signed_bias": statistics.fmean(
                    row["z_signal_signed"] - row["population_signal"] for row in selected
                ),
                "z_clipping_fraction": statistics.fmean(
                    float(row["z_clipped"]) for row in selected
                ),
                "mean_fisher_cosine": statistics.fmean(
                    row["fisher_cosine"]
                    for row in selected
                    if row["fisher_cosine"] is not None
                ),
                "mean_fisher_squared_error": statistics.fmean(
                    row["fisher_squared_error"] for row in selected
                ),
            }
        )
    return output


def summarize_anchor_audit(
    config: RetrospectiveConfig,
    rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    groups = _group_summary(rows)
    improved = [row["z_action_mae"] < row["historical_trend_action_mae"] for row in groups]
    reductions = [row["relative_action_mae_reduction"] for row in groups]
    clipping = statistics.median(row["z_clipping_fraction"] for row in groups)
    if all(improved) and statistics.median(reductions) >= 0.25 and clipping < 0.5:
        status = "supported"
    elif sum(improved) >= 3 or any(row["mean_fisher_cosine"] > 0.0 for row in groups):
        status = "inconclusive"
    else:
        status = "rejected"
    return {
        "study": "E9.1",
        "config_hash": config.config_hash,
        "status": status,
        "decision_rule": {
            "groups_improved": sum(improved),
            "group_count": len(groups),
            "median_relative_action_mae_reduction": statistics.median(reductions),
            "median_clipping_fraction": clipping,
        },
        "group_summaries": groups,
        "diagnostics": diagnostics,
    }


def _source_contract(config: RetrospectiveConfig, inputs: dict[str, Any]) -> dict[str, Any]:
    files = {
        "oracle": (
            "config.json",
            "manifest.json",
            "reference_states.pt",
            "reference_fishers.pt",
            "local_mle_parameters.pt",
            "oracle_estimates.json",
        ),
        "single_lap": ("config.json", "manifest.json", "trajectories.pt", "trajectory_metrics.json"),
        "double_lap": ("config.json", "manifest.json", "trajectories.pt", "trajectory_metrics.json"),
    }
    return {
        "run_ids": {
            "oracle": config.oracle_run_id,
            "single_lap": config.single_lap_run_id,
            "double_lap": config.double_lap_run_id,
        },
        "paths": {
            name: str(path.relative_to(Path(__file__).parents[3]))
            for name, path in inputs["paths"].items()
        },
        "file_sha256": {
            name: {file: file_sha256(inputs["paths"][name] / file) for file in names}
            for name, names in files.items()
        },
        "partition_hash": inputs["partition_hash"],
        "initial_state_hash": inputs["initial_state_hash"],
        "parameter_count": 512,
        "source_artifacts_mutated": False,
    }


def run_e9_1(config: RetrospectiveConfig, repo_root: Path):
    started = time.perf_counter()
    rows, vectors, context = build_anchor_audit(config, repo_root)
    summary = summarize_anchor_audit(config, rows, context["diagnostics"])
    summary["elapsed_seconds"] = time.perf_counter() - started
    contract = _source_contract(config, context["inputs"])
    return rows, vectors, summary, contract


def _innovation_risks(
    inputs: dict[str, Any],
    angles: list[float],
    fisher: FisherRepresentation,
    config: RetrospectiveConfig,
) -> Tensor:
    values = []
    for angle in angles:
        shape, _ = covariance_risk(
            _local_values(inputs, angle, config.oracle_sample_size),
            fisher,
            local_sample_size=config.oracle_sample_size,
        )
        values.append(shape / config.deployed_batch_size)
    return torch.tensor(values, dtype=torch.float64)


def build_smoothing_audit(
    config: RetrospectiveConfig,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    anchor_rows, vectors, context = build_anchor_audit(config, repo_root)
    inputs = context["inputs"]
    anchor_lookup = {
        (row["design"], row["schedule"], row["condition"], row["step"]): row
        for row in anchor_rows
    }
    output: list[dict[str, Any]] = []
    scalar_moments: dict[tuple[str, str, str], float] = {}
    for design in ("single_lap", "double_lap"):
        run = inputs["runs"][design]
        for schedule in config.schedule_kinds:
            rows = run.trajectory_metrics[schedule][config.condition]
            stored = vectors[design][schedule][config.condition]
            z_values = stored["z"]
            population = stored["population_displacements"]
            candidates: tuple[tuple[str, float | None], ...] = (
                ("instantaneous", None),
                *tuple(
                    (f"vector_x{multiplier:g}", half_life)
                    for multiplier, half_life in zip(
                        (0.5, 1.0, 2.0),
                        config.smoothing_half_lives_degrees[design],
                        strict=True,
                    )
                ),
            )
            z_coordinates = torch.tensor(
                [float(row["cumulative_angular_degrees"]) for row in rows[1:-1]],
                dtype=torch.float64,
            )
            for label, half_life in candidates:
                scalar_key = (design, schedule, label)
                scalar_moments[scalar_key] = 0.0
                for z_index in range(z_values.shape[0]):
                    step = z_index + 1
                    anchor = anchor_lookup[(design, schedule, config.condition, step)]
                    coordinates = z_coordinates[: z_index + 1]
                    weights = exponential_window_weights(coordinates, half_life=half_life)
                    filtered = weighted_vector(z_values[: z_index + 1], weights)
                    left = float(rows[step]["angle_degrees"])
                    fisher = _fisher(inputs, left, config.oracle_rank)
                    raw_energy = quadratic_energy(filtered, fisher)
                    innovation_angles = [
                        float(row["angle_degrees"]) for row in rows[1 : step + 2]
                    ]
                    risks = _innovation_risks(inputs, innovation_angles, fisher, config)
                    correction = innovation_noise_risk(weights, risks)
                    signed = raw_energy - correction
                    clipped = max(0.0, signed)
                    filtered_population = weighted_vector(
                        population[1 : step + 1], weights
                    )
                    local_linearity_energy = quadratic_energy(filtered_population, fisher)
                    target_signal = anchor["population_signal"]
                    q = anchor["q"]
                    action = marginal_action(
                        clipped,
                        q,
                        anchor["old_covariance_shape"],
                        anchor["new_covariance_shape"],
                        config.deployed_batch_size,
                    )
                    gain = 1.0 if half_life is None else 1.0 - 2.0 ** (
                        -max(
                            0.0,
                            float(rows[step]["cumulative_angular_degrees"])
                            - float(rows[step - 1]["cumulative_angular_degrees"]),
                        )
                        / half_life
                    )
                    scalar_moments[scalar_key] = (
                        (1.0 - gain) * scalar_moments[scalar_key]
                        + gain * anchor["z_signal_clipped"]
                    )
                    scalar_action = marginal_action(
                        scalar_moments[scalar_key],
                        q,
                        anchor["old_covariance_shape"],
                        anchor["new_covariance_shape"],
                        config.deployed_batch_size,
                    )
                    output.append(
                        {
                            **{
                                name: anchor[name]
                                for name in (
                                    "design",
                                    "schedule",
                                    "step",
                                    "angle_degrees",
                                    "next_angle_degrees",
                                    "leg_id",
                                    "reversal_step",
                                    "steps_since_reversal",
                                    "reversal_window",
                                    "eligible",
                                    "historical_trend_pi",
                                    "population_pi",
                                    "population_signal",
                                    "q",
                                )
                            },
                            "estimator": label,
                            "half_life_degrees": half_life,
                            "weight_count": weights.numel(),
                            "weight_concentration": float(weights.square().sum()),
                            "signal_raw": raw_energy,
                            "noise_correction": correction,
                            "signal_signed": signed,
                            "signal_clipped": clipped,
                            "clipped": signed < 0.0,
                            "action": action,
                            "scalar_energy_action": scalar_action,
                            "local_linearity_energy": local_linearity_energy,
                            "local_linearity_bias": local_linearity_energy - target_signal,
                            "fisher_squared_error": quadratic_energy(
                                filtered - population[step], fisher
                            ),
                        }
                    )
    return output, vectors, context


def _smoothing_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["design"], row["schedule"], row["estimator"]) for row in rows})
    output = []
    for design, schedule, estimator in keys:
        selected = [
            row
            for row in rows
            if row["design"] == design
            and row["schedule"] == schedule
            and row["estimator"] == estimator
            and row["eligible"]
        ]
        reversal = [row for row in selected if row["reversal_window"]]
        output.append(
            {
                "design": design,
                "schedule": schedule,
                "estimator": estimator,
                "half_life_degrees": selected[0]["half_life_degrees"],
                "eligible_transition_count": len(selected),
                "action_mae": statistics.fmean(
                    abs(row["action"] - row["population_pi"]) for row in selected
                ),
                "scalar_energy_action_mae": statistics.fmean(
                    abs(row["scalar_energy_action"] - row["population_pi"])
                    for row in selected
                ),
                "historical_action_mae": statistics.fmean(
                    abs(row["historical_trend_pi"] - row["population_pi"])
                    for row in selected
                ),
                "signal_mae": statistics.fmean(
                    abs(row["signal_clipped"] - row["population_signal"])
                    for row in selected
                ),
                "signal_signed_bias": statistics.fmean(
                    row["signal_signed"] - row["population_signal"] for row in selected
                ),
                "local_linearity_absolute_bias": statistics.fmean(
                    abs(row["local_linearity_bias"]) for row in selected
                ),
                "clipping_fraction": statistics.fmean(
                    float(row["clipped"]) for row in selected
                ),
                "reversal_action_mae": (
                    None
                    if not reversal
                    else statistics.fmean(
                        abs(row["action"] - row["population_pi"]) for row in reversal
                    )
                ),
            }
        )
    return output


def summarize_smoothing_audit(
    config: RetrospectiveConfig,
    rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    summaries = _smoothing_summaries(rows)
    labels = sorted(
        {
            row["estimator"]
            for row in summaries
            if row["estimator"] != "instantaneous"
        }
    )
    candidate_scores = {}
    for label in labels:
        selected = [row for row in summaries if row["estimator"] == label]
        candidate_scores[label] = statistics.median(row["action_mae"] for row in selected)
    selected_label = min(candidate_scores, key=candidate_scores.get)
    selected = [row for row in summaries if row["estimator"] == selected_label]
    instant = {
        (row["design"], row["schedule"]): row
        for row in summaries
        if row["estimator"] == "instantaneous"
    }
    improvement_count = 0
    signal_improvement_count = 0
    catastrophic = []
    for row in selected:
        baseline = instant[(row["design"], row["schedule"])]
        if row["action_mae"] < min(
            baseline["action_mae"], row["historical_action_mae"]
        ):
            improvement_count += 1
        if row["signal_mae"] < baseline["signal_mae"]:
            signal_improvement_count += 1
        if (
            row["reversal_action_mae"] is not None
            and baseline["reversal_action_mae"] is not None
            and row["reversal_action_mae"]
            > 2.0 * baseline["reversal_action_mae"] + 0.02
        ):
            catastrophic.append((row["design"], row["schedule"]))
    sorted_candidates = sorted(candidate_scores.values())
    stable_neighbors = (
        len(sorted_candidates) >= 2
        and sorted_candidates[1] <= 1.25 * sorted_candidates[0] + 0.005
    )
    supported = (
        improvement_count >= 3
        and signal_improvement_count >= 3
        and not catastrophic
        and stable_neighbors
    )
    status = "supported" if supported else "inconclusive"
    return {
        "study": "E9.2",
        "config_hash": config.config_hash,
        "status": status,
        "selected_estimator": selected_label,
        "selection_uses_predictive_outcomes": False,
        "decision_rule": {
            "groups_improving_action_over_both_baselines": improvement_count,
            "groups_improving_signal_over_instantaneous": signal_improvement_count,
            "catastrophic_reversal_groups": catastrophic,
            "stable_neighbor_sensitivity": stable_neighbors,
        },
        "candidate_median_action_mae": candidate_scores,
        "group_summaries": summaries,
        "diagnostics": diagnostics,
    }


def run_e9_2(config: RetrospectiveConfig, repo_root: Path):
    started = time.perf_counter()
    rows, vectors, context = build_smoothing_audit(config, repo_root)
    summary = summarize_smoothing_audit(config, rows, context["diagnostics"])
    summary["elapsed_seconds"] = time.perf_counter() - started
    contract = _source_contract(config, context["inputs"])
    return rows, vectors, summary, contract
