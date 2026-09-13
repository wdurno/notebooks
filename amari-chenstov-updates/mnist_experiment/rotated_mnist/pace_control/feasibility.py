"""Artifact-only Fisher-speed feasibility map for Plan 10 Phase 1."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import torch

from src.representations import FisherRepresentation, representation_from_artifact

from ..phase6_artifacts import load_completed_phase6_oracle
from ..phase6_oracle import angle_key, covariance_risk, paired_displacement_covariance_risk
from .artifacts import file_sha256
from .config import FeasibilityConfig
from .theory import centered_pace, fixed_pi_q_update, movement_energy_target


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _fisher(artifacts: dict[str, Any], angle: float, rank: int) -> FisherRepresentation:
    value = artifacts[angle_key(angle)][str(rank)]["representation"]
    return representation_from_artifact(value, device="cpu").to(dtype=torch.float64)


def _linear_grid(config: FeasibilityConfig) -> tuple[float, ...]:
    count = round(30.0 / config.map_increment_degrees)
    values = tuple(round(index * config.map_increment_degrees, 12) for index in range(count + 1))
    if not math.isclose(values[-1], 30.0, abs_tol=1e-12):
        raise ValueError("map increment does not partition [0, 30]")
    return values


def _interpolate(angles: tuple[float, ...], values: list[float], angle: float) -> float:
    if angle <= angles[0]:
        return values[0]
    if angle >= angles[-1]:
        return values[-1]
    right = next(index for index, candidate in enumerate(angles) if candidate >= angle)
    if angles[right] == angle:
        return values[right]
    left = right - 1
    weight = (angle - angles[left]) / (angles[right] - angles[left])
    return (1.0 - weight) * values[left] + weight * values[right]


def _corrected_energy(
    left: float,
    right: float,
    *,
    parameters: dict[str, torch.Tensor],
    local: dict[str, dict[str, torch.Tensor]],
    fisher: FisherRepresentation,
    local_sample_size: int,
    local_replicates: int,
    reference_sample_size: int,
) -> tuple[float, float, float]:
    displacement = (parameters[angle_key(right)] - parameters[angle_key(left)]).to(torch.float64)
    raw = float(fisher.quadratic(displacement))
    noise_shape, _ = paired_displacement_covariance_risk(
        local[angle_key(left)][str(local_sample_size)][:local_replicates],
        local[angle_key(right)][str(local_sample_size)][:local_replicates],
        fisher,
        local_sample_size=local_sample_size,
    )
    correction = noise_shape / reference_sample_size
    return raw, correction, max(0.0, raw - correction)


def build_feasibility_map(
    config: FeasibilityConfig, repo_root: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    source_path = _resolve(repo_root, config.oracle_run_path)
    oracle = load_completed_phase6_oracle(source_path)
    if source_path.name != config.oracle_run_id:
        raise ValueError("Plan 10 oracle identity differs from configuration")
    if oracle.config.reference.fit_sample_size < 1:
        raise ValueError("Plan 6 reference sample size is invalid")
    references = torch.load(source_path / "reference_states.pt", map_location="cpu", weights_only=True)
    fishers = torch.load(source_path / "reference_fishers.pt", map_location="cpu", weights_only=True)
    local = torch.load(source_path / "local_mle_parameters.pt", map_location="cpu", weights_only=True)
    parameters = references["parameters"]
    angles = _linear_grid(config)
    missing = [
        angle
        for angle in angles
        if angle_key(angle) not in parameters
        or angle_key(angle) not in fishers
        or angle_key(angle) not in local
    ]
    if missing:
        raise ValueError(f"Plan 6 oracle lacks required map angles: {missing}")

    covariance_shapes: list[float] = []
    fisher_speeds: list[float] = []
    map_rows: list[dict[str, Any]] = []
    for index, angle in enumerate(angles):
        fisher = _fisher(fishers, angle, config.fisher_rank)
        shape, _ = covariance_risk(
            local[angle_key(angle)][str(config.local_mle_sample_size)][: config.local_mle_replicates],
            fisher,
            local_sample_size=config.local_mle_sample_size,
        )
        covariance_shapes.append(shape)
        left_index = max(0, index - 1)
        right_index = min(len(angles) - 1, index + 1)
        if left_index == right_index:
            raise ValueError("Fisher-speed map requires at least two angles")
        left = angles[left_index]
        right = angles[right_index]
        raw, correction, energy = _corrected_energy(
            left,
            right,
            parameters=parameters,
            local=local,
            fisher=fisher,
            local_sample_size=config.local_mle_sample_size,
            local_replicates=config.local_mle_replicates,
            reference_sample_size=oracle.config.reference.fit_sample_size,
        )
        span = right - left
        speed = energy / (span * span)
        fisher_speeds.append(speed)
        map_rows.append(
            {
                "angle_degrees": angle,
                "derivative_left_degrees": left,
                "derivative_right_degrees": right,
                "derivative_span_degrees": span,
                "derivative_energy_raw": raw,
                "derivative_noise_correction": correction,
                "fisher_speed": speed,
                "covariance_shape": shape,
            }
        )

    validation_rows: list[dict[str, Any]] = []
    for index, angle in enumerate(angles):
        fisher = _fisher(fishers, angle, config.fisher_rank)
        for direction in (-1, 1):
            prior_energy = None
            for increment in config.validation_increments_degrees:
                right = round(angle + direction * increment, 12)
                if right < 0.0 or right > 30.0 or angle_key(right) not in parameters:
                    continue
                raw, correction, energy = _corrected_energy(
                    angle,
                    right,
                    parameters=parameters,
                    local=local,
                    fisher=fisher,
                    local_sample_size=config.local_mle_sample_size,
                    local_replicates=config.local_mle_replicates,
                    reference_sample_size=oracle.config.reference.fit_sample_size,
                )
                prediction = fisher_speeds[index] * increment * increment
                relative_error = abs(energy - prediction) / max(energy, prediction, 1e-12)
                monotone = prior_energy is None or energy + 1e-12 >= prior_energy
                validation_rows.append(
                    {
                        "angle_degrees": angle,
                        "direction": direction,
                        "increment_degrees": increment,
                        "target_angle_degrees": right,
                        "energy_raw": raw,
                        "noise_correction": correction,
                        "energy": energy,
                        "quadratic_prediction": prediction,
                        "relative_error": relative_error,
                        "monotone_from_previous_increment": monotone,
                    }
                )
                prior_energy = energy

    route_rows: list[dict[str, Any]] = []
    q = config.initial_q
    angle = config.route_knots_degrees[0]
    step = 0
    lower_hits = 0
    upper_hits = 0
    infeasible = 0
    for leg, target_angle in enumerate(config.route_knots_degrees[1:]):
        direction = 1.0 if target_angle > angle else -1.0
        while not math.isclose(angle, target_angle, abs_tol=1e-12):
            if step >= config.maximum_route_steps:
                break
            old_shape = _interpolate(angles, covariance_shapes, angle)
            speed = _interpolate(angles, fisher_speeds, angle)
            candidate = config.minimum_pace_degrees
            target = float("nan")
            feasible = True
            for _ in range(16):
                prospective_angle = min(30.0, max(0.0, angle + direction * candidate))
                new_shape = _interpolate(angles, covariance_shapes, prospective_angle)
                target = movement_energy_target(
                    config.fixed_pi,
                    q=q,
                    old_covariance_shape=old_shape,
                    new_covariance_shape=new_shape,
                    batch_size=config.batch_size,
                )
                if target < 0.0 or speed <= 0.0:
                    feasible = False
                    candidate = config.minimum_pace_degrees
                    break
                updated = centered_pace(target, speed)
                if abs(updated - candidate) < 1e-10:
                    candidate = updated
                    break
                candidate = updated
            unclipped = candidate
            applied = min(config.maximum_pace_degrees, max(config.minimum_pace_degrees, candidate))
            bound = None
            if candidate <= config.minimum_pace_degrees:
                bound = "lower"
                lower_hits += 1
            elif candidate >= config.maximum_pace_degrees:
                bound = "upper"
                upper_hits += 1
            remaining = abs(target_angle - angle)
            applied = min(applied, remaining)
            next_angle = round(angle + direction * applied, 12)
            if not feasible:
                infeasible += 1
            route_rows.append(
                {
                    "step": step,
                    "leg": leg,
                    "angle_degrees": angle,
                    "target_knot_degrees": target_angle,
                    "direction": int(direction),
                    "q": q,
                    "old_covariance_shape": old_shape,
                    "fisher_speed": speed,
                    "movement_target": target,
                    "pace_unclipped_degrees": unclipped,
                    "pace_degrees": applied,
                    "pace_bound": bound,
                    "feasible": feasible,
                    "next_angle_degrees": next_angle,
                    "cumulative_observations": (step + 1) * config.batch_size,
                }
            )
            q = fixed_pi_q_update(q, config.fixed_pi, config.batch_size)
            angle = next_angle
            step += 1
        if step >= config.maximum_route_steps:
            break

    relative_errors = [row["relative_error"] for row in validation_rows]
    monotone_rows = [row for row in validation_rows if row["increment_degrees"] > min(config.validation_increments_degrees)]
    monotone_fraction = statistics.fmean(float(row["monotone_from_previous_increment"]) for row in monotone_rows)
    median_error = statistics.median(relative_errors)
    bound_fraction = (lower_hits + upper_hits) / max(len(route_rows), 1)
    route_complete = math.isclose(angle, config.route_knots_degrees[-1], abs_tol=1e-9) and len(route_rows) < config.maximum_route_steps
    checks = {
        "all_targets_feasible": infeasible == 0,
        "finite_step_energy_monotone": monotone_fraction >= config.minimum_monotone_fraction,
        "quadratic_approximation_useful": median_error <= config.maximum_median_relative_error,
        "route_complete": route_complete,
        "bound_occupancy_acceptable": bound_fraction <= config.maximum_bound_fraction,
    }
    summary = {
        "gate_pass": all(checks.values()),
        "gate_checks": checks,
        "map_angle_count": len(angles),
        "validation_pair_count": len(validation_rows),
        "median_relative_approximation_error": median_error,
        "monotone_fraction": monotone_fraction,
        "route_steps": len(route_rows),
        "route_observations": len(route_rows) * config.batch_size,
        "route_complete": route_complete,
        "infeasible_steps": infeasible,
        "lower_bound_steps": lower_hits,
        "upper_bound_steps": upper_hits,
        "bound_fraction": bound_fraction,
        "final_q": q,
        "fisher_speed_minimum": min(fisher_speeds),
        "fisher_speed_median": statistics.median(fisher_speeds),
        "fisher_speed_maximum": max(fisher_speeds),
        "covariance_shape_minimum": min(covariance_shapes),
        "covariance_shape_median": statistics.median(covariance_shapes),
        "covariance_shape_maximum": max(covariance_shapes),
    }
    source_contract = {
        "oracle_run_id": source_path.name,
        "oracle_path": config.oracle_run_path,
        "oracle_config_hash": oracle.manifest["config_hash"],
        "reference_fit_sample_size": oracle.config.reference.fit_sample_size,
        "source_partition_hash": oracle.source_contract["source_partition_hash"],
        "source_file_sha256": {
            name: file_sha256(source_path / name)
            for name in ("manifest.json", "reference_states.pt", "reference_fishers.pt", "local_mle_parameters.pt")
        },
        "reference_interpolation_used": False,
        "scalar_map_interpolation_used": True,
    }
    return map_rows, validation_rows, route_rows, summary, source_contract
