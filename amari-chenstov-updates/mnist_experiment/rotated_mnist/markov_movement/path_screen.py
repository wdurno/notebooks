"""Population-reference path opportunity screen for Plan 9 E9.10."""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from ..phase6_oracle import angle_key, estimate_oracle
from .artifacts import file_sha256, read_json, validate_completed
from .config import ExtensionConfig, PathScreenConfig, RetrospectiveConfig
from .estimators import marginal_action, quadratic_energy
from .retrospective import _fisher, _load_inputs, _local_values


_E9_7_REQUIRED = ("source_contract.json", "audit_rows.json", "summary.json")
_E9_1_REQUIRED = (
    "source_contract.json",
    "audit_rows.json",
    "vectors.pt",
    "summary.json",
)
_KNOTS = (0.0, 15.0, 30.0, 0.0, 15.0, 30.0)


def _schedule(transitions_per_arrow: int) -> list[dict[str, Any]]:
    rows = []
    step = 0
    direction_regime = 0
    previous_direction = None
    for arrow, (left, right) in enumerate(zip(_KNOTS, _KNOTS[1:])):
        direction = 1 if right > left else -1
        if previous_direction is not None and direction != previous_direction:
            direction_regime += 1
        previous_direction = direction
        for offset in range(transitions_per_arrow):
            fraction_left = offset / transitions_per_arrow
            fraction_right = (offset + 1) / transitions_per_arrow
            rows.append(
                {
                    "step": step,
                    "arrow": arrow,
                    "direction": direction,
                    "direction_regime": direction_regime,
                    "angle_degrees": left + (right - left) * fraction_left,
                    "next_angle_degrees": left + (right - left) * fraction_right,
                }
            )
            step += 1
    return rows


def _longest_run(rows: list[dict[str, Any]], threshold: float) -> int:
    longest = 0
    current = 0
    previous_step = None
    previous_regime = None
    for row in rows:
        contiguous = (
            previous_step is not None
            and row["step"] == previous_step + 1
            and row["direction_regime"] == previous_regime
        )
        if row["action_lift"] >= threshold:
            current = current + 1 if contiguous else 1
            longest = max(longest, current)
        else:
            current = 0
        previous_step = row["step"]
        previous_regime = row["direction_regime"]
    return longest


def _drift_variation(
    inputs: dict[str, Any],
    left: float,
    right: float,
    fisher,
) -> float:
    direction = 1.0 if right > left else -1.0
    count = int(round(abs(right - left) / 0.75))
    if count < 1:
        raise ValueError("path-screen transition is below the reference grid")
    subangles = [left + direction * 0.75 * index for index in range(count + 1)]
    parameters = inputs["references"]["parameters"]
    movements = torch.stack(
        [
            (
                parameters[angle_key(subangles[index + 1])]
                - parameters[angle_key(subangles[index])]
            ).to(torch.float64)
            for index in range(count)
        ]
    )
    mean = movements.mean(dim=0)
    denominator = quadratic_energy(mean, fisher)
    if denominator <= 0.0:
        return 0.0
    variation = statistics.fmean(
        quadratic_energy(row - mean, fisher) for row in movements
    )
    return variation / denominator


def run_path_screen(
    config: PathScreenConfig,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    e9_7_path = repo_root / config.e9_7_run_path
    validate_completed(e9_7_path, required=_E9_7_REQUIRED)
    e9_7_config = ExtensionConfig.from_mapping(read_json(e9_7_path / "config.json"))
    if e9_7_config.run_id != config.e9_7_run_id:
        raise ValueError("E9.10 source identity differs")
    e9_7_contract = read_json(e9_7_path / "source_contract.json")
    e9_1_path = repo_root / e9_7_contract["source_paths"]["e9_1"]
    validate_completed(e9_1_path, required=_E9_1_REQUIRED)
    retrospective_config = RetrospectiveConfig.from_mapping(
        read_json(e9_1_path / "config.json")
    )
    inputs = _load_inputs(retrospective_config, repo_root)
    oracle = inputs["oracle"]
    reference_sample_size = oracle.config.reference.fit_sample_size
    local_sample_size = retrospective_config.oracle_sample_size

    output = []
    candidate_summaries = []
    qualified = []
    for transitions_per_arrow in config.transitions_per_arrow:
        q = 1.0 / config.initial_effective_size
        candidate_rows = []
        for row in _schedule(transitions_per_arrow):
            left = float(row["angle_degrees"])
            right = float(row["next_angle_degrees"])
            fisher = _fisher(inputs, left, retrospective_config.oracle_rank)
            estimate = estimate_oracle(
                inputs["references"]["parameters"][angle_key(left)],
                inputs["references"]["parameters"][angle_key(right)],
                _local_values(inputs, left, local_sample_size),
                _local_values(inputs, right, local_sample_size),
                fisher,
                reference_sample_size=reference_sample_size,
                local_sample_size=local_sample_size,
                q=q,
                deployed_batch_size=config.deployed_batch_size,
            )
            covariance_action = marginal_action(
                0.0,
                q,
                estimate.old_covariance_shape_risk,
                estimate.new_covariance_shape_risk,
                config.deployed_batch_size,
            )
            variation = _drift_variation(inputs, left, right, fisher)
            result = {
                **row,
                "transitions_per_arrow": transitions_per_arrow,
                "angular_step_degrees": abs(right - left),
                "q": q,
                "signal": estimate.signal,
                "rho": config.deployed_batch_size
                * estimate.signal
                / estimate.new_covariance_shape_risk,
                "marginal_action": estimate.pi,
                "covariance_action": covariance_action,
                "action_lift": estimate.pi - covariance_action,
                "drift_variation_ratio": variation,
                "reference_precision_inherited": True,
            }
            output.append(result)
            candidate_rows.append(result)
            q = (1.0 - config.fixed_pi) ** 2 * q + (
                config.fixed_pi**2 / config.deployed_batch_size
            )
        longest = _longest_run(candidate_rows, config.action_lift_threshold)
        mean_variation = statistics.fmean(
            row["drift_variation_ratio"] for row in candidate_rows
        )
        qualifies = (
            longest >= config.sustained_steps
            and mean_variation <= config.maximum_mean_drift_variation_ratio
        )
        summary = {
            "transitions_per_arrow": transitions_per_arrow,
            "angular_step_degrees": 15.0 / transitions_per_arrow,
            "transition_count": len(candidate_rows),
            "mean_rho": statistics.fmean(row["rho"] for row in candidate_rows),
            "maximum_rho": max(row["rho"] for row in candidate_rows),
            "mean_action_lift": statistics.fmean(
                row["action_lift"] for row in candidate_rows
            ),
            "maximum_action_lift": max(row["action_lift"] for row in candidate_rows),
            "longest_threshold_run": longest,
            "mean_drift_variation_ratio": mean_variation,
            "maximum_drift_variation_ratio": max(
                row["drift_variation_ratio"] for row in candidate_rows
            ),
            "qualified": qualifies,
        }
        candidate_summaries.append(summary)
        if qualifies:
            qualified.append(summary)
    selected = (
        None
        if not qualified
        else max(qualified, key=lambda row: row["transitions_per_arrow"])
    )
    summary = {
        "study": "E9.10",
        "status": "supported" if selected is not None else "rejected",
        "config_hash": config.config_hash,
        "qualified_path_found": selected is not None,
        "selected_candidate": selected,
        "candidate_summaries": candidate_summaries,
        "predictive_outcomes_used_for_selection": False,
        "deployed_batch_size_changed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    contract = {
        "source_run_id": config.e9_7_run_id,
        "source_path": config.e9_7_run_path,
        "source_sha256": {
            name: file_sha256(e9_7_path / name)
            for name in (*_E9_7_REQUIRED, "config.json", "manifest.json")
        },
        "population_reference_run_id": retrospective_config.oracle_run_id,
        "population_reference_grid_degrees": 0.75,
        "source_artifacts_mutated": False,
    }
    if not all(
        not isinstance(value, float) or math.isfinite(value)
        for row in output
        for value in row.values()
    ):
        raise RuntimeError("E9.10 produced nonfinite values")
    return output, summary, contract
