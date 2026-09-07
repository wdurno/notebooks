"""Artifact-only opportunity and cross-moment studies for Plan 9."""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ..phase6_oracle import angle_key
from .artifacts import file_sha256, read_json, validate_completed
from .config import ExtensionConfig, RetrospectiveConfig
from .estimators import (
    exponential_window_weights,
    marginal_action,
    quadratic_energy,
    weighted_cross_moment,
)
from .retrospective import _fisher, _load_inputs, _risk_inner


_RETROSPECTIVE_REQUIRED = (
    "source_contract.json",
    "audit_rows.json",
    "vectors.pt",
    "summary.json",
)


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_source(path: Path, expected_run_id: str) -> dict[str, Any]:
    validate_completed(path, required=_RETROSPECTIVE_REQUIRED)
    if path.name != expected_run_id:
        raise ValueError("Plan 9 extension source run ID differs")
    config_value = read_json(path / "config.json")
    config = RetrospectiveConfig.from_mapping(config_value)
    return {
        "path": path,
        "config": config,
        "rows": read_json(path / "audit_rows.json"),
        "vectors": torch.load(path / "vectors.pt", map_location="cpu", weights_only=True),
        "summary": read_json(path / "summary.json"),
    }


def _load_extension_inputs(config: ExtensionConfig, repo_root: Path) -> dict[str, Any]:
    e9_1 = _load_source(_resolve(repo_root, config.e9_1_run_path), config.e9_1_run_id)
    e9_2 = _load_source(_resolve(repo_root, config.e9_2_run_path), config.e9_2_run_id)
    if e9_1["config"].study != "e9_1" or e9_2["config"].study != "e9_2":
        raise ValueError("Plan 9 extension source studies differ")
    retrospective = _load_inputs(e9_1["config"], repo_root)
    return {"e9_1": e9_1, "e9_2": e9_2, "retrospective": retrospective}


def _source_contract(
    config: ExtensionConfig,
    inputs: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    sources = {name: inputs[name]["path"] for name in ("e9_1", "e9_2")}
    return {
        "source_run_ids": {
            "e9_1": config.e9_1_run_id,
            "e9_2": config.e9_2_run_id,
        },
        "source_paths": {
            name: str(path.relative_to(repo_root)) for name, path in sources.items()
        },
        "source_sha256": {
            name: {
                artifact: file_sha256(path / artifact)
                for artifact in (*_RETROSPECTIVE_REQUIRED, "config.json", "manifest.json")
            }
            for name, path in sources.items()
        },
        "source_artifacts_mutated": False,
    }


def _longest_threshold_run(rows: list[dict[str, Any]], threshold: float) -> int:
    longest = 0
    current = 0
    previous_step = None
    previous_leg = None
    for row in sorted(rows, key=lambda item: int(item["step"])):
        contiguous = (
            previous_step is not None
            and int(row["step"]) == previous_step + 1
            and int(row["leg_id"]) == previous_leg
        )
        if row["eligible"] and row["population_action_lift"] >= threshold:
            current = current + 1 if contiguous else 1
            longest = max(longest, current)
        else:
            current = 0
        previous_step = int(row["step"])
        previous_leg = int(row["leg_id"])
    return longest


def build_opportunity_audit(
    config: ExtensionConfig,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    inputs = _load_extension_inputs(config, repo_root)
    source = inputs["e9_1"]
    retrospective = inputs["retrospective"]
    output = []
    for row in source["rows"]:
        stored = source["vectors"][row["design"]][row["schedule"]][row["condition"]]
        step = int(row["step"])
        fisher = _fisher(retrospective, float(row["angle_degrees"]), source["config"].oracle_rank)
        population = stored["population_displacements"][step].to(dtype=torch.float64)
        learner = stored["parameters"][step].to(dtype=torch.float64)
        reference = retrospective["references"]["parameters"][
            angle_key(float(row["angle_degrees"]))
        ].to(dtype=torch.float64)
        anchor_error = learner - reference
        conditional_signal = quadratic_energy(population - anchor_error, fisher)
        batch_size = source["config"].deployed_batch_size
        population_signal = float(row["population_signal"])
        new_shape = float(row["new_covariance_shape"])
        covariance_action = marginal_action(
            0.0,
            float(row["q"]),
            float(row["old_covariance_shape"]),
            new_shape,
            batch_size,
        )
        conditional_action = marginal_action(
            conditional_signal,
            float(row["q"]),
            float(row["old_covariance_shape"]),
            new_shape,
            batch_size,
        )
        output.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "design",
                        "schedule",
                        "condition",
                        "step",
                        "leg_id",
                        "angle_degrees",
                        "next_angle_degrees",
                        "cumulative_angular_degrees",
                        "cold_start",
                        "eligible",
                        "q",
                        "old_covariance_shape",
                        "new_covariance_shape",
                    )
                },
                "population_signal": population_signal,
                "population_rho": batch_size * population_signal / new_shape,
                "population_action": float(row["population_pi"]),
                "covariance_action": covariance_action,
                "population_action_lift": float(row["population_pi"]) - covariance_action,
                "conditional_signal": conditional_signal,
                "conditional_rho": batch_size * conditional_signal / new_shape,
                "conditional_action": conditional_action,
                "conditional_action_lift": conditional_action - covariance_action,
                "anchor_error_energy": quadratic_energy(anchor_error, fisher),
                "marginal_centered_signal_identified": False,
            }
        )

    groups = []
    for design, schedule in sorted(
        {
            (row["design"], row["schedule"])
            for row in output
            if row["condition"] == "decomposed_edr"
        }
    ):
        selected = [
            row
            for row in output
            if row["design"] == design
            and row["schedule"] == schedule
            and row["condition"] == "decomposed_edr"
            and row["eligible"]
        ]
        threshold_runs = {
            f"{threshold:.2f}": _longest_threshold_run(selected, threshold)
            for threshold in config.opportunity_thresholds
        }
        groups.append(
            {
                "design": design,
                "schedule": schedule,
                "eligible_transition_count": len(selected),
                "mean_population_rho": statistics.fmean(
                    row["population_rho"] for row in selected
                ),
                "maximum_population_rho": max(row["population_rho"] for row in selected),
                "mean_population_action_lift": statistics.fmean(
                    row["population_action_lift"] for row in selected
                ),
                "maximum_population_action_lift": max(
                    row["population_action_lift"] for row in selected
                ),
                "mean_conditional_rho": statistics.fmean(
                    row["conditional_rho"] for row in selected
                ),
                "mean_conditional_action_lift": statistics.fmean(
                    row["conditional_action_lift"] for row in selected
                ),
                "longest_sustained_action_lift": threshold_runs,
            }
        )

    primary_key = f"{config.primary_opportunity_threshold:.2f}"
    sensitivity_key = f"{min(config.opportunity_thresholds):.2f}"
    if any(row["longest_sustained_action_lift"][primary_key] >= config.sustained_steps for row in groups):
        classification = "present"
    elif any(
        row["longest_sustained_action_lift"][sensitivity_key] >= config.sustained_steps
        for row in groups
    ):
        classification = "weak"
    else:
        classification = "absent"
    summary = {
        "study": "E9.7",
        "status": "supported",
        "config_hash": config.config_hash,
        "opportunity_classification": classification,
        "marginal_centered_signal_identified": False,
        "primary_threshold": config.primary_opportunity_threshold,
        "sustained_steps": config.sustained_steps,
        "group_summaries": groups,
        "diagnostics": {
            "row_count": len(output),
            "all_finite": all(
                not isinstance(value, float) or math.isfinite(value)
                for row in output
                for value in row.values()
            ),
        },
    }
    return output, summary, _source_contract(config, inputs, repo_root)


def _cross_window_rows(
    rows: list[dict[str, Any]],
    target: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["design"] == target["design"]
        and row["schedule"] == target["schedule"]
        and int(row["step"]) <= int(target["step"])
        and int(row["leg_id"]) == int(target["leg_id"])
    ]


def _lag_diagnostics(
    rows: list[dict[str, Any]],
    vectors: dict[str, Any],
    retrospective: dict[str, Any],
    rank: int,
    maximum_lag: int = 12,
) -> list[dict[str, Any]]:
    output = []
    for design, schedule in sorted({(row["design"], row["schedule"]) for row in rows}):
        selected = sorted(
            [
                row
                for row in rows
                if row["design"] == design
                and row["schedule"] == schedule
                and row["condition"] == "decomposed_edr"
            ],
            key=lambda item: int(item["step"]),
        )
        stored = vectors[design][schedule]["decomposed_edr"]
        residuals = {
            int(row["step"]): stored["z"][int(row["step"]) - 1].to(dtype=torch.float64)
            - stored["population_displacements"][int(row["step"])].to(dtype=torch.float64)
            for row in selected
        }
        by_step = {int(row["step"]): row for row in selected}
        for lag in range(1, maximum_lag + 1):
            products = []
            for right_step, right_row in by_step.items():
                left_step = right_step - lag
                left_row = by_step.get(left_step)
                if (
                    left_row is None
                    or int(left_row["leg_id"]) != int(right_row["leg_id"])
                    or not left_row["eligible"]
                    or not right_row["eligible"]
                ):
                    continue
                fisher = _fisher(
                    retrospective, float(right_row["angle_degrees"]), rank
                )
                products.append(
                    _risk_inner(residuals[left_step], residuals[right_step], fisher)
                )
            if products:
                output.append(
                    {
                        "design": design,
                        "schedule": schedule,
                        "lag": lag,
                        "pair_count": len(products),
                        "mean_residual_cross_moment": statistics.fmean(products),
                        "median_residual_cross_moment": statistics.median(products),
                        "standard_deviation": (
                            statistics.stdev(products) if len(products) > 1 else None
                        ),
                    }
                )
    return output


def build_cross_moment_audit(
    config: ExtensionConfig,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    inputs = _load_extension_inputs(config, repo_root)
    source = inputs["e9_1"]
    retrospective = inputs["retrospective"]
    primary_rows = [
        row for row in source["rows"] if row["condition"] == "decomposed_edr"
    ]
    e9_2_lookup = {
        (row["design"], row["schedule"], int(row["step"])): row
        for row in inputs["e9_2"]["rows"]
        if row["estimator"] == "vector_x2"
    }
    output = []
    for target in primary_rows:
        window_rows = _cross_window_rows(primary_rows, target)
        stored = source["vectors"][target["design"]][target["schedule"]][
            "decomposed_edr"
        ]
        steps = [int(row["step"]) for row in window_rows]
        vectors = torch.stack(
            [stored["z"][step - 1].to(dtype=torch.float64) for step in steps]
        )
        population = torch.stack(
            [
                stored["population_displacements"][step].to(dtype=torch.float64)
                for step in steps
            ]
        )
        coordinates = torch.tensor(
            [float(row["cumulative_angular_degrees"]) for row in window_rows],
            dtype=torch.float64,
        )
        fisher = _fisher(
            retrospective,
            float(target["angle_degrees"]),
            source["config"].oracle_rank,
        )
        for half_life in config.cross_half_lives_degrees[target["design"]]:
            weights = exponential_window_weights(coordinates, half_life=half_life)
            for lag_exclusion in config.lag_exclusions:
                if len(steps) < lag_exclusion + 2:
                    continue
                estimate, pair_details = weighted_cross_moment(
                    vectors,
                    weights,
                    fisher,
                    lag_exclusion=lag_exclusion,
                )
                population_cross, _ = weighted_cross_moment(
                    population,
                    weights,
                    fisher,
                    lag_exclusion=lag_exclusion,
                )
                clipped = max(0.0, estimate)
                action = marginal_action(
                    clipped,
                    float(target["q"]),
                    float(target["old_covariance_shape"]),
                    float(target["new_covariance_shape"]),
                    source["config"].deployed_batch_size,
                )
                baseline = e9_2_lookup[
                    (target["design"], target["schedule"], int(target["step"]))
                ]
                output.append(
                    {
                        **{
                            key: target[key]
                            for key in (
                                "design",
                                "schedule",
                                "step",
                                "leg_id",
                                "angle_degrees",
                                "next_angle_degrees",
                                "cumulative_angular_degrees",
                                "reversal_window",
                                "cold_start",
                                "eligible",
                                "q",
                                "old_covariance_shape",
                                "new_covariance_shape",
                                "population_signal",
                                "population_pi",
                                "historical_trend_pi",
                            )
                        },
                        "half_life_degrees": half_life,
                        "lag_exclusion": lag_exclusion,
                        "window_observation_count": len(steps),
                        **pair_details,
                        "cross_signal_signed": estimate,
                        "cross_signal_clipped": clipped,
                        "cross_clipped": estimate < 0.0,
                        "cross_action": action,
                        "population_cross_signal": population_cross,
                        "local_linearity_bias": population_cross
                        - float(target["population_signal"]),
                        "e9_2_signal_signed": float(baseline["signal_signed"]),
                        "e9_2_signal_clipped": float(baseline["signal_clipped"]),
                        "e9_2_action": float(baseline["action"]),
                    }
                )

    lag_rows = _lag_diagnostics(
        primary_rows,
        source["vectors"],
        retrospective,
        source["config"].oracle_rank,
    )
    summaries = []
    for design, schedule, half_life, lag in sorted(
        {
            (
                row["design"],
                row["schedule"],
                row["half_life_degrees"],
                row["lag_exclusion"],
            )
            for row in output
        }
    ):
        selected = [
            row
            for row in output
            if row["design"] == design
            and row["schedule"] == schedule
            and row["half_life_degrees"] == half_life
            and row["lag_exclusion"] == lag
            and row["eligible"]
        ]
        if not selected:
            continue
        summaries.append(
            {
                "design": design,
                "schedule": schedule,
                "half_life_degrees": half_life,
                "lag_exclusion": lag,
                "eligible_transition_count": len(selected),
                "signed_signal_mae": statistics.fmean(
                    abs(row["cross_signal_signed"] - row["population_signal"])
                    for row in selected
                ),
                "clipped_signal_mae": statistics.fmean(
                    abs(row["cross_signal_clipped"] - row["population_signal"])
                    for row in selected
                ),
                "signed_signal_bias": statistics.fmean(
                    row["cross_signal_signed"] - row["population_signal"]
                    for row in selected
                ),
                "action_mae": statistics.fmean(
                    abs(row["cross_action"] - row["population_pi"])
                    for row in selected
                ),
                "e9_2_signed_signal_mae": statistics.fmean(
                    abs(row["e9_2_signal_signed"] - row["population_signal"])
                    for row in selected
                ),
                "e9_2_action_mae": statistics.fmean(
                    abs(row["e9_2_action"] - row["population_pi"])
                    for row in selected
                ),
                "historical_action_mae": statistics.fmean(
                    abs(row["historical_trend_pi"] - row["population_pi"])
                    for row in selected
                ),
                "clipping_fraction": statistics.fmean(
                    float(row["cross_clipped"]) for row in selected
                ),
                "mean_effective_pair_count": statistics.fmean(
                    row["effective_pair_count"] for row in selected
                ),
                "local_linearity_absolute_bias": statistics.fmean(
                    abs(row["local_linearity_bias"]) for row in selected
                ),
            }
        )

    primary = [
        row
        for row in summaries
        if row["half_life_degrees"]
        == config.primary_cross_half_lives_degrees[row["design"]]
        and row["lag_exclusion"] == config.primary_lag_exclusion
    ]
    improving_signal = sum(
        row["signed_signal_mae"] < row["e9_2_signed_signal_mae"] for row in primary
    )
    low_clipping = sum(row["clipping_fraction"] < 0.75 for row in primary)
    sensitivities = defaultdict(list)
    for row in summaries:
        sensitivities[(row["design"], row["schedule"])].append(row)
    stable_groups = 0
    for key, rows in sensitivities.items():
        primary_row = next(
            row
            for row in rows
            if row["half_life_degrees"]
            == config.primary_cross_half_lives_degrees[key[0]]
            and row["lag_exclusion"] == config.primary_lag_exclusion
        )
        neighbor_mae = [
            row["signed_signal_mae"]
            for row in rows
            if row["lag_exclusion"] in (1, 2)
        ]
        if statistics.median(neighbor_mae) <= 1.5 * primary_row["signed_signal_mae"]:
            stable_groups += 1
    supported = improving_signal >= 3 and low_clipping >= 3 and stable_groups >= 3
    status = "supported" if supported else "inconclusive"
    summary = {
        "study": "E9.8",
        "status": status,
        "config_hash": config.config_hash,
        "primary_estimator": {
            "pair_weight": "product_of_exponential_observation_weights",
            "lag_exclusion": config.primary_lag_exclusion,
            "half_lives_degrees": config.primary_cross_half_lives_degrees,
            "same_leg_only": True,
        },
        "decision_rule": {
            "groups_improving_signed_signal_mae": improving_signal,
            "groups_with_clipping_below_0.75": low_clipping,
            "groups_stable_under_sensitivity": stable_groups,
            "group_count": len(primary),
        },
        "group_summaries": summaries,
        "diagnostics": {
            "row_count": len(output),
            "lag_diagnostic_count": len(lag_rows),
            "all_finite": all(
                not isinstance(value, float) or math.isfinite(value)
                for row in output
                for value in row.values()
            ),
        },
    }
    return output, lag_rows, summary, _source_contract(config, inputs, repo_root)


def run_e9_7(config: ExtensionConfig, repo_root: Path):
    started = time.perf_counter()
    rows, summary, contract = build_opportunity_audit(config, repo_root)
    summary["elapsed_seconds"] = time.perf_counter() - started
    return rows, None, summary, contract


def run_e9_8(config: ExtensionConfig, repo_root: Path):
    started = time.perf_counter()
    rows, lag_rows, summary, contract = build_cross_moment_audit(config, repo_root)
    summary["elapsed_seconds"] = time.perf_counter() - started
    return rows, lag_rows, summary, contract
