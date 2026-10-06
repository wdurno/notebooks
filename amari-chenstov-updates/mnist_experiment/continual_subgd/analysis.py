"""Artifact-only Plan 13 selection and trajectory summaries."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from scipy.stats import t as student_t

from mnist_experiment.rotated_mnist.artifacts import _read_json

from .artifacts import UnitStore
from .conditions import Condition, default_probe_conditions, phase2_conditions
from .geometry import AdaptationGeometry, half_life_gain, projector_distance, random_geometry
from .mixture import MIXTURE_TRAJECTORY_REQUIRED, mixture_trajectory_unit
from .trajectory import BURN_IN_REQUIRED, TRAJECTORY_REQUIRED


ANALYSIS_REQUIRED = ("summary.json", "selection.json")


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty collection")
    return statistics.fmean(values)


def _standard_error(values: list[float]) -> float:
    return 0.0 if len(values) < 2 else statistics.stdev(values) / math.sqrt(len(values))


def _completed_rotation_result(
    store: UnitStore,
    phase: str,
    index: int,
    schedule: str,
    condition: Condition,
    *,
    burn_in_steps: int,
    rank: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    path = store.completed(
        _trajectory_unit(
            store,
            phase,
            index,
            schedule,
            condition,
            burn_in_steps=burn_in_steps,
            rank=rank,
        ),
        TRAJECTORY_REQUIRED,
    )
    if path is None:
        raise RuntimeError(f"missing trajectory {phase}/{index}/{schedule}/{condition.name}")
    return (
        _read_json(path / "summary.json"),
        _read_json(path / "metrics.json"),
        _read_json(path / "checks.json"),
    )


def _aggregate_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["schedule_kind"], row["condition"]["name"])].append(row)
    return [
        {
            "schedule": schedule,
            "condition": condition,
            "mean_current_nll_auc": _mean(
                [row["post_burn_in_current_nll_auc"] for row in values]
            ),
            "mean_current_accuracy_auc": _mean(
                [row["post_burn_in_current_accuracy_auc"] for row in values]
            ),
            "mean_worst_panel_nll_auc": _mean(
                [row["post_burn_in_worst_panel_nll_auc"] for row in values]
            ),
            "mean_wall_seconds": _mean([row["total_wall_time_seconds"] for row in values]),
            "replicas": len(values),
        }
        for (schedule, condition), values in sorted(grouped.items())
    ]


def _paired_effect(
    method: list[float],
    comparator: list[float],
) -> dict[str, Any]:
    if len(method) != len(comparator) or not method:
        raise ValueError("paired effect requires equal nonempty vectors")
    gains = [reference - treatment for treatment, reference in zip(method, comparator)]
    mean = _mean(gains)
    se = _standard_error(gains)
    sd = 0.0 if len(gains) < 2 else statistics.stdev(gains)
    critical = 0.0 if len(gains) < 2 else float(student_t.ppf(0.975, len(gains) - 1))
    return {
        "mean_gain": mean,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
        "standardized_effect": None if sd == 0 else mean / sd,
        "median_gain": statistics.median(gains),
        "favorable_replicas": sum(value > 0 for value in gains),
        "replicas": len(gains),
        "gains": gains,
    }


def _trajectory_unit(
    store: UnitStore,
    phase: str,
    index: int,
    schedule: str,
    condition: Condition,
    *,
    burn_in_steps: int,
    rank: int,
) -> dict[str, Any]:
    return store.unit(
        phase,
        "trajectory",
        index,
        schedule=schedule,
        condition=condition.name,
        detail={
            "burn_in_steps": burn_in_steps,
            "adaptation_rank": rank,
            "condition": condition.mapping(),
        },
    )


def _burn_in_unit(
    store: UnitStore,
    phase: str,
    index: int,
    schedule: str,
    burn_in_steps: int,
) -> dict[str, Any]:
    return store.unit(
        phase,
        "burn_in",
        index,
        schedule=schedule,
        detail={"burn_in_steps": burn_in_steps},
    )


def _full_measurements(
    store: UnitStore,
    phase: str,
    index: int,
    schedule: str,
    *,
    burn_in_steps: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    burn_path = store.completed(
        _burn_in_unit(store, phase, index, schedule, burn_in_steps),
        BURN_IN_REQUIRED,
    )
    condition = Condition("full_space", "full_space")
    trajectory_path = store.completed(
        _trajectory_unit(
            store,
            phase,
            index,
            schedule,
            condition,
            burn_in_steps=burn_in_steps,
            rank=rank,
        ),
        TRAJECTORY_REQUIRED,
    )
    if burn_path is None or trajectory_path is None:
        raise RuntimeError(f"missing full-space geometry artifacts for {index}/{schedule}")
    burn = torch.load(burn_path / "burn_in.pt", map_location="cpu", weights_only=False)
    trajectory = torch.load(
        trajectory_path / "trajectory.pt",
        map_location="cpu",
        weights_only=False,
    )
    displacements = torch.cat(
        (
            burn["shadow_displacements"].to(torch.float64),
            trajectory["shadow_displacements"].to(torch.float64),
        )
    )
    gradients = torch.cat(
        (
            burn["gradient_proxies"].to(torch.float64),
            trajectory["gradient_proxies"].to(torch.float64),
        )
    )
    rows = _read_json(trajectory_path / "metrics.json")
    return displacements, gradients, rows


def run_phase0_analysis(
    store: UnitStore,
    *,
    conditions: tuple[Condition, ...],
    burn_in_steps: int,
    rank: int,
    resume: bool,
) -> Path:
    phase = "phase0"
    detail = {
        "burn_in_steps": burn_in_steps,
        "adaptation_rank": rank,
        "conditions": [condition.mapping() for condition in conditions],
    }
    unit = store.unit(phase, "analysis", 1, detail=detail)
    session = store.begin(unit, ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    schedule_checks = []
    for schedule in ("linear", "sigmoid"):
        first_evaluations = []
        first_archive_traces = []
        objective_values = []
        no_update_hashes = []
        for condition in conditions:
            _, metrics, checks = _completed_rotation_result(
                store,
                phase,
                1,
                schedule,
                condition,
                burn_in_steps=burn_in_steps,
                rank=rank,
            )
            first = metrics[0]
            first_evaluations.append(
                (float(first["current_nll"]), float(first["current_accuracy"]))
            )
            first_archive_traces.append(float(first["archive_trace"]))
            if condition.kind != "no_update":
                objective = (
                    first["optimizer"]
                    if condition.kind == "full_space"
                    else first["shadow"]["objective"]
                )
                objective_values.append(float(objective["objective_before"]))
            if condition.kind == "no_update":
                no_update_hashes = [row["parameter_hash"] for row in metrics]
            if not checks["all_finite"] or not checks["burn_in_pairing"]:
                raise RuntimeError(f"Phase 0 trajectory contract failed for {schedule}/{condition.name}")
        tolerance = 1e-7
        schedule_checks.append(
            {
                "schedule": schedule,
                "identical_pre_treatment_evaluation": max(
                    abs(value - first_evaluations[0][column])
                    for value_pair in first_evaluations
                    for column, value in enumerate(value_pair)
                ) <= tolerance,
                "identical_pre_treatment_archive": max(first_archive_traces)
                - min(first_archive_traces)
                <= tolerance,
                "condition_independent_objective": max(objective_values)
                - min(objective_values)
                <= tolerance,
                "no_update_parameters_constant": len(set(no_update_hashes)) == 1,
            }
        )
    passed = all(
        all(value for key, value in row.items() if key != "schedule")
        for row in schedule_checks
    )
    if not passed:
        raise RuntimeError(f"Phase 0 control contract failed: {schedule_checks}")
    selection = {
        "integrity_gate_passed": True,
        "plan12_asset_builder_reused": True,
        "rotation_pi": 0.025,
        "isotropic_ridge_ratio": 0.1,
        "classification": "phase0_contract_reproduced",
    }
    session.write_json(
        "summary.json",
        {"phase": phase, "checks": schedule_checks, "selection": selection},
    )
    session.write_json("selection.json", selection)
    return store.finish(session, ANALYSIS_REQUIRED)


def _residual_ratio(geometry: AdaptationGeometry, observations: torch.Tensor) -> float:
    if observations.shape[0] == 0:
        raise ValueError("held-out geometry evaluation is empty")
    coefficients = observations @ geometry.basis
    residual = observations - coefficients @ geometry.basis.mT
    return float(residual.square().sum() / observations.square().sum().clamp_min(1e-30))


def run_phase1_analysis(store: UnitStore, *, resume: bool) -> Path:
    unit = store.unit("phase1", "analysis", 1)
    session = store.begin(unit, ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    minimum_k = min(store.study.burn_in_candidates)
    minimum_rank = min(store.study.rank_candidates)
    rows: list[dict[str, Any]] = []
    geometries: dict[tuple[str, int, int], list[AdaptationGeometry]] = defaultdict(list)
    replay_sequences: dict[tuple[str, int], torch.Tensor] = {}
    replay_metadata: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for index in range(1, store.study.phase1_replicas + 1):
        for schedule in ("linear", "sigmoid"):
            displacements, gradients, metrics = _full_measurements(
                store,
                "phase1",
                index,
                schedule,
                burn_in_steps=minimum_k,
                rank=minimum_rank,
            )
            replay_sequences[(schedule, index)] = displacements
            replay_metadata[(schedule, index)] = metrics
            for burn_in in store.study.burn_in_candidates:
                for rank in store.study.rank_candidates:
                    if rank > burn_in or burn_in >= displacements.shape[0]:
                        continue
                    learned = AdaptationGeometry.from_observations(
                        displacements[:burn_in],
                        rank,
                    )
                    proxy = AdaptationGeometry.from_observations(
                        gradients[:burn_in],
                        rank,
                    )
                    random = random_geometry(
                        displacements.shape[1],
                        learned.eigenvalues,
                        seed=store.study.seed(
                            f"phase1_random:{schedule}:K={burn_in}:r={rank}",
                            index,
                        ),
                    )
                    held_out = displacements[burn_in:]
                    row = {
                        "schedule": schedule,
                        "replica_index": index,
                        "burn_in_steps": burn_in,
                        "rank": rank,
                        "held_out_residual_ratio": _residual_ratio(learned, held_out),
                        "random_residual_ratio": _residual_ratio(random, held_out),
                        "gradient_proxy_residual_ratio": _residual_ratio(proxy, held_out),
                        "gradient_displacement_projector_distance": projector_distance(
                            learned.basis,
                            proxy.basis,
                        ),
                        "burn_in_explained_ratio": 1 - _residual_ratio(
                            learned,
                            displacements[:burn_in],
                        ),
                    }
                    rows.append(row)
                    geometries[(schedule, burn_in, rank)].append(learned)

    aggregate = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["burn_in_steps"], row["rank"])].append(row)
    for (burn_in, rank), values in sorted(grouped.items()):
        residuals = [float(row["held_out_residual_ratio"]) for row in values]
        random_residuals = [float(row["random_residual_ratio"]) for row in values]
        proxy_residuals = [float(row["gradient_proxy_residual_ratio"]) for row in values]
        stability = []
        for schedule in ("linear", "sigmoid"):
            candidates = geometries[(schedule, burn_in, rank)]
            for left in range(len(candidates)):
                for right in range(left + 1, len(candidates)):
                    stability.append(projector_distance(candidates[left].basis, candidates[right].basis))
        aggregate.append(
            {
                "burn_in_steps": burn_in,
                "rank": rank,
                "mean_held_out_residual_ratio": _mean(residuals),
                "se_held_out_residual_ratio": _standard_error(residuals),
                "mean_random_residual_ratio": _mean(random_residuals),
                "mean_gradient_proxy_residual_ratio": _mean(proxy_residuals),
                "mean_projector_instability": None if not stability else _mean(stability),
                "units": len(values),
            }
        )
    best = min(aggregate, key=lambda row: row["mean_held_out_residual_ratio"])
    threshold = best["mean_held_out_residual_ratio"] + best["se_held_out_residual_ratio"]
    eligible = [row for row in aggregate if row["mean_held_out_residual_ratio"] <= threshold]
    selected = min(
        eligible,
        key=lambda row: (
            row["rank"],
            row["burn_in_steps"],
            row["mean_held_out_residual_ratio"],
        ),
    )

    calibration_rows = []
    innovation_values: list[float] = []
    for half_life in store.study.covariance_half_lives:
        for (schedule, index), displacements in replay_sequences.items():
            burn_in = int(selected["burn_in_steps"])
            rank = int(selected["rank"])
            geometry = AdaptationGeometry.from_observations(displacements[:burn_in], rank)
            metadata = {int(row["step"]): row for row in replay_metadata[(schedule, index)]}
            values = []
            distances = []
            for step in range(burn_in, displacements.shape[0]):
                observation = displacements[step]
                innovation = geometry.innovation(observation)
                updated = geometry.update(observation, half_life_gain(half_life))
                distances.append(projector_distance(geometry.basis, updated.basis))
                geometry = updated
                values.append(
                    {
                        "step": step,
                        "innovation": innovation,
                        "knot": bool(metadata.get(step, {}).get("knot", False)),
                    }
                )
                innovation_values.append(innovation)
            calibration_rows.append(
                {
                    "schedule": schedule,
                    "replica_index": index,
                    "half_life": half_life,
                    "mean_innovation": _mean([row["innovation"] for row in values]),
                    "mean_knot_innovation": (
                        None
                        if not [row for row in values if row["knot"]]
                        else _mean([row["innovation"] for row in values if row["knot"]])
                    ),
                    "mean_projector_distance": _mean(distances),
                }
            )

    probe_rows = []
    for condition in default_probe_conditions():
        assert condition.controller is not None
        state_value = 0.0
        alphas = []
        betas = []
        for value in innovation_values:
            tau = condition.controller.tau
            state_value = (1 - tau) * state_value + tau * value
            alphas.append(1 / (1 + condition.controller.alpha_scale * state_value))
            fraction = condition.controller.beta_scale * state_value / (
                1 + condition.controller.beta_scale * state_value
            )
            betas.append(
                condition.controller.beta_min
                + (condition.controller.beta_max - condition.controller.beta_min) * fraction
            )
        probe_rows.append(
            {
                "condition": condition.mapping(),
                "alpha_min": min(alphas),
                "alpha_max": max(alphas),
                "beta_min_realized": min(betas),
                "beta_max_realized": max(betas),
                "nonsaturated": max(alphas) - min(alphas) >= 0.05,
            }
        )
    survivors = [row["condition"] for row in probe_rows if row["nonsaturated"]]
    if not survivors:
        survivors = [probe_rows[0]["condition"]]
    low_rank_supported = selected["mean_held_out_residual_ratio"] < selected["mean_random_residual_ratio"]
    selection = {
        "burn_in_steps": int(selected["burn_in_steps"]),
        "rank": int(selected["rank"]),
        "candidate_half_lives": list(store.study.covariance_half_lives),
        "low_rank_supported_against_random": low_rank_supported,
        "gradient_proxy_promoted": False,
        "probe_conditions": survivors,
        "selection_rule": "smallest rank then shortest burn-in within one SE of best displacement reconstruction",
    }
    summary = {
        "phase": "phase1",
        "rows": rows,
        "aggregate": aggregate,
        "selected": selected,
        "calibration": calibration_rows,
        "controller_operating_points": probe_rows,
        "classification": (
            "low_rank_structure_supported"
            if low_rank_supported
            else "low_rank_structure_not_better_than_random"
        ),
    }
    session.write_json("summary.json", summary)
    session.write_json("selection.json", selection)
    return store.finish(session, ANALYSIS_REQUIRED)


def run_phase2_analysis(
    store: UnitStore,
    *,
    burn_in_steps: int,
    rank: int,
    resume: bool,
) -> Path:
    unit = store.unit(
        "phase2",
        "analysis",
        1,
        detail={"burn_in_steps": burn_in_steps, "adaptation_rank": rank},
    )
    session = store.begin(unit, ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    conditions = phase2_conditions(store.study.covariance_half_lives)
    rows = []
    for index in range(1, store.study.phase2_replicas + 1):
        for schedule in ("linear", "sigmoid"):
            for condition in conditions:
                path = store.completed(
                    _trajectory_unit(
                        store,
                        "phase2",
                        index,
                        schedule,
                        condition,
                        burn_in_steps=burn_in_steps,
                        rank=rank,
                    ),
                    TRAJECTORY_REQUIRED,
                )
                if path is None:
                    raise RuntimeError(f"missing Phase 2 trajectory {index}/{schedule}/{condition.name}")
                rows.append(_read_json(path / "summary.json"))
    aggregate = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["schedule_kind"], row["condition"]["name"])].append(row)
    for (schedule, name), values in sorted(grouped.items()):
        aggregate.append(
            {
                "schedule": schedule,
                "condition": name,
                "mean_current_nll_auc": _mean([row["post_burn_in_current_nll_auc"] for row in values]),
                "mean_current_accuracy_auc": _mean([row["post_burn_in_current_accuracy_auc"] for row in values]),
                "mean_worst_panel_nll_auc": _mean([row["post_burn_in_worst_panel_nll_auc"] for row in values]),
                "mean_wall_seconds": _mean([row["total_wall_time_seconds"] for row in values]),
                "replicas": len(values),
            }
        )
    online = [row for row in aggregate if row["condition"].startswith("online_subgd_h")]
    static_by_schedule = {
        row["schedule"]: row for row in aggregate if row["condition"] == "static_subgd"
    }
    candidates = []
    for half_life in store.study.covariance_half_lives:
        name = f"online_subgd_h{half_life:g}"
        values = [row for row in online if row["condition"] == name]
        retention_harm = _mean(
            [
                (row["mean_worst_panel_nll_auc"] - static_by_schedule[row["schedule"]]["mean_worst_panel_nll_auc"])
                / static_by_schedule[row["schedule"]]["mean_worst_panel_nll_auc"]
                for row in values
            ]
        )
        candidates.append(
            {
                "name": name,
                "half_life": half_life,
                "mean_current_nll_auc": _mean([row["mean_current_nll_auc"] for row in values]),
                "mean_retention_harm": retention_harm,
                "mean_wall_seconds": _mean([row["mean_wall_seconds"] for row in values]),
                "eligible": retention_harm <= 0.01,
            }
        )
    eligible = [row for row in candidates if row["eligible"]] or candidates
    selected = min(
        eligible,
        key=lambda row: (
            row["mean_current_nll_auc"],
            row["mean_wall_seconds"],
            -row["half_life"],
        ),
    )
    selection = {
        "online_condition": selected["name"],
        "online_half_life": selected["half_life"],
        "fallback_used": not any(row["eligible"] for row in candidates),
        "candidate_rows": candidates,
    }
    summary = {
        "phase": "phase2",
        "aggregate": aggregate,
        "selection": selection,
    }
    session.write_json("summary.json", summary)
    session.write_json("selection.json", selection)
    return store.finish(session, ANALYSIS_REQUIRED)


def run_phase3_probe_analysis(
    store: UnitStore,
    *,
    phase: str,
    replica_count: int,
    conditions: tuple[Condition, ...],
    burn_in_steps: int,
    rank: int,
    resume: bool,
) -> Path:
    unit = store.unit(
        phase,
        "analysis",
        1,
        detail={
            "burn_in_steps": burn_in_steps,
            "adaptation_rank": rank,
            "conditions": [condition.mapping() for condition in conditions],
        },
    )
    session = store.begin(unit, ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    full_summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    adaptive: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        if condition.kind == "adaptive_subgd":
            adaptive[condition.name] = {
                "condition": condition.mapping(),
                "alphas": [],
                "betas": [],
                "knot_gains": [],
                "ordinary_gains": [],
                "projector_distances": [],
                "nlls": [],
                "finite": True,
            }
    for index in range(1, replica_count + 1):
        for schedule in ("linear", "sigmoid"):
            for condition in conditions:
                summary, metrics, checks = _completed_rotation_result(
                    store,
                    phase,
                    index,
                    schedule,
                    condition,
                    burn_in_steps=burn_in_steps,
                    rank=rank,
                )
                if condition.kind == "full_space":
                    full_summaries[schedule].append(summary)
                if condition.kind != "adaptive_subgd":
                    continue
                bucket = adaptive[condition.name]
                bucket["nlls"].append(float(summary["post_burn_in_current_nll_auc"]))
                bucket["finite"] = bucket["finite"] and bool(checks["all_finite"])
                for row in metrics:
                    geometry = row.get("geometry")
                    if geometry is None:
                        continue
                    bucket["alphas"].append(float(geometry["alpha"]))
                    bucket["betas"].append(float(geometry["beta"]))
                    bucket["projector_distances"].append(float(geometry["projector_distance"]))
                    gain = float(geometry["orthogonal_gain"])
                    bucket["knot_gains" if row["knot"] else "ordinary_gains"].append(gain)
    full_mean = _mean(
        [
            float(row["post_burn_in_current_nll_auc"])
            for values in full_summaries.values()
            for row in values
        ]
    )
    diagnostics = []
    for name, bucket in adaptive.items():
        alpha_range = max(bucket["alphas"]) - min(bucket["alphas"])
        beta_range = max(bucket["betas"]) - min(bucket["betas"])
        ordinary = statistics.median(bucket["ordinary_gains"])
        knot_increase = (
            0.0
            if not bucket["knot_gains"]
            else max(bucket["knot_gains"]) - ordinary
        )
        max_rotation = max(bucket["projector_distances"])
        relative_nll = _mean(bucket["nlls"]) / full_mean
        mechanically_healthy = (
            bucket["finite"]
            and (store.study.smoke or alpha_range >= 0.05)
            and (store.study.smoke or beta_range >= 1e-4)
            and (store.study.smoke or max_rotation >= 1e-6)
            and (store.study.smoke or knot_increase > 1e-4)
            and relative_nll <= 2.0
        )
        diagnostics.append(
            {
                "name": name,
                "condition": bucket["condition"],
                "all_finite": bucket["finite"],
                "alpha_range": alpha_range,
                "beta_range": beta_range,
                "knot_orthogonal_gain_increase": knot_increase,
                "max_projector_distance": max_rotation,
                "relative_current_nll_to_full": relative_nll,
                "mechanically_healthy": mechanically_healthy,
            }
        )
    survivors = [row for row in diagnostics if row["mechanically_healthy"]]
    fallback_used = not survivors
    if fallback_used:
        survivors = [
            max(
                diagnostics,
                key=lambda row: (
                    row["alpha_range"] + row["beta_range"] + row["max_projector_distance"],
                    -row["relative_current_nll_to_full"],
                ),
            )
        ]
    selection = {
        "survivor_conditions": [row["condition"] for row in survivors],
        "fallback_used": fallback_used,
        "classification": (
            "mechanically_healthy_controller_available"
            if not fallback_used
            else "universal_probe_failure_least_pathological_carried"
        ),
    }
    session.write_json(
        "summary.json",
        {"phase": phase, "diagnostics": diagnostics, "selection": selection},
    )
    session.write_json("selection.json", selection)
    return store.finish(session, ANALYSIS_REQUIRED)


def run_phase3_stage_analysis(
    store: UnitStore,
    *,
    phase: str,
    trajectory_phase: str | None = None,
    replica_count: int,
    conditions: tuple[Condition, ...],
    candidate_names: tuple[str, ...],
    burn_in_steps: int,
    rank: int,
    resume: bool,
    final_stage: bool = False,
) -> Path:
    data_phase = phase if trajectory_phase is None else trajectory_phase
    unit = store.unit(
        phase,
        "analysis",
        1,
        detail={
            "burn_in_steps": burn_in_steps,
            "adaptation_rank": rank,
            "candidate_names": list(candidate_names),
            "trajectory_phase": data_phase,
        },
    )
    session = store.begin(unit, ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    summaries = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index in range(1, replica_count + 1):
        for schedule in ("linear", "sigmoid"):
            for condition in conditions:
                summary, _, _ = _completed_rotation_result(
                    store,
                    data_phase,
                    index,
                    schedule,
                    condition,
                    burn_in_steps=burn_in_steps,
                    rank=rank,
                )
                summaries.append(summary)
                by_key[(schedule, condition.name)].append(summary)
    aggregate = _aggregate_summaries(summaries)
    candidate_rows = []
    for name in candidate_names:
        current_harms = []
        retention_harms = []
        nlls = []
        for schedule in ("linear", "sigmoid"):
            candidate = by_key[(schedule, name)]
            full = by_key[(schedule, "full_space")]
            for treatment, control in zip(candidate, full):
                current_harms.append(
                    (treatment["post_burn_in_current_nll_auc"] - control["post_burn_in_current_nll_auc"])
                    / control["post_burn_in_current_nll_auc"]
                )
                retention_harms.append(
                    (treatment["post_burn_in_worst_panel_nll_auc"] - control["post_burn_in_worst_panel_nll_auc"])
                    / control["post_burn_in_worst_panel_nll_auc"]
                )
                nlls.append(float(treatment["post_burn_in_current_nll_auc"]))
        candidate_rows.append(
            {
                "condition": name,
                "mean_current_nll_auc": _mean(nlls),
                "mean_current_harm": _mean(current_harms),
                "mean_retention_harm": _mean(retention_harms),
                "eligible": _mean(retention_harms) <= 0.01,
            }
        )
    eligible = [row for row in candidate_rows if row["eligible"]] or candidate_rows
    selected_row = min(
        eligible,
        key=lambda row: (row["mean_current_nll_auc"], row["mean_retention_harm"]),
    )
    selected = next(condition for condition in conditions if condition.name == selected_row["condition"])
    selection: dict[str, Any] = {
        "selected_condition": selected.mapping(),
        "fallback_used": not any(row["eligible"] for row in candidate_rows),
        "candidate_rows": candidate_rows,
    }
    if final_stage:
        sample_sizes = []
        for schedule in ("linear", "sigmoid"):
            method = [row["post_burn_in_current_nll_auc"] for row in by_key[(schedule, selected.name)]]
            full = [row["post_burn_in_current_nll_auc"] for row in by_key[(schedule, "full_space")]]
            differences = [reference - treatment for treatment, reference in zip(method, full)]
            delta = 0.01 * _mean(full)
            if len(differences) < 2 or statistics.stdev(differences) == 0:
                projected = store.study.phase4_replicas
            else:
                projected = math.ceil(
                    (1.96 + 0.84) ** 2 * statistics.variance(differences) / (delta**2)
                )
            sample_sizes.append(
                {
                    "schedule": schedule,
                    "raw_projected_replicas": projected,
                    "smallest_effect_current_nll_auc": delta,
                }
            )
        frozen_count = max(
            1 if store.study.smoke else 16,
            min(store.study.phase4_replicas, max(row["raw_projected_replicas"] for row in sample_sizes)),
        )
        selection["phase4_replica_count"] = frozen_count
        selection["phase4_power_projection"] = sample_sizes
        nonadaptive_names = [
            condition.name
            for condition in conditions
            if condition.kind in {"static_subgd", "online_subgd"}
        ]
        nonadaptive_rows = [
            row for row in aggregate if row["condition"] in nonadaptive_names
        ]
        pooled: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in nonadaptive_rows:
            pooled[row["condition"]].append(row)
        selected_nonadaptive = min(
            pooled,
            key=lambda name: _mean([row["mean_current_nll_auc"] for row in pooled[name]]),
        )
        selection["selected_nonadaptive_condition"] = next(
            condition.mapping() for condition in conditions if condition.name == selected_nonadaptive
        )
    session.write_json(
        "summary.json",
        {"phase": phase, "aggregate": aggregate, "selection": selection},
    )
    session.write_json("selection.json", selection)
    return store.finish(session, ANALYSIS_REQUIRED)


def run_phase4_analysis(
    store: UnitStore,
    *,
    replica_count: int,
    conditions: tuple[Condition, ...],
    adaptive_name: str,
    burn_in_steps: int,
    rank: int,
    resume: bool,
) -> Path:
    phase = "phase4"
    unit = store.unit(
        phase,
        "analysis",
        1,
        detail={
            "burn_in_steps": burn_in_steps,
            "adaptation_rank": rank,
            "replica_count": replica_count,
            "conditions": [condition.mapping() for condition in conditions],
        },
    )
    session = store.begin(unit, ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    all_rows = []
    for index in range(1, replica_count + 1):
        for schedule in ("linear", "sigmoid"):
            for condition in conditions:
                summary, _, _ = _completed_rotation_result(
                    store,
                    phase,
                    index,
                    schedule,
                    condition,
                    burn_in_steps=burn_in_steps,
                    rank=rank,
                )
                by_key[(schedule, condition.name)].append(summary)
                all_rows.append(summary)
    comparisons = []
    schedule_gate: dict[str, bool] = {}
    schedule_harm: dict[str, bool] = {}
    for schedule in ("linear", "sigmoid"):
        method = by_key[(schedule, adaptive_name)]
        full = by_key[(schedule, "full_space")]
        random = by_key[(schedule, "random_rank_matched")]
        current = _paired_effect(
            [row["post_burn_in_current_nll_auc"] for row in method],
            [row["post_burn_in_current_nll_auc"] for row in full],
        )
        retention = _paired_effect(
            [row["post_burn_in_worst_panel_nll_auc"] for row in method],
            [row["post_burn_in_worst_panel_nll_auc"] for row in full],
        )
        versus_random = _paired_effect(
            [row["post_burn_in_current_nll_auc"] for row in method],
            [row["post_burn_in_current_nll_auc"] for row in random],
        )
        retention_versus_random = _paired_effect(
            [row["post_burn_in_worst_panel_nll_auc"] for row in method],
            [row["post_burn_in_worst_panel_nll_auc"] for row in random],
        )
        full_current = _mean([row["post_burn_in_current_nll_auc"] for row in full])
        full_retention = _mean([row["post_burn_in_worst_panel_nll_auc"] for row in full])
        current_case = (
            current["mean_gain"] >= 0.01 * full_current
            and retention["mean_gain"] >= -0.01 * full_retention
        )
        retention_case = (
            retention["mean_gain"] >= 0.01 * full_retention
            and current["mean_gain"] >= -0.01 * full_current
        )
        schedule_gate[schedule] = (
            (
                current_case
                and current["ci95_low"] > 0
                and versus_random["mean_gain"] > 0
            )
            or (
                retention_case
                and retention["ci95_low"] > 0
                and retention_versus_random["mean_gain"] > 0
            )
        )
        schedule_harm[schedule] = (
            current["mean_gain"] < -0.01 * full_current
            or retention["mean_gain"] < -0.01 * full_retention
        )
        comparisons.append(
            {
                "schedule": schedule,
                "adaptive_vs_full_current_nll": current,
                "adaptive_vs_full_retention_nll": retention,
                "adaptive_vs_random_current_nll": versus_random,
                "adaptive_vs_random_retention_nll": retention_versus_random,
                "promotion_gate": schedule_gate[schedule],
                "material_harm": schedule_harm[schedule],
            }
        )
    promoted = any(schedule_gate.values()) and not any(
        schedule_harm[schedule]
        for schedule in schedule_harm
        if not schedule_gate[schedule]
    )
    adaptive = next(condition for condition in conditions if condition.name == adaptive_name)
    selection = {
        "promoted": promoted,
        "dynamic_transport_condition": adaptive.mapping(),
        "classification": "promising" if promoted else "not_promoted_diagnostic_best_bet",
    }
    session.write_json(
        "summary.json",
        {
            "phase": phase,
            "aggregate": _aggregate_summaries(all_rows),
            "comparisons": comparisons,
            "selection": selection,
        },
    )
    session.write_json("selection.json", selection)
    return store.finish(session, ANALYSIS_REQUIRED)


def run_phase5_analysis(
    store: UnitStore,
    *,
    conditions: tuple[Condition, ...],
    dynamic_name: str,
    burn_in_steps: int,
    rank: int,
    resume: bool,
) -> Path:
    phase = "phase5"
    unit = store.unit(
        phase,
        "analysis",
        1,
        environment="digit9_mixture",
        detail={
            "burn_in_steps": burn_in_steps,
            "adaptation_rank": rank,
            "replica_count": store.study.phase5_replicas,
            "conditions": [condition.mapping() for condition in conditions],
        },
    )
    session = store.begin(unit, ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = []
    for index in range(1, store.study.phase5_replicas + 1):
        for condition in conditions:
            path = store.completed(
                mixture_trajectory_unit(
                    store,
                    index,
                    condition,
                    burn_in_steps=burn_in_steps,
                    rank=rank,
                ),
                MIXTURE_TRAJECTORY_REQUIRED,
            )
            if path is None:
                raise RuntimeError(f"missing Phase 5 trajectory {index}/{condition.name}")
            summary = _read_json(path / "summary.json")
            rows.append(summary)
            by_condition[condition.name].append(summary)
    full = by_condition["full_space"]
    comparisons = []
    for condition in conditions:
        if condition.name == "full_space":
            continue
        method = by_condition[condition.name]
        comparisons.append(
            {
                "condition": condition.name,
                "current_nll_vs_full": _paired_effect(
                    [row["post_burn_in_current_nll_auc"] for row in method],
                    [row["post_burn_in_current_nll_auc"] for row in full],
                ),
                "retention_nll_vs_full": _paired_effect(
                    [row["post_burn_in_retention_nll_auc"] for row in method],
                    [row["post_burn_in_retention_nll_auc"] for row in full],
                ),
            }
        )
    dynamic = next(row for row in comparisons if row["condition"] == dynamic_name)
    selection = {
        "transport_only": True,
        "dynamic_condition": dynamic_name,
        "dynamic_current_nll_mean_gain": dynamic["current_nll_vs_full"]["mean_gain"],
        "dynamic_retention_nll_mean_gain": dynamic["retention_nll_vs_full"]["mean_gain"],
        "rotation_and_mixture_evidence_pooled": False,
    }
    session.write_json(
        "summary.json",
        {
            "phase": phase,
            "environment": "digit9_mixture",
            "aggregate": _aggregate_summaries(rows),
            "comparisons": comparisons,
            "selection": selection,
        },
    )
    session.write_json("selection.json", selection)
    return store.finish(session, ANALYSIS_REQUIRED)


__all__ = [
    "ANALYSIS_REQUIRED",
    "_burn_in_unit",
    "_trajectory_unit",
    "run_phase0_analysis",
    "run_phase1_analysis",
    "run_phase2_analysis",
    "run_phase3_probe_analysis",
    "run_phase3_stage_analysis",
    "run_phase4_analysis",
    "run_phase5_analysis",
]
