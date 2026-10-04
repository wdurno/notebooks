"""Artifact-only statistical summaries and frozen Plan 12 decisions."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..artifacts import _read_json
from .artifacts import UnitStore
from .local_response import (
    BRANCH_REQUIRED,
    REFERENCE_REQUIRED,
    TARGET_REQUIRED,
    LocalCondition,
    local_conditions,
)
from .mp_calibration import CALIBRATION_REQUIRED
from .trajectory import TRAJECTORY_REQUIRED, TrajectoryCondition


PHASE1_ANALYSIS_REQUIRED = ("summary.json", "selection.json")
TRAJECTORY_ANALYSIS_REQUIRED = ("summary.json",)
PHASE3_ANALYSIS_REQUIRED = ("summary.json", "selection.json")
MINIMUM_RELATIVE_MSE_IMPROVEMENT = 0.005
TARGET_REPEAT_TOLERANCE = 1e-2
TARGET_RELATIVE_GRADIENT_TOLERANCE = 0.05


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty collection")
    return statistics.fmean(values)


def _optional_mean(values: list[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return None if not available else _mean(available)


def _target_fit_valid(summary: dict[str, Any]) -> bool:
    if summary["condition"]["no_update"]:
        return True
    health = summary["fit_health"]
    repeat_health = summary["repeat_fit_health"]
    relative = float(health["relative_final_gradient_norm"])
    repeat_relative = float(repeat_health["relative_final_gradient_norm"])
    return (
        summary["repeat_parameter_distance"] <= TARGET_REPEAT_TOLERANCE
        and math.isfinite(relative)
        and math.isfinite(repeat_relative)
        and relative <= TARGET_RELATIVE_GRADIENT_TOLERANCE
        and repeat_relative <= TARGET_RELATIVE_GRADIENT_TOLERANCE
        and float(health["objective_decrease"]) >= 0
        and float(repeat_health["objective_decrease"]) >= 0
    )


def _components(vector: Tensor, basis: Tensor) -> dict[str, float]:
    resolved = basis.mT @ vector
    unresolved = vector - basis @ resolved
    return {
        "full": float(vector.square().sum()),
        "resolved": float(resolved.square().sum()),
        "unresolved": float(unresolved.square().sum()),
    }


def _identified_logits(logits: Tensor) -> Tensor:
    """Remove the likelihood-invariant common-class logit shift."""
    return logits - logits.mean(dim=-1, keepdim=True)


def decompose_estimates(
    estimates: Tensor,
    penalized_target: Tensor,
    unregularized_reference: Tensor,
    resolved_basis: Tensor,
    reference_fisher: Tensor,
) -> dict[str, Any]:
    """Separate conditional variance, estimator bias, and ridge bias."""
    if estimates.ndim != 2 or penalized_target.shape != estimates.shape[1:] or unregularized_reference.shape != penalized_target.shape:
        raise ValueError("bias-variance tensors are incompatible")
    if reference_fisher.shape != (estimates.shape[1], estimates.shape[1]):
        raise ValueError("reference Fisher has the wrong shape")
    mean = estimates.mean(dim=0)
    centered = estimates - mean
    variance = float(centered.square().sum(dim=1).mean())
    fisher_variance = float(torch.einsum("bi,ij,bj->b", centered, reference_fisher, centered).mean())
    estimator_bias = mean - penalized_target
    regularization_bias = penalized_target - unregularized_reference
    total_bias = mean - unregularized_reference
    return {
        "parameter_variance": variance,
        "fisher_variance": fisher_variance,
        "estimator_bias_squared": _components(estimator_bias, resolved_basis),
        "regularization_bias_squared": _components(regularization_bias, resolved_basis),
        "total_bias_squared": _components(total_bias, resolved_basis),
        "fisher_estimator_bias_squared": float(estimator_bias @ reference_fisher @ estimator_bias),
        "fisher_regularization_bias_squared": float(regularization_bias @ reference_fisher @ regularization_bias),
        "fisher_total_bias_squared": float(total_bias @ reference_fisher @ total_bias),
        "fisher_total_mse": fisher_variance + float(total_bias @ reference_fisher @ total_bias),
    }


def _has_adjacent_ratios(rows: list[dict[str, Any]], all_candidates: list[dict[str, Any]]) -> bool:
    positions = {
        row["ridge_ratio"]: index
        for index, row in enumerate(sorted(all_candidates, key=lambda item: item["ridge_ratio"]))
    }
    selected = sorted(positions[row["ridge_ratio"]] for row in rows)
    return any(right == left + 1 for left, right in zip(selected, selected[1:]))


def _local_paths(
    store: UnitStore,
    anchor_index: int,
    condition: LocalCondition,
) -> tuple[Path, list[Path]]:
    target_unit = store.unit(
        "phase1",
        "target",
        anchor_index,
        condition=condition.name,
        detail=condition.mapping(),
    )
    target = store.completed(target_unit, TARGET_REQUIRED)
    if target is None:
        raise RuntimeError(f"missing local target {anchor_index}/{condition.name}")
    branches = []
    for batch_index in range(1, store.study.local_batches_per_anchor + 1):
        unit_index = (anchor_index - 1) * store.study.local_batches_per_anchor + batch_index
        unit = store.unit(
            "phase1",
            "branch",
            unit_index,
            condition=condition.name,
            detail={"anchor_index": anchor_index, "batch_index": batch_index, **condition.mapping()},
        )
        path = store.completed(unit, BRANCH_REQUIRED)
        if path is None:
            raise RuntimeError(f"missing local branch {anchor_index}/{batch_index}/{condition.name}")
        branches.append(path)
    return target, branches


def run_phase1_analysis(store: UnitStore, *, resume: bool) -> Path:
    unit = store.unit("phase1", "analysis", 1)
    session = store.begin(unit, PHASE1_ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, PHASE1_ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    conditions = local_conditions(store.study.ridge_scale_ratios)
    rows: list[dict[str, Any]] = []
    anchor_count = 2 * store.study.anchors_per_schedule
    for anchor_index in range(1, anchor_count + 1):
        reference_unit = store.unit("phase1", "reference", anchor_index)
        reference_path = store.completed(reference_unit, REFERENCE_REQUIRED)
        if reference_path is None:
            raise RuntimeError(f"missing Plan 12 reference for anchor {anchor_index}")
        reference = torch.load(reference_path / "reference.pt", map_location="cpu", weights_only=False)
        fisher = reference["fisher"].to(torch.float64)
        basis = reference["resolved_basis"].to(torch.float64)
        current_target_path, _ = _local_paths(store, anchor_index, LocalCondition("current_only", current_only=True))
        current_target = torch.load(current_target_path / "target.pt", map_location="cpu", weights_only=False)
        current_target_summary = _read_json(current_target_path / "summary.json")
        current_target_valid = _target_fit_valid(current_target_summary)
        current_logits = _identified_logits(current_target["current_logits"].to(torch.float64))
        current_probabilities = current_target["current_probabilities"].to(torch.float64)
        current_parameter = current_target["parameter"].to(torch.float64)
        for condition in conditions:
            target_path, branch_paths = _local_paths(store, anchor_index, condition)
            target_artifact = torch.load(target_path / "target.pt", map_location="cpu", weights_only=False)
            target_summary = _read_json(target_path / "summary.json")
            penalized_target_valid = _target_fit_valid(target_summary)
            branch_artifacts = [
                torch.load(path / "estimate.pt", map_location="cpu", weights_only=False)
                for path in branch_paths
            ]
            branch_summaries = [_read_json(path / "summary.json") for path in branch_paths]
            logits = torch.stack([
                _identified_logits(artifact["current_logits"].to(torch.float64))
                for artifact in branch_artifacts
            ])
            mean_logits = logits.mean(dim=0)
            target_logits = _identified_logits(target_artifact["current_logits"].to(torch.float64))
            functional_variance = float((logits - mean_logits).square().mean(dim=(1, 2)).mean())
            functional_estimator_bias = float((mean_logits - target_logits).square().mean()) if penalized_target_valid else None
            functional_regularization_bias = float((target_logits - current_logits).square().mean()) if penalized_target_valid and current_target_valid else None
            functional_total_bias = float((mean_logits - current_logits).square().mean()) if current_target_valid else None
            probabilities = torch.stack([
                artifact["current_probabilities"].to(torch.float64)
                for artifact in branch_artifacts
            ])
            mean_probabilities = probabilities.mean(dim=0)
            target_probabilities = target_artifact["current_probabilities"].to(torch.float64)
            probability_variance = float((probabilities - mean_probabilities).square().mean(dim=(1, 2)).mean())
            probability_estimator_bias = float((mean_probabilities - target_probabilities).square().mean()) if penalized_target_valid else None
            probability_regularization_bias = float((target_probabilities - current_probabilities).square().mean()) if penalized_target_valid and current_target_valid else None
            probability_total_bias = float((mean_probabilities - current_probabilities).square().mean()) if current_target_valid else None
            row: dict[str, Any] = {
                "anchor_index": anchor_index,
                "schedule": "linear" if anchor_index <= store.study.anchors_per_schedule else "sigmoid",
                "condition": condition.name,
                "ridge_geometry": condition.ridge_geometry,
                "ridge_ratio": condition.ridge_ratio,
                "kappa": target_summary["kappa"],
                "target_repeat_distance": target_summary["repeat_parameter_distance"],
                "current_reference_valid": current_target_valid,
                "penalized_target_valid": penalized_target_valid,
                "target_valid": current_target_valid and penalized_target_valid,
                "functional_variance": functional_variance,
                "functional_estimator_bias_squared": functional_estimator_bias,
                "functional_regularization_bias_squared": functional_regularization_bias,
                "functional_total_bias_squared": functional_total_bias,
                "functional_total_mse": None if functional_total_bias is None else functional_variance + functional_total_bias,
                "probability_variance": probability_variance,
                "probability_estimator_bias_squared": probability_estimator_bias,
                "probability_regularization_bias_squared": probability_regularization_bias,
                "probability_total_bias_squared": probability_total_bias,
                "probability_total_mse": None if probability_total_bias is None else probability_variance + probability_total_bias,
                "mean_current_nll": _mean([summary["metrics"]["current_nll"] for summary in branch_summaries]),
                "mean_current_accuracy": _mean([summary["metrics"]["current_accuracy"] for summary in branch_summaries]),
                "mean_current_brier": _mean([summary["metrics"]["current_brier"] for summary in branch_summaries]),
                "mean_worst_retention_nll": _mean([
                    max(summary["metrics"][f"retention_{angle}_nll"] for angle in ("000", "015", "030"))
                    for summary in branch_summaries
                ]),
                "mean_final_gradient_norm": _mean([
                    float(summary["fit_health"]["final_gradient_norm"] or 0.0) for summary in branch_summaries
                ]),
            }
            if condition.chart:
                estimates = torch.stack([artifact["parameter"].to(torch.float64) for artifact in branch_artifacts])
                target_parameter = target_artifact["parameter"].to(torch.float64)
                decomposition = decompose_estimates(
                        estimates,
                        target_parameter,
                        current_parameter,
                        basis,
                        fisher,
                    )
                if not current_target_valid:
                    for key in (
                        "estimator_bias_squared",
                        "regularization_bias_squared",
                        "total_bias_squared",
                        "fisher_estimator_bias_squared",
                        "fisher_regularization_bias_squared",
                        "fisher_total_bias_squared",
                        "fisher_total_mse",
                    ):
                        decomposition[key] = None
                elif not penalized_target_valid:
                    for key in (
                        "estimator_bias_squared",
                        "regularization_bias_squared",
                        "fisher_estimator_bias_squared",
                        "fisher_regularization_bias_squared",
                    ):
                        decomposition[key] = None
                row.update(decomposition)
            else:
                row.update(
                    {
                        "parameter_variance": None,
                        "fisher_variance": None,
                        "estimator_bias_squared": None,
                        "regularization_bias_squared": None,
                        "total_bias_squared": None,
                        "fisher_estimator_bias_squared": None,
                        "fisher_regularization_bias_squared": None,
                        "fisher_total_bias_squared": None,
                        "fisher_total_mse": None,
                    }
                )
            rows.append(row)

    aggregate = []
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition.name]
        aggregate.append(
            {
                "condition": condition.name,
                "ridge_geometry": condition.ridge_geometry,
                "ridge_ratio": condition.ridge_ratio,
                "target_valid_fraction": _mean([float(row["target_valid"]) for row in subset]),
                "current_reference_valid_fraction": _mean([
                    float(row["current_reference_valid"]) for row in subset
                ]),
                "fisher_total_mse": _optional_mean([row["fisher_total_mse"] for row in subset]),
                "fisher_variance": _optional_mean([row["fisher_variance"] for row in subset]),
                "fisher_total_bias_squared": _optional_mean([row["fisher_total_bias_squared"] for row in subset]),
                "parameter_variance": _optional_mean([row["parameter_variance"] for row in subset]),
                "functional_total_mse": _optional_mean([row["functional_total_mse"] for row in subset]),
                "probability_total_mse": _optional_mean([row["probability_total_mse"] for row in subset]),
                "current_nll": _mean([row["mean_current_nll"] for row in subset]),
                "current_accuracy": _mean([row["mean_current_accuracy"] for row in subset]),
                "worst_retention_nll": _mean([row["mean_worst_retention_nll"] for row in subset]),
                "heldout_current_plus_retention_nll": _mean([
                    row["mean_current_nll"] + row["mean_worst_retention_nll"] for row in subset
                ]),
            }
        )
    baseline = next(row for row in aggregate if row["condition"] == "gauge_no_ridge")
    selection: dict[str, Any] = {"baseline": baseline}
    for geometry in ("isotropic", "tail"):
        candidates = [row for row in aggregate if row["ridge_geometry"] == geometry and row["ridge_ratio"] > 0]
        use_fisher_mse = baseline["fisher_total_mse"] is not None and any(
            row["fisher_total_mse"] is not None and row["current_reference_valid_fraction"] >= 0.5
            for row in candidates
        )
        metric = "fisher_total_mse" if use_fisher_mse else "heldout_current_plus_retention_nll"
        baseline_value = float(baseline[metric])
        eligible = [
            row
            for row in candidates
            if row[metric] is not None and (not use_fisher_mse or row["current_reference_valid_fraction"] >= 0.5)
        ]
        improving = [
            row
            for row in eligible
            if row[metric] <= baseline_value * (1.0 - MINIMUM_RELATIVE_MSE_IMPROVEMENT)
            and row["current_nll"] <= baseline["current_nll"] + 0.01
        ]
        pool = improving or eligible
        best_value = min(row[metric] for row in pool)
        practically_tied = [
            row
            for row in pool
            if row[metric] <= best_value * (1.0 + MINIMUM_RELATIVE_MSE_IMPROVEMENT)
        ]
        chosen = min(practically_tied, key=lambda row: row["ridge_ratio"])
        if _has_adjacent_ratios(improving, candidates):
            classification = "promising_contiguous_region"
        elif improving:
            classification = "isolated_improvement_fallback"
        else:
            classification = "likely_unfavorable_fallback"
        selection[geometry] = {
            "classification": classification,
            "selected_condition": chosen["condition"],
            "selected_ratio": chosen["ridge_ratio"],
            "eligible_improving_conditions": [row["condition"] for row in improving],
            "selection_metric": (
                "mean_reference_fisher_weighted_total_mse"
                if use_fisher_mse
                else "mean_heldout_current_plus_worst_retention_nll"
            ),
            "minimum_relative_improvement": MINIMUM_RELATIVE_MSE_IMPROVEMENT,
            "practical_tie_relative_tolerance": MINIMUM_RELATIVE_MSE_IMPROVEMENT,
        }
    summary = {"rows": rows, "aggregate": aggregate, "selection": selection}
    session.write_json("summary.json", summary)
    session.write_json("selection.json", selection)
    return store.finish(session, PHASE1_ANALYSIS_REQUIRED)


def _effect(values: list[float]) -> dict[str, float]:
    mean = _mean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = standard_deviation / len(values) ** 0.5
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "replicas": len(values),
        "positive_count": sum(value > 0 for value in values),
    }


def run_trajectory_analysis(
    store: UnitStore,
    phase: str,
    conditions: tuple[TrajectoryCondition, ...],
    replicas: int,
    *,
    resume: bool,
) -> Path:
    unit = store.unit(phase, "analysis", 1, detail={"conditions": [item.mapping() for item in conditions], "replicas": replicas})
    session = store.begin(unit, TRAJECTORY_ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, TRAJECTORY_ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    summaries: dict[tuple[str, str], list[dict[str, Any]]] = {}
    paths: dict[tuple[str, str], list[Path]] = {}
    metric_cache: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    for schedule in ("linear", "sigmoid"):
        for condition in conditions:
            key = (schedule, condition.name)
            summaries[key], paths[key] = [], []
            metric_cache[key] = []
            for index in range(1, replicas + 1):
                trajectory_unit = store.unit(
                    phase,
                    "trajectory",
                    index,
                    schedule=schedule,
                    condition=condition.name,
                    detail=condition.mapping(),
                )
                path = store.completed(trajectory_unit, TRAJECTORY_REQUIRED)
                if path is None:
                    raise RuntimeError(f"missing {phase} trajectory {index}/{schedule}/{condition.name}")
                summaries[key].append(_read_json(path / "summary.json"))
                paths[key].append(path)
                metric_cache[key].append(_read_json(path / "metrics.json"))
    result_rows = []
    paired = []
    for schedule in ("linear", "sigmoid"):
        baseline = summaries[(schedule, "gauge_no_ridge")]
        baseline_metrics = metric_cache[(schedule, "gauge_no_ridge")]
        for condition in conditions:
            values = summaries[(schedule, condition.name)]
            row = {
                "schedule": schedule,
                "condition": condition.name,
                "mean_current_nll_auc": _mean([value["current_nll_auc"] for value in values]),
                "mean_current_accuracy_auc": _mean([value["current_accuracy_auc"] for value in values]),
                "mean_current_brier_auc": _mean([value["current_brier_auc"] for value in values]),
                "mean_worst_panel_nll_auc": _mean([value["worst_panel_nll_auc"] for value in values]),
                "mean_wall_seconds": _mean([value["total_wall_time_seconds"] for value in values]),
            }
            if condition.chart:
                trajectories = torch.stack([
                    torch.load(path / "trajectory.pt", map_location="cpu", weights_only=False)["parameters"].to(torch.float64)
                    for path in paths[(schedule, condition.name)]
                ])
                row["mean_across_replica_parameter_variance"] = float(
                    trajectories.var(dim=0, correction=1).sum(dim=1).mean()
                )
            else:
                row["mean_across_replica_parameter_variance"] = None
            metrics = metric_cache[(schedule, condition.name)]
            recommendations = [
                float(step["shadow_pi"]["recommendation_pi"])
                for replica in metrics
                for step in replica
                if step["shadow_pi"] is not None
            ]
            sandwich = [
                float(step["sandwich"]["covariance_trace"])
                for replica in metrics
                for step in replica
                if step["sandwich"] is not None
            ]
            row["shadow_pi_mean"] = _mean(recommendations)
            row["shadow_pi_variance"] = statistics.pvariance(recommendations)
            row["mean_penalized_sandwich_trace"] = _mean(sandwich)
            per_replica_pi = [
                _mean([
                    float(step["shadow_pi"]["recommendation_pi"])
                    for step in replica
                    if step["shadow_pi"] is not None
                ])
                for replica in metrics
            ]
            row["across_replica_shadow_pi_variance"] = statistics.pvariance(per_replica_pi)
            shadow_steps = [step["shadow_pi"] for replica in metrics for step in replica if step["shadow_pi"] is not None]
            row["shadow_pi_boundary_fraction"] = _mean([
                float(step["lower_bound_active"] or step["upper_bound_active"] or step["unsupported_scale_fallback"])
                for step in shadow_steps
            ])
            ridge_steps = [step["ridge"] for replica in metrics for step in replica if step["ridge"] is not None]
            row["mean_total_displacement_squared"] = _mean([float(step["total_squared"]) for step in ridge_steps])
            row["mean_resolved_displacement_squared"] = _mean([float(step["resolved_squared"]) for step in ridge_steps])
            row["mean_unresolved_displacement_squared"] = _mean([float(step["unresolved_squared"]) for step in ridge_steps])
            updates = [step["fisher_update"] for replica in metrics for step in replica if step["fisher_update"] is not None]
            row["mean_compression_relative_frobenius_error"] = _mean([
                float(step["compression_relative_frobenius_error"]) for step in updates
            ])
            proposals = [step["proposal"] for replica in metrics for step in replica if step["proposal"] is not None]
            row["mean_final_gradient_norm"] = _mean([float(step["final_gradient_norm"]) for step in proposals])
            result_rows.append(row)
            if condition.name != "gauge_no_ridge":
                baseline_replica_pi = [
                    _mean([
                        float(step["shadow_pi"]["recommendation_pi"])
                        for step in replica
                        if step["shadow_pi"] is not None
                    ])
                    for replica in baseline_metrics
                ]
                paired.append(
                    {
                        "schedule": schedule,
                        "condition": condition.name,
                        "nll_auc_gain": _effect([
                            baseline[index]["current_nll_auc"] - values[index]["current_nll_auc"]
                            for index in range(replicas)
                        ]),
                        "accuracy_auc_gain": _effect([
                            values[index]["current_accuracy_auc"] - baseline[index]["current_accuracy_auc"]
                            for index in range(replicas)
                        ]),
                        "shadow_pi_mean_shift": _effect([
                            per_replica_pi[index] - baseline_replica_pi[index]
                            for index in range(replicas)
                        ]),
                    }
                )
    summary = {
        "phase": phase,
        "replicas": replicas,
        "conditions": [item.mapping() for item in conditions],
        "rows": result_rows,
        "paired_against_gauge_no_ridge": paired,
        "pi_reference_status": "unavailable_without_a_valid_high_sample_local_reference; empirical stability and sandwich calibration only",
    }
    session.write_json("summary.json", summary)
    return store.finish(session, TRAJECTORY_ANALYSIS_REQUIRED)


def run_phase3_analysis(store: UnitStore, *, resume: bool) -> Path:
    unit = store.unit("phase3", "analysis", 1)
    session = store.begin(unit, PHASE3_ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, PHASE3_ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    rows = []
    for anchor_index in range(1, 2 * store.study.anchors_per_schedule + 1):
        calibration_unit = store.unit("phase3", "calibration", anchor_index)
        path = store.completed(calibration_unit, CALIBRATION_REQUIRED)
        if path is None:
            raise RuntimeError(f"missing calibration checkpoint {anchor_index}")
        rows.append(_read_json(path / "summary.json"))
    theoretical = [row for row in rows if row["theoretical"] is not None]
    valid = [row for row in theoretical if row["status"] == "complete"]
    fallback = len(valid) != len(rows)
    if valid and not fallback:
        selected_ratio = statistics.median(row["selected_scale_ratio"] for row in valid)
        source = "empirical_99_percent_tail_maximum"
    elif theoretical:
        selected_ratio = statistics.median(
            row["theoretical"]["edge"]["upper_quantile"] / row["mean_eigenvalue"]
            for row in theoretical
        )
        source = "deformed_mp_theoretical_fallback"
    else:
        selected_ratio = statistics.median(
            row["heuristic_kappa"] / row["mean_eigenvalue"] for row in rows
        )
        source = "point_zero_one_top_eigenvalue_fallback"
    selection = {
        "classification": "empirical_selector" if not fallback else "theoretical_or_heuristic_fallback",
        "selected_scale_ratio": selected_ratio,
        "selection_source": source,
        "valid_checkpoint_count": len(valid),
        "failed_checkpoint_count": len(rows) - len(valid),
        "fallback": fallback,
    }
    summary = {
        "rows": rows,
        "selection": selection,
        "mean_empirical_to_theoretical_ratio": None if not theoretical else _mean([row["theoretical"]["empirical_to_theoretical_ratio"] for row in theoretical]),
        "pseudo_tail_pit_values": [row["pseudo_tail"]["empirical_pit"] for row in theoretical if row["pseudo_tail"] is not None],
    }
    session.write_json("summary.json", summary)
    session.write_json("selection.json", selection)
    return store.finish(session, PHASE3_ANALYSIS_REQUIRED)
