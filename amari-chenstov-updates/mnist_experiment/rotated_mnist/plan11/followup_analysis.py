"""Artifact-only progress and fixed-size inference for the .025 follow-up."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import t

from ..artifacts import RotatedArtifactError, _read_json
from .followup import DEFAULT_CONFIG, DEFAULT_ROOT, POLICIES, REPLICAS, SCHEDULES, contract, make_store, planned_units
from .config import Study
from .run import ASSET_REQUIRED, TRAJECTORY_REQUIRED


def _inference(differences: list[float]) -> dict[str, Any]:
    n = len(differences)
    mean = statistics.fmean(differences)
    sd = statistics.stdev(differences)
    sem = sd / math.sqrt(n)
    critical = float(t.ppf(0.975, n - 1))
    if sem == 0:
        p_value = 1.0 if mean == 0 else 0.0
    else:
        p_value = float(2 * t.sf(abs(mean / sem), n - 1))
    return {
        "mean_gain": mean,
        "standard_deviation": sd,
        "confidence_low": mean - critical * sem,
        "confidence_high": mean + critical * sem,
        "two_sided_p": p_value,
        "favorable_at_nominal_005": mean > 0 and p_value < 0.05,
    }


def _running_auc(exposure: list[int], values: list[float]) -> list[float]:
    result = [values[0]]
    area = 0.0
    for index in range(1, len(values)):
        area += (values[index - 1] + values[index]) * (exposure[index] - exposure[index - 1]) / 2
        result.append(area / (exposure[index] - exposure[0]))
    return result


def _curve(path: Path, schedule: str, points: int, samples_per_step: int) -> dict[str, list[float] | list[int]]:
    records = _read_json(path / "metrics.json")
    if not isinstance(records, list) or len(records) != points:
        raise RotatedArtifactError(f"incompatible trajectory length: {path}")
    exposure: list[int] = []
    angle: list[float] = []
    nll: list[float] = []
    accuracy: list[float] = []
    try:
        for index, row in enumerate(records):
            if row["step"] != index or row["schedule_kind"] != schedule:
                raise ValueError("step or schedule differs")
            if row["observations_before_evaluation"] != index * samples_per_step:
                raise ValueError("pre-update exposure clock differs")
            exposure.append(int(row["observations_before_evaluation"]))
            angle.append(float(row["angle_degrees"]))
            nll.append(float(row["current_nll"]))
            accuracy.append(float(row["current_environment_accuracy"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RotatedArtifactError(f"incompatible trajectory rows: {path}: {exc}") from exc
    if any(not math.isfinite(value) for value in (*angle, *nll, *accuracy)):
        raise RotatedArtifactError(f"nonfinite trajectory metric: {path}")
    if any(value < 0 for value in nll) or any(not 0 <= value <= 1 for value in accuracy):
        raise RotatedArtifactError(f"invalid trajectory metric range: {path}")
    return {"exposure": exposure, "angle": angle, "nll": nll, "accuracy": accuracy}


def _mean_trajectory(
    exposure: list[int], angle: list[float], sums: dict[str, list[float]], count: int
) -> dict[str, list[float] | list[int]]:
    result: dict[str, list[float] | list[int]] = {"exposure": exposure, "angle": angle}
    for key, values in sums.items():
        mean_values = [value / count for value in values]
        result[key] = mean_values
        result[key.replace("_nll", "_nll_auc").replace("_accuracy", "_accuracy_auc")] = _running_auc(exposure, mean_values)
    return result


def load_progress(
    repo_root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    study = Study.from_path(repo_root / config_path)
    if study.smoke:
        raise ValueError("production notebook cannot analyze smoke replicas")
    store = make_store(repo_root / output_root, study, repo_root)
    contract_path = store.root / "contract.json"
    ledger_path = store.root / "ledger.json"
    if not contract_path.is_file() or not ledger_path.is_file():
        raise RotatedArtifactError("focused .025 follow-up ledger has not been frozen")
    if _read_json(contract_path) != contract(store):
        raise RotatedArtifactError("focused .025 follow-up contract differs")
    units = planned_units(store)
    if _read_json(ledger_path) != {"units": units}:
        raise RotatedArtifactError("focused .025 follow-up ledger differs")

    by_schedule: dict[str, dict[str, Any]] = {}
    unit_lookup = {
        (unit["replica_index"], unit["schedule"], unit["policy"]["gain"]): unit
        for unit in units
    }
    validated_assets: set[int] = set()
    points = 3 * study.protocol.rotation.transitions_per_arrow + 1
    samples_per_step = study.protocol.data.samples_per_step
    for schedule in SCHEDULES:
        rows = []
        incomplete = []
        reference_exposure: list[int] | None = None
        reference_angle: list[float] | None = None
        curve_sums: dict[str, list[float]] = {}
        for index in range(1, REPLICAS + 1):
            fixed = store.completed(unit_lookup[index, schedule, POLICIES[0].gain], TRAJECTORY_REQUIRED)
            blend = store.completed(unit_lookup[index, schedule, POLICIES[1].gain], TRAJECTORY_REQUIRED)
            if fixed is None or blend is None:
                incomplete.append(index)
                continue
            if index not in validated_assets:
                asset = store.completed(store.unit(units[0]["phase"], index, None, None), ASSET_REQUIRED)
                if asset is None:
                    raise RotatedArtifactError(f"paired replica {index} lacks completed shared assets")
                asset_summary = _read_json(asset / "summary.json")
                if asset_summary.get("stream_identity_paired") is not True:
                    raise RotatedArtifactError(f"paired replica {index} has unpaired stream identities")
                validated_assets.add(index)
            fixed_summary = _read_json(fixed / "summary.json")
            blend_summary = _read_json(blend / "summary.json")
            for path, summary, policy in zip((fixed, blend), (fixed_summary, blend_summary), POLICIES):
                checks = _read_json(path / "checks.json")
                if (
                    summary.get("schedule_kind") != schedule
                    or summary.get("anchor") != policy.anchor
                    or summary.get("gain") != policy.gain
                    or any(checks.get(key) is not True for key in (
                        "all_decisions_predictable", "all_finite",
                        "every_action_matches_blend", "shared_initial_model",
                    ))
                    or not math.isfinite(float(checks.get("maximum_q_recursion_error", math.inf)))
                    or float(checks.get("maximum_q_recursion_error", math.inf)) > 1e-10
                ):
                    raise RotatedArtifactError(f"invalid pairing or action checks in {schedule} replica {index}")
            fixed_nll = float(fixed_summary["environment_nll_auc"])
            blend_nll = float(blend_summary["environment_nll_auc"])
            if not math.isfinite(fixed_nll) or not math.isfinite(blend_nll):
                raise RotatedArtifactError(f"nonfinite NLL AUC in {schedule} replica {index}")
            fixed_accuracy_auc = float(fixed_summary["environment_accuracy_auc"])
            blend_accuracy_auc = float(blend_summary["environment_accuracy_auc"])
            if not 0 <= fixed_accuracy_auc <= 1 or not 0 <= blend_accuracy_auc <= 1:
                raise RotatedArtifactError(f"invalid accuracy AUC in {schedule} replica {index}")
            for label, path, summary in (
                ("fixed", fixed, fixed_summary), ("blend", blend, blend_summary)
            ):
                curve = _curve(path, schedule, points, samples_per_step)
                if reference_exposure is None:
                    reference_exposure = curve["exposure"]
                    reference_angle = curve["angle"]
                elif curve["exposure"] != reference_exposure or curve["angle"] != reference_angle:
                    raise RotatedArtifactError(f"unaligned {schedule} trajectory: {path}")
                for metric, summary_key in (("nll", "environment_nll_auc"), ("accuracy", "environment_accuracy_auc")):
                    observed = _running_auc(curve["exposure"], curve[metric])[-1]
                    if abs(observed - float(summary[summary_key])) > 1e-8:
                        raise RotatedArtifactError(f"trajectory AUC differs from summary: {path}")
                    key = f"{label}_{metric}"
                    if key not in curve_sums:
                        curve_sums[key] = [0.0] * points
                    for position, value in enumerate(curve[metric]):
                        curve_sums[key][position] += value
            fixed_seconds = float(fixed_summary["total_wall_time_seconds"])
            blend_seconds = float(blend_summary["total_wall_time_seconds"])
            fallback = float(blend_summary["unsupported_scale_fallback_fraction"])
            if any(not math.isfinite(value) for value in (fixed_seconds, blend_seconds, fallback)):
                raise RotatedArtifactError(f"nonfinite resource diagnostic in {schedule} replica {index}")
            rows.append({
                "replica_index": index,
                "fixed_nll_auc": fixed_nll,
                "blend_nll_auc": blend_nll,
                "fixed_accuracy_auc": fixed_accuracy_auc,
                "blend_accuracy_auc": blend_accuracy_auc,
                "gain": fixed_nll - blend_nll,
                "pair_wall_seconds": fixed_seconds + blend_seconds,
                "blend_fallback_fraction": fallback,
            })
        differences = [row["gain"] for row in rows]
        result = {
            "status": "complete" if not incomplete else "incomplete",
            "planned_pairs": REPLICAS,
            "completed_pairs": len(rows),
            "incomplete_replica_indices": incomplete,
            "rows": rows,
            "descriptive_mean_gain": statistics.fmean(differences) if differences else None,
            "descriptive_median_gain": statistics.median(differences) if differences else None,
            "positive_pairs": sum(value > 0 for value in differences),
            "mean_pair_wall_seconds": statistics.fmean(row["pair_wall_seconds"] for row in rows) if rows else None,
            "maximum_blend_fallback_fraction": max((row["blend_fallback_fraction"] for row in rows), default=None),
            "mean_trajectory": _mean_trajectory(reference_exposure, reference_angle, curve_sums, len(rows)) if rows else None,
            "inference": _inference(differences) if not incomplete else None,
        }
        by_schedule[schedule] = result
    return {"contract": contract(store), "schedules": by_schedule}
