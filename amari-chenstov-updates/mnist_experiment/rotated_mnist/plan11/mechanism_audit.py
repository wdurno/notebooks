"""Artifact-only mechanistic audit for the completed Plan 11 Phase 3 study."""

from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, t

from ..artifacts import RotatedArtifactError, _read_json
from .analysis import sigmoid_policies
from .artifacts import UnitStore, file_hash
from .config import ANCHORS, GAINS, Policy, Study
from .followup import (
    DEFAULT_CONFIG as FOLLOWUP_CONFIG,
    DEFAULT_ROOT as FOLLOWUP_ROOT,
    POLICIES as FOLLOWUP_POLICIES,
    REPLICAS as FOLLOWUP_REPLICAS,
    SCHEDULES,
    contract as followup_contract,
    make_store as make_followup_store,
    planned_units as followup_units,
)
from .followup_analysis import load_progress
from .orchestrate import DEFAULT_ROOT as DEVELOPMENT_ROOT
from .run import ASSET_REQUIRED, TRAJECTORY_REQUIRED


DEVELOPMENT_CONFIG = Path("mnist_experiment/rotated_mnist/plan11/configs/development.json")
ANALYSIS_SOURCE = Path("mnist_experiment/rotated_mnist/plan11/mechanism_audit.py")
POLICY_LABELS = {0.0: "fixed", 0.025: "blend"}
PREDICTIVE_FIELDS = (
    "current_nll",
    "current_environment_accuracy",
    "current_brier",
    "current_expected_calibration_error",
    "current_nine_nll",
    "current_non_nine_nll",
    "current_nine_recall",
)
LEG_LABELS = ("first ascent", "descent", "second ascent")


def normalized_auc(x: list[int] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Return a trapezoidal AUC normalized to the supplied exposure range."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.shape != x_array.shape or len(x_array) < 2:
        raise ValueError("normalized AUC needs aligned one-dimensional arrays")
    width = float(x_array[-1] - x_array[0])
    if width <= 0 or np.any(np.diff(x_array) <= 0) or not np.all(np.isfinite(y_array)):
        raise ValueError("normalized AUC has an invalid exposure domain")
    return float(np.trapezoid(y_array, x_array) / width)


def _correlation(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    if mask.sum() < 3:
        return math.nan
    left_array = left_array[mask]
    right_array = right_array[mask]
    if np.ptp(left_array) == 0 or np.ptp(right_array) == 0:
        return math.nan
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _spearman(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    return _correlation(rankdata(left), rankdata(right))


def _sample_summary(values: list[float]) -> dict[str, Any]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("sample summary requires finite replicated values")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    sem = standard_deviation / math.sqrt(len(values))
    critical = float(t.ppf(0.975, len(values) - 1))
    p_value = 1.0 if sem == 0 and mean == 0 else (0.0 if sem == 0 else float(2 * t.sf(abs(mean / sem), len(values) - 1)))
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "standard_deviation": standard_deviation,
        "confidence_low": mean - critical * sem,
        "confidence_high": mean + critical * sem,
        "two_sided_p": p_value,
        "positive_count": sum(value > 0 for value in values),
    }


def _validate_development(repo_root: Path) -> tuple[dict[str, Any], UnitStore]:
    study = Study.from_path(repo_root / DEVELOPMENT_CONFIG)
    if study.smoke:
        raise RotatedArtifactError("Phase 3A cannot read a smoke development study")
    store = UnitStore(repo_root / DEVELOPMENT_ROOT, study, repo_root)
    expected_contract = {
        "study": study.mapping(),
        "study_hash": study.config_hash,
        "source_hashes": store.sources,
    }
    contract_path = store.root / "contracts" / "development.json"
    if _read_json(contract_path) != expected_contract:
        raise RotatedArtifactError("Plan 11 development contract differs")

    linear_policies = tuple(Policy(anchor, gain) for anchor in ANCHORS for gain in GAINS)
    linear_units = [
        store.unit("development", index, "linear", policy.mapping())
        for index in range(1, 13)
        for policy in linear_policies
    ]
    linear_ledger = {
        "study_hash": study.config_hash,
        "source_hashes": store.sources,
        "units": linear_units,
    }
    if _read_json(store.root / "ledgers" / "development_linear.json") != linear_ledger:
        raise RotatedArtifactError("Plan 11 linear development ledger differs")

    selection_path = store.root / "decisions" / "linear_selection.json"
    selection = _read_json(selection_path)
    if selection.get("study_hash") != study.config_hash:
        raise RotatedArtifactError("Plan 11 linear selection study differs")
    sigmoid_policy_values = sigmoid_policies(selection)
    sigmoid_units = [
        store.unit("development", index, "sigmoid", policy.mapping())
        for index in range(1, 13)
        for policy in sigmoid_policy_values
    ]
    sigmoid_ledger = {
        "study_hash": study.config_hash,
        "source_hashes": store.sources,
        "units": sigmoid_units,
    }
    if _read_json(store.root / "ledgers" / "development_sigmoid.json") != sigmoid_ledger:
        raise RotatedArtifactError("Plan 11 sigmoid development ledger differs")

    decision_path = store.root / "decisions" / "development_decision.json"
    decision = _read_json(decision_path)
    if decision.get("study_hash") != study.config_hash or decision.get("status") != "stopped":
        raise RotatedArtifactError("Plan 11 development decision differs")

    for unit in (*linear_units, *sigmoid_units):
        if store.completed(unit, TRAJECTORY_REQUIRED) is None:
            raise RotatedArtifactError("Plan 11 development trajectory is incomplete")
    for index in range(1, 13):
        asset = store.unit("development", index, None, None)
        if store.completed(asset, ASSET_REQUIRED) is None:
            raise RotatedArtifactError("Plan 11 development asset is incomplete")

    return (
        {
            "study_hash": study.config_hash,
            "validated_trajectory_count": len(linear_units) + len(sigmoid_units),
            "validated_asset_count": 12,
            "linear_ledger_sha256": file_hash(store.root / "ledgers" / "development_linear.json"),
            "sigmoid_ledger_sha256": file_hash(store.root / "ledgers" / "development_sigmoid.json"),
            "selection_sha256": file_hash(selection_path),
            "decision_sha256": file_hash(decision_path),
            "decision": decision,
        },
        store,
    )


def _followup_context(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], UnitStore, dict[tuple[int, str, float], dict[str, Any]]]:
    progress = load_progress(repo_root)
    study = Study.from_path(repo_root / FOLLOWUP_CONFIG)
    store = make_followup_store(repo_root / FOLLOWUP_ROOT, study, repo_root)
    contract_path = store.root / "contract.json"
    ledger_path = store.root / "ledger.json"
    if _read_json(contract_path) != followup_contract(store):
        raise RotatedArtifactError("Plan 11 follow-up contract changed after validation")
    units = followup_units(store)
    if _read_json(ledger_path) != {"units": units}:
        raise RotatedArtifactError("Plan 11 follow-up ledger changed after validation")
    lookup = {
        (unit["replica_index"], unit["schedule"], unit["policy"]["gain"]): unit
        for unit in units
    }
    return (
        {
            "study_hash": study.config_hash,
            "validated_trajectory_count": len(units),
            "validated_asset_count": FOLLOWUP_REPLICAS,
            "contract_sha256": file_hash(contract_path),
            "ledger_sha256": file_hash(ledger_path),
        },
        progress,
        store,
        lookup,
    )


def _extract(row: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = row
    for key in path:
        if value is None or not isinstance(value, dict) or key not in value:
            return math.nan
        value = value[key]
    if value is None:
        return math.nan
    result = float(value)
    return result if math.isfinite(result) else math.nan


MECHANIC_PATHS: dict[str, tuple[str, ...]] = {
    "applied_pi": ("controller", "applied_pi"),
    "recommendation_pi": ("controller", "recommendation_pi"),
    "movement_premium": ("controller", "discounted_movement_premium"),
    "q": ("controller_acceptance", "state_after", "q"),
    "ewc_odds": ("controller", "ewc_odds"),
    "displacement_norm": ("proposal", "displacement_norm"),
    "fisher_displacement_norm": ("proposal", "fisher_weighted_displacement_norm"),
    "ewc_penalty": ("proposal", "ewc_penalty_after"),
    "objective_decrease": ("proposal", "objective_decrease"),
    "optimizer_evaluations": ("proposal", "optimizer_function_evaluations"),
    "previous_fisher_trace": ("fisher_update", "previous_trace"),
    "candidate_fisher_trace": ("fisher_update", "candidate_trace"),
    "fresh_fisher_trace": ("fisher_update", "fresh_trace"),
}


def _mean_from_sums(sums: np.ndarray, counts: np.ndarray) -> list[float]:
    result = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result.tolist()


def _add_finite(sums: np.ndarray, counts: np.ndarray, values: np.ndarray) -> None:
    mask = np.isfinite(values)
    sums[mask] += values[mask]
    counts[mask] += 1


def _policy_metrics(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RotatedArtifactError(f"nonfinite Phase 3 field: {field}")
    return values


def _leg_auc(exposure: np.ndarray, values: np.ndarray, leg: int) -> float:
    start = 40 * leg
    stop = start + 41
    return normalized_auc(exposure[start:stop], values[start:stop])


def _lag_correlations(action: np.ndarray, outcome: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for lag in range(0, 21):
        action_slice = action[8 : len(action) - lag]
        outcome_slice = outcome[9 + lag :]
        if len(action_slice) != len(outcome_slice):
            raise RuntimeError("lagged Plan 11 arrays do not align")
        rows.append({"lag_updates": lag, "correlation": _correlation(action_slice, outcome_slice)})
    return rows


def _schedule_audit(
    schedule: str,
    progress: dict[str, Any],
    store: UnitStore,
    lookup: dict[tuple[int, str, float], dict[str, Any]],
) -> dict[str, Any]:
    points = 121
    transitions = 120
    predictive_sums = {
        label: {field: np.zeros(points) for field in PREDICTIVE_FIELDS}
        for label in POLICY_LABELS.values()
    }
    class_recall_sums = {label: np.zeros((points, 10)) for label in POLICY_LABELS.values()}
    confusion_sums = {label: np.zeros((10, 10)) for label in POLICY_LABELS.values()}
    mechanic_sums = {
        label: {name: np.zeros(transitions) for name in MECHANIC_PATHS}
        for label in POLICY_LABELS.values()
    }
    mechanic_counts = {
        label: {name: np.zeros(transitions, dtype=int) for name in MECHANIC_PATHS}
        for label in POLICY_LABELS.values()
    }
    hash_matches = np.zeros(points, dtype=int)
    same_action_pairs = 0
    lanczos_seed_matches = 0
    lanczos_seed_comparisons = 0
    replica_rows: list[dict[str, Any]] = []
    class_recall_gains: list[list[float]] = []
    exposure: np.ndarray | None = None
    angle: np.ndarray | None = None

    for index in range(1, FOLLOWUP_REPLICAS + 1):
        pair_rows: dict[str, list[dict[str, Any]]] = {}
        pair_arrays: dict[str, dict[str, np.ndarray]] = {}
        paths: dict[str, Path] = {}
        for gain, label in POLICY_LABELS.items():
            path = store.paths(lookup[index, schedule, gain])[0]
            rows = _read_json(path / "metrics.json")
            if not isinstance(rows, list) or len(rows) != points:
                raise RotatedArtifactError(f"invalid Phase 3 metric rows: {path}")
            this_exposure = np.asarray([row["observations_before_evaluation"] for row in rows], dtype=int)
            this_angle = np.asarray([row["angle_degrees"] for row in rows], dtype=float)
            if exposure is None:
                exposure, angle = this_exposure, this_angle
            elif not np.array_equal(exposure, this_exposure) or not np.array_equal(angle, this_angle):
                raise RotatedArtifactError(f"unaligned Phase 3 mechanism rows: {path}")
            pair_rows[label] = rows
            paths[label] = path
            pair_arrays[label] = {}
            for field in PREDICTIVE_FIELDS:
                values = _policy_metrics(rows, field)
                pair_arrays[label][field] = values
                predictive_sums[label][field] += values
            recall = np.asarray([row["current_per_class_recall"] for row in rows], dtype=float)
            confusion = np.asarray([row["current_confusion_matrix"] for row in rows], dtype=float)
            if recall.shape != (points, 10) or confusion.shape != (points, 10, 10):
                raise RotatedArtifactError(f"invalid class diagnostics: {path}")
            if not np.all(np.isfinite(recall)) or not np.all(np.isfinite(confusion)):
                raise RotatedArtifactError(f"nonfinite class diagnostics: {path}")
            class_recall_sums[label] += recall
            confusion_sums[label] += confusion.sum(axis=0)
            pair_arrays[label]["class_recall"] = recall
            transition_rows = rows[:-1]
            for name, key_path in MECHANIC_PATHS.items():
                values = np.asarray([_extract(row, key_path) for row in transition_rows])
                pair_arrays[label][name] = values
                _add_finite(mechanic_sums[label][name], mechanic_counts[label][name], values)

        fixed_rows, blend_rows = pair_rows["fixed"], pair_rows["blend"]
        hash_matches += np.asarray(
            [fixed["parameter_hash"] == blend["parameter_hash"] for fixed, blend in zip(fixed_rows, blend_rows)],
            dtype=int,
        )
        fixed_actions = pair_arrays["fixed"]["applied_pi"]
        blend_actions = pair_arrays["blend"]["applied_pi"]
        if np.array_equal(fixed_actions[:8], blend_actions[:8]):
            same_action_pairs += 1
        else:
            raise RotatedArtifactError(f"cold-start actions differ in {schedule} replica {index}")
        for fixed, blend in zip(fixed_rows[:-1], blend_rows[:-1]):
            fixed_lanczos = (fixed.get("fisher_update") or {}).get("lanczos")
            blend_lanczos = (blend.get("fisher_update") or {}).get("lanczos")
            if fixed_lanczos is None and blend_lanczos is None:
                continue
            if fixed_lanczos is None or blend_lanczos is None:
                raise RotatedArtifactError(f"unpaired Lanczos update in {schedule} replica {index}")
            lanczos_seed_comparisons += 1
            lanczos_seed_matches += fixed_lanczos["seed"] == blend_lanczos["seed"]

        if exposure is None:
            raise RuntimeError("Phase 3 exposure was not initialized")
        fixed = pair_arrays["fixed"]
        blend = pair_arrays["blend"]
        nll_gain = normalized_auc(exposure, fixed["current_nll"]) - normalized_auc(exposure, blend["current_nll"])
        accuracy_gain = normalized_auc(exposure, blend["current_environment_accuracy"]) - normalized_auc(exposure, fixed["current_environment_accuracy"])
        brier_gain = normalized_auc(exposure, fixed["current_brier"]) - normalized_auc(exposure, blend["current_brier"])
        calibration_gain = normalized_auc(exposure, fixed["current_expected_calibration_error"]) - normalized_auc(exposure, blend["current_expected_calibration_error"])
        nine_nll_gain = normalized_auc(exposure, fixed["current_nine_nll"]) - normalized_auc(exposure, blend["current_nine_nll"])
        non_nine_nll_gain = normalized_auc(exposure, fixed["current_non_nine_nll"]) - normalized_auc(exposure, blend["current_non_nine_nll"])
        nine_recall_gain = normalized_auc(exposure, blend["current_nine_recall"]) - normalized_auc(exposure, fixed["current_nine_recall"])
        recall_gain = [
            normalized_auc(exposure, blend["class_recall"][:, label])
            - normalized_auc(exposure, fixed["class_recall"][:, label])
            for label in range(10)
        ]
        class_recall_gains.append(recall_gain)
        action_excess = blend["applied_pi"] - fixed["applied_pi"]
        recommendation_excess = blend["recommendation_pi"] - fixed["recommendation_pi"]
        movement_excess = blend["movement_premium"] - fixed["movement_premium"]
        q_excess = blend["q"] - fixed["q"]
        displacement_excess = blend["displacement_norm"] - fixed["displacement_norm"]
        ewc_odds_excess = blend["ewc_odds"] - fixed["ewc_odds"]
        cold_exposure = exposure[:9]
        cold_nll_gain = normalized_auc(cold_exposure, fixed["current_nll"][:9]) - normalized_auc(cold_exposure, blend["current_nll"][:9])
        post_nll_gain = normalized_auc(exposure[8:], fixed["current_nll"][8:]) - normalized_auc(exposure[8:], blend["current_nll"][8:])
        replica_rows.append(
            {
                "replica_index": index,
                "nll_gain": nll_gain,
                "accuracy_gain": accuracy_gain,
                "brier_gain": brier_gain,
                "calibration_gain": calibration_gain,
                "nine_nll_gain": nine_nll_gain,
                "non_nine_nll_gain": non_nine_nll_gain,
                "nine_recall_gain": nine_recall_gain,
                "mean_action_excess": float(np.mean(action_excess[8:])),
                "mean_recommendation_excess": float(np.mean(recommendation_excess[8:])),
                "mean_movement_excess": float(np.mean(movement_excess[8:])),
                "mean_q_excess": float(np.mean(q_excess[8:])),
                "mean_displacement_excess": float(np.mean(displacement_excess[8:])),
                "mean_ewc_odds_excess": float(np.mean(ewc_odds_excess[8:])),
                "cold_start_nll_gain": cold_nll_gain,
                "release_nll_gain": float(fixed["current_nll"][8] - blend["current_nll"][8]),
                "post_cold_nll_gain": post_nll_gain,
                "leg_nll_gains": [
                    _leg_auc(exposure, fixed["current_nll"], leg)
                    - _leg_auc(exposure, blend["current_nll"], leg)
                    for leg in range(3)
                ],
                "leg_accuracy_gains": [
                    _leg_auc(exposure, blend["current_environment_accuracy"], leg)
                    - _leg_auc(exposure, fixed["current_environment_accuracy"], leg)
                    for leg in range(3)
                ],
            }
        )

    if exposure is None or angle is None:
        raise RuntimeError("Phase 3 schedule produced no replicas")
    predictive_means = {
        label: {field: (values / FOLLOWUP_REPLICAS).tolist() for field, values in fields.items()}
        for label, fields in predictive_sums.items()
    }
    mechanic_means = {
        label: {
            name: _mean_from_sums(mechanic_sums[label][name], mechanic_counts[label][name])
            for name in MECHANIC_PATHS
        }
        for label in POLICY_LABELS.values()
    }
    nll_curve_gain = np.asarray(predictive_means["fixed"]["current_nll"]) - np.asarray(predictive_means["blend"]["current_nll"])
    action_curve_excess = np.asarray(mechanic_means["blend"]["applied_pi"]) - np.asarray(mechanic_means["fixed"]["applied_pi"])
    class_array = np.asarray(class_recall_gains)
    normalized_confusions = {}
    for label, matrix in confusion_sums.items():
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized_confusions[label] = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)
    terminal = [row["nll_gain"] for row in replica_rows]
    action = [row["mean_action_excess"] for row in replica_rows]
    cold = [row["cold_start_nll_gain"] for row in replica_rows]
    post = [row["post_cold_nll_gain"] for row in replica_rows]
    first_divergence = next((step for step, count in enumerate(hash_matches) if count < FOLLOWUP_REPLICAS), None)
    return {
        "status": progress["status"],
        "inference": progress["inference"],
        "exposure": exposure.tolist(),
        "angle": angle.tolist(),
        "predictive_means": predictive_means,
        "mechanic_means": mechanic_means,
        "replica_rows": replica_rows,
        "primary": _sample_summary(terminal),
        "secondary": {
            "accuracy_gain": _sample_summary([row["accuracy_gain"] for row in replica_rows]),
            "brier_gain": _sample_summary([row["brier_gain"] for row in replica_rows]),
            "calibration_gain": _sample_summary([row["calibration_gain"] for row in replica_rows]),
            "nine_nll_gain": _sample_summary([row["nine_nll_gain"] for row in replica_rows]),
            "non_nine_nll_gain": _sample_summary([row["non_nine_nll_gain"] for row in replica_rows]),
            "nine_recall_gain": _sample_summary([row["nine_recall_gain"] for row in replica_rows]),
        },
        "legs": [
            {
                "leg": label,
                "nll_gain": _sample_summary([row["leg_nll_gains"][leg] for row in replica_rows]),
                "accuracy_gain": _sample_summary([row["leg_accuracy_gains"][leg] for row in replica_rows]),
            }
            for leg, label in enumerate(LEG_LABELS)
        ],
        "class_recall": [
            {"class": label, **_sample_summary(class_array[:, label].tolist())}
            for label in range(10)
        ],
        "confusion_difference": (normalized_confusions["blend"] - normalized_confusions["fixed"]).tolist(),
        "heterogeneity": {
            "action_vs_terminal_pearson": _correlation(action, terminal),
            "action_vs_terminal_spearman": _spearman(action, terminal),
            "cold_vs_post_pearson": _correlation(cold, post),
            "cold_vs_post_spearman": _spearman(cold, post),
            "release_gain_standard_deviation": statistics.stdev(row["release_nll_gain"] for row in replica_rows),
            "cold_auc_gain_standard_deviation": statistics.stdev(cold),
            "post_cold_gain_standard_deviation": statistics.stdev(post),
        },
        "lag_correlations": _lag_correlations(action_curve_excess, nll_curve_gain),
        "numerical_pairing": {
            "same_action_cold_start_pairs": same_action_pairs,
            "planned_pairs": FOLLOWUP_REPLICAS,
            "lanczos_seed_matches": lanczos_seed_matches,
            "lanczos_seed_comparisons": lanczos_seed_comparisons,
            "parameter_hash_match_fraction": (hash_matches / FOLLOWUP_REPLICAS).tolist(),
            "first_parameter_divergence_step": first_divergence,
            "release_nll_gain": _sample_summary([row["release_nll_gain"] for row in replica_rows]),
            "cold_start_nll_auc_gain": _sample_summary(cold),
        },
    }


def _development_reversal(store: UnitStore, progress: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for schedule in SCHEDULES:
        development_gains = []
        for index in range(1, 13):
            fixed = store.paths(store.unit("development", index, schedule, Policy(0.025, 0.0).mapping()))[0]
            blend = store.paths(store.unit("development", index, schedule, Policy(0.025, 0.025).mapping()))[0]
            fixed_summary = _read_json(fixed / "summary.json")
            blend_summary = _read_json(blend / "summary.json")
            development_gains.append(float(fixed_summary["environment_nll_auc"]) - float(blend_summary["environment_nll_auc"]))
        confirmation_gains = [row["gain"] for row in progress["schedules"][schedule]["rows"]]
        result.append(
            {
                "schedule": schedule,
                "development": _sample_summary(development_gains),
                "confirmation": _sample_summary(confirmation_gains),
                "mean_change": statistics.fmean(confirmation_gains) - statistics.fmean(development_gains),
            }
        )
    return result


def load_mechanism_audit(repo_root: Path) -> dict[str, Any]:
    """Validate immutable Plan 11 evidence and calculate exploratory summaries."""

    development_provenance, development_store = _validate_development(repo_root)
    followup_provenance, progress, followup_store, lookup = _followup_context(repo_root)
    schedules = {
        schedule: _schedule_audit(
            schedule,
            progress["schedules"][schedule],
            followup_store,
            lookup,
        )
        for schedule in SCHEDULES
    }
    return {
        "analysis_contract": {
            "phase": "plan11_phase3a_mechanism_audit_v1",
            "analysis_source": str(ANALYSIS_SOURCE),
            "analysis_source_sha256": hashlib.sha256((repo_root / ANALYSIS_SOURCE).read_bytes()).hexdigest(),
            "exploratory": True,
            "changes_phase3_inference": False,
            "trains_or_mutates": False,
        },
        "provenance": {
            "development": development_provenance,
            "phase3": followup_provenance,
        },
        "schedules": schedules,
        "development_to_confirmation": _development_reversal(development_store, progress),
    }
