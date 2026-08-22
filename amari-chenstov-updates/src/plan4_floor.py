"""Exploratory Plan 4 sensitivity analysis for a lower controller floor."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import load_config
from .initialization import replica_bundle_id, replica_design_hash
from .plan4_analysis import PI_SIGNAL_THRESHOLD, _event_mask
from .plan4_challenge import (
    PLAN4_CHALLENGE_BUNDLE_SCHEMA_VERSION,
    SCHEDULES,
    Plan4ChallengeError,
    _challenge_config,
    _read_json,
    _sha256,
    load_screen_bundle,
    status_rows,
)
from .schedules import resolve_schedule


PLAN4_FLOOR_ANALYSIS_SCHEMA_VERSION = 1
FLOOR = 0.025
FLOOR_CONDITIONS = (
    "fixed-pi005",
    "fixed-pi0025",
    "adaptive-fisher-pimin0025-h005",
)
NEW_FLOOR_CONDITIONS = FLOOR_CONDITIONS[1:]
PREDICTIVE_FIELDS = (
    "environment_accuracy",
    "nine_ovr_accuracy",
    "nine_precision",
    "nine_recall",
    "non_nine_accuracy",
    "nll",
    "expected_calibration_error",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def build_floor_bundle(
    repo_root: str | Path,
    baseline_bundle_path: str | Path,
    *,
    device: str = "cuda",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Pair two lower-floor treatments with the completed fixed-.05 runs."""

    root = Path(repo_root)
    baseline_bundle, baseline_manifest = load_screen_bundle(baseline_bundle_path)
    if baseline_manifest.get("stage") != "realized-actuation-screen":
        raise Plan4ChallengeError("floor sensitivity requires the actuation screen")
    baseline_status = status_rows(baseline_bundle, root)
    if any(row["run_state"] != "completed" for row in baseline_status):
        raise Plan4ChallengeError("the fixed-.05 baseline runs must be complete")

    configs: dict[str, dict[str, Any]] = {}
    entries = []
    for schedule in SCHEDULES:
        baseline = [
            row
            for row in baseline_status
            if row["schedule"] == schedule and row["condition"] == "fixed-pi005"
        ]
        if len(baseline) != 1:
            raise Plan4ChallengeError(f"missing fixed-.05 baseline for {schedule}")
        baseline_row = baseline[0]
        baseline_mapping = _read_json(Path(baseline_row["config_path"]))
        relative = f"configs/{schedule}/fixed-pi005.json"
        configs[relative] = baseline_mapping
        entries.append(
            {
                **{
                    key: baseline_row[key]
                    for key in (
                        "schedule",
                        "condition",
                        "risk_metric",
                        "config_hash",
                        "run_id",
                        "cache_root",
                        "replica_bundle_id",
                        "replica_design_hash",
                        "source_archive",
                        "source_replica_bundle",
                    )
                },
                "config_file": relative,
                "reused_completed_baseline": True,
            }
        )
        for condition in NEW_FLOOR_CONDITIONS:
            config = _challenge_config(
                baseline_mapping,
                source_archive=baseline_row["source_archive"],
                schedule=schedule,
                condition=condition,
                device=device,
            )
            if replica_design_hash(config) != baseline_row["replica_design_hash"]:
                raise Plan4ChallengeError("floor treatment broke schedule pairing")
            relative = f"configs/{schedule}/{condition}.json"
            configs[relative] = config.to_mapping()
            entries.append(
                {
                    "schedule": schedule,
                    "condition": condition,
                    "risk_metric": config.controller.risk_metric,
                    "config_file": relative,
                    "config_hash": config.config_hash,
                    "run_id": config.run_id,
                    "cache_root": config.cache_root,
                    "replica_bundle_id": replica_bundle_id(config),
                    "replica_design_hash": replica_design_hash(config),
                    "source_archive": baseline_row["source_archive"],
                    "source_replica_bundle": baseline_row["source_replica_bundle"],
                    "reused_completed_baseline": False,
                }
            )
    identity = {
        "schema_version": PLAN4_CHALLENGE_BUNDLE_SCHEMA_VERSION,
        "floor_analysis_schema_version": PLAN4_FLOOR_ANALYSIS_SCHEMA_VERSION,
        "builder_code_sha256": _sha256(Path(__file__)),
        "challenge_code_sha256": _sha256(Path(__file__).with_name("plan4_challenge.py")),
        "phase": 5,
        "stage": "exploratory-floor-sensitivity",
        "device": device,
        "pi_min": FLOOR,
        "schedules": list(SCHEDULES),
        "conditions": list(FLOOR_CONDITIONS),
        "baseline_bundle": str(baseline_bundle),
        "baseline_manifest_sha256": _sha256(baseline_bundle / "manifest.json"),
        "config_hashes": [entry["config_hash"] for entry in entries],
        "confirmatory": False,
    }
    bundle_id = f"plan4-floor-sensitivity__{_hash(identity)[:12]}"
    return (
        {
            **identity,
            "bundle_id": bundle_id,
            "entry_count": len(entries),
            "new_run_count": len(SCHEDULES) * len(NEW_FLOOR_CONDITIONS),
            "estimated_seconds": 100.0 * len(SCHEDULES) * len(NEW_FLOOR_CONDITIONS),
            "entries": entries,
        },
        configs,
    )


def write_floor_bundle(
    repo_root: str | Path,
    baseline_bundle_path: str | Path,
    *,
    device: str = "cuda",
) -> Path:
    root = Path(repo_root)
    manifest, configs = build_floor_bundle(root, baseline_bundle_path, device=device)
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "challenge"
        / "bundles"
        / manifest["bundle_id"]
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4ChallengeError(f"incomplete floor bundle: {destination}")
        if _read_json(destination / "manifest.json") != manifest:
            raise Plan4ChallengeError("completed floor bundle differs")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=incomplete))
    try:
        for relative, mapping in configs.items():
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(mapping, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _finite_array(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    return np.asarray(
        [
            math.nan if row["classification"].get(field) is None else float(row["classification"][field])
            for row in rows
        ],
        dtype=np.float64,
    )


def _condition_predictive_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for field in PREDICTIVE_FIELDS:
        values = _finite_array(rows, field)
        finite = np.isfinite(values)
        result[field] = {
            "finite_fraction": float(finite.mean()),
            "finite_step_mean": float(values[finite].mean()) if finite.any() else None,
            "endpoint": float(values[-1]) if math.isfinite(float(values[-1])) else None,
        }
    return result


def _paired_predictive_summary(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    event: np.ndarray,
) -> dict[str, Any]:
    """Return left-minus-right differences on common finite evaluations."""

    if len(left_rows) != len(right_rows) or len(left_rows) != event.size + 1:
        raise Plan4ChallengeError("paired predictive trajectories do not align")
    event_outcomes = np.flatnonzero(event) + 1
    result = {}
    for field in PREDICTIVE_FIELDS:
        left = _finite_array(left_rows, field)
        right = _finite_array(right_rows, field)
        common = np.isfinite(left) & np.isfinite(right)
        event_common = common[event_outcomes]
        event_difference = left[event_outcomes] - right[event_outcomes]
        result[field] = {
            "common_step_count": int(common.sum()),
            "full_mean_difference": (
                float((left[common] - right[common]).mean()) if common.any() else None
            ),
            "event_outcome_count": int(event_common.sum()),
            "event_mean_difference": (
                float(event_difference[event_common].mean())
                if event_common.any()
                else None
            ),
            "endpoint_difference": (
                float(left[-1] - right[-1]) if common[-1] else None
            ),
        }
    return result


def _floor_controller_summary(
    rows: Sequence[Mapping[str, Any]],
    event: np.ndarray,
) -> dict[str, Any]:
    updates = [row for row in rows if isinstance(row.get("controller_decision"), Mapping)]
    if len(updates) != event.size:
        raise Plan4ChallengeError("controller trace does not align with event")
    decisions = [row["controller_decision"] for row in updates]
    applied = np.asarray([float(row["applied_pi"]) for row in decisions])
    signal = np.asarray([float(row["signal_energy"]) for row in decisions])
    old = np.asarray([float(row["old_covariance_risk"]) for row in decisions])
    new = np.asarray([float(row["new_covariance_risk"]) for row in decisions])
    risk = (1.0 - applied) ** 2 * (signal + old) + applied**2 * new
    fixed = (1.0 - FLOOR) ** 2 * (signal + old) + FLOOR**2 * new
    event_applied = applied[event]
    event_risk = float(risk[event].sum())
    event_fixed = float(fixed[event].sum())
    correlation = None
    if float(np.ptp(signal[event])) > 0.0 and float(np.ptp(event_applied)) > 0.0:
        candidate = float(np.corrcoef(signal[event], event_applied)[0, 1])
        correlation = candidate if math.isfinite(candidate) else None
    return {
        "event_transition_count": int(event.sum()),
        "event_steps_above_007": int(np.sum(event_applied > PI_SIGNAL_THRESHOLD)),
        "event_applied_pi_range": float(np.ptp(event_applied)),
        "event_signal_action_correlation": correlation,
        "event_relative_risk_reduction_vs_fixed_0025": (
            (event_fixed - event_risk) / max(event_fixed, 1e-15)
        ),
        "full_applied_pi_min": float(applied.min()),
        "full_applied_pi_mean": float(applied.mean()),
        "full_applied_pi_max": float(applied.max()),
        "lower_bound_fraction": float(np.mean(applied <= FLOOR + 1e-12)),
        "trajectory": [
            {
                "step": int(row["step"]),
                "p": float(row["p"]),
                "in_event_window": bool(event[index]),
                "applied_pi": float(decision["applied_pi"]),
                "plugin_pi": float(decision["plugin_pi"]),
                "signal_energy": float(decision["signal_energy"]),
            }
            for index, (row, decision) in enumerate(zip(updates, decisions, strict=True))
        ],
    }


def build_floor_analysis(
    bundle_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    bundle, manifest = load_screen_bundle(bundle_path)
    if manifest.get("stage") != "exploratory-floor-sensitivity":
        raise Plan4ChallengeError("not a floor-sensitivity bundle")
    status = status_rows(bundle, repo_root)
    if any(row["run_state"] != "completed" for row in status):
        raise Plan4ChallengeError("all floor-sensitivity runs must complete")
    schedules = {}
    initial_hashes = set()
    uniform_hashes = set()
    provenance = []
    for schedule in SCHEDULES:
        members = [row for row in status if row["schedule"] == schedule]
        if {row["condition"] for row in members} != set(FLOOR_CONDITIONS):
            raise Plan4ChallengeError(f"floor conditions differ for {schedule}")
        loaded = {}
        for member in members:
            run_path = Path(member["run_path"])
            metric_path = run_path / "plan3_hybrid_metrics.json"
            schedule_path = run_path / "schedule_trajectory.json"
            metrics = _read_json(metric_path)
            schedule_artifact = _read_json(schedule_path)
            loaded[member["condition"]] = metrics
            initial_hashes.add(metrics["pairing"]["initial_parameter_hash"])
            uniform_hashes.add(schedule_artifact["uniform_stream_hash"])
            provenance.append(
                {
                    "run_id": member["run_id"],
                    "schedule": schedule,
                    "condition": member["condition"],
                    "metrics_sha256": _sha256(metric_path),
                    "schedule_sha256": _sha256(schedule_path),
                }
            )
        resolved = resolve_schedule(load_config([row for row in members if row["condition"] == "fixed-pi0025"][0]["config_path"]).data)
        event = _event_mask(resolved)
        rows = {name: value["condition_steps"] for name, value in loaded.items()}
        adaptive = rows["adaptive-fisher-pimin0025-h005"]
        conditions = {
            name: {
                "predictive": _condition_predictive_summary(condition_rows),
                "resources": loaded[name]["resource_ledger"],
            }
            for name, condition_rows in rows.items()
        }
        conditions["adaptive-fisher-pimin0025-h005"]["controller"] = (
            _floor_controller_summary(adaptive, event)
        )
        schedules[schedule] = {
            "conditions": conditions,
            "adaptive_minus_fixed_0025": _paired_predictive_summary(
                adaptive, rows["fixed-pi0025"], event
            ),
            "adaptive_minus_fixed_005": _paired_predictive_summary(
                adaptive, rows["fixed-pi005"], event
            ),
            "fixed_0025_minus_fixed_005": _paired_predictive_summary(
                rows["fixed-pi0025"], rows["fixed-pi005"], event
            ),
        }
    if len(initial_hashes) != 1 or len(uniform_hashes) != 1:
        raise Plan4ChallengeError("floor sensitivity lost pairing")
    return {
        "schema_version": PLAN4_FLOOR_ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "plan4_exploratory_floor_sensitivity",
        "confirmatory": False,
        "pi_min": FLOOR,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": _sha256(bundle / "manifest.json"),
        "common_initial_parameter_hash": next(iter(initial_hashes)),
        "common_uniform_stream_hash": next(iter(uniform_hashes)),
        "input_provenance": provenance,
        "schedules": schedules,
    }


def write_floor_analysis(bundle_path: str | Path, repo_root: str | Path) -> Path:
    root = Path(repo_root)
    analysis = build_floor_analysis(bundle_path, root)
    identity = {
        "schema_version": analysis["schema_version"],
        "analysis_kind": analysis["analysis_kind"],
        "bundle_id": analysis["bundle_id"],
        "bundle_manifest_sha256": analysis["bundle_manifest_sha256"],
        "input_provenance": analysis["input_provenance"],
    }
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "challenge"
        / "floor_analysis"
        / f"floor_sensitivity__{_hash(identity)[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4ChallengeError(f"incomplete floor analysis: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
