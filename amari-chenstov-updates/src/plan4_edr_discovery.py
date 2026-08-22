"""Cold-start discovery amendment for Plan 4 EDR control."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import ExperimentConfig, load_config
from .initialization import replica_bundle_id, replica_design_hash
from .plan4_analysis import _event_mask
from .plan4_challenge import (
    Plan4ChallengeError,
    _read_json,
    _sha256,
    load_screen_bundle,
    status_rows,
)
from .plan4_edr import (
    EDR_CONDITION,
    _edr_config,
    _edr_controller_summary,
    _predictive_trajectory,
)
from .plan4_floor import (
    PREDICTIVE_FIELDS,
    _condition_predictive_summary,
    _finite_array,
    _paired_predictive_summary,
)
from .schedules import resolve_schedule


PLAN4_EDR_DISCOVERY_BUNDLE_SCHEMA_VERSION = 1
PLAN4_EDR_DISCOVERY_ANALYSIS_SCHEMA_VERSION = 1
DISCOVERY_CONDITION = "edr-fisher-pimin001-h04-cold005"
DISCOVERY_CONDITIONS = (
    "fixed-pi005",
    "fixed-pi0025",
    EDR_CONDITION,
    DISCOVERY_CONDITION,
)
PRIMARY_COMPARATOR = "fixed-pi005"
SECONDARY_TOLERANCE = 0.02


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _discovery_config(source: Mapping[str, Any]) -> ExperimentConfig:
    return _edr_config(
        source,
        schedule="linear",
        cold_start_pi=0.05,
        experiment_suffix="edr-discovery",
    )


def build_discovery_bundle(
    edr_bundle_path: str | Path,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    source_bundle, source_manifest = load_screen_bundle(edr_bundle_path)
    if source_manifest.get("stage") != "edr-predictive-development":
        raise Plan4ChallengeError("discovery amendment requires the Phase 6 EDR bundle")
    status = status_rows(source_bundle, root)
    if any(row["run_state"] != "completed" for row in status):
        raise Plan4ChallengeError("discovery amendment inputs must be complete")
    linear = [row for row in status if row["schedule"] == "linear"]
    by_condition = {row["condition"]: row for row in linear}
    required = set(DISCOVERY_CONDITIONS[:-1])
    if not required.issubset(by_condition):
        raise Plan4ChallengeError("linear discovery controls are incomplete")

    entries = []
    configs: dict[str, dict[str, Any]] = {}
    for condition in DISCOVERY_CONDITIONS[:-1]:
        member = by_condition[condition]
        relative = f"configs/linear/{condition}.json"
        configs[relative] = _read_json(Path(member["config_path"]))
        entries.append(
            {
                **{
                    key: member[key]
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
                "reused_completed_control": True,
            }
        )

    source_edr = by_condition[EDR_CONDITION]
    discovery = _discovery_config(_read_json(Path(source_edr["config_path"])))
    if (
        replica_design_hash(discovery) != source_edr["replica_design_hash"]
        or replica_bundle_id(discovery) != source_edr["replica_bundle_id"]
    ):
        raise Plan4ChallengeError("cold-start amendment broke linear pairing")
    relative = f"configs/linear/{DISCOVERY_CONDITION}.json"
    configs[relative] = discovery.to_mapping()
    entries.append(
        {
            "schedule": "linear",
            "condition": DISCOVERY_CONDITION,
            "risk_metric": discovery.controller.risk_metric,
            "config_file": relative,
            "config_hash": discovery.config_hash,
            "run_id": discovery.run_id,
            "cache_root": discovery.cache_root,
            "replica_bundle_id": replica_bundle_id(discovery),
            "replica_design_hash": replica_design_hash(discovery),
            "source_archive": source_edr["source_archive"],
            "source_replica_bundle": source_edr["source_replica_bundle"],
            "reused_completed_control": False,
        }
    )

    identity = {
        "schema_version": PLAN4_EDR_DISCOVERY_BUNDLE_SCHEMA_VERSION,
        "phase": 6,
        "stage": "edr-cold-start-discovery",
        "source_bundle_id": source_manifest["bundle_id"],
        "source_manifest_sha256": _sha256(source_bundle / "manifest.json"),
        "schedule": "linear",
        "conditions": list(DISCOVERY_CONDITIONS),
        "primary_comparator": PRIMARY_COMPARATOR,
        "cold_start_pi": 0.05,
        "pi_min": 0.01,
        "action_half_life_steps": 4.0,
        "config_hashes": [entry["config_hash"] for entry in entries],
        "new_run_count": 1,
        "confirmatory": False,
    }
    return (
        {
            **identity,
            "bundle_id": f"plan4-edr-discovery__{_hash(identity)[:12]}",
            "entry_count": len(entries),
            "estimated_seconds": 67.0,
            "entries": entries,
        },
        configs,
    )


def write_discovery_bundle(
    edr_bundle_path: str | Path,
    repo_root: str | Path,
) -> Path:
    root = Path(repo_root)
    manifest, configs = build_discovery_bundle(edr_bundle_path, root)
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "edr"
        / "discovery"
        / "bundles"
        / manifest["bundle_id"]
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4ChallengeError(f"incomplete discovery bundle: {destination}")
        if _read_json(destination / "manifest.json") != manifest:
            raise Plan4ChallengeError("completed discovery bundle differs")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent))
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


def _paired_outcome_summary(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    selected: np.ndarray,
) -> dict[str, Any]:
    if len(left_rows) != len(right_rows) or selected.shape != (len(left_rows),):
        raise Plan4ChallengeError("discovery outcome windows do not align")
    result = {}
    for field in PREDICTIVE_FIELDS:
        left = _finite_array(left_rows, field)
        right = _finite_array(right_rows, field)
        common = selected & np.isfinite(left) & np.isfinite(right)
        result[field] = {
            "common_step_count": int(common.sum()),
            "mean_difference": (
                float((left[common] - right[common]).mean()) if common.any() else None
            ),
        }
    return result


def _discovery_controller_summary(
    rows: Sequence[Mapping[str, Any]],
    event: np.ndarray,
) -> dict[str, Any]:
    summary = _edr_controller_summary(rows, event)
    trajectory = summary["trajectory"]
    live = [row for row in trajectory if not row["cold_start_active"]]
    if len(live) < 10:
        raise Plan4ChallengeError("discovery trajectory has fewer than ten live actions")
    tail = live[-10:]
    summary.update(
        {
            "live_transition_count": len(live),
            "first_live_pi": float(live[0]["applied_pi"]),
            "post_cold_fraction_below_005": float(
                np.mean([row["applied_pi"] < 0.05 for row in live])
            ),
            "last_ten_pi_mean": float(
                np.mean([row["applied_pi"] for row in tail])
            ),
            "last_ten_pi_min": float(min(row["applied_pi"] for row in tail)),
            "last_ten_pi_max": float(max(row["applied_pi"] for row in tail)),
            "last_ten_all_below_005": all(
                row["applied_pi"] < 0.05 for row in tail
            ),
            "last_ten_floor_fraction": float(
                np.mean([row["lower_bound_active"] for row in tail])
            ),
        }
    )
    summary["mechanical_discovery"] = bool(
        summary["last_ten_all_below_005"]
        and summary["last_ten_floor_fraction"] == 0.0
    )
    return summary


def build_discovery_analysis(
    bundle_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    bundle, manifest = load_screen_bundle(bundle_path)
    if manifest.get("stage") != "edr-cold-start-discovery":
        raise Plan4ChallengeError("not an EDR cold-start discovery bundle")
    status = status_rows(bundle, repo_root)
    if any(row["run_state"] != "completed" for row in status):
        raise Plan4ChallengeError("all discovery conditions must complete")
    if {row["condition"] for row in status} != set(DISCOVERY_CONDITIONS):
        raise Plan4ChallengeError("discovery conditions differ from the manifest")

    loaded = {}
    provenance = []
    initial_hashes = set()
    uniform_hashes = set()
    for member in status:
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
                "condition": member["condition"],
                "metrics_sha256": _sha256(metric_path),
                "schedule_sha256": _sha256(schedule_path),
            }
        )
    if len(initial_hashes) != 1 or len(uniform_hashes) != 1:
        raise Plan4ChallengeError("cold-start discovery lost pairing")

    discovery_member = next(
        row for row in status if row["condition"] == DISCOVERY_CONDITION
    )
    resolved = resolve_schedule(load_config(discovery_member["config_path"]).data)
    event = _event_mask(resolved)
    rows = {name: value["condition_steps"] for name, value in loaded.items()}
    discovery_rows = rows[DISCOVERY_CONDITION]
    controller = _discovery_controller_summary(discovery_rows, event)

    cold_steps = {
        row["step"]
        for row in discovery_rows
        if isinstance(row.get("controller_decision"), Mapping)
        and row["controller_decision"]["cold_start_active"]
    }
    cold = np.asarray([row["step"] in cold_steps for row in discovery_rows])
    live = np.asarray(
        [
            isinstance(row.get("controller_decision"), Mapping)
            and not row["controller_decision"]["cold_start_active"]
            for row in discovery_rows
        ]
    )
    fixed_rows = rows[PRIMARY_COMPARATOR]
    cold_parameter_parity = all(
        left["parameter_hash"] == right["parameter_hash"]
        for left, right, selected in zip(
            discovery_rows, fixed_rows, cold, strict=True
        )
        if selected
    )
    if not cold_parameter_parity:
        raise Plan4ChallengeError("EDR and fixed .05 diverged during cold start")

    comparisons = {
        "discovery_minus_fixed_005": {
            "full": _paired_predictive_summary(discovery_rows, fixed_rows, event),
            "cold": _paired_outcome_summary(discovery_rows, fixed_rows, cold),
            "live": _paired_outcome_summary(discovery_rows, fixed_rows, live),
        },
        "discovery_minus_cold_0025": {
            "full": _paired_predictive_summary(
                discovery_rows, rows[EDR_CONDITION], event
            ),
            "cold": _paired_outcome_summary(
                discovery_rows, rows[EDR_CONDITION], cold
            ),
            "live": _paired_outcome_summary(
                discovery_rows, rows[EDR_CONDITION], live
            ),
        },
        "discovery_minus_fixed_0025_hindsight": {
            "full": _paired_predictive_summary(
                discovery_rows, rows["fixed-pi0025"], event
            ),
            "cold": _paired_outcome_summary(
                discovery_rows, rows["fixed-pi0025"], cold
            ),
            "live": _paired_outcome_summary(
                discovery_rows, rows["fixed-pi0025"], live
            ),
        },
    }
    live_primary = comparisons["discovery_minus_fixed_005"]["live"]
    regression_checks = {
        field: value["mean_difference"] < -SECONDARY_TOLERANCE
        for field, value in live_primary.items()
        if field not in {"nll", "expected_calibration_error"}
        and value["mean_difference"] is not None
    }
    ece = live_primary["expected_calibration_error"]["mean_difference"]
    regression_checks["expected_calibration_error"] = bool(
        ece is not None and ece > SECONDARY_TOLERANCE
    )
    nll = live_primary["nll"]["mean_difference"]
    predictive_discovery = bool(
        nll is not None
        and nll < 0.0
        and not any(regression_checks.values())
    )
    conditions = {
        name: {
            "predictive": _condition_predictive_summary(condition_rows),
            "predictive_trajectory": _predictive_trajectory(condition_rows),
            "resources": loaded[name]["resource_ledger"],
        }
        for name, condition_rows in rows.items()
    }
    conditions[DISCOVERY_CONDITION]["controller"] = controller
    return {
        "schema_version": PLAN4_EDR_DISCOVERY_ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "plan4_edr_cold_start_discovery",
        "confirmatory": False,
        "primary_comparator": PRIMARY_COMPARATOR,
        "fixed_0025_is_hindsight_only": True,
        "secondary_materiality_tolerance": SECONDARY_TOLERANCE,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": _sha256(bundle / "manifest.json"),
        "common_initial_parameter_hash": next(iter(initial_hashes)),
        "common_uniform_stream_hash": next(iter(uniform_hashes)),
        "cold_prefix_parameter_parity_with_fixed_005": cold_parameter_parity,
        "input_provenance": provenance,
        "conditions": conditions,
        "comparisons": comparisons,
        "interpretation_gate": {
            "mechanical_discovery": controller["mechanical_discovery"],
            "predictive_discovery": predictive_discovery,
            "live_nll_difference_vs_fixed_005": nll,
            "material_secondary_regressions": regression_checks,
            "authorizes_fresh_replica_confirmation": bool(
                controller["mechanical_discovery"] and predictive_discovery
            ),
        },
    }


def write_discovery_analysis(bundle_path: str | Path, repo_root: str | Path) -> Path:
    root = Path(repo_root)
    analysis = build_discovery_analysis(bundle_path, root)
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
        / "edr"
        / "discovery"
        / "analysis"
        / f"edr_discovery__{_hash(identity)[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4ChallengeError(f"incomplete discovery analysis: {destination}")
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
