"""Immutable Plan 4 realized-actuation challenge construction and analysis."""

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

from .config import ExperimentConfig, load_config
from .initialization import replica_bundle_id, replica_design_hash
from .plan4_analysis import (
    MIN_PI_RANGE,
    MIN_RISK_REDUCTION,
    MIN_SIGNAL_TRANSITIONS,
    PI_MIN,
    PI_SIGNAL_THRESHOLD,
    _event_mask,
)
from .schedules import resolve_schedule


PLAN4_CHALLENGE_BUNDLE_SCHEMA_VERSION = 1
PLAN4_ACTUATION_ANALYSIS_SCHEMA_VERSION = 1
SCHEDULES = ("linear", "logistic-k32", "logistic-k64", "logistic-k128", "logistic-k256")
CONDITIONS = ("fixed-pi005", "adaptive-euclidean-h020", "adaptive-fisher-h005")
SOURCE_PATTERN = "mnist_lfu_plan2-low-data_m008_fixed-ewc-pi005__replica-{replica:04d}__*"


class Plan4ChallengeError(RuntimeError):
    """Raised when a realized-actuation challenge invariant fails."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Plan4ChallengeError(f"could not read {path}: {exc}") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_run(repo_root: Path, replica: int) -> Path:
    candidates = sorted(
        path
        for path in (
            repo_root / "cache" / "mnist_experiment" / "phase8_runs"
        ).glob(SOURCE_PATTERN.format(replica=replica))
        if (path / "COMPLETED").is_file()
    )
    if len(candidates) != 1:
        raise Plan4ChallengeError(
            f"expected one completed Plan 2 source for replica {replica}, found {len(candidates)}"
        )
    return candidates[0]


def _schedule_mapping(name: str) -> dict[str, Any]:
    if name == "linear":
        return {
            "kind": "linear",
            "p_start": 0.0,
            "p_end": 0.2,
            "center_fraction": None,
            "steepness": None,
        }
    if name not in SCHEDULES:
        raise ValueError(f"unsupported challenge schedule: {name}")
    return {
        "kind": "normalized_logistic",
        "p_start": 0.0,
        "p_end": 0.2,
        "center_fraction": 0.5,
        "steepness": float(name.removeprefix("logistic-k")),
    }


def _condition_mapping(name: str) -> dict[str, Any]:
    if name == "fixed-pi005":
        return {
            "policy": "fixed_unified",
            "fixed_pi": 0.05,
            "pi_min": 0.05,
            "pi_max": 0.05,
            "trend_half_life_p": 0.2,
            "risk_metric": "euclidean",
        }
    if name == "adaptive-euclidean-h020":
        return {
            "policy": "optimal_plugin",
            "fixed_pi": 0.5,
            "pi_min": 0.05,
            "pi_max": 0.95,
            "trend_half_life_p": 0.2,
            "risk_metric": "euclidean",
        }
    if name == "adaptive-fisher-h005":
        return {
            "policy": "optimal_plugin",
            "fixed_pi": 0.5,
            "pi_min": 0.05,
            "pi_max": 0.95,
            "trend_half_life_p": 0.05,
            "risk_metric": "fisher",
        }
    if name == "fixed-pi0025":
        return {
            "policy": "fixed_unified",
            "fixed_pi": 0.025,
            "pi_min": 0.025,
            "pi_max": 0.025,
            "trend_half_life_p": 0.2,
            "risk_metric": "euclidean",
        }
    if name == "adaptive-fisher-pimin0025-h005":
        return {
            "policy": "optimal_plugin",
            "fixed_pi": 0.5,
            "pi_min": 0.025,
            "pi_max": 0.95,
            "trend_half_life_p": 0.05,
            "risk_metric": "fisher",
        }
    raise ValueError(f"unsupported challenge condition: {name}")


def _challenge_config(
    source_mapping: Mapping[str, Any],
    *,
    source_archive: str,
    schedule: str,
    condition: str,
    device: str,
) -> ExperimentConfig:
    mapping = json.loads(json.dumps(source_mapping))
    floor_sensitivity = "0025" in condition
    mapping.update(
        {
            "schema_version": 19 if floor_sensitivity else 18,
            "artifact_schema_version": 10,
            "metric_schema_version": 14,
            "cache_root": "cache/mnist_experiment/plan4_challenge_runs",
            "experiment": (
                f"mnist_plan4-phase5-floor-sensitivity_{schedule}_{condition}"
                if floor_sensitivity
                else f"mnist_plan4-phase5-screen_{schedule}_{condition}"
            ),
            "replay": {
                "capacity": 0,
                "policy": "fifo",
                "max_steps": None,
                "mode": "hybrid",
                "archive_initialization_artifact": source_archive,
            },
        }
    )
    mapping["data"]["schedule"] = _schedule_mapping(schedule)
    mapping["runtime"]["device"] = device
    mapping["controller"].update(
        {
            **_condition_mapping(condition),
            "oracle_mode": "none",
            "reference_optimum_artifact": None,
        }
    )
    return ExperimentConfig.from_mapping(mapping)


def build_screen_bundle(
    repo_root: str | Path,
    *,
    replica: int = 1,
    device: str = "cuda",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repo_root)
    source = _source_run(root, replica)
    source_config_path = source / "config.json"
    source_metrics_path = source / "phase8_metrics.json"
    source_archive_path = source / "phase8_checkpoints.pt"
    if not source_archive_path.is_file():
        raise Plan4ChallengeError("source EWC archive is missing")
    source_mapping = _read_json(source_config_path)
    source_metrics = _read_json(source_metrics_path)
    source_replica_bundle_id = source_metrics.get("replica_bundle_id")
    if not isinstance(source_replica_bundle_id, str):
        raise Plan4ChallengeError("source replica bundle provenance is missing")
    source_replica_bundle = str(
        Path(source_mapping["cache_root"]).parent
        / "replicas"
        / source_replica_bundle_id
    )
    source_archive = str(source_archive_path.relative_to(root))
    configs: dict[str, dict[str, Any]] = {}
    entries = []
    for schedule in SCHEDULES:
        schedule_design_hash = None
        schedule_bundle_id = None
        for condition in CONDITIONS:
            config = _challenge_config(
                source_mapping,
                source_archive=source_archive,
                schedule=schedule,
                condition=condition,
                device=device,
            )
            design_hash = replica_design_hash(config)
            bundle_id = replica_bundle_id(config)
            if schedule_design_hash is None:
                schedule_design_hash = design_hash
                schedule_bundle_id = bundle_id
            elif (design_hash, bundle_id) != (schedule_design_hash, schedule_bundle_id):
                raise Plan4ChallengeError("paired conditions have different replica designs")
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
                    "replica_bundle_id": bundle_id,
                    "replica_design_hash": design_hash,
                    "source_archive": source_archive,
                    "source_replica_bundle": source_replica_bundle,
                }
            )
    identity = {
        "schema_version": PLAN4_CHALLENGE_BUNDLE_SCHEMA_VERSION,
        "builder_code_sha256": _sha256(Path(__file__)),
        "phase": 5,
        "stage": "realized-actuation-screen",
        "device": device,
        "replica": replica,
        "schedules": list(SCHEDULES),
        "conditions": list(CONDITIONS),
        "source_run_id": source.name,
        "source_config_sha256": _sha256(source_config_path),
        "source_archive_sha256": _sha256(source_archive_path),
        "config_hashes": [entry["config_hash"] for entry in entries],
        "selection_uses_predictive_metrics": False,
    }
    bundle_id = f"plan4-actuation-screen__r{replica:04d}__{_hash(identity)[:12]}"
    manifest = {
        **identity,
        "bundle_id": bundle_id,
        "entry_count": len(entries),
        "estimated_seconds": 100.0 * len(entries),
        "entries": entries,
    }
    return manifest, configs


def write_screen_bundle(
    repo_root: str | Path,
    *,
    replica: int = 1,
    device: str = "cuda",
) -> Path:
    root = Path(repo_root)
    manifest, configs = build_screen_bundle(root, replica=replica, device=device)
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
            raise Plan4ChallengeError(f"incomplete challenge bundle: {destination}")
        if _read_json(destination / "manifest.json") != manifest:
            raise Plan4ChallengeError("completed challenge bundle differs")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=incomplete))
    try:
        for relative, mapping in configs.items():
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_screen_bundle(path: str | Path) -> tuple[Path, dict[str, Any]]:
    bundle = Path(path)
    if not (bundle / "COMPLETED").is_file():
        raise Plan4ChallengeError(f"challenge bundle is incomplete: {bundle}")
    manifest = _read_json(bundle / "manifest.json")
    if manifest.get("schema_version") != PLAN4_CHALLENGE_BUNDLE_SCHEMA_VERSION:
        raise Plan4ChallengeError("unsupported challenge bundle schema")
    return bundle, manifest


def status_rows(bundle_path: str | Path, repo_root: str | Path) -> list[dict[str, Any]]:
    bundle, manifest = load_screen_bundle(bundle_path)
    root = Path(repo_root)
    rows = []
    for entry in manifest["entries"]:
        config_path = bundle / entry["config_file"]
        config = load_config(config_path)
        final = root / entry["cache_root"] / entry["run_id"]
        incomplete = root / entry["cache_root"] / ".incomplete" / entry["run_id"]
        if (final / "COMPLETED").is_file():
            state, run_path = "completed", final
        elif incomplete.exists():
            state, run_path = "incomplete", incomplete
        elif final.exists():
            state, run_path = "invalid", final
        else:
            state, run_path = "missing", final
        replica_path = (
            root
            / Path(config.cache_root).parent
            / "replicas"
            / replica_bundle_id(config)
        )
        rows.append(
            {
                **entry,
                "config_path": str(config_path),
                "run_path": str(run_path),
                "run_state": state,
                "replica_path": str(replica_path),
                "replica_state": (
                    "completed" if (replica_path / "COMPLETED").is_file() else "missing"
                ),
            }
        )
    return rows


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(np.ptp(left)) == 0.0 or float(np.ptp(right)) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _controller_summary(rows: Sequence[Mapping[str, Any]], event: np.ndarray) -> dict[str, Any]:
    updates = [row for row in rows if isinstance(row.get("controller_decision"), Mapping)]
    if len(updates) != event.size:
        raise Plan4ChallengeError("controller trace does not align with event transitions")
    decisions = [row["controller_decision"] for row in updates]
    applied = np.asarray([float(row["applied_pi"]) for row in decisions])
    plugin = np.asarray([float(row.get("plugin_pi", row["raw_pi"])) for row in decisions])
    signal = np.asarray([float(row.get("signal_energy", 0.0)) for row in decisions])
    old = np.asarray([float(row.get("old_covariance_risk", 0.0)) for row in decisions])
    new = np.asarray([float(row.get("new_covariance_risk", 0.0)) for row in decisions])
    action_risk = (1.0 - applied) ** 2 * (signal + old) + applied**2 * new
    fixed_risk = (1.0 - PI_MIN) ** 2 * (signal + old) + PI_MIN**2 * new
    event_applied = applied[event]
    event_plugin = plugin[event]
    event_signal = signal[event]
    action_total = float(action_risk[event].sum())
    fixed_total = float(fixed_risk[event].sum())
    correlation = _correlation(event_signal, event_applied)
    result = {
        "event_transition_count": int(event.sum()),
        "event_signal_transition_count": int(np.sum(event_applied > PI_SIGNAL_THRESHOLD)),
        "event_applied_pi_min": float(event_applied.min()),
        "event_applied_pi_max": float(event_applied.max()),
        "event_applied_pi_range": float(np.ptp(event_applied)),
        "event_ungated_pi_min": float(event_plugin.min()),
        "event_ungated_pi_max": float(event_plugin.max()),
        "event_ungated_pi_range": float(np.ptp(event_plugin)),
        "event_signal_action_correlation": correlation,
        "event_action_risk": action_total,
        "event_fixed_005_counterfactual_risk": fixed_total,
        "event_relative_risk_reduction_vs_fixed_005": (
            (fixed_total - action_total) / max(fixed_total, 1e-15)
        ),
        "full_applied_pi_min": float(applied.min()),
        "full_applied_pi_mean": float(applied.mean()),
        "full_applied_pi_max": float(applied.max()),
        "lower_bound_fraction": float(np.mean(applied <= PI_MIN + 1e-12)),
        "cold_start_fraction": float(np.mean([bool(row["cold_start_active"]) for row in decisions])),
        "zero_information_fallback_count": int(
            np.sum([bool(row.get("zero_information_fallback", False)) for row in decisions])
        ),
    }
    result["passes_actuation_gate"] = bool(
        result["event_signal_transition_count"] >= MIN_SIGNAL_TRANSITIONS
        and result["event_applied_pi_range"] >= MIN_PI_RANGE
        and correlation is not None
        and correlation > 0.0
        and result["event_relative_risk_reduction_vs_fixed_005"] >= MIN_RISK_REDUCTION
        and result["zero_information_fallback_count"] == 0
    )
    result["trajectory"] = [
        {
            "step": int(row["step"]),
            "p": float(row["p"]),
            "in_event_window": bool(event[index]),
            "applied_pi": float(decision["applied_pi"]),
            "raw_pi": float(decision["raw_pi"]),
            "ungated_plugin_pi": float(decision.get("plugin_pi", decision["raw_pi"])),
            "signal_energy": float(decision.get("signal_energy", 0.0)),
            "old_covariance_risk": float(decision.get("old_covariance_risk", 0.0)),
            "new_covariance_risk": float(decision.get("new_covariance_risk", 0.0)),
            "cold_start_active": bool(decision["cold_start_active"]),
        }
        for index, (row, decision) in enumerate(zip(updates, decisions, strict=True))
    ]
    return result


def build_actuation_analysis(bundle_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    bundle, manifest = load_screen_bundle(bundle_path)
    root = Path(repo_root)
    status = status_rows(bundle, root)
    if any(row["run_state"] != "completed" for row in status):
        raise Plan4ChallengeError("all challenge runs must complete before analysis")
    schedules: dict[str, Any] = {}
    uniform_hashes = set()
    initial_hashes = set()
    input_provenance = []
    for schedule_name in SCHEDULES:
        members = [row for row in status if row["schedule"] == schedule_name]
        conditions = {}
        schedule_hashes = set()
        schedule_uniforms = set()
        for member in members:
            run_path = Path(member["run_path"])
            metric_path = run_path / "plan3_hybrid_metrics.json"
            schedule_path = run_path / "schedule_trajectory.json"
            metrics = _read_json(metric_path)
            schedule = _read_json(schedule_path)
            rows = metrics["condition_steps"]
            resolved = resolve_schedule(load_config(member["config_path"]).data)
            event = _event_mask(resolved)
            summary = _controller_summary(rows, event)
            conditions[member["condition"]] = summary
            schedule_hashes.add(schedule["schedule_hash"])
            schedule_uniforms.add(schedule["uniform_stream_hash"])
            uniform_hashes.add(schedule["uniform_stream_hash"])
            initial_hashes.add(metrics["pairing"]["initial_parameter_hash"])
            input_provenance.append(
                {
                    "run_id": member["run_id"],
                    "schedule": schedule_name,
                    "condition": member["condition"],
                    "config_hash": member["config_hash"],
                    "metrics_sha256": _sha256(metric_path),
                    "schedule_sha256": _sha256(schedule_path),
                }
            )
        if len(schedule_hashes) != 1 or len(schedule_uniforms) != 1:
            raise Plan4ChallengeError("within-schedule pairing differs")
        fisher = conditions["adaptive-fisher-h005"]
        schedules[schedule_name] = {
            "schedule_hash": next(iter(schedule_hashes)),
            "uniform_stream_hash": next(iter(schedule_uniforms)),
            "passes_actuation_gate": fisher["passes_actuation_gate"],
            "conditions": conditions,
        }
    if len(uniform_hashes) != 1 or len(initial_hashes) != 1:
        raise Plan4ChallengeError("cross-schedule common randomness or initialization differs")
    passing = [name for name in SCHEDULES if name != "linear" and schedules[name]["passes_actuation_gate"]]
    passing.sort(
        key=lambda name: (
            -schedules[name]["conditions"]["adaptive-fisher-h005"]["event_relative_risk_reduction_vs_fixed_005"],
            -schedules[name]["conditions"]["adaptive-fisher-h005"]["event_applied_pi_range"],
            float(name.removeprefix("logistic-k")),
        )
    )
    selected = passing[0] if passing else None
    return {
        "schema_version": PLAN4_ACTUATION_ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "plan4_realized_actuation_only",
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": _sha256(bundle / "manifest.json"),
        "selection_uses_predictive_metrics": False,
        "selection_input_whitelist": [
            "controller_decision",
            "controller_state_before",
            "controller_acceptance",
            "step",
            "p",
            "schedule_trajectory",
        ],
        "sealed_predictive_fields": ["classification", "resource_ledger"],
        "gate": {
            "pi_signal_threshold": PI_SIGNAL_THRESHOLD,
            "minimum_signal_transitions": MIN_SIGNAL_TRANSITIONS,
            "minimum_pi_range": MIN_PI_RANGE,
            "minimum_relative_risk_reduction": MIN_RISK_REDUCTION,
            "positive_signal_action_correlation": True,
        },
        "decision": "proceed-to-predictive-confirmation" if selected else "stop",
        "selected_schedule": selected,
        "common_uniform_stream_hash": next(iter(uniform_hashes)),
        "common_initial_parameter_hash": next(iter(initial_hashes)),
        "input_provenance": input_provenance,
        "schedules": schedules,
    }


def write_actuation_analysis(bundle_path: str | Path, repo_root: str | Path) -> Path:
    root = Path(repo_root)
    analysis = build_actuation_analysis(bundle_path, root)
    identity = {
        "schema_version": analysis["schema_version"],
        "analysis_kind": analysis["analysis_kind"],
        "bundle_id": analysis["bundle_id"],
        "bundle_manifest_sha256": analysis["bundle_manifest_sha256"],
        "input_provenance": analysis["input_provenance"],
        "gate": analysis["gate"],
    }
    destination = (
        root
        / "cache"
        / "mnist_experiment"
        / "plan4"
        / "challenge"
        / "analysis"
        / f"phase5_actuation__{_hash(identity)[:12]}"
    )
    if destination.exists():
        if not (destination / "COMPLETED").is_file():
            raise Plan4ChallengeError(f"incomplete actuation analysis: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent))
    try:
        (temporary / "summary.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "manifest.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "COMPLETED").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
