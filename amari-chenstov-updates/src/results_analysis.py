"""Strict read-only loaders and summaries for MNIST experiment results."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import MANIFEST_SCHEMA_VERSION
from .classification_backfill import (
    default_classification_backfill_root,
    overlay_classification_backfill,
)
from .config import ExperimentConfig
from .exposure import EXPOSURE_FIELDS, stream_exposure_rows
from .phase9 import Phase9Bundle, phase9_status_rows

PHASE3_REFERENCE_SCHEMA_VERSION = 1
PHASE4_METHODS_BY_SCHEMA = {
    1: ("ema", "ac_only", "full_lfu", "periodic_fresh"),
    2: (
        "ema",
        "ac_only",
        "full_lfu",
        "periodic_fresh",
        "ridge_ac_only",
        "ridge_full_lfu",
    ),
}
PHASE6_METHODS_BY_SCHEMA = {
    1: (
        "ema",
        "ac_only",
        "full_lfu",
        "periodic_fresh",
        "ridge_ac_only",
        "ridge_full_lfu",
    ),
}
PHASE7_FIXED_METRIC_SCHEMA_VERSION = 1
PHASE7_COUPLED_METRIC_SCHEMA_VERSION = 1
PHASE7_LEGACY_LANCZOS_SHA256 = (
    "643bb7562ad7ad2d087229414d7adf114f575b342f462be619f90be920691634"
)
PHASE7_DIAGONAL_ERROR_INSTABILITY_THRESHOLD = 0.1
PHASE7_DENSE_ERROR_INSTABILITY_THRESHOLD = 1.0
PHASE8_METRIC_SCHEMA_VERSIONS = (2, 3, 4, 5, 6, 7, 8)
PHASE8_TRAJECTORY_SCHEMA_BY_METRIC = {
    2: 2,
    3: 2,
    4: 3,
    5: 4,
    6: 4,
    7: 4,
    8: 4,
}
PHASE8_CONTROLLER_STATE_SCHEMA_BY_METRIC = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 4,
    7: 4,
    8: 4,
}
PHASE8_ORACLE_PATH_SCHEMA_BY_METRIC = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 4,
    7: 4,
    8: 4,
}
RIDGE_METHODS = frozenset({"ridge_ac_only", "ridge_full_lfu"})
DEFAULT_PHASE3_RUN_IDS = (
    "mnist_lfu_phase3_final__replica-0000__e216d98c2f668d87",
    "mnist_lfu_phase3_interior_checkpoints__replica-0000__548b52840b2d4d0f",
)
DEFAULT_PHASE3_P_VALUES = (0.25, 0.5, 0.75)
DEFAULT_PHASE4_BASELINE_RUN_IDS = (
    "mnist_lfu_phase4_ridge_pilot__replica-0000__8294cf954c52cb00",
)


class AnalysisArtifactError(RuntimeError):
    """Raised when a result artifact cannot be analyzed without guessing."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisArtifactError(f"could not read {path}: {exc}") from exc


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisArtifactError(f"{context} must be an object")
    return value


def _require_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    context: str,
) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise AnalysisArtifactError(f"{context} is missing keys: {missing}")


@dataclasses.dataclass(frozen=True)
class _RunEnvelope:
    path: Path
    run_id: str
    config: ExperimentConfig
    manifest: Mapping[str, Any]


def _load_envelope(path: str | Path) -> _RunEnvelope:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise AnalysisArtifactError(f"run is incomplete: {run_path}")
    config_mapping = _require_mapping(
        _read_json(run_path / "config.json"),
        f"{run_path}/config.json",
    )
    manifest = _require_mapping(
        _read_json(run_path / "manifest.json"),
        f"{run_path}/manifest.json",
    )
    try:
        config = ExperimentConfig.from_mapping(config_mapping)
    except ValueError as exc:
        raise AnalysisArtifactError(
            f"invalid configuration in {run_path}: {exc}"
        ) from exc
    _require_keys(
        manifest,
        {
            "manifest_schema_version",
            "run_id",
            "config_hash",
            "status",
        },
        f"{run_path}/manifest.json",
    )
    if manifest["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise AnalysisArtifactError(
            f"unsupported manifest schema in {run_path}: "
            f"{manifest['manifest_schema_version']}"
        )
    if manifest["status"] != "completed":
        raise AnalysisArtifactError(f"manifest is not completed: {run_path}")
    if manifest["run_id"] != run_path.name or config.run_id != run_path.name:
        raise AnalysisArtifactError(f"run identity mismatch: {run_path}")
    if manifest["config_hash"] != config.config_hash:
        raise AnalysisArtifactError(f"configuration hash mismatch: {run_path}")
    return _RunEnvelope(
        path=run_path,
        run_id=run_path.name,
        config=config,
        manifest=manifest,
    )


@dataclasses.dataclass(frozen=True)
class Phase3Run:
    path: Path
    run_id: str
    config: ExperimentConfig
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]


def load_phase3_run(path: str | Path) -> Phase3Run:
    envelope = _load_envelope(path)
    metrics = _require_mapping(
        _read_json(envelope.path / "phase3_metrics.json"),
        f"{envelope.path}/phase3_metrics.json",
    )
    _require_keys(
        metrics,
        {
            "reference_fisher_schema_version",
            "replica_bundle_id",
            "device",
            "peak_cuda_memory_bytes",
            "checkpoints",
        },
        f"{envelope.path}/phase3_metrics.json",
    )
    if (
        metrics["reference_fisher_schema_version"]
        != PHASE3_REFERENCE_SCHEMA_VERSION
    ):
        raise AnalysisArtifactError(
            f"unsupported Phase 3 schema in {envelope.path}: "
            f"{metrics['reference_fisher_schema_version']}"
        )
    checkpoints = metrics["checkpoints"]
    if not isinstance(checkpoints, list) or not checkpoints:
        raise AnalysisArtifactError(
            f"Phase 3 run has no checkpoints: {envelope.path}"
        )
    configured = tuple(envelope.config.reference.stencil_p_values)
    observed = tuple(float(row["p"]) for row in checkpoints)
    if observed != configured:
        raise AnalysisArtifactError(
            f"Phase 3 checkpoint grid does not match its config: {envelope.path}"
        )
    return Phase3Run(
        path=envelope.path,
        run_id=envelope.run_id,
        config=envelope.config,
        manifest=envelope.manifest,
        metrics=metrics,
    )


def load_selected_phase3_runs(
    root: str | Path,
    run_ids: Sequence[str] = DEFAULT_PHASE3_RUN_IDS,
    *,
    expected_p_values: Sequence[float] = DEFAULT_PHASE3_P_VALUES,
) -> tuple[Phase3Run, ...]:
    runs = tuple(load_phase3_run(Path(root) / run_id) for run_id in run_ids)
    observed_p = sorted(
        float(checkpoint["p"])
        for run in runs
        for checkpoint in run.metrics["checkpoints"]
    )
    if observed_p != sorted(float(value) for value in expected_p_values):
        raise AnalysisArtifactError(
            "selected Phase 3 runs do not cover exactly "
            f"{tuple(expected_p_values)}; observed {tuple(observed_p)}"
        )
    bundle_ids = {run.metrics["replica_bundle_id"] for run in runs}
    if len(bundle_ids) != 1:
        raise AnalysisArtifactError(
            "selected Phase 3 runs use incompatible replica bundles"
        )
    return runs


def phase3_convergence_rows(runs: Sequence[Phase3Run]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for checkpoint in run.metrics["checkpoints"]:
            for entry in checkpoint["convergence"]:
                rows.append(
                    {
                        "run_id": run.run_id,
                        "p": float(checkpoint["p"]),
                        **entry,
                    }
                )
    return rows


def phase3_stencil_rows(runs: Sequence[Phase3Run]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for checkpoint in run.metrics["checkpoints"]:
            p_value = float(checkpoint["p"])
            for direction in checkpoint["directions"]:
                shared = {
                    "run_id": run.run_id,
                    "p": p_value,
                    "direction_index": direction["direction_index"],
                    "sample_size": direction["sample_size"],
                    "lfu_elapsed_seconds": direction["lfu_elapsed_seconds"],
                }
                for entry in direction["weighted_stencils"]:
                    rows.append(
                        {
                            **shared,
                            "stencil_kind": "weighted",
                            "target": "full_lfu",
                            **entry,
                        }
                    )
                for entry in direction["fixed_measure_stencils"]:
                    rows.append(
                        {
                            **shared,
                            "stencil_kind": "unweighted",
                            "target": "residual",
                            "plus_weight_mean": None,
                            "minus_weight_mean": None,
                            "plus_effective_sample_size": None,
                            "minus_effective_sample_size": None,
                            "relative_to_next_epsilon": None,
                            **entry,
                        }
                    )
    return rows


def phase3_diagnostic_rows(runs: Sequence[Phase3Run]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for checkpoint in run.metrics["checkpoints"]:
            convergence = checkpoint["convergence"]
            directions = checkpoint["directions"]
            rows.append(
                {
                    "run_id": run.run_id,
                    "p": float(checkpoint["p"]),
                    "selected_reference_sample_size": (
                        run.config.reference.sample_size
                    ),
                    "reference_elapsed_seconds": convergence[-1][
                        "elapsed_seconds"
                    ],
                    "lfu_elapsed_seconds": max(
                        row["lfu_elapsed_seconds"]
                        for row in directions
                        if row["sample_size"]
                        == run.config.reference.sample_size
                    ),
                    "minimum_eigenvalue": checkpoint["eigendecomposition"][
                        "minimum_eigenvalue"
                    ],
                    "maximum_eigenvalue": checkpoint["eigendecomposition"][
                        "maximum_eigenvalue"
                    ],
                    "eigendecomposition_elapsed_seconds": checkpoint[
                        "eigendecomposition"
                    ]["elapsed_seconds"],
                    "peak_cuda_memory_bytes": run.metrics[
                        "peak_cuda_memory_bytes"
                    ],
                }
            )
    return rows


def _phase4_design_mapping(config: ExperimentConfig) -> dict[str, Any]:
    estimator = config.estimator
    return {
        "replica_id": config.replica_id,
        "replica_seed": config.replica_seed,
        "data": dataclasses.asdict(config.data),
        "runtime": dataclasses.asdict(config.runtime),
        "reference": dataclasses.asdict(config.reference),
        "initialization": dataclasses.asdict(config.initialization),
        "optimizer": dataclasses.asdict(config.optimizer),
        "controller": dataclasses.asdict(config.controller),
        "estimator": {
            "representation": estimator.representation,
            "ema_gain": estimator.ema_gain,
            "fresh_fisher_cadence": estimator.fresh_fisher_cadence,
            "low_rank": estimator.low_rank,
        },
    }


def _load_trajectory_metadata(path: Path) -> Mapping[str, Any]:
    try:
        import torch

        artifact = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise AnalysisArtifactError(
            f"could not read trajectory metadata {path}: {exc}"
        ) from exc
    artifact = _require_mapping(artifact, str(path))
    _require_keys(
        artifact,
        {
            "schema_version",
            "content_hash",
            "p_values",
            "parameter_layout",
        },
        str(path),
    )
    if artifact["schema_version"] != 1:
        raise AnalysisArtifactError(
            f"unsupported trajectory schema in {path}: "
            f"{artifact['schema_version']}"
        )
    return {
        "content_hash": artifact["content_hash"],
        "p_values": tuple(float(value) for value in artifact["p_values"]),
        "parameter_layout": artifact["parameter_layout"],
    }


def _structural_parameter_layout(layout: Mapping[str, Any]) -> dict[str, Any]:
    parameters = layout.get("parameters")
    if not isinstance(parameters, list):
        raise AnalysisArtifactError("parameter layout entries must be a list")
    return {
        "total_numel": layout.get("total_numel"),
        "parameters": [
            {
                key: entry[key]
                for key in ("name", "shape", "start", "stop")
                if key in entry
            }
            for entry in parameters
        ],
    }


@dataclasses.dataclass(frozen=True)
class Phase4Run:
    path: Path
    run_id: str
    config: ExperimentConfig
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    metric_schema_version: int
    trajectory_hash: str
    p_values: tuple[float, ...]
    parameter_layout_hash: str
    design_hash: str


def load_phase4_run(path: str | Path) -> Phase4Run:
    envelope = _load_envelope(path)
    metrics = _require_mapping(
        _read_json(envelope.path / "phase4_metrics.json"),
        f"{envelope.path}/phase4_metrics.json",
    )
    _require_keys(
        metrics,
        {
            "phase4_metric_schema_version",
            "replica_bundle_id",
            "trajectory_hash",
            "methods",
            "condition_steps",
            "common_steps",
        },
        f"{envelope.path}/phase4_metrics.json",
    )
    schema = metrics["phase4_metric_schema_version"]
    if schema not in PHASE4_METHODS_BY_SCHEMA:
        raise AnalysisArtifactError(
            f"unsupported Phase 4 metric schema in {envelope.path}: {schema}"
        )
    expected_methods = PHASE4_METHODS_BY_SCHEMA[schema]
    if tuple(metrics["methods"]) != expected_methods:
        raise AnalysisArtifactError(
            f"Phase 4 method set does not match schema {schema}: {envelope.path}"
        )
    trajectory = _load_trajectory_metadata(
        envelope.path / "phase4_trajectory.pt"
    )
    if trajectory["content_hash"] != metrics["trajectory_hash"]:
        raise AnalysisArtifactError(
            f"trajectory hash mismatch in {envelope.path}"
        )
    p_values = trajectory["p_values"]
    if len(p_values) != envelope.config.data.num_p_steps:
        raise AnalysisArtifactError(
            f"trajectory length does not match config in {envelope.path}"
        )
    layout = _require_mapping(
        trajectory["parameter_layout"],
        f"{envelope.path}/phase4_trajectory.pt parameter_layout",
    )
    if layout.get("total_numel") != 512:
        raise AnalysisArtifactError(
            f"unexpected parameter count in {envelope.path}: "
            f"{layout.get('total_numel')}"
        )
    condition_steps = metrics["condition_steps"]
    if not isinstance(condition_steps, list):
        raise AnalysisArtifactError(
            f"condition_steps must be a list in {envelope.path}"
        )
    for method in expected_methods:
        method_rows = [row for row in condition_steps if row.get("method") == method]
        if len(method_rows) != len(p_values):
            raise AnalysisArtifactError(
                f"condition {method} does not align with trajectory in "
                f"{envelope.path}"
            )
        observed_p = tuple(float(row["p"]) for row in method_rows)
        if observed_p != p_values:
            raise AnalysisArtifactError(
                f"condition {method} has a different p grid in {envelope.path}"
            )
    unknown_methods = {
        row.get("method") for row in condition_steps
    } - set(expected_methods)
    if unknown_methods:
        raise AnalysisArtifactError(
            f"unknown condition rows in {envelope.path}: "
            f"{sorted(unknown_methods)}"
        )
    return Phase4Run(
        path=envelope.path,
        run_id=envelope.run_id,
        config=envelope.config,
        manifest=envelope.manifest,
        metrics=metrics,
        metric_schema_version=int(schema),
        trajectory_hash=metrics["trajectory_hash"],
        p_values=p_values,
        parameter_layout_hash=_canonical_hash(
            _structural_parameter_layout(layout)
        ),
        design_hash=_canonical_hash(_phase4_design_mapping(envelope.config)),
    )


def discover_phase4_runs(root: str | Path) -> tuple[Phase4Run, ...]:
    run_root = Path(root)
    if not run_root.is_dir():
        raise AnalysisArtifactError(f"Phase 4 run root does not exist: {run_root}")
    runs = []
    for path in sorted(run_root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if (path / "phase4_metrics.json").exists():
            runs.append(load_phase4_run(path))
    return tuple(runs)


def select_principal_phase4_runs(
    runs: Sequence[Phase4Run],
    *,
    required_run_ids: Sequence[str] = DEFAULT_PHASE4_BASELINE_RUN_IDS,
    experiment_prefixes: Sequence[str] = ("mnist_lfu_phase5_",),
) -> tuple[Phase4Run, ...]:
    """Select accepted baselines and explicitly named Phase 5 experiments."""

    by_id = {run.run_id: run for run in runs}
    missing = sorted(set(required_run_ids) - set(by_id))
    if missing:
        raise AnalysisArtifactError(
            f"required Phase 4 baseline runs are missing: {missing}"
        )
    selected_ids = set(required_run_ids)
    selected_ids.update(
        run.run_id
        for run in runs
        if any(
            run.config.experiment.startswith(prefix)
            for prefix in experiment_prefixes
        )
    )
    return tuple(run for run in runs if run.run_id in selected_ids)


def _condition_label(method: str, half_life: float | None) -> str:
    if method not in RIDGE_METHODS:
        return method
    return f"{method} [h={half_life:g}]"


def normalized_phase4_condition_rows(
    runs: Sequence[Phase4Run],
) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        config = run.config
        half_life = config.estimator.ridge_half_life_steps
        for source in run.metrics["condition_steps"]:
            method = source["method"]
            method_half_life = half_life if method in RIDGE_METHODS else None
            rows.append(
                {
                    **source,
                    "ridge": source.get("ridge"),
                    "run_id": run.run_id,
                    "run_path": str(run.path),
                    "experiment": config.experiment,
                    "metric_schema_version": run.metric_schema_version,
                    "config_schema_version": config.schema_version,
                    "replica_id": config.replica_id,
                    "replica_seed": config.replica_seed,
                    "replica_bundle_id": run.metrics["replica_bundle_id"],
                    "trajectory_hash": run.trajectory_hash,
                    "design_hash": run.design_hash,
                    "parameter_layout_hash": run.parameter_layout_hash,
                    "num_p_steps": config.data.num_p_steps,
                    "samples_per_step": config.data.samples_per_step,
                    "online_observation_budget": (
                        config.data.num_p_steps
                        * config.data.samples_per_step
                    ),
                    "ema_gain": config.estimator.ema_gain,
                    "fresh_fisher_cadence": (
                        config.estimator.fresh_fisher_cadence
                    ),
                    "ridge_half_life_steps": method_half_life,
                    "ridge_amplitude_epsilon": (
                        config.estimator.ridge_amplitude_epsilon
                    ),
                    "ridge_coherence_threshold": (
                        config.estimator.ridge_coherence_threshold
                    ),
                    "condition_label": _condition_label(
                        method,
                        method_half_life,
                    ),
                }
            )
    return rows


def _evidence_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["replica_id"],
        row["trajectory_hash"],
        row["design_hash"],
        row["method"],
        (
            row["ridge_half_life_steps"]
            if row["method"] in RIDGE_METHODS
            else None
        ),
    )


def select_phase4_condition_evidence(
    runs: Sequence[Phase4Run],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply schema precedence without counting repeated controls twice."""

    rows = normalized_phase4_condition_rows(runs)
    sources: dict[tuple[Any, ...], dict[str, tuple[int, str]]] = defaultdict(dict)
    for row in rows:
        sources[_evidence_key(row)][row["run_id"]] = (
            row["metric_schema_version"],
            row["run_id"],
        )
    selected_source = {
        key: max(candidates, key=lambda run_id: candidates[run_id])
        for key, candidates in sources.items()
    }
    selected = [
        row
        for row in rows
        if row["run_id"] == selected_source[_evidence_key(row)]
    ]
    selected_counts = defaultdict(int)
    available_counts = defaultdict(int)
    for row in rows:
        available_counts[row["run_id"]] += 1
    for row in selected:
        selected_counts[row["run_id"]] += 1

    inventory = []
    for run in runs:
        selected_count = selected_counts[run.run_id]
        available_count = available_counts[run.run_id]
        if selected_count == available_count:
            status = "selected"
        elif selected_count == 0:
            status = "superseded"
        else:
            status = "partially selected"
        inventory.append(
            {
                "run_id": run.run_id,
                "experiment": run.config.experiment,
                "replica_id": run.config.replica_id,
                "metric_schema_version": run.metric_schema_version,
                "num_p_steps": run.config.data.num_p_steps,
                "samples_per_step": run.config.data.samples_per_step,
                "ema_gain": run.config.estimator.ema_gain,
                "ridge_half_life_steps": (
                    run.config.estimator.ridge_half_life_steps
                ),
                "trajectory_hash": run.trajectory_hash,
                "available_condition_rows": available_count,
                "selected_condition_rows": selected_count,
                "selection_status": status,
            }
        )
    return selected, inventory


def phase4_replica_summaries(
    selected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize each condition within each replica over interior p values."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        if 0.0 < float(row["p"]) < 1.0:
            key = (
                row["run_id"],
                row["replica_id"],
                row["design_hash"],
                row["method"],
                row["ridge_half_life_steps"],
            )
            grouped[key].append(row)
    summaries = []
    for rows in grouped.values():
        first = rows[0]
        summaries.append(
            {
                key: first[key]
                for key in (
                    "run_id",
                    "replica_id",
                    "replica_seed",
                    "trajectory_hash",
                    "design_hash",
                    "method",
                    "condition_label",
                    "num_p_steps",
                    "samples_per_step",
                    "online_observation_budget",
                    "ema_gain",
                    "ridge_half_life_steps",
                )
            }
            | {
                "interior_step_count": len(rows),
                "mean_relative_frobenius_error": sum(
                    float(row["relative_frobenius_error"]) for row in rows
                )
                / len(rows),
                "maximum_relative_frobenius_error": max(
                    float(row["relative_frobenius_error"]) for row in rows
                ),
                "mean_applied_correction_fro": sum(
                    float(row["applied_correction_fro"]) for row in rows
                )
                / len(rows),
                "maximum_relative_projection_distance": max(
                    float(row["projection"]["relative_projection_distance"])
                    for row in rows
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["num_p_steps"],
            row["samples_per_step"],
            row["ema_gain"],
            row["condition_label"],
            row["replica_id"],
        ),
    )


def phase4_cost_rows(runs: Sequence[Phase4Run]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        metrics = run.metrics
        artifact_bytes = metrics.get("artifact_payload_bytes_before_metrics")
        rows.append(
            {
                "run_id": run.run_id,
                "num_p_steps": run.config.data.num_p_steps,
                "samples_per_step": run.config.data.samples_per_step,
                "ema_gain": run.config.estimator.ema_gain,
                "ridge_half_life_steps": (
                    run.config.estimator.ridge_half_life_steps
                ),
                "trajectory_elapsed_seconds": metrics["driver"][
                    "trajectory_elapsed_seconds"
                ],
                "replay_elapsed_seconds": metrics["replay_elapsed_seconds"],
                "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
                "peak_process_rss_bytes": metrics["peak_process_rss_bytes"],
                "artifact_payload_bytes": artifact_bytes,
            }
        )
    return rows


def validate_common_parameter_layout(runs: Sequence[Phase4Run]) -> str:
    hashes = {run.parameter_layout_hash for run in runs}
    if not hashes:
        raise AnalysisArtifactError("no Phase 4 runs were supplied")
    if len(hashes) != 1:
        raise AnalysisArtifactError(
            "Phase 4 runs use incompatible parameter layouts"
        )
    return next(iter(hashes))


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


@dataclasses.dataclass(frozen=True)
class Phase6Run:
    path: Path
    run_id: str
    config: ExperimentConfig
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    metric_schema_version: int
    methods: tuple[str, ...]
    p_values: tuple[float, ...]
    trajectory_hashes: Mapping[str, str]
    parameter_layout_hash: str


def _load_phase6_trajectory(path: Path) -> Mapping[str, Any]:
    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise AnalysisArtifactError(
            f"could not read Phase 6 trajectory {path}: {exc}"
        ) from exc
    value = _require_mapping(value, str(path))
    _require_keys(
        value,
        {
            "schema_version",
            "stream_plan_hash",
            "p_values",
            "observation_indices",
            "parameter_layout",
            "conditions",
        },
        str(path),
    )
    if value["schema_version"] != 1:
        raise AnalysisArtifactError(
            f"unsupported Phase 6 trajectory schema in {path}: "
            f"{value['schema_version']}"
        )
    return value


def load_phase6_run(path: str | Path) -> Phase6Run:
    envelope = _load_envelope(path)
    metrics_path = envelope.path / "phase6_metrics.json"
    metrics = _require_mapping(_read_json(metrics_path), str(metrics_path))
    _require_keys(
        metrics,
        {
            "phase6_metric_schema_version",
            "replica_bundle_id",
            "stream_plan_hash",
            "methods",
            "adaptation",
            "pairing",
            "path_hashes",
            "path_divergence",
            "condition_steps",
            "references",
        },
        str(metrics_path),
    )
    schema = metrics["phase6_metric_schema_version"]
    if schema not in PHASE6_METHODS_BY_SCHEMA:
        raise AnalysisArtifactError(
            f"unsupported Phase 6 metric schema in {envelope.path}: {schema}"
        )
    methods = tuple(metrics["methods"])
    if methods != PHASE6_METHODS_BY_SCHEMA[schema]:
        raise AnalysisArtifactError(
            f"Phase 6 method set does not match schema {schema}: {envelope.path}"
        )
    trajectory = _load_phase6_trajectory(
        envelope.path / "phase6_trajectories.pt"
    )
    p_values = tuple(float(value) for value in trajectory["p_values"])
    if len(p_values) != envelope.config.data.num_p_steps:
        raise AnalysisArtifactError(
            f"Phase 6 trajectory length does not match config: {envelope.path}"
        )
    expected_grid = tuple(
        step / (envelope.config.data.num_p_steps - 1)
        for step in range(envelope.config.data.num_p_steps)
    )
    if p_values != expected_grid:
        raise AnalysisArtifactError(
            f"Phase 6 trajectory has an unexpected p grid: {envelope.path}"
        )
    if trajectory["stream_plan_hash"] != metrics["stream_plan_hash"]:
        raise AnalysisArtifactError(
            f"Phase 6 stream hash mismatch: {envelope.path}"
        )
    conditions = _require_mapping(
        trajectory["conditions"],
        f"{envelope.path}/phase6_trajectories.pt conditions",
    )
    if set(conditions) != set(methods):
        raise AnalysisArtifactError(
            f"Phase 6 trajectory method set is incompatible: {envelope.path}"
        )
    trajectory_hashes = _require_mapping(
        metrics["path_hashes"],
        f"{metrics_path} path_hashes",
    )
    for method in methods:
        condition = _require_mapping(
            conditions[method],
            f"{envelope.path} trajectory condition {method}",
        )
        if condition.get("content_hash") != trajectory_hashes.get(method):
            raise AnalysisArtifactError(
                f"Phase 6 trajectory hash mismatch for {method}: "
                f"{envelope.path}"
            )
        parameters = condition.get("parameters")
        displacements = condition.get("displacements")
        if (
            getattr(parameters, "ndim", None) != 2
            or parameters.shape[0] != len(p_values)
            or getattr(displacements, "shape", None)
            != (len(p_values) - 1, parameters.shape[1])
        ):
            raise AnalysisArtifactError(
                f"Phase 6 trajectory shape is invalid for {method}: "
                f"{envelope.path}"
            )
        try:
            import torch

            consistent = torch.equal(
                displacements,
                parameters[1:] - parameters[:-1],
            )
        except ImportError as exc:
            raise AnalysisArtifactError(
                "PyTorch is required to validate Phase 6 trajectories"
            ) from exc
        if not consistent:
            raise AnalysisArtifactError(
                f"Phase 6 displacements do not match parameters for {method}: "
                f"{envelope.path}"
            )

    adaptation = _require_mapping(
        metrics["adaptation"],
        f"{metrics_path} adaptation",
    )
    _require_keys(
        adaptation,
        {
            "adaptation_weight",
            "pi_max",
            "ewc_multiplier",
            "effective_ewc_strength",
            "objective_normalization",
            "post_optimization_scaling",
        },
        f"{metrics_path} adaptation",
    )
    pi_value = float(adaptation["adaptation_weight"])
    pi_max = float(adaptation["pi_max"])
    expected_strength = float(adaptation["ewc_multiplier"]) * (
        (1.0 - pi_value) / pi_value
    )
    if not 0.0 < pi_value <= pi_max < 1.0:
        raise AnalysisArtifactError(
            f"Phase 6 adaptation weights are invalid: {envelope.path}"
        )
    if not math.isclose(
        float(adaptation["effective_ewc_strength"]),
        expected_strength,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AnalysisArtifactError(
            f"Phase 6 EWC odds are inconsistent: {envelope.path}"
        )
    if (
        adaptation["objective_normalization"]
        != "mean_new_loss_plus_old_to_new_odds"
        or adaptation["post_optimization_scaling"] is not False
    ):
        raise AnalysisArtifactError(
            f"Phase 6 objective convention is incompatible: {envelope.path}"
        )

    condition_steps = metrics["condition_steps"]
    references = metrics["references"]
    if not isinstance(condition_steps, list) or not isinstance(references, list):
        raise AnalysisArtifactError(
            f"Phase 6 step and reference rows must be lists: {envelope.path}"
        )
    for method in methods:
        rows = [row for row in condition_steps if row.get("method") == method]
        reference_rows = [
            row for row in references if row.get("method") == method
        ]
        if len(rows) != len(p_values) or len(reference_rows) != len(p_values):
            raise AnalysisArtifactError(
                f"Phase 6 rows do not align for {method}: {envelope.path}"
            )
        if tuple(float(row["p"]) for row in rows) != p_values:
            raise AnalysisArtifactError(
                f"Phase 6 condition p grid differs for {method}: "
                f"{envelope.path}"
            )
        for step, (row, reference_row) in enumerate(
            zip(rows, reference_rows, strict=True)
        ):
            if row.get("step") != step or reference_row.get("step") != step:
                raise AnalysisArtifactError(
                    f"Phase 6 step order differs for {method}: {envelope.path}"
                )
            if (
                row.get("parameter_hash")
                != row.get("reference_parameter_hash")
                or row.get("parameter_hash")
                != reference_row.get("parameter_hash")
            ):
                raise AnalysisArtifactError(
                    f"Phase 6 reference is not keyed to the {method} path at "
                    f"step {step}: {envelope.path}"
                )
            proposal = row.get("proposal")
            if step + 1 == len(p_values):
                if proposal is not None:
                    raise AnalysisArtifactError(
                        f"Phase 6 final row unexpectedly has a proposal: "
                        f"{envelope.path}"
                    )
                continue
            proposal = _require_mapping(
                proposal,
                f"{metrics_path} {method} proposal step {step}",
            )
            if (
                proposal.get("post_optimization_scaling_applied") is not False
                or not math.isclose(
                    float(proposal["displacement_norm"]),
                    float(proposal["accepted_displacement_norm"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise AnalysisArtifactError(
                    f"Phase 6 proposal scaling audit failed for {method} at "
                    f"step {step}: {envelope.path}"
                )
            if not math.isclose(
                float(proposal["effective_ewc_strength"]),
                expected_strength,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AnalysisArtifactError(
                    f"Phase 6 proposal used inconsistent EWC odds for {method} "
                    f"at step {step}: {envelope.path}"
                )

    layout = _require_mapping(
        trajectory["parameter_layout"],
        f"{envelope.path}/phase6_trajectories.pt parameter_layout",
    )
    return Phase6Run(
        path=envelope.path,
        run_id=envelope.run_id,
        config=envelope.config,
        manifest=envelope.manifest,
        metrics=metrics,
        metric_schema_version=int(schema),
        methods=methods,
        p_values=p_values,
        trajectory_hashes={
            method: str(trajectory_hashes[method]) for method in methods
        },
        parameter_layout_hash=_canonical_hash(
            _structural_parameter_layout(layout)
        ),
    )


def discover_phase6_runs(root: str | Path) -> tuple[Phase6Run, ...]:
    run_root = Path(root)
    if not run_root.is_dir():
        raise AnalysisArtifactError(f"Phase 6 run root does not exist: {run_root}")
    runs = []
    for path in sorted(run_root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if (path / "phase6_metrics.json").exists():
            runs.append(load_phase6_run(path))
    if not runs:
        raise AnalysisArtifactError(
            f"no completed Phase 6 runs were found in {run_root}"
        )
    return tuple(runs)


def phase6_condition_rows(
    runs: Sequence[Phase6Run],
) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for source in run.metrics["condition_steps"]:
            proposal = source.get("proposal")
            rows.append(
                {
                    **source,
                    "run_id": run.run_id,
                    "run_path": str(run.path),
                    "experiment": run.config.experiment,
                    "replica_id": run.config.replica_id,
                    "replica_seed": run.config.replica_seed,
                    "replica_bundle_id": run.metrics["replica_bundle_id"],
                    "trajectory_hash": run.trajectory_hashes[source["method"]],
                    "parameter_layout_hash": run.parameter_layout_hash,
                    "num_p_steps": run.config.data.num_p_steps,
                    "samples_per_step": run.config.data.samples_per_step,
                    "proposal_data_loss_before": (
                        None
                        if proposal is None
                        else proposal["data_loss_before"]
                    ),
                    "proposal_data_loss_after": (
                        None
                        if proposal is None
                        else proposal["data_loss_after"]
                    ),
                    "proposal_ewc_penalty": (
                        None
                        if proposal is None
                        else proposal["ewc_penalty_after"]
                    ),
                    "proposal_displacement_norm": (
                        None
                        if proposal is None
                        else proposal["displacement_norm"]
                    ),
                    "proposal_fisher_weighted_norm": (
                        None
                        if proposal is None
                        else proposal["fisher_weighted_displacement_norm"]
                    ),
                }
            )
    return rows


def phase6_replica_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["run_id"],
                row["replica_id"],
                row["method"],
            )
        ].append(row)
    summaries = []
    for condition_rows in grouped.values():
        ordered = sorted(condition_rows, key=lambda row: row["step"])
        first = ordered[0]
        final = ordered[-1]
        interior = [
            row for row in ordered if 0.0 < float(row["p"]) < 1.0
        ]
        summaries.append(
            {
                "run_id": first["run_id"],
                "replica_id": first["replica_id"],
                "method": first["method"],
                "trajectory_hash": first["trajectory_hash"],
                "adaptation_weight": first["adaptation_weight"],
                "effective_ewc_strength": first["effective_ewc_strength"],
                "interior_step_count": len(interior),
                "mean_interior_fisher_error": (
                    None
                    if not interior
                    else sum(
                        float(row["relative_frobenius_error"])
                        for row in interior
                    )
                    / len(interior)
                ),
                "final_non_nine_accuracy": final["before_non_nine_accuracy"],
                "final_nine_accuracy": final["before_nine_accuracy"],
                "final_balanced_accuracy": final["before_balanced_accuracy"],
                "initial_nine_nll": first["before_nine_nll"],
                "final_nine_nll": final["before_nine_nll"],
                "nine_nll_improvement": (
                    float(first["before_nine_nll"])
                    - float(final["before_nine_nll"])
                ),
                "final_non_nine_nll": final["before_non_nine_nll"],
                "non_nine_forgetting": (
                    float(first["before_non_nine_accuracy"])
                    - float(final["before_non_nine_accuracy"])
                ),
                "final_distance_from_initial": final[
                    "distance_from_initial"
                ],
            }
        )
    return sorted(
        summaries,
        key=lambda row: (row["run_id"], row["method"]),
    )


@dataclasses.dataclass(frozen=True)
class Phase7FixedRun:
    path: Path
    run_id: str
    config: ExperimentConfig
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    p_values: tuple[float, ...]
    rank_grid: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Phase7CoupledRun:
    path: Path
    run_id: str
    config: ExperimentConfig
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    p_values: tuple[float, ...]
    methods: tuple[str, ...]
    selected_rank: int


def _require_phase7_files(path: Path, names: Iterable[str]) -> None:
    missing = sorted(name for name in names if not (path / name).is_file())
    if missing:
        raise AnalysisArtifactError(
            f"Phase 7 run is missing required artifacts {missing}: {path}"
        )


def load_phase7_fixed_run(path: str | Path) -> Phase7FixedRun:
    envelope = _load_envelope(path)
    metrics_path = envelope.path / "phase7_metrics.json"
    metrics = _require_mapping(_read_json(metrics_path), str(metrics_path))
    _require_keys(
        metrics,
        {
            "phase7_metric_schema_version",
            "run_kind",
            "trajectory_hash",
            "parameter_count",
            "rank_grid",
            "condition_status",
            "condition_steps",
            "legacy_lanczos_source_hash",
            "expected_legacy_lanczos_source_hash",
        },
        str(metrics_path),
    )
    if (
        metrics["phase7_metric_schema_version"]
        != PHASE7_FIXED_METRIC_SCHEMA_VERSION
        or metrics["run_kind"] != "fixed_rank_sweep"
    ):
        raise AnalysisArtifactError(
            f"unsupported Phase 7 fixed schema in {envelope.path}"
        )
    rank_grid = tuple(int(value) for value in metrics["rank_grid"])
    configured_grid = envelope.config.estimator.low_rank_grid
    if configured_grid is None or rank_grid != tuple(configured_grid):
        raise AnalysisArtifactError(
            f"Phase 7 rank grid differs from its config: {envelope.path}"
        )
    if (
        metrics["legacy_lanczos_source_hash"]
        != PHASE7_LEGACY_LANCZOS_SHA256
        or metrics["expected_legacy_lanczos_source_hash"]
        != PHASE7_LEGACY_LANCZOS_SHA256
    ):
        raise AnalysisArtifactError(
            f"Phase 7 Lanczos source hash is incompatible: {envelope.path}"
        )
    expected_conditions = {
        "diagonal" if rank == 0 else f"low_rank_diagonal_r{rank}"
        for rank in rank_grid
    }
    statuses = _require_mapping(
        metrics["condition_status"],
        f"{metrics_path} condition_status",
    )
    if set(statuses) != expected_conditions:
        raise AnalysisArtifactError(
            f"Phase 7 condition set differs from rank grid: {envelope.path}"
        )
    rows = metrics["condition_steps"]
    if not isinstance(rows, list) or not rows:
        raise AnalysisArtifactError(
            f"Phase 7 fixed run has no scalar rows: {envelope.path}"
        )
    p_values = tuple(
        index / (envelope.config.data.num_p_steps - 1)
        for index in range(envelope.config.data.num_p_steps)
    )
    for condition in expected_conditions:
        condition_rows = [
            row
            for row in rows
            if row.get("condition") == condition
            and row.get("status") == "completed"
        ]
        artifact_status = statuses[condition].get("status")
        if artifact_status == "completed":
            if len(condition_rows) != len(p_values):
                raise AnalysisArtifactError(
                    f"Phase 7 rows do not align for {condition}: "
                    f"{envelope.path}"
                )
            if tuple(float(row["p"]) for row in condition_rows) != p_values:
                raise AnalysisArtifactError(
                    f"Phase 7 p grid differs for {condition}: {envelope.path}"
                )
        elif artifact_status != "failed":
            raise AnalysisArtifactError(
                f"unknown Phase 7 condition status for {condition}: "
                f"{artifact_status}"
            )
    _require_phase7_files(
        envelope.path,
        {
            "phase7_trajectory.pt",
            "phase7_representations.pt",
            "phase7_dense_checkpoints.pt",
            "phase7_reference_plans.pt",
        },
    )
    return Phase7FixedRun(
        path=envelope.path,
        run_id=envelope.run_id,
        config=envelope.config,
        manifest=envelope.manifest,
        metrics=metrics,
        p_values=p_values,
        rank_grid=rank_grid,
    )


def load_phase7_coupled_run(path: str | Path) -> Phase7CoupledRun:
    envelope = _load_envelope(path)
    metrics_path = envelope.path / "phase7_coupled_metrics.json"
    metrics = _require_mapping(_read_json(metrics_path), str(metrics_path))
    _require_keys(
        metrics,
        {
            "phase7_coupled_metric_schema_version",
            "run_kind",
            "stream_plan_hash",
            "methods",
            "selected_rank",
            "adaptation",
            "pairing",
            "path_hashes",
            "condition_steps",
            "references",
        },
        str(metrics_path),
    )
    if (
        metrics["phase7_coupled_metric_schema_version"]
        != PHASE7_COUPLED_METRIC_SCHEMA_VERSION
        or metrics["run_kind"] != "structured_coupled"
    ):
        raise AnalysisArtifactError(
            f"unsupported Phase 7 coupled schema in {envelope.path}"
        )
    selected_rank = int(metrics["selected_rank"])
    methods = tuple(metrics["methods"])
    expected_methods = (
        "dense_ridge_full",
        "diagonal_ridge_full",
        f"low_rank_diagonal_r{selected_rank}",
    )
    if methods != expected_methods:
        raise AnalysisArtifactError(
            f"Phase 7 coupled methods are incompatible: {envelope.path}"
        )
    if (
        envelope.config.estimator.low_rank != selected_rank
        or envelope.config.estimator.low_rank_grid != [0, selected_rank]
    ):
        raise AnalysisArtifactError(
            f"Phase 7 selected rank differs from its config: {envelope.path}"
        )
    adaptation = _require_mapping(
        metrics["adaptation"],
        f"{metrics_path} adaptation",
    )
    pi_value = float(adaptation["adaptation_weight"])
    expected_strength = float(adaptation["ewc_multiplier"]) * (
        (1.0 - pi_value) / pi_value
    )
    if (
        adaptation.get("objective_normalization")
        != "mean_new_loss_plus_old_to_new_odds"
        or adaptation.get("post_optimization_scaling") is not False
        or not math.isclose(
            float(adaptation["effective_ewc_strength"]),
            expected_strength,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise AnalysisArtifactError(
            f"Phase 7 coupled EWC convention is incompatible: {envelope.path}"
        )
    p_values = tuple(
        index / (envelope.config.data.num_p_steps - 1)
        for index in range(envelope.config.data.num_p_steps)
    )
    condition_rows = metrics["condition_steps"]
    references = metrics["references"]
    if not isinstance(condition_rows, list) or not isinstance(references, list):
        raise AnalysisArtifactError(
            f"Phase 7 coupled scalar rows must be lists: {envelope.path}"
        )
    for method in methods:
        rows = [row for row in condition_rows if row.get("method") == method]
        reference_rows = [
            row for row in references if row.get("method") == method
        ]
        if len(rows) != len(p_values) or len(reference_rows) != len(p_values):
            raise AnalysisArtifactError(
                f"Phase 7 coupled rows do not align for {method}: "
                f"{envelope.path}"
            )
        if tuple(float(row["p"]) for row in rows) != p_values:
            raise AnalysisArtifactError(
                f"Phase 7 coupled p grid differs for {method}: "
                f"{envelope.path}"
            )
        for step, row in enumerate(rows):
            if row.get("step") != step:
                raise AnalysisArtifactError(
                    f"Phase 7 coupled step order differs for {method}: "
                    f"{envelope.path}"
                )
            proposal = row.get("proposal")
            if step + 1 == len(p_values):
                if proposal is not None:
                    raise AnalysisArtifactError(
                        f"Phase 7 final row has a proposal: {envelope.path}"
                    )
            elif (
                not isinstance(proposal, Mapping)
                or proposal.get("post_optimization_scaling_applied") is not False
                or not math.isclose(
                    float(proposal["effective_ewc_strength"]),
                    expected_strength,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise AnalysisArtifactError(
                    f"Phase 7 proposal audit failed for {method} at step "
                    f"{step}: {envelope.path}"
                )
    _require_phase7_files(
        envelope.path,
        {
            "phase7_coupled_trajectories.pt",
            "phase7_coupled_checkpoints.pt",
            "phase7_coupled_reference_plans.pt",
        },
    )
    return Phase7CoupledRun(
        path=envelope.path,
        run_id=envelope.run_id,
        config=envelope.config,
        manifest=envelope.manifest,
        metrics=metrics,
        p_values=p_values,
        methods=methods,
        selected_rank=selected_rank,
    )


def discover_phase7_runs(
    root: str | Path,
) -> tuple[tuple[Phase7FixedRun, ...], tuple[Phase7CoupledRun, ...]]:
    run_root = Path(root)
    if not run_root.is_dir():
        raise AnalysisArtifactError(f"Phase 7 run root does not exist: {run_root}")
    fixed = []
    coupled = []
    for path in sorted(run_root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if (path / "phase7_metrics.json").is_file():
            fixed.append(load_phase7_fixed_run(path))
        if (path / "phase7_coupled_metrics.json").is_file():
            coupled.append(load_phase7_coupled_run(path))
    if not fixed and not coupled:
        raise AnalysisArtifactError(
            f"no completed Phase 7 runs were found in {run_root}"
        )
    return tuple(fixed), tuple(coupled)


def _phase7_numerical_status(
    rows: Sequence[Mapping[str, Any]],
    artifact_status: str,
) -> str:
    if artifact_status != "completed":
        return artifact_status
    if any(
        float(row["relative_frobenius_error_to_dense"])
        > PHASE7_DENSE_ERROR_INSTABILITY_THRESHOLD
        or float(row.get("lanczos_represented_diagonal_relative_error", 0.0))
        > PHASE7_DIAGONAL_ERROR_INSTABILITY_THRESHOLD
        for row in rows
    ):
        return "numerically_unstable"
    return "completed"


def phase7_fixed_rows(
    runs: Sequence[Phase7FixedRun],
) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        statuses = run.metrics["condition_status"]
        by_condition = defaultdict(list)
        for source in run.metrics["condition_steps"]:
            by_condition[source["condition"]].append(source)
        for condition, condition_rows in by_condition.items():
            numerical_status = _phase7_numerical_status(
                [
                    row
                    for row in condition_rows
                    if row.get("status") == "completed"
                ],
                statuses[condition]["status"],
            )
            for source in condition_rows:
                rows.append(
                    {
                        **source,
                        "run_id": run.run_id,
                        "run_path": str(run.path),
                        "experiment": run.config.experiment,
                        "replica_id": run.config.replica_id,
                        "trajectory_hash": run.metrics["trajectory_hash"],
                        "numerical_status": numerical_status,
                    }
                )
    return rows


def phase7_fixed_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            grouped[(row["run_id"], row["condition"])].append(row)
    summaries = []
    for condition_rows in grouped.values():
        ordered = sorted(condition_rows, key=lambda row: row["step"])
        first = ordered[0]
        evaluation_rows = ordered[1:] or ordered
        summaries.append(
            {
                "run_id": first["run_id"],
                "condition": first["condition"],
                "representation": first["representation"],
                "requested_rank": first["requested_rank"],
                "mean_realized_rank": sum(
                    float(row["realized_rank"]) for row in evaluation_rows
                )
                / len(evaluation_rows),
                "numerical_status": first["numerical_status"],
                "mean_dense_frobenius_error": sum(
                    float(row["relative_frobenius_error_to_dense"])
                    for row in evaluation_rows
                )
                / len(evaluation_rows),
                "mean_reference_frobenius_error": sum(
                    float(row["relative_frobenius_error_to_reference"])
                    for row in evaluation_rows
                )
                / len(evaluation_rows),
                "mean_matvec_error": sum(
                    float(row["fixed_probe_matvec_relative_error"])
                    for row in evaluation_rows
                )
                / len(evaluation_rows),
                "mean_quadratic_error": sum(
                    float(row["mean_probe_quadratic_relative_error"])
                    for row in evaluation_rows
                )
                / len(evaluation_rows),
                "mean_leading_eigenvector_alignment": sum(
                    float(row["leading_eigenvector_alignment"])
                    for row in evaluation_rows
                )
                / len(evaluation_rows),
                "maximum_lanczos_diagonal_error": max(
                    float(
                        row.get(
                            "lanczos_represented_diagonal_relative_error",
                            0.0,
                        )
                    )
                    for row in evaluation_rows
                ),
                "representation_storage_bytes": max(
                    int(row["representation_storage_bytes"])
                    for row in evaluation_rows
                ),
                "mean_update_elapsed_seconds": sum(
                    float(row["update_elapsed_seconds"])
                    for row in evaluation_rows
                )
                / len(evaluation_rows),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (row["run_id"], row["requested_rank"]),
    )


def phase7_coupled_rows(
    runs: Sequence[Phase7CoupledRun],
) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for source in run.metrics["condition_steps"]:
            rows.append(
                {
                    **source,
                    "run_id": run.run_id,
                    "run_path": str(run.path),
                    "experiment": run.config.experiment,
                    "replica_id": run.config.replica_id,
                    "selected_rank": run.selected_rank,
                    "trajectory_hash": run.metrics["path_hashes"][
                        source["method"]
                    ],
                }
            )
    return rows


def phase7_coupled_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_id"], row["method"])].append(row)
    summaries = []
    for condition_rows in grouped.values():
        ordered = sorted(condition_rows, key=lambda row: row["step"])
        first = ordered[0]
        final = ordered[-1]
        evaluation_rows = ordered[1:] or ordered
        summaries.append(
            {
                "run_id": first["run_id"],
                "method": first["method"],
                "representation": first["representation"],
                "selected_rank": first["selected_rank"],
                "mean_fisher_error": sum(
                    float(row["relative_frobenius_error"])
                    for row in evaluation_rows
                )
                / len(evaluation_rows),
                "final_fisher_error": float(
                    final["relative_frobenius_error"]
                ),
                "final_nll": float(final["before_nll"]),
                "final_nine_nll": float(final["before_nine_nll"]),
                "final_non_nine_nll": float(final["before_non_nine_nll"]),
                "final_balanced_accuracy": float(
                    final["before_balanced_accuracy"]
                ),
                "final_distance_from_initial": float(
                    final["distance_from_initial"]
                ),
                "mean_update_elapsed_seconds": sum(
                    float(row["update_elapsed_seconds"])
                    for row in ordered
                )
                / len(ordered),
            }
        )
    return sorted(summaries, key=lambda row: (row["run_id"], row["method"]))


@dataclasses.dataclass(frozen=True)
class Phase8ControllerRun:
    path: Path
    run_id: str
    config: ExperimentConfig
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    p_values: tuple[float, ...]
    methods: tuple[str, ...]
    exposures: tuple[Mapping[str, Any], ...]


def _read_torch_mapping(path: Path) -> Mapping[str, Any]:
    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise AnalysisArtifactError(f"could not read {path}: {exc}") from exc
    return _require_mapping(value, str(path))


def load_phase8_controller_run(path: str | Path) -> Phase8ControllerRun:
    """Strictly load notebook-sized Phase 8 artifacts without checkpoints."""

    envelope = _load_envelope(path)
    if envelope.config.schema_version not in {7, 8, 9, 10, 11, 12}:
        raise AnalysisArtifactError(
            "Phase 8 requires config schema 7-12: "
            f"{envelope.path}"
        )
    metrics_path = envelope.path / "phase8_metrics.json"
    metrics = _require_mapping(_read_json(metrics_path), str(metrics_path))
    _require_keys(
        metrics,
        {
            "phase8_metric_schema_version",
            "run_kind",
            "methods",
            "policy",
            "unified_pi_contract",
            "post_optimization_scaling",
            "oracle_path_hash",
            "condition_steps",
            "path_hashes",
        },
        str(metrics_path),
    )
    metric_schema = metrics["phase8_metric_schema_version"]
    if metric_schema not in PHASE8_METRIC_SCHEMA_VERSIONS:
        raise AnalysisArtifactError(
            f"unsupported Phase 8 metric schema in {metrics_path}: "
            f"{metrics['phase8_metric_schema_version']}"
        )
    if metrics["run_kind"] != "unified_controller":
        raise AnalysisArtifactError(f"invalid Phase 8 run kind in {metrics_path}")
    if metrics["unified_pi_contract"] is not True:
        raise AnalysisArtifactError(f"Phase 8 run is not unified: {metrics_path}")
    if metrics["post_optimization_scaling"] is not False:
        raise AnalysisArtifactError(
            f"Phase 8 run applied forbidden post-scaling: {metrics_path}"
        )
    if metric_schema >= 3:
        _require_keys(
            metrics,
            {"fisher_inverse_used", "trace_estimator"},
            str(metrics_path),
        )
        if metrics["fisher_inverse_used"] is not False:
            raise AnalysisArtifactError(
                f"Phase 8 run used a forbidden Fisher inverse: {metrics_path}"
            )
        if metrics["trace_estimator"] != (
            "accepted_displacement_residual_moments"
        ):
            raise AnalysisArtifactError(
                f"unsupported Phase 8 trace estimator in {metrics_path}"
            )
    if metric_schema >= 4:
        _require_keys(
            metrics,
            {
                "oracle_convergence_contract",
                "residual_dependence",
                "references",
            },
            str(metrics_path),
        )
    if metric_schema >= 5:
        semantic_contract = {
            "controller_displacement_law": "u_approx_pi_times_drift_plus_noise",
            "trend_estimand": "environmental_parameter_displacement",
            "trend_observation": "accepted_displacement_divided_by_pi",
            "covariance_residual": "u_minus_pi_times_predictable_drift",
            "oracle_covariance_residual": "u_minus_pi_times_oracle_drift",
            "ewc_optimality_diagnostic": "final_objective_gradient",
        }
        _require_keys(metrics, semantic_contract, str(metrics_path))
        mismatched = {
            name: metrics[name]
            for name, expected in semantic_contract.items()
            if metrics[name] != expected
        }
        if mismatched:
            raise AnalysisArtifactError(
                f"Phase 8 corrected estimator contract is invalid: {mismatched}"
            )
        _require_keys(
            metrics,
            {"oracle_path_provenance"},
            str(metrics_path),
        )
    if metric_schema >= 6:
        optimizer_contract = {
            "optimizer_budget_role": "fixed_compute_budget_not_convergence_claim",
            "optimizer_accounting": (
                "structured_iterations_and_function_evaluations"
            ),
        }
        _require_keys(metrics, optimizer_contract, str(metrics_path))
        mismatched = {
            name: metrics[name]
            for name, expected in optimizer_contract.items()
            if metrics[name] != expected
        }
        if mismatched:
            raise AnalysisArtifactError(
                f"Phase 8 optimizer-budget contract is invalid: {mismatched}"
            )
    if metric_schema >= 7:
        _require_keys(
            metrics,
            {"exposure_accounting", "calibration_contract"},
            str(metrics_path),
        )
        if metrics["exposure_accounting"] != (
            "ordered_stream_batches_with_optimizer_consumption"
        ):
            raise AnalysisArtifactError(
                "Phase 8 exposure-accounting contract is invalid"
            )
        calibration_contract = _require_mapping(
            metrics["calibration_contract"],
            f"{metrics_path}:calibration_contract",
        )
        if calibration_contract != {
            "brier": "mean_multiclass_probability_squared_error",
            "expected_calibration_error_bins": 15,
            "expected_calibration_error_binning": (
                "equal_width_maximum_probability"
            ),
        }:
            raise AnalysisArtifactError(
                "Phase 8 calibration contract is invalid"
            )
    if metric_schema >= 8:
        classification_contract = {
            "nine_accuracy_legacy_semantics": (
                "recall_conditioned_on_true_nine"
            ),
            "nine_recall": "true_positive_rate_conditioned_on_true_nine",
            "nine_false_positive_rate": (
                "predicted_nine_conditioned_on_true_non_nine"
            ),
            "nine_ovr_accuracy": "prevalence_adjusted_at_row_p",
            "nine_precision": (
                "prevalence_adjusted_at_row_p_null_when_undefined"
            ),
            "environment_accuracy": "multiclass_prevalence_adjusted_at_row_p",
        }
        _require_keys(metrics, {"classification_contract"}, str(metrics_path))
        if metrics["classification_contract"] != classification_contract:
            raise AnalysisArtifactError(
                "Phase 8 classification contract is invalid"
            )

    trajectory_path = envelope.path / "phase8_trajectories.pt"
    controller_path = envelope.path / "phase8_controller_states.pt"
    oracle_path = envelope.path / "phase8_reference_optimum.pt"
    checkpoint_path = envelope.path / "phase8_checkpoints.pt"
    for required_path in (
        trajectory_path,
        controller_path,
        oracle_path,
        checkpoint_path,
    ):
        if not required_path.is_file():
            raise AnalysisArtifactError(
                f"Phase 8 run is missing required artifact: {required_path}"
            )

    trajectory = _read_torch_mapping(trajectory_path)
    controller = _read_torch_mapping(controller_path)
    oracle = _read_torch_mapping(oracle_path)
    if trajectory.get("schema_version") != (
        PHASE8_TRAJECTORY_SCHEMA_BY_METRIC[metric_schema]
    ):
        raise AnalysisArtifactError(
            f"unsupported Phase 8 trajectory schema in {trajectory_path}"
        )
    if controller.get("schema_version") != (
        PHASE8_CONTROLLER_STATE_SCHEMA_BY_METRIC[metric_schema]
    ):
        raise AnalysisArtifactError(
            f"unsupported Phase 8 controller-state schema in {controller_path}"
        )
    if oracle.get("schema_version") != (
        PHASE8_ORACLE_PATH_SCHEMA_BY_METRIC[metric_schema]
    ):
        raise AnalysisArtifactError(
            f"unsupported Phase 8 oracle-path schema in {oracle_path}"
        )
    methods = tuple(str(method) for method in metrics["methods"])
    if set(trajectory.get("conditions", {})) != set(methods):
        raise AnalysisArtifactError("Phase 8 trajectory methods do not match metrics")
    if set(controller.get("conditions", {})) != set(methods):
        raise AnalysisArtifactError(
            "Phase 8 controller-state methods do not match metrics"
        )
    oracle_body = _require_mapping(oracle.get("path"), f"{oracle_path}:path")
    if oracle_body.get("content_hash") != metrics["oracle_path_hash"]:
        raise AnalysisArtifactError("Phase 8 oracle path hash does not match metrics")
    p_values = tuple(float(value) for value in trajectory.get("p_values", ()))
    if not p_values:
        raise AnalysisArtifactError("Phase 8 trajectory has no p values")
    expected_rows = len(p_values) * len(methods)
    if len(metrics["condition_steps"]) != expected_rows:
        raise AnalysisArtifactError("Phase 8 scalar row count is incomplete")
    if any(row.get("same_pi_consumed") is not True for row in metrics["condition_steps"]):
        raise AnalysisArtifactError("Phase 8 run violated the unified pi contract")

    observation_indices = trajectory.get("observation_indices")
    class_labels = trajectory.get("class_labels")
    if observation_indices is None and class_labels is None:
        exposures: tuple[Mapping[str, Any], ...] = ()
    elif observation_indices is None or class_labels is None:
        raise AnalysisArtifactError(
            "Phase 8 trajectory has incomplete stream-plan observations"
        )
    else:
        try:
            exposures = stream_exposure_rows(observation_indices, class_labels)
        except (TypeError, ValueError) as exc:
            raise AnalysisArtifactError(
                f"invalid Phase 8 stream exposure data: {exc}"
            ) from exc
        if len(exposures) != len(p_values):
            raise AnalysisArtifactError(
                "Phase 8 exposure rows do not align with the p grid"
            )

    if metric_schema >= 6:
        for row in metrics["condition_steps"]:
            proposal = row.get("proposal")
            if proposal is None:
                continue
            proposal = _require_mapping(
                proposal, f"{metrics_path}:proposal step {row.get('step')}"
            )
            _require_keys(
                proposal,
                {
                    "inner_steps",
                    "optimizer_iterations",
                    "optimizer_function_evaluations",
                    "stopping_reason",
                },
                f"{metrics_path}:proposal step {row.get('step')}",
            )
            for name in (
                "inner_steps",
                "optimizer_iterations",
                "optimizer_function_evaluations",
            ):
                value = proposal[name]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise AnalysisArtifactError(
                        f"invalid Phase 8 optimizer accounting {name}={value}"
                    )
    if metric_schema >= 7:
        if not exposures:
            raise AnalysisArtifactError(
                "Phase 8 metric schema 7 requires stream exposure data"
            )
        exposure_by_step = {int(row["step"]): row for row in exposures}
        calibration_fields = {
            "before_brier",
            "before_non_nine_brier",
            "before_nine_brier",
            "before_expected_calibration_error",
            "before_calibration_bin_count",
            "after_brier",
            "after_non_nine_brier",
            "after_nine_brier",
            "after_expected_calibration_error",
            "after_calibration_bin_count",
        }
        for row in metrics["condition_steps"]:
            step = int(row["step"])
            expected_exposure = exposure_by_step.get(step)
            if expected_exposure is None:
                raise AnalysisArtifactError(
                    f"Phase 8 scalar row has unknown exposure step {step}"
                )
            _require_keys(
                row,
                set(EXPOSURE_FIELDS) | calibration_fields,
                f"{metrics_path}:condition step {step}",
            )
            mismatched_exposure = {
                name: (row[name], expected_exposure[name])
                for name in EXPOSURE_FIELDS
                if row[name] != expected_exposure[name]
            }
            if mismatched_exposure:
                raise AnalysisArtifactError(
                    "Phase 8 scalar exposure data does not match its stream: "
                    f"{mismatched_exposure}"
                )
    if metric_schema >= 8:
        classification_names = {
            "nine_metric_prevalence",
            "nine_true_positive_count",
            "nine_false_positive_count",
            "nine_true_negative_count",
            "nine_false_negative_count",
            "nine_recall",
            "nine_false_positive_rate",
            "nine_specificity",
            "nine_ovr_accuracy",
            "nine_precision",
            "environment_accuracy",
        }
        for row in metrics["condition_steps"]:
            for stage in ("before", "after"):
                _require_keys(
                    row,
                    {f"{stage}_{name}" for name in classification_names},
                    f"{metrics_path}:condition step {row.get('step')}",
                )
                prevalence = float(row[f"{stage}_nine_metric_prevalence"])
                if prevalence != float(row["p"]):
                    raise AnalysisArtifactError(
                        "Phase 8 classification prevalence does not match p"
                    )
                recall = float(row[f"{stage}_nine_recall"])
                if recall != float(row[f"{stage}_nine_accuracy"]):
                    raise AnalysisArtifactError(
                        "Phase 8 legacy nine accuracy is not nine recall"
                    )
    return Phase8ControllerRun(
        path=envelope.path,
        run_id=envelope.run_id,
        config=envelope.config,
        manifest=envelope.manifest,
        metrics=metrics,
        p_values=p_values,
        methods=methods,
        exposures=exposures,
    )


def discover_phase8_controller_runs(
    root: str | Path,
) -> tuple[Phase8ControllerRun, ...]:
    run_root = Path(root)
    if not run_root.is_dir():
        raise AnalysisArtifactError(f"Phase 8 run root does not exist: {run_root}")
    runs = []
    for path in sorted(run_root.iterdir()):
        if (
            path.is_dir()
            and not path.name.startswith(".")
            and (path / "phase8_metrics.json").is_file()
        ):
            runs.append(load_phase8_controller_run(path))
    if not runs:
        raise AnalysisArtifactError(
            f"no completed Phase 8 controller runs were found in {run_root}"
        )
    return tuple(runs)


def _environment_metric(
    source: Mapping[str, Any],
    metric: str,
    *,
    stage: str = "before",
) -> float | None:
    nine_key = f"{stage}_nine_{metric}"
    non_nine_key = f"{stage}_non_nine_{metric}"
    if nine_key not in source or non_nine_key not in source:
        return None
    p = float(source["p"])
    return (1.0 - p) * float(source[non_nine_key]) + p * float(
        source[nine_key]
    )


def phase8_controller_rows(
    runs: Sequence[Phase8ControllerRun],
    *,
    classification_backfill_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        run_rows = []
        exposure_by_step = {
            int(row["step"]): row for row in run.exposures
        }
        for source in run.metrics["condition_steps"]:
            exposure = exposure_by_step.get(int(source["step"]), {})
            run_rows.append(
                {
                    **source,
                    **exposure,
                    "run_id": run.run_id,
                    "run_path": str(run.path),
                    "experiment": run.config.experiment,
                    "replica_id": run.config.replica_id,
                    "samples_per_step": run.config.data.samples_per_step,
                    "policy": run.metrics["policy"],
                    "metric_schema": run.metrics[
                        "phase8_metric_schema_version"
                    ],
                    "trajectory_hash": run.metrics["path_hashes"][
                        source["method"]
                    ],
                    "environment_accuracy": _environment_metric(
                        source, "accuracy"
                    ),
                    "environment_nll": _environment_metric(source, "nll"),
                    "environment_brier": _environment_metric(
                        source, "brier"
                    ),
                    "after_environment_accuracy": _environment_metric(
                        source, "accuracy", stage="after"
                    ),
                    "after_environment_nll": _environment_metric(
                        source, "nll", stage="after"
                    ),
                    "after_environment_brier": _environment_metric(
                        source, "brier", stage="after"
                    ),
                    "theoretical_status": (
                        "legacy_spectral_trace_smoke"
                        if run.metrics["phase8_metric_schema_version"] == 2
                        else "corrected_normalized_drift_phase8"
                        if run.metrics["phase8_metric_schema_version"] >= 5
                        else "legacy_attenuated_trend_phase8"
                    ),
                }
            )
        if run.metrics["phase8_metric_schema_version"] >= 8:
            run_rows = [
                {**row, "classification_metric_source": "runtime_schema8"}
                for row in run_rows
            ]
        elif classification_backfill_root is not None:
            run_rows = overlay_classification_backfill(
                run_rows,
                run.path,
                classification_backfill_root,
            )
        rows.extend(run_rows)
    return rows


def _normalized_trapezoid_auc(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> float | None:
    points = sorted(
        (
            (float(row["p"]), float(row[key]))
            for row in rows
            if row.get(key) is not None
        ),
        key=lambda point: point[0],
    )
    if not points:
        return None
    if len(points) == 1:
        return points[0][1]
    span = points[-1][0] - points[0][0]
    if span <= 0.0:
        return None
    area = sum(
        0.5 * (left[1] + right[1]) * (right[0] - left[0])
        for left, right in zip(points, points[1:])
    )
    return area / span


def phase8_controller_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_id"], row["method"])].append(row)
    summaries = []
    for condition_rows in grouped.values():
        ordered = sorted(condition_rows, key=lambda row: row["step"])
        first = ordered[0]
        final = ordered[-1]
        controller_rows = [row["controller"] for row in ordered]
        trace_error_name = (
            "trace_relative_error_to_oracle_residual"
            if "trace_relative_error_to_oracle_residual" in first
            else "trace_relative_error_to_oracle"
        )
        trace_errors = [
            row[trace_error_name]
            for row in ordered
            if row.get(trace_error_name) is not None
        ]
        proposal_rows = [
            row["proposal"]
            for row in ordered
            if isinstance(row.get("proposal"), Mapping)
            and row["proposal"].get("optimization_guard") is not None
        ]
        relative_gradient_norms = [
            float(row["relative_final_gradient_norm"])
            for row in proposal_rows
            if row.get("relative_final_gradient_norm") is not None
        ]
        final_gradient_rms = [
            float(row["final_gradient_rms"])
            for row in proposal_rows
            if row.get("final_gradient_rms") is not None
        ]
        optimizer_iterations = [
            int(row["optimizer_iterations"])
            for row in proposal_rows
            if row.get("optimizer_iterations") is not None
        ]
        optimizer_evaluations = [
            int(row["optimizer_function_evaluations"])
            for row in proposal_rows
            if row.get("optimizer_function_evaluations") is not None
        ]
        summaries.append(
            {
                "run_id": first["run_id"],
                "policy": first["policy"],
                "method": first["method"],
                "mean_pi": sum(float(row["applied_pi"]) for row in controller_rows)
                / len(controller_rows),
                "lower_bound_frequency": sum(
                    bool(row["lower_bound_active"]) for row in controller_rows
                )
                / len(controller_rows),
                "upper_bound_frequency": sum(
                    bool(row["upper_bound_active"]) for row in controller_rows
                )
                / len(controller_rows),
                "mean_fisher_error": sum(
                    float(row["relative_frobenius_error"])
                    for row in ordered[1:] or ordered
                )
                / len(ordered[1:] or ordered),
                "mean_trace_relative_error": (
                    None
                    if not trace_errors
                    else sum(float(value) for value in trace_errors)
                    / len(trace_errors)
                ),
                "backtracking_rejections": (
                    None
                    if not proposal_rows
                    else sum(
                        int(row["backtracking_rejections"])
                        for row in proposal_rows
                    )
                ),
                "maximum_backtracks": (
                    None
                    if not proposal_rows
                    else max(
                        int(row["maximum_backtracks"])
                        for row in proposal_rows
                    )
                ),
                "minimum_learning_rate": (
                    None
                    if not proposal_rows
                    else min(
                        float(row["minimum_learning_rate"])
                        for row in proposal_rows
                    )
                ),
                "mean_optimizer_iterations": (
                    None
                    if not optimizer_iterations
                    else sum(optimizer_iterations) / len(optimizer_iterations)
                ),
                "maximum_optimizer_iterations": (
                    None if not optimizer_iterations else max(optimizer_iterations)
                ),
                "mean_optimizer_function_evaluations": (
                    None
                    if not optimizer_evaluations
                    else sum(optimizer_evaluations) / len(optimizer_evaluations)
                ),
                "maximum_optimizer_function_evaluations": (
                    None if not optimizer_evaluations else max(optimizer_evaluations)
                ),
                "mean_relative_final_gradient_norm": (
                    None
                    if not relative_gradient_norms
                    else sum(relative_gradient_norms)
                    / len(relative_gradient_norms)
                ),
                "maximum_relative_final_gradient_norm": (
                    None
                    if not relative_gradient_norms
                    else max(relative_gradient_norms)
                ),
                "mean_final_gradient_rms": (
                    None
                    if not final_gradient_rms
                    else sum(final_gradient_rms) / len(final_gradient_rms)
                ),
                "environment_accuracy_auc": _normalized_trapezoid_auc(
                    ordered, "environment_accuracy"
                ),
                "environment_nll_auc": _normalized_trapezoid_auc(
                    ordered, "environment_nll"
                ),
                "nine_accuracy_auc": _normalized_trapezoid_auc(
                    ordered, "before_nine_accuracy"
                ),
                "nine_nll_auc": _normalized_trapezoid_auc(
                    ordered, "before_nine_nll"
                ),
                "non_nine_accuracy_auc": _normalized_trapezoid_auc(
                    ordered, "before_non_nine_accuracy"
                ),
                "non_nine_nll_auc": _normalized_trapezoid_auc(
                    ordered, "before_non_nine_nll"
                ),
                "final_environment_accuracy": final.get(
                    "environment_accuracy"
                ),
                "final_environment_nll": final.get("environment_nll"),
                "final_nine_accuracy": final.get("before_nine_accuracy"),
                "final_nine_nll": final.get("before_nine_nll"),
                "final_non_nine_accuracy": final.get(
                    "before_non_nine_accuracy"
                ),
                "final_non_nine_nll": final.get("before_non_nine_nll"),
                "final_balanced_accuracy": float(final["before_balanced_accuracy"]),
                "final_parameter_squared_error_to_oracle": float(
                    final["parameter_squared_error_to_oracle"]
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (row["run_id"], row["method"]),
    )


PHASE9_PAIRED_METRICS = (
    "environment_accuracy_auc",
    "environment_nll_auc",
    "nine_accuracy_auc",
    "nine_nll_auc",
    "non_nine_accuracy_auc",
    "non_nine_nll_auc",
    "final_nine_accuracy",
    "final_nine_nll",
    "final_non_nine_accuracy",
    "final_non_nine_nll",
    "final_balanced_accuracy",
    "mean_fisher_error",
)
PHASE9_LOWER_IS_BETTER = frozenset(
    metric
    for metric in PHASE9_PAIRED_METRICS
    if "nll" in metric or "error" in metric
)
_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _deduplicated_phase9_entries(
    bundles: Sequence[Phase9Bundle],
) -> list[dict[str, Any]]:
    by_run_id: dict[str, dict[str, Any]] = {}
    for bundle in bundles:
        for source in bundle.entries:
            entry = dict(source)
            run_id = str(entry["run_id"])
            existing = by_run_id.get(run_id)
            if existing is None:
                entry["bundle_ids"] = [bundle.bundle_id]
                by_run_id[run_id] = entry
                continue
            comparable = {
                key: value
                for key, value in existing.items()
                if key != "bundle_ids"
            }
            if comparable != entry:
                raise AnalysisArtifactError(
                    f"Phase 9 run {run_id} has conflicting bundle metadata"
                )
            existing["bundle_ids"].append(bundle.bundle_id)
    return sorted(
        by_run_id.values(),
        key=lambda row: (
            row["replica_index"],
            row["profile"],
            row["kind"],
            row["cell"],
        ),
    )


def phase9_inventory_rows(
    bundles: Sequence[Phase9Bundle],
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    status_by_run: dict[str, dict[str, Any]] = {}
    for bundle in bundles:
        for status in phase9_status_rows(bundle, repo_root):
            run_id = status["run_id"]
            existing = status_by_run.get(run_id)
            if existing is not None and (
                existing["run_state"] != status["run_state"]
                or existing["initialization_state"]
                != status["initialization_state"]
            ):
                raise AnalysisArtifactError(
                    f"Phase 9 run {run_id} has conflicting artifact states"
                )
            status_by_run[run_id] = status
    rows = []
    for entry in _deduplicated_phase9_entries(bundles):
        status = status_by_run[entry["run_id"]]
        rows.append(
            {
                **entry,
                "bundle_count": len(entry["bundle_ids"]),
                "run_state": status["run_state"],
                "initialization_state": status["initialization_state"],
                "run_path": status["run_path"],
                "config_path": status["config_path"],
            }
        )
    return rows


def phase9_replica_rows(
    bundles: Sequence[Phase9Bundle],
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    rows = []
    for entry in phase9_inventory_rows(bundles, root):
        if entry["run_state"] != "completed":
            continue
        run = load_phase8_controller_run(entry["run_path"])
        summaries = phase8_controller_summaries(phase8_controller_rows([run]))
        for summary in summaries:
            rows.append(
                {
                    **summary,
                    "profile": entry["profile"],
                    "cell": entry["cell"],
                    "kind": entry["kind"],
                    "replica_index": entry["replica_index"],
                    "replica_id": entry["replica_id"],
                    "replica_seed": entry["replica_seed"],
                    "replica_bundle_id": entry["replica_bundle_id"],
                    "control_run_id": entry["control_run_id"],
                    "factors": entry["factors"],
                    "bundle_ids": entry["bundle_ids"],
                    "run_path": entry["run_path"],
                    "total_elapsed_seconds": float(
                        run.metrics["total_elapsed_seconds"]
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["replica_index"],
            row["profile"],
            row["kind"],
            row["cell"],
            row["method"],
        ),
    )


def phase9_paired_rows(
    replica_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_run_method = {
        (row["run_id"], row["method"]): row for row in replica_rows
    }
    pairs = []
    for treatment in replica_rows:
        control_run_id = treatment.get("control_run_id")
        if treatment.get("kind") != "treatment" or control_run_id is None:
            continue
        control = by_run_method.get((control_run_id, treatment["method"]))
        if control is None:
            continue
        row = {
            "profile": treatment["profile"],
            "cell": treatment["cell"],
            "method": treatment["method"],
            "replica_index": treatment["replica_index"],
            "replica_id": treatment["replica_id"],
            "replica_seed": treatment["replica_seed"],
            "replica_bundle_id": treatment["replica_bundle_id"],
            "treatment_run_id": treatment["run_id"],
            "control_run_id": control["run_id"],
            "treatment_run_path": treatment.get("run_path"),
            "control_run_path": control.get("run_path"),
            "factors": treatment["factors"],
        }
        for metric in PHASE9_PAIRED_METRICS:
            treatment_value = treatment.get(metric)
            control_value = control.get(metric)
            row[f"treatment_{metric}"] = treatment_value
            row[f"control_{metric}"] = control_value
            row[f"delta_{metric}"] = (
                None
                if treatment_value is None or control_value is None
                else float(treatment_value) - float(control_value)
            )
        pairs.append(row)
    return sorted(
        pairs,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["replica_index"],
        ),
    )


def phase9_paired_aggregates(
    paired_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped[(row["profile"], row["cell"], row["method"])].append(row)
    aggregates = []
    for (profile, cell, method), group in grouped.items():
        for metric in PHASE9_PAIRED_METRICS:
            values = [
                float(row[f"delta_{metric}"])
                for row in group
                if row.get(f"delta_{metric}") is not None
            ]
            if not values:
                continue
            count = len(values)
            mean = sum(values) / count
            if count < 2:
                standard_error = ci_low = ci_high = None
            else:
                variance = sum((value - mean) ** 2 for value in values) / (
                    count - 1
                )
                standard_error = math.sqrt(variance / count)
                critical = _T_CRITICAL_975.get(count - 1, 1.96)
                half_width = critical * standard_error
                ci_low = mean - half_width
                ci_high = mean + half_width
            aggregates.append(
                {
                    "profile": profile,
                    "cell": cell,
                    "method": method,
                    "metric": metric,
                    "lower_is_better": metric in PHASE9_LOWER_IS_BETTER,
                    "replica_count": count,
                    "mean_paired_effect": mean,
                    "standard_error": standard_error,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "initial_replica_target_met": count >= 5,
                    "treatment_run_ids": [
                        row["treatment_run_id"] for row in group
                    ],
                    "control_run_ids": [row["control_run_id"] for row in group],
                }
            )
    return sorted(
        aggregates,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["metric"],
        ),
    )


def phase9_progress_rows(
    inventory_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    control_states = {
        row["run_id"]: row["run_state"]
        for row in inventory_rows
    }
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in inventory_rows:
        if row["kind"] == "treatment":
            grouped[(row["profile"], row["cell"])].append(row)
    rows = []
    for (profile, cell), entries in grouped.items():
        completed_treatments = sum(
            entry["run_state"] == "completed" for entry in entries
        )
        completed_pairs = sum(
            entry["run_state"] == "completed"
            and control_states.get(entry["control_run_id"]) == "completed"
            for entry in entries
        )
        rows.append(
            {
                "profile": profile,
                "cell": cell,
                "expected_replicas": len(entries),
                "completed_treatments": completed_treatments,
                "completed_pairs": completed_pairs,
                "initial_target": 5,
                "initial_target_met": completed_pairs >= 5,
            }
        )
    return sorted(rows, key=lambda row: (row["profile"], row["cell"]))

PLAN2_TRAJECTORY_METRICS = (
    "after_nine_ovr_accuracy",
    "after_nine_precision",
    "after_nine_recall",
    "after_environment_accuracy",
    "after_non_nine_accuracy",
    "after_nine_nll",
    "after_non_nine_nll",
    "after_environment_nll",
    "after_brier",
    "after_non_nine_brier",
    "after_nine_brier",
    "after_environment_brier",
    "after_expected_calibration_error",
)
PLAN2_PRIMARY_PREDICTIVE_METRICS = (
    "after_nine_ovr_accuracy",
    "after_nine_precision",
    "after_nine_recall",
    "after_environment_accuracy",
)
PLAN2_EXPOSURE_COORDINATES = (
    "after_cumulative_observations",
    "after_cumulative_nine_observations",
    "after_cumulative_non_nine_observations",
    "after_cumulative_unique_observations",
    "after_cumulative_unique_nine_observations",
    "after_cumulative_unique_non_nine_observations",
)


def phase9_trajectory_rows(
    bundles: Sequence[Phase9Bundle],
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    """Load completed Phase 9 scalar trajectories with exposure provenance."""

    root = Path(repo_root)
    rows = []
    for entry in phase9_inventory_rows(bundles, root):
        if entry["run_state"] != "completed":
            continue
        run = load_phase8_controller_run(entry["run_path"])
        for source in phase8_controller_rows(
            [run],
            classification_backfill_root=default_classification_backfill_root(
                root
            ),
        ):
            rows.append(
                {
                    **source,
                    "profile": entry["profile"],
                    "cell": entry["cell"],
                    "kind": entry["kind"],
                    "replica_index": entry["replica_index"],
                    "replica_seed": entry["replica_seed"],
                    "replica_bundle_id": entry["replica_bundle_id"],
                    "control_run_id": entry["control_run_id"],
                    "factors": entry["factors"],
                    "bundle_ids": entry["bundle_ids"],
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["replica_index"],
            row["step"],
        ),
    )


def _sample_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    count = len(values)
    if count == 0:
        raise ValueError("sample interval requires at least one value")
    mean = sum(values) / count
    if count < 2:
        standard_deviation = standard_error = ci_low = ci_high = None
    else:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        standard_deviation = math.sqrt(variance)
        standard_error = standard_deviation / math.sqrt(count)
        critical = _T_CRITICAL_975.get(count - 1, 1.96)
        half_width = critical * standard_error
        ci_low = mean - half_width
        ci_high = mean + half_width
    return {
        "replica_count": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def phase9_expected_trajectory_rows(
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = PLAN2_TRAJECTORY_METRICS,
) -> list[dict[str, Any]]:
    """Aggregate pointwise expected trajectories with replicas as units."""

    grouped: dict[
        tuple[str, str, str, int, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in trajectory_rows:
        grouped[
            (
                str(row["profile"]),
                str(row["cell"]),
                str(row["method"]),
                int(row["samples_per_step"]),
                int(row["step"]),
            )
        ].append(row)

    results = []
    for (profile, cell, method, samples_per_step, step), group in grouped.items():
        replica_ids = [str(row["replica_id"]) for row in group]
        if len(replica_ids) != len(set(replica_ids)):
            raise AnalysisArtifactError(
                "duplicate replica trajectory point for "
                f"{profile}:{cell}:m={samples_per_step}:{step}"
            )
        p_values = {float(row["p"]) for row in group}
        if len(p_values) != 1:
            raise AnalysisArtifactError(
                f"unaligned p values for {profile}:{cell}:m={samples_per_step}:{step}"
            )
        coordinate_means = {
            f"mean_{name}": (
                None
                if not any(row.get(name) is not None for row in group)
                else sum(
                    float(row[name])
                    for row in group
                    if row.get(name) is not None
                )
                / sum(row.get(name) is not None for row in group)
            )
            for name in PLAN2_EXPOSURE_COORDINATES
        }
        for metric in metrics:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
            ]
            if not values:
                continue
            interval = _sample_interval(values)
            results.append(
                {
                    "profile": profile,
                    "cell": cell,
                    "method": method,
                    "step": step,
                    "p": next(iter(p_values)),
                    "samples_per_step": samples_per_step,
                    "metric": metric,
                    "completed_replica_count": len(group),
                    "metric_complete": len(values) == len(group),
                    **coordinate_means,
                    **interval,
                }
            )
    return sorted(
        results,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["samples_per_step"],
            row["metric"],
            row["step"],
        ),
    )


def phase9_durable_crossing_rows(
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str = "after_nine_ovr_accuracy",
    threshold: float = 0.90,
) -> list[dict[str, Any]]:
    """Find the first expected-trajectory point that stays above a threshold."""

    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    grouped: dict[
        tuple[str, str, str, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in expected_rows:
        if row.get("metric") == metric:
            grouped[
                (
                    str(row["profile"]),
                    str(row["cell"]),
                    str(row["method"]),
                    int(row["samples_per_step"]),
                )
            ].append(row)

    results = []
    for (profile, cell, method, samples_per_step), group in grouped.items():
        ordered = sorted(group, key=lambda row: int(row["step"]))
        crossing = next(
            (
                row
                for index, row in enumerate(ordered)
                if float(row["mean"]) >= threshold
                and all(
                    float(later["mean"]) >= threshold
                    for later in ordered[index:]
                )
            ),
            None,
        )
        base = {
            "profile": profile,
            "cell": cell,
            "method": method,
            "samples_per_step": samples_per_step,
            "metric": metric,
            "threshold": threshold,
            "crossed": crossing is not None,
        }
        if crossing is None:
            results.append(
                {
                    **base,
                    "step": None,
                    "p": None,
                    "mean": None,
                    "standard_deviation": None,
                    "replica_count": min(
                        int(row["replica_count"]) for row in ordered
                    ),
                    **{
                        f"mean_{name}": None
                        for name in PLAN2_EXPOSURE_COORDINATES
                    },
                }
            )
        else:
            results.append(
                {
                    **base,
                    "step": crossing["step"],
                    "p": crossing["p"],
                    "mean": crossing["mean"],
                    "standard_deviation": crossing["standard_deviation"],
                    "replica_count": crossing["replica_count"],
                    **{
                        f"mean_{name}": crossing.get(f"mean_{name}")
                        for name in PLAN2_EXPOSURE_COORDINATES
                    },
                }
            )
    return sorted(
        results,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["samples_per_step"],
        ),
    )


def phase9_joint_durable_crossing_rows(
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Find the first point after which every requested mean stays acceptable."""

    requested = dict(thresholds)
    if not requested:
        raise ValueError("thresholds must be nonempty")
    for metric, threshold in requested.items():
        if not metric or not math.isfinite(threshold):
            raise ValueError("threshold metrics and values must be finite")

    grouped: dict[
        tuple[str, str, str, int], dict[int, dict[str, Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in expected_rows:
        metric = str(row.get("metric"))
        if metric not in requested:
            continue
        key = (
            str(row["profile"]),
            str(row["cell"]),
            str(row["method"]),
            int(row["samples_per_step"]),
        )
        step = int(row["step"])
        if metric in grouped[key][step]:
            raise AnalysisArtifactError(
                f"duplicate expected metric at {key}:step={step}:{metric}"
            )
        grouped[key][step][metric] = row

    results = []
    for (profile, cell, method, samples_per_step), by_step in grouped.items():
        ordered_steps = sorted(by_step)

        def acceptable(step: int) -> bool:
            rows = by_step[step]
            return all(
                metric in rows
                and bool(rows[metric].get("metric_complete"))
                and float(rows[metric]["mean"]) >= threshold
                for metric, threshold in requested.items()
            )

        crossing_step = next(
            (
                step
                for index, step in enumerate(ordered_steps)
                if acceptable(step)
                and all(acceptable(later) for later in ordered_steps[index:])
            ),
            None,
        )
        base = {
            "profile": profile,
            "cell": cell,
            "method": method,
            "samples_per_step": samples_per_step,
            "thresholds": requested,
            "crossed": crossing_step is not None,
        }
        if crossing_step is None:
            results.append(
                {
                    **base,
                    "step": None,
                    "p": None,
                    "minimum_replica_count": min(
                        int(row["replica_count"])
                        for rows in by_step.values()
                        for row in rows.values()
                    ),
                    **{
                        f"mean_{name}": None
                        for name in PLAN2_EXPOSURE_COORDINATES
                    },
                }
            )
            continue
        selected = by_step[crossing_step]
        representative = selected[next(iter(requested))]
        results.append(
            {
                **base,
                "step": crossing_step,
                "p": representative["p"],
                "minimum_replica_count": min(
                    int(row["replica_count"]) for row in selected.values()
                ),
                **{
                    f"mean_{name}": representative.get(f"mean_{name}")
                    for name in PLAN2_EXPOSURE_COORDINATES
                },
            }
        )
    return sorted(
        results,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["samples_per_step"],
        ),
    )


def phase9_equivalent_data_multiplier_rows(
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_cell: str,
    treatment_cells: Sequence[str],
    metrics: Sequence[str] = PLAN2_PRIMARY_PREDICTIVE_METRICS,
    practical_margins: Mapping[str, float] | None = None,
    exposure_key: str = "mean_after_cumulative_unique_nine_observations",
) -> list[dict[str, Any]]:
    """Bracket baseline exposure needed to match each treatment trajectory."""

    requested = tuple(metrics)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("metrics must be unique and nonempty")
    treatments = tuple(treatment_cells)
    if not treatments or len(treatments) != len(set(treatments)):
        raise ValueError("treatment_cells must be unique and nonempty")
    margins = {metric: 0.0 for metric in requested}
    if practical_margins is not None:
        unknown = set(practical_margins) - set(requested)
        if unknown:
            raise ValueError(f"practical margins contain unknown metrics: {unknown}")
        margins.update(
            {metric: float(value) for metric, value in practical_margins.items()}
        )
    if any(not math.isfinite(value) or value < 0.0 for value in margins.values()):
        raise ValueError("practical margins must be finite and nonnegative")

    indexed: dict[
        tuple[str, str, str, int, int], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in expected_rows:
        metric = str(row.get("metric"))
        if metric not in requested:
            continue
        key = (
            str(row["profile"]),
            str(row["cell"]),
            str(row["method"]),
            int(row["samples_per_step"]),
            int(row["step"]),
        )
        if metric in indexed[key]:
            raise AnalysisArtifactError(f"duplicate expected metric at {key}:{metric}")
        indexed[key][metric] = row

    def complete_metric_set(rows: Mapping[str, Mapping[str, Any]]) -> bool:
        return all(
            metric in rows
            and bool(rows[metric].get("metric_complete"))
            and rows[metric].get(exposure_key) is not None
            for metric in requested
        )

    baseline_by_step: dict[
        tuple[str, str, int], list[tuple[int, dict[str, Mapping[str, Any]]]]
    ] = defaultdict(list)
    for key, rows in indexed.items():
        profile, cell, method, samples_per_step, step = key
        if cell == baseline_cell and complete_metric_set(rows):
            baseline_by_step[(profile, method, step)].append(
                (samples_per_step, rows)
            )

    results = []
    for key, target_rows in indexed.items():
        profile, cell, method, target_m, step = key
        if cell not in treatments or not complete_metric_set(target_rows):
            continue
        representative = target_rows[requested[0]]
        target_exposure = float(representative[exposure_key])
        if target_exposure <= 0.0 or not math.isfinite(target_exposure):
            continue
        p = float(representative["p"])
        candidates = []
        for candidate_m, candidate_rows in baseline_by_step.get(
            (profile, method, step), ()
        ):
            candidate_rep = candidate_rows[requested[0]]
            if not math.isclose(float(candidate_rep["p"]), p, abs_tol=1e-12):
                raise AnalysisArtifactError(
                    f"unaligned EDM p values for {profile}:{method}:step={step}"
                )
            exposure = float(candidate_rep[exposure_key])
            if exposure < 0.0 or not math.isfinite(exposure):
                continue
            candidates.append((exposure, candidate_m, candidate_rows))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if not candidates:
            continue

        comparisons = [*requested, "joint_primary"]
        for comparison in comparisons:
            comparison_metrics = (
                requested if comparison == "joint_primary" else (comparison,)
            )

            def matches(rows: Mapping[str, Mapping[str, Any]]) -> bool:
                return all(
                    float(rows[metric]["mean"])
                    >= float(target_rows[metric]["mean"]) - margins[metric]
                    for metric in comparison_metrics
                )

            matched_index = next(
                (
                    index
                    for index, (_, _, candidate_rows) in enumerate(candidates)
                    if matches(candidate_rows)
                ),
                None,
            )
            if matched_index is None:
                lower_exposure, lower_m, _ = candidates[-1]
                upper_exposure = upper_m = matched_rows = None
                status = "right_censored"
            else:
                upper_exposure, upper_m, matched_rows = candidates[matched_index]
                if matched_index == 0:
                    lower_exposure = lower_m = None
                    status = "left_censored"
                else:
                    lower_exposure, lower_m, _ = candidates[matched_index - 1]
                    status = "bracketed"
            if matched_index is None:
                count_rows = (target_rows, candidates[-1][2])
            elif matched_index == 0:
                count_rows = (target_rows, candidates[0][2])
            else:
                count_rows = (
                    target_rows,
                    candidates[matched_index - 1][2],
                    candidates[matched_index][2],
                )
            results.append(
                {
                    "profile": profile,
                    "treatment_cell": cell,
                    "baseline_cell": baseline_cell,
                    "method": method,
                    "step": step,
                    "p": p,
                    "treatment_samples_per_step": target_m,
                    "comparison": comparison,
                    "comparison_metrics": comparison_metrics,
                    "practical_margins": {
                        metric: margins[metric] for metric in comparison_metrics
                    },
                    "exposure_key": exposure_key,
                    "treatment_exposure": target_exposure,
                    "target_means": {
                        metric: float(target_rows[metric]["mean"])
                        for metric in comparison_metrics
                    },
                    "status": status,
                    "baseline_samples_per_step_lower": lower_m,
                    "baseline_exposure_lower": lower_exposure,
                    "multiplier_lower": (
                        None
                        if lower_exposure is None
                        else lower_exposure / target_exposure
                    ),
                    "baseline_samples_per_step_upper": upper_m,
                    "baseline_exposure_upper": upper_exposure,
                    "multiplier_upper": (
                        None
                        if upper_exposure is None
                        else upper_exposure / target_exposure
                    ),
                    "matched_baseline_means": (
                        None
                        if matched_rows is None
                        else {
                            metric: float(matched_rows[metric]["mean"])
                            for metric in comparison_metrics
                        }
                    ),
                    "minimum_replica_count": min(
                        int(row["replica_count"])
                        for rows in count_rows
                        for row in rows.values()
                    ),
                }
            )
    return sorted(
        results,
        key=lambda row: (
            row["profile"],
            row["treatment_cell"],
            row["method"],
            row["treatment_samples_per_step"],
            row["comparison"],
            row["step"],
        ),
    )


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile values must be nonempty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def phase9_paired_dominance_bootstrap_rows(
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_cell: str,
    treatment_cells: Sequence[str],
    treatment_samples_per_step: int,
    baseline_samples_per_step: Sequence[int],
    steps: Sequence[int],
    metrics: Sequence[str] = PLAN2_PRIMARY_PREDICTIVE_METRICS,
    practical_margins: Mapping[str, float] | None = None,
    maximum_draws: int = 10_000,
    seed: int = 1729,
    exposure_key: str = "after_cumulative_unique_nine_observations",
) -> list[dict[str, Any]]:
    """Estimate paired bootstrap support for baseline dominance at fixed states."""

    requested = tuple(metrics)
    treatments = tuple(treatment_cells)
    baseline_sizes = tuple(sorted(set(baseline_samples_per_step)))
    requested_steps = tuple(sorted(set(steps)))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("metrics must be unique and nonempty")
    if not treatments or len(treatments) != len(set(treatments)):
        raise ValueError("treatment_cells must be unique and nonempty")
    if not baseline_sizes or any(value <= 0 for value in baseline_sizes):
        raise ValueError("baseline samples_per_step values must be positive")
    if not requested_steps or any(value < 0 for value in requested_steps):
        raise ValueError("steps must be nonnegative")
    if maximum_draws <= 0:
        raise ValueError("maximum_draws must be positive")
    margins = {metric: 0.0 for metric in requested}
    if practical_margins is not None:
        unknown = set(practical_margins) - set(requested)
        if unknown:
            raise ValueError(f"practical margins contain unknown metrics: {unknown}")
        margins.update(
            {metric: float(value) for metric, value in practical_margins.items()}
        )
    if any(not math.isfinite(value) or value < 0.0 for value in margins.values()):
        raise ValueError("practical margins must be finite and nonnegative")

    indexed: dict[
        tuple[str, str, str, int, int], dict[int, Mapping[str, Any]]
    ] = defaultdict(dict)
    allowed_cells = {baseline_cell, *treatments}
    allowed_sizes = {treatment_samples_per_step, *baseline_sizes}
    for row in trajectory_rows:
        cell = str(row["cell"])
        samples_per_step = int(row["samples_per_step"])
        step = int(row["step"])
        if (
            cell not in allowed_cells
            or samples_per_step not in allowed_sizes
            or step not in requested_steps
        ):
            continue
        key = (
            str(row["profile"]),
            cell,
            str(row["method"]),
            samples_per_step,
            step,
        )
        replica = int(row["replica_index"])
        if replica in indexed[key]:
            raise AnalysisArtifactError(f"duplicate paired-bootstrap row: {key}:{replica}")
        indexed[key][replica] = row

    results = []
    rng = random.Random(seed)
    for key, treatment_by_replica in indexed.items():
        profile, treatment_cell, method, samples_per_step, step = key
        if (
            treatment_cell not in treatments
            or samples_per_step != treatment_samples_per_step
        ):
            continue
        for baseline_m in baseline_sizes:
            baseline_by_replica = indexed.get(
                (profile, baseline_cell, method, baseline_m, step)
            )
            if baseline_by_replica is None:
                continue
            replicas = sorted(
                set(treatment_by_replica).intersection(baseline_by_replica)
            )
            replicas = [
                replica
                for replica in replicas
                if all(
                    treatment_by_replica[replica].get(metric) is not None
                    and baseline_by_replica[replica].get(metric) is not None
                    for metric in requested
                )
                and treatment_by_replica[replica].get(exposure_key) is not None
                and baseline_by_replica[replica].get(exposure_key) is not None
            ]
            if len(replicas) < 2:
                continue
            p_values = {
                float(rows[replica]["p"])
                for rows in (treatment_by_replica, baseline_by_replica)
                for replica in replicas
            }
            if len(p_values) != 1:
                raise AnalysisArtifactError(
                    f"unaligned paired-bootstrap p values at {profile}:{step}"
                )

            exact_draw_count = len(replicas) ** len(replicas)
            if exact_draw_count <= maximum_draws:
                draws: Iterable[tuple[int, ...]] = itertools.product(
                    replicas, repeat=len(replicas)
                )
                draw_count = exact_draw_count
                bootstrap_mode = "exact"
            else:
                draws = (
                    tuple(rng.choice(replicas) for _ in replicas)
                    for _ in range(maximum_draws)
                )
                draw_count = maximum_draws
                bootstrap_mode = "monte_carlo"

            deltas = {metric: [] for metric in requested}
            metric_match_counts = {metric: 0 for metric in requested}
            joint_match_count = 0
            for draw in draws:
                draw_deltas = {}
                for metric in requested:
                    delta = sum(
                        float(baseline_by_replica[replica][metric])
                        - float(treatment_by_replica[replica][metric])
                        + margins[metric]
                        for replica in draw
                    ) / len(draw)
                    deltas[metric].append(delta)
                    draw_deltas[metric] = delta
                    metric_match_counts[metric] += delta >= 0.0
                joint_match_count += all(
                    delta >= 0.0 for delta in draw_deltas.values()
                )

            treatment_exposure = sum(
                float(treatment_by_replica[replica][exposure_key])
                for replica in replicas
            ) / len(replicas)
            baseline_exposure = sum(
                float(baseline_by_replica[replica][exposure_key])
                for replica in replicas
            ) / len(replicas)
            results.append(
                {
                    "profile": profile,
                    "treatment_cell": treatment_cell,
                    "baseline_cell": baseline_cell,
                    "method": method,
                    "step": step,
                    "p": next(iter(p_values)),
                    "treatment_samples_per_step": treatment_samples_per_step,
                    "baseline_samples_per_step": baseline_m,
                    "replica_count": len(replicas),
                    "replica_indices": replicas,
                    "bootstrap_mode": bootstrap_mode,
                    "bootstrap_draw_count": draw_count,
                    "practical_margins": margins,
                    "exposure_key": exposure_key,
                    "treatment_exposure": treatment_exposure,
                    "baseline_exposure": baseline_exposure,
                    "exposure_multiplier": baseline_exposure / treatment_exposure,
                    "mean_metric_deltas": {
                        metric: sum(values) / len(values)
                        for metric, values in deltas.items()
                    },
                    "metric_delta_ci95": {
                        metric: (
                            _linear_quantile(values, 0.025),
                            _linear_quantile(values, 0.975),
                        )
                        for metric, values in deltas.items()
                    },
                    "metric_match_probabilities": {
                        metric: count / draw_count
                        for metric, count in metric_match_counts.items()
                    },
                    "joint_match_probability": joint_match_count / draw_count,
                }
            )
    return sorted(
        results,
        key=lambda row: (
            row["profile"],
            row["treatment_cell"],
            row["method"],
            row["treatment_samples_per_step"],
            row["baseline_samples_per_step"],
            row["step"],
        ),
    )


def phase9_metric_availability_rows(
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = PLAN2_TRAJECTORY_METRICS,
) -> list[dict[str, Any]]:
    """Report complete-replica coverage for requested trajectory metrics."""

    grouped: dict[
        tuple[str, str, str, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in trajectory_rows:
        grouped[
            (
                str(row["profile"]),
                str(row["cell"]),
                str(row["method"]),
                int(row["samples_per_step"]),
            )
        ].append(row)

    results = []
    for (profile, cell, method, samples_per_step), group in grouped.items():
        by_replica: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in group:
            by_replica[str(row["replica_id"])].append(row)
        for metric in metrics:
            complete = sum(
                all(metric in row for row in replica_rows)
                for replica_rows in by_replica.values()
            )
            results.append(
                {
                    "profile": profile,
                    "cell": cell,
                    "method": method,
                    "samples_per_step": samples_per_step,
                    "metric": metric,
                    "completed_replicas": len(by_replica),
                    "replicas_with_complete_metric": complete,
                    "metric_available": complete > 0,
                    "metric_complete": complete == len(by_replica),
                }
            )
    return sorted(
        results,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["samples_per_step"],
            row["metric"],
        ),
    )


def require_phase9_metrics(
    availability_rows: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    cells: Sequence[str],
    metrics: Sequence[str],
) -> None:
    """Fail clearly when a requested condition or scalar metric is unavailable."""

    indexed = {
        (str(row["profile"]), str(row["cell"]), str(row["metric"])): row
        for row in availability_rows
    }
    missing = []
    for cell in cells:
        for metric in metrics:
            row = indexed.get((profile, cell, metric))
            if row is None or not row.get("metric_available"):
                missing.append(f"{profile}:{cell}:{metric}")
    if missing:
        raise AnalysisArtifactError(
            "requested Phase 9 trajectory metrics are unavailable: "
            + ", ".join(missing)
        )


def _normalized_auc_by_key(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> float | None:
    points = sorted(
        (
            (float(row[x_key]), float(row[y_key]))
            for row in rows
            if row.get(x_key) is not None and row.get(y_key) is not None
        ),
        key=lambda point: point[0],
    )
    if not points:
        return None
    if any(right[0] < left[0] for left, right in zip(points, points[1:])):
        raise AnalysisArtifactError(f"{x_key} is not monotone")
    if len(points) == 1 or points[-1][0] == points[0][0]:
        return points[0][1]
    area = sum(
        0.5 * (left[1] + right[1]) * (right[0] - left[0])
        for left, right in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def phase9_exposure_auc_rows(
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = (
        "after_nine_ovr_accuracy",
        "after_nine_precision",
        "after_nine_recall",
        "after_environment_accuracy",
    ),
) -> list[dict[str, Any]]:
    """Return per-replica p- and observation-normalized trajectory AUCs."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[(str(row["run_id"]), str(row["method"]))].append(row)

    results = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: int(row["step"]))
        first = ordered[0]
        result = {
            "profile": first["profile"],
            "cell": first["cell"],
            "kind": first["kind"],
            "method": first["method"],
            "run_id": first["run_id"],
            "control_run_id": first["control_run_id"],
            "replica_id": first["replica_id"],
            "replica_index": first["replica_index"],
            "samples_per_step": first["samples_per_step"],
        }
        for metric in metrics:
            result[f"{metric}_p_auc"] = _normalized_auc_by_key(
                ordered, x_key="p", y_key=metric
            )
            result[f"{metric}_observation_auc"] = _normalized_auc_by_key(
                ordered,
                x_key="after_cumulative_observations",
                y_key=metric,
            )
        results.append(result)
    return sorted(
        results,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["samples_per_step"],
            row["replica_index"],
        ),
    )


def phase9_paired_auc_rows(
    auc_rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    """Pair arbitrary AUC metrics by immutable control run identity."""

    by_run_method = {
        (str(row["run_id"]), str(row["method"])): row for row in auc_rows
    }
    results = []
    for treatment in auc_rows:
        control_run_id = treatment.get("control_run_id")
        if treatment.get("kind") != "treatment" or control_run_id is None:
            continue
        control = by_run_method.get(
            (str(control_run_id), str(treatment["method"]))
        )
        if control is None:
            continue
        result = {
            "profile": treatment["profile"],
            "cell": treatment["cell"],
            "method": treatment["method"],
            "samples_per_step": treatment["samples_per_step"],
            "replica_id": treatment["replica_id"],
            "replica_index": treatment["replica_index"],
            "treatment_run_id": treatment["run_id"],
            "control_run_id": control_run_id,
        }
        for metric in metrics:
            treatment_value = treatment.get(metric)
            control_value = control.get(metric)
            result[f"treatment_{metric}"] = treatment_value
            result[f"control_{metric}"] = control_value
            result[f"delta_{metric}"] = (
                None
                if treatment_value is None or control_value is None
                else float(treatment_value) - float(control_value)
            )
        results.append(result)
    return sorted(
        results,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["samples_per_step"],
            row["replica_index"],
        ),
    )


def phase9_fixed_budget_rows(
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    observation_budgets: Sequence[int],
    metrics: Sequence[str] = PLAN2_TRAJECTORY_METRICS,
) -> list[dict[str, Any]]:
    """Select each replica's latest post-update state within fixed budgets."""

    budgets = tuple(sorted(set(observation_budgets)))
    if not budgets or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in budgets
    ):
        raise ValueError("observation budgets must be nonnegative integers")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[(str(row["run_id"]), str(row["method"]))].append(row)

    results = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: int(row["step"]))
        first = ordered[0]
        if any(
            row.get("after_cumulative_observations") is None for row in ordered
        ):
            continue
        for budget in budgets:
            eligible = [
                row
                for row in ordered
                if int(row["after_cumulative_observations"]) <= budget
            ]
            if not eligible:
                continue
            selected = max(
                eligible,
                key=lambda row: (
                    int(row["after_cumulative_observations"]),
                    int(row["step"]),
                ),
            )
            results.append(
                {
                    "profile": first["profile"],
                    "cell": first["cell"],
                    "kind": first["kind"],
                    "method": first["method"],
                    "run_id": first["run_id"],
                    "control_run_id": first["control_run_id"],
                    "replica_id": first["replica_id"],
                    "replica_index": first["replica_index"],
                    "samples_per_step": first["samples_per_step"],
                    "requested_observation_budget": budget,
                    "observed_cumulative_observations": selected[
                        "after_cumulative_observations"
                    ],
                    "observed_cumulative_nine_observations": selected[
                        "after_cumulative_nine_observations"
                    ],
                    "observed_cumulative_unique_nine_observations": selected[
                        "after_cumulative_unique_nine_observations"
                    ],
                    "step": selected["step"],
                    "p": selected["p"],
                    **{metric: selected.get(metric) for metric in metrics},
                }
            )
    return sorted(
        results,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["samples_per_step"],
            row["replica_index"],
            row["requested_observation_budget"],
        ),
    )


def phase9_paired_fixed_budget_rows(
    budget_rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    """Pair fixed-budget outcomes by control run, method, and budget."""

    by_key = {
        (
            str(row["run_id"]),
            str(row["method"]),
            int(row["requested_observation_budget"]),
        ): row
        for row in budget_rows
    }
    results = []
    for treatment in budget_rows:
        control_run_id = treatment.get("control_run_id")
        if treatment.get("kind") != "treatment" or control_run_id is None:
            continue
        key = (
            str(control_run_id),
            str(treatment["method"]),
            int(treatment["requested_observation_budget"]),
        )
        control = by_key.get(key)
        if control is None:
            continue
        result = {
            "profile": treatment["profile"],
            "cell": treatment["cell"],
            "method": treatment["method"],
            "samples_per_step": treatment["samples_per_step"],
            "replica_id": treatment["replica_id"],
            "replica_index": treatment["replica_index"],
            "requested_observation_budget": treatment[
                "requested_observation_budget"
            ],
            "treatment_run_id": treatment["run_id"],
            "control_run_id": control_run_id,
        }
        for metric in metrics:
            treatment_value = treatment.get(metric)
            control_value = control.get(metric)
            result[f"treatment_{metric}"] = treatment_value
            result[f"control_{metric}"] = control_value
            result[f"delta_{metric}"] = (
                None
                if treatment_value is None or control_value is None
                else float(treatment_value) - float(control_value)
            )
        results.append(result)
    return sorted(
        results,
        key=lambda row: (
            row["profile"],
            row["cell"],
            row["method"],
            row["samples_per_step"],
            row["replica_index"],
            row["requested_observation_budget"],
        ),
    )
