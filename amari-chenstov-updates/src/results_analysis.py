"""Strict read-only loaders and summaries for MNIST experiment results."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import MANIFEST_SCHEMA_VERSION
from .config import ExperimentConfig

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
