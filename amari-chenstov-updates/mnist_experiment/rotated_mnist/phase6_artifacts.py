"""Immutable artifact lifecycle for Plan 6 diagnostics and oracle runs."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

from src.seeding import SEED_SCHEMA_VERSION, derive_seed_map

from .artifacts import (
    MANIFEST_SCHEMA_VERSION,
    RotatedArtifactError,
    RotatedCompletedRunError,
    RotatedIncompleteRunError,
    RotatedRunSession,
    _read_json,
    _utc_now,
    _write_json,
    runtime_metadata,
)
from .phase6_config import (
    PHASE6_ARTIFACT_SCHEMA_VERSION,
    PHASE6_METRIC_SCHEMA_VERSION,
    Phase6DebiasConfig,
    Phase6OracleConfig,
)


PHASE6_SEED_COMPONENTS = (
    "plan6_reference_fit",
    "plan6_reference_loader",
    "plan6_fisher_scores",
    "plan6_fisher_lanczos",
    "plan6_local_mle_samples",
    "plan6_local_mle_loader",
)

PHASE6_DEBIAS_REQUIRED_ARTIFACTS = (
    "source_contract.json",
    "debiased_recommendations.json",
    "audit_summary.json",
)

PHASE6_ORACLE_REQUIRED_ARTIFACTS = (
    "source_contract.json",
    "reference_contract.json",
    "reference_metrics.json",
    "reference_states.pt",
    "reference_fishers.pt",
    "local_mle_metrics.json",
    "local_mle_parameters.pt",
    "oracle_estimates.json",
    "convergence.json",
    "run_summary.json",
)


class Phase6RunStore:
    def __init__(self, root: str | Path, *, run_kind: str, label: str):
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"
        self.run_kind = run_kind
        self.label = label

    def begin(self, config, repo_root: str | Path, *, resume: bool = False):
        config.validate()
        final_path = self.root / config.run_id
        working_path = self.incomplete_root / config.run_id
        if final_path.exists():
            if (final_path / "COMPLETED").is_file():
                raise RotatedCompletedRunError(
                    f"{self.label} already completed: {config.run_id}"
                )
            raise RotatedArtifactError(
                f"final {self.label} exists without COMPLETED: {final_path}"
            )
        if working_path.exists():
            if not resume:
                raise RotatedIncompleteRunError(
                    f"incomplete {self.label} exists; pass --resume: {config.run_id}"
                )
            if _read_json(working_path / "config.json") != config.to_mapping():
                raise RotatedArtifactError(f"incomplete {self.label} config differs")
            manifest = _read_json(working_path / "manifest.json")
            if manifest.get("status") != "incomplete":
                raise RotatedArtifactError(f"incomplete {self.label} manifest is invalid")
            return RotatedRunSession(config.run_id, working_path, final_path)

        working_path.mkdir(parents=True, exist_ok=False)
        _write_json(working_path / "config.json", config.to_mapping())
        _write_json(
            working_path / "manifest.json",
            {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "artifact_schema_version": config.artifact_schema_version,
                "metric_schema_version": config.metric_schema_version,
                "seed_schema_version": SEED_SCHEMA_VERSION,
                "run_kind": self.run_kind,
                "run_id": config.run_id,
                "experiment": config.experiment,
                "replica_id": config.replica_id,
                "config_hash": config.config_hash,
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
                "seeds": derive_seed_map(
                    config.replica_seed, PHASE6_SEED_COMPONENTS
                ),
                "runtime": runtime_metadata(repo_root, config),
            },
        )
        return RotatedRunSession(config.run_id, working_path, final_path)


def _validated_manifest(path: Path, config, *, run_kind: str) -> dict[str, Any]:
    manifest = _read_json(path / "manifest.json")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_schema_version")
        != PHASE6_ARTIFACT_SCHEMA_VERSION
        or manifest.get("metric_schema_version") != PHASE6_METRIC_SCHEMA_VERSION
        or manifest.get("run_kind") != run_kind
        or manifest.get("status") != "completed"
        or manifest.get("config_hash") != config.config_hash
        or manifest.get("run_id") != config.run_id
        or path.name != config.run_id
    ):
        raise RotatedArtifactError("Plan 6 manifest is incompatible")
    return manifest


def _require_completed(path: Path, names: tuple[str, ...]) -> None:
    if not (path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(f"Plan 6 run is incomplete: {path}")
    for name in ("config.json", "manifest.json", *names):
        if not (path / name).is_file():
            raise RotatedArtifactError(f"completed Plan 6 run is missing {name}")


@dataclasses.dataclass(frozen=True)
class LoadedPhase6DebiasRun:
    path: Path
    config: Phase6DebiasConfig
    manifest: dict[str, Any]
    source_contract: dict[str, Any]
    recommendations: dict[str, Any]
    audit_summary: dict[str, Any]


def load_completed_phase6_debias(path: str | Path) -> LoadedPhase6DebiasRun:
    run_path = Path(path)
    _require_completed(run_path, PHASE6_DEBIAS_REQUIRED_ARTIFACTS)
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("Plan 6 debias config must be an object")
    config = Phase6DebiasConfig.from_mapping(config_value)
    manifest = _validated_manifest(
        run_path, config, run_kind="phase6_artifact_only_trend_variance_audit"
    )
    source_contract = _read_json(run_path / "source_contract.json")
    recommendations = _read_json(run_path / "debiased_recommendations.json")
    summary = _read_json(run_path / "audit_summary.json")
    if (
        source_contract.get("source_run_id") != config.source_run_id
        or summary.get("source_run_id") != config.source_run_id
        or summary.get("config_hash") != config.config_hash
        or set(recommendations) != set(config.schedule_kinds)
    ):
        raise RotatedArtifactError("Plan 6 debias contents are incompatible")
    return LoadedPhase6DebiasRun(
        path=run_path,
        config=config,
        manifest=manifest,
        source_contract=source_contract,
        recommendations=recommendations,
        audit_summary=summary,
    )


@dataclasses.dataclass(frozen=True)
class LoadedPhase6OracleRun:
    path: Path
    config: Phase6OracleConfig
    manifest: dict[str, Any]
    source_contract: dict[str, Any]
    reference_contract: dict[str, Any]
    reference_metrics: tuple[dict[str, Any], ...]
    local_mle_metrics: dict[str, Any]
    oracle_estimates: dict[str, Any]
    convergence: dict[str, Any]
    run_summary: dict[str, Any]


def load_completed_phase6_oracle(path: str | Path) -> LoadedPhase6OracleRun:
    run_path = Path(path)
    _require_completed(run_path, PHASE6_ORACLE_REQUIRED_ARTIFACTS)
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("Plan 6 oracle config must be an object")
    config = Phase6OracleConfig.from_mapping(config_value)
    run_kind = f"phase6_{config.mode}_instantaneous_oracle"
    manifest = _validated_manifest(run_path, config, run_kind=run_kind)
    source_contract = _read_json(run_path / "source_contract.json")
    reference_contract = _read_json(run_path / "reference_contract.json")
    reference_metrics = _read_json(run_path / "reference_metrics.json")
    local_metrics = _read_json(run_path / "local_mle_metrics.json")
    estimates = _read_json(run_path / "oracle_estimates.json")
    convergence = _read_json(run_path / "convergence.json")
    summary = _read_json(run_path / "run_summary.json")
    if (
        source_contract.get("source_run_id") != config.source_run_id
        or summary.get("source_run_id") != config.source_run_id
        or summary.get("config_hash") != config.config_hash
        or summary.get("reference_contract_hash")
        != reference_contract.get("content_hash")
        or set(estimates) != set(config.schedule_kinds)
        or not isinstance(reference_metrics, list)
        or not isinstance(local_metrics, dict)
        or not isinstance(convergence, dict)
    ):
        raise RotatedArtifactError("Plan 6 oracle contents are incompatible")
    return LoadedPhase6OracleRun(
        path=run_path,
        config=config,
        manifest=manifest,
        source_contract=source_contract,
        reference_contract=reference_contract,
        reference_metrics=tuple(reference_metrics),
        local_mle_metrics=local_metrics,
        oracle_estimates=estimates,
        convergence=convergence,
        run_summary=summary,
    )
