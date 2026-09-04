"""Strict configuration for the artifact-only Plan 7 coefficient audit."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config import RotatedConfigError, RuntimeConfig, _integer, _strict_keys
from .phase5_single_lap_config import (
    SINGLE_LAP_EDR_CONDITION,
    SINGLE_LAP_SCHEDULES,
)
from .phase6_config import PHASE6_SOURCE_RUN_ID


PHASE7_CONFIG_SCHEMA_VERSION = 1
PHASE7_METRIC_SCHEMA_VERSION = 1
PHASE7_ARTIFACT_SCHEMA_VERSION = 1
PHASE7_DEBIAS_RUN_ID = (
    "rotated_mnist_phase6_debias_primary__"
    "replica-0001__e1c9d975a0351e2e"
)
PHASE7_ORACLE_RUN_ID = (
    "rotated_mnist_phase6_oracle_full_path__"
    "replica-0001__7c7dc5936d08fb91"
)
PHASE7_CONTROL_CONDITIONS = ("fixed_pi0025", "fixed_pi005")


def _positive(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RotatedConfigError(f"{name} must be finite and positive")
    return float(value)


@dataclasses.dataclass(frozen=True)
class Phase7AuditConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    source_run_path: str
    source_run_id: str
    debias_run_path: str
    debias_run_id: str
    oracle_run_path: str
    oracle_run_id: str
    schedule_kinds: tuple[str, ...]
    primary_condition: str
    control_conditions: tuple[str, ...]
    deployed_batch_size: int
    oracle_sample_size: int
    oracle_replicate_count: int
    primary_rank: int
    sensitivity_ranks: tuple[int, ...]
    variance_scale: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase7AuditConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "Phase 7 audit configuration")
        for name in ("schedule_kinds", "control_conditions", "sensitivity_ranks"):
            if not isinstance(value[name], list):
                raise RotatedConfigError(f"{name} must be a list")
        result = cls(
            schema_version=_integer(
                value["schema_version"], minimum=1, name="schema_version"
            ),
            metric_schema_version=_integer(
                value["metric_schema_version"],
                minimum=1,
                name="metric_schema_version",
            ),
            artifact_schema_version=_integer(
                value["artifact_schema_version"],
                minimum=1,
                name="artifact_schema_version",
            ),
            experiment=str(value["experiment"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(
                value["replica_seed"], minimum=0, name="replica_seed"
            ),
            source_run_path=str(value["source_run_path"]),
            source_run_id=str(value["source_run_id"]),
            debias_run_path=str(value["debias_run_path"]),
            debias_run_id=str(value["debias_run_id"]),
            oracle_run_path=str(value["oracle_run_path"]),
            oracle_run_id=str(value["oracle_run_id"]),
            schedule_kinds=tuple(str(item) for item in value["schedule_kinds"]),
            primary_condition=str(value["primary_condition"]),
            control_conditions=tuple(
                str(item) for item in value["control_conditions"]
            ),
            deployed_batch_size=_integer(
                value["deployed_batch_size"],
                minimum=1,
                name="deployed_batch_size",
            ),
            oracle_sample_size=_integer(
                value["oracle_sample_size"],
                minimum=2,
                name="oracle_sample_size",
            ),
            oracle_replicate_count=_integer(
                value["oracle_replicate_count"],
                minimum=2,
                name="oracle_replicate_count",
            ),
            primary_rank=_integer(
                value["primary_rank"], minimum=1, name="primary_rank"
            ),
            sensitivity_ranks=tuple(
                _integer(item, minimum=1, name="sensitivity rank")
                for item in value["sensitivity_ranks"]
            ),
            variance_scale=_positive(value["variance_scale"], name="variance_scale"),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != PHASE7_CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 7 config schema")
        if self.metric_schema_version != PHASE7_METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 7 metric schema")
        if self.artifact_schema_version != PHASE7_ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 7 artifact schema")
        if not self.experiment.startswith("rotated_mnist_phase7_anchor_audit_"):
            raise RotatedConfigError("invalid Phase 7 experiment name")
        if not self.replica_id or self.replica_seed >= 2**63:
            raise RotatedConfigError("invalid Phase 7 replica identity")
        paths_and_ids = (
            (self.source_run_path, self.source_run_id),
            (self.debias_run_path, self.debias_run_id),
            (self.oracle_run_path, self.oracle_run_id),
        )
        if any(Path(path).name != run_id for path, run_id in paths_and_ids):
            raise RotatedConfigError("input artifact path and run ID differ")
        if self.source_run_id != PHASE6_SOURCE_RUN_ID:
            raise RotatedConfigError("Plan 7 source is not authoritative")
        if self.debias_run_id != PHASE7_DEBIAS_RUN_ID:
            raise RotatedConfigError("Plan 7 debias run is not authoritative")
        if self.oracle_run_id != PHASE7_ORACLE_RUN_ID:
            raise RotatedConfigError("Plan 7 oracle run is not authoritative")
        if self.schedule_kinds != SINGLE_LAP_SCHEDULES:
            raise RotatedConfigError("Plan 7 requires linear and sigmoid schedules")
        if self.primary_condition != SINGLE_LAP_EDR_CONDITION:
            raise RotatedConfigError("Plan 7 primary condition must be EDR")
        if self.control_conditions != PHASE7_CONTROL_CONDITIONS:
            raise RotatedConfigError("Plan 7 controls must be fixed .025 and .05")
        if self.deployed_batch_size != 4:
            raise RotatedConfigError("Plan 7 requires deployed m=4")
        ranks = (self.primary_rank, *self.sensitivity_ranks)
        if len(set(ranks)) != len(ranks):
            raise RotatedConfigError("primary and sensitivity ranks must be distinct")
        if self.runtime.device != "cpu" or self.runtime.dtype != "float64":
            raise RotatedConfigError("Plan 7 audit must use CPU float64 arithmetic")

    @property
    def conditions(self) -> tuple[str, ...]:
        return (self.primary_condition, *self.control_conditions)

    @property
    def ranks(self) -> tuple[int, ...]:
        return (self.primary_rank, *self.sensitivity_ranks)

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["schedule_kinds"] = list(self.schedule_kinds)
        value["control_conditions"] = list(self.control_conditions)
        value["sensitivity_ranks"] = list(self.sensitivity_ranks)
        return value

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_mapping(), allow_nan=False, separators=(",", ":"), sort_keys=True
        )

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        return f"{self.experiment}__{self.replica_id}__{self.config_hash[:16]}"


def load_phase7_audit_config(path: str | Path) -> Phase7AuditConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Phase 7 audit configuration must be an object")
    return Phase7AuditConfig.from_mapping(value)
