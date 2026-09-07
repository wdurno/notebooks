"""Strict configuration for the Plan 7 movement-premium audit."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config import RotatedConfigError, RuntimeConfig, _integer, _strict_keys


MOVEMENT_CONFIG_SCHEMA_VERSION = 1
MOVEMENT_METRIC_SCHEMA_VERSION = 1
MOVEMENT_ARTIFACT_SCHEMA_VERSION = 1
MOVEMENT_SCHEDULES = ("linear", "sigmoid")
MOVEMENT_CONDITION = "decomposed_edr"

AUTHORITATIVE_ORACLE_RUN_ID = (
    "rotated_mnist_phase6_oracle_full_path__"
    "replica-0001__7c7dc5936d08fb91"
)
AUTHORITATIVE_COEFFICIENT_RUN_ID = (
    "rotated_mnist_phase7_anchor_audit_primary_v3__"
    "replica-0001__4406a8e7634c0ad8"
)
AUTHORITATIVE_SINGLE_RUN_ID = (
    "rotated_mnist_phase8_single_lap_rechallenge__"
    "replica-0001__6445ab53bb10cd25"
)
AUTHORITATIVE_DOUBLE_RUN_ID = (
    "rotated_mnist_phase8_double_lap_reversal_stress__"
    "replica-0001__f035d4b8cdf6d894"
)


def _positive_float(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RotatedConfigError(f"{name} must be finite and positive")
    return float(value)


@dataclasses.dataclass(frozen=True)
class MovementPremiumAuditConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    oracle_run_path: str
    oracle_run_id: str
    coefficient_run_path: str
    coefficient_run_id: str
    single_lap_run_path: str
    single_lap_run_id: str
    double_lap_run_path: str
    double_lap_run_id: str
    schedule_kinds: tuple[str, ...]
    condition: str
    deployed_batch_size: int
    oracle_sample_size: int
    oracle_replicate_count: int
    oracle_rank: int
    cold_start_steps: int
    reversal_window_steps: int
    ratio_floor: float
    attribution_tolerance: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "MovementPremiumAuditConfig":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "Plan 7 movement-premium configuration",
        )
        if not isinstance(value["schedule_kinds"], list):
            raise RotatedConfigError("schedule_kinds must be a list")
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
            oracle_run_path=str(value["oracle_run_path"]),
            oracle_run_id=str(value["oracle_run_id"]),
            coefficient_run_path=str(value["coefficient_run_path"]),
            coefficient_run_id=str(value["coefficient_run_id"]),
            single_lap_run_path=str(value["single_lap_run_path"]),
            single_lap_run_id=str(value["single_lap_run_id"]),
            double_lap_run_path=str(value["double_lap_run_path"]),
            double_lap_run_id=str(value["double_lap_run_id"]),
            schedule_kinds=tuple(str(item) for item in value["schedule_kinds"]),
            condition=str(value["condition"]),
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
            oracle_rank=_integer(
                value["oracle_rank"], minimum=1, name="oracle_rank"
            ),
            cold_start_steps=_integer(
                value["cold_start_steps"],
                minimum=0,
                name="cold_start_steps",
            ),
            reversal_window_steps=_integer(
                value["reversal_window_steps"],
                minimum=1,
                name="reversal_window_steps",
            ),
            ratio_floor=_positive_float(
                value["ratio_floor"], name="ratio_floor"
            ),
            attribution_tolerance=_positive_float(
                value["attribution_tolerance"], name="attribution_tolerance"
            ),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != MOVEMENT_CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported movement-audit config schema")
        if self.metric_schema_version != MOVEMENT_METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported movement-audit metric schema")
        if self.artifact_schema_version != MOVEMENT_ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported movement-audit artifact schema")
        if not self.experiment.startswith("rotated_mnist_phase7_movement_audit_"):
            raise RotatedConfigError("invalid movement-audit experiment name")
        if not self.replica_id or self.replica_seed >= 2**63:
            raise RotatedConfigError("invalid movement-audit replica identity")
        paths_and_ids = (
            (self.oracle_run_path, self.oracle_run_id),
            (self.coefficient_run_path, self.coefficient_run_id),
            (self.single_lap_run_path, self.single_lap_run_id),
            (self.double_lap_run_path, self.double_lap_run_id),
        )
        if any(Path(path).name != run_id for path, run_id in paths_and_ids):
            raise RotatedConfigError("input artifact path and run ID differ")
        expected_ids = (
            AUTHORITATIVE_ORACLE_RUN_ID,
            AUTHORITATIVE_COEFFICIENT_RUN_ID,
            AUTHORITATIVE_SINGLE_RUN_ID,
            AUTHORITATIVE_DOUBLE_RUN_ID,
        )
        if tuple(run_id for _, run_id in paths_and_ids) != expected_ids:
            raise RotatedConfigError("movement audit requires authoritative inputs")
        if self.schedule_kinds != MOVEMENT_SCHEDULES:
            raise RotatedConfigError("movement audit requires both schedules")
        if self.condition != MOVEMENT_CONDITION:
            raise RotatedConfigError("movement audit requires decomposed_edr")
        frozen = (
            ("deployed_batch_size", self.deployed_batch_size, 4),
            ("oracle_sample_size", self.oracle_sample_size, 2048),
            ("oracle_replicate_count", self.oracle_replicate_count, 64),
            ("oracle_rank", self.oracle_rank, 16),
            ("cold_start_steps", self.cold_start_steps, 8),
            ("reversal_window_steps", self.reversal_window_steps, 8),
        )
        for name, actual, expected in frozen:
            if actual != expected:
                raise RotatedConfigError(f"{name} is frozen at {expected}")
        if self.runtime.device != "cpu" or self.runtime.dtype != "float64":
            raise RotatedConfigError("movement audit must use CPU float64 arithmetic")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["schedule_kinds"] = list(self.schedule_kinds)
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


def load_movement_premium_audit_config(
    path: str | Path,
) -> MovementPremiumAuditConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("movement-audit configuration must be an object")
    return MovementPremiumAuditConfig.from_mapping(value)
