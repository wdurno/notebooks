"""Strict configuration for the Phase 5 slow-trend single-lap retry."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config import (
    DataConfig,
    InitializationConfig,
    RotatedConfigError,
    RotationConfig,
    RuntimeConfig,
    _integer,
    _positive_float,
    _strict_keys,
)
from .phase3_config import Phase3FisherConfig, Phase3LearnerConfig
from .phase5_double_lap_config import (
    DoubleLapControllerConfig,
    RotatedDoubleLapConfig,
)


SINGLE_LAP_CONFIG_SCHEMA_VERSION = 1
SINGLE_LAP_METRIC_SCHEMA_VERSION = 1
SINGLE_LAP_ARTIFACT_SCHEMA_VERSION = 1
SINGLE_LAP_KNOTS = (0.0, 30.0, 0.0)
SINGLE_LAP_SCHEDULES = ("linear", "sigmoid")
SINGLE_LAP_EDR_CONDITION = "edr_slowtrend_slowaction"
SINGLE_LAP_CONDITIONS = (
    "current_only",
    "fixed_pi0025",
    "fixed_pi005",
    SINGLE_LAP_EDR_CONDITION,
)
SINGLE_LAP_FIXED_PI = {
    "fixed_pi0025": 0.025,
    "fixed_pi005": 0.05,
}


@dataclasses.dataclass(frozen=True)
class RotatedSlowSingleLapConfig(RotatedDoubleLapConfig):
    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "RotatedSlowSingleLapConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "slow single-lap configuration")
        if not isinstance(value["conditions"], list) or not isinstance(
            value["schedule_kinds"], list
        ):
            raise RotatedConfigError("conditions and schedule_kinds must be lists")
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
            conditions=tuple(str(item) for item in value["conditions"]),
            schedule_kinds=tuple(str(item) for item in value["schedule_kinds"]),
            sigmoid_kappa=_positive_float(
                value["sigmoid_kappa"], name="sigmoid_kappa"
            ),
            rotation=RotationConfig.from_mapping(value["rotation"]),
            data=DataConfig.from_mapping(value["data"]),
            initialization=InitializationConfig.from_mapping(
                value["initialization"]
            ),
            learner=Phase3LearnerConfig.from_mapping(value["learner"]),
            fisher=Phase3FisherConfig.from_mapping(value["fisher"]),
            controller=DoubleLapControllerConfig.from_mapping_with_contract(
                value["controller"],
                expected_trend_half_life=7.5,
                contract_name="slow single-lap",
            ),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != SINGLE_LAP_CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported slow single-lap config schema")
        if self.metric_schema_version != SINGLE_LAP_METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported slow single-lap metric schema")
        if self.artifact_schema_version != SINGLE_LAP_ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported slow single-lap artifact schema")
        if not self.experiment.startswith("rotated_mnist_phase5_slow_single_lap_"):
            raise RotatedConfigError("slow single-lap experiment name is invalid")
        if not self.replica_id or self.replica_seed >= 2**63:
            raise RotatedConfigError("slow single-lap replica identity is invalid")
        if self.conditions != SINGLE_LAP_CONDITIONS:
            raise RotatedConfigError(
                f"slow single-lap conditions must be {list(SINGLE_LAP_CONDITIONS)}"
            )
        if self.schedule_kinds != SINGLE_LAP_SCHEDULES:
            raise RotatedConfigError(
                "slow single-lap requires linear and sigmoid schedules"
            )
        if not math.isclose(self.sigmoid_kappa, 8.0, rel_tol=0.0, abs_tol=1e-15):
            raise RotatedConfigError("slow single-lap sigmoid kappa is frozen at 8")
        if self.rotation.knots_degrees != SINGLE_LAP_KNOTS:
            raise RotatedConfigError("slow single-lap rotation knots are frozen")
        expected_transitions = 2 if "smoke" in self.experiment else 40
        if self.rotation.transitions_per_arrow != expected_transitions:
            raise RotatedConfigError(
                "slow single-lap transitions per leg are frozen at "
                f"{expected_transitions}"
            )
        if self.data.samples_per_step != 4 or self.data.stream_width != 4:
            raise RotatedConfigError("slow single-lap experiment is frozen at m=4")
        if self.fisher.initial_sample_size > self.data.reference_pool_size:
            raise RotatedConfigError(
                "initial Fisher sample size exceeds the reference partition"
            )
        self.rotation.validate()


def load_single_lap_config(path: str | Path) -> RotatedSlowSingleLapConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("slow single-lap configuration must be an object")
    return RotatedSlowSingleLapConfig.from_mapping(value)
