"""Strict configuration for the Phase 5 double-lap EDR retry."""

from __future__ import annotations

import dataclasses
import hashlib
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


DOUBLE_LAP_CONFIG_SCHEMA_VERSION = 1
DOUBLE_LAP_METRIC_SCHEMA_VERSION = 1
DOUBLE_LAP_ARTIFACT_SCHEMA_VERSION = 1
DOUBLE_LAP_KNOTS = (0.0, 30.0, 0.0, 30.0)
DOUBLE_LAP_SCHEDULES = ("linear", "sigmoid")
DOUBLE_LAP_CONDITIONS = (
    "current_only",
    "fixed_pi0025",
    "fixed_pi005",
    "fixed_pi0075",
    "fixed_pi010",
    "edr_fasttrend_slowaction",
)
DOUBLE_LAP_FIXED_PI = {
    "fixed_pi0025": 0.025,
    "fixed_pi005": 0.05,
    "fixed_pi0075": 0.075,
    "fixed_pi010": 0.10,
}


@dataclasses.dataclass(frozen=True)
class DoubleLapControllerConfig:
    risk_metric: str
    pi_min: float
    pi_max: float
    cold_start_pi: float
    cold_start_steps: int
    trend_half_life_degrees: float
    action_half_life_steps: float
    trace_epsilon: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DoubleLapControllerConfig":
        return cls.from_mapping_with_contract(
            value,
            expected_trend_half_life=1.875,
            contract_name="double-lap",
        )

    @classmethod
    def from_mapping_with_contract(
        cls,
        value: Mapping[str, Any],
        *,
        expected_trend_half_life: float,
        contract_name: str,
    ) -> "DoubleLapControllerConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "controller")
        result = cls(
            risk_metric=str(value["risk_metric"]),
            pi_min=_positive_float(value["pi_min"], name="controller.pi_min"),
            pi_max=_positive_float(value["pi_max"], name="controller.pi_max"),
            cold_start_pi=_positive_float(
                value["cold_start_pi"], name="controller.cold_start_pi"
            ),
            cold_start_steps=_integer(
                value["cold_start_steps"],
                minimum=0,
                name="controller.cold_start_steps",
            ),
            trend_half_life_degrees=_positive_float(
                value["trend_half_life_degrees"],
                name="controller.trend_half_life_degrees",
            ),
            action_half_life_steps=_positive_float(
                value["action_half_life_steps"],
                name="controller.action_half_life_steps",
            ),
            trace_epsilon=_positive_float(
                value["trace_epsilon"], name="controller.trace_epsilon"
            ),
        )
        frozen = {
            "pi_min": (result.pi_min, 0.01),
            "pi_max": (result.pi_max, 0.95),
            "cold_start_pi": (result.cold_start_pi, 0.05),
            "trend_half_life_degrees": (
                result.trend_half_life_degrees,
                expected_trend_half_life,
            ),
            "action_half_life_steps": (result.action_half_life_steps, 8.0),
        }
        if result.risk_metric != "fisher":
            raise RotatedConfigError(f"{contract_name} EDR requires Fisher risk")
        if result.cold_start_steps != 8:
            raise RotatedConfigError(
                f"{contract_name} cold start is frozen at 8 updates"
            )
        for name, (actual, expected_value) in frozen.items():
            if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-15):
                raise RotatedConfigError(
                    f"{contract_name} controller {name} is frozen at {expected_value}"
                )
        return result


@dataclasses.dataclass(frozen=True)
class RotatedDoubleLapConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    conditions: tuple[str, ...]
    schedule_kinds: tuple[str, ...]
    sigmoid_kappa: float
    rotation: RotationConfig
    data: DataConfig
    initialization: InitializationConfig
    learner: Phase3LearnerConfig
    fisher: Phase3FisherConfig
    controller: DoubleLapControllerConfig
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RotatedDoubleLapConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "double-lap configuration")
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
            controller=DoubleLapControllerConfig.from_mapping(value["controller"]),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != DOUBLE_LAP_CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported double-lap configuration schema")
        if self.metric_schema_version != DOUBLE_LAP_METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported double-lap metric schema")
        if self.artifact_schema_version != DOUBLE_LAP_ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported double-lap artifact schema")
        if not self.experiment.startswith("rotated_mnist_phase5_double_lap_"):
            raise RotatedConfigError("double-lap experiment name is invalid")
        if not self.replica_id or self.replica_seed >= 2**63:
            raise RotatedConfigError("double-lap replica identity is invalid")
        if self.conditions != DOUBLE_LAP_CONDITIONS:
            raise RotatedConfigError(
                f"double-lap conditions must be {list(DOUBLE_LAP_CONDITIONS)}"
            )
        if self.schedule_kinds != DOUBLE_LAP_SCHEDULES:
            raise RotatedConfigError("double-lap requires linear and sigmoid schedules")
        if not math.isclose(self.sigmoid_kappa, 8.0, rel_tol=0.0, abs_tol=1e-15):
            raise RotatedConfigError("double-lap sigmoid kappa is frozen at 8")
        if self.rotation.knots_degrees != DOUBLE_LAP_KNOTS:
            raise RotatedConfigError("double-lap rotation knots are frozen")
        expected_transitions = 2 if "smoke" in self.experiment else 40
        if self.rotation.transitions_per_arrow != expected_transitions:
            raise RotatedConfigError(
                f"double-lap transitions per leg are frozen at {expected_transitions}"
            )
        if self.data.samples_per_step != 4 or self.data.stream_width != 4:
            raise RotatedConfigError("double-lap experiment is frozen at m=4")
        if self.fisher.initial_sample_size > self.data.reference_pool_size:
            raise RotatedConfigError(
                "initial Fisher sample size exceeds the reference partition"
            )
        self.rotation.validate()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_schema_version": self.metric_schema_version,
            "artifact_schema_version": self.artifact_schema_version,
            "experiment": self.experiment,
            "replica_id": self.replica_id,
            "replica_seed": self.replica_seed,
            "conditions": list(self.conditions),
            "schedule_kinds": list(self.schedule_kinds),
            "sigmoid_kappa": self.sigmoid_kappa,
            "rotation": self.rotation.to_mapping(),
            "data": dataclasses.asdict(self.data),
            "initialization": dataclasses.asdict(self.initialization),
            "learner": dataclasses.asdict(self.learner),
            "fisher": dataclasses.asdict(self.fisher),
            "controller": dataclasses.asdict(self.controller),
            "runtime": dataclasses.asdict(self.runtime),
        }

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


def load_double_lap_config(path: str | Path) -> RotatedDoubleLapConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("double-lap configuration must be an object")
    return RotatedDoubleLapConfig.from_mapping(value)
