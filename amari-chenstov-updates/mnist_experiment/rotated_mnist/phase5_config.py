"""Strict configuration for the Plan 5 Phase 5 EDR challenge."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config import RotatedConfigError, _integer, _positive_float, _strict_keys


PHASE5_CONFIG_SCHEMA_VERSION = 1
PHASE5_METRIC_SCHEMA_VERSION = 1
PHASE5_ARTIFACT_SCHEMA_VERSION = 1
PHASE5_CONDITIONS = (
    "fixed_pi001",
    "fixed_pi0025",
    "fixed_pi005",
    "fixed_pi010",
    "edr_cold005",
)
PHASE5_NEW_CONDITIONS = tuple(
    condition for condition in PHASE5_CONDITIONS if condition != "fixed_pi005"
)
PHASE5_FIXED_PI = {
    "fixed_pi001": 0.01,
    "fixed_pi0025": 0.025,
    "fixed_pi005": 0.05,
    "fixed_pi010": 0.10,
}


@dataclasses.dataclass(frozen=True)
class Phase5ControllerConfig:
    risk_metric: str
    pi_min: float
    pi_max: float
    cold_start_pi: float
    trend_half_life_degrees: float
    action_half_life_steps: float
    trace_epsilon: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase5ControllerConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "controller")
        result = cls(
            risk_metric=str(value["risk_metric"]),
            pi_min=_positive_float(value["pi_min"], name="controller.pi_min"),
            pi_max=_positive_float(value["pi_max"], name="controller.pi_max"),
            cold_start_pi=_positive_float(
                value["cold_start_pi"], name="controller.cold_start_pi"
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
        if result.risk_metric != "fisher":
            raise RotatedConfigError("Phase 5 EDR risk metric must be Fisher")
        expected_values = {
            "pi_min": (result.pi_min, 0.01),
            "pi_max": (result.pi_max, 0.95),
            "cold_start_pi": (result.cold_start_pi, 0.05),
            "trend_half_life_degrees": (
                result.trend_half_life_degrees,
                7.5,
            ),
            "action_half_life_steps": (result.action_half_life_steps, 4.0),
        }
        for name, (actual, expected_value) in expected_values.items():
            if not math.isclose(
                actual, expected_value, rel_tol=0.0, abs_tol=1e-15
            ):
                raise RotatedConfigError(
                    f"Phase 5 controller {name} is frozen at {expected_value}"
                )
        return result


@dataclasses.dataclass(frozen=True)
class RotatedPhase5Config:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    source_phase4_run: str
    source_phase4_run_id: str
    source_phase4_config_hash: str
    conditions: tuple[str, ...]
    fixed_pis: tuple[float, ...]
    controller: Phase5ControllerConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RotatedPhase5Config":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "Phase 5 configuration")
        conditions = value["conditions"]
        fixed_pis = value["fixed_pis"]
        if not isinstance(conditions, list) or not isinstance(fixed_pis, list):
            raise RotatedConfigError("conditions and fixed_pis must be lists")
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
            source_phase4_run=str(value["source_phase4_run"]),
            source_phase4_run_id=str(value["source_phase4_run_id"]),
            source_phase4_config_hash=str(value["source_phase4_config_hash"]),
            conditions=tuple(str(item) for item in conditions),
            fixed_pis=tuple(float(item) for item in fixed_pis),
            controller=Phase5ControllerConfig.from_mapping(value["controller"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != PHASE5_CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 5 configuration schema")
        if self.metric_schema_version != PHASE5_METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 5 metric schema")
        if self.artifact_schema_version != PHASE5_ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 5 artifact schema")
        if not self.experiment.startswith("rotated_mnist_phase5_"):
            raise RotatedConfigError(
                "Phase 5 experiment must start with 'rotated_mnist_phase5_'"
            )
        if not self.replica_id or not self.source_phase4_run:
            raise RotatedConfigError("Phase 5 replica and source path must be nonempty")
        if self.conditions != PHASE5_CONDITIONS:
            raise RotatedConfigError(
                f"Phase 5 conditions must be {list(PHASE5_CONDITIONS)}"
            )
        if self.fixed_pis != (0.01, 0.025, 0.05, 0.10):
            raise RotatedConfigError("Phase 5 fixed pi bracket is frozen")
        for name, value in {
            "source_phase4_run_id": self.source_phase4_run_id,
            "source_phase4_config_hash": self.source_phase4_config_hash,
        }.items():
            if not value:
                raise RotatedConfigError(f"{name} must be nonempty")
        if len(self.source_phase4_config_hash) != 64:
            raise RotatedConfigError("source Phase 4 config hash must be SHA-256")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_schema_version": self.metric_schema_version,
            "artifact_schema_version": self.artifact_schema_version,
            "experiment": self.experiment,
            "replica_id": self.replica_id,
            "source_phase4_run": self.source_phase4_run,
            "source_phase4_run_id": self.source_phase4_run_id,
            "source_phase4_config_hash": self.source_phase4_config_hash,
            "conditions": list(self.conditions),
            "fixed_pis": list(self.fixed_pis),
            "controller": dataclasses.asdict(self.controller),
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


def load_phase5_config(path: str | Path) -> RotatedPhase5Config:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Phase 5 configuration must be a JSON object")
    return RotatedPhase5Config.from_mapping(value)
