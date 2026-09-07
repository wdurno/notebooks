"""Strict configuration for the decomposed-EDR rechallenge."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config import (
    RotatedConfigError,
    RuntimeConfig,
    _integer,
    _positive_float,
    _strict_keys,
)


PHASE8_CONFIG_SCHEMA_VERSION = 1
PHASE8_METRIC_SCHEMA_VERSION = 1
PHASE8_ARTIFACT_SCHEMA_VERSION = 1
PHASE8_SOURCE_KINDS = ("single_lap", "double_lap", "canonical")
PHASE8_CONDITIONS = (
    "fixed_pi005_sentinel",
    "fixed_pi0025",
    "tracked_q_covariance",
    "decomposed_edr",
    "hybrid_b032_decomposed_edr",
)


@dataclasses.dataclass(frozen=True)
class Phase8ControllerConfig:
    pi_min: float
    pi_max: float
    cold_start_pi: float
    cold_start_steps: int
    trend_half_life_degrees: float
    movement_half_life_steps: float
    trace_epsilon: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase8ControllerConfig":
        _strict_keys(value, {field.name for field in dataclasses.fields(cls)}, "controller")
        result = cls(
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
            movement_half_life_steps=_positive_float(
                value["movement_half_life_steps"],
                name="controller.movement_half_life_steps",
            ),
            trace_epsilon=_positive_float(
                value["trace_epsilon"], name="controller.trace_epsilon"
            ),
        )
        if result.pi_min > result.pi_max:
            raise RotatedConfigError("controller.pi_min cannot exceed pi_max")
        for name, actual, expected in (
            ("pi_min", result.pi_min, 0.01),
            ("pi_max", result.pi_max, 0.95),
            ("cold_start_pi", result.cold_start_pi, 0.05),
            ("movement_half_life_steps", result.movement_half_life_steps, 8.0),
        ):
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
                raise RotatedConfigError(f"Plan 8 controller {name} is frozen at {expected}")
        return result


@dataclasses.dataclass(frozen=True)
class Phase8CompatibilityConfig:
    accuracy_auc_tolerance: float
    nll_auc_tolerance: float
    final_accuracy_tolerance: float

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "Phase8CompatibilityConfig":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "compatibility",
        )
        return cls(
            accuracy_auc_tolerance=_positive_float(
                value["accuracy_auc_tolerance"],
                name="compatibility.accuracy_auc_tolerance",
            ),
            nll_auc_tolerance=_positive_float(
                value["nll_auc_tolerance"],
                name="compatibility.nll_auc_tolerance",
            ),
            final_accuracy_tolerance=_positive_float(
                value["final_accuracy_tolerance"],
                name="compatibility.final_accuracy_tolerance",
            ),
        )


@dataclasses.dataclass(frozen=True)
class Phase8Config:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    source_kind: str
    source_artifact: str
    conditions: tuple[str, ...]
    transition_limit: int | None
    controller: Phase8ControllerConfig
    compatibility: Phase8CompatibilityConfig
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase8Config":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "Plan 8 configuration",
        )
        conditions = value["conditions"]
        if not isinstance(conditions, list):
            raise RotatedConfigError("conditions must be a list")
        limit = value["transition_limit"]
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
            source_kind=str(value["source_kind"]),
            source_artifact=str(value["source_artifact"]),
            conditions=tuple(str(item) for item in conditions),
            transition_limit=(
                None
                if limit is None
                else _integer(limit, minimum=1, name="transition_limit")
            ),
            controller=Phase8ControllerConfig.from_mapping(value["controller"]),
            compatibility=Phase8CompatibilityConfig.from_mapping(
                value["compatibility"]
            ),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != PHASE8_CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Plan 8 configuration schema")
        if self.metric_schema_version != PHASE8_METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Plan 8 metric schema")
        if self.artifact_schema_version != PHASE8_ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Plan 8 artifact schema")
        if not self.experiment.startswith("rotated_mnist_phase8_"):
            raise RotatedConfigError("Plan 8 experiment name is invalid")
        if not self.replica_id or self.replica_seed >= 2**63:
            raise RotatedConfigError("Plan 8 replica identity is invalid")
        if self.source_kind not in PHASE8_SOURCE_KINDS:
            raise RotatedConfigError("unsupported Plan 8 source kind")
        if not self.source_artifact:
            raise RotatedConfigError("source_artifact must be nonempty")
        if (
            not self.conditions
            or len(set(self.conditions)) != len(self.conditions)
            or any(item not in PHASE8_CONDITIONS for item in self.conditions)
        ):
            raise RotatedConfigError("Plan 8 conditions are invalid")
        if "decomposed_edr" not in self.conditions:
            raise RotatedConfigError("Plan 8 requires decomposed_edr")
        expected_trend = 1.875 if self.source_kind == "double_lap" else 7.5
        if not math.isclose(
            self.controller.trend_half_life_degrees,
            expected_trend,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RotatedConfigError(
                f"{self.source_kind} trend half-life is frozen at {expected_trend}"
            )
        expected_cold = 1 if "smoke" in self.experiment else 8
        if self.controller.cold_start_steps != expected_cold:
            raise RotatedConfigError(
                f"Plan 8 cold_start_steps is frozen at {expected_cold}"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_schema_version": self.metric_schema_version,
            "artifact_schema_version": self.artifact_schema_version,
            "experiment": self.experiment,
            "replica_id": self.replica_id,
            "replica_seed": self.replica_seed,
            "source_kind": self.source_kind,
            "source_artifact": self.source_artifact,
            "conditions": list(self.conditions),
            "transition_limit": self.transition_limit,
            "controller": dataclasses.asdict(self.controller),
            "compatibility": dataclasses.asdict(self.compatibility),
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


def load_phase8_config(path: str | Path) -> Phase8Config:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Plan 8 configuration must be an object")
    return Phase8Config.from_mapping(value)
