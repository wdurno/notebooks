"""Strict configurations for Plan 9 gain studies E9.12--E9.14."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import RotatedConfigError, RuntimeConfig, _integer, _strict_keys
from .config import (
    PLAN9_ARTIFACT_SCHEMA_VERSION,
    PLAN9_CONFIG_SCHEMA_VERSION,
    PLAN9_METRIC_SCHEMA_VERSION,
    _positive,
    _relative_path,
)


E9_1_RUN_ID = (
    "rotated_mnist_plan9_e9_1_anchor_cancellation__"
    "replica-0001__b1234a5779ce2e42"
)
E9_9_RUN_ID = (
    "rotated_mnist_plan9_e9_9_affinity_attribution_v2__"
    "replica-0001__c40471e46aa06d8f"
)


def _probability(value: Any, *, name: str) -> float:
    result = _positive(value, name=name)
    if result > 1.0:
        raise RotatedConfigError(f"{name} must not exceed one")
    return result


@dataclasses.dataclass(frozen=True)
class GainContractConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    study: str
    experiment: str
    replica_id: str
    replica_seed: int
    e9_1_run_path: str
    e9_1_run_id: str
    e9_9_run_path: str
    e9_9_run_id: str
    condition: str
    fisher_rank: int
    scalar_half_lives_steps: tuple[float, ...]
    primary_half_life_steps: float
    trace_relative_tolerance: float
    displacement_relative_tolerance: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GainContractConfig":
        _strict_keys(value, {field.name for field in dataclasses.fields(cls)}, "E9.12")
        half_lives = value["scalar_half_lives_steps"]
        if not isinstance(half_lives, list):
            raise RotatedConfigError("scalar_half_lives_steps must be a list")
        result = cls(
            schema_version=_integer(value["schema_version"], minimum=1, name="schema_version"),
            metric_schema_version=_integer(
                value["metric_schema_version"], minimum=1, name="metric_schema_version"
            ),
            artifact_schema_version=_integer(
                value["artifact_schema_version"], minimum=1, name="artifact_schema_version"
            ),
            study=str(value["study"]),
            experiment=str(value["experiment"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(value["replica_seed"], minimum=0, name="replica_seed"),
            e9_1_run_path=_relative_path(value["e9_1_run_path"], name="e9_1_run_path"),
            e9_1_run_id=str(value["e9_1_run_id"]),
            e9_9_run_path=_relative_path(value["e9_9_run_path"], name="e9_9_run_path"),
            e9_9_run_id=str(value["e9_9_run_id"]),
            condition=str(value["condition"]),
            fisher_rank=_integer(value["fisher_rank"], minimum=1, name="fisher_rank"),
            scalar_half_lives_steps=tuple(
                _positive(item, name="scalar_half_lives_steps") for item in half_lives
            ),
            primary_half_life_steps=_positive(
                value["primary_half_life_steps"], name="primary_half_life_steps"
            ),
            trace_relative_tolerance=_positive(
                value["trace_relative_tolerance"], name="trace_relative_tolerance"
            ),
            displacement_relative_tolerance=_positive(
                value["displacement_relative_tolerance"],
                name="displacement_relative_tolerance",
            ),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != PLAN9_CONFIG_SCHEMA_VERSION
            or self.metric_schema_version != PLAN9_METRIC_SCHEMA_VERSION
            or self.artifact_schema_version != PLAN9_ARTIFACT_SCHEMA_VERSION
            or self.study != "e9_12"
            or not self.experiment.startswith("rotated_mnist_plan9_e9_12_")
        ):
            raise RotatedConfigError("invalid E9.12 identity or schema")
        if Path(self.e9_1_run_path).name != self.e9_1_run_id or self.e9_1_run_id != E9_1_RUN_ID:
            raise RotatedConfigError("E9.12 E9.1 source is not authoritative")
        if Path(self.e9_9_run_path).name != self.e9_9_run_id or self.e9_9_run_id != E9_9_RUN_ID:
            raise RotatedConfigError("E9.12 E9.9 source is not authoritative")
        if (
            self.condition != "decomposed_edr"
            or self.fisher_rank != 8
            or self.scalar_half_lives_steps != (4.0, 8.0, 16.0)
            or self.primary_half_life_steps != 8.0
        ):
            raise RotatedConfigError("E9.12 treatment constants drifted")
        if self.runtime.device != "cuda" or self.runtime.dtype != "float32":
            raise RotatedConfigError("E9.12 authoritative replay requires CUDA float32")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["scalar_half_lives_steps"] = list(self.scalar_half_lives_steps)
        return value

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_mapping(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @property
    def run_id(self) -> str:
        return f"{self.experiment}__{self.replica_id}__{self.config_hash[:16]}"


@dataclasses.dataclass(frozen=True)
class GainEstimatorConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    study: str
    experiment: str
    replica_id: str
    replica_seed: int
    e9_12_run_path: str
    e9_12_run_id: str
    fast_half_lives_steps: tuple[float, ...]
    primary_half_life_steps: float
    fisher_rank: int
    relative_damping: float
    solver_residual_tolerance: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GainEstimatorConfig":
        _strict_keys(value, {field.name for field in dataclasses.fields(cls)}, "E9.13")
        half_lives = value["fast_half_lives_steps"]
        if not isinstance(half_lives, list):
            raise RotatedConfigError("fast_half_lives_steps must be a list")
        result = cls(
            schema_version=_integer(value["schema_version"], minimum=1, name="schema_version"),
            metric_schema_version=_integer(value["metric_schema_version"], minimum=1, name="metric_schema_version"),
            artifact_schema_version=_integer(value["artifact_schema_version"], minimum=1, name="artifact_schema_version"),
            study=str(value["study"]),
            experiment=str(value["experiment"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(value["replica_seed"], minimum=0, name="replica_seed"),
            e9_12_run_path=_relative_path(value["e9_12_run_path"], name="e9_12_run_path"),
            e9_12_run_id=str(value["e9_12_run_id"]),
            fast_half_lives_steps=tuple(_positive(item, name="fast_half_lives_steps") for item in half_lives),
            primary_half_life_steps=_positive(value["primary_half_life_steps"], name="primary_half_life_steps"),
            fisher_rank=_integer(value["fisher_rank"], minimum=1, name="fisher_rank"),
            relative_damping=_positive(value["relative_damping"], name="relative_damping"),
            solver_residual_tolerance=_positive(
                value["solver_residual_tolerance"], name="solver_residual_tolerance"
            ),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != PLAN9_CONFIG_SCHEMA_VERSION
            or self.metric_schema_version != PLAN9_METRIC_SCHEMA_VERSION
            or self.artifact_schema_version != PLAN9_ARTIFACT_SCHEMA_VERSION
            or self.study != "e9_13"
            or not self.experiment.startswith("rotated_mnist_plan9_e9_13_")
        ):
            raise RotatedConfigError("invalid E9.13 identity or schema")
        if Path(self.e9_12_run_path).name != self.e9_12_run_id:
            raise RotatedConfigError("E9.13 source path and ID differ")
        if (
            self.fast_half_lives_steps != (4.0, 8.0, 16.0)
            or self.primary_half_life_steps != 8.0
            or self.fisher_rank != 8
        ):
            raise RotatedConfigError("E9.13 estimator family drifted")
        if self.runtime.device != "cpu" or self.runtime.dtype != "float64":
            raise RotatedConfigError("E9.13 structured arithmetic requires CPU float64")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["fast_half_lives_steps"] = list(self.fast_half_lives_steps)
        return value

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_mapping(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @property
    def run_id(self) -> str:
        return f"{self.experiment}__{self.replica_id}__{self.config_hash[:16]}"


@dataclasses.dataclass(frozen=True)
class GainCalibrationConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    study: str
    experiment: str
    replica_id: str
    replica_seed: int
    e9_13_run_path: str
    e9_13_run_id: str
    mean_energy_reduction_threshold: float
    minimum_passing_groups: int
    reversal_ratio_limit: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GainCalibrationConfig":
        _strict_keys(value, {field.name for field in dataclasses.fields(cls)}, "E9.14")
        result = cls(
            schema_version=_integer(value["schema_version"], minimum=1, name="schema_version"),
            metric_schema_version=_integer(value["metric_schema_version"], minimum=1, name="metric_schema_version"),
            artifact_schema_version=_integer(value["artifact_schema_version"], minimum=1, name="artifact_schema_version"),
            study=str(value["study"]),
            experiment=str(value["experiment"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(value["replica_seed"], minimum=0, name="replica_seed"),
            e9_13_run_path=_relative_path(value["e9_13_run_path"], name="e9_13_run_path"),
            e9_13_run_id=str(value["e9_13_run_id"]),
            mean_energy_reduction_threshold=_probability(
                value["mean_energy_reduction_threshold"], name="mean_energy_reduction_threshold"
            ),
            minimum_passing_groups=_integer(
                value["minimum_passing_groups"], minimum=1, name="minimum_passing_groups"
            ),
            reversal_ratio_limit=_positive(value["reversal_ratio_limit"], name="reversal_ratio_limit"),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != PLAN9_CONFIG_SCHEMA_VERSION
            or self.metric_schema_version != PLAN9_METRIC_SCHEMA_VERSION
            or self.artifact_schema_version != PLAN9_ARTIFACT_SCHEMA_VERSION
            or self.study != "e9_14"
            or not self.experiment.startswith("rotated_mnist_plan9_e9_14_")
        ):
            raise RotatedConfigError("invalid E9.14 identity or schema")
        if Path(self.e9_13_run_path).name != self.e9_13_run_id:
            raise RotatedConfigError("E9.14 source path and ID differ")
        if self.mean_energy_reduction_threshold != 0.25 or self.minimum_passing_groups != 3:
            raise RotatedConfigError("E9.14 decision threshold drifted")
        if self.runtime.device != "cpu" or self.runtime.dtype != "float64":
            raise RotatedConfigError("E9.14 analysis requires CPU float64")

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_mapping(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @property
    def run_id(self) -> str:
        return f"{self.experiment}__{self.replica_id}__{self.config_hash[:16]}"


def _load(path: str | Path, cls):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load gain config: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("gain configuration must be an object")
    return cls.from_mapping(value)


def load_gain_contract_config(path: str | Path) -> GainContractConfig:
    return _load(path, GainContractConfig)


def load_gain_estimator_config(path: str | Path) -> GainEstimatorConfig:
    return _load(path, GainEstimatorConfig)


def load_gain_calibration_config(path: str | Path) -> GainCalibrationConfig:
    return _load(path, GainCalibrationConfig)
