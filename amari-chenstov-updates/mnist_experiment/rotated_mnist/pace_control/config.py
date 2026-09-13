"""Strict portable configurations for Plan 10 feasibility and fidelity studies."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import RotatedConfigError, RuntimeConfig, _integer, _strict_keys


PLAN10_SCHEMA_VERSION = 1
PLAN10_METRIC_SCHEMA_VERSION = 1
PLAN10_ARTIFACT_SCHEMA_VERSION = 2


def _positive(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RotatedConfigError(f"{name} must be finite and positive")
    return float(value)


def _unit(value: Any, name: str) -> float:
    result = _positive(value, name)
    if result >= 1.0:
        raise RotatedConfigError(f"{name} must be below one")
    return result


def _relative_path(value: Any, name: str) -> str:
    result = str(value)
    path = Path(result)
    if not result or path.is_absolute() or ".." in path.parts:
        raise RotatedConfigError(f"{name} must be a repository-relative path")
    return result


@dataclasses.dataclass(frozen=True)
class FeasibilityConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    oracle_run_path: str
    oracle_run_id: str
    fixed_pi: float
    batch_size: int
    initial_q: float
    fisher_rank: int
    local_mle_sample_size: int
    local_mle_replicates: int
    map_increment_degrees: float
    validation_increments_degrees: tuple[float, ...]
    minimum_pace_degrees: float
    maximum_pace_degrees: float
    route_knots_degrees: tuple[float, ...]
    maximum_route_steps: int
    maximum_median_relative_error: float
    minimum_monotone_fraction: float
    maximum_bound_fraction: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeasibilityConfig":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "Plan 10 feasibility configuration",
        )
        result = cls(
            schema_version=_integer(value["schema_version"], minimum=1, name="schema_version"),
            metric_schema_version=_integer(value["metric_schema_version"], minimum=1, name="metric_schema_version"),
            artifact_schema_version=_integer(value["artifact_schema_version"], minimum=1, name="artifact_schema_version"),
            experiment=str(value["experiment"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(value["replica_seed"], minimum=0, name="replica_seed"),
            oracle_run_path=_relative_path(value["oracle_run_path"], "oracle_run_path"),
            oracle_run_id=str(value["oracle_run_id"]),
            fixed_pi=_unit(value["fixed_pi"], "fixed_pi"),
            batch_size=_integer(value["batch_size"], minimum=1, name="batch_size"),
            initial_q=_positive(value["initial_q"], "initial_q"),
            fisher_rank=_integer(value["fisher_rank"], minimum=1, name="fisher_rank"),
            local_mle_sample_size=_integer(value["local_mle_sample_size"], minimum=2, name="local_mle_sample_size"),
            local_mle_replicates=_integer(value["local_mle_replicates"], minimum=2, name="local_mle_replicates"),
            map_increment_degrees=_positive(value["map_increment_degrees"], "map_increment_degrees"),
            validation_increments_degrees=tuple(_positive(item, "validation_increments_degrees") for item in value["validation_increments_degrees"]),
            minimum_pace_degrees=_positive(value["minimum_pace_degrees"], "minimum_pace_degrees"),
            maximum_pace_degrees=_positive(value["maximum_pace_degrees"], "maximum_pace_degrees"),
            route_knots_degrees=tuple(float(item) for item in value["route_knots_degrees"]),
            maximum_route_steps=_integer(value["maximum_route_steps"], minimum=1, name="maximum_route_steps"),
            maximum_median_relative_error=_positive(value["maximum_median_relative_error"], "maximum_median_relative_error"),
            minimum_monotone_fraction=_unit(value["minimum_monotone_fraction"], "minimum_monotone_fraction"),
            maximum_bound_fraction=_unit(value["maximum_bound_fraction"], "maximum_bound_fraction"),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (self.schema_version, self.metric_schema_version, self.artifact_schema_version) != (1, 1, 2):
            raise RotatedConfigError("unsupported Plan 10 feasibility schema")
        if not self.experiment.startswith("rotated_mnist_plan10_phase1_"):
            raise RotatedConfigError("invalid Plan 10 feasibility experiment")
        if Path(self.oracle_run_path).name != self.oracle_run_id:
            raise RotatedConfigError("oracle path and run ID differ")
        if self.minimum_pace_degrees >= self.maximum_pace_degrees:
            raise RotatedConfigError("pace bounds are not ordered")
        if self.validation_increments_degrees != (0.75, 1.5):
            raise RotatedConfigError("Phase 1 validation increments are frozen")
        if self.route_knots_degrees != (0.0, 15.0, 30.0, 0.0, 15.0, 30.0):
            raise RotatedConfigError("Phase 1 route is frozen")
        if any(not 0.0 <= angle <= 30.0 for angle in self.route_knots_degrees):
            raise RotatedConfigError("route angle is outside [0, 30]")
        if self.runtime.device != "cpu" or self.runtime.dtype != "float64":
            raise RotatedConfigError("Phase 1 is an artifact-only CPU float64 audit")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["validation_increments_degrees"] = list(self.validation_increments_degrees)
        value["route_knots_degrees"] = list(self.route_knots_degrees)
        return value

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_mapping(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        return f"{self.experiment}__{self.replica_id}__{self.config_hash[:16]}"


def load_feasibility_config(path: str | Path) -> FeasibilityConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load Plan 10 feasibility config: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Plan 10 feasibility config must be an object")
    return FeasibilityConfig.from_mapping(value)
