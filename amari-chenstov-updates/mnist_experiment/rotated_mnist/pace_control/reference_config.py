"""Configuration for the Plan 10 multi-start population-reference repair."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import RotatedConfigError, RuntimeConfig, _integer, _strict_keys


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


def _relative(value: Any, name: str) -> str:
    result = str(value)
    path = Path(result)
    if not result or path.is_absolute() or ".." in path.parts:
        raise RotatedConfigError(f"{name} must be a repository-relative path")
    return result


@dataclasses.dataclass(frozen=True)
class ReferenceRepairConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    oracle_run_path: str
    oracle_run_id: str
    finite_risk_run_path: str
    finite_risk_run_id: str
    grid_increment_degrees: float
    candidate_start_count: int
    refinement_epochs: int
    learning_rate: float
    minimum_delta: float
    training_sample_size: int
    selection_size: int
    heldout_size: int
    batch_size: int
    minimum_own_best_fraction: float
    maximum_median_regret: float
    minimum_nonnegative_fraction: float
    minimum_local_nonnegative_fraction: float
    local_increment_limit_degrees: float
    minimum_monotone_fraction: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReferenceRepairConfig":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "Plan 10 reference-repair configuration",
        )
        result = cls(
            schema_version=_integer(value["schema_version"], minimum=1, name="schema_version"),
            metric_schema_version=_integer(value["metric_schema_version"], minimum=1, name="metric_schema_version"),
            artifact_schema_version=_integer(value["artifact_schema_version"], minimum=1, name="artifact_schema_version"),
            experiment=str(value["experiment"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(value["replica_seed"], minimum=0, name="replica_seed"),
            oracle_run_path=_relative(value["oracle_run_path"], "oracle_run_path"),
            oracle_run_id=str(value["oracle_run_id"]),
            finite_risk_run_path=_relative(value["finite_risk_run_path"], "finite_risk_run_path"),
            finite_risk_run_id=str(value["finite_risk_run_id"]),
            grid_increment_degrees=_positive(value["grid_increment_degrees"], "grid_increment_degrees"),
            candidate_start_count=_integer(value["candidate_start_count"], minimum=2, name="candidate_start_count"),
            refinement_epochs=_integer(value["refinement_epochs"], minimum=1, name="refinement_epochs"),
            learning_rate=_positive(value["learning_rate"], "learning_rate"),
            minimum_delta=_positive(value["minimum_delta"], "minimum_delta"),
            training_sample_size=_integer(value["training_sample_size"], minimum=1, name="training_sample_size"),
            selection_size=_integer(value["selection_size"], minimum=1, name="selection_size"),
            heldout_size=_integer(value["heldout_size"], minimum=1, name="heldout_size"),
            batch_size=_integer(value["batch_size"], minimum=1, name="batch_size"),
            minimum_own_best_fraction=_unit(value["minimum_own_best_fraction"], "minimum_own_best_fraction"),
            maximum_median_regret=_positive(value["maximum_median_regret"], "maximum_median_regret"),
            minimum_nonnegative_fraction=_unit(value["minimum_nonnegative_fraction"], "minimum_nonnegative_fraction"),
            minimum_local_nonnegative_fraction=_unit(value["minimum_local_nonnegative_fraction"], "minimum_local_nonnegative_fraction"),
            local_increment_limit_degrees=_positive(value["local_increment_limit_degrees"], "local_increment_limit_degrees"),
            minimum_monotone_fraction=_unit(value["minimum_monotone_fraction"], "minimum_monotone_fraction"),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (self.schema_version, self.metric_schema_version, self.artifact_schema_version) != (1, 1, 2):
            raise RotatedConfigError("unsupported Plan 10 reference-repair schema")
        if not self.experiment.startswith("rotated_mnist_plan10_phase1c_"):
            raise RotatedConfigError("invalid Plan 10 reference-repair experiment")
        if Path(self.oracle_run_path).name != self.oracle_run_id:
            raise RotatedConfigError("oracle path and run ID differ")
        if Path(self.finite_risk_run_path).name != self.finite_risk_run_id:
            raise RotatedConfigError("finite-risk path and run ID differ")
        frozen = (
            (self.grid_increment_degrees, 0.75),
            (self.candidate_start_count, 3),
            (self.refinement_epochs, 12),
            (self.training_sample_size, 10_000),
            (self.selection_size, 2_000),
            (self.heldout_size, 8_000),
            (self.local_increment_limit_degrees, 3.0),
        )
        if any(actual != expected for actual, expected in frozen):
            raise RotatedConfigError("Plan 10 reference-repair design drifted")
        if self.runtime.dtype != "float32":
            raise RotatedConfigError("reference repair requires float32 model fitting")

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_mapping(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        return f"{self.experiment}__{self.replica_id}__{self.config_hash[:16]}"


def load_reference_repair_config(path: str | Path) -> ReferenceRepairConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load Plan 10 reference-repair config: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Plan 10 reference-repair config must be an object")
    return ReferenceRepairConfig.from_mapping(value)
