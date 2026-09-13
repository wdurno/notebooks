"""Strict configuration for the Plan 10 finite excess-risk recovery."""

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
class FiniteRiskConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    oracle_run_path: str
    oracle_run_id: str
    derivative_run_path: str
    derivative_run_id: str
    fixed_pi: float
    batch_size: int
    initial_q: float
    fisher_rank: int
    local_mle_sample_size: int
    local_mle_replicates: int
    grid_increment_degrees: float
    heldout_offset: int
    heldout_size: int
    evaluation_batch_size: int
    route_knots_degrees: tuple[float, ...]
    maximum_route_steps: int
    minimum_nonnegative_fraction: float
    minimum_monotone_fraction: float
    minimum_target_attainment_fraction: float
    maximum_median_target_error: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FiniteRiskConfig":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "Plan 10 finite-risk configuration",
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
            derivative_run_path=_relative(value["derivative_run_path"], "derivative_run_path"),
            derivative_run_id=str(value["derivative_run_id"]),
            fixed_pi=_unit(value["fixed_pi"], "fixed_pi"),
            batch_size=_integer(value["batch_size"], minimum=1, name="batch_size"),
            initial_q=_positive(value["initial_q"], "initial_q"),
            fisher_rank=_integer(value["fisher_rank"], minimum=1, name="fisher_rank"),
            local_mle_sample_size=_integer(value["local_mle_sample_size"], minimum=2, name="local_mle_sample_size"),
            local_mle_replicates=_integer(value["local_mle_replicates"], minimum=2, name="local_mle_replicates"),
            grid_increment_degrees=_positive(value["grid_increment_degrees"], "grid_increment_degrees"),
            heldout_offset=_integer(value["heldout_offset"], minimum=0, name="heldout_offset"),
            heldout_size=_integer(value["heldout_size"], minimum=1, name="heldout_size"),
            evaluation_batch_size=_integer(value["evaluation_batch_size"], minimum=1, name="evaluation_batch_size"),
            route_knots_degrees=tuple(float(item) for item in value["route_knots_degrees"]),
            maximum_route_steps=_integer(value["maximum_route_steps"], minimum=1, name="maximum_route_steps"),
            minimum_nonnegative_fraction=_unit(value["minimum_nonnegative_fraction"], "minimum_nonnegative_fraction"),
            minimum_monotone_fraction=_unit(value["minimum_monotone_fraction"], "minimum_monotone_fraction"),
            minimum_target_attainment_fraction=_unit(value["minimum_target_attainment_fraction"], "minimum_target_attainment_fraction"),
            maximum_median_target_error=_unit(value["maximum_median_target_error"], "maximum_median_target_error"),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (self.schema_version, self.metric_schema_version, self.artifact_schema_version) != (1, 1, 2):
            raise RotatedConfigError("unsupported Plan 10 finite-risk schema")
        if not self.experiment.startswith("rotated_mnist_plan10_phase1b_"):
            raise RotatedConfigError("invalid Plan 10 finite-risk experiment")
        if Path(self.oracle_run_path).name != self.oracle_run_id:
            raise RotatedConfigError("oracle path and run ID differ")
        if Path(self.derivative_run_path).name != self.derivative_run_id:
            raise RotatedConfigError("derivative path and run ID differ")
        if self.route_knots_degrees != (0.0, 15.0, 30.0, 0.0, 15.0, 30.0):
            raise RotatedConfigError("finite-risk route is frozen")
        if self.grid_increment_degrees != 0.75:
            raise RotatedConfigError("finite-risk grid is frozen at 0.75 degrees")
        if self.heldout_offset != 2_000 or self.heldout_size != 8_000:
            raise RotatedConfigError("finite-risk held-out panel is frozen")
        if self.runtime.dtype != "float32":
            raise RotatedConfigError("finite-risk model evaluation requires float32")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
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


def load_finite_risk_config(path: str | Path) -> FiniteRiskConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load Plan 10 finite-risk config: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Plan 10 finite-risk config must be an object")
    return FiniteRiskConfig.from_mapping(value)
