"""Strict configuration for the Plan 10 Phase 1d response surface."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import RotatedConfigError, RuntimeConfig, _integer, _strict_keys


PACE_GRID = (0.0, 0.375, 0.75, 1.125, 1.5, 3.0)
PI_GRID = (0.0125, 0.025, 0.0375, 0.05, 0.075, 0.10, 0.15)
ANCHORS_BY_STAGE = {
    "smoke": (20, 60),
    "coarse": (20, 60),
    "full": (20, 40, 60, 100),
    "expansion": (20, 40, 60, 100),
}
REPLICATES_BY_STAGE = {"smoke": 2, "coarse": 16, "full": 32, "expansion": 128}


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


def _optional_relative(value: Any, name: str) -> str | None:
    return None if value is None else _relative(value, name)


@dataclasses.dataclass(frozen=True)
class ResponseSurfaceConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    stage: str
    replica_id: str
    replica_seed: int
    source_run_path: str
    source_run_id: str
    prerequisite_run_path: str | None
    prerequisite_run_id: str | None
    source_schedule: str
    source_condition: str
    anchor_steps: tuple[int, ...]
    pace_degrees: tuple[float, ...]
    pi_values: tuple[float, ...]
    fixed_pi: float
    batch_size: int
    replicate_count: int
    evaluation_batch_size: int
    bootstrap_replicates: int
    confidence_level: float
    practical_nll_margin: float
    physical_pace_minimum: float
    physical_pace_maximum: float
    minimum_monotone_fraction: float
    minimum_interior_anchor_count: int
    maximum_fit_failure_fraction: float
    crossing_low_pi: float
    crossing_high_pi: float
    checkpoint_interval_units: int
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResponseSurfaceConfig":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "Plan 10 response-surface configuration",
        )
        result = cls(
            schema_version=_integer(value["schema_version"], minimum=1, name="schema_version"),
            metric_schema_version=_integer(value["metric_schema_version"], minimum=1, name="metric_schema_version"),
            artifact_schema_version=_integer(value["artifact_schema_version"], minimum=1, name="artifact_schema_version"),
            experiment=str(value["experiment"]),
            stage=str(value["stage"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(value["replica_seed"], minimum=0, name="replica_seed"),
            source_run_path=_relative(value["source_run_path"], "source_run_path"),
            source_run_id=str(value["source_run_id"]),
            prerequisite_run_path=_optional_relative(
                value["prerequisite_run_path"], "prerequisite_run_path"
            ),
            prerequisite_run_id=(
                None
                if value["prerequisite_run_id"] is None
                else str(value["prerequisite_run_id"])
            ),
            source_schedule=str(value["source_schedule"]),
            source_condition=str(value["source_condition"]),
            anchor_steps=tuple(
                _integer(item, minimum=1, name="anchor_steps")
                for item in value["anchor_steps"]
            ),
            pace_degrees=tuple(float(item) for item in value["pace_degrees"]),
            pi_values=tuple(float(item) for item in value["pi_values"]),
            fixed_pi=_unit(value["fixed_pi"], "fixed_pi"),
            batch_size=_integer(value["batch_size"], minimum=1, name="batch_size"),
            replicate_count=_integer(value["replicate_count"], minimum=1, name="replicate_count"),
            evaluation_batch_size=_integer(value["evaluation_batch_size"], minimum=1, name="evaluation_batch_size"),
            bootstrap_replicates=_integer(value["bootstrap_replicates"], minimum=100, name="bootstrap_replicates"),
            confidence_level=_unit(value["confidence_level"], "confidence_level"),
            practical_nll_margin=_positive(value["practical_nll_margin"], "practical_nll_margin"),
            physical_pace_minimum=_positive(value["physical_pace_minimum"], "physical_pace_minimum"),
            physical_pace_maximum=_positive(value["physical_pace_maximum"], "physical_pace_maximum"),
            minimum_monotone_fraction=_unit(value["minimum_monotone_fraction"], "minimum_monotone_fraction"),
            minimum_interior_anchor_count=_integer(value["minimum_interior_anchor_count"], minimum=1, name="minimum_interior_anchor_count"),
            maximum_fit_failure_fraction=float(value["maximum_fit_failure_fraction"]),
            crossing_low_pi=_unit(value["crossing_low_pi"], "crossing_low_pi"),
            crossing_high_pi=_unit(value["crossing_high_pi"], "crossing_high_pi"),
            checkpoint_interval_units=_integer(value["checkpoint_interval_units"], minimum=1, name="checkpoint_interval_units"),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (self.schema_version, self.metric_schema_version, self.artifact_schema_version) != (1, 1, 2):
            raise RotatedConfigError("unsupported Plan 10 response-surface schema")
        if self.stage not in ANCHORS_BY_STAGE:
            raise RotatedConfigError("response-surface stage is invalid")
        if not self.experiment.startswith(
            f"rotated_mnist_plan10_phase1d_{self.stage}_"
        ):
            raise RotatedConfigError("response-surface experiment does not match its stage")
        if Path(self.source_run_path).name != self.source_run_id:
            raise RotatedConfigError("response-surface source path and run ID differ")
        if (self.prerequisite_run_path is None) != (self.prerequisite_run_id is None):
            raise RotatedConfigError("response-surface prerequisite fields must agree")
        if self.stage == "smoke":
            if self.prerequisite_run_path is not None:
                raise RotatedConfigError("response-surface smoke cannot have a prerequisite")
        elif (
            self.prerequisite_run_path is None
            or Path(self.prerequisite_run_path).name != self.prerequisite_run_id
        ):
            raise RotatedConfigError("response-surface stage requires its immutable prerequisite")
        if self.source_schedule != "linear" or self.source_condition != "fixed_pi005":
            raise RotatedConfigError("response-surface source treatment is frozen")
        if self.anchor_steps != ANCHORS_BY_STAGE[self.stage]:
            raise RotatedConfigError("response-surface anchors drifted")
        if self.pace_degrees != PACE_GRID or self.pi_values != PI_GRID:
            raise RotatedConfigError("response-surface treatment grid drifted")
        if self.fixed_pi != 0.05 or self.batch_size != 4:
            raise RotatedConfigError("response-surface learner contract drifted")
        if self.replicate_count != REPLICATES_BY_STAGE[self.stage]:
            raise RotatedConfigError("response-surface replicate count drifted")
        if self.confidence_level != 0.95 or self.practical_nll_margin != 0.002:
            raise RotatedConfigError("response-surface decision threshold drifted")
        if (self.physical_pace_minimum, self.physical_pace_maximum) != (0.375, 1.5):
            raise RotatedConfigError("response-surface physical pace bounds drifted")
        if self.minimum_monotone_fraction != 0.8:
            raise RotatedConfigError("response-surface monotonicity gate drifted")
        expected_interior = 1 if self.stage in {"smoke", "coarse"} else 3
        if self.minimum_interior_anchor_count != expected_interior:
            raise RotatedConfigError("response-surface interior gate drifted")
        if self.maximum_fit_failure_fraction != 0.01:
            raise RotatedConfigError("response-surface failure gate drifted")
        if (self.crossing_low_pi, self.crossing_high_pi) != (0.025, 0.075):
            raise RotatedConfigError("response-surface crossing contrast drifted")
        if self.runtime.device != "cuda" or self.runtime.dtype != "float32":
            raise RotatedConfigError("response-surface execution requires CUDA float32")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["anchor_steps"] = list(self.anchor_steps)
        value["pace_degrees"] = list(self.pace_degrees)
        value["pi_values"] = list(self.pi_values)
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


def load_response_surface_config(path: str | Path) -> ResponseSurfaceConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(
            f"could not load Plan 10 response-surface config: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Plan 10 response-surface config must be an object")
    return ResponseSurfaceConfig.from_mapping(value)
