"""Strict configuration for the paired Plan 5 Phase 3 transfer pilot."""

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


PHASE3_CONFIG_SCHEMA_VERSION = 1
PHASE3_METRIC_SCHEMA_VERSION = 1
PHASE3_ARTIFACT_SCHEMA_VERSION = 1
PHASE3_CONDITIONS = ("current_only", "ewc_fixed_pi005")


@dataclasses.dataclass(frozen=True)
class Phase3LearnerConfig:
    optimizer: str
    learning_rate: float
    inner_steps: int
    lbfgs_history_size: int
    lbfgs_max_eval_factor: float
    lbfgs_tolerance_grad: float
    lbfgs_tolerance_change: float
    lbfgs_line_search_fn: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase3LearnerConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "learner")
        result = cls(
            optimizer=str(value["optimizer"]),
            learning_rate=_positive_float(
                value["learning_rate"], name="learner.learning_rate"
            ),
            inner_steps=_integer(
                value["inner_steps"], minimum=1, name="learner.inner_steps"
            ),
            lbfgs_history_size=_integer(
                value["lbfgs_history_size"],
                minimum=1,
                name="learner.lbfgs_history_size",
            ),
            lbfgs_max_eval_factor=_positive_float(
                value["lbfgs_max_eval_factor"],
                name="learner.lbfgs_max_eval_factor",
            ),
            lbfgs_tolerance_grad=_positive_float(
                value["lbfgs_tolerance_grad"],
                name="learner.lbfgs_tolerance_grad",
            ),
            lbfgs_tolerance_change=_positive_float(
                value["lbfgs_tolerance_change"],
                name="learner.lbfgs_tolerance_change",
            ),
            lbfgs_line_search_fn=str(value["lbfgs_line_search_fn"]),
        )
        if result.optimizer != "lbfgs":
            raise RotatedConfigError("Phase 3 learner.optimizer must be 'lbfgs'")
        if result.lbfgs_max_eval_factor < 1.0:
            raise RotatedConfigError(
                "learner.lbfgs_max_eval_factor must be at least one"
            )
        if result.lbfgs_line_search_fn != "strong_wolfe":
            raise RotatedConfigError(
                "learner.lbfgs_line_search_fn must be 'strong_wolfe'"
            )
        return result


@dataclasses.dataclass(frozen=True)
class Phase3FisherConfig:
    method: str
    representation: str
    rank: int
    initial_sample_size: int
    chunk_size: int
    matrix_dtype: str
    fixed_pi: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase3FisherConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "fisher")
        result = cls(
            method=str(value["method"]),
            representation=str(value["representation"]),
            rank=_integer(value["rank"], minimum=0, name="fisher.rank"),
            initial_sample_size=_integer(
                value["initial_sample_size"],
                minimum=1,
                name="fisher.initial_sample_size",
            ),
            chunk_size=_integer(
                value["chunk_size"], minimum=1, name="fisher.chunk_size"
            ),
            matrix_dtype=str(value["matrix_dtype"]),
            fixed_pi=_positive_float(value["fixed_pi"], name="fisher.fixed_pi"),
        )
        if result.method != "ema":
            raise RotatedConfigError("Phase 3 fisher.method must be 'ema'")
        if result.representation != "low_rank_diagonal":
            raise RotatedConfigError(
                "Phase 3 Fisher representation must be low_rank_diagonal"
            )
        if result.rank != 8:
            raise RotatedConfigError("Phase 3 Fisher rank is frozen at eight")
        if result.matrix_dtype != "float64":
            raise RotatedConfigError("Phase 3 Fisher matrices must use float64")
        if not 0.0 < result.fixed_pi < 1.0:
            raise RotatedConfigError("fisher.fixed_pi must be in (0, 1)")
        if not math.isclose(result.fixed_pi, 0.05, rel_tol=0.0, abs_tol=1e-15):
            raise RotatedConfigError("Phase 3 fixed pi is frozen at .05")
        return result


@dataclasses.dataclass(frozen=True)
class RotatedPhase3Config:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    conditions: tuple[str, ...]
    rotation: RotationConfig
    data: DataConfig
    initialization: InitializationConfig
    learner: Phase3LearnerConfig
    fisher: Phase3FisherConfig
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RotatedPhase3Config":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "Phase 3 configuration")
        conditions = value["conditions"]
        if not isinstance(conditions, list):
            raise RotatedConfigError("conditions must be a list")
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
            conditions=tuple(str(condition) for condition in conditions),
            rotation=RotationConfig.from_mapping(value["rotation"]),
            data=DataConfig.from_mapping(value["data"]),
            initialization=InitializationConfig.from_mapping(
                value["initialization"]
            ),
            learner=Phase3LearnerConfig.from_mapping(value["learner"]),
            fisher=Phase3FisherConfig.from_mapping(value["fisher"]),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != PHASE3_CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 3 configuration schema")
        if self.metric_schema_version != PHASE3_METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 3 metric schema")
        if self.artifact_schema_version != PHASE3_ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 3 artifact schema")
        if not self.experiment.startswith("rotated_mnist_phase3_"):
            raise RotatedConfigError(
                "Phase 3 experiment must start with 'rotated_mnist_phase3_'"
            )
        if not self.replica_id:
            raise RotatedConfigError("replica_id must be nonempty")
        if self.replica_seed >= 2**63:
            raise RotatedConfigError("replica_seed must be less than 2**63")
        if self.conditions != PHASE3_CONDITIONS:
            raise RotatedConfigError(
                f"Phase 3 conditions must be {list(PHASE3_CONDITIONS)}"
            )
        if self.rotation.knots_degrees != (0.0, 15.0, 30.0):
            raise RotatedConfigError("Phase 3 runs only the first rotation ascent")
        if self.data.samples_per_step != 8:
            raise RotatedConfigError("Phase 3 pilot is frozen at m=8")
        if self.data.samples_per_step != self.data.stream_width:
            raise RotatedConfigError("Phase 3 pilot requires one paired stream width")
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
            "rotation": self.rotation.to_mapping(),
            "data": dataclasses.asdict(self.data),
            "initialization": dataclasses.asdict(self.initialization),
            "learner": dataclasses.asdict(self.learner),
            "fisher": dataclasses.asdict(self.fisher),
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


def load_phase3_config(path: str | Path) -> RotatedPhase3Config:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Phase 3 configuration must be a JSON object")
    return RotatedPhase3Config.from_mapping(value)
