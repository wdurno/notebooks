"""Strict configuration for the paired Plan 5 Phase 4 memory screen."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .config import (
    DataConfig,
    InitializationConfig,
    RotatedConfigError,
    RotationConfig,
    RuntimeConfig,
    _integer,
    _strict_keys,
)
from .phase3_config import Phase3FisherConfig, Phase3LearnerConfig


PHASE4_CONFIG_SCHEMA_VERSION = 1
PHASE4_METRIC_SCHEMA_VERSION = 1
PHASE4_ARTIFACT_SCHEMA_VERSION = 1
PHASE4_CONDITIONS = (
    "current_only",
    "ewc_fixed_pi005",
    "replay_b032",
    "hybrid_b032_fixed_pi005",
    "replay_unbounded",
)
PHASE4_KNOTS = (0.0, 15.0, 30.0, 0.0, 15.0, 30.0)


@dataclasses.dataclass(frozen=True)
class Phase4ReplayConfig:
    policy: str
    bounded_capacity: int
    logical_pixel_dtype: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase4ReplayConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "replay")
        result = cls(
            policy=str(value["policy"]),
            bounded_capacity=_integer(
                value["bounded_capacity"],
                minimum=1,
                name="replay.bounded_capacity",
            ),
            logical_pixel_dtype=str(value["logical_pixel_dtype"]),
        )
        if result.policy != "fifo":
            raise RotatedConfigError("Phase 4 replay policy must be FIFO")
        if result.bounded_capacity != 32:
            raise RotatedConfigError("Phase 4 bounded replay capacity is frozen at 32")
        if result.logical_pixel_dtype != "float32":
            raise RotatedConfigError(
                "Phase 4 exact transformed replay is accounted as float32"
            )
        return result


@dataclasses.dataclass(frozen=True)
class RotatedPhase4Config:
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
    replay: Phase4ReplayConfig
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RotatedPhase4Config":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "Phase 4 configuration")
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
            replay=Phase4ReplayConfig.from_mapping(value["replay"]),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != PHASE4_CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 4 configuration schema")
        if self.metric_schema_version != PHASE4_METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 4 metric schema")
        if self.artifact_schema_version != PHASE4_ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 4 artifact schema")
        if not self.experiment.startswith("rotated_mnist_phase4_"):
            raise RotatedConfigError(
                "Phase 4 experiment must start with 'rotated_mnist_phase4_'"
            )
        if not self.replica_id:
            raise RotatedConfigError("replica_id must be nonempty")
        if self.replica_seed >= 2**63:
            raise RotatedConfigError("replica_seed must be less than 2**63")
        if self.conditions != PHASE4_CONDITIONS:
            raise RotatedConfigError(
                f"Phase 4 conditions must be {list(PHASE4_CONDITIONS)}"
            )
        if self.rotation.knots_degrees != PHASE4_KNOTS:
            raise RotatedConfigError("Phase 4 requires the complete repeated path")
        if self.data.samples_per_step != 8 or self.data.stream_width != 8:
            raise RotatedConfigError("Phase 4 is frozen at one paired m=8 stream")
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
            "replay": dataclasses.asdict(self.replay),
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


def load_phase4_config(path: str | Path) -> RotatedPhase4Config:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Phase 4 configuration must be a JSON object")
    return RotatedPhase4Config.from_mapping(value)
