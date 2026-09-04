"""Strict configuration for the Plan 5 learnability audit."""

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
    RotationConfig,
    RotatedConfigError,
    RuntimeConfig,
)


AUDIT_CONFIG_SCHEMA_VERSION = 1
AUDIT_METRIC_SCHEMA_VERSION = 1
AUDIT_ARTIFACT_SCHEMA_VERSION = 1


def _strict_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise RotatedConfigError(f"{name} is missing fields {sorted(missing)}")
    if unknown:
        raise RotatedConfigError(f"{name} has unknown fields {sorted(unknown)}")


def _integer(value: Any, *, minimum: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RotatedConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value: Any, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RotatedConfigError(f"{name} must be numeric")
    result = float(value)
    invalid = result < 0.0 if allow_zero else result <= 0.0
    if not math.isfinite(result) or invalid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise RotatedConfigError(f"{name} must be finite and {qualifier}")
    return result


@dataclasses.dataclass(frozen=True)
class ReferenceFitConfig:
    angles_degrees: tuple[float, ...]
    fit_pool_size: int
    fit_sample_size: int
    validation_size: int
    endpoint_repeat_count: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    minimum_delta: float
    fisher_chunk_size: int
    fisher_matrix_dtype: str
    edr_batch_size: int
    edr_fixed_pi: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReferenceFitConfig":
        expected = {
            "angles_degrees",
            "fit_pool_size",
            "fit_sample_size",
            "validation_size",
            "endpoint_repeat_count",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "patience",
            "minimum_delta",
            "fisher_chunk_size",
            "fisher_matrix_dtype",
            "edr_batch_size",
            "edr_fixed_pi",
        }
        _strict_keys(value, expected, "reference")
        raw_angles = value["angles_degrees"]
        if not isinstance(raw_angles, list) or len(raw_angles) < 2:
            raise RotatedConfigError("reference.angles_degrees requires >= 2 angles")
        result = cls(
            angles_degrees=tuple(
                _positive(angle, name="reference angle", allow_zero=True)
                for angle in raw_angles
            ),
            fit_pool_size=_integer(
                value["fit_pool_size"], minimum=2, name="reference.fit_pool_size"
            ),
            fit_sample_size=_integer(
                value["fit_sample_size"],
                minimum=1,
                name="reference.fit_sample_size",
            ),
            validation_size=_integer(
                value["validation_size"],
                minimum=1,
                name="reference.validation_size",
            ),
            endpoint_repeat_count=_integer(
                value["endpoint_repeat_count"],
                minimum=1,
                name="reference.endpoint_repeat_count",
            ),
            optimizer=str(value["optimizer"]),
            learning_rate=_positive(
                value["learning_rate"], name="reference.learning_rate"
            ),
            weight_decay=_positive(
                value["weight_decay"],
                name="reference.weight_decay",
                allow_zero=True,
            ),
            batch_size=_integer(
                value["batch_size"], minimum=1, name="reference.batch_size"
            ),
            max_epochs=_integer(
                value["max_epochs"], minimum=1, name="reference.max_epochs"
            ),
            patience=_integer(
                value["patience"], minimum=1, name="reference.patience"
            ),
            minimum_delta=_positive(
                value["minimum_delta"],
                name="reference.minimum_delta",
                allow_zero=True,
            ),
            fisher_chunk_size=_integer(
                value["fisher_chunk_size"],
                minimum=1,
                name="reference.fisher_chunk_size",
            ),
            fisher_matrix_dtype=str(value["fisher_matrix_dtype"]),
            edr_batch_size=_integer(
                value["edr_batch_size"], minimum=1, name="reference.edr_batch_size"
            ),
            edr_fixed_pi=_positive(
                value["edr_fixed_pi"], name="reference.edr_fixed_pi"
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if tuple(sorted(self.angles_degrees)) != self.angles_degrees:
            raise RotatedConfigError("reference angles must be strictly increasing")
        if len(set(self.angles_degrees)) != len(self.angles_degrees):
            raise RotatedConfigError("reference angles must be unique")
        if self.angles_degrees[0] != 0.0 or self.angles_degrees[-1] != 30.0:
            raise RotatedConfigError("reference angles must span 0 through 30 degrees")
        if self.fit_sample_size > self.fit_pool_size:
            raise RotatedConfigError("fit_sample_size cannot exceed fit_pool_size")
        if self.optimizer != "adam":
            raise RotatedConfigError("reference.optimizer must be 'adam'")
        if self.patience > self.max_epochs:
            raise RotatedConfigError("reference.patience cannot exceed max_epochs")
        if self.fisher_matrix_dtype != "float64":
            raise RotatedConfigError("reference Fisher matrices must use float64")
        if not 0.0 < self.edr_fixed_pi < 1.0:
            raise RotatedConfigError("reference.edr_fixed_pi must be in (0, 1)")


@dataclasses.dataclass(frozen=True)
class GateConfig:
    minimum_environment_accuracy_30: float
    minimum_learning_room: float
    minimum_fisher_signal_to_noise: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GateConfig":
        expected = {
            "minimum_environment_accuracy_30",
            "minimum_learning_room",
            "minimum_fisher_signal_to_noise",
        }
        _strict_keys(value, expected, "gate")
        result = cls(
            minimum_environment_accuracy_30=_positive(
                value["minimum_environment_accuracy_30"],
                name="gate.minimum_environment_accuracy_30",
            ),
            minimum_learning_room=_positive(
                value["minimum_learning_room"],
                name="gate.minimum_learning_room",
                allow_zero=True,
            ),
            minimum_fisher_signal_to_noise=_positive(
                value["minimum_fisher_signal_to_noise"],
                name="gate.minimum_fisher_signal_to_noise",
            ),
        )
        if result.minimum_environment_accuracy_30 > 1.0:
            raise RotatedConfigError("gate accuracy threshold must be <= 1")
        return result


@dataclasses.dataclass(frozen=True)
class RotatedAuditConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    rotation: RotationConfig
    data: DataConfig
    initialization: InitializationConfig
    reference: ReferenceFitConfig
    gate: GateConfig
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RotatedAuditConfig":
        expected = {
            "schema_version",
            "metric_schema_version",
            "artifact_schema_version",
            "experiment",
            "replica_id",
            "replica_seed",
            "rotation",
            "data",
            "initialization",
            "reference",
            "gate",
            "runtime",
        }
        _strict_keys(value, expected, "audit configuration")
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
            rotation=RotationConfig.from_mapping(value["rotation"]),
            data=DataConfig.from_mapping(value["data"]),
            initialization=InitializationConfig.from_mapping(value["initialization"]),
            reference=ReferenceFitConfig.from_mapping(value["reference"]),
            gate=GateConfig.from_mapping(value["gate"]),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != AUDIT_CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported rotated audit config schema")
        if self.metric_schema_version != AUDIT_METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported rotated audit metric schema")
        if self.artifact_schema_version != AUDIT_ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported rotated audit artifact schema")
        if not self.experiment.startswith("rotated_mnist_phase2_"):
            raise RotatedConfigError("audit experiment must start with rotated_mnist_phase2_")
        if not self.replica_id or self.replica_seed >= 2**63:
            raise RotatedConfigError("audit replica identity is invalid")
        self.rotation.validate()
        self.reference.validate()
        if self.reference.fit_pool_size >= self.data.reference_pool_size:
            raise RotatedConfigError(
                "reference pool must leave observations for Fisher estimation"
            )
        if self.reference.validation_size >= self.data.evaluation_size:
            raise RotatedConfigError(
                "validation split must leave a reporting evaluation panel"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_schema_version": self.metric_schema_version,
            "artifact_schema_version": self.artifact_schema_version,
            "experiment": self.experiment,
            "replica_id": self.replica_id,
            "replica_seed": self.replica_seed,
            "rotation": self.rotation.to_mapping(),
            "data": dataclasses.asdict(self.data),
            "initialization": dataclasses.asdict(self.initialization),
            "reference": {
                **dataclasses.asdict(self.reference),
                "angles_degrees": list(self.reference.angles_degrees),
            },
            "gate": dataclasses.asdict(self.gate),
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


def load_audit_config(path: str | Path) -> RotatedAuditConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load audit config {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("audit configuration must be a JSON object")
    return RotatedAuditConfig.from_mapping(value)
