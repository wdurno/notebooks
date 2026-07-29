"""Typed, versioned experiment configuration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, TypeVar

CONFIG_SCHEMA_VERSION = 4
ARTIFACT_SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_T = TypeVar("_T")


class ConfigError(ValueError):
    """Raised when an experiment configuration violates its schema."""


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _construct_dataclass(cls: type[_T], value: Any, context: str) -> _T:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be an object")

    expected = {field.name for field in dataclasses.fields(cls)}
    supplied = set(value)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ConfigError(f"invalid {context}: {', '.join(details)}")
    return cls(**value)


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ConfigError(
            f"{name} must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'"
        )


@dataclasses.dataclass(frozen=True)
class DataConfig:
    num_p_steps: int
    samples_per_step: int
    non_nine_sampling: str
    initialization_size: int
    online_pool_size: int
    reference_pool_size: int
    evaluation_size: int

    def validate(self) -> None:
        if not _is_integer(self.num_p_steps) or self.num_p_steps < 2:
            raise ConfigError("data.num_p_steps must be an integer >= 2")
        if not _is_integer(self.samples_per_step) or self.samples_per_step < 1:
            raise ConfigError("data.samples_per_step must be an integer >= 1")
        if self.non_nine_sampling not in {"empirical", "balanced"}:
            raise ConfigError(
                "data.non_nine_sampling must be 'empirical' or 'balanced'"
            )
        for name, value in {
            "initialization_size": self.initialization_size,
            "online_pool_size": self.online_pool_size,
            "reference_pool_size": self.reference_pool_size,
            "evaluation_size": self.evaluation_size,
        }.items():
            if not _is_integer(value) or value < 1:
                raise ConfigError(f"data.{name} must be an integer >= 1")


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    device: str
    training_dtype: str
    matrix_dtype: str
    deterministic_algorithms: bool

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ConfigError("runtime.device must be 'cpu', 'cuda', or 'auto'")
        if self.training_dtype not in {"float32", "float64"}:
            raise ConfigError(
                "runtime.training_dtype must be 'float32' or 'float64'"
            )
        if self.matrix_dtype not in {"float32", "float64"}:
            raise ConfigError(
                "runtime.matrix_dtype must be 'float32' or 'float64'"
            )
        if not isinstance(self.deterministic_algorithms, bool):
            raise ConfigError(
                "runtime.deterministic_algorithms must be a boolean"
            )


@dataclasses.dataclass(frozen=True)
class EstimatorConfig:
    method: str
    representation: str
    ema_gain: float
    fresh_fisher_cadence: int | None
    low_rank: int | None

    def validate(self) -> None:
        if self.method not in {"ema", "ac_only", "full_lfu", "periodic_fresh"}:
            raise ConfigError(f"unsupported estimator.method: {self.method}")
        if self.representation not in {
            "dense",
            "diagonal",
            "low_rank_diagonal",
        }:
            raise ConfigError(
                f"unsupported estimator.representation: {self.representation}"
            )
        if not _is_finite_number(self.ema_gain) or not (
            0.0 < float(self.ema_gain) <= 1.0
        ):
            raise ConfigError("estimator.ema_gain must be in (0, 1]")
        if self.fresh_fisher_cadence is not None and (
            not _is_integer(self.fresh_fisher_cadence)
            or self.fresh_fisher_cadence < 1
        ):
            raise ConfigError(
                "estimator.fresh_fisher_cadence must be null or an integer >= 1"
            )
        if self.low_rank is not None and (
            not _is_integer(self.low_rank) or self.low_rank < 0
        ):
            raise ConfigError("estimator.low_rank must be null or an integer >= 0")
        if self.representation == "low_rank_diagonal" and self.low_rank is None:
            raise ConfigError(
                "estimator.low_rank is required for low_rank_diagonal"
            )
        if self.representation != "low_rank_diagonal" and self.low_rank is not None:
            raise ConfigError(
                "estimator.low_rank must be null unless representation is "
                "low_rank_diagonal"
            )


@dataclasses.dataclass(frozen=True)
class ReferenceConfig:
    sample_size: int
    chunk_size: int
    derivative_dtype: str
    convergence_sample_sizes: list[int]
    stencil_epsilons: list[float]
    stencil_p_values: list[float]
    stencil_direction_count: int
    calibration_steps: int
    calibration_batch_size: int
    calibration_learning_rate: float

    def validate(self) -> None:
        if not _is_integer(self.sample_size) or self.sample_size < 1:
            raise ConfigError("reference.sample_size must be an integer >= 1")
        if not _is_integer(self.chunk_size) or self.chunk_size < 1:
            raise ConfigError("reference.chunk_size must be an integer >= 1")
        if self.chunk_size > self.sample_size:
            raise ConfigError(
                "reference.chunk_size cannot exceed reference.sample_size"
            )
        if self.derivative_dtype not in {"float32", "float64"}:
            raise ConfigError(
                "reference.derivative_dtype must be 'float32' or 'float64'"
            )
        if (
            not isinstance(self.convergence_sample_sizes, list)
            or not self.convergence_sample_sizes
            or any(
                not _is_integer(value) or value < 1
                for value in self.convergence_sample_sizes
            )
            or self.convergence_sample_sizes
            != sorted(set(self.convergence_sample_sizes))
            or self.convergence_sample_sizes[-1] != self.sample_size
        ):
            raise ConfigError(
                "reference.convergence_sample_sizes must be unique increasing "
                "positive integers ending at reference.sample_size"
            )
        if (
            not isinstance(self.stencil_epsilons, list)
            or not self.stencil_epsilons
            or any(
                not _is_finite_number(value) or float(value) <= 0.0
                for value in self.stencil_epsilons
            )
            or self.stencil_epsilons
            != sorted(set(self.stencil_epsilons))
        ):
            raise ConfigError(
                "reference.stencil_epsilons must be unique increasing "
                "positive finite numbers"
            )
        if (
            not isinstance(self.stencil_p_values, list)
            or not self.stencil_p_values
            or any(
                not _is_finite_number(value) or not 0.0 < float(value) < 1.0
                for value in self.stencil_p_values
            )
            or self.stencil_p_values != sorted(set(self.stencil_p_values))
        ):
            raise ConfigError(
                "reference.stencil_p_values must be unique increasing "
                "interior probabilities"
            )
        for name, value in {
            "stencil_direction_count": self.stencil_direction_count,
            "calibration_steps": self.calibration_steps,
            "calibration_batch_size": self.calibration_batch_size,
        }.items():
            if not _is_integer(value) or value < 1:
                raise ConfigError(f"reference.{name} must be an integer >= 1")
        if not _is_finite_number(self.calibration_learning_rate) or not (
            float(self.calibration_learning_rate) > 0.0
        ):
            raise ConfigError(
                "reference.calibration_learning_rate must be positive"
            )


@dataclasses.dataclass(frozen=True)
class InitializationConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    target_non_nine_accuracy: float | None
    num_workers: int

    def validate(self) -> None:
        if self.optimizer not in {"sgd", "adam"}:
            raise ConfigError("initialization.optimizer must be 'sgd' or 'adam'")
        if not _is_finite_number(self.learning_rate) or not (
            float(self.learning_rate) > 0.0
        ):
            raise ConfigError("initialization.learning_rate must be positive")
        if not _is_finite_number(self.weight_decay) or (
            float(self.weight_decay) < 0.0
        ):
            raise ConfigError("initialization.weight_decay must be nonnegative")
        if not _is_integer(self.batch_size) or self.batch_size < 1:
            raise ConfigError("initialization.batch_size must be an integer >= 1")
        if not _is_integer(self.max_epochs) or self.max_epochs < 1:
            raise ConfigError("initialization.max_epochs must be an integer >= 1")
        if self.target_non_nine_accuracy is not None and (
            not _is_finite_number(self.target_non_nine_accuracy)
            or not 0.0 < float(self.target_non_nine_accuracy) <= 1.0
        ):
            raise ConfigError(
                "initialization.target_non_nine_accuracy must be null or in (0, 1]"
            )
        if not _is_integer(self.num_workers) or self.num_workers < 0:
            raise ConfigError(
                "initialization.num_workers must be an integer >= 0"
            )


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    inner_steps: int
    ewc_strength: float

    def validate(self) -> None:
        if self.name not in {"sgd", "adam"}:
            raise ConfigError("optimizer.name must be 'sgd' or 'adam'")
        if not _is_finite_number(self.learning_rate) or not (
            float(self.learning_rate) > 0.0
        ):
            raise ConfigError("optimizer.learning_rate must be positive")
        if not _is_integer(self.inner_steps) or self.inner_steps < 1:
            raise ConfigError("optimizer.inner_steps must be an integer >= 1")
        if not _is_finite_number(self.ewc_strength) or (
            float(self.ewc_strength) < 0.0
        ):
            raise ConfigError("optimizer.ewc_strength must be nonnegative")


@dataclasses.dataclass(frozen=True)
class ControllerConfig:
    policy: str
    fixed_pi: float
    pi_max: float
    damping: float
    epsilon: float

    def validate(self) -> None:
        if self.policy not in {
            "uncontrolled",
            "fixed",
            "optimal_plugin",
            "optimal_capped",
            "optimal_oracle",
        }:
            raise ConfigError(f"unsupported controller.policy: {self.policy}")
        for name, value in {
            "fixed_pi": self.fixed_pi,
            "pi_max": self.pi_max,
        }.items():
            if not _is_finite_number(value) or not (
                0.0 <= float(value) <= 1.0
            ):
                raise ConfigError(f"controller.{name} must be in [0, 1]")
        if not _is_finite_number(self.damping) or not (
            float(self.damping) > 0.0
        ):
            raise ConfigError("controller.damping must be positive")
        if not _is_finite_number(self.epsilon) or not (
            float(self.epsilon) > 0.0
        ):
            raise ConfigError("controller.epsilon must be positive")


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    artifact_schema_version: int
    metric_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    cache_root: str
    data: DataConfig
    runtime: RuntimeConfig
    estimator: EstimatorConfig
    reference: ReferenceConfig
    initialization: InitializationConfig
    optimizer: OptimizerConfig
    controller: ControllerConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        if not isinstance(value, Mapping):
            raise ConfigError("configuration root must be an object")

        expected = {field.name for field in dataclasses.fields(cls)}
        supplied = set(value)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise ConfigError(f"invalid configuration root: {', '.join(details)}")

        config = cls(
            schema_version=value["schema_version"],
            artifact_schema_version=value["artifact_schema_version"],
            metric_schema_version=value["metric_schema_version"],
            experiment=value["experiment"],
            replica_id=value["replica_id"],
            replica_seed=value["replica_seed"],
            cache_root=value["cache_root"],
            data=_construct_dataclass(DataConfig, value["data"], "data"),
            runtime=_construct_dataclass(RuntimeConfig, value["runtime"], "runtime"),
            estimator=_construct_dataclass(
                EstimatorConfig, value["estimator"], "estimator"
            ),
            reference=_construct_dataclass(
                ReferenceConfig, value["reference"], "reference"
            ),
            initialization=_construct_dataclass(
                InitializationConfig,
                value["initialization"],
                "initialization",
            ),
            optimizer=_construct_dataclass(
                OptimizerConfig, value["optimizer"], "optimizer"
            ),
            controller=_construct_dataclass(
                ControllerConfig, value["controller"], "controller"
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigError(
                f"schema_version must be {CONFIG_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        if self.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ConfigError(
                f"artifact_schema_version must be {ARTIFACT_SCHEMA_VERSION}, "
                f"got {self.artifact_schema_version}"
            )
        if self.metric_schema_version != METRIC_SCHEMA_VERSION:
            raise ConfigError(
                f"metric_schema_version must be {METRIC_SCHEMA_VERSION}, "
                f"got {self.metric_schema_version}"
            )
        _validate_identifier(self.experiment, "experiment")
        _validate_identifier(self.replica_id, "replica_id")
        if not _is_integer(self.replica_seed) or not (
            0 <= self.replica_seed < 2**63
        ):
            raise ConfigError("replica_seed must be an integer in [0, 2**63)")
        if not isinstance(self.cache_root, str) or not self.cache_root.strip():
            raise ConfigError("cache_root must be a nonempty path string")

        self.data.validate()
        self.runtime.validate()
        self.estimator.validate()
        self.reference.validate()
        self.initialization.validate()
        self.optimizer.validate()
        self.controller.validate()

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_mapping(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        return (
            f"{self.experiment}__{self.replica_id}__{self.config_hash[:16]}"
        )


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not read configuration {config_path}: {exc}") from exc
    return ExperimentConfig.from_mapping(value)
