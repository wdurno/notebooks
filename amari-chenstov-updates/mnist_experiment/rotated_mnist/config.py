"""Strict, versioned configuration for the detachable Plan 5 experiment."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1


class RotatedConfigError(ValueError):
    """Raised when a rotated-MNIST configuration is invalid."""


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


def _positive_float(value: Any, *, name: str, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise RotatedConfigError(f"{name} must be a finite {qualifier} number")
    return float(value)


@dataclasses.dataclass(frozen=True)
class RotationConfig:
    knots_degrees: tuple[float, ...]
    transitions_per_arrow: int
    interpolation: str
    expand: bool
    fill: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RotationConfig":
        expected = {
            "knots_degrees",
            "transitions_per_arrow",
            "interpolation",
            "expand",
            "fill",
        }
        _strict_keys(value, expected, "rotation")
        knots_raw = value["knots_degrees"]
        if not isinstance(knots_raw, list) or len(knots_raw) < 2:
            raise RotatedConfigError("rotation.knots_degrees must contain >= 2 values")
        knots = tuple(
            _positive_float(angle, name="rotation angle", allow_zero=True)
            for angle in knots_raw
        )
        if any(angle > 360.0 for angle in knots):
            raise RotatedConfigError("rotation angles must be in [0, 360]")
        result = cls(
            knots_degrees=knots,
            transitions_per_arrow=_integer(
                value["transitions_per_arrow"],
                minimum=1,
                name="rotation.transitions_per_arrow",
            ),
            interpolation=str(value["interpolation"]),
            expand=value["expand"],
            fill=_positive_float(
                value["fill"], name="rotation.fill", allow_zero=True
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.interpolation != "bilinear":
            raise RotatedConfigError("rotation.interpolation must be 'bilinear'")
        if self.expand is not False:
            raise RotatedConfigError("rotation.expand must be false")
        if self.fill != 0.0:
            raise RotatedConfigError("rotation.fill must be 0.0")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["knots_degrees"] = list(self.knots_degrees)
        return value


@dataclasses.dataclass(frozen=True)
class DataConfig:
    samples_per_step: int
    stream_width: int
    initialization_size: int
    online_pool_size: int
    reference_pool_size: int
    evaluation_size: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DataConfig":
        expected = {
            "samples_per_step",
            "stream_width",
            "initialization_size",
            "online_pool_size",
            "reference_pool_size",
            "evaluation_size",
        }
        _strict_keys(value, expected, "data")
        result = cls(
            **{
                name: _integer(value[name], minimum=1, name=f"data.{name}")
                for name in expected
            }
        )
        if result.samples_per_step > result.stream_width:
            raise RotatedConfigError(
                "data.samples_per_step cannot exceed data.stream_width"
            )
        return result


@dataclasses.dataclass(frozen=True)
class InitializationConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    target_accuracy: float | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InitializationConfig":
        expected = {
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "target_accuracy",
        }
        _strict_keys(value, expected, "initialization")
        target = value["target_accuracy"]
        if target is not None:
            target = _positive_float(target, name="initialization.target_accuracy")
            if target > 1.0:
                raise RotatedConfigError(
                    "initialization.target_accuracy must be null or in (0, 1]"
                )
        result = cls(
            optimizer=str(value["optimizer"]),
            learning_rate=_positive_float(
                value["learning_rate"], name="initialization.learning_rate"
            ),
            weight_decay=_positive_float(
                value["weight_decay"],
                name="initialization.weight_decay",
                allow_zero=True,
            ),
            batch_size=_integer(
                value["batch_size"], minimum=1, name="initialization.batch_size"
            ),
            max_epochs=_integer(
                value["max_epochs"], minimum=1, name="initialization.max_epochs"
            ),
            target_accuracy=target,
        )
        if result.optimizer not in {"sgd", "adam"}:
            raise RotatedConfigError(
                "initialization.optimizer must be 'sgd' or 'adam'"
            )
        return result


@dataclasses.dataclass(frozen=True)
class LearnerConfig:
    condition: str
    optimizer: str
    learning_rate: float
    inner_steps: int
    lbfgs_history_size: int
    lbfgs_max_eval_factor: float
    lbfgs_tolerance_grad: float
    lbfgs_tolerance_change: float
    lbfgs_line_search_fn: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LearnerConfig":
        expected = {
            "condition",
            "optimizer",
            "learning_rate",
            "inner_steps",
            "lbfgs_history_size",
            "lbfgs_max_eval_factor",
            "lbfgs_tolerance_grad",
            "lbfgs_tolerance_change",
            "lbfgs_line_search_fn",
        }
        _strict_keys(value, expected, "learner")
        result = cls(
            condition=str(value["condition"]),
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
        if result.condition != "current_only":
            raise RotatedConfigError(
                "Phase 1 learner.condition must be 'current_only'"
            )
        if result.optimizer != "lbfgs":
            raise RotatedConfigError("Phase 1 learner.optimizer must be 'lbfgs'")
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
class RuntimeConfig:
    device: str
    dtype: str
    deterministic_algorithms: bool
    num_workers: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        expected = {"device", "dtype", "deterministic_algorithms", "num_workers"}
        _strict_keys(value, expected, "runtime")
        result = cls(
            device=str(value["device"]),
            dtype=str(value["dtype"]),
            deterministic_algorithms=value["deterministic_algorithms"],
            num_workers=_integer(
                value["num_workers"], minimum=0, name="runtime.num_workers"
            ),
        )
        if result.device not in {"cpu", "cuda", "auto"}:
            raise RotatedConfigError("runtime.device must be cpu, cuda, or auto")
        if result.dtype not in {"float32", "float64"}:
            raise RotatedConfigError("runtime.dtype must be float32 or float64")
        if not isinstance(result.deterministic_algorithms, bool):
            raise RotatedConfigError(
                "runtime.deterministic_algorithms must be a boolean"
            )
        return result


@dataclasses.dataclass(frozen=True)
class RotatedExperimentConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    rotation: RotationConfig
    data: DataConfig
    initialization: InitializationConfig
    learner: LearnerConfig
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RotatedExperimentConfig":
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
            "learner",
            "runtime",
        }
        _strict_keys(value, expected, "configuration")
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
            initialization=InitializationConfig.from_mapping(
                value["initialization"]
            ),
            learner=LearnerConfig.from_mapping(value["learner"]),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported rotated configuration schema")
        if self.metric_schema_version != METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported rotated metric schema")
        if self.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported rotated artifact schema")
        if not self.experiment.startswith("rotated_mnist_"):
            raise RotatedConfigError("experiment must start with 'rotated_mnist_'")
        if not self.replica_id:
            raise RotatedConfigError("replica_id must be nonempty")
        if self.replica_seed >= 2**63:
            raise RotatedConfigError("replica_seed must be less than 2**63")
        self.rotation.validate()

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
            "learner": dataclasses.asdict(self.learner),
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


def load_config(path: str | Path) -> RotatedExperimentConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("configuration must be a JSON object")
    return RotatedExperimentConfig.from_mapping(value)
