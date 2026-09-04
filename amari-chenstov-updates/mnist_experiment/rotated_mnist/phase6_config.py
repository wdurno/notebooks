"""Strict configuration contracts for Plan 6 oracle calibration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config import RotatedConfigError, RuntimeConfig, _integer, _strict_keys


PHASE6_CONFIG_SCHEMA_VERSION = 1
PHASE6_METRIC_SCHEMA_VERSION = 1
PHASE6_ARTIFACT_SCHEMA_VERSION = 1
PHASE6_SOURCE_RUN_ID = (
    "rotated_mnist_phase5_slow_single_lap_development__"
    "replica-0001__30f2161a06eb8d50"
)
PHASE6_SCHEDULES = ("linear", "sigmoid")
PHASE6_CONDITION = "edr_slowtrend_slowaction"


def _positive(value: Any, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RotatedConfigError(f"{name} must be numeric")
    result = float(value)
    invalid = result < 0.0 if allow_zero else result <= 0.0
    if not math.isfinite(result) or invalid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise RotatedConfigError(f"{name} must be finite and {qualifier}")
    return result


def _positive_tuple(
    value: Any,
    *,
    name: str,
    integer: bool = False,
) -> tuple[float, ...] | tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RotatedConfigError(f"{name} must be a nonempty list")
    if integer:
        result = tuple(
            _integer(item, minimum=1, name=f"{name} entry") for item in value
        )
    else:
        result = tuple(_positive(item, name=f"{name} entry") for item in value)
    if tuple(sorted(set(result))) != result:
        raise RotatedConfigError(f"{name} must be strictly increasing")
    return result


@dataclasses.dataclass(frozen=True)
class Phase6DebiasConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    source_run_path: str
    source_run_id: str
    schedule_kinds: tuple[str, ...]
    condition: str
    deployed_batch_size: int
    variance_scales: tuple[float, ...]
    pi_min: float
    pi_max: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase6DebiasConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "Phase 6 debias configuration")
        if not isinstance(value["schedule_kinds"], list):
            raise RotatedConfigError("schedule_kinds must be a list")
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
            source_run_path=str(value["source_run_path"]),
            source_run_id=str(value["source_run_id"]),
            schedule_kinds=tuple(str(item) for item in value["schedule_kinds"]),
            condition=str(value["condition"]),
            deployed_batch_size=_integer(
                value["deployed_batch_size"],
                minimum=1,
                name="deployed_batch_size",
            ),
            variance_scales=tuple(
                _positive_tuple(value["variance_scales"], name="variance_scales")
            ),
            pi_min=_positive(value["pi_min"], name="pi_min"),
            pi_max=_positive(value["pi_max"], name="pi_max"),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != PHASE6_CONFIG_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 6 config schema")
        if self.metric_schema_version != PHASE6_METRIC_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 6 metric schema")
        if self.artifact_schema_version != PHASE6_ARTIFACT_SCHEMA_VERSION:
            raise RotatedConfigError("unsupported Phase 6 artifact schema")
        if not self.experiment.startswith("rotated_mnist_phase6_debias_"):
            raise RotatedConfigError("invalid Phase 6 debias experiment name")
        if not self.replica_id or self.replica_seed >= 2**63:
            raise RotatedConfigError("invalid Phase 6 replica identity")
        if Path(self.source_run_path).name != self.source_run_id:
            raise RotatedConfigError("source path and run ID do not match")
        if self.source_run_id != PHASE6_SOURCE_RUN_ID:
            raise RotatedConfigError("Phase 6 source run is not authoritative")
        if self.schedule_kinds != PHASE6_SCHEDULES:
            raise RotatedConfigError("Phase 6 requires linear and sigmoid schedules")
        if self.condition != PHASE6_CONDITION:
            raise RotatedConfigError("Phase 6 condition is not the frozen EDR path")
        if self.deployed_batch_size != 4:
            raise RotatedConfigError("deployed m is frozen at 4")
        if not 0.0 < self.pi_min < self.pi_max <= 1.0:
            raise RotatedConfigError("pi bounds must satisfy 0 < min < max <= 1")
        if 1.0 not in self.variance_scales:
            raise RotatedConfigError("variance scales must include the primary value 1")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["schedule_kinds"] = list(self.schedule_kinds)
        value["variance_scales"] = list(self.variance_scales)
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


def load_phase6_debias_config(path: str | Path) -> Phase6DebiasConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Phase 6 debias configuration must be an object")
    return Phase6DebiasConfig.from_mapping(value)


@dataclasses.dataclass(frozen=True)
class Phase6ReferenceConfig:
    fit_pool_size: int
    fit_sample_size: int
    validation_size: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    initial_max_epochs: int
    max_epochs: int
    patience: int
    minimum_delta: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase6ReferenceConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "Phase 6 reference configuration")
        result = cls(
            fit_pool_size=_integer(
                value["fit_pool_size"], minimum=2, name="reference.fit_pool_size"
            ),
            fit_sample_size=_integer(
                value["fit_sample_size"], minimum=2, name="reference.fit_sample_size"
            ),
            validation_size=_integer(
                value["validation_size"], minimum=1, name="reference.validation_size"
            ),
            batch_size=_integer(
                value["batch_size"], minimum=1, name="reference.batch_size"
            ),
            learning_rate=_positive(
                value["learning_rate"], name="reference.learning_rate"
            ),
            weight_decay=_positive(
                value["weight_decay"],
                name="reference.weight_decay",
                allow_zero=True,
            ),
            initial_max_epochs=_integer(
                value["initial_max_epochs"],
                minimum=1,
                name="reference.initial_max_epochs",
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
        )
        if result.fit_sample_size > result.fit_pool_size:
            raise RotatedConfigError("reference fit sample exceeds its pool")
        if result.patience > result.max_epochs:
            raise RotatedConfigError("reference patience exceeds max epochs")
        if result.initial_max_epochs < result.max_epochs:
            raise RotatedConfigError(
                "initial reference epoch budget must be at least the continuation budget"
            )
        return result


@dataclasses.dataclass(frozen=True)
class Phase6FisherConfig:
    sample_size: int
    chunk_size: int
    ranks: tuple[int, ...]
    matrix_dtype: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase6FisherConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "Phase 6 Fisher configuration")
        result = cls(
            sample_size=_integer(
                value["sample_size"], minimum=2, name="fisher.sample_size"
            ),
            chunk_size=_integer(
                value["chunk_size"], minimum=1, name="fisher.chunk_size"
            ),
            ranks=tuple(
                _positive_tuple(value["ranks"], name="fisher.ranks", integer=True)
            ),
            matrix_dtype=str(value["matrix_dtype"]),
        )
        if result.matrix_dtype != "float64":
            raise RotatedConfigError("oracle Fisher accumulation requires float64")
        return result


@dataclasses.dataclass(frozen=True)
class Phase6LocalMLEConfig:
    sample_sizes: tuple[int, ...]
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    replicate_checkpoints: tuple[int, ...]
    maximum_replicates: int
    bootstrap_replicates: int
    sigma_multiple: float
    pi_half_width_tolerance: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase6LocalMLEConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "Phase 6 local-MLE configuration")
        result = cls(
            sample_sizes=tuple(
                _positive_tuple(
                    value["sample_sizes"], name="local_mle.sample_sizes", integer=True
                )
            ),
            batch_size=_integer(
                value["batch_size"], minimum=1, name="local_mle.batch_size"
            ),
            learning_rate=_positive(
                value["learning_rate"], name="local_mle.learning_rate"
            ),
            weight_decay=_positive(
                value["weight_decay"],
                name="local_mle.weight_decay",
                allow_zero=True,
            ),
            max_epochs=_integer(
                value["max_epochs"], minimum=1, name="local_mle.max_epochs"
            ),
            replicate_checkpoints=tuple(
                _positive_tuple(
                    value["replicate_checkpoints"],
                    name="local_mle.replicate_checkpoints",
                    integer=True,
                )
            ),
            maximum_replicates=_integer(
                value["maximum_replicates"],
                minimum=2,
                name="local_mle.maximum_replicates",
            ),
            bootstrap_replicates=_integer(
                value["bootstrap_replicates"],
                minimum=20,
                name="local_mle.bootstrap_replicates",
            ),
            sigma_multiple=_positive(
                value["sigma_multiple"], name="local_mle.sigma_multiple"
            ),
            pi_half_width_tolerance=_positive(
                value["pi_half_width_tolerance"],
                name="local_mle.pi_half_width_tolerance",
            ),
        )
        if result.replicate_checkpoints[-1] != result.maximum_replicates:
            raise RotatedConfigError(
                "the last local-MLE checkpoint must equal maximum_replicates"
            )
        return result


@dataclasses.dataclass(frozen=True)
class Phase6OracleConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    mode: str
    source_run_path: str
    source_run_id: str
    reference_run_path: str | None
    schedule_kinds: tuple[str, ...]
    condition: str
    transition_steps: dict[str, tuple[int, ...]]
    deployed_batch_size: int
    reference: Phase6ReferenceConfig
    fisher: Phase6FisherConfig
    local_mle: Phase6LocalMLEConfig
    max_wall_time_seconds: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase6OracleConfig":
        expected = {field.name for field in dataclasses.fields(cls)}
        _strict_keys(value, expected, "Phase 6 oracle configuration")
        schedules = value["schedule_kinds"]
        raw_steps = value["transition_steps"]
        if not isinstance(schedules, list) or not isinstance(raw_steps, Mapping):
            raise RotatedConfigError("oracle schedules and transition steps are invalid")
        converted_steps = {}
        for name, steps in raw_steps.items():
            if not isinstance(steps, list):
                raise RotatedConfigError("transition step values must be lists")
            converted_steps[str(name)] = tuple(
                _integer(step, minimum=0, name="transition step") for step in steps
            )
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
            mode=str(value["mode"]),
            source_run_path=str(value["source_run_path"]),
            source_run_id=str(value["source_run_id"]),
            reference_run_path=(
                None
                if value["reference_run_path"] is None
                else str(value["reference_run_path"])
            ),
            schedule_kinds=tuple(str(item) for item in schedules),
            condition=str(value["condition"]),
            transition_steps=converted_steps,
            deployed_batch_size=_integer(
                value["deployed_batch_size"],
                minimum=1,
                name="deployed_batch_size",
            ),
            reference=Phase6ReferenceConfig.from_mapping(value["reference"]),
            fisher=Phase6FisherConfig.from_mapping(value["fisher"]),
            local_mle=Phase6LocalMLEConfig.from_mapping(value["local_mle"]),
            max_wall_time_seconds=_positive(
                value["max_wall_time_seconds"], name="max_wall_time_seconds"
            ),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != PHASE6_CONFIG_SCHEMA_VERSION
            or self.metric_schema_version != PHASE6_METRIC_SCHEMA_VERSION
            or self.artifact_schema_version != PHASE6_ARTIFACT_SCHEMA_VERSION
        ):
            raise RotatedConfigError("unsupported Phase 6 oracle schema")
        if not self.experiment.startswith("rotated_mnist_phase6_oracle_"):
            raise RotatedConfigError("invalid Phase 6 oracle experiment name")
        if self.mode not in {"smoke", "pilot", "full"}:
            raise RotatedConfigError("oracle mode must be smoke, pilot, or full")
        if not self.replica_id or self.replica_seed >= 2**63:
            raise RotatedConfigError("invalid Phase 6 oracle replica identity")
        if Path(self.source_run_path).name != self.source_run_id:
            raise RotatedConfigError("source path and run ID do not match")
        if self.source_run_id != PHASE6_SOURCE_RUN_ID:
            raise RotatedConfigError("Phase 6 source run is not authoritative")
        if self.schedule_kinds != PHASE6_SCHEDULES:
            raise RotatedConfigError("oracle requires linear and sigmoid schedules")
        if self.condition != PHASE6_CONDITION:
            raise RotatedConfigError("oracle condition is not the frozen EDR path")
        if set(self.transition_steps) != set(self.schedule_kinds):
            raise RotatedConfigError("transition steps must cover each schedule")
        for schedule, steps in self.transition_steps.items():
            if tuple(sorted(set(steps))) != steps:
                raise RotatedConfigError(
                    f"{schedule} transition steps must be strictly increasing"
                )
        if self.mode == "full" and any(self.transition_steps.values()):
            raise RotatedConfigError("full mode derives all transitions; lists must be empty")
        if self.mode != "full" and any(not steps for steps in self.transition_steps.values()):
            raise RotatedConfigError("smoke and pilot require explicit transition steps")
        if self.deployed_batch_size != 4:
            raise RotatedConfigError("deployed m is frozen at 4")
        if self.fisher.sample_size > (
            20000 - self.reference.fit_pool_size
        ):
            raise RotatedConfigError("Fisher sample exceeds the held-out reference tail")
        if max(self.local_mle.sample_sizes) > self.reference.fit_pool_size:
            raise RotatedConfigError("local-MLE sample exceeds its bootstrap pool")
        if max(self.fisher.ranks) > 512:
            raise RotatedConfigError("Fisher rank exceeds canonical parameter count")
        if self.reference_run_path is not None and self.mode != "full":
            raise RotatedConfigError("only a full run may reuse pilot references")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["schedule_kinds"] = list(self.schedule_kinds)
        value["transition_steps"] = {
            name: list(self.transition_steps[name]) for name in self.schedule_kinds
        }
        value["fisher"]["ranks"] = list(self.fisher.ranks)
        value["local_mle"]["sample_sizes"] = list(self.local_mle.sample_sizes)
        value["local_mle"]["replicate_checkpoints"] = list(
            self.local_mle.replicate_checkpoints
        )
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


def load_phase6_oracle_config(path: str | Path) -> Phase6OracleConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Phase 6 oracle configuration must be an object")
    return Phase6OracleConfig.from_mapping(value)
