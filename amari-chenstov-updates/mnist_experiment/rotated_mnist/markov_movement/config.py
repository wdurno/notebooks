"""Strict configurations for Plan 9 studies."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import RotatedConfigError, RuntimeConfig, _integer, _strict_keys
from ..phase7_movement_config import (
    AUTHORITATIVE_DOUBLE_RUN_ID,
    AUTHORITATIVE_ORACLE_RUN_ID,
    AUTHORITATIVE_SINGLE_RUN_ID,
)


PLAN9_CONFIG_SCHEMA_VERSION = 1
PLAN9_METRIC_SCHEMA_VERSION = 1
PLAN9_ARTIFACT_SCHEMA_VERSION = 1
PLAN9_SCHEDULES = ("linear", "sigmoid")
PLAN9_DESIGNS = ("single_lap", "double_lap")


def _positive(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RotatedConfigError(f"{name} must be finite and positive")
    return float(value)


def _relative_path(value: Any, *, name: str) -> str:
    result = str(value)
    path = Path(result)
    if not result or path.is_absolute() or ".." in path.parts:
        raise RotatedConfigError(f"{name} must be a repository-relative path")
    return result


def _unit_interval(value: Any, *, name: str) -> float:
    result = _positive(value, name=name)
    if result >= 1.0:
        raise RotatedConfigError(f"{name} must be below one")
    return result


@dataclasses.dataclass(frozen=True)
class RetrospectiveConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    study: str
    experiment: str
    replica_id: str
    replica_seed: int
    oracle_run_path: str
    oracle_run_id: str
    single_lap_run_path: str
    single_lap_run_id: str
    double_lap_run_path: str
    double_lap_run_id: str
    condition: str
    schedule_kinds: tuple[str, ...]
    deployed_batch_size: int
    oracle_sample_size: int
    oracle_replicate_count: int
    oracle_rank: int
    cold_start_steps: int
    reversal_window_steps: int
    smoothing_half_lives_degrees: dict[str, tuple[float, ...]]
    ratio_floor: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RetrospectiveConfig":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "Plan 9 retrospective configuration",
        )
        if not isinstance(value["schedule_kinds"], list) or not isinstance(
            value["smoothing_half_lives_degrees"], Mapping
        ):
            raise RotatedConfigError("Plan 9 retrospective collections are invalid")
        smoothing = {
            str(design): tuple(
                _positive(item, name=f"smoothing_half_lives_degrees.{design}")
                for item in items
            )
            for design, items in value["smoothing_half_lives_degrees"].items()
        }
        result = cls(
            schema_version=_integer(value["schema_version"], minimum=1, name="schema_version"),
            metric_schema_version=_integer(
                value["metric_schema_version"], minimum=1, name="metric_schema_version"
            ),
            artifact_schema_version=_integer(
                value["artifact_schema_version"], minimum=1, name="artifact_schema_version"
            ),
            study=str(value["study"]),
            experiment=str(value["experiment"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(value["replica_seed"], minimum=0, name="replica_seed"),
            oracle_run_path=_relative_path(value["oracle_run_path"], name="oracle_run_path"),
            oracle_run_id=str(value["oracle_run_id"]),
            single_lap_run_path=_relative_path(
                value["single_lap_run_path"], name="single_lap_run_path"
            ),
            single_lap_run_id=str(value["single_lap_run_id"]),
            double_lap_run_path=_relative_path(
                value["double_lap_run_path"], name="double_lap_run_path"
            ),
            double_lap_run_id=str(value["double_lap_run_id"]),
            condition=str(value["condition"]),
            schedule_kinds=tuple(str(item) for item in value["schedule_kinds"]),
            deployed_batch_size=_integer(
                value["deployed_batch_size"], minimum=1, name="deployed_batch_size"
            ),
            oracle_sample_size=_integer(
                value["oracle_sample_size"], minimum=2, name="oracle_sample_size"
            ),
            oracle_replicate_count=_integer(
                value["oracle_replicate_count"], minimum=2, name="oracle_replicate_count"
            ),
            oracle_rank=_integer(value["oracle_rank"], minimum=1, name="oracle_rank"),
            cold_start_steps=_integer(
                value["cold_start_steps"], minimum=0, name="cold_start_steps"
            ),
            reversal_window_steps=_integer(
                value["reversal_window_steps"], minimum=1, name="reversal_window_steps"
            ),
            smoothing_half_lives_degrees=smoothing,
            ratio_floor=_positive(value["ratio_floor"], name="ratio_floor"),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != PLAN9_CONFIG_SCHEMA_VERSION
            or self.metric_schema_version != PLAN9_METRIC_SCHEMA_VERSION
            or self.artifact_schema_version != PLAN9_ARTIFACT_SCHEMA_VERSION
        ):
            raise RotatedConfigError("unsupported Plan 9 retrospective schema")
        if self.study not in {"e9_1", "e9_2"} or not self.experiment.startswith(
            f"rotated_mnist_plan9_{self.study}_"
        ):
            raise RotatedConfigError("invalid Plan 9 retrospective study identity")
        if not self.replica_id or self.replica_seed >= 2**63:
            raise RotatedConfigError("invalid Plan 9 replica identity")
        paths = (
            (self.oracle_run_path, self.oracle_run_id, AUTHORITATIVE_ORACLE_RUN_ID),
            (self.single_lap_run_path, self.single_lap_run_id, AUTHORITATIVE_SINGLE_RUN_ID),
            (self.double_lap_run_path, self.double_lap_run_id, AUTHORITATIVE_DOUBLE_RUN_ID),
        )
        if any(Path(path).name != run_id or run_id != expected for path, run_id, expected in paths):
            raise RotatedConfigError("Plan 9 requires the frozen authoritative sources")
        if self.condition != "decomposed_edr" or self.schedule_kinds != PLAN9_SCHEDULES:
            raise RotatedConfigError("Plan 9 retrospective treatment is frozen")
        frozen = (
            (self.deployed_batch_size, 4),
            (self.oracle_sample_size, 2048),
            (self.oracle_replicate_count, 64),
            (self.oracle_rank, 16),
            (self.cold_start_steps, 8),
            (self.reversal_window_steps, 8),
        )
        if any(actual != expected for actual, expected in frozen):
            raise RotatedConfigError("Plan 9 retrospective constants drifted")
        expected_smoothing = {
            "single_lap": (3.75, 7.5, 15.0),
            "double_lap": (0.9375, 1.875, 3.75),
        }
        if self.smoothing_half_lives_degrees != expected_smoothing:
            raise RotatedConfigError("Plan 9 smoothing family is frozen")
        if self.runtime.device != "cpu" or self.runtime.dtype != "float64":
            raise RotatedConfigError("Plan 9 retrospective arithmetic is CPU float64")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["schedule_kinds"] = list(self.schedule_kinds)
        value["smoothing_half_lives_degrees"] = {
            key: list(items) for key, items in self.smoothing_half_lives_degrees.items()
        }
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


def load_retrospective_config(path: str | Path) -> RetrospectiveConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load Plan 9 config: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Plan 9 retrospective config must be an object")
    return RetrospectiveConfig.from_mapping(value)


@dataclasses.dataclass(frozen=True)
class ExtensionConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    study: str
    experiment: str
    replica_id: str
    replica_seed: int
    e9_1_run_path: str
    e9_1_run_id: str
    e9_2_run_path: str
    e9_2_run_id: str
    opportunity_thresholds: tuple[float, ...]
    primary_opportunity_threshold: float
    sustained_steps: int
    cross_half_lives_degrees: dict[str, tuple[float, ...]]
    primary_cross_half_lives_degrees: dict[str, float]
    lag_exclusions: tuple[int, ...]
    primary_lag_exclusion: int
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExtensionConfig":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "Plan 9 extension configuration",
        )
        if (
            not isinstance(value["opportunity_thresholds"], list)
            or not isinstance(value["cross_half_lives_degrees"], Mapping)
            or not isinstance(value["primary_cross_half_lives_degrees"], Mapping)
            or not isinstance(value["lag_exclusions"], list)
        ):
            raise RotatedConfigError("Plan 9 extension collections are invalid")
        result = cls(
            schema_version=_integer(value["schema_version"], minimum=1, name="schema_version"),
            metric_schema_version=_integer(
                value["metric_schema_version"], minimum=1, name="metric_schema_version"
            ),
            artifact_schema_version=_integer(
                value["artifact_schema_version"], minimum=1, name="artifact_schema_version"
            ),
            study=str(value["study"]),
            experiment=str(value["experiment"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(value["replica_seed"], minimum=0, name="replica_seed"),
            e9_1_run_path=_relative_path(value["e9_1_run_path"], name="e9_1_run_path"),
            e9_1_run_id=str(value["e9_1_run_id"]),
            e9_2_run_path=_relative_path(value["e9_2_run_path"], name="e9_2_run_path"),
            e9_2_run_id=str(value["e9_2_run_id"]),
            opportunity_thresholds=tuple(
                _unit_interval(item, name="opportunity_thresholds")
                for item in value["opportunity_thresholds"]
            ),
            primary_opportunity_threshold=_unit_interval(
                value["primary_opportunity_threshold"],
                name="primary_opportunity_threshold",
            ),
            sustained_steps=_integer(
                value["sustained_steps"], minimum=1, name="sustained_steps"
            ),
            cross_half_lives_degrees={
                str(design): tuple(
                    _positive(item, name=f"cross_half_lives_degrees.{design}")
                    for item in items
                )
                for design, items in value["cross_half_lives_degrees"].items()
            },
            primary_cross_half_lives_degrees={
                str(design): _positive(
                    item, name=f"primary_cross_half_lives_degrees.{design}"
                )
                for design, item in value["primary_cross_half_lives_degrees"].items()
            },
            lag_exclusions=tuple(
                _integer(item, minimum=0, name="lag_exclusions")
                for item in value["lag_exclusions"]
            ),
            primary_lag_exclusion=_integer(
                value["primary_lag_exclusion"],
                minimum=0,
                name="primary_lag_exclusion",
            ),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != PLAN9_CONFIG_SCHEMA_VERSION
            or self.metric_schema_version != PLAN9_METRIC_SCHEMA_VERSION
            or self.artifact_schema_version != PLAN9_ARTIFACT_SCHEMA_VERSION
        ):
            raise RotatedConfigError("unsupported Plan 9 extension schema")
        if self.study not in {"e9_7", "e9_8"} or not self.experiment.startswith(
            f"rotated_mnist_plan9_{self.study}_"
        ):
            raise RotatedConfigError("invalid Plan 9 extension identity")
        if not self.replica_id or self.replica_seed >= 2**63:
            raise RotatedConfigError("invalid Plan 9 extension replica identity")
        expected_runs = {
            "e9_1": "rotated_mnist_plan9_e9_1_anchor_cancellation__replica-0001__b1234a5779ce2e42",
            "e9_2": "rotated_mnist_plan9_e9_2_smooth_drift_v2__replica-0001__b4c838dcc0b327bf",
        }
        for path, run_id, expected in (
            (self.e9_1_run_path, self.e9_1_run_id, expected_runs["e9_1"]),
            (self.e9_2_run_path, self.e9_2_run_id, expected_runs["e9_2"]),
        ):
            if Path(path).name != run_id or run_id != expected:
                raise RotatedConfigError("Plan 9 extension source identity drifted")
        if (
            self.opportunity_thresholds != (0.01, 0.02, 0.05)
            or self.primary_opportunity_threshold != 0.02
            or self.sustained_steps != 8
        ):
            raise RotatedConfigError("Plan 9 opportunity gate drifted")
        expected_half_lives = {
            "single_lap": (7.5, 15.0, 30.0),
            "double_lap": (1.875, 3.75, 7.5),
        }
        expected_primary = {"single_lap": 15.0, "double_lap": 3.75}
        if (
            self.cross_half_lives_degrees != expected_half_lives
            or self.primary_cross_half_lives_degrees != expected_primary
            or self.lag_exclusions != (1, 2, 4)
            or self.primary_lag_exclusion != 1
        ):
            raise RotatedConfigError("Plan 9 cross-moment family drifted")
        if self.runtime.device != "cpu" or self.runtime.dtype != "float64":
            raise RotatedConfigError("Plan 9 extension arithmetic is CPU float64")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["opportunity_thresholds"] = list(self.opportunity_thresholds)
        value["lag_exclusions"] = list(self.lag_exclusions)
        value["cross_half_lives_degrees"] = {
            key: list(items) for key, items in self.cross_half_lives_degrees.items()
        }
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


def load_extension_config(path: str | Path) -> ExtensionConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load Plan 9 extension config: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Plan 9 extension config must be an object")
    return ExtensionConfig.from_mapping(value)


@dataclasses.dataclass(frozen=True)
class AttributionConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    study: str
    experiment: str
    replica_id: str
    replica_seed: int
    e9_1_run_path: str
    e9_1_run_id: str
    condition: str
    optimizer_step_budgets: tuple[int, ...]
    primary_optimizer_step_budget: int
    sensitivity_transitions_per_group: int
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AttributionConfig":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "Plan 9 attribution configuration",
        )
        if not isinstance(value["optimizer_step_budgets"], list):
            raise RotatedConfigError("optimizer_step_budgets must be a list")
        result = cls(
            schema_version=_integer(value["schema_version"], minimum=1, name="schema_version"),
            metric_schema_version=_integer(
                value["metric_schema_version"], minimum=1, name="metric_schema_version"
            ),
            artifact_schema_version=_integer(
                value["artifact_schema_version"], minimum=1, name="artifact_schema_version"
            ),
            study=str(value["study"]),
            experiment=str(value["experiment"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(value["replica_seed"], minimum=0, name="replica_seed"),
            e9_1_run_path=_relative_path(value["e9_1_run_path"], name="e9_1_run_path"),
            e9_1_run_id=str(value["e9_1_run_id"]),
            condition=str(value["condition"]),
            optimizer_step_budgets=tuple(
                _integer(item, minimum=1, name="optimizer_step_budgets")
                for item in value["optimizer_step_budgets"]
            ),
            primary_optimizer_step_budget=_integer(
                value["primary_optimizer_step_budget"],
                minimum=1,
                name="primary_optimizer_step_budget",
            ),
            sensitivity_transitions_per_group=_integer(
                value["sensitivity_transitions_per_group"],
                minimum=2,
                name="sensitivity_transitions_per_group",
            ),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != PLAN9_CONFIG_SCHEMA_VERSION
            or self.metric_schema_version != PLAN9_METRIC_SCHEMA_VERSION
            or self.artifact_schema_version != PLAN9_ARTIFACT_SCHEMA_VERSION
            or self.study != "e9_9"
            or not self.experiment.startswith("rotated_mnist_plan9_e9_9_")
        ):
            raise RotatedConfigError("invalid Plan 9 attribution identity")
        expected = "rotated_mnist_plan9_e9_1_anchor_cancellation__replica-0001__b1234a5779ce2e42"
        if Path(self.e9_1_run_path).name != self.e9_1_run_id or self.e9_1_run_id != expected:
            raise RotatedConfigError("Plan 9 attribution source identity drifted")
        if (
            self.condition != "decomposed_edr"
            or self.optimizer_step_budgets != (50, 100)
            or self.primary_optimizer_step_budget != 50
            or self.sensitivity_transitions_per_group != 12
        ):
            raise RotatedConfigError("Plan 9 attribution design drifted")
        if self.runtime.device != "cuda" or self.runtime.dtype != "float32":
            raise RotatedConfigError("Plan 9 attribution fits require CUDA float32")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["optimizer_step_budgets"] = list(self.optimizer_step_budgets)
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


@dataclasses.dataclass(frozen=True)
class PathScreenConfig:
    schema_version: int
    metric_schema_version: int
    artifact_schema_version: int
    study: str
    experiment: str
    replica_id: str
    replica_seed: int
    e9_7_run_path: str
    e9_7_run_id: str
    transitions_per_arrow: tuple[int, ...]
    fixed_pi: float
    initial_effective_size: int
    deployed_batch_size: int
    action_lift_threshold: float
    sustained_steps: int
    maximum_mean_drift_variation_ratio: float
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PathScreenConfig":
        _strict_keys(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "Plan 9 path-screen configuration",
        )
        if not isinstance(value["transitions_per_arrow"], list):
            raise RotatedConfigError("transitions_per_arrow must be a list")
        result = cls(
            schema_version=_integer(value["schema_version"], minimum=1, name="schema_version"),
            metric_schema_version=_integer(
                value["metric_schema_version"], minimum=1, name="metric_schema_version"
            ),
            artifact_schema_version=_integer(
                value["artifact_schema_version"], minimum=1, name="artifact_schema_version"
            ),
            study=str(value["study"]),
            experiment=str(value["experiment"]),
            replica_id=str(value["replica_id"]),
            replica_seed=_integer(value["replica_seed"], minimum=0, name="replica_seed"),
            e9_7_run_path=_relative_path(value["e9_7_run_path"], name="e9_7_run_path"),
            e9_7_run_id=str(value["e9_7_run_id"]),
            transitions_per_arrow=tuple(
                _integer(item, minimum=1, name="transitions_per_arrow")
                for item in value["transitions_per_arrow"]
            ),
            fixed_pi=_unit_interval(value["fixed_pi"], name="fixed_pi"),
            initial_effective_size=_integer(
                value["initial_effective_size"], minimum=1, name="initial_effective_size"
            ),
            deployed_batch_size=_integer(
                value["deployed_batch_size"], minimum=1, name="deployed_batch_size"
            ),
            action_lift_threshold=_unit_interval(
                value["action_lift_threshold"], name="action_lift_threshold"
            ),
            sustained_steps=_integer(
                value["sustained_steps"], minimum=1, name="sustained_steps"
            ),
            maximum_mean_drift_variation_ratio=_positive(
                value["maximum_mean_drift_variation_ratio"],
                name="maximum_mean_drift_variation_ratio",
            ),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != PLAN9_CONFIG_SCHEMA_VERSION
            or self.metric_schema_version != PLAN9_METRIC_SCHEMA_VERSION
            or self.artifact_schema_version != PLAN9_ARTIFACT_SCHEMA_VERSION
            or self.study != "e9_10"
            or not self.experiment.startswith("rotated_mnist_plan9_e9_10_")
        ):
            raise RotatedConfigError("invalid Plan 9 path-screen identity")
        expected = "rotated_mnist_plan9_e9_7_movement_opportunity__replica-0001__6109e38ad6876642"
        if Path(self.e9_7_run_path).name != self.e9_7_run_id or self.e9_7_run_id != expected:
            raise RotatedConfigError("Plan 9 path-screen source identity drifted")
        if (
            self.transitions_per_arrow != (20, 10, 5, 2)
            or self.fixed_pi != 0.05
            or self.initial_effective_size != 30_000
            or self.deployed_batch_size != 4
            or self.action_lift_threshold != 0.02
            or self.sustained_steps != 8
            or self.maximum_mean_drift_variation_ratio != 0.25
        ):
            raise RotatedConfigError("Plan 9 path-screen design drifted")
        if self.runtime.device != "cpu" or self.runtime.dtype != "float64":
            raise RotatedConfigError("Plan 9 path screen requires CPU float64")

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["transitions_per_arrow"] = list(self.transitions_per_arrow)
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


def _load_strict_config(path: str | Path, cls):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedConfigError(f"could not load Plan 9 config: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RotatedConfigError("Plan 9 config must be an object")
    return cls.from_mapping(value)


def load_attribution_config(path: str | Path) -> AttributionConfig:
    return _load_strict_config(path, AttributionConfig)


def load_path_screen_config(path: str | Path) -> PathScreenConfig:
    return _load_strict_config(path, PathScreenConfig)
