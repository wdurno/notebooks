"""Configuration and comparisons for local EWC fit-budget calibration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .config import ConfigError, OptimizerConfig, RuntimeConfig

FIT_CALIBRATION_SCHEMA_VERSION = 2
SUPPORTED_FIT_CALIBRATION_SCHEMA_VERSIONS = (1, 2)
FIT_CALIBRATION_ARTIFACT_SCHEMA_VERSION = 1
FIT_CALIBRATION_METRIC_SCHEMA_VERSION = 2
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclasses.dataclass(frozen=True)
class FitCalibrationConfig:
    schema_version: int
    artifact_schema_version: int
    metric_schema_version: int
    experiment: str
    replica_id: str
    replica_seed: int
    cache_root: str
    source_run: str
    checkpoint_steps: list[int]
    inner_step_budgets: list[int]
    runtime: RuntimeConfig
    optimizer: OptimizerConfig | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FitCalibrationConfig":
        if not isinstance(value, Mapping):
            raise ConfigError("fit-calibration configuration must be an object")
        schema_version = value.get("schema_version")
        expected = {field.name for field in dataclasses.fields(cls)}
        if schema_version == 1:
            expected.remove("optimizer")
        supplied = set(value)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unknown = sorted(supplied - expected)
            raise ConfigError(
                "invalid fit-calibration configuration: "
                f"missing={missing}, unknown={unknown}"
            )
        runtime_value = value["runtime"]
        if not isinstance(runtime_value, Mapping):
            raise ConfigError("runtime must be an object")
        runtime_expected = {
            field.name for field in dataclasses.fields(RuntimeConfig)
        }
        if set(runtime_value) != runtime_expected:
            raise ConfigError("invalid fit-calibration runtime configuration")
        optimizer = None
        if schema_version == 2:
            optimizer_value = value["optimizer"]
            if not isinstance(optimizer_value, Mapping):
                raise ConfigError("optimizer must be an object")
            optimizer_expected = {
                field.name for field in dataclasses.fields(OptimizerConfig)
            }
            if set(optimizer_value) != optimizer_expected:
                raise ConfigError("invalid fit-calibration optimizer configuration")
            optimizer = OptimizerConfig(**optimizer_value)
        config_mapping = {
            **dict(value),
            "checkpoint_steps": list(value["checkpoint_steps"]),
            "inner_step_budgets": list(value["inner_step_budgets"]),
            "runtime": RuntimeConfig(**runtime_value),
        }
        if schema_version == 2:
            config_mapping["optimizer"] = optimizer
        config = cls(**config_mapping)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version not in SUPPORTED_FIT_CALIBRATION_SCHEMA_VERSIONS:
            raise ConfigError("fit-calibration schema_version must be 1 or 2")
        if (
            self.artifact_schema_version
            != FIT_CALIBRATION_ARTIFACT_SCHEMA_VERSION
        ):
            raise ConfigError("fit-calibration artifact_schema_version must be 1")
        expected_metric_schema = 1 if self.schema_version == 1 else 2
        if self.metric_schema_version != expected_metric_schema:
            raise ConfigError(
                "fit-calibration metric_schema_version must match schema version"
            )
        for name, value in {
            "experiment": self.experiment,
            "replica_id": self.replica_id,
        }.items():
            if (
                not isinstance(value, str)
                or not _IDENTIFIER_PATTERN.fullmatch(value)
            ):
                raise ConfigError(f"{name} is not a valid identifier")
        if (
            not isinstance(self.replica_seed, int)
            or isinstance(self.replica_seed, bool)
            or not 0 <= self.replica_seed < 2**63
        ):
            raise ConfigError("replica_seed must be an integer in [0, 2**63)")
        for name, value in {
            "cache_root": self.cache_root,
            "source_run": self.source_run,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{name} must be a nonempty path")
        for name, values, minimum in (
            ("checkpoint_steps", self.checkpoint_steps, 0),
            ("inner_step_budgets", self.inner_step_budgets, 1),
        ):
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(item, int)
                    or isinstance(item, bool)
                    or item < minimum
                    for item in values
                )
                or values != sorted(set(values))
            ):
                raise ConfigError(
                    f"{name} must be a nonempty increasing list of unique integers"
                )
        self.runtime.validate()
        if self.schema_version == 1 and self.optimizer is not None:
            raise ConfigError("schema-v1 fit calibration cannot override optimizer")
        if self.schema_version == 2:
            if self.optimizer is None:
                raise ConfigError("schema-v2 fit calibration requires optimizer")
            self.optimizer.validate()

    def to_mapping(self) -> dict[str, Any]:
        mapping = dataclasses.asdict(self)
        if self.schema_version == 1:
            mapping.pop("optimizer")
        return mapping

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
        return f"{self.experiment}__{self.replica_id}__{self.config_hash[:16]}"


def load_fit_calibration_config(path: str | Path) -> FitCalibrationConfig:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not read configuration {source}: {exc}") from exc
    return FitCalibrationConfig.from_mapping(value)


def displacement_comparison(
    estimate: Tensor,
    reference: Tensor,
) -> dict[str, float | None]:
    if estimate.shape != reference.shape or estimate.ndim != 1:
        raise ValueError("displacements must be equal-length vectors")
    if (
        estimate.dtype != reference.dtype
        or estimate.device != reference.device
        or not torch.isfinite(estimate).all()
        or not torch.isfinite(reference).all()
    ):
        raise ValueError("displacements must be finite and share dtype/device")
    difference = torch.linalg.vector_norm(estimate - reference)
    reference_norm = torch.linalg.vector_norm(reference)
    denominator = torch.linalg.vector_norm(estimate) * reference_norm
    return {
        "distance": float(difference),
        "relative_distance_to_reference": float(
            difference / reference_norm.clamp_min(torch.finfo(reference.dtype).tiny)
        ),
        "cosine": (
            None
            if float(denominator) == 0.0
            else float((estimate @ reference) / denominator)
        ),
    }
