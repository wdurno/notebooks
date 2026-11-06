"""Plan 17 settings on the frozen Plan 16 learner/data contract."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

from mnist_experiment.diagonal_lfu.config import (
    Config as BaseConfig,
    PROTOCOL as BASE_PROTOCOL,
    SCHEMA_VERSION,
    content_hash,
)


PROTOCOL = "plan17-kalman-pi-v1"
TUPLE_FIELDS = ("control_cs", "ema_half_lives", "process_scales", "schedule_factors")


@dataclasses.dataclass(frozen=True)
class Config(BaseConfig):
    protocol: str = PROTOCOL
    seed: int = 171101
    pi_min: float = 1e-4
    pi_max: float = .2
    cold_start: int = 16
    noise_half_life: float = 32.
    anchor_refresh: int = 64
    observer_floor_relative: float = 1e-6
    observer_floor_absolute: float = 1e-12
    ema_half_lives: tuple[float, ...] = (8., 16., 32.)
    process_scales: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    schedule_factors: tuple[float, ...] = (1., 1.25)
    calibration_size: int = 256
    shadow_replicas: int = 1
    candidates_per_family: int = 2

    def validate(self) -> None:
        # Construct the frozen validator, not replace(self), which would recurse.
        shared = {field.name: getattr(self, field.name) for field in dataclasses.fields(BaseConfig)}
        shared["protocol"] = BASE_PROTOCOL
        BaseConfig(**shared)
        if self.protocol != PROTOCOL:
            raise ValueError("unsupported Plan 17 protocol")
        for name in ("pi_min", "pi_max", "noise_half_life", "observer_floor_relative",
                     "observer_floor_absolute"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.pi_min <= self.pi_max < 1:
            raise ValueError("actions must satisfy 0 < pi_min <= pi_max < 1")
        if type(self.cold_start) is not int or not 1 <= self.cold_start < self.num_p_steps:
            raise ValueError("cold_start must be positive and leave a post-cold-start arrival")
        if type(self.anchor_refresh) is not int or self.anchor_refresh < 0:
            raise ValueError("anchor_refresh must be a nonnegative integer (zero fixes the anchor)")
        for name in ("calibration_size", "shadow_replicas", "candidates_per_family"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.calibration_size <= self.initialization_size:
            raise ValueError("calibration_size must be between 2 and initialization_size")
        if self.shadow_replicas > 2 or self.candidates_per_family > 2:
            raise ValueError("shadow_replicas and candidates_per_family are bounded at two")
        for name in TUPLE_FIELDS[1:]:
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not values
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{name} must be a nonempty tuple of distinct finite positive numbers")
        if any(factor not in (1., 1.25) for factor in self.schedule_factors):
            raise ValueError("schedule_factors must be drawn from (1, 1.25)")
        if not self.smoke:
            if self.hold_steps < 5 * max(self.noise_half_life, *self.ema_half_lives):
                raise ValueError("production holds must span five observer/noise half-lives")
            if self.hold_steps < self.anchor_refresh:
                raise ValueError("production holds must span an anchor_refresh interval")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Config":
        if not isinstance(values, Mapping):
            raise ValueError("configuration must be a mapping")
        resolved = dict(values)
        for name in TUPLE_FIELDS:
            if name in resolved:
                if not isinstance(resolved[name], (list, tuple)):
                    raise ValueError(f"{name} must be a sequence")
                resolved[name] = tuple(resolved[name])
        return super().from_mapping(resolved)

    @property
    def scientific_hash(self) -> str:
        values = self.mapping()
        for name in ("device", "threads", "dev_replicas", "main_replicas", "stationary_replicas",
                     "shadow_replicas"):
            values.pop(name)
        return content_hash(values)
