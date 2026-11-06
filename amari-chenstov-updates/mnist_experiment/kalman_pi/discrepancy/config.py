"""Immutable settings for padded registration with sustained gradual drift."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.seeding import derive_component_seed

from ..registration.config import ROLE_COUNTS, TRAIN_ROLES, content_hash
from . import PROTOCOL, SCHEMA_VERSION


TUPLE_FIELDS = ("role_counts", "schedule_factors", "amplitudes", "process_scales",
                "ridge_candidates", "ema_half_lives", "retention_ridges")


@dataclasses.dataclass(frozen=True)
class Config:
    schema_version: int = SCHEMA_VERSION
    protocol: str = PROTOCOL
    seed: int = 171105
    device: str = "cuda"
    dtype: str = "float32"
    fisher_dtype: str = "float64"
    threads: int = 2
    smoke: bool = False
    samples_per_step: int = 4
    hold_steps: int = 352
    ramp_steps: int = 304
    initialization_size: int = 24000
    initial_epochs: int = 15
    initial_max_epochs: int = 30
    initial_batch_size: int = 256
    initial_lr: float = .001
    validation_size: int = 1000
    reference_size: int = 1024
    reference_max_size: int = 2048
    calibration_size: int = 2048
    diagnostic_size: int = 256
    capability_size: int = 256
    capability_epochs: int = 20
    eval_per_class: int = 100
    eval_every: int = 16
    max_iter: int = 50
    max_eval: int = 75
    learning_rate: float = 1.
    history_size: int = 100
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    chunk_size: int = 8
    fixed_pi: float = .025
    ridge_ratio: float = .1
    ridge_candidates: tuple[float, ...] = (0., .01, .1)
    # Retained for compatible helper APIs; no EMA or retention-companion arms.
    retention_ridges: tuple[float, ...] = ()
    retention_tolerance_pixels: float = .10
    ema_half_lives: tuple[float, ...] = ()
    noise_half_life: float = 32.
    cold_start: int = 16
    pi_min: float = .0001
    pi_max: float = .2
    anchor_refresh: int = 0
    observer_floor_relative: float = 1e-6
    observer_floor_absolute: float = 1e-12
    process_scales: tuple[float, ...] = (1e-5, 1e-4, 1e-3)
    schedule_factors: tuple[float, ...] = (1., 1.25)
    amplitudes: tuple[float, ...] = (25., 45.)
    angle_max: float = 25.
    angle_std: float = 6.
    angle_limit: float = 70.
    role_counts: tuple[int, ...] = ROLE_COUNTS
    bootstrap_draws: int = 2000
    main_replicas: int = 8
    stationary_replicas: int = 4
    dev_replicas: int = 2
    shadow_replicas: int = 2
    candidates_per_family: int = 2
    max_bytes: int = 500_000_000_000
    min_free_bytes: int = 20_000_000_000

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_p_steps(self) -> int:
        return 2 * self.hold_steps + 2 * self.ramp_steps

    @property
    def role_sizes(self) -> dict[str, int]:
        return dict(zip(TRAIN_ROLES, self.role_counts, strict=True))

    @property
    def scale_factors(self) -> tuple[float, ...]:
        return self.schedule_factors

    @property
    def fresh_paths(self) -> int:
        return 4 * (len(self.amplitudes) * self.main_replicas + self.stationary_replicas)

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported discrepancy schema_version")
        if self.protocol != PROTOCOL:
            raise ValueError("unsupported discrepancy protocol")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(self.smoke) is not bool:
            raise ValueError("smoke must be a boolean")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.dtype not in ("float32", "float64") or self.fisher_dtype != "float64":
            raise ValueError("training uses float32/float64; Fisher accumulation must use float64")
        for name in ("threads", "samples_per_step", "hold_steps", "ramp_steps",
                     "initialization_size", "initial_epochs", "initial_max_epochs",
                     "initial_batch_size", "validation_size", "reference_size",
                     "reference_max_size", "calibration_size", "diagnostic_size",
                     "capability_size", "capability_epochs", "eval_per_class", "eval_every",
                     "max_iter", "max_eval", "history_size", "chunk_size", "cold_start",
                     "bootstrap_draws", "shadow_replicas", "candidates_per_family",
                     "max_bytes", "min_free_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, bound in (("dev_replicas", 2), ("main_replicas", 8), ("stationary_replicas", 4)):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= bound:
                raise ValueError(f"{name} must be an integer in [0, {bound}]")
        for name in ("initial_lr", "learning_rate", "tolerance_grad", "tolerance_change",
                     "noise_half_life", "observer_floor_relative", "observer_floor_absolute",
                     "angle_max", "angle_std", "angle_limit", "retention_tolerance_pixels"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("pi_min", "pi_max", "fixed_pi", "ridge_ratio"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.pi_min <= self.fixed_pi <= self.pi_max < 1 or self.ridge_ratio < 0:
            raise ValueError("require 0 < pi_min <= fixed_pi <= pi_max < 1 and nonnegative ridge")
        if self.ramp_steps < 2 or self.cold_start >= self.num_p_steps:
            raise ValueError("ramps need both endpoints; cold_start must leave a later arrival")
        if self.max_eval < self.max_iter or self.initial_max_epochs < self.initial_epochs:
            raise ValueError("maximum solver/initialization budgets must cover initial budgets")
        if not 2 <= self.calibration_size <= self.initialization_size:
            raise ValueError("calibration must reuse at least two already-seen initialization examples")
        if self.reference_size > self.reference_max_size:
            raise ValueError("reference_size must not exceed its nested reserve")
        if self.angle_limit != 70. or self.angle_std != 6.:
            raise ValueError("rotation law requires std=6 and support=[-70,70]")
        if type(self.anchor_refresh) is not int or self.anchor_refresh != 0:
            raise ValueError("this protocol uses a fixed initialization anchor")
        if self.shadow_replicas > 2 or self.candidates_per_family > 2:
            raise ValueError("observer development is bounded at two replicas/candidates per family")
        if (not isinstance(self.role_counts, tuple) or self.role_counts != ROLE_COUNTS
                or any(type(n) is not int for n in self.role_counts)):
            raise ValueError("study-wide role_counts must equal the fixed allocation")
        for name in ("schedule_factors", "amplitudes", "process_scales", "ridge_candidates"):
            values = getattr(self, name)
            minimum = 0 if name == "ridge_candidates" else 1e-300
            if (not isinstance(values, tuple) or not values
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < minimum
                           for v in values) or len(set(values)) != len(values)):
                raise ValueError(f"{name} must contain distinct finite valid numbers")
        if self.schedule_factors != (1., 1.25) or self.amplitudes != (25., 45.):
            raise ValueError("length factors and amplitudes are frozen at (1,1.25) and (25,45)")
        if self.angle_max not in self.amplitudes:
            raise ValueError("angle_max must be one of the configured amplitudes")
        if self.ema_half_lives != () or self.retention_ridges != ():
            raise ValueError("Phase 7 has no EMA or retention-companion conditions")
        if (self.initialization_size > self.role_sizes["initialization"]
                or self.validation_size > self.role_sizes["validation"]
                or self.reference_max_size > self.role_sizes["reference"]
                or 3 * self.diagnostic_size + self.capability_size > self.role_sizes["diagnostic"]):
            raise ValueError("requested samples exceed their study-wide source roles")
        if not self.smoke:
            if self.samples_per_step != 4:
                raise ValueError("production arrivals contain four new pairs")
            if self.initial_max_epochs > 30 or self.capability_epochs > 20:
                raise ValueError("initialization/capability epochs exceed the Phase 7 allowance")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Config":
        if not isinstance(values, Mapping):
            raise ValueError("configuration must be a mapping")
        unknown = set(values) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        resolved = dict(values)
        for name in TUPLE_FIELDS:
            if name in resolved:
                if not isinstance(resolved[name], (tuple, list)):
                    raise ValueError(f"{name} must be a sequence")
                resolved[name] = tuple(resolved[name])
        return cls(**resolved)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def scientific_hash(self) -> str:
        values = self.mapping()
        for name in ("device", "threads", "dev_replicas", "main_replicas",
                     "stationary_replicas", "shadow_replicas", "max_bytes", "min_free_bytes"):
            values.pop(name)
        return content_hash(values)

    def seed_for(self, cohort: str, replica: int, component: str) -> int:
        if not isinstance(cohort, str) or not cohort or not isinstance(component, str) or not component:
            raise ValueError("cohort and component must be nonempty strings")
        if type(replica) is not int or replica < 0:
            raise ValueError("replica must be a nonnegative integer")
        return derive_component_seed(self.seed, f"{self.protocol}:{cohort}:{replica}:{component}")


def load(path: str | Path) -> Config:
    return Config.from_json(path)


def from_mapping(values: Mapping[str, Any]) -> Config:
    return Config.from_mapping(values)


def smoke_config(**overrides: Any) -> Config:
    values = dict(smoke=True, device="cpu", threads=1, hold_steps=2, ramp_steps=2,
                  initialization_size=16, initial_epochs=1, initial_max_epochs=1,
                  initial_batch_size=8, validation_size=10, reference_size=4,
                  reference_max_size=8, calibration_size=8, diagnostic_size=4,
                  capability_size=4, capability_epochs=1, eval_per_class=1, eval_every=2,
                  max_iter=2, max_eval=4, history_size=3, chunk_size=2, cold_start=2,
                  bootstrap_draws=16)
    return Config.from_mapping(values | overrides)
