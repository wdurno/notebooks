"""Versioned scientific settings for the bounded Plan 16 study."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.seeding import derive_component_seed


SCHEMA_VERSION = 1
PROTOCOL = "plan16-diagonal-lfu-v1"
CONDITIONS = ("ema", "paired_fd", "analytic_lfu")


def content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class Config:
    schema_version: int = SCHEMA_VERSION
    protocol: str = PROTOCOL
    seed: int = 161101
    smoke: bool = False
    device: str = "cuda"
    dtype: str = "float32"
    fisher_dtype: str = "float64"
    threads: int = 2
    samples_per_step: int = 4
    hold_steps: int = 352
    ramp_steps: int = 128
    p_low: float = .005
    p_high: float = .1
    initialization_size: int = 2048
    reference_size: int = 1024
    validation_size: int = 512
    initial_epochs: int = 20
    initial_batch_size: int = 128
    initial_lr: float = .003
    eval_per_class: int = 200
    eval_every: int = 16
    max_iter: int = 50
    max_eval: int = 75
    learning_rate: float = 1.
    history_size: int = 100
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    control_cs: tuple[float, ...] = (.01, .025, .05)
    ridge_ratio: float = .1
    dev_replicas: int = 2
    main_replicas: int = 8
    stationary_replicas: int = 4
    chunk_size: int = 8
    bootstrap_draws: int = 2000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported Plan 16 schema_version")
        if self.protocol != PROTOCOL:
            raise ValueError("unsupported Plan 16 protocol")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(self.smoke) is not bool:
            raise ValueError("smoke must be a boolean")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.dtype not in ("float32", "float64") or self.fisher_dtype != "float64":
            raise ValueError("training requires float32/float64; Fisher accumulation requires float64")
        for name in (
            "threads", "samples_per_step", "hold_steps", "ramp_steps", "initialization_size",
            "reference_size", "validation_size", "initial_epochs", "initial_batch_size",
            "eval_per_class", "eval_every", "max_iter", "max_eval", "history_size",
            "chunk_size", "bootstrap_draws",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dev_replicas", "main_replicas", "stationary_replicas"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.ramp_steps < 2:
            raise ValueError("ramp_steps must include both endpoints (at least two)")
        for name in ("p_low", "p_high", "initial_lr", "learning_rate", "ridge_ratio",
                     "tolerance_grad", "tolerance_change"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.p_low < self.p_high <= 1:
            raise ValueError("prevalences must satisfy 0 <= p_low < p_high <= 1")
        if self.ridge_ratio < 0 or self.tolerance_grad <= 0 or self.tolerance_change <= 0:
            raise ValueError("ridge must be nonnegative and solver tolerances positive")
        if self.initial_lr <= 0 or self.learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.max_eval < self.max_iter:
            raise ValueError("max_eval must be at least max_iter")
        if not isinstance(self.control_cs, tuple) or not self.control_cs:
            raise ValueError("control_cs must be a nonempty tuple")
        if any(type(c) not in (int, float) or not math.isfinite(c) or not 0 < c < 1
               for c in self.control_cs):
            raise ValueError("control_cs must contain finite actions strictly between zero and one")
        if len(set(self.control_cs)) != len(self.control_cs):
            raise ValueError("control_cs must not contain duplicates")
        half_life = math.log(.5) / math.log1p(-min(self.control_cs))
        if not self.smoke and self.hold_steps <= 5 * half_life:
            raise ValueError("production holds must exceed five half-lives of the smallest action")

    @property
    def num_p_steps(self) -> int:
        return 3 * self.hold_steps + 2 * self.ramp_steps

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Config":
        if not isinstance(values, Mapping):
            raise ValueError("configuration must be a mapping")
        unknown = set(values) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        resolved = dict(values)
        if "control_cs" in resolved:
            if not isinstance(resolved["control_cs"], (list, tuple)):
                raise ValueError("control_cs must be a sequence")
            resolved["control_cs"] = tuple(resolved["control_cs"])
        return cls(**resolved)

    @classmethod
    def from_json(cls, path: Path | str) -> "Config":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def scientific_hash(self) -> str:
        # Placement and future replica targets do not change an existing trajectory.
        values = self.mapping()
        for name in ("device", "threads", "dev_replicas", "main_replicas", "stationary_replicas"):
            values.pop(name)
        return content_hash(values)

    def seed_for(self, cohort: str, replica: int, component: str) -> int:
        if not isinstance(cohort, str) or not cohort or not isinstance(component, str) or not component:
            raise ValueError("cohort and component must be nonempty strings")
        if type(replica) is not int or replica < 0:
            raise ValueError("replica must be a nonnegative integer")
        return derive_component_seed(self.seed, f"{self.protocol}:{cohort}:{replica}:{component}")
