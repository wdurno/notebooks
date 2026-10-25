"""Resolved, versioned settings for the regression study and small acceptance."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

from src.seeding import derive_component_seed

from . import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
METHODS = ("fd_replay_ewc", "replay_ewc", "ewc_only")
SCHEMA = 1


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Config:
    protocol: str = PROTOCOL
    schema_version: int = SCHEMA
    smoke: bool = False
    seed: int = 2026102408
    device: str = "cuda"
    dtype: str = "float32"
    threads: int = 2
    angle_scale: float = 45.
    init_size: int = 30_000
    init_validation_size: int = 1_000
    reference_size: int = 1_024
    init_epochs: int = 15
    init_batch: int = 256
    init_learning_rate: float = .001
    eval_per_class: int = 100
    evaluation_chunk: int = 256
    fisher_chunk: int = 16
    nominal_batch: int = 4
    ramp_groups: int = 40
    novel_probability: float = .1
    novel_count: int = 100
    screen_novel_count: int = 30
    calibration_replicas: int = 2
    allocation_replicas: int = 2
    refinement_replicas: int = 2
    confirmation_replicas: int = 12
    preferred_min_confirmation: int = 8
    reference_capacity: int = 64
    reference_replay: int = 16
    capacities: tuple[int, ...] = (16, 64, 256)
    replay_counts: tuple[int, ...] = (4, 16, 64)
    ridge_grid: tuple[float, ...] = (.1, 1., 10.)
    strength_ratios: tuple[float, ...] = (.03, .3, 3.)
    smaller_budgets: tuple[int, ...] = (25, 50, 75)
    retention_margin_degrees: float = 1.
    acquisition_tie_fraction: float = .01
    envelope_factor: float = 1.25
    pi: float = .025
    probes: int = 2
    fd_step: float = .01
    max_iter: int = 100
    max_eval: int = 150
    audit_multiplier: int = 2
    audit_objective_tolerance: float = .001
    numerical_tolerance: float = .05
    bootstrap_samples: int = 2_000
    analysis_reserve_seconds: float = 1_800.
    confirmation_reserve_seconds: float = 36_000.
    max_artifact_bytes: int = 500_000_000_000
    free_reserve_bytes: int = 2_000_000_000

    def mapping(self) -> dict:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    @property
    def scientific_hash(self) -> str:
        return content_hash(self.mapping())

    def seed_for(self, family: str, replica: int, component: str) -> int:
        return derive_component_seed(self.seed, f"{PROTOCOL}:{family}:{replica}:{component}")

    def validate(self) -> None:
        if self.protocol != PROTOCOL or self.schema_version != SCHEMA:
            raise ValueError("unsupported regression protocol/schema")
        if self.device not in ("cpu", "cuda") or self.dtype not in ("float32", "float64"):
            raise ValueError("unsupported device/dtype")
        if type(self.smoke) is not bool:
            raise ValueError("smoke must be boolean")
        for name in ("seed", "threads", "init_size", "init_validation_size", "reference_size",
                     "init_epochs", "init_batch", "eval_per_class", "evaluation_chunk",
                     "fisher_chunk", "nominal_batch", "ramp_groups", "novel_count",
                     "screen_novel_count", "calibration_replicas", "allocation_replicas",
                     "confirmation_replicas", "preferred_min_confirmation", "reference_capacity",
                     "reference_replay", "probes", "max_iter", "max_eval", "audit_multiplier",
                     "bootstrap_samples", "max_artifact_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        if type(self.refinement_replicas) is not int or self.refinement_replicas < 0:
            raise ValueError("invalid refinement replica count")
        if type(self.free_reserve_bytes) is not int or self.free_reserve_bytes < 0:
            raise ValueError("invalid filesystem reserve")
        for name in ("init_learning_rate", "retention_margin_degrees", "acquisition_tie_fraction",
                     "envelope_factor", "fd_step", "audit_objective_tolerance",
                     "numerical_tolerance", "analysis_reserve_seconds", "confirmation_reserve_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.angle_scale != 45. or self.nominal_batch != 4:
            raise ValueError("likelihood scale and archive exposure unit are frozen")
        if not 0 < self.pi < 1 or not 0 < self.novel_probability <= 1 or self.ramp_groups < 2:
            raise ValueError("invalid archive rate or drip schedule")
        if not 1 <= self.screen_novel_count < self.novel_count:
            raise ValueError("screen must precede full novel budget")
        if self.reference_replay > self.reference_capacity:
            raise ValueError("reference replay exceeds its capacity")
        for name in ("capacities", "replay_counts", "smaller_budgets"):
            values = getattr(self, name)
            if not values or any(type(v) is not int or v <= 0 for v in values) or len(set(values)) != len(values):
                raise ValueError(f"invalid {name}")
        if max(self.smaller_budgets) >= self.novel_count:
            raise ValueError("smaller budgets must be below the terminal budget")
        for name in ("ridge_grid", "strength_ratios"):
            values = getattr(self, name)
            if not values or any(not math.isfinite(v) or v <= 0 for v in values) or len(set(values)) != len(values):
                raise ValueError(f"invalid {name}")
        if self.max_artifact_bytes > 500_000_000_000:
            raise ValueError("the shared 500 GB ceiling cannot be increased")
        if not self.smoke and (self.novel_count != 100 or self.novel_probability != .1 or self.ramp_groups != 40):
            raise ValueError("changing the production data scope requires a versioned protocol")

    @classmethod
    def read(cls, path: Path | str) -> "Config":
        values = json.loads(Path(path).read_text())
        for name in ("capacities", "replay_counts", "ridge_grid", "strength_ratios", "smaller_budgets"):
            if name in values:
                values[name] = tuple(values[name])
        result = cls(**values)
        result.validate()
        return result


@dataclasses.dataclass(frozen=True)
class Recipe:
    method: str
    capacity: int
    replay_count: int
    ridge: float = 1.
    strength: float = 0.
    fd_step: float = .01
    probes: int = 2
    max_iter: int = 100
    max_eval: int = 150

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError("unknown regression treatment")
        if type(self.capacity) is not int or type(self.replay_count) is not int:
            raise ValueError("capacity and replay count must be integers")
        if self.method == "ewc_only":
            if self.capacity != 0 or self.replay_count != 0:
                raise ValueError("EWC-only requires C=b=0")
        elif not 1 <= self.replay_count <= self.capacity:
            raise ValueError("replay policy requires 1 <= b <= C")
        for name in ("ridge", "strength"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if self.method != "fd_replay_ewc" and self.strength:
            raise ValueError("control cannot have FD strength")
        if self.method == "fd_replay_ewc" and not self.strength:
            raise ValueError("FD must have positive calibrated strength")
        if not math.isfinite(self.fd_step) or self.fd_step <= 0:
            raise ValueError("invalid finite-difference step")
        for name in ("probes", "max_iter", "max_eval"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")

    def mapping(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def key(self) -> str:
        return f"{self.method}-{content_hash(self.mapping())[:16]}"

    @classmethod
    def from_mapping(cls, value: dict) -> "Recipe":
        mapping = dict(value)
        recorded_key = mapping.pop("recipe_key", None)
        recipe = cls(**mapping)
        if recorded_key is not None and recorded_key != recipe.key:
            raise ValueError("recipe identity disagrees with its settings")
        return recipe

    @classmethod
    def from_config(cls, cfg: Config, method: str, *, capacity=None, replay_count=None,
                    ridge=1., strength=0., fd_step=None, policy=None) -> "Recipe":
        capacity = cfg.reference_capacity if capacity is None else capacity
        replay_count = cfg.reference_replay if replay_count is None else replay_count
        if method == "ewc_only":
            capacity, replay_count = 0, 0
        policy = policy or {"max_iter": cfg.max_iter, "max_eval": cfg.max_eval}
        return cls(method, capacity, replay_count, ridge, strength,
                   cfg.fd_step if fd_step is None else fd_step, cfg.probes, **policy)
