"""Versioned Phase 9 settings, independent of the frozen regression protocol."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path

from mnist_experiment.fisher_memory_regression import PROTOCOL as ENGINE_PROTOCOL
from mnist_experiment.fisher_memory_regression.config import Config as EngineConfig
from mnist_experiment.fisher_memory_regression.config import METHODS, Recipe, content_hash
from src.seeding import derive_component_seed

from . import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
SCHEMA = 1


@dataclass(frozen=True)
class Context:
    key: str
    family: str
    gamma_max: float = 1.
    blur_max: float = 0.

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key or "--" in self.key:
            raise ValueError("context key must be nonempty and cannot contain '--'")
        if self.family not in ("gamma", "blur", "composite", "combined"):
            raise ValueError("context family must be gamma, blur, composite, or combined")
        if not math.isfinite(self.gamma_max) or self.gamma_max < 1:
            raise ValueError("gamma_max must be finite and at least one")
        if not math.isfinite(self.blur_max) or self.blur_max < 0:
            raise ValueError("blur_max must be finite and nonnegative")
        if self.family == "gamma" and (self.gamma_max <= 1 or self.blur_max != 0):
            raise ValueError("gamma context requires gamma_max > 1 and zero blur")
        if self.family == "blur" and (self.gamma_max != 1 or self.blur_max <= 0):
            raise ValueError("blur context requires nominal gamma and positive blur")
        if self.family in ("composite", "combined") and (self.gamma_max <= 1 or self.blur_max <= 0):
            raise ValueError("composite context requires gamma and blur")

    def mapping(self) -> dict:
        return dict(key=self.key, family=self.family,
                    gamma_max=self.gamma_max, blur_max=self.blur_max)

    @classmethod
    def from_mapping(cls, value: dict) -> "Context":
        return cls(**value)


def default_contexts() -> tuple[Context, ...]:
    return (Context("gamma18", "gamma", 1.8), Context("gamma30", "gamma", 3.),
            Context("blur06", "blur", blur_max=.6), Context("blur12", "blur", blur_max=1.2))


@dataclass(frozen=True)
class Config(EngineConfig):
    protocol: str = PROTOCOL
    schema_version: int = SCHEMA
    seed: int = 2026102509
    max_init_epochs: int = 45
    init_epoch_stages: tuple[int, ...] = (15, 30, 45)
    contexts: tuple[Context, ...] = field(default_factory=default_contexts)
    calibration_replicas: int = 2
    allocation_replicas: int = 2
    refinement_replicas: int = 2
    finalist_extra_replicas: int = 2
    confirmation_replicas: int = 12
    reference_capacity: int = 256
    reference_replay: int = 16
    capacities: tuple[int, ...] = (256,)
    replay_counts: tuple[int, ...] = (4, 16)
    ridge_grid: tuple[float, ...] = (1., 10., 100.)
    strength_ratios: tuple[float, ...] = (.1, 1., 10.)
    retention_margin_degrees: float = 1.
    matching_tolerance: float = .2
    historical_strength: float = 2.6356432540763373
    envelope_factor: float = 2.5
    refinement_settings: int = 8
    diagnostic_base_size: int = 512
    diagnostic_novel_size: int = 256
    diagnostic_epochs: int = 5
    max_diagnostic_epochs: int = 10
    diagnostic_batch: int = 128
    diagnostic_learning_rate: float = .001

    def seed_for(self, family: str, replica: int, component: str) -> int:
        family = family.split("--", 1)[0]
        return derive_component_seed(self.seed, f"{PROTOCOL}:{family}:{replica}:{component}")

    def validate(self) -> None:
        if self.protocol != PROTOCOL or self.schema_version != SCHEMA:
            raise ValueError("unsupported continuous-shift protocol/schema")
        values = {entry.name: getattr(self, entry.name) for entry in fields(EngineConfig)}
        values["protocol"] = ENGINE_PROTOCOL
        EngineConfig(**values).validate()
        for name in ("max_init_epochs", "diagnostic_base_size", "diagnostic_novel_size",
                     "diagnostic_epochs", "max_diagnostic_epochs", "diagnostic_batch"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("matching_tolerance", "historical_strength", "diagnostic_learning_rate"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        if self.matching_tolerance >= self.retention_margin_degrees:
            raise ValueError("matching tolerance must be smaller than the retention ceiling")
        stages = self.init_epoch_stages
        if (not isinstance(stages, tuple) or not stages
                or any(type(epoch) is not int or epoch <= 0 for epoch in stages)
                or tuple(sorted(set(stages))) != stages
                or stages[-1] != self.max_init_epochs or self.init_epochs not in stages):
            raise ValueError("invalid initialization epoch stages or maximum")
        if self.diagnostic_epochs > self.max_diagnostic_epochs:
            raise ValueError("diagnostic epochs exceed their maximum")
        if type(self.finalist_extra_replicas) is not int or self.finalist_extra_replicas < 0:
            raise ValueError("invalid finalist extra replica count")
        if type(self.refinement_settings) is not int or not 0 <= self.refinement_settings <= 8:
            raise ValueError("at most eight additional settings per context")
        if (not isinstance(self.contexts, tuple) or not 1 <= len(self.contexts) <= 5
                or any(not isinstance(context, Context) for context in self.contexts)
                or len({context.key for context in self.contexts}) != len(self.contexts)):
            raise ValueError("contexts must have distinct keys, with at most five contexts")
        if not self.smoke:
            frozen = dict(max_init_epochs=45, init_epoch_stages=(15, 30, 45),
                          calibration_replicas=2, allocation_replicas=2, refinement_replicas=2,
                          finalist_extra_replicas=2,
                          reference_capacity=256, reference_replay=16,
                          capacities=(256,), replay_counts=(4, 16), ridge_grid=(1., 10., 100.),
                          strength_ratios=(.1, 1., 10.), retention_margin_degrees=1.,
                          matching_tolerance=.2, envelope_factor=2.5,
                          diagnostic_base_size=512, diagnostic_novel_size=256)
            if any(getattr(self, name) != value for name, value in frozen.items()):
                raise ValueError("changing the production Phase 9 scope requires a new protocol")
            if self.confirmation_replicas not in (8, 10, 12):
                raise ValueError("production confirmation requires eight, ten, or twelve replicas")
            families = [context.family for context in self.contexts]
            if families.count("gamma") != 2 or families.count("blur") != 2:
                raise ValueError("production requires two gamma and two blur contexts")

    @classmethod
    def from_mapping(cls, values: dict) -> "Config":
        values = dict(values)
        for name in ("capacities", "replay_counts", "ridge_grid", "strength_ratios",
                     "smaller_budgets", "init_epoch_stages"):
            if name in values:
                values[name] = tuple(values[name])
        if "contexts" in values:
            values["contexts"] = tuple(Context.from_mapping(value) for value in values["contexts"])
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def read(cls, path: Path | str) -> "Config":
        return cls.from_mapping(json.loads(Path(path).read_text()))

    @classmethod
    def smoke_config(cls, device="cpu") -> "Config":
        return cls(smoke=True, device=device, dtype="float64", init_size=40,
                   init_validation_size=20, reference_size=12, init_epochs=1,
                   max_init_epochs=1, init_epoch_stages=(1,), init_batch=20,
                   contexts=(default_contexts()[0], default_contexts()[2]),
                   diagnostic_base_size=8, diagnostic_novel_size=4,
                   diagnostic_epochs=1, max_diagnostic_epochs=2, diagnostic_batch=4,
                   eval_per_class=1, novel_count=3, screen_novel_count=1,
                   ramp_groups=2, novel_probability=.5, refinement_replicas=0,
                   finalist_extra_replicas=0, confirmation_replicas=2,
                   preferred_min_confirmation=2, reference_capacity=2, reference_replay=1,
                   capacities=(2,), replay_counts=(1,), ridge_grid=(1.,), strength_ratios=(.1,),
                   max_iter=2, max_eval=4, smaller_budgets=(1, 2), bootstrap_samples=50,
                   refinement_settings=0, confirmation_reserve_seconds=30.,
                   analysis_reserve_seconds=10., free_reserve_bytes=268435456)
