"""Phase 8.1 settings; the Phase 8 likelihood and numerical engine are reused."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

from src.seeding import derive_component_seed
from mnist_experiment.fisher_memory_regression import PROTOCOL as ENGINE_PROTOCOL
from mnist_experiment.fisher_memory_regression.config import Config as EngineConfig

from . import PROTOCOL


@dataclass(frozen=True)
class Config(EngineConfig):
    protocol: str = PROTOCOL
    seed: int = 2026102581
    calibration_replicas: int = 4
    refinement_replicas: int = 2
    reference_capacity: int = 256
    reference_replay: int = 16
    capacities: tuple[int, ...] = (256,)
    replay_counts: tuple[int, ...] = (4, 16)
    ridge_grid: tuple[float, ...] = (1., 3., 10., 30.)
    strength_ratios: tuple[float, ...] = (.01, .1, 1.)
    retention_margin_degrees: float = 1.3342351463
    matching_tolerance: float = .20
    historical_strength: float = 263.5643254076337
    learning_time_limit: float = .0585204394
    memory_ceiling_bytes: int = 9942912204
    refinement_settings: int = 20
    confirmation_reserve_seconds: float = 7200.
    analysis_reserve_seconds: float = 900.

    def seed_for(self, family, replica, component):
        return derive_component_seed(self.seed, f"{PROTOCOL}:{family}:{replica}:{component}")

    def validate(self):
        if self.protocol != PROTOCOL:
            raise ValueError("unsupported retention protocol")
        # Validate inherited engine settings without claiming the old protocol.
        values = {field.name: getattr(self, field.name) for field in fields(EngineConfig)}
        values["protocol"] = ENGINE_PROTOCOL
        EngineConfig(**values).validate()
        for name in ("matching_tolerance", "historical_strength", "learning_time_limit"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        if self.matching_tolerance >= self.retention_margin_degrees:
            raise ValueError("matching tolerance must be smaller than the ceiling")
        if type(self.refinement_settings) is not int or not 0 <= self.refinement_settings <= 20:
            raise ValueError("at most 20 additional settings")
        if type(self.memory_ceiling_bytes) is not int or self.memory_ceiling_bytes <= 0:
            raise ValueError("invalid memory ceiling")
        if not self.smoke and (self.calibration_replicas != 4 or self.refinement_replicas != 2
                               or self.capacities != (256,) or self.replay_counts != (4, 16)):
            raise ValueError("changing the production search scope requires a new protocol")

    @classmethod
    def smoke_config(cls, device="cpu"):
        return cls(smoke=True, device=device, dtype="float64", init_size=40,
            init_validation_size=20, reference_size=12, init_epochs=1, init_batch=20,
            eval_per_class=1, novel_count=3, screen_novel_count=1,
            ramp_groups=2, novel_probability=.5, calibration_replicas=2,
            refinement_replicas=0, confirmation_replicas=2, preferred_min_confirmation=2,
            reference_capacity=2, reference_replay=1, capacities=(2,), replay_counts=(1,),
            ridge_grid=(1.,), strength_ratios=(.01,), max_iter=2, max_eval=4,
            smaller_budgets=(1,2), bootstrap_samples=50, refinement_settings=0,
            confirmation_reserve_seconds=30., analysis_reserve_seconds=10.,
            free_reserve_bytes=268435456, learning_time_limit=100.)
