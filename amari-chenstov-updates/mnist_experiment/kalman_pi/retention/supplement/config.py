"""Three approved arms, with new draws from the primary study's data roles."""

from __future__ import annotations

import dataclasses

from mnist_experiment.kalman_pi.config import Config as ObserverConfig, PROTOCOL as OBSERVER_PROTOCOL
from mnist_experiment.kalman_pi.retention.config import Config as PrimaryConfig
from mnist_experiment.diagonal_lfu.config import content_hash
from src.seeding import derive_component_seed


PROTOCOL = "plan17-zero-ridge-confirmation-v1"
ARMS = {"fixed_ridge": {"condition": "fixed", "ridge": .1},
        "fixed_zero": {"condition": "fixed", "ridge": 0.},
        "kalman_zero": {"condition": "kalman_eta0.0001", "ridge": 0.}}


@dataclasses.dataclass(frozen=True)
class Config(ObserverConfig):
    protocol: str = PROTOCOL
    seed: int = 171103
    partition_seed: int = 171102
    anchor_refresh: int = 0
    calibration_size: int = 2048
    ema_half_lives: tuple[float, ...] = (16.,)
    process_scales: tuple[float, ...] = (.0001,)
    control_cs: tuple[float, ...] = (.025,)
    main_replicas: int = 8
    stationary_replicas: int = 0

    def validate(self):
        shared = {f.name: getattr(self, f.name) for f in dataclasses.fields(ObserverConfig)}
        shared["protocol"] = OBSERVER_PROTOCOL
        ObserverConfig(**shared)
        if self.protocol != PROTOCOL:
            raise ValueError("unsupported zero-ridge protocol")
        if type(self.partition_seed) is not int or not 0 <= self.partition_seed < 2**63:
            raise ValueError("partition seed must be an integer in [0,2**63)")
        if (self.anchor_refresh != 0 or self.control_cs != (.025,) or self.ridge_ratio != .1
                or self.ema_half_lives != (16.,) or self.process_scales != (.0001,)):
            raise ValueError("Phase 5E freezes its reference and observer settings")
        if self.stationary_replicas != 0 or self.main_replicas < 1:
            raise ValueError("Phase 5E has only changing-schedule confirmation replicas")
        if not self.smoke:
            if self.seed != 171103 or self.partition_seed != 171102 or self.main_replicas != 8:
                raise ValueError("the supplemental cohort and seed namespaces are frozen")
            primary = PrimaryConfig().mapping()
            exempt = {"protocol", "seed", "main_replicas", "stationary_replicas", "device", "threads",
                      "hold_steps", "ramp_steps"}
            for field in dataclasses.fields(ObserverConfig):
                if field.name not in exempt and getattr(self, field.name) != primary[field.name]:
                    raise ValueError(f"Phase 5E preserves the primary learner setting {field.name}")
            # The asset builder resolves the predrawn factor with dataclasses.replace.
            if (self.hold_steps, self.ramp_steps) not in ((352, 128), (440, 160)):
                raise ValueError("production horizons must be the approved base or resolved 1.25 factor")

    @classmethod
    def from_mapping(cls, values):
        cfg = super().from_mapping(values)
        if not cfg.smoke and (cfg.hold_steps, cfg.ramp_steps) != (352, 128):
            raise ValueError("input configuration must use the base horizon; assets resolve the factor")
        return cfg

    def seed_for(self, cohort, replica, component):
        if cohort == "study-partition":
            # Repartitioning would leak primary development holdouts into confirmation.
            return derive_component_seed(self.partition_seed,
                f"plan17-retention-calibration-v1:{cohort}:{replica}:{component}")
        return super().seed_for(cohort, replica, component)

    @property
    def scientific_hash(self):
        values = self.mapping()
        for name in ("device", "threads"):
            values.pop(name)
        return content_hash(values)
