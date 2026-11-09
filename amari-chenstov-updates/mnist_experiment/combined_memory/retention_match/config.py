"""Versioned calibration settings with the unchanged Phase 5 learning law."""

from dataclasses import dataclass
import math

from src.seeding import derive_component_seed
from ..broad_drift.config import Config as BroadConfig
from . import PROTOCOL, SCHEMA_VERSION


@dataclass(frozen=True)
class Config(BroadConfig):
    protocol: str = PROTOCOL
    schema_version: int = SCHEMA_VERSION
    seed: int = 18110806
    main_replicas: int = 4
    fd_strength: float = .5691316767656495
    development_replicas: int = 4
    refinements_per_arm: int = 2
    force_calibration_failure: bool = False

    def broad_mapping(self):
        values = self.mapping()
        for name in ("fd_strength", "development_replicas", "refinements_per_arm",
                     "force_calibration_failure"):
            values.pop(name)
        values.update(protocol=BroadConfig().protocol, schema_version=1)
        return values

    def validate(self):
        if self.protocol != PROTOCOL or self.schema_version != SCHEMA_VERSION:
            raise ValueError("incompatible retention-match protocol/schema")
        BroadConfig.from_mapping(self.broad_mapping())
        if (self.development_replicas != 4 or self.refinements_per_arm != 2
                or type(self.force_calibration_failure) is not bool
                or not math.isfinite(self.fd_strength) or self.fd_strength <= 0):
            raise ValueError("invalid bounded retention calibration settings")
        if not self.smoke and (self.fd_strength != .5691316767656495
                or self.main_replicas != 4 or self.stationary_replicas != 2
                or self.retention_tolerance != .1 or self.force_calibration_failure
                or self.max_iter != 100 or self.max_eval != 150
                or self.fd_dtype != "float32"):
            raise ValueError("production settings differ from the approved Phase 6 repair")

    def seed_for(self, cohort, replica, component):
        if type(replica) is not int or replica < 0:
            raise ValueError("replica index must be nonnegative")
        if (cohort, replica, component) == ("study-partition", 0, "test"):
            return BroadConfig().seed_for(cohort, replica, component)
        if cohort == "development" and replica < 2:
            return BroadConfig(seed=18110705).seed_for(cohort, replica, component)
        return derive_component_seed(self.seed, f"{PROTOCOL}:{cohort}:{replica}:{component}")

    @classmethod
    def smoke_config(cls, device="cpu"):
        values = BroadConfig.smoke_config(device).mapping()
        values.update(protocol=PROTOCOL, seed=18110806, main_replicas=2,
                      stationary_replicas=1, fd_strength=.1, retention_tolerance=10.)
        return cls.from_mapping(values)
