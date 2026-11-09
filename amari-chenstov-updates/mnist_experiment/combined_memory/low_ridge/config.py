"""Low-ridge follow-up settings, retaining the frozen broad-drift learner."""

from dataclasses import dataclass
import math

from src.seeding import derive_component_seed
from ..broad_drift.config import Config as BroadConfig
from ..retention_match.config import Config as HistoryConfig
from . import PROTOCOL, SCHEMA_VERSION


@dataclass(frozen=True)
class Config(HistoryConfig):
    protocol: str = PROTOCOL
    schema_version: int = SCHEMA_VERSION
    seed: int = 18110907
    main_replicas: int = 8
    refinements_per_arm: int = 5

    def validate(self):
        if self.protocol != PROTOCOL or self.schema_version != SCHEMA_VERSION:
            raise ValueError("incompatible low-ridge protocol/schema")
        inherited = self.broad_mapping()
        # Only the cohort allocation changes; retain every inherited learner check.
        if not self.smoke:
            inherited["main_replicas"] = 4
        BroadConfig.from_mapping(inherited)
        if (self.development_replicas != 4 or self.refinements_per_arm != 5
                or type(self.force_calibration_failure) is not bool
                or not math.isfinite(self.fd_strength) or self.fd_strength <= 0):
            raise ValueError("invalid low-ridge calibration settings")
        if not self.smoke and (self.fd_strength != .5691316767656495
                or self.main_replicas != 8 or self.stationary_replicas != 2
                or self.retention_tolerance != .1 or self.force_calibration_failure
                or self.max_iter != 100 or self.max_eval != 150 or self.fd_dtype != "float32"):
            raise ValueError("production settings differ from approved Phase 7")

    def seed_for(self, cohort, replica, component):
        if type(replica) is not int or replica < 0:
            raise ValueError("replica must be nonnegative integer")
        if cohort == "development" or (cohort, replica, component) == ("study-partition", 0, "test"):
            return HistoryConfig().seed_for(cohort, replica, component)
        if not isinstance(cohort, str) or not cohort or not isinstance(component, str) or not component:
            raise ValueError("nonempty seed component and cohort required")
        return derive_component_seed(self.seed, f"{PROTOCOL}:{cohort}:{replica}:{component}")

    @classmethod
    def smoke_config(cls, device="cpu"):
        values = HistoryConfig.smoke_config(device).mapping()
        values.update(protocol=PROTOCOL, seed=18110907, main_replicas=4,
                      stationary_replicas=1, refinements_per_arm=5)
        return cls.from_mapping(values)
