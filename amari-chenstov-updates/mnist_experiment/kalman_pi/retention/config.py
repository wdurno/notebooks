"""Bounded Phase 5 settings, separate from the completed observer study."""

from __future__ import annotations

import dataclasses
import math

from mnist_experiment.kalman_pi.config import Config as ObserverConfig, PROTOCOL as OBSERVER_PROTOCOL
from mnist_experiment.diagonal_lfu.config import content_hash


PROTOCOL = "plan17-retention-calibration-v1"
CONDITIONS = ("fixed", "covariance", "ema_h16", "kalman_eta0.0001")
RIDGES = (0., .01, .03, .1, .3, 1.)


@dataclasses.dataclass(frozen=True)
class Config(ObserverConfig):
    protocol: str = PROTOCOL
    seed: int = 171102
    anchor_refresh: int = 0
    calibration_size: int = 2048
    ema_half_lives: tuple[float, ...] = (16.,)
    process_scales: tuple[float, ...] = (.0001,)
    control_cs: tuple[float, ...] = (.025,)
    ridges: tuple[float, ...] = RIDGES
    tolerance_pp: float = .5
    development_main: int = 4
    development_stationary: int = 2
    max_refinements: int = 2

    def validate(self):
        values = {f.name: getattr(self, f.name) for f in dataclasses.fields(ObserverConfig)}
        values["protocol"] = OBSERVER_PROTOCOL
        ObserverConfig(**values)
        if self.protocol != PROTOCOL:
            raise ValueError("unsupported retention protocol")
        if (not isinstance(self.ridges, tuple) or not self.ridges
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 3 for v in self.ridges)
                or tuple(sorted(set(self.ridges))) != self.ridges):
            raise ValueError("ridges must be sorted distinct finite values in [0,3]")
        if self.tolerance_pp != .5:
            raise ValueError("the development retention band is frozen at +/-0.5 pp")
        if self.control_cs != (.025,) or self.ridge_ratio != .1:
            raise ValueError("the reference is fixed pi=.025, ridge=.1")
        if self.anchor_refresh != 0 or self.ema_half_lives != (16.,) or self.process_scales != (.0001,):
            raise ValueError("Phase 5 retains the selected fixed-anchor observers")
        for name in ("development_main", "development_stationary"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_refinements) is not int or not 0 <= self.max_refinements <= 2:
            raise ValueError("at most two refinement values per policy/schedule")
        if not self.smoke:
            if self.ridges != RIDGES or (self.development_main, self.development_stationary) != (4, 2):
                raise ValueError("production development grid/cohort is fixed by Phase 5")
            if self.schedule_factors != (1., 1.25) or self.calibration_size != 2048:
                raise ValueError("production uses both horizon factors and full initialization noise calibration")
            if any(n % 2 for n in (self.main_replicas, self.stationary_replicas)):
                raise ValueError("fresh cohort sizes must balance both horizon factors")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values)
        if "ridges" in values:
            if not isinstance(values["ridges"], (list, tuple)):
                raise ValueError("ridges must be a sequence")
            values["ridges"] = tuple(values["ridges"])
        return super().from_mapping(values)

    @property
    def scientific_hash(self):
        # Targets and selection rules are part of the follow-up identity.
        values = self.mapping()
        for name in ("device", "threads"):
            values.pop(name)
        return content_hash(values)
