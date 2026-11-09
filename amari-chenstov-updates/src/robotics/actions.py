"""Plan19 legacy mixture, followed by a single explicit SI-unit calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .protocol import Control

ACTION_NAMES = (
    "drive-left", "drive-right", "drive-forward", "drive-backward",
    "look-left", "look-right", "look-up", "look-forward",
)
ACTION_VECTORS = (
    (0., 0., -1., 1.), (0., 0., 1., 1.), (0., 0., 0., 1.),
    (0., 0., 0., -1.), (1., 0., 0., 0.), (-1., 0., 0., 0.),
    (0., 1., 0., 0.), (0., 0., 0., 0.),
)
MAPPING_VERSION = "plan19-absolute-camera-si-v1"


def action_distribution_to_vector(probabilities: Iterable[float]) -> tuple[float, ...]:
    probabilities = tuple(float(p) for p in probabilities)
    total = sum(probabilities)
    if (len(probabilities) != 8 or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(total, 1., abs_tol=1e-6, rel_tol=0)):
        raise ValueError("expected eight finite nonnegative probabilities summing to one")
    # Float32 softmax roundoff is not a different action distribution. Restore
    # convex weights only after validating the small normalization error.
    probabilities = tuple(p / total for p in probabilities)
    return tuple(sum(p * vector[i] for p, vector in zip(probabilities, ACTION_VECTORS))
                 for i in range(4))


@dataclass(frozen=True)
class Calibration:
    """Unvalidated physical defaults, NOT measured Pi servo settings.

    Weighted normalized camera zero means the nominal viewing angle, not
    electrical servo zero. No turn-implies-drive legacy side effect survives.
    """

    pan_limit_rad: float = math.radians(60)
    tilt_neutral_rad: float = math.radians(-8)
    tilt_up_range_rad: float = math.radians(38)
    steering_limit_rad: float = math.radians(25)
    forward_limit_mps: float = .20
    reverse_limit_mps: float = .10
    version: str = MAPPING_VERSION
    physically_validated: bool = False

    def __post_init__(self) -> None:
        limits = (self.pan_limit_rad, self.tilt_up_range_rad, self.steering_limit_rad,
                  self.forward_limit_mps, self.reverse_limit_mps)
        if any(not math.isfinite(v) or v <= 0 for v in limits):
            raise ValueError("calibration limits must be finite and positive")
        if not math.isfinite(self.tilt_neutral_rad):
            raise ValueError("camera neutral must be finite")

    def map_distribution(self, probabilities: Iterable[float]) -> Control:
        pan, tilt, steer, drive = action_distribution_to_vector(probabilities)
        speed_scale = self.forward_limit_mps if drive >= 0 else self.reverse_limit_mps
        return Control(pan * self.pan_limit_rad,
                       self.tilt_neutral_rad + tilt * self.tilt_up_range_rad,
                       steer * self.steering_limit_rad, drive * speed_scale)

    def validate(self, controls: Control) -> None:
        for value, low, high in (
            (controls.pan_rad, -self.pan_limit_rad, self.pan_limit_rad),
            (controls.tilt_rad, self.tilt_neutral_rad,
             self.tilt_neutral_rad + self.tilt_up_range_rad),
            (controls.steering_rad, -self.steering_limit_rad, self.steering_limit_rad),
            (controls.drive_mps, -self.reverse_limit_mps, self.forward_limit_mps),
        ):
            if not math.isfinite(value) or not low - 1e-12 <= value <= high + 1e-12:
                raise ValueError("control outside calibrated bounds")

    def neutral(self, previous: Control | None = None) -> Control:
        return Control(previous.pan_rad if previous else 0.,
                       previous.tilt_rad if previous else self.tilt_neutral_rad, 0., 0.)
