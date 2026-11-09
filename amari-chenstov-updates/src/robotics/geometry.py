"""Privileged geometric labels. Never included in a learner observation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TerminalGeometry:
    clearance_m: float
    heading_error_rad: float
    speed_mps: float
    contact: bool


def target_geometry(position_xy: tuple[float, float], yaw_rad: float,
                    speed_mps: float, target_xy: tuple[float, float],
                    contact: bool, *, half_length_m: float = .16,
                    target_radius_m: float = .10) -> TerminalGeometry:
    """Distance from front-bumper CENTER to target's circular surface.

    Heading is chassis +x toward target from the chassis center. Contacts
    include wheels/chassis against obstacle, target or arena, never the floor.
    """
    x, y = position_xy
    tx, ty = target_xy
    bumper = (x + half_length_m * math.cos(yaw_rad),
              y + half_length_m * math.sin(yaw_rad))
    clearance = math.hypot(tx - bumper[0], ty - bumper[1]) - target_radius_m
    heading = math.atan2(ty - y, tx - x) - yaw_rad
    heading = math.atan2(math.sin(heading), math.cos(heading))
    return TerminalGeometry(clearance, heading, speed_mps, contact)


def in_success_region(geometry: TerminalGeometry) -> bool:
    return (not geometry.contact and .25 - 1e-10 <= geometry.clearance_m <= .45 + 1e-10
            and abs(geometry.heading_error_rad) <= math.radians(15) + 1e-10
            and geometry.speed_mps <= .02 + 1e-10)


def terminal_event(geometry: TerminalGeometry, held_s: float,
                   elapsed_s: float, deadline_s: float) -> str | None:
    if geometry.contact:
        return "collision"
    if in_success_region(geometry) and held_s >= 1. - 1e-10:
        return "success"
    if elapsed_s >= deadline_s - 1e-10:
        return "deadline"
    return None
