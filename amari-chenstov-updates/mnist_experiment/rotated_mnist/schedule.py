"""Resolved repeated-angle schedules for Plan 5."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping

from .config import RotationConfig


ROTATION_SCHEDULE_SCHEMA_VERSION = 1
PRINCIPAL_KNOTS_DEGREES = (0.0, 15.0, 30.0, 0.0, 15.0, 30.0)


def _hash(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class RotationSchedule:
    angles_degrees: tuple[float, ...]
    leg_ids: tuple[int, ...]
    directions_to_next: tuple[int, ...]
    knot_flags: tuple[bool, ...]
    cumulative_degrees: tuple[float, ...]
    knots_degrees: tuple[float, ...]
    transitions_per_arrow: int
    schema_version: int = ROTATION_SCHEDULE_SCHEMA_VERSION

    @property
    def num_points(self) -> int:
        return len(self.angles_degrees)

    @property
    def num_transitions(self) -> int:
        return self.num_points - 1

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        for name in (
            "angles_degrees",
            "leg_ids",
            "directions_to_next",
            "knot_flags",
            "cumulative_degrees",
            "knots_degrees",
        ):
            value[name] = list(value[name])
        return value

    @property
    def content_hash(self) -> str:
        return _hash(self.to_mapping())

    def validate(self) -> None:
        if self.schema_version != ROTATION_SCHEDULE_SCHEMA_VERSION:
            raise ValueError("unsupported rotation schedule schema")
        expected_points = (
            (len(self.knots_degrees) - 1) * self.transitions_per_arrow + 1
        )
        if self.num_points != expected_points:
            raise ValueError("rotation schedule has an inconsistent point count")
        fields = (
            self.leg_ids,
            self.directions_to_next,
            self.knot_flags,
            self.cumulative_degrees,
        )
        if any(len(field) != self.num_points for field in fields):
            raise ValueError("rotation schedule arrays have inconsistent lengths")
        if any(not math.isfinite(value) for value in self.angles_degrees):
            raise ValueError("rotation schedule contains non-finite angles")
        if self.directions_to_next[-1] != 0:
            raise ValueError("final schedule point must have zero next direction")
        for step in range(self.num_transitions):
            difference = self.angles_degrees[step + 1] - self.angles_degrees[step]
            expected_direction = 1 if difference > 0.0 else -1 if difference < 0.0 else 0
            if self.directions_to_next[step] != expected_direction:
                raise ValueError("rotation direction metadata is inconsistent")
            expected_distance = self.cumulative_degrees[step] + abs(difference)
            if not math.isclose(
                self.cumulative_degrees[step + 1],
                expected_distance,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("rotation cumulative distance is inconsistent")
        knot_indices = {
            index * self.transitions_per_arrow
            for index in range(len(self.knots_degrees))
        }
        if self.knot_flags != tuple(
            step in knot_indices for step in range(self.num_points)
        ):
            raise ValueError("rotation knot metadata is inconsistent")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RotationSchedule":
        expected = {
            "angles_degrees",
            "leg_ids",
            "directions_to_next",
            "knot_flags",
            "cumulative_degrees",
            "knots_degrees",
            "transitions_per_arrow",
            "schema_version",
        }
        if set(value) != expected:
            raise ValueError("rotation schedule mapping has invalid fields")
        converted = dict(value)
        for name in (
            "angles_degrees",
            "leg_ids",
            "directions_to_next",
            "knot_flags",
            "cumulative_degrees",
            "knots_degrees",
        ):
            converted[name] = tuple(converted[name])
        schedule = cls(**converted)
        schedule.validate()
        return schedule


def resolve_rotation_schedule(config: RotationConfig) -> RotationSchedule:
    config.validate()
    angles: list[float] = []
    leg_ids: list[int] = []
    transitions = config.transitions_per_arrow
    for leg, (left, right) in enumerate(
        zip(config.knots_degrees[:-1], config.knots_degrees[1:], strict=True)
    ):
        count = transitions if leg + 1 < len(config.knots_degrees) - 1 else transitions + 1
        for offset in range(count):
            angles.append(left + offset * (right - left) / transitions)
            leg_ids.append(leg)

    directions = []
    cumulative = [0.0]
    for left, right in zip(angles[:-1], angles[1:], strict=True):
        difference = right - left
        directions.append(1 if difference > 0.0 else -1 if difference < 0.0 else 0)
        cumulative.append(cumulative[-1] + abs(difference))
    directions.append(0)
    knot_indices = {
        index * transitions for index in range(len(config.knots_degrees))
    }
    schedule = RotationSchedule(
        angles_degrees=tuple(angles),
        leg_ids=tuple(leg_ids),
        directions_to_next=tuple(directions),
        knot_flags=tuple(step in knot_indices for step in range(len(angles))),
        cumulative_degrees=tuple(cumulative),
        knots_degrees=config.knots_degrees,
        transitions_per_arrow=transitions,
    )
    schedule.validate()
    return schedule
