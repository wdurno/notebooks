"""Canonical PiCar action names, distributions, and control vectors."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Iterable, Mapping


ACTION_NAMES = (
    "drive-left",
    "drive-right",
    "drive-forward",
    "drive-backward",
    "look-left",
    "look-right",
    "look-up",
    "look-forward",
)

ACTION_VECTOR_KEYS = ("pan", "tilt", "turn", "drive")
ACTION_VECTOR_BOUNDS = MappingProxyType(
    {
        "pan": (-1.0, 1.0),
        "tilt": (0.0, 1.0),
        "turn": (-1.0, 1.0),
        "drive": (-1.0, 1.0),
    }
)

ACTION_VECTORS = MappingProxyType(
    {
        "drive-left": {"pan": 0.0, "tilt": 0.0, "turn": -1.0, "drive": 1.0},
        "drive-right": {"pan": 0.0, "tilt": 0.0, "turn": 1.0, "drive": 1.0},
        "drive-forward": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
        "drive-backward": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": -1.0},
        "look-left": {"pan": 1.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
        "look-right": {"pan": -1.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
        "look-up": {"pan": 0.0, "tilt": 1.0, "turn": 0.0, "drive": 0.0},
        "look-forward": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
    }
)

ActionDistribution = tuple[float, ...]
ActionVector = dict[str, float]


def action_count() -> int:
    return len(ACTION_NAMES)


def action_name_to_index(action_name: str) -> int:
    try:
        return ACTION_NAMES.index(action_name)
    except ValueError as exc:
        raise ValueError(f"Unknown action name: {action_name}") from exc


def action_index_to_name(action_index: int) -> str:
    try:
        return ACTION_NAMES[int(action_index)]
    except IndexError as exc:
        raise ValueError(f"Unknown action index: {action_index}") from exc


def normalize_action_name(text: str, *, default_action: str = "look-forward") -> str:
    candidate = (text or "").strip().lower()
    if not candidate:
        return default_action
    if candidate in ACTION_NAMES:
        return candidate
    candidate = candidate.replace("_", "-")
    if candidate in ACTION_NAMES:
        return candidate
    for action_name in ACTION_NAMES:
        if action_name in candidate:
            return action_name
    return default_action


def action_name_to_distribution(action_name: str) -> ActionDistribution:
    action_index = action_name_to_index(action_name)
    return tuple(1.0 if idx == action_index else 0.0 for idx in range(action_count()))


def distribution_to_action_name(distribution: Iterable[float]) -> str:
    values = validate_action_distribution(distribution)
    action_index = max(range(len(values)), key=values.__getitem__)
    return action_index_to_name(action_index)


def validate_action_distribution(
    distribution: Iterable[float],
    *,
    tolerance: float = 1e-6,
) -> ActionDistribution:
    values = tuple(float(value) for value in distribution)
    if len(values) != action_count():
        raise ValueError(f"Expected {action_count()} action probabilities")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Action distribution contains non-finite values")
    if any(value < -tolerance for value in values):
        raise ValueError("Action distribution contains negative probabilities")
    probability_sum = sum(values)
    if abs(probability_sum - 1.0) > tolerance:
        raise ValueError("Action distribution must sum to 1")
    return tuple(0.0 if value < 0.0 and abs(value) <= tolerance else value for value in values)


def action_vector_for_name(action_name: str) -> ActionVector:
    action_name_to_index(action_name)
    return dict(ACTION_VECTORS[action_name])


def action_distribution_to_vector(distribution: Iterable[float]) -> ActionVector:
    values = validate_action_distribution(distribution)
    mixed = {key: 0.0 for key in ACTION_VECTOR_KEYS}
    for action_name, probability in zip(ACTION_NAMES, values):
        if probability == 0.0:
            continue
        vector = ACTION_VECTORS[action_name]
        for key in ACTION_VECTOR_KEYS:
            mixed[key] += probability * float(vector[key])
    return clamp_action_vector(mixed)


def validate_action_vector(action_vector: Mapping[str, float]) -> ActionVector:
    missing = [key for key in ACTION_VECTOR_KEYS if key not in action_vector]
    if missing:
        raise ValueError(f"Action vector missing keys: {', '.join(missing)}")
    out = {}
    for key in ACTION_VECTOR_KEYS:
        value = float(action_vector[key])
        if not math.isfinite(value):
            raise ValueError(f"Action vector value for `{key}` is not finite")
        low, high = ACTION_VECTOR_BOUNDS[key]
        if value < low or value > high:
            raise ValueError(f"Action vector value for `{key}` must be in [{low}, {high}]")
        out[key] = value
    return out


def clamp_action_vector(action_vector: Mapping[str, float]) -> ActionVector:
    missing = [key for key in ACTION_VECTOR_KEYS if key not in action_vector]
    if missing:
        raise ValueError(f"Action vector missing keys: {', '.join(missing)}")
    out = {}
    for key in ACTION_VECTOR_KEYS:
        value = float(action_vector[key])
        if not math.isfinite(value):
            raise ValueError(f"Action vector value for `{key}` is not finite")
        low, high = ACTION_VECTOR_BOUNDS[key]
        out[key] = min(max(value, low), high)
    return out
