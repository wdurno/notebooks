from __future__ import annotations

from typing import Iterable

import torch


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

_ACTION_TO_VECTOR = {
    "drive-left": {"pan": 0.0, "tilt": 0.0, "turn": -1.0, "drive": 0.0},
    "drive-right": {"pan": 0.0, "tilt": 0.0, "turn": 1.0, "drive": 0.0},
    "drive-forward": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
    "drive-backward": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": -1.0},
    "look-left": {"pan": 1.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
    "look-right": {"pan": -1.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
    "look-up": {"pan": 0.0, "tilt": 1.0, "turn": 0.0, "drive": 0.0},
    "look-forward": {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
}


def action_name_to_index(action_name: str) -> int:
    try:
        return ACTION_NAMES.index(action_name)
    except ValueError as exc:
        raise ValueError(f"Unknown action name: {action_name}") from exc


def action_index_to_name(action_index: int) -> str:
    try:
        return ACTION_NAMES[action_index]
    except IndexError as exc:
        raise ValueError(f"Unknown action index: {action_index}") from exc


def action_name_to_one_hot(
    action_name: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    action_index = action_name_to_index(action_name)
    one_hot = torch.zeros(len(ACTION_NAMES), dtype=dtype, device=device)
    one_hot[action_index] = 1.0
    return one_hot


def action_vector_for_name(action_name: str) -> dict[str, float]:
    return dict(_ACTION_TO_VECTOR[action_name])


def action_vector_for_index(action_index: int) -> dict[str, float]:
    return action_vector_for_name(action_index_to_name(action_index))


def one_hot_to_action_name(one_hot: torch.Tensor) -> str:
    if one_hot.ndim != 1 or one_hot.shape[0] != len(ACTION_NAMES):
        raise ValueError(f"Expected one-hot vector of shape [{len(ACTION_NAMES)}]")
    action_index = int(torch.argmax(one_hot).item())
    return action_index_to_name(action_index)


def one_hot_to_action_vector(one_hot: torch.Tensor) -> dict[str, float]:
    if one_hot.ndim != 1 or one_hot.shape[0] != len(ACTION_NAMES):
        raise ValueError(f"Expected one-hot vector of shape [{len(ACTION_NAMES)}]")
    values = one_hot.detach().to(dtype=torch.float32)
    out = {key: 0.0 for key in ACTION_VECTOR_KEYS}
    for idx, weight in enumerate(values):
        if float(weight) == 0.0:
            continue
        vector = action_vector_for_index(idx)
        for key in ACTION_VECTOR_KEYS:
            out[key] += float(weight) * float(vector[key])
    return out


def mix_action_vectors(
    agentic_vector: dict[str, float],
    value_vector: dict[str, float],
    t: float,
) -> dict[str, float]:
    if t < 0.0 or t > 1.0:
        raise ValueError("`t` must be in [0, 1]")
    return {
        key: (1.0 - t) * float(agentic_vector[key]) + t * float(value_vector[key])
        for key in ACTION_VECTOR_KEYS
    }


def normalized_action_name(text: str, *, default_action: str = "look-forward") -> str:
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
