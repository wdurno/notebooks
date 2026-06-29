from __future__ import annotations

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
ACTION_VECTOR_BOUNDS = {
    "pan": (-1.0, 1.0),
    "tilt": (0.0, 1.0),
    "turn": (-1.0, 1.0),
    "drive": (-1.0, 1.0),
}

_ACTION_TO_VECTOR = {
    # Turning actions include forward drive so the robot can change heading
    # while steering, instead of only rotating wheels in place.
    "drive-left": {"pan": 0.0, "tilt": 0.0, "turn": -1.0, "drive": 1.0},
    "drive-right": {"pan": 0.0, "tilt": 0.0, "turn": 1.0, "drive": 1.0},
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
    mixed = {
        key: (1.0 - t) * float(agentic_vector[key]) + t * float(value_vector[key])
        for key in ACTION_VECTOR_KEYS
    }
    return clamp_action_vector(mixed)


def action_vector_to_tensor(
    action_vector: dict[str, float],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(
        [float(action_vector[key]) for key in ACTION_VECTOR_KEYS],
        dtype=dtype,
        device=device,
    )


def tensor_to_action_vector(action_tensor: torch.Tensor) -> dict[str, float]:
    if action_tensor.ndim != 1 or action_tensor.shape[0] != len(ACTION_VECTOR_KEYS):
        raise ValueError(f"Expected action tensor of shape [{len(ACTION_VECTOR_KEYS)}]")
    values = action_tensor.detach().to(dtype=torch.float32)
    return clamp_action_vector(
        {
            key: float(values[idx].item())
            for idx, key in enumerate(ACTION_VECTOR_KEYS)
        }
    )


def clamp_action_vector(action_vector: dict[str, float]) -> dict[str, float]:
    clamped = {}
    for key in ACTION_VECTOR_KEYS:
        low, high = ACTION_VECTOR_BOUNDS[key]
        value = float(action_vector[key])
        clamped[key] = min(max(value, low), high)
    return clamped


def mix_action_tensors(agentic_tensor: torch.Tensor, actor_tensor: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
    if agentic_tensor.shape != actor_tensor.shape:
        raise ValueError("Agentic and actor action tensors must have the same shape")
    if agentic_tensor.shape[-1] != len(ACTION_VECTOR_KEYS):
        raise ValueError(f"Expected last action dimension to be {len(ACTION_VECTOR_KEYS)}")
    t_tensor = torch.as_tensor(t, dtype=agentic_tensor.dtype, device=agentic_tensor.device)
    if t_tensor.ndim == 0:
        t_tensor = t_tensor.reshape(1)
    while t_tensor.ndim < agentic_tensor.ndim:
        t_tensor = t_tensor.unsqueeze(-1)
    mixed = (1.0 - t_tensor) * agentic_tensor + t_tensor * actor_tensor
    return clamp_action_tensor(mixed)


def clamp_action_tensor(action_tensor: torch.Tensor) -> torch.Tensor:
    if action_tensor.shape[-1] != len(ACTION_VECTOR_KEYS):
        raise ValueError(f"Expected last action dimension to be {len(ACTION_VECTOR_KEYS)}")
    low = torch.tensor([ACTION_VECTOR_BOUNDS[key][0] for key in ACTION_VECTOR_KEYS], dtype=action_tensor.dtype, device=action_tensor.device)
    high = torch.tensor([ACTION_VECTOR_BOUNDS[key][1] for key in ACTION_VECTOR_KEYS], dtype=action_tensor.dtype, device=action_tensor.device)
    return torch.minimum(torch.maximum(action_tensor, low), high)


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
