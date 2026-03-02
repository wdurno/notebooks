from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


ActionVector = dict[str, float]


@dataclass(frozen=True)
class ModelObservation:
    """Single-step observation passed from the environment into the model."""

    image_rgb: Any
    messages: list[dict[str, Any]]
    t: float
    last_reward: float = 0.0
    done: bool = False
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelActionOutput:
    """Structured action returned by the model for one observation."""

    agentic_action_name: str
    agentic_action_one_hot: torch.Tensor
    value_logits: torch.Tensor
    value_action_index: int
    value_action_one_hot: torch.Tensor
    agentic_action_vector: ActionVector
    value_action_vector: ActionVector
    mixed_action_vector: ActionVector
    generated_text: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    """One replay-buffer transition for SSR-backed Q-learning."""

    observation: ModelObservation
    action_index: int
    reward: float
    next_observation: ModelObservation
    done: bool
    target_text: Optional[str] = None
    target_action_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionBatch:
    """Collated batch sampled from the model replay buffer."""

    observations: list[ModelObservation]
    action_index: torch.Tensor
    reward: torch.Tensor
    next_observations: list[ModelObservation]
    done: torch.Tensor
    target_text: list[Optional[str]]
    target_action_name: list[Optional[str]]
    metadata: list[dict[str, Any]]
