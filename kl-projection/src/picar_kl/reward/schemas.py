"""Reward prompt and result schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RewardPromptSpec:
    prompt_id: str
    prompt_text: str
    task_text: str
    min_reward: float = 0.0
    max_reward: float = 10.0


@dataclass(frozen=True)
class RewardResult:
    prompt_id: str
    raw_text: str
    reward: float
    clipped_reward: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_text(self) -> str:
        return str(self.metadata.get("task_text") or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "raw_text": self.raw_text,
            "reward": self.reward,
            "clipped_reward": self.clipped_reward,
            "metadata": dict(self.metadata),
        }
