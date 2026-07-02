"""Reward scoring configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    prompt_id: str = "reward_prompt_1"
    generation_max_new_tokens: int = 16
    allow_downloads: bool = False
