"""Frozen/base VLM reward scoring."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from picar_kl.reward.config import RewardConfig
from picar_kl.reward.prompts import RewardPromptRegistry
from picar_kl.reward.schemas import RewardResult


def parse_reward_text(text: str, *, min_reward: float = 0.0, max_reward: float = 10.0) -> tuple[float, float]:
    candidate = (text or "").strip()
    if not candidate:
        raise ValueError("Reward output was empty")
    value = _extract_reward_value(candidate)
    clipped = min(max(value, min_reward), max_reward)
    return value, clipped


@dataclass
class ConstantRewardScorer:
    """Small explicit scorer for tests and smoke runs without a VLM."""

    reward: float = 0.0
    prompt_id: str = "reward_prompt_1"
    task_text: str = "find the red ball"

    def score(self, image_rgb: Any) -> RewardResult:
        del image_rgb
        value = float(self.reward)
        return RewardResult(
            prompt_id=self.prompt_id,
            raw_text=json.dumps({"reward": value}),
            reward=value,
            clipped_reward=value,
            metadata={"task_text": self.task_text},
        )


class FrozenVLMRewardScorer:
    """Score images with the base VLM and a selected reward prompt."""

    def __init__(
        self,
        config: RewardConfig | None = None,
        *,
        registry: RewardPromptRegistry | None = None,
        runtime: Any | None = None,
        runtime_config: Any | None = None,
    ):
        self.config = config or RewardConfig()
        self.registry = registry or RewardPromptRegistry.default()
        self._runtime = runtime
        self._runtime_config = runtime_config

    def score(self, image_rgb: Any) -> RewardResult:
        prompt_spec = self.registry.get(self.config.prompt_id)
        runtime = self._load_runtime()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt_spec.prompt_text}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Score the image and return JSON."},
                ],
            },
        ]
        raw_text, metadata = runtime.generate_completion(
            image_rgb=image_rgb,
            messages=messages,
            max_new_tokens=int(self.config.generation_max_new_tokens),
            do_sample=False,
            use_base_model=True,
        )
        reward, clipped_reward = parse_reward_text(
            raw_text,
            min_reward=prompt_spec.min_reward,
            max_reward=prompt_spec.max_reward,
        )
        return RewardResult(
            prompt_id=prompt_spec.prompt_id,
            raw_text=raw_text,
            reward=reward,
            clipped_reward=clipped_reward,
            metadata={
                "prompt_text": prompt_spec.prompt_text,
                "task_text": prompt_spec.task_text,
                "runtime": dict(metadata),
            },
        )

    def _load_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        from picar_kl.vlm.runtime import QwenRuntime, QwenRuntimeConfig

        config = self._runtime_config or QwenRuntimeConfig(allow_downloads=bool(self.config.allow_downloads))
        self._runtime = QwenRuntime.from_config(config)
        return self._runtime


def _extract_reward_value(text: str) -> float:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and "reward" in payload:
        return float(payload["reward"])

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "reward" in payload:
            return float(payload["reward"])

    numeric_match = re.search(r"-?\d+(?:\.\d+)?", text)
    if numeric_match:
        return float(numeric_match.group(0))
    raise ValueError(f"Could not parse a numeric reward from: {text!r}")
