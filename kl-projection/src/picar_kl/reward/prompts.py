"""Reward prompt registry."""

from __future__ import annotations

from dataclasses import dataclass

from picar_kl.reward.schemas import RewardPromptSpec


REWARD_PROMPT_1 = """
You are a reinforcement learning assistant in charge of deciding rewards.
You are receiving images from a mobile, robotic camera.
Primary visual task: finding a red ball.
If you see a red ball in the image, return a reward in [1, 10], otherwise zero.
If the ball is far away, return a 1.
If the ball is close enough to fill the screen while still being fully visible, return a 10.
If the ball is too close, entirely filling the screen, return a 1.
For intermediary distances, return interpolated values in (1, 10).
So, the robot only gets the highest score when the red ball is the right distance from the camera.
Return JSON with exactly one key: `reward`.
Example output when no red ball is visible:
{"reward": 0}
Example output when the red ball is centered and at a good distance:
{"reward": 9}
""".strip()


@dataclass(frozen=True)
class RewardPromptRegistry:
    prompts: dict[str, RewardPromptSpec]

    @classmethod
    def default(cls) -> "RewardPromptRegistry":
        spec = RewardPromptSpec(
            prompt_id="reward_prompt_1",
            prompt_text=REWARD_PROMPT_1,
            task_text="find the red ball",
            min_reward=0.0,
            max_reward=10.0,
        )
        return cls(prompts={spec.prompt_id: spec})

    def get(self, prompt_id: str) -> RewardPromptSpec:
        try:
            return self.prompts[prompt_id]
        except KeyError as exc:
            raise KeyError(f"Unknown reward prompt: {prompt_id}") from exc
