"""Rolling episode context shared by experiment phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from picar_kl.context.budget import PromptBudgetResult, apply_prompt_token_window
from picar_kl.context.config import ContextConfig
from picar_kl.context.messages import (
    build_assistant_action_message,
    build_control_system_message,
    build_goal_system_message,
    build_operator_status_message,
    copy_message,
    ensure_latest_user_image_placeholder,
    is_goal_system_message,
)
from picar_kl.context.protocols import PromptTokenizer


@dataclass(frozen=True)
class ContextRender:
    messages: list[dict[str, Any]]
    budget: PromptBudgetResult

    @property
    def metadata(self) -> dict[str, Any]:
        return self.budget.metadata()


@dataclass
class EpisodeContext:
    """Mutable rolling message state for one robot episode."""

    config: ContextConfig = field(default_factory=ContextConfig)
    history: list[dict[str, Any]] = field(default_factory=list)
    last_reward: float = 0.0
    _last_reward_result: Any | None = None

    def add_observation(
        self,
        *,
        user_texts: list[str],
        reward_result: Any,
        step_index: int,
        tokenizer: PromptTokenizer | None = None,
    ) -> ContextRender:
        self._last_reward_result = reward_result
        user_message = build_operator_status_message(
            user_texts=user_texts,
            step_index=step_index,
            last_reward=self.last_reward,
            current_reward=float(reward_result.clipped_reward),
            reward_prompt_id=str(reward_result.prompt_id),
        )
        self.history = self._with_goal_message([*self.history, user_message], reward_result=reward_result)
        return self.render(tokenizer=tokenizer)

    def add_assistant_action(
        self,
        *,
        action_name: str,
        generated_text: str = "",
        reward_result: Any | None = None,
    ) -> None:
        resolved_reward = reward_result or self._last_reward_result
        if resolved_reward is None:
            raise ValueError("reward_result is required before assistant action history can be updated")
        assistant_message = build_assistant_action_message(action_name=action_name, generated_text=generated_text)
        self.history = self._with_goal_message([*self.history, assistant_message], reward_result=resolved_reward)
        self.last_reward = float(resolved_reward.clipped_reward)

    def render(self, *, tokenizer: PromptTokenizer | None = None) -> ContextRender:
        messages = [build_control_system_message(), *[copy_message(message) for message in self.history]]
        messages = ensure_latest_user_image_placeholder(messages)
        budget = apply_prompt_token_window(
            messages,
            tokenizer=tokenizer,
            token_window=self.config.prompt_token_window,
        )
        return ContextRender(messages=budget.messages, budget=budget)

    def _with_goal_message(self, history: list[dict[str, Any]], *, reward_result: Any) -> list[dict[str, Any]]:
        goal = build_goal_system_message(
            task_text=str(getattr(reward_result, "task_text", "") or ""),
            reward_prompt_id=str(reward_result.prompt_id),
        )
        non_goal = [copy_message(message) for message in history if not is_goal_system_message(message)]
        if self.config.history_window <= 1:
            return [goal]
        tail = non_goal[-(int(self.config.history_window) - 1) :]
        return [goal, *tail]
