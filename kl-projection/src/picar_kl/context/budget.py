"""Prompt token budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from picar_kl.context.messages import (
    copy_message,
    is_goal_system_message,
    message_with_single_text,
    text_from_message,
)
from picar_kl.context.protocols import PromptTokenizer


@dataclass(frozen=True)
class PromptBudgetResult:
    messages: list[dict[str, Any]]
    prompt_tokens: int | None
    token_window: int | None
    truncated: bool = False
    dropped_messages: int = 0
    clipped_latest: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prompt_token_window": self.token_window,
            "prompt_truncated": self.truncated,
            "prompt_dropped_messages": self.dropped_messages,
            "prompt_clipped_latest": self.clipped_latest,
        }


def apply_prompt_token_window(
    messages: list[dict[str, Any]],
    *,
    tokenizer: PromptTokenizer | None,
    token_window: int | None,
) -> PromptBudgetResult:
    copied = [copy_message(message) for message in messages]
    if token_window is None or int(token_window) <= 0 or tokenizer is None:
        return PromptBudgetResult(
            messages=copied,
            prompt_tokens=_count(tokenizer, copied),
            token_window=token_window,
        )

    window = int(token_window)
    full_tokens = _count(tokenizer, copied)
    if full_tokens is None or full_tokens <= window:
        return PromptBudgetResult(messages=copied, prompt_tokens=full_tokens, token_window=window)

    mandatory_indices = _mandatory_indices(copied)
    selected = set(mandatory_indices)
    mandatory_messages = [copied[idx] for idx in mandatory_indices]
    mandatory_tokens = _count(tokenizer, mandatory_messages)
    if mandatory_tokens is None:
        return PromptBudgetResult(messages=copied, prompt_tokens=full_tokens, token_window=window)
    if mandatory_tokens > window:
        return PromptBudgetResult(
            messages=mandatory_messages,
            prompt_tokens=mandatory_tokens,
            token_window=window,
            truncated=True,
            dropped_messages=max(0, len(copied) - len(mandatory_messages)),
        )

    for idx in range(len(copied) - 1, -1, -1):
        if idx in selected:
            continue
        trial_indices = sorted((*selected, idx))
        trial_messages = [copied[item_idx] for item_idx in trial_indices]
        trial_tokens = _count(tokenizer, trial_messages)
        if trial_tokens is None:
            return PromptBudgetResult(messages=copied, prompt_tokens=full_tokens, token_window=window)
        if trial_tokens <= window:
            selected.add(idx)

    truncated_messages = [copied[idx] for idx in range(len(copied)) if idx in selected]
    clipped_latest = False
    latest_idx = _latest_text_message_index(copied)
    if latest_idx is not None and latest_idx not in selected:
        clipped = _clip_latest_message(
            tokenizer=tokenizer,
            base_messages=truncated_messages,
            original_message=copied[latest_idx],
            token_window=window,
        )
        if clipped is not None:
            truncated_messages = [*truncated_messages, clipped]
            clipped_latest = True

    return PromptBudgetResult(
        messages=truncated_messages,
        prompt_tokens=_count(tokenizer, truncated_messages),
        token_window=window,
        truncated=True,
        dropped_messages=max(0, len(copied) - len(truncated_messages)),
        clipped_latest=clipped_latest,
    )


def _count(tokenizer: PromptTokenizer | None, messages: list[dict[str, Any]]) -> int | None:
    if tokenizer is None:
        return None
    return tokenizer.count_prompt_tokens(messages)


def _mandatory_indices(messages: list[dict[str, Any]]) -> list[int]:
    indices: list[int] = []
    if messages:
        indices.append(0)
    for idx, message in enumerate(messages):
        if is_goal_system_message(message) and idx not in indices:
            indices.append(idx)
    return sorted(indices)


def _latest_text_message_index(messages: list[dict[str, Any]]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if text_from_message(messages[idx]):
            return idx
    return None


def _clip_latest_message(
    *,
    tokenizer: PromptTokenizer,
    base_messages: list[dict[str, Any]],
    original_message: dict[str, Any],
    token_window: int,
) -> dict[str, Any] | None:
    full_text = text_from_message(original_message)
    if not full_text:
        return None
    marker = " [truncated]"
    low = 1
    high = len(full_text)
    best_message = None
    while low <= high:
        mid = (low + high) // 2
        clipped_text = f"{full_text[:mid].rstrip()}{marker}"
        candidate = message_with_single_text(original_message, clipped_text)
        candidate_tokens = _count(tokenizer, [*base_messages, candidate])
        if candidate_tokens is None:
            return None
        if candidate_tokens <= token_window:
            best_message = candidate
            low = mid + 1
        else:
            high = mid - 1
    if best_message is not None:
        return best_message
    tiny = message_with_single_text(original_message, marker.strip())
    tiny_tokens = _count(tokenizer, [*base_messages, tiny])
    if tiny_tokens is not None and tiny_tokens <= token_window:
        return tiny
    return None
