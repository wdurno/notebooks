"""Protocols used by shared context code."""

from __future__ import annotations

from typing import Any, Protocol


class PromptTokenizer(Protocol):
    """Minimal prompt-token interface implemented by VLM runtimes."""

    def count_prompt_tokens(self, messages: list[dict[str, Any]]) -> int | None:
        ...
