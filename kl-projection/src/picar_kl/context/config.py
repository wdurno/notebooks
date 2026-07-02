"""Shared context configuration for experiment phases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextConfig:
    """Bounds for rolling VLM context."""

    history_window: int = 180
    prompt_token_window: int | None = 8000

    def __post_init__(self) -> None:
        if int(self.history_window) < 1:
            raise ValueError(f"history_window must be >= 1, got {self.history_window}")
        if self.prompt_token_window is not None and int(self.prompt_token_window) < 0:
            raise ValueError(f"prompt_token_window must be >= 0, got {self.prompt_token_window}")
