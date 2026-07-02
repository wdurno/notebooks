"""Shared bounded VLM context machinery."""

from .config import ContextConfig
from .state import ContextRender, EpisodeContext

__all__ = ["ContextConfig", "ContextRender", "EpisodeContext"]
