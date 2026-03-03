from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueuedSpeechEvent:
    text: str
    received_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardPromptSpec:
    prompt_id: str
    prompt_text: str
    min_reward: float = 0.0
    max_reward: float = 10.0


@dataclass(frozen=True)
class RewardResult:
    prompt_id: str
    raw_text: str
    reward: float
    clipped_reward: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingSummary:
    triggered: bool
    replay_size: int
    pi: float | None = None
    loss: float | None = None
    memorized: int | None = None


@dataclass(frozen=True)
class ExperimentPaths:
    run_dir: Path
    metadata_path: Path
    blobs_dir: Path
    logs_dir: Path
    metrics_dir: Path
    artifacts_dir: Path
