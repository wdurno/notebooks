from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ExperimentPhase = Literal["init", "tune", "retask"]
ExperimentUpdateMode = Literal["auto", "batch", "online"]


@dataclass(frozen=True)
class ExperimentRunConfig:
    phase: ExperimentPhase
    update_mode: ExperimentUpdateMode = "auto"
    picar_host: str | None = None
    deterministic_coding: bool = False
    history_window: int = 12
    all_images: bool = False
    prompt_token_window: int = 512
    init_t: float = 0.0
    fixed_t: float | None = None
    t_step: float = 0.001
    t_log_every: int = 1
    load_snapshot: Path | None = None
    load_latest_from_model_root: bool = False
    reward_prompt: str = "find the red ball"
    snapshot_keep: int = 3
    data_root: Path | None = None
    model_root: Path | None = None
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 0.1
    fit_iters: int | None = None
    train_every_steps: int = 16
    min_replay_size: int = 32
    memorize_every_steps: int = 64
    memorize_n: int = 64
    memorize_random_idx: bool = False
    run_uuid: str | None = None


@dataclass(frozen=True)
class ExperimentRunSummary:
    run_uuid: str
    phase: ExperimentPhase
    steps_completed: int
    data_run_dir: Path
    model_run_dir: Path
