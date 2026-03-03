from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def default_demo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return default_demo_root() / "data"


@dataclass(frozen=True)
class PiCarControlConfig:
    host: str = "127.0.0.1:5000"
    repo_root: Path = Path("/home/evan/Documents/picar-v-rl-env")
    x_resize: int | None = None
    y_resize: int | None = None


@dataclass(frozen=True)
class SpeechStreamConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    blocksize: int = 0
    amplitude_threshold: float = 0.015
    silence_seconds: float = 0.8
    min_speech_seconds: float = 0.25
    max_event_queue_size: int = 128
    callback_queue_size: int = 256
    poll_interval_seconds: float = 0.05


@dataclass(frozen=True)
class RewardConfig:
    prompt_id: str = "reward_prompt_1"
    generation_max_new_tokens: int = 32
    min_reward: float = 0.0
    max_reward: float = 10.0
    allow_downloads: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    train_every_steps: int = 16
    min_replay_size: int = 32
    batch_size: int = 32
    fit_iters: int = 1
    memorize_every_steps: int = 64
    memorize_n: int = 64
    memorize_random_idx: bool = True
    t_start: float = 0.0
    t_end: float = 1.0
    t_ramp_steps: int = 1_000


@dataclass(frozen=True)
class EnvConfig:
    demo_root: Path = field(default_factory=default_demo_root)
    data_dir: Path = field(default_factory=default_data_dir)
    experiment_name_prefix: str = "picar-rl-env"
    history_window: int = 12
    checkpoint_basename: str = "picar_policy"
    write_frame_blobs: bool = True
    picar: PiCarControlConfig = field(default_factory=PiCarControlConfig)
    speech: SpeechStreamConfig = field(default_factory=SpeechStreamConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
