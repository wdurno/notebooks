from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def default_demo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_dir() -> Path:
    return default_demo_root() / "model"


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "qwen2.5-vl-3b"
    model_dir: Path = field(default_factory=default_model_dir)
    model_subdir: str = "vlm/qwen2.5-vl-3b/base"
    finetune_subdir: str = "vlm/qwen2.5-vl-3b/finetunes"
    allow_downloads: bool = True
    deterministic_coding: bool = False
    all_images: bool = False
    prompt_token_window: int | None = 512
    learning_rate: float = 1e-2
    gamma: float = 0.99
    alpha: float = 1.0
    beta: float = 1.0
    token_pg_weight: float = 1.0
    value_loss_weight: float = 1.0
    value_head_bias: bool = True
    optimizer_name: str = "sgd"
    critic_hidden_size: Optional[int] = None
    target_update_tau: float = 0.05
    actor_anchor_weight: float = 1.0
    generation_max_new_tokens: int = 64
    generation_temperature: float = 0.3
    generation_top_p: float = 0.95
    generation_top_k: int = 0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "lm_head")
    default_t: float = 0.0
    hidden_size: Optional[int] = None
