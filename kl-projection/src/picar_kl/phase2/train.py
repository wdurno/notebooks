"""Offline phase 2 KL-projection training."""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from picar_kl.phase2.cache import EncodingCacheConfig, VisualEncodingCache
from picar_kl.phase2.dataset import (
    Phase2Batch,
    Phase2WindowBatch,
    Phase2WindowExample,
    collate_phase2_windows,
    load_phase2_windows,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for phase 2 KL training") from exc
    return torch


@dataclass(frozen=True)
class Phase2TrainingConfig:
    data_roots: tuple[Path, ...] = (Path("artifacts/data/phase1"),)
    cache_root: Path = Path("artifacts/data/phase2/encodings")
    cache_model_name: str = "qwen2.5-vl-3b"
    cache_manifest_path: Path = Path("artifacts/manifests/tracked/models/vlm_models.json")
    cache_encoder_id: str = "qwen2.5-vl-get-image-features-pooler-output-v1"
    cache_config_hash: str = "output_dtype=float16"
    output_root: Path = Path("experiments/runs/phase2")
    checkpoint_root: Path = Path("artifacts/models/phase2")
    run_name: str | None = None
    context_steps: int = 4
    prediction_steps: int = 4
    window_stride: int | None = None
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 1e-3
    model_dim: int = 128
    conditioning_dim: int = 32
    conditioning_hidden_dim: int = 128
    token_type_dim: int = 8
    lstm_hidden_dim: int = 128
    lstm_layers: int = 1
    dropout: float = 0.0
    seed: int = 0
    max_windows: int | None = None
    save_checkpoint: bool = True
    allow_partial_cache: bool = False


@dataclass(frozen=True)
class Phase2TrainingResult:
    run_id: str
    run_dir: Path
    summary_path: Path
    checkpoint_path: Path | None
    losses: tuple[float, ...]
    final_loss: float
    valid_steps: int
    window_count: int

    @property
    def sequence_count(self) -> int:
        return self.window_count


def run_phase2_training(config: Phase2TrainingConfig) -> Phase2TrainingResult:
    config = validate_phase2_training_config(config)
    torch = _require_torch()
    from picar_kl.models.phase2 import Phase2KLModel, Phase2KLModelConfig
    from picar_kl.training.kl_projection import masked_action_kl_loss

    _seed_everything(config.seed, torch=torch)
    cache = VisualEncodingCache(
        EncodingCacheConfig(
            cache_root=config.cache_root,
            model_name=config.cache_model_name,
            manifest_path=config.cache_manifest_path,
            encoder_id=config.cache_encoder_id,
            config_hash=config.cache_config_hash,
        )
    )
    windows = load_phase2_windows(
        config.data_roots,
        cache=cache,
        context_steps=config.context_steps,
        prediction_steps=config.prediction_steps,
        stride=config.window_stride,
        allow_missing_cached_encodings=config.allow_partial_cache,
    )
    if config.max_windows is not None:
        windows = windows[: config.max_windows]
    if not windows:
        raise ValueError("phase 2 training found no cached prefix-target windows")

    first_shape = windows[0].target.visual_token_shape
    _, visual_dim = first_shape
    model = Phase2KLModel(
        Phase2KLModelConfig(
            visual_dim=visual_dim,
            model_dim=config.model_dim,
            conditioning_dim=config.conditioning_dim,
            conditioning_hidden_dim=config.conditioning_hidden_dim,
            token_type_dim=config.token_type_dim,
            lstm_hidden_dim=config.lstm_hidden_dim,
            lstm_layers=config.lstm_layers,
            dropout=config.dropout,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    valid_steps = 0
    for _ in range(config.epochs):
        for batch_windows in _iter_batches(windows, config.batch_size):
            batch = collate_phase2_windows(batch_windows)
            tensors = phase2_window_batch_to_tensors(batch, torch=torch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(
                prefix_visual_tokens=tensors["prefix_visual_tokens"],
                prefix_actions=tensors["prefix_actions"],
                prefix_step_mask=tensors["prefix_step_mask"],
                target_visual_tokens=tensors["target_visual_tokens"],
                target_previous_actions=tensors["target_previous_actions"],
                target_step_mask=tensors["target_step_mask"],
            )
            loss = masked_action_kl_loss(
                logits=output.logits,
                target_distributions=tensors["target_distributions"],
                mask=tensors["target_step_mask"],
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            valid_steps += int(tensors["target_step_mask"].sum().item())

    run_id = _run_id(config.run_name)
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = None
    if config.save_checkpoint:
        checkpoint_dir = config.checkpoint_root / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_path = checkpoint_dir / "policy.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model.config),
                "training_config": _jsonable(asdict(config)),
                "losses": losses,
            },
            checkpoint_path,
        )

    final_loss = float(losses[-1])
    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase2-kl-projection",
        "conditioning_mode": "trainable-prefix-head",
        "config": _jsonable(asdict(config)),
        "dataset": {
            "window_count": len(windows),
            "valid_target_steps": valid_steps,
            "visual_token_shape": list(first_shape),
            "context_steps": config.context_steps,
            "prediction_steps": config.prediction_steps,
            "allow_partial_cache": config.allow_partial_cache,
        },
        "training": {
            "losses": losses,
            "final_loss": final_loss,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        },
        "artifacts": {
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        },
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return Phase2TrainingResult(
        run_id=run_id,
        run_dir=run_dir,
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        losses=tuple(float(item) for item in losses),
        final_loss=final_loss,
        valid_steps=valid_steps,
        window_count=len(windows),
    )


def phase2_batch_to_tensors(batch: Phase2Batch, *, conditioning_dim: int, torch: Any | None = None) -> dict[str, Any]:
    torch = torch or _require_torch()
    visual_tokens = torch.as_tensor(batch.visual_tokens, dtype=torch.float32)
    previous_actions = torch.as_tensor(batch.previous_actions, dtype=torch.float32)
    target_distributions = torch.as_tensor(batch.target_distributions, dtype=torch.float32)
    step_mask = torch.as_tensor(batch.step_mask, dtype=torch.bool)
    conditioning = torch.zeros(
        (visual_tokens.shape[0], visual_tokens.shape[1], int(conditioning_dim)),
        dtype=torch.float32,
    )
    return {
        "visual_tokens": visual_tokens,
        "previous_actions": previous_actions,
        "target_distributions": target_distributions,
        "step_mask": step_mask,
        "conditioning": conditioning,
    }


def phase2_window_batch_to_tensors(batch: Phase2WindowBatch, *, torch: Any | None = None) -> dict[str, Any]:
    torch = torch or _require_torch()
    return {
        "prefix_visual_tokens": torch.as_tensor(batch.prefix.visual_tokens, dtype=torch.float32),
        "prefix_actions": torch.as_tensor(batch.prefix.target_distributions, dtype=torch.float32),
        "prefix_step_mask": torch.as_tensor(batch.prefix.step_mask, dtype=torch.bool),
        "target_visual_tokens": torch.as_tensor(batch.target.visual_tokens, dtype=torch.float32),
        "target_previous_actions": torch.as_tensor(batch.target.previous_actions, dtype=torch.float32),
        "target_distributions": torch.as_tensor(batch.target.target_distributions, dtype=torch.float32),
        "target_step_mask": torch.as_tensor(batch.target.step_mask, dtype=torch.bool),
    }


def validate_phase2_training_config(config: Phase2TrainingConfig) -> Phase2TrainingConfig:
    if not config.data_roots:
        raise ValueError("at least one data root is required")
    if int(config.context_steps) < 1:
        raise ValueError("context_steps must be >= 1")
    if int(config.prediction_steps) < 1:
        raise ValueError("prediction_steps must be >= 1")
    if config.window_stride is not None and int(config.window_stride) < 1:
        raise ValueError("window_stride must be >= 1 when provided")
    if int(config.batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if int(config.epochs) < 1:
        raise ValueError("epochs must be >= 1")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be > 0")
    for name in ("model_dim", "conditioning_dim", "conditioning_hidden_dim", "token_type_dim", "lstm_hidden_dim", "lstm_layers"):
        if int(getattr(config, name)) < 1:
            raise ValueError(f"{name} must be >= 1")
    if config.max_windows is not None and int(config.max_windows) < 1:
        raise ValueError("max_windows must be >= 1 when provided")
    return config


def _iter_batches(
    windows: Sequence[Phase2WindowExample],
    batch_size: int,
) -> Iterator[list[Phase2WindowExample]]:
    indices = list(range(len(windows)))
    random.shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        yield [windows[idx] for idx in indices[start : start + int(batch_size)]]


def _seed_everything(seed: int, *, torch: Any) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _run_id(run_name: str | None) -> str:
    if run_name:
        return run_name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
