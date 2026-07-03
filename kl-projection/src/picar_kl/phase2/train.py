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
    load_phase2_windows_with_stats,
)
from picar_kl.phase2.metrics import Phase2Breakout, action_metrics_from_arrays, merge_metric_sums


FIT_MODE_WINDOW_SAMPLING = "window_sampling_fit"
FIT_MODE_FULL_SEQUENCE = "full_sequence_fit"
FIT_MODES = (FIT_MODE_WINDOW_SAMPLING, FIT_MODE_FULL_SEQUENCE)


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
    fit_mode: str = FIT_MODE_WINDOW_SAMPLING
    validation_fraction: float = 0.2
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
    device: str = "auto"


@dataclass(frozen=True)
class Phase2TrainingResult:
    run_id: str
    run_dir: Path
    artifact_dir: Path
    summary_path: Path
    metrics_path: Path
    checkpoint_path: Path | None
    losses: tuple[float, ...]
    final_loss: float
    valid_steps: int
    window_count: int
    train_window_count: int
    validation_window_count: int
    device: str

    @property
    def sequence_count(self) -> int:
        return self.window_count


def run_phase2_training(config: Phase2TrainingConfig) -> Phase2TrainingResult:
    config = validate_phase2_training_config(config)
    torch = _require_torch()
    from picar_kl.models.phase2 import Phase2KLModel, Phase2KLModelConfig
    from picar_kl.training.kl_projection import masked_action_kl_loss

    _seed_everything(config.seed, torch=torch)
    device = resolve_torch_device(config.device, torch=torch)
    cache_config = EncodingCacheConfig(
        cache_root=config.cache_root,
        model_name=config.cache_model_name,
        manifest_path=config.cache_manifest_path,
        encoder_id=config.cache_encoder_id,
        config_hash=config.cache_config_hash,
    )
    cache = VisualEncodingCache(cache_config)
    load_result = load_phase2_windows_with_stats(
        config.data_roots,
        cache=cache,
        context_steps=config.context_steps,
        prediction_steps=config.prediction_steps,
        stride=config.window_stride,
        allow_missing_cached_encodings=config.allow_partial_cache,
    )
    windows = _prepare_windows(load_result.windows, config=config)
    if not windows:
        raise ValueError("phase 2 training found no cached prefix-target windows")

    train_windows, eval_splits = split_windows_for_fit(windows, config=config)
    if not train_windows:
        raise ValueError("phase 2 training produced no training windows")

    first_shape = train_windows[0].target.visual_token_shape
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    run_id = _run_id(config.run_name)
    run_dir = config.output_root / run_id
    artifact_dir = config.checkpoint_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = artifact_dir / "metrics.jsonl"
    config_path = artifact_dir / "config.json"
    config_path.write_text(json.dumps(_jsonable(asdict(config)), indent=2, sort_keys=True), encoding="utf-8")

    losses: list[float] = []
    valid_steps = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(config.epochs):
            batch_windows_iter = _iter_batches(
                train_windows,
                config.batch_size,
                shuffle=config.fit_mode == FIT_MODE_WINDOW_SAMPLING,
            )
            for batch_index, batch_windows in enumerate(batch_windows_iter):
                batch = collate_phase2_windows(batch_windows)
                tensors = phase2_window_batch_to_tensors(batch, torch=torch, device=device)
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
                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)
                batch_steps = int(tensors["target_step_mask"].sum().item())
                valid_steps += batch_steps
                batch_metrics = action_metrics_from_arrays(
                    probabilities=output.policy.probabilities.detach().cpu().numpy(),
                    targets=tensors["target_distributions"].detach().cpu().numpy(),
                    mask=tensors["target_step_mask"].detach().cpu().numpy(),
                )
                _write_metric_row(
                    metrics_file,
                    row={
                        "row_type": "train_batch",
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "loss": loss_value,
                        "metrics": batch_metrics,
                        "breakout": _breakout_for_windows(
                            batch_windows,
                            config=config,
                            split_name="train",
                            split_strategy=_split_strategy(config),
                            sampling_policy=_sampling_policy(config),
                        ).to_dict(),
                    },
                )

            for split_name, split_windows in eval_splits.items():
                eval_row = evaluate_phase2_windows(
                    model=model,
                    windows=split_windows,
                    config=config,
                    split_name=split_name,
                    split_strategy=_split_strategy(config),
                    sampling_policy=_sampling_policy(config),
                    torch=torch,
                    loss_fn=masked_action_kl_loss,
                    device=device,
                )
                eval_row["epoch"] = epoch
                _write_metric_row(metrics_file, row=eval_row)

    final_loss = float(losses[-1])
    checkpoint_path = None
    if config.save_checkpoint:
        checkpoint_path = artifact_dir / "policy.pt"
        torch.save(
            {
                "model_state_dict": _state_dict_to_cpu(model.state_dict()),
                "model_config": asdict(model.config),
                "training_config": _jsonable(asdict(config)),
                "losses": losses,
            },
            checkpoint_path,
        )

    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase2-kl-projection",
        "conditioning_mode": "trainable-prefix-head",
        "fit_mode": config.fit_mode,
        "config": _jsonable(asdict(config)),
        "breakout": _breakout_for_windows(
            train_windows,
            config=config,
            split_name="train",
            split_strategy=_split_strategy(config),
            sampling_policy=_sampling_policy(config),
        ).to_dict(),
        "dataset": {
            "window_count": len(windows),
            "selected_window_count": len(windows),
            "loaded_window_count": len(load_result.windows),
            "candidate_window_count": load_result.candidate_window_count,
            "skipped_window_count": load_result.skipped_window_count,
            "skipped_record_count": load_result.skipped_record_count,
            "skipped_missing_cache_count": load_result.skipped_missing_cache_count,
            "skipped_no_action_count": load_result.skipped_no_action_count,
            "train_window_count": len(train_windows),
            "validation_window_count": len(eval_splits.get("validation", [])),
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
            "device_requested": config.device,
            "device": str(device),
        },
        "artifacts": {
            "artifact_dir": str(artifact_dir),
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "metrics_path": str(metrics_path),
            "config_path": str(config_path),
        },
        "source_cache": _jsonable(asdict(cache_config)),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return Phase2TrainingResult(
        run_id=run_id,
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        summary_path=summary_path,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        losses=tuple(float(item) for item in losses),
        final_loss=final_loss,
        valid_steps=valid_steps,
        window_count=len(windows),
        train_window_count=len(train_windows),
        validation_window_count=len(eval_splits.get("validation", [])),
        device=str(device),
    )


def phase2_batch_to_tensors(
    batch: Phase2Batch,
    *,
    conditioning_dim: int,
    torch: Any | None = None,
    device: Any | None = None,
) -> dict[str, Any]:
    torch = torch or _require_torch()
    visual_tokens = torch.as_tensor(batch.visual_tokens, dtype=torch.float32, device=device)
    previous_actions = torch.as_tensor(batch.previous_actions, dtype=torch.float32, device=device)
    target_distributions = torch.as_tensor(batch.target_distributions, dtype=torch.float32, device=device)
    step_mask = torch.as_tensor(batch.step_mask, dtype=torch.bool, device=device)
    conditioning = torch.zeros(
        (visual_tokens.shape[0], visual_tokens.shape[1], int(conditioning_dim)),
        dtype=torch.float32,
        device=device,
    )
    return {
        "visual_tokens": visual_tokens,
        "previous_actions": previous_actions,
        "target_distributions": target_distributions,
        "step_mask": step_mask,
        "conditioning": conditioning,
    }


def phase2_window_batch_to_tensors(
    batch: Phase2WindowBatch,
    *,
    torch: Any | None = None,
    device: Any | None = None,
) -> dict[str, Any]:
    torch = torch or _require_torch()
    return {
        "prefix_visual_tokens": torch.as_tensor(batch.prefix.visual_tokens, dtype=torch.float32, device=device),
        "prefix_actions": torch.as_tensor(batch.prefix.target_distributions, dtype=torch.float32, device=device),
        "prefix_step_mask": torch.as_tensor(batch.prefix.step_mask, dtype=torch.bool, device=device),
        "target_visual_tokens": torch.as_tensor(batch.target.visual_tokens, dtype=torch.float32, device=device),
        "target_previous_actions": torch.as_tensor(batch.target.previous_actions, dtype=torch.float32, device=device),
        "target_distributions": torch.as_tensor(batch.target.target_distributions, dtype=torch.float32, device=device),
        "target_step_mask": torch.as_tensor(batch.target.step_mask, dtype=torch.bool, device=device),
    }


def evaluate_phase2_windows(
    *,
    model: Any,
    windows: Sequence[Phase2WindowExample],
    config: Phase2TrainingConfig,
    split_name: str,
    split_strategy: str,
    sampling_policy: str,
    torch: Any,
    loss_fn: Any,
    device: Any | None = None,
) -> dict[str, Any]:
    model.eval()
    loss_rows: list[float] = []
    metric_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_windows in _iter_batches(windows, config.batch_size, shuffle=False):
            batch = collate_phase2_windows(batch_windows)
            tensors = phase2_window_batch_to_tensors(batch, torch=torch, device=device)
            output = model(
                prefix_visual_tokens=tensors["prefix_visual_tokens"],
                prefix_actions=tensors["prefix_actions"],
                prefix_step_mask=tensors["prefix_step_mask"],
                target_visual_tokens=tensors["target_visual_tokens"],
                target_previous_actions=tensors["target_previous_actions"],
                target_step_mask=tensors["target_step_mask"],
            )
            loss = loss_fn(
                logits=output.logits,
                target_distributions=tensors["target_distributions"],
                mask=tensors["target_step_mask"],
            )
            loss_rows.append(float(loss.detach().cpu().item()))
            metric_rows.append(
                action_metrics_from_arrays(
                    probabilities=output.policy.probabilities.detach().cpu().numpy(),
                    targets=tensors["target_distributions"].detach().cpu().numpy(),
                    mask=tensors["target_step_mask"].detach().cpu().numpy(),
                )
            )
    return {
        "row_type": "eval_epoch",
        "split_name": split_name,
        "loss": None if not loss_rows else float(np.mean(loss_rows)),
        "metrics": merge_metric_sums(metric_rows),
        "breakout": _breakout_for_windows(
            windows,
            config=config,
            split_name=split_name,
            split_strategy=split_strategy,
            sampling_policy=sampling_policy,
        ).to_dict(),
    }



def resolve_device_name(device: str) -> str:
    name = str(device).strip().lower()
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return name


def resolve_torch_device(device: str, *, torch: Any) -> Any:
    name = resolve_device_name(device)
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for phase 2 training, but torch.cuda.is_available() is false")
    return torch.device(name)


def _state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: value.detach().cpu() if hasattr(value, "detach") else value for key, value in state_dict.items()}

def split_windows_for_fit(
    windows: Sequence[Phase2WindowExample],
    *,
    config: Phase2TrainingConfig,
) -> tuple[list[Phase2WindowExample], dict[str, list[Phase2WindowExample]]]:
    prepared = list(windows)
    if config.fit_mode == FIT_MODE_FULL_SEQUENCE:
        return prepared, {"runtime": prepared}
    rng = random.Random(config.seed)
    rng.shuffle(prepared)
    validation_count = int(len(prepared) * float(config.validation_fraction))
    if config.validation_fraction > 0.0 and len(prepared) > 1:
        validation_count = max(1, validation_count)
    validation = prepared[:validation_count]
    train = prepared[validation_count:]
    if not train and validation:
        train = [validation.pop()]
    eval_splits = {"train": train}
    if validation:
        eval_splits["validation"] = validation
    return train, eval_splits


def validate_phase2_training_config(config: Phase2TrainingConfig) -> Phase2TrainingConfig:
    if not config.data_roots:
        raise ValueError("at least one data root is required")
    if config.fit_mode not in FIT_MODES:
        raise ValueError(f"fit_mode must be one of: {', '.join(FIT_MODES)}")
    if float(config.validation_fraction) < 0.0 or float(config.validation_fraction) >= 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
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
    resolve_device_name(config.device)
    return config


def _prepare_windows(windows: Sequence[Phase2WindowExample], *, config: Phase2TrainingConfig) -> list[Phase2WindowExample]:
    prepared = list(windows)
    if config.fit_mode == FIT_MODE_WINDOW_SAMPLING:
        rng = random.Random(config.seed)
        rng.shuffle(prepared)
    if config.max_windows is not None:
        prepared = prepared[: config.max_windows]
    return prepared


def _iter_batches(
    windows: Sequence[Phase2WindowExample],
    batch_size: int,
    *,
    shuffle: bool,
) -> Iterator[list[Phase2WindowExample]]:
    indices = list(range(len(windows)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        yield [windows[idx] for idx in indices[start : start + int(batch_size)]]


def _breakout_for_windows(
    windows: Sequence[Phase2WindowExample],
    *,
    config: Phase2TrainingConfig,
    split_name: str,
    split_strategy: str,
    sampling_policy: str,
) -> Phase2Breakout:
    return Phase2Breakout(
        fit_mode=config.fit_mode,
        split_name=split_name,
        split_strategy=split_strategy,
        run_ids=tuple(sorted(_run_ids_for_windows(windows))),
        context_steps=config.context_steps,
        prediction_steps=config.prediction_steps,
        window_stride=config.window_stride,
        sampling_policy=sampling_policy,
        random_seed=config.seed if config.fit_mode == FIT_MODE_WINDOW_SAMPLING else None,
        metadata={"window_count": len(windows)},
    )


def _run_ids_for_windows(windows: Sequence[Phase2WindowExample]) -> set[str]:
    run_ids: set[str] = set()
    for window in windows:
        for sequence in (window.prefix, window.target):
            for step in sequence.steps:
                run_uuid = step.metadata.get("run_uuid")
                if run_uuid is not None:
                    run_ids.add(str(run_uuid))
    return run_ids


def _split_strategy(config: Phase2TrainingConfig) -> str:
    if config.fit_mode == FIT_MODE_WINDOW_SAMPLING:
        return "random_window"
    return "full_sequence"


def _sampling_policy(config: Phase2TrainingConfig) -> str:
    if config.fit_mode == FIT_MODE_WINDOW_SAMPLING:
        return "shuffle_without_replacement"
    return "sequential"


def _write_metric_row(handle: Any, *, row: dict[str, Any]) -> None:
    handle.write(json.dumps(_jsonable(row), sort_keys=True) + '\n')
    handle.flush()

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
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
