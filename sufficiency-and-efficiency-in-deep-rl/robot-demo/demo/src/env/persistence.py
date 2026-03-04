from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import ExperimentPaths


def create_experiment_paths(data_dir: Path, experiment_name_prefix: str) -> ExperimentPaths:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    base_name = f"{timestamp}-{experiment_name_prefix}"
    run_dir = data_dir / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = data_dir / f"{base_name}-resume-{suffix:02d}"
        suffix += 1
    blobs_dir = run_dir / "blobs"
    logs_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    artifacts_dir = run_dir / "artifacts"
    for path in (blobs_dir, logs_dir, metrics_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        run_dir=run_dir,
        metadata_path=run_dir / "metadata.json",
        blobs_dir=blobs_dir,
        logs_dir=logs_dir,
        metrics_dir=metrics_dir,
        artifacts_dir=artifacts_dir,
    )


def write_metadata(paths: ExperimentPaths, metadata: dict[str, Any]) -> None:
    payload = _to_jsonable(metadata)
    paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return None


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(record), sort_keys=True))
        handle.write("\n")
    return None


def save_frame_array(path: Path, image_rgb: Any) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to persist frame arrays") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, image_rgb)
    return None


def save_replay_snapshot(replay_buffer: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    replay_buffer.save(path)
    return None


def save_model_artifacts(model: Any, artifacts_dir: Path, checkpoint_basename: str) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to persist model artifacts") from exc

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    base_path = artifacts_dir / checkpoint_basename
    if hasattr(model, "save"):
        model.save(str(base_path))

    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(trainable_state, artifacts_dir / f"{checkpoint_basename}.trainable.pt")

    actor_head = getattr(model, "actor_head", None)
    if actor_head is not None:
        torch.save(actor_head.state_dict(), artifacts_dir / f"{checkpoint_basename}.actor_head.pt")

    critic = getattr(model, "critic", None)
    if critic is not None:
        torch.save(critic.state_dict(), artifacts_dir / f"{checkpoint_basename}.critic.pt")

    backbone_model = getattr(getattr(model, "backbone", None), "model", None)
    if backbone_model is not None and hasattr(backbone_model, "save_pretrained"):
        adapter_dir = artifacts_dir / f"{checkpoint_basename}-lora"
        backbone_model.save_pretrained(adapter_dir)
    return None


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
