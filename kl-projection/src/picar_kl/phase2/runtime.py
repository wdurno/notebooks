"""Phase 2 policy loading and offline replay."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from picar_kl.actions import action_distribution_to_vector, distribution_to_action_name, validate_action_distribution
from picar_kl.phase2.cache import EncodingCacheConfig, VisualEncodingCache
from picar_kl.phase2.dataset import Phase2WindowExample, collate_phase2_windows, load_phase2_windows_with_stats
from picar_kl.phase2.train import phase2_window_batch_to_tensors, resolve_torch_device
from picar_kl.records import ActionRecord


POLICY_CHECKPOINT_FILENAME = "policy.pt"
SUMMARY_FILENAME = "summary.json"
CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.jsonl"


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for phase 2 runtime") from exc
    return torch


@dataclass(frozen=True)
class Phase2FitArtifact:
    fit_id: str
    artifact_dir: Path
    checkpoint_path: Path
    summary_path: Path | None = None
    config_path: Path | None = None
    metrics_path: Path | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def prediction_steps(self) -> int:
        dataset = self.summary.get("dataset") or {}
        config = self.summary.get("config") or {}
        value = dataset.get("prediction_steps", config.get("prediction_steps"))
        if value is None:
            raise ValueError(f"Phase 2 fit artifact lacks prediction_steps: {self.artifact_dir}")
        return int(value)

    @property
    def context_steps(self) -> int:
        dataset = self.summary.get("dataset") or {}
        config = self.summary.get("config") or {}
        value = dataset.get("context_steps", config.get("context_steps"))
        if value is None:
            raise ValueError(f"Phase 2 fit artifact lacks context_steps: {self.artifact_dir}")
        return int(value)

    @property
    def window_stride(self) -> int | None:
        config = self.summary.get("config") or {}
        value = config.get("window_stride")
        return None if value is None else int(value)

    @property
    def source_cache_config(self) -> EncodingCacheConfig:
        payload = self.summary.get("source_cache") or {}
        return EncodingCacheConfig(
            cache_root=Path(payload.get("cache_root") or "artifacts/data/phase2/encodings"),
            model_name=str(payload.get("model_name") or "qwen2.5-vl-3b"),
            manifest_path=Path(payload.get("manifest_path") or "artifacts/manifests/tracked/models/vlm_models.json"),
            encoder_id=str(payload.get("encoder_id") or "qwen2.5-vl-get-image-features-pooler-output-v1"),
            config_hash=str(payload.get("config_hash") or "output_dtype=float16"),
        )


@dataclass(frozen=True)
class LoadedPhase2Policy:
    artifact: Phase2FitArtifact
    model: Any
    device: str
    model_config: dict[str, Any]
    training_config: dict[str, Any]


@dataclass(frozen=True)
class Phase2ReplayStep:
    window_index: int
    target_index: int
    run_uuid: str | None
    step_index: int | None
    source_action_name: str | None
    action: ActionRecord
    conditioning_norm: float
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "target_index": self.target_index,
            "run_uuid": self.run_uuid,
            "step_index": self.step_index,
            "source_action_name": self.source_action_name,
            "action": self.action.to_dict(),
            "conditioning_norm": self.conditioning_norm,
            "latency_seconds": self.latency_seconds,
        }


@dataclass(frozen=True)
class Phase2ReplayResult:
    fit_id: str
    checkpoint_path: Path
    device: str
    context_steps: int
    prediction_steps: int
    window_count: int
    replayed_step_count: int
    skipped_window_count: int
    skipped_record_count: int
    steps: tuple[Phase2ReplayStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_id": self.fit_id,
            "checkpoint_path": str(self.checkpoint_path),
            "device": self.device,
            "context_steps": self.context_steps,
            "prediction_steps": self.prediction_steps,
            "window_count": self.window_count,
            "replayed_step_count": self.replayed_step_count,
            "skipped_window_count": self.skipped_window_count,
            "skipped_record_count": self.skipped_record_count,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class Phase2ReplayConfig:
    data_roots: tuple[Path, ...] = (Path("artifacts/data/phase1"),)
    checkpoint_root: Path = Path("artifacts/models/phase2")
    fit_id: str | None = None
    checkpoint_path: Path | None = None
    cache_root: Path | None = None
    device: str = "auto"
    max_windows: int | None = None
    allow_partial_cache: bool = False


def select_phase2_fit_artifact(
    *,
    checkpoint_root: Path = Path("artifacts/models/phase2"),
    fit_id: str | None = None,
    checkpoint_path: Path | None = None,
) -> Phase2FitArtifact:
    if fit_id and checkpoint_path is not None:
        raise ValueError("Specify either fit_id or checkpoint_path, not both")
    if checkpoint_path is not None:
        return _artifact_from_checkpoint(Path(checkpoint_path))
    root = Path(checkpoint_root)
    if fit_id:
        return _artifact_from_dir(root / fit_id)
    return latest_phase2_fit_artifact(root)


def latest_phase2_fit_artifact(checkpoint_root: Path = Path("artifacts/models/phase2")) -> Phase2FitArtifact:
    root = Path(checkpoint_root)
    if not root.exists():
        raise FileNotFoundError(root)
    candidates = []
    for artifact_dir in root.iterdir():
        if not artifact_dir.is_dir():
            continue
        checkpoint_path = artifact_dir / POLICY_CHECKPOINT_FILENAME
        if checkpoint_path.exists():
            candidates.append(artifact_dir)
    if not candidates:
        raise FileNotFoundError(f"No phase 2 policy checkpoints found under {root}")
    latest = max(candidates, key=lambda path: (path / POLICY_CHECKPOINT_FILENAME).stat().st_mtime)
    return _artifact_from_dir(latest)


def load_phase2_policy(
    *,
    checkpoint_root: Path = Path("artifacts/models/phase2"),
    fit_id: str | None = None,
    checkpoint_path: Path | None = None,
    device: str = "auto",
) -> LoadedPhase2Policy:
    torch = _require_torch()
    from picar_kl.models.phase2 import Phase2KLModel, Phase2KLModelConfig

    artifact = select_phase2_fit_artifact(
        checkpoint_root=checkpoint_root,
        fit_id=fit_id,
        checkpoint_path=checkpoint_path,
    )
    torch_device = resolve_torch_device(device, torch=torch)
    checkpoint = torch.load(artifact.checkpoint_path, map_location=torch_device)
    model_config = dict(checkpoint["model_config"])
    model = Phase2KLModel(Phase2KLModelConfig(**model_config)).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return LoadedPhase2Policy(
        artifact=artifact,
        model=model,
        device=str(torch_device),
        model_config=model_config,
        training_config=dict(checkpoint.get("training_config") or {}),
    )


def replay_phase2_policy(config: Phase2ReplayConfig) -> Phase2ReplayResult:
    loaded = load_phase2_policy(
        checkpoint_root=config.checkpoint_root,
        fit_id=config.fit_id,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
    )
    cache_config = loaded.artifact.source_cache_config
    if config.cache_root is not None:
        cache_config = EncodingCacheConfig(
            cache_root=config.cache_root,
            model_name=cache_config.model_name,
            manifest_path=cache_config.manifest_path,
            encoder_id=cache_config.encoder_id,
            config_hash=cache_config.config_hash,
        )
    cache = VisualEncodingCache(cache_config)
    load_result = load_phase2_windows_with_stats(
        config.data_roots,
        cache=cache,
        context_steps=loaded.artifact.context_steps,
        prediction_steps=loaded.artifact.prediction_steps,
        stride=loaded.artifact.window_stride,
        allow_missing_cached_encodings=config.allow_partial_cache,
    )
    windows = list(load_result.windows)
    if config.max_windows is not None:
        windows = windows[: int(config.max_windows)]
    steps = replay_phase2_windows(loaded, windows)
    return Phase2ReplayResult(
        fit_id=loaded.artifact.fit_id,
        checkpoint_path=loaded.artifact.checkpoint_path,
        device=loaded.device,
        context_steps=loaded.artifact.context_steps,
        prediction_steps=loaded.artifact.prediction_steps,
        window_count=len(windows),
        replayed_step_count=len(steps),
        skipped_window_count=load_result.skipped_window_count,
        skipped_record_count=load_result.skipped_record_count,
        steps=tuple(steps),
    )


def replay_phase2_windows(
    loaded: LoadedPhase2Policy,
    windows: Sequence[Phase2WindowExample],
) -> list[Phase2ReplayStep]:
    if not windows:
        return []
    torch = _require_torch()
    torch_device = torch.device(loaded.device)
    batch = collate_phase2_windows(windows)
    tensors = phase2_window_batch_to_tensors(batch, torch=torch, device=torch_device)
    start = time.perf_counter()
    with torch.no_grad():
        output = loaded.model(
            prefix_visual_tokens=tensors["prefix_visual_tokens"],
            prefix_actions=tensors["prefix_actions"],
            prefix_step_mask=tensors["prefix_step_mask"],
            target_visual_tokens=tensors["target_visual_tokens"],
            target_previous_actions=tensors["target_previous_actions"],
            target_step_mask=tensors["target_step_mask"],
        )
    elapsed = time.perf_counter() - start
    probabilities = output.policy.probabilities.detach().cpu().numpy()
    step_mask = tensors["target_step_mask"].detach().cpu().numpy().astype(bool)
    conditioning = output.conditioning.detach().cpu().numpy()

    replay_steps: list[Phase2ReplayStep] = []
    valid_count = max(1, int(step_mask.sum()))
    per_step_latency = elapsed / valid_count
    for window_index, window in enumerate(windows):
        conditioning_norm = float(np.linalg.norm(conditioning[window_index]))
        for target_index, step in enumerate(window.target.steps):
            if not bool(step_mask[window_index, target_index]):
                continue
            distribution = validate_action_distribution(probabilities[window_index, target_index].tolist(), tolerance=1e-5)
            replay_steps.append(
                Phase2ReplayStep(
                    window_index=window_index,
                    target_index=target_index,
                    run_uuid=step.metadata.get("run_uuid"),
                    step_index=None if step.metadata.get("step_index") is None else int(step.metadata["step_index"]),
                    source_action_name=step.metadata.get("action_name"),
                    action=ActionRecord.from_distribution(
                        distribution,
                        source="phase2-lstm-replay",
                        executed_vector=action_distribution_to_vector(distribution),
                        metadata={
                            "fit_id": loaded.artifact.fit_id,
                            "target_index": target_index,
                            "window_index": window_index,
                        },
                    ),
                    conditioning_norm=conditioning_norm,
                    latency_seconds=per_step_latency,
                )
            )
    return replay_steps


def _artifact_from_checkpoint(checkpoint_path: Path) -> Phase2FitArtifact:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    return _artifact_from_dir(checkpoint_path.parent, checkpoint_path=checkpoint_path)


def _artifact_from_dir(artifact_dir: Path, *, checkpoint_path: Path | None = None) -> Phase2FitArtifact:
    artifact_dir = Path(artifact_dir)
    checkpoint_path = checkpoint_path or artifact_dir / POLICY_CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    summary_path = artifact_dir / SUMMARY_FILENAME
    config_path = artifact_dir / CONFIG_FILENAME
    metrics_path = artifact_dir / METRICS_FILENAME
    summary = _load_json_if_exists(summary_path)
    return Phase2FitArtifact(
        fit_id=artifact_dir.name,
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path if summary_path.exists() else None,
        config_path=config_path if config_path.exists() else None,
        metrics_path=metrics_path if metrics_path.exists() else None,
        summary=summary,
    )


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
