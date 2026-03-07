from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ObservationStore:
    """Persist per-step observations and compressed frame blobs."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.images_dir = self.run_dir / "images"
        self.observations_path = self.run_dir / "observations.jsonl"
        self.run_meta_path = self.run_dir / "run_meta.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def write_run_metadata(self, metadata: dict[str, Any]) -> None:
        payload = _to_jsonable(metadata)
        _atomic_write_json(self.run_meta_path, payload)

    def append_observation(
        self,
        *,
        observation: Any,
        reward: float | None,
        user_texts: list[str],
        action: Any | None,
        training: Any | None,
        action_receipt: Any | None,
        source: str,
    ) -> None:
        image_path = self._save_compressed_image(observation.image_rgb, step_index=int(observation.step_index))
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "step_index": int(observation.step_index),
            "t": float(observation.t),
            "last_reward": float(observation.last_reward),
            "reward": None if reward is None else float(reward),
            "done": bool(observation.done),
            "metadata": _to_jsonable(getattr(observation, "metadata", {})),
            "messages": _to_jsonable(getattr(observation, "messages", [])),
            "user_texts": list(user_texts),
            "action": _action_to_record(action),
            "training": _to_jsonable(training),
            "action_receipt": _to_jsonable(action_receipt),
            "image_path": str(image_path.relative_to(self.run_dir)),
        }
        self._append_jsonl(record)

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        self.observations_path.parent.mkdir(parents=True, exist_ok=True)
        with self.observations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(record), sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _save_compressed_image(self, image_rgb: Any, *, step_index: int) -> Path:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for observation persistence") from exc

        image = np.asarray(image_rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        path = self.images_dir / f"step_{step_index:06d}.npz"
        np.savez_compressed(path, image=image)
        return path


def _action_to_record(action: Any | None) -> dict[str, Any] | None:
    if action is None:
        return None
    keys = (
        "agentic_action_name",
        "agentic_action_vector",
        "actor_action_vector",
        "executed_action_vector",
        "critic_value",
        "generated_text",
        "logp_beta_sum",
    )
    record = {}
    for key in keys:
        if hasattr(action, key):
            record[key] = _to_jsonable(getattr(action, key))
    return record


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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
