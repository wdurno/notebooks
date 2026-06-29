"""Persist phase 1 observations and compressed images."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from picar_kl.records import Phase1ObservationRecord


class Phase1ObservationStore:
    """Append-only phase 1 run storage."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.images_dir = self.run_dir / "images"
        self.observations_path = self.run_dir / "observations.jsonl"
        self.run_meta_path = self.run_dir / "run_meta.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def write_run_metadata(self, metadata: dict[str, Any]) -> None:
        payload = dict(metadata)
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        _atomic_write_json(self.run_meta_path, payload)

    def append(self, record: Phase1ObservationRecord, *, image_rgb: Any) -> Phase1ObservationRecord:
        image_path = self._save_image(image_rgb, step_index=record.step_index)
        stored_record = replace(
            record,
            run_dir=self.run_dir,
            image_path=image_path.relative_to(self.run_dir),
        )
        self._append_jsonl(stored_record.to_dict())
        return stored_record

    def _save_image(self, image_rgb: Any, *, step_index: int) -> Path:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for phase 1 observation persistence") from exc

        image = np.asarray(image_rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        path = self.images_dir / f"step_{int(step_index):06d}.npz"
        np.savez_compressed(path, image=image)
        return path

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        self.observations_path.parent.mkdir(parents=True, exist_ok=True)
        with self.observations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


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
