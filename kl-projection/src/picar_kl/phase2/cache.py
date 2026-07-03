"""Precomputed visual-token encoding cache for phase 2."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from picar_kl.models.visual import VisualTokenEncoding
from picar_kl.records import Phase1ObservationRecord


MANIFEST_FILENAME = "manifest.json"
ENCODINGS_DIRNAME = "tensors"


@dataclass(frozen=True)
class EncodingCacheConfig:
    cache_root: Path = Path("artifacts/data/phase2/encodings")
    model_name: str = "qwen2.5-vl-3b"
    manifest_path: Path = Path("artifacts/manifests/tracked/models/vlm_models.json")
    encoder_id: str = "qwen2.5-vl-get-image-features-pooler-output-v1"
    config_hash: str = "default"


@dataclass(frozen=True)
class EncodingCacheEntry:
    cache_key: str
    run_uuid: str | None
    step_index: int
    source_image_path: str
    encoding_path: str
    model_name: str
    manifest_path: str
    encoder_id: str
    config_hash: str
    image_shape: tuple[int, ...]
    encoding_shape: tuple[int, int]
    dtype: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EncodingCacheEntry":
        return cls(
            cache_key=str(payload["cache_key"]),
            run_uuid=None if payload.get("run_uuid") is None else str(payload["run_uuid"]),
            step_index=int(payload["step_index"]),
            source_image_path=str(payload["source_image_path"]),
            encoding_path=str(payload["encoding_path"]),
            model_name=str(payload["model_name"]),
            manifest_path=str(payload["manifest_path"]),
            encoder_id=str(payload["encoder_id"]),
            config_hash=str(payload["config_hash"]),
            image_shape=tuple(int(item) for item in payload["image_shape"]),
            encoding_shape=tuple(int(item) for item in payload["encoding_shape"]),
            dtype=str(payload["dtype"]),
            created_at=str(payload["created_at"]),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_shape"] = list(self.image_shape)
        payload["encoding_shape"] = list(self.encoding_shape)
        return payload


@dataclass(frozen=True)
class EncodingCacheManifest:
    version: int = 1
    entries: dict[str, EncodingCacheEntry] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EncodingCacheManifest":
        entries = {
            str(key): EncodingCacheEntry.from_dict(value)
            for key, value in dict(payload.get("entries") or {}).items()
        }
        return cls(version=int(payload.get("version", 1)), entries=entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "entries": {key: entry.to_dict() for key, entry in sorted(self.entries.items())},
        }


class VisualEncodingCache:
    """Disk cache for visual-token encodings."""

    def __init__(self, config: EncodingCacheConfig | None = None):
        self.config = config or EncodingCacheConfig()
        self.cache_root = Path(self.config.cache_root)
        self.tensors_dir = self.cache_root / ENCODINGS_DIRNAME
        self.manifest_path = self.cache_root / MANIFEST_FILENAME

    def load_manifest(self) -> EncodingCacheManifest:
        if not self.manifest_path.exists():
            return EncodingCacheManifest()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Encoding cache manifest must be a JSON object: {self.manifest_path}")
        return EncodingCacheManifest.from_dict(payload)

    def has(self, record: Phase1ObservationRecord) -> bool:
        key = self.cache_key(record)
        entry = self.load_manifest().entries.get(key)
        return bool(entry and self.path_for_entry(entry).exists())

    def load(self, record: Phase1ObservationRecord) -> VisualTokenEncoding:
        key = self.cache_key(record)
        entry = self.load_manifest().entries.get(key)
        if entry is None:
            raise FileNotFoundError(f"No cached visual encoding for key {key}")
        path = self.path_for_entry(entry)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = np.load(path, allow_pickle=False)
        if "tokens" not in payload:
            raise ValueError(f"Encoding tensor file lacks `tokens`: {path}")
        return VisualTokenEncoding(tokens=payload["tokens"], metadata=dict(entry.metadata))

    def store(
        self,
        record: Phase1ObservationRecord,
        *,
        image_shape: tuple[int, ...],
        encoding: VisualTokenEncoding,
    ) -> EncodingCacheEntry:
        key = self.cache_key(record)
        tensor_rel_path = Path(ENCODINGS_DIRNAME) / f"{key}.npz"
        tensor_path = self.cache_root / tensor_rel_path
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tensor_path, tokens=np.asarray(encoding.tokens))

        entry = EncodingCacheEntry(
            cache_key=key,
            run_uuid=record.run_uuid,
            step_index=int(record.step_index),
            source_image_path=str(record.image_file or record.image_path),
            encoding_path=str(tensor_rel_path),
            model_name=self.config.model_name,
            manifest_path=str(self.config.manifest_path),
            encoder_id=self.config.encoder_id,
            config_hash=self.config.config_hash,
            image_shape=tuple(int(item) for item in image_shape),
            encoding_shape=encoding.shape,
            dtype=encoding.dtype,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=dict(encoding.metadata),
        )
        manifest = self.load_manifest()
        entries = dict(manifest.entries)
        entries[key] = entry
        self._write_manifest(EncodingCacheManifest(version=manifest.version, entries=entries))
        return entry

    def cache_key(self, record: Phase1ObservationRecord) -> str:
        image_file = record.image_file
        image_stat = None
        if image_file is not None and image_file.exists():
            stat = image_file.stat()
            image_stat = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        payload = {
            "run_uuid": record.run_uuid,
            "step_index": int(record.step_index),
            "image_path": None if image_file is None else str(image_file),
            "image_stat": image_stat,
            "model_name": self.config.model_name,
            "manifest_path": str(self.config.manifest_path),
            "encoder_id": self.config.encoder_id,
            "config_hash": self.config.config_hash,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    def path_for_entry(self, entry: EncodingCacheEntry) -> Path:
        path = Path(entry.encoding_path)
        if path.is_absolute():
            return path
        return self.cache_root / path

    def _write_manifest(self, manifest: EncodingCacheManifest) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(manifest.to_dict()), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.manifest_path)


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
