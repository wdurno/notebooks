from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import default_model_dir


def manifest_dir(model_dir: Path | None = None) -> Path:
    return (model_dir or default_model_dir()) / "manifests"


def load_manifest(filename: str, model_dir: Path | None = None) -> Dict[str, Any]:
    path = manifest_dir(model_dir) / filename
    if not path.exists():
        path = _tracked_manifest_dir() / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stt_manifest(model_dir: Path | None = None) -> Dict[str, Any]:
    return load_manifest("stt_models.json", model_dir=model_dir)


def load_tts_manifest(model_dir: Path | None = None) -> Dict[str, Any]:
    return load_manifest("tts_voices.json", model_dir=model_dir)


def _tracked_manifest_dir() -> Path:
    return Path(__file__).resolve().parents[6] / "artifacts" / "manifests" / "tracked" / "models"
