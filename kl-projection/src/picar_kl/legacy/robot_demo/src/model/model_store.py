from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import ModelConfig


class ModelAssetError(RuntimeError):
    pass


class ModelStore:
    """Resolve and optionally download base VLM assets under `demo/model/`."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_dir = config.model_dir
        self.cache_dir = self.model_dir / "cache" / "downloads"

    def load_manifest(self) -> dict[str, Any]:
        manifest_path = self.model_dir / "manifests" / "vlm_models.json"
        if not manifest_path.exists():
            manifest_path = _tracked_manifest_dir() / "vlm_models.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def model_spec(self, model_name: str) -> dict[str, Any]:
        manifest = self.load_manifest()
        try:
            return manifest["models"][model_name]
        except KeyError as exc:
            raise ModelAssetError(f"Unknown VLM model: {model_name}") from exc

    def ensure_base_model(self) -> Path:
        spec = self.model_spec(self.config.model_name)
        target_dir = self.model_dir / spec["target_dir"]
        required_files = [target_dir / item for item in spec["required_files"]]
        if all(path.exists() for path in required_files):
            return target_dir
        if not self.config.allow_downloads:
            missing = ", ".join(str(path.relative_to(self.model_dir)) for path in required_files if not path.exists())
            raise ModelAssetError(f"Missing VLM assets for {self.config.model_name}: {missing}")
        return self._download_snapshot(spec=spec, target_dir=target_dir)

    def _download_snapshot(self, spec: dict[str, Any], target_dir: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ModelAssetError("huggingface_hub is required to download the base VLM") from exc

        staging_dir = self.cache_dir / "vlm" / spec["name"]
        staging_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec["repo_id"],
            local_dir=staging_dir,
            local_dir_use_symlinks=False,
            allow_patterns=spec.get("allow_patterns", spec["required_files"]),
        )
        # Copy the full staged snapshot into the final model directory, excluding
        # Hugging Face cache bookkeeping. The real Qwen weights are typically
        # sharded safetensors files, so copying only `required_files` would
        # leave the download incomplete and unusable by `from_pretrained`.
        for src in staging_dir.rglob("*"):
            if not src.is_file():
                continue
            if ".cache" in src.parts:
                continue
            dst = target_dir / src.relative_to(staging_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return target_dir


def _tracked_manifest_dir() -> Path:
    return Path(__file__).resolve().parents[6] / "artifacts" / "manifests" / "tracked" / "models"
