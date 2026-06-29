"""Resolve tracked model manifests and local model assets."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ModelAssetError(RuntimeError):
    pass


@dataclass(frozen=True)
class VLMModelSpec:
    name: str
    repo_id: str
    target_dir: str
    allow_patterns: tuple[str, ...]
    required_files: tuple[str, ...]


class ModelStore:
    """Small model asset resolver for current experiment manifests."""

    def __init__(
        self,
        *,
        model_root: Path = Path("artifacts/models"),
        manifest_path: Path = Path("artifacts/manifests/tracked/models/vlm_models.json"),
        allow_downloads: bool = False,
    ):
        self.model_root = Path(model_root)
        self.manifest_path = Path(manifest_path)
        self.allow_downloads = bool(allow_downloads)

    def load_vlm_manifest(self) -> dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ModelAssetError(f"VLM manifest must be a JSON object: {self.manifest_path}")
        return payload

    def default_vlm_name(self) -> str:
        manifest = self.load_vlm_manifest()
        default = manifest.get("default")
        if not default:
            raise ModelAssetError(f"VLM manifest lacks default model: {self.manifest_path}")
        return str(default)

    def vlm_spec(self, model_name: str | None = None) -> VLMModelSpec:
        manifest = self.load_vlm_manifest()
        resolved_name = model_name or str(manifest.get("default") or "")
        try:
            payload = manifest["models"][resolved_name]
        except KeyError as exc:
            raise ModelAssetError(f"Unknown VLM model: {resolved_name}") from exc
        return VLMModelSpec(
            name=str(payload["name"]),
            repo_id=str(payload["repo_id"]),
            target_dir=str(payload["target_dir"]),
            allow_patterns=tuple(str(item) for item in payload.get("allow_patterns", ())),
            required_files=tuple(str(item) for item in payload.get("required_files", ())),
        )

    def ensure_vlm_model(self, model_name: str | None = None) -> Path:
        spec = self.vlm_spec(model_name)
        target_dir = self.model_root / spec.target_dir
        missing = [target_dir / item for item in spec.required_files if not (target_dir / item).exists()]
        if not missing:
            return target_dir
        if not self.allow_downloads:
            missing_text = ", ".join(str(path) for path in missing)
            raise ModelAssetError(f"Missing VLM assets for {spec.name}: {missing_text}")
        return self._download_snapshot(spec, target_dir)

    def _download_snapshot(self, spec: VLMModelSpec, target_dir: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ModelAssetError("huggingface_hub is required to download VLM assets") from exc

        staging_dir = self.model_root / "cache" / "downloads" / "vlm" / spec.name
        staging_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=staging_dir,
            local_dir_use_symlinks=False,
            allow_patterns=list(spec.allow_patterns or spec.required_files),
        )
        for src in staging_dir.rglob("*"):
            if not src.is_file() or ".cache" in src.parts:
                continue
            dst = target_dir / src.relative_to(staging_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return target_dir
