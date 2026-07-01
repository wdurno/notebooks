from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from .config import SpeechConfig
from .errors import MissingDependencyError, ModelAssetError
from .manifests import load_stt_manifest, load_tts_manifest


class ModelStore:
    def __init__(self, config: SpeechConfig):
        self.config = config
        self.model_dir = config.model_dir
        self.cache_dir = self.model_dir / "cache" / "downloads"

    def stt_spec(self, model_name: str) -> Dict[str, Any]:
        manifest = load_stt_manifest(self.model_dir)
        try:
            return manifest["models"][model_name]
        except KeyError as exc:
            raise ModelAssetError(f"Unknown STT model: {model_name}") from exc

    def tts_spec(self, voice_id: str) -> Dict[str, Any]:
        manifest = load_tts_manifest(self.model_dir)
        try:
            return manifest["voices"][voice_id]
        except KeyError as exc:
            raise ModelAssetError(f"Unknown TTS voice: {voice_id}") from exc

    def ensure_stt_model(self, model_name: str, allow_downloads: bool | None = None) -> Path:
        spec = self.stt_spec(model_name)
        target_dir = self.model_dir / spec["target_dir"]
        required_files = [target_dir / relative_path for relative_path in spec["required_files"]]
        if all(path.exists() for path in required_files):
            return target_dir
        if allow_downloads is None:
            allow_downloads = self.config.allow_downloads
        if not allow_downloads:
            missing = ", ".join(str(path.relative_to(self.model_dir)) for path in required_files if not path.exists())
            raise ModelAssetError(f"Missing STT model assets for {model_name}: {missing}")
        return self._download_stt_snapshot(spec=spec, target_dir=target_dir)

    def ensure_tts_voice(self, voice_id: str, allow_downloads: bool | None = None) -> tuple[Path, Path]:
        spec = self.tts_spec(voice_id)
        target_dir = self.model_dir / spec["target_dir"]
        model_path = target_dir / spec["model_filename"]
        config_path = target_dir / spec["config_filename"]
        if model_path.exists() and config_path.exists():
            return model_path, config_path
        if allow_downloads is None:
            allow_downloads = self.config.allow_downloads
        if not allow_downloads:
            raise ModelAssetError(f"Missing TTS voice assets for {voice_id} under {target_dir}")
        return self._download_tts_voice(spec=spec, target_dir=target_dir)

    def _download_stt_snapshot(self, spec: Dict[str, Any], target_dir: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise MissingDependencyError(
                "huggingface_hub is required to download STT model assets"
            ) from exc

        staging_dir = self.cache_dir / "stt" / spec["name"]
        staging_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=spec["repo_id"],
            local_dir=staging_dir,
            local_dir_use_symlinks=False,
            allow_patterns=spec["required_files"],
        )
        for relative_path in spec["required_files"]:
            src = staging_dir / relative_path
            dst = target_dir / relative_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return target_dir

    def _download_tts_voice(self, spec: Dict[str, Any], target_dir: Path) -> tuple[Path, Path]:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise MissingDependencyError(
                "huggingface_hub is required to download TTS voice assets"
            ) from exc

        target_dir.mkdir(parents=True, exist_ok=True)
        for item in spec["files"]:
            downloaded = hf_hub_download(
                repo_id=spec["repo_id"],
                filename=item["repo_path"],
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
            final_path = target_dir / item["local_name"]
            if Path(downloaded) != final_path:
                shutil.copy2(downloaded, final_path)
        return target_dir / spec["model_filename"], target_dir / spec["config_filename"]
