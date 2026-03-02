from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import SpeechConfig
from .errors import MissingDependencyError, SpeechError
from .model_store import ModelStore


@dataclass(frozen=True)
class SynthesisResult:
    audio_path: Path
    sample_rate: int
    voice_id: str
    backend: str


class PiperTTS:
    def __init__(self, config: SpeechConfig):
        self.config = config
        self.store = ModelStore(config)

    def synthesize_to_file(self, text: str, output_path: Path) -> SynthesisResult:
        model_path, config_path = self.store.ensure_tts_voice(self.config.tts.voice_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executable = self._resolve_executable()
        if executable is None:
            raise MissingDependencyError(
                "Piper executable not found. Install piper-tts or set TTSConfig.piper_executable."
            )

        command = [
            executable,
            "--model",
            str(model_path),
            "--output_file",
            str(output_path),
            "--length_scale",
            str(self.config.tts.length_scale),
            "--noise_scale",
            str(self.config.tts.noise_scale),
            "--noise_w",
            str(self.config.tts.noise_w),
        ]
        if self.config.tts.resolved_device() == "cuda":
            command.append("--cuda")

        completed = subprocess.run(
            command,
            input=text,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise SpeechError(
                f"Piper synthesis failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            )
        sample_rate = self._read_sample_rate(config_path)
        return SynthesisResult(
            audio_path=output_path,
            sample_rate=sample_rate,
            voice_id=self.config.tts.voice_id,
            backend="piper-cli",
        )

    def _resolve_executable(self) -> str | None:
        configured = Path(self.config.tts.piper_executable)
        if configured.is_file():
            return str(configured)

        from_path = shutil.which(self.config.tts.piper_executable)
        if from_path is not None:
            return from_path

        venv_bin = Path(sys.executable).parent / self.config.tts.piper_executable
        if venv_bin.is_file():
            return str(venv_bin)
        return None

    def synthesize_to_tempfile(self, text: str) -> SynthesisResult:
        fd, raw_path = tempfile.mkstemp(suffix=".wav", prefix="speech-tts-")
        Path(raw_path).unlink(missing_ok=True)
        return self.synthesize_to_file(text=text, output_path=Path(raw_path))

    def _read_sample_rate(self, config_path: Path) -> int:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        audio_data = data.get("audio", {})
        sample_rate = audio_data.get("sample_rate")
        if sample_rate is None:
            raise SpeechError(f"Missing sample_rate in Piper config: {config_path}")
        return int(sample_rate)
