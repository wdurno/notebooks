from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import torch
except Exception:
    torch = None


def default_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def default_demo_root() -> Path:
    # Compatibility name from the staged legacy speech code.
    return default_project_root()


def default_model_dir() -> Path:
    return default_project_root() / "artifacts" / "models"


def choose_runtime_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(frozen=True)
class STTConfig:
    model_name: str = "whisper-large-v3-turbo-int8"
    prefer_gpu: bool = True
    device: Optional[str] = None
    compute_type_cpu: str = "int8"
    compute_type_cuda: str = "int8_float16"
    beam_size: int = 5
    vad_filter: bool = True
    language: Optional[str] = "en"

    def resolved_device(self) -> str:
        return self.device or choose_runtime_device(prefer_gpu=self.prefer_gpu)


@dataclass(frozen=True)
class TTSConfig:
    voice_id: str = "en_US-lessac-medium"
    prefer_gpu: bool = False
    device: Optional[str] = None
    speaker: Optional[int] = None
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    piper_executable: str = "piper"

    def resolved_device(self) -> str:
        return self.device or choose_runtime_device(prefer_gpu=self.prefer_gpu)


@dataclass(frozen=True)
class AudioIOConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    blocksize: int = 0


@dataclass(frozen=True)
class SpeechStreamConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    blocksize: int = 0
    amplitude_threshold: float = 0.015
    silence_seconds: float = 0.8
    min_speech_seconds: float = 0.25
    max_event_queue_size: int = 128
    callback_queue_size: int = 256
    utterance_queue_size: int = 32
    poll_interval_seconds: float = 0.05


@dataclass(frozen=True)
class SpeechConfig:
    project_root: Path = field(default_factory=default_project_root)
    model_dir: Path = field(default_factory=default_model_dir)
    allow_downloads: bool = True
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioIOConfig = field(default_factory=AudioIOConfig)

    @property
    def demo_root(self) -> Path:
        # Compatibility property for legacy call sites while they are being retired.
        return self.project_root
