"""Optional speech adapters for phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SpeechAdapterError(RuntimeError):
    pass


class EmptySpeechSource:
    def drain_texts(self) -> list[str]:
        return []


class QueueSpeechSource:
    """Simple in-memory speech source useful for tests and scripted runs."""

    def __init__(self, texts: list[str] | None = None):
        self._texts = list(texts or [])

    def append(self, text: str) -> None:
        self._texts.append(str(text))

    def drain_texts(self) -> list[str]:
        out = list(self._texts)
        self._texts.clear()
        return out


@dataclass(frozen=True)
class LegacySpeechConfig:
    model_dir: Path = Path("artifacts/models")
    allow_downloads: bool = True


class LegacySpeechSource:
    """Blocking STT source backed by copied speech code."""

    def __init__(self, service: Any):
        self.service = service

    @classmethod
    def from_config(cls, config: LegacySpeechConfig | None = None) -> "LegacySpeechSource":
        service = _build_legacy_speech_service(config or LegacySpeechConfig())
        return cls(service)

    def drain_texts(self) -> list[str]:
        result = self.service.listen_once()
        text = str(getattr(result, "text", "")).strip()
        return [text] if text else []


class LegacySpeaker:
    """TTS speaker backed by copied speech code."""

    def __init__(self, service: Any):
        self.service = service

    @classmethod
    def from_config(cls, config: LegacySpeechConfig | None = None) -> "LegacySpeaker":
        service = _build_legacy_speech_service(config or LegacySpeechConfig())
        return cls(service)

    def speak(self, text: str) -> None:
        if text.strip():
            self.service.speak(text)


def _build_legacy_speech_service(config: LegacySpeechConfig) -> Any:
    try:
        from picar_kl.legacy.robot_demo.src.speech.config import SpeechConfig
        from picar_kl.legacy.robot_demo.src.speech.service import SpeechService
    except ImportError as exc:
        raise SpeechAdapterError("Copied speech service is unavailable") from exc

    return SpeechService(
        SpeechConfig(
            model_dir=config.model_dir,
            allow_downloads=bool(config.allow_downloads),
        )
    )
