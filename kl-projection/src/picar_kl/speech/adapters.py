"""Optional speech adapters for phase runtimes."""

from __future__ import annotations

from typing import Any

from picar_kl.speech.config import SpeechConfig, SpeechStreamConfig
from picar_kl.speech.service import SpeechService
from picar_kl.speech.stream import ContinuousSpeechStream


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


class SpeechServiceSource:
    """Blocking STT source backed by the current speech service."""

    def __init__(self, service: Any):
        self.service = service

    @classmethod
    def from_config(cls, config: SpeechConfig | None = None) -> "SpeechServiceSource":
        return cls(_build_speech_service(config or SpeechConfig()))

    def drain_texts(self) -> list[str]:
        result = self.service.listen_once()
        text = str(getattr(result, "text", "")).strip()
        return [text] if text else []


class StreamingSpeechSource:
    """Non-blocking STT source backed by a continuous microphone stream."""

    def __init__(self, stream: ContinuousSpeechStream):
        self.stream = stream

    @classmethod
    def from_service(
        cls,
        service: Any,
        config: SpeechStreamConfig | None = None,
    ) -> "StreamingSpeechSource":
        return cls(ContinuousSpeechStream(config or SpeechStreamConfig(), transcriber=service.stt))

    @classmethod
    def from_config(
        cls,
        config: SpeechConfig | None = None,
        stream_config: SpeechStreamConfig | None = None,
    ) -> "StreamingSpeechSource":
        return cls.from_service(_build_speech_service(config or SpeechConfig()), stream_config)

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()

    def drain_texts(self) -> list[str]:
        return [event.text for event in self.stream.drain() if event.text.strip()]


class SpeechServiceSpeaker:
    """TTS speaker backed by the current speech service."""

    def __init__(self, service: Any):
        self.service = service

    @classmethod
    def from_config(cls, config: SpeechConfig | None = None) -> "SpeechServiceSpeaker":
        return cls(_build_speech_service(config or SpeechConfig()))

    def speak(self, text: str) -> None:
        if text.strip():
            self.service.speak(text)


def _build_speech_service(config: SpeechConfig) -> SpeechService:
    try:
        return SpeechService(config)
    except ImportError as exc:
        raise SpeechAdapterError("Speech service is unavailable") from exc


LegacySpeechConfig = SpeechConfig
LegacySpeechSource = SpeechServiceSource
LegacySpeaker = SpeechServiceSpeaker
