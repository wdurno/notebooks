from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from picar_kl.speech.config import SpeechStreamConfig
from picar_kl.speech.errors import MissingDependencyError, SpeechError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueuedSpeechEvent:
    text: str
    received_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UtteranceSegmenter:
    sample_rate: int
    amplitude_threshold: float
    silence_seconds: float
    min_speech_seconds: float
    _active_chunks: list[Any] = field(default_factory=list)
    _speech_samples: int = 0
    _silence_samples: int = 0

    def ingest(self, samples: Any) -> list[Any]:
        np = _numpy()
        chunk = np.asarray(samples, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        if chunk.size == 0:
            return []

        rms = float(np.sqrt(np.mean(np.square(chunk))))
        utterances = []
        silence_limit = int(self.sample_rate * self.silence_seconds)
        min_speech_samples = int(self.sample_rate * self.min_speech_seconds)
        if rms >= self.amplitude_threshold:
            self._active_chunks.append(chunk.copy())
            self._speech_samples += int(chunk.size)
            self._silence_samples = 0
        elif self._active_chunks:
            self._active_chunks.append(chunk.copy())
            self._silence_samples += int(chunk.size)
            if self._silence_samples >= silence_limit:
                utterance = np.concatenate(self._active_chunks, axis=0)
                if self._speech_samples >= min_speech_samples:
                    utterances.append(utterance)
                self.reset()
        return utterances

    def flush(self) -> list[Any]:
        np = _numpy()
        if not self._active_chunks:
            return []
        utterance = np.concatenate(self._active_chunks, axis=0)
        min_speech_samples = int(self.sample_rate * self.min_speech_seconds)
        self.reset()
        if utterance.size >= min_speech_samples:
            return [utterance]
        return []

    def reset(self) -> None:
        self._active_chunks.clear()
        self._speech_samples = 0
        self._silence_samples = 0


class ContinuousSpeechStream:
    """Background microphone listener that yields completed utterance texts."""

    def __init__(
        self,
        config: SpeechStreamConfig,
        *,
        transcriber: Any,
        sounddevice_module: Any | None = None,
    ):
        self.config = config
        self.transcriber = transcriber
        self._sounddevice = sounddevice_module
        self._segmenter = UtteranceSegmenter(
            sample_rate=config.sample_rate,
            amplitude_threshold=config.amplitude_threshold,
            silence_seconds=config.silence_seconds,
            min_speech_seconds=config.min_speech_seconds,
        )
        self._audio_queue: queue.Queue[Any] = queue.Queue(maxsize=config.callback_queue_size)
        self._event_queue: queue.Queue[QueuedSpeechEvent] = queue.Queue(maxsize=config.max_event_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._stream = None

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return None
        sd = self._sounddevice or self._load_sounddevice()

        def callback(indata, frame_count, time_info, status):
            del frame_count, time_info
            if status:
                LOGGER.warning("[stt] input stream status=%s", status)
                return None
            try:
                self._audio_queue.put_nowait(indata.copy())
            except queue.Full:
                LOGGER.warning("[stt] dropped audio chunk because callback queue is full")
            return None

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.blocksize,
            callback=callback,
        )
        self._stream.start()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, name="continuous-speech-stream", daemon=True)
        self._worker_thread.start()
        return None

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._stream = None
        return None

    def drain(self) -> list[QueuedSpeechEvent]:
        events = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                return events

    def push_audio(self, samples: Any) -> None:
        for utterance in self._segmenter.ingest(samples):
            self._transcribe_and_enqueue(utterance)
        return None

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                samples = self._audio_queue.get(timeout=self.config.poll_interval_seconds)
            except queue.Empty:
                continue
            for utterance in self._segmenter.ingest(samples):
                self._transcribe_and_enqueue(utterance)
        for utterance in self._segmenter.flush():
            self._transcribe_and_enqueue(utterance)
        return None

    def _transcribe_and_enqueue(self, utterance: Any) -> None:
        try:
            result = self.transcriber.transcribe_array(utterance, self.config.sample_rate)
        except SpeechError:
            raise
        except Exception as exc:
            LOGGER.exception("[stt] transcription failed")
            raise SpeechError("Continuous speech transcription failed") from exc
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            return None
        LOGGER.info("[stt] captured text=%r", text)
        event = QueuedSpeechEvent(text=text, received_at=time.time(), metadata={"segments": getattr(result, "segments", [])})
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            LOGGER.warning("[stt] dropped utterance because event queue is full")
        return None

    def _load_sounddevice(self):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise MissingDependencyError("sounddevice is required for continuous speech streaming") from exc
        return sd


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise MissingDependencyError("numpy is required for continuous speech streaming") from exc
    return np
