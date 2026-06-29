from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np

from .config import SpeechConfig
from .errors import MissingDependencyError
from .model_store import ModelStore


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: Optional[str]
    language_probability: Optional[float]
    duration_seconds: Optional[float]
    segments: list[dict[str, Any]]


class FasterWhisperSTT:
    def __init__(self, config: SpeechConfig):
        self.config = config
        self.store = ModelStore(config)
        self._model = None

    def transcribe_file(self, wav_path: Path) -> TranscriptionResult:
        model = self._load_model()
        segments, info = model.transcribe(
            str(wav_path),
            beam_size=self.config.stt.beam_size,
            vad_filter=self.config.stt.vad_filter,
            language=self.config.stt.language,
        )
        segment_list = [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            for segment in segments
        ]
        text = " ".join(segment["text"].strip() for segment in segment_list).strip()
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration_seconds=getattr(info, "duration", None),
            segments=segment_list,
        )

    def transcribe_array(self, samples: "np.ndarray", sample_rate: int) -> TranscriptionResult:
        from .audio_io import AudioIO, AudioIOConfig

        fd, raw_path = tempfile.mkstemp(suffix=".wav", prefix="speech-stt-")
        Path(raw_path).unlink(missing_ok=True)
        wav_path = Path(raw_path)
        AudioIO(AudioIOConfig(sample_rate=sample_rate)).write_wav(wav_path, samples, sample_rate)
        return self.transcribe_file(wav_path)

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MissingDependencyError(
                "faster-whisper is required for STT inference"
            ) from exc

        model_dir = self.store.ensure_stt_model(self.config.stt.model_name)
        device = self.config.stt.resolved_device()
        compute_type = (
            self.config.stt.compute_type_cuda if device == "cuda" else self.config.stt.compute_type_cpu
        )
        self._model = WhisperModel(
            str(model_dir),
            device=device,
            compute_type=compute_type,
        )
        return self._model
