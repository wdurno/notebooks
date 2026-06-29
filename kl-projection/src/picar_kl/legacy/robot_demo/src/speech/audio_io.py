from __future__ import annotations

import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import numpy as np

from .config import AudioIOConfig
from .errors import AudioDeviceError, MissingDependencyError


@dataclass(frozen=True)
class RecordedAudio:
    wav_path: Path
    sample_rate: int
    channels: int
    duration_seconds: float


class AudioIO:
    def __init__(self, config: AudioIOConfig):
        self.config = config

    def list_devices(self) -> list[str]:
        sd = self._sounddevice()
        return [str(device["name"]) for device in sd.query_devices()]

    def record_until_enter(self, prompt: str = "Press Enter to stop recording.") -> RecordedAudio:
        sd = self._sounddevice()
        np = self._numpy()
        frames: List[np.ndarray] = []
        stop_event = threading.Event()

        def callback(indata, frame_count, time_info, status):
            del frame_count, time_info
            if status:
                raise AudioDeviceError(str(status))
            frames.append(indata.copy())
            if stop_event.is_set():
                raise sd.CallbackStop()

        print(prompt, flush=True)
        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            device=self.config.input_device,
            blocksize=self.config.blocksize,
            callback=callback,
        ):
            input()
            stop_event.set()

        if not frames:
            raise AudioDeviceError("No audio frames were recorded")

        audio = np.concatenate(frames, axis=0)
        fd, raw_path = tempfile.mkstemp(suffix=".wav", prefix="speech-recording-")
        Path(raw_path).unlink(missing_ok=True)
        wav_path = Path(raw_path)
        self.write_wav(wav_path, audio, self.config.sample_rate)
        duration_seconds = float(audio.shape[0]) / float(self.config.sample_rate)
        return RecordedAudio(
            wav_path=wav_path,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            duration_seconds=duration_seconds,
        )

    def play_wav(self, wav_path: Path) -> None:
        sd = self._sounddevice()
        np = self._numpy()
        with wave.open(str(wav_path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.readframes(handle.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
        sd.play(audio, samplerate=sample_rate, device=self.config.output_device)
        sd.wait()

    def write_wav(self, path: Path, samples: "np.ndarray", sample_rate: int) -> None:
        np = self._numpy()
        samples = np.asarray(samples)
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        channels = 1 if pcm.ndim == 1 else pcm.shape[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())

    def _sounddevice(self):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise MissingDependencyError(
                "sounddevice is required for microphone and speaker I/O"
            ) from exc
        return sd

    def _numpy(self):
        try:
            import numpy as np
        except ImportError as exc:
            raise MissingDependencyError("numpy is required for audio sample processing") from exc
        return np
