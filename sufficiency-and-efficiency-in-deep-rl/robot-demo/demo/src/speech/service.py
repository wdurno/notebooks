from __future__ import annotations

from dataclasses import dataclass

from .audio_io import AudioIO, RecordedAudio
from .config import SpeechConfig
from .stt import FasterWhisperSTT, TranscriptionResult
from .tts import PiperTTS, SynthesisResult


@dataclass(frozen=True)
class SpeechRoundTrip:
    recording: RecordedAudio
    transcription: TranscriptionResult
    synthesis: SynthesisResult


class SpeechService:
    def __init__(self, config: SpeechConfig):
        self.config = config
        self.audio = AudioIO(config.audio)
        self.stt = FasterWhisperSTT(config)
        self.tts = PiperTTS(config)

    def listen_once(self) -> TranscriptionResult:
        recording = self.audio.record_until_enter()
        return self.stt.transcribe_file(recording.wav_path)

    def speak(self, text: str) -> SynthesisResult:
        synthesis = self.tts.synthesize_to_tempfile(text)
        self.audio.play_wav(synthesis.audio_path)
        return synthesis

    def round_trip(self) -> SpeechRoundTrip:
        recording = self.audio.record_until_enter()
        transcription = self.stt.transcribe_file(recording.wav_path)
        synthesis = self.tts.synthesize_to_tempfile(transcription.text)
        self.audio.play_wav(synthesis.audio_path)
        return SpeechRoundTrip(
            recording=recording,
            transcription=transcription,
            synthesis=synthesis,
        )
