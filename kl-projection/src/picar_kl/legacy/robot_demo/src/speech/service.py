from __future__ import annotations

from dataclasses import dataclass

from .audio_io import AudioIO, RecordedAudio
from .config import SpeechConfig
from .stt import FasterWhisperSTT, TranscriptionResult
from .tts import PiperTTS, SynthesisResult


@dataclass(frozen=True)
class SpeechRoundTrip:
    """Structured output for one microphone-to-speaker interaction."""

    recording: RecordedAudio
    transcription: TranscriptionResult
    synthesis: SynthesisResult


class SpeechService:
    """High-level speech facade for the demo.

    This service coordinates microphone input, speech-to-text inference,
    text-to-speech synthesis, and speaker playback behind a small interface
    that the rest of the application can call directly.
    """

    def __init__(self, config: SpeechConfig):
        """Build a speech service from the provided runtime configuration."""

        self.config = config
        self.audio = AudioIO(config.audio)
        self.stt = FasterWhisperSTT(config)
        self.tts = PiperTTS(config)

    def listen_once(self) -> TranscriptionResult:
        """Record one utterance from the microphone and transcribe it."""

        recording = self.audio.record_until_enter()
        return self.stt.transcribe_file(recording.wav_path)

    def speak(self, text: str) -> SynthesisResult:
        """Synthesize text to speech and play it through the active output device."""

        synthesis = self.tts.synthesize_to_tempfile(text)
        self.audio.play_wav(synthesis.audio_path)
        return synthesis

    def round_trip(self) -> SpeechRoundTrip:
        """Record speech, transcribe it, synthesize the transcript, and play it back."""

        recording = self.audio.record_until_enter()
        transcription = self.stt.transcribe_file(recording.wav_path)
        synthesis = self.tts.synthesize_to_tempfile(transcription.text)
        self.audio.play_wav(synthesis.audio_path)
        return SpeechRoundTrip(
            recording=recording,
            transcription=transcription,
            synthesis=synthesis,
        )
