from types import SimpleNamespace

import pytest

from picar_kl.speech.errors import SpeechError
from picar_kl.speech.service import SpeechService


class FakeAudio:
    def __init__(self):
        self.played = []

    def record_until_enter(self):
        return SimpleNamespace(wav_path="recording.wav")

    def play_wav(self, wav_path):
        self.played.append(wav_path)


class BlankSTT:
    def transcribe_file(self, wav_path):
        del wav_path
        return SimpleNamespace(text="   ")


class FakeTTS:
    def synthesize_to_tempfile(self, text):
        return SimpleNamespace(audio_path=f"{text}.wav")


def test_speak_rejects_blank_text():
    service = SpeechService.__new__(SpeechService)

    with pytest.raises(SpeechError, match="non-empty text"):
        service.speak("  ")


def test_round_trip_rejects_empty_transcript_before_tts():
    service = SpeechService.__new__(SpeechService)
    service.audio = FakeAudio()
    service.stt = BlankSTT()
    service.tts = FakeTTS()

    with pytest.raises(SpeechError, match="empty transcript"):
        service.round_trip()

    assert service.audio.played == []
