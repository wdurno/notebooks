from pathlib import Path

import pytest

from picar_kl.speech.config import SpeechConfig
from picar_kl.speech.errors import SpeechError
from picar_kl.speech.tts import PiperTTS


def test_piper_tts_rejects_blank_text_before_model_lookup(tmp_path):
    tts = PiperTTS.__new__(PiperTTS)
    tts.config = SpeechConfig(model_dir=tmp_path)
    tts.store = object()

    with pytest.raises(SpeechError, match="non-empty text"):
        tts.synthesize_to_file("   ", Path(tmp_path / "blank.wav"))
