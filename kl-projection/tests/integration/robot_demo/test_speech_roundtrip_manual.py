from pathlib import Path

import pytest

from picar_kl.speech.config import SpeechConfig
from picar_kl.speech.errors import MissingDependencyError, SpeechError
from picar_kl.speech.service import SpeechService


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.integration
def test_manual_speech_roundtrip():
    config = SpeechConfig(model_dir=PROJECT_ROOT / "artifacts" / "models")
    service = SpeechService(config)

    print("Manual speech test")
    print(f"Detected audio devices: {service.audio.list_devices()}")
    print("1. Press Enter to begin speaking.")
    input()
    print("2. Speak into the microphone.")
    print("3. Press Enter again to stop recording.")

    try:
        result = service.round_trip()
    except MissingDependencyError as exc:
        pytest.skip(str(exc))
    except SpeechError as exc:
        pytest.fail(f"Speech round-trip failed: {exc}")

    print(f"Transcribed text: {result.transcription.text}")
    assert result.transcription.text.strip()
