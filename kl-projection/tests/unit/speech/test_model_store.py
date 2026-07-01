from pathlib import Path

import pytest

from picar_kl.speech.config import SpeechConfig
from picar_kl.speech.errors import ModelAssetError
from picar_kl.speech.model_store import ModelStore


def test_speech_model_store_reports_missing_stt_assets(tmp_path):
    store = ModelStore(SpeechConfig(model_dir=tmp_path / "models", allow_downloads=False))

    with pytest.raises(ModelAssetError, match="Missing STT model assets"):
        store.ensure_stt_model("whisper-large-v3-turbo-int8")


def test_speech_model_store_reports_missing_tts_assets(tmp_path):
    store = ModelStore(SpeechConfig(model_dir=tmp_path / "models", allow_downloads=False))

    with pytest.raises(ModelAssetError, match="Missing TTS voice assets"):
        store.ensure_tts_voice("en_US-lessac-medium")


def test_speech_model_store_uses_existing_tts_assets(tmp_path):
    model_root = tmp_path / "models"
    voice_dir = model_root / "tts" / "piper" / "en_US-lessac-medium"
    model_path = voice_dir / "en_US-lessac-medium.onnx"
    config_path = voice_dir / "en_US-lessac-medium.onnx.json"
    voice_dir.mkdir(parents=True)
    model_path.write_bytes(b"model")
    config_path.write_text("{}")

    store = ModelStore(SpeechConfig(model_dir=model_root, allow_downloads=False))

    assert store.ensure_tts_voice("en_US-lessac-medium") == (model_path, config_path)
