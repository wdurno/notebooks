from picar_kl.speech.manifests import load_stt_manifest, load_tts_manifest


def test_speech_manifests_fall_back_to_tracked_project_manifests(tmp_path):
    stt_manifest = load_stt_manifest(tmp_path / "models")
    tts_manifest = load_tts_manifest(tmp_path / "models")

    assert stt_manifest["default"] == "whisper-large-v3-turbo-int8"
    assert tts_manifest["default"] == "en_US-lessac-medium"
