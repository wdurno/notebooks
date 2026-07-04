from picar_kl.speech.config import STTConfig, TTSConfig


def test_stt_prefers_gpu_by_default():
    assert STTConfig().prefer_gpu is True


def test_tts_uses_cpu_by_default():
    config = TTSConfig()

    assert config.prefer_gpu is False
    assert config.resolved_device() == "cpu"


def test_tts_allows_explicit_cuda_device():
    assert TTSConfig(device="cuda").resolved_device() == "cuda"
