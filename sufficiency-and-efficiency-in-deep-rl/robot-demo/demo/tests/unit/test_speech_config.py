from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from speech.config import SpeechConfig, choose_runtime_device, default_demo_root, default_model_dir
from speech.manifests import load_stt_manifest, load_tts_manifest
from speech.model_store import ModelStore


def test_default_paths_match_demo_layout():
    assert default_demo_root() == PROJECT_ROOT / "demo"
    assert default_model_dir() == PROJECT_ROOT / "demo" / "model"


def test_device_selection_allows_cpu_override():
    assert choose_runtime_device(prefer_gpu=False) == "cpu"


def test_manifests_load_and_resolve_targets():
    config = SpeechConfig()
    store = ModelStore(config)
    stt_manifest = load_stt_manifest(config.model_dir)
    tts_manifest = load_tts_manifest(config.model_dir)

    assert stt_manifest["default"] == config.stt.model_name
    assert tts_manifest["default"] == config.tts.voice_id
    assert store.stt_spec(config.stt.model_name)["target_dir"] == "stt/whisper-large-v3-turbo-int8"
    assert store.tts_spec(config.tts.voice_id)["target_dir"] == "tts/piper/en_US-lessac-medium"
