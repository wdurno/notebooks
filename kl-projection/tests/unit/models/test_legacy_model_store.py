from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src" / "picar_kl" / "legacy" / "robot_demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.config import ModelConfig
from model.model_store import ModelStore


def test_legacy_model_store_falls_back_to_tracked_project_manifest(tmp_path):
    store = ModelStore(ModelConfig(model_dir=tmp_path / "models", allow_downloads=False))

    spec = store.model_spec("qwen2.5-vl-3b")

    assert spec["repo_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
