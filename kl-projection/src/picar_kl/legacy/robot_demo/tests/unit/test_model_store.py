from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.config import ModelConfig
from model.model_store import ModelStore


def test_ensure_base_model_uses_existing_assets_without_download(tmp_path):
    model_dir = tmp_path / "model"
    target_dir = model_dir / "vlm" / "qwen2.5-vl-3b" / "base"
    manifest_dir = model_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    (manifest_dir / "vlm_models.json").write_text(
        """
        {
          "default": "qwen2.5-vl-3b",
          "models": {
            "qwen2.5-vl-3b": {
              "name": "qwen2.5-vl-3b",
              "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
              "target_dir": "vlm/qwen2.5-vl-3b/base",
              "required_files": [
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json"
              ]
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    for filename in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (target_dir / filename).write_text("{}", encoding="utf-8")

    store = ModelStore(ModelConfig(model_dir=model_dir, allow_downloads=False))

    resolved = store.ensure_base_model()

    assert resolved == Path(target_dir)
