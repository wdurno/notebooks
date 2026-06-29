import json

import pytest

from picar_kl.models.store import ModelAssetError, ModelStore


def test_model_store_resolves_existing_vlm_assets(tmp_path):
    manifest_path = tmp_path / "vlm_models.json"
    model_root = tmp_path / "models"
    target = model_root / "vlm" / "qwen"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "default": "qwen",
                "models": {
                    "qwen": {
                        "name": "qwen",
                        "repo_id": "repo/qwen",
                        "target_dir": "vlm/qwen",
                        "allow_patterns": ["*.json"],
                        "required_files": ["config.json"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    store = ModelStore(model_root=model_root, manifest_path=manifest_path)

    assert store.default_vlm_name() == "qwen"
    assert store.ensure_vlm_model() == target


def test_model_store_fails_clearly_when_assets_missing(tmp_path):
    manifest_path = tmp_path / "vlm_models.json"
    manifest_path.write_text(
        json.dumps(
            {
                "default": "qwen",
                "models": {
                    "qwen": {
                        "name": "qwen",
                        "repo_id": "repo/qwen",
                        "target_dir": "vlm/qwen",
                        "required_files": ["config.json"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    store = ModelStore(model_root=tmp_path / "models", manifest_path=manifest_path)

    with pytest.raises(ModelAssetError, match="Missing VLM assets"):
        store.ensure_vlm_model()
