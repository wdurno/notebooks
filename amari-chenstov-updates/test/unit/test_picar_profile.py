import json

import pytest

from numerical_experiment.picar.preflight import inspect_model_assets


def test_incomplete_or_traversing_assets_rejected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
    with pytest.raises(ValueError, match="no safetensors"):
        inspect_model_assets(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "../outside"}}))
    with pytest.raises(ValueError, match="unsafe"):
        inspect_model_assets(tmp_path)


def test_small_asset_manifest(tmp_path):
    for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "config.json").write_text('{"model_type":"qwen2_5_vl"}')
    (tmp_path / "model.safetensors").write_bytes(b"test-only")
    manifest = inspect_model_assets(tmp_path, hash_weights=True)
    assert manifest["total_bytes"] == 9
    assert len(manifest["files"]["model.safetensors"]["sha256"]) == 64
