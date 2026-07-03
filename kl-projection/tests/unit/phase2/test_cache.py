import numpy as np

from picar_kl.records import Phase1ObservationRecord
from picar_kl.models.visual import VisualTokenEncoding
from picar_kl.phase2.cache import EncodingCacheConfig, VisualEncodingCache


def _record(tmp_path, *, step_index=0):
    run_dir = tmp_path / "run-a"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / f"step_{step_index:06d}.npz"
    np.savez_compressed(image_path, image=np.zeros((4, 5, 3), dtype=np.uint8))
    return Phase1ObservationRecord(
        run_uuid="run-a",
        run_dir=run_dir,
        timestamp=None,
        source="step",
        step_index=step_index,
        image_path=image_path.relative_to(run_dir),
        messages=[],
        user_texts=[],
        action=None,
    )


def test_encoding_cache_stores_manifest_and_tensor(tmp_path):
    cache = VisualEncodingCache(
        EncodingCacheConfig(
            cache_root=tmp_path / "cache",
            model_name="qwen-test",
            manifest_path=tmp_path / "vlm_models.json",
            encoder_id="encoder-test",
            config_hash="abc",
        )
    )
    record = _record(tmp_path)
    encoding = VisualTokenEncoding(
        tokens=np.arange(12, dtype=np.float32).reshape(3, 4),
        metadata={"source": "unit-test"},
    )

    entry = cache.store(record, image_shape=(4, 5, 3), encoding=encoding)

    assert cache.has(record)
    assert entry.encoding_shape == (3, 4)
    manifest = cache.load_manifest()
    assert entry.cache_key in manifest.entries
    loaded = cache.load(record)
    assert loaded.tokens.shape == (3, 4)
    assert loaded.metadata["source"] == "unit-test"


def test_encoding_cache_key_changes_with_step(tmp_path):
    cache = VisualEncodingCache(EncodingCacheConfig(cache_root=tmp_path / "cache"))
    first = _record(tmp_path, step_index=0)
    second = _record(tmp_path, step_index=1)

    assert cache.cache_key(first) != cache.cache_key(second)
