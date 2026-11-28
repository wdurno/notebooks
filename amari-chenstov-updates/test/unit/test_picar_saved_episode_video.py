"""Original camera-frame recovery uses scored pixel identities, not new episodes."""

import hashlib

import numpy as np
import pytest

from numerical_experiment.picar_saved_episode_video import episode_images


def save_frame(root, value):
    import io

    array = np.full((4, 6, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    path = root / (hashlib.sha256(payload).hexdigest() + ".npy")
    path.write_bytes(payload)
    return {"shape": list(array.shape), "dtype": "uint8",
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}, array, path


def episode(targets):
    return {"scores": [{"diagnostic": {"target": target}} for target in targets]}


def test_original_observation_order_and_repeated_frames(tmp_path):
    first, a, _ = save_frame(tmp_path, 1)
    second, b, _ = save_frame(tmp_path, 2)
    rows = episode_images(episode([second, first, second]), tmp_path)
    for (_, actual), expected in zip(rows, (b, a, b)):
        assert np.array_equal(actual, expected)


def test_missing_image_is_not_replaced_by_rendering(tmp_path):
    target, _, path = save_frame(tmp_path, 1)
    path.unlink()
    with pytest.raises(ValueError, match="unavailable"):
        episode_images(episode([target]), tmp_path)


def test_scored_image_shape_must_match(tmp_path):
    target, _, _ = save_frame(tmp_path, 1)
    target["shape"] = [6, 4, 3]
    with pytest.raises(ValueError, match="identity differs"):
        episode_images(episode([target]), tmp_path)
