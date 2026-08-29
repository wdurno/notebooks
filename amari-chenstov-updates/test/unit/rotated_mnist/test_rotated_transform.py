import torch
from pathlib import Path

from mnist_experiment.rotated_mnist.config import load_config
from mnist_experiment.rotated_mnist.transform import (
    rotate_mnist_tensor,
    tensor_content_hash,
)


SMOKE_CONFIG = (
    Path(__file__).parents[3]
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase1_smoke.json"
)


def test_rotation_is_deterministic_and_shape_preserving() -> None:
    config = load_config(SMOKE_CONFIG).rotation
    image = torch.arange(28 * 28, dtype=torch.float32).reshape(1, 28, 28)
    image /= image.numel() - 1

    first = rotate_mnist_tensor(image, 15.0, config)
    second = rotate_mnist_tensor(image, 15.0, config)

    assert first.shape == image.shape
    assert first.dtype == image.dtype
    assert torch.equal(first, second)
    assert tensor_content_hash(first) == tensor_content_hash(second)
    assert float(first[0, 0, 0]) == 0.0


def test_rotation_rejects_noncanonical_inputs() -> None:
    config = load_config(SMOKE_CONFIG).rotation
    bad = torch.zeros(28, 28)

    try:
        rotate_mnist_tensor(bad, 15.0, config)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("noncanonical image shape was accepted")
