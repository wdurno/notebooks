"""Deterministic image rotation and content hashing."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate

from .config import RotationConfig


ROTATION_TRANSFORM_SCHEMA_VERSION = 1


def tensor_content_hash(tensor: Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rotate_mnist_tensor(
    image: Tensor,
    angle_degrees: float,
    config: RotationConfig,
) -> Tensor:
    """Rotate one float MNIST tensor without augmentation or resizing."""

    config.validate()
    if image.ndim != 3 or image.shape != (1, 28, 28):
        raise ValueError("MNIST image must have shape (1, 28, 28)")
    if not image.is_floating_point() or not bool(torch.isfinite(image).all()):
        raise ValueError("MNIST image must be a finite floating-point tensor")
    if image.device.type != "cpu":
        raise ValueError("rotation stream materialization requires CPU tensors")
    if (
        isinstance(angle_degrees, bool)
        or not isinstance(angle_degrees, (int, float))
        or not math.isfinite(float(angle_degrees))
    ):
        raise ValueError("rotation angle must be finite")
    result = rotate(
        image,
        float(angle_degrees),
        interpolation=InterpolationMode.BILINEAR,
        expand=config.expand,
        fill=[config.fill],
    )
    if result.shape != image.shape or result.dtype != image.dtype:
        raise RuntimeError("rotation changed MNIST tensor shape or dtype")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("rotation produced non-finite pixels")
    return result.contiguous()


def rotate_mnist_batch(
    images: Tensor,
    angle_degrees: float,
    config: RotationConfig,
) -> Tensor:
    """Apply the versioned transform to a CPU batch without changing semantics."""

    config.validate()
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("MNIST image batch must have shape (batch, 1, 28, 28)")
    if images.shape[0] == 0:
        raise ValueError("MNIST image batch cannot be empty")
    if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
        raise ValueError("MNIST image batch must be finite and floating point")
    if images.device.type != "cpu":
        raise ValueError("rotation materialization requires CPU tensors")
    if (
        isinstance(angle_degrees, bool)
        or not isinstance(angle_degrees, (int, float))
        or not math.isfinite(float(angle_degrees))
    ):
        raise ValueError("rotation angle must be finite")
    result = rotate(
        images,
        float(angle_degrees),
        interpolation=InterpolationMode.BILINEAR,
        expand=config.expand,
        fill=[config.fill],
    )
    if result.shape != images.shape or result.dtype != images.dtype:
        raise RuntimeError("rotation changed MNIST batch shape or dtype")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("rotation produced non-finite pixels")
    return result.contiguous()
