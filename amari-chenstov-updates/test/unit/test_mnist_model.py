import pytest
import torch
from torch import nn

from src.mnist_model import (
    CANONICAL_PARAMETER_COUNT,
    CanonicalMnistCNN,
    build_canonical_model,
    configure_torch_runtime,
)


def test_canonical_model_has_required_shape_and_parameter_count() -> None:
    model = CanonicalMnistCNN()
    inputs = torch.zeros(4, 1, 28, 28)

    outputs = model(inputs)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert outputs.shape == (4, 10)
    assert parameter_count == CANONICAL_PARAMETER_COUNT == 512
    assert [type(module) for module in model.features] == [
        nn.Conv2d,
        nn.SiLU,
        nn.AvgPool2d,
        nn.Conv2d,
        nn.SiLU,
        nn.AdaptiveAvgPool2d,
    ]
    assert isinstance(model.classifier, nn.Linear)


def test_canonical_model_contains_no_forbidden_modules() -> None:
    model = CanonicalMnistCNN()
    forbidden = (nn.Dropout, nn.modules.batchnorm._BatchNorm, nn.ReLU, nn.MaxPool2d)

    assert not any(
        isinstance(module, forbidden)
        for module in model.modules()
    )


def test_canonical_initialization_is_deterministic_by_seed() -> None:
    first, first_layout = build_canonical_model(123, dtype=torch.float64)
    second, second_layout = build_canonical_model(123, dtype=torch.float64)
    different, _ = build_canonical_model(124, dtype=torch.float64)

    assert first_layout.metadata() == second_layout.metadata()
    for first_parameter, second_parameter in zip(
        first.parameters(),
        second.parameters(),
        strict=True,
    ):
        assert torch.equal(first_parameter, second_parameter)
    assert any(
        not torch.equal(first_parameter, different_parameter)
        for first_parameter, different_parameter in zip(
            first.parameters(),
            different.parameters(),
            strict=True,
        )
    )


def test_runtime_can_warn_for_unsupported_deterministic_operations() -> None:
    try:
        configure_torch_runtime(
            deterministic_algorithms=True,
            warn_only=True,
        )

        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        configure_torch_runtime(deterministic_algorithms=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_canonical_model_initializes_and_runs_on_gpu() -> None:
    model, layout = build_canonical_model(
        123,
        device="cuda",
        dtype=torch.float32,
    )

    outputs = model(torch.zeros(2, 1, 28, 28, device="cuda"))

    assert outputs.shape == (2, 10)
    assert outputs.device.type == "cuda"
    assert layout.total_numel == 512
