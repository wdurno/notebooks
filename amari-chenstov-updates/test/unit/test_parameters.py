import pytest
import torch
from torch import nn

from src.parameters import ParameterLayout, ParameterLayoutError


class HeterogeneousModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.arange(6, dtype=torch.float64).reshape(2, 3))
        self.bias = nn.Parameter(torch.tensor([7.0, 8.0], dtype=torch.float64))
        self.register_parameter(
            "frozen",
            nn.Parameter(torch.tensor([9.0], dtype=torch.float64), requires_grad=False),
        )


def test_parameter_layout_flatten_restore_round_trip() -> None:
    model = HeterogeneousModel()
    layout = ParameterLayout.from_module(model)

    flat = layout.flatten_module(model, detach=True)
    restored = layout.unflatten_named(flat)

    assert layout.names == ("matrix", "bias")
    assert layout.total_numel == 8
    assert torch.equal(restored["matrix"], model.matrix)
    assert torch.equal(restored["bias"], model.bias)
    assert layout.flatten_named(restored).equal(flat)


def test_parameter_layout_flattens_batched_parameter_pytrees() -> None:
    model = HeterogeneousModel()
    layout = ParameterLayout.from_module(model)
    batched = {
        "matrix": torch.stack((model.matrix, model.matrix + 10)),
        "bias": torch.stack((model.bias, model.bias + 10)),
    }

    result = layout.flatten_batched_named(batched)

    assert result.shape == (2, 8)
    assert torch.equal(result[0], layout.flatten_module(model))
    assert torch.equal(result[1], result[0] + 10)


def test_parameter_layout_detects_shape_and_metadata_mismatches() -> None:
    model = HeterogeneousModel()
    layout = ParameterLayout.from_module(model)

    with pytest.raises(ParameterLayoutError, match="shape mismatch"):
        layout.flatten_named(
            {
                "matrix": torch.zeros(3, 2, dtype=torch.float64),
                "bias": model.bias,
            }
        )

    with pytest.raises(ParameterLayoutError, match="dtype mismatch"):
        layout.flatten_named(
            {
                "matrix": model.matrix.float(),
                "bias": model.bias,
            }
        )

    metadata = layout.metadata()
    metadata["total_numel"] += 1
    with pytest.raises(ParameterLayoutError, match="metadata"):
        layout.assert_metadata(metadata)


def test_detached_flatten_is_an_independent_snapshot() -> None:
    model = HeterogeneousModel()
    layout = ParameterLayout.from_module(model)
    snapshot = layout.flatten_module(model, detach=True)

    with torch.no_grad():
        model.matrix.add_(100)

    assert not torch.equal(snapshot, layout.flatten_module(model))


def test_parameter_vector_can_be_copied_back_into_module() -> None:
    model = HeterogeneousModel()
    layout = ParameterLayout.from_module(model)
    original = layout.flatten_module(model, detach=True)
    replacement = original + 3

    layout.copy_vector_to_module(model, replacement)

    assert torch.equal(layout.flatten_module(model), replacement)
