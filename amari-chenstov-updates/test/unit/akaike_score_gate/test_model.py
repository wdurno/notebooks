import pytest
import torch

from mnist_experiment.akaike_score_gate.model import (
    PARAMETER_GROUPS, active_indices, build_model, layer_indices,
)
from mnist_experiment.rotated_mnist.plan12.gauge import helmert_contrast
from src.parameters import ParameterLayoutError


@pytest.mark.parametrize("width,count,depth_counts", [
    (5, 501, [54, 84, 114, 239, 461, 501]),
    (6, 559, [63, 105, 147, 297, 519, 559]),
])
def test_parameter_chart_and_nested_depth_counts(width, count, depth_counts):
    model, layout = build_model(12, width, dtype=torch.float64)
    assert layout.total_numel == count
    assert tuple(dict(model.named_children())) == PARAMETER_GROUPS
    assert tuple(layer_indices(layout)) == PARAMETER_GROUPS
    previous = set()
    for depth, expected in enumerate(depth_counts, 1):
        indices, metadata = active_indices(layout, depth)
        assert indices.dtype == torch.int64 and indices.device.type == "cpu"
        assert indices.numel() == expected == metadata["active_count"]
        assert previous <= set(indices.tolist())
        assert not set(metadata["candidate_indices"]) & set(indices.tolist())
        assert metadata["indices"] == indices.tolist()
        previous = set(indices.tolist())
    assert active_indices(layout, 1)[1]["name"] == "output_only"
    assert active_indices(layout, 4)[1]["name"] == "dense_head"
    assert active_indices(layout, 6)[1]["candidate_indices"] == []
    assert all(p.requires_grad for p in model.parameters())


def test_seed_pairing_does_not_change_global_rng_and_layout_roundtrips():
    state = torch.random.get_rng_state().clone()
    model, layout = build_model(4)
    assert torch.equal(state, torch.random.get_rng_state())
    twin, _ = build_model(4)
    other, _ = build_model(5)
    vector = layout.flatten_module(model, detach=True)
    assert torch.equal(vector, layout.flatten_module(twin))
    assert not torch.equal(vector, layout.flatten_module(other))
    layout.copy_vector_to_module(other, vector)
    assert torch.equal(vector, layout.flatten_module(other))
    layout.assert_metadata(layout.metadata())
    with pytest.raises(ParameterLayoutError):
        layout.assert_metadata(build_model(4, 6)[1].metadata())


def test_helmert_chart_ten_probabilities_and_hidden_nonlinearities():
    model, _ = build_model(2, dtype=torch.float64)
    x = torch.randn(2, 1, 28, 28, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(30))
    contrast = model.output.contrast
    assert torch.equal(contrast, helmert_contrast())
    torch.testing.assert_close(contrast.mT @ contrast, torch.eye(9, dtype=torch.float64))
    torch.testing.assert_close(contrast.sum(0), torch.zeros(9, dtype=torch.float64))
    logits = model(x)
    assert logits.shape == (2, 10)
    torch.testing.assert_close(logits.sum(1), torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(2, dtype=torch.float64))
    f = torch.nn.functional
    features = f.adaptive_avg_pool2d(f.silu(model.conv2(
        f.avg_pool2d(f.silu(model.conv1(x)), 2))), (2, 2)).flatten(1)
    expected = model.output(f.silu(model.hidden3(f.silu(model.hidden2(f.silu(model.hidden1(features)))))))
    torch.testing.assert_close(logits, expected)
    affine_only = model.output(model.hidden3(model.hidden2(model.hidden1(features))))
    assert not torch.allclose(logits, affine_only)


@pytest.mark.parametrize("depth", [0, 7, True, 1.5])
def test_invalid_depth_rejected(depth):
    with pytest.raises(ValueError, match="depth"):
        active_indices(build_model(0)[1], depth)


def test_old_parameter_chart_is_not_accepted():
    from src.mnist_model import build_canonical_model

    with pytest.raises(ValueError, match="chart"):
        active_indices(build_canonical_model(0)[1], 1)
