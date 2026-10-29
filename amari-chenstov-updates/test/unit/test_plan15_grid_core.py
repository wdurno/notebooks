"""Focused CPU model/Gaussian-curvature tests, without experimental fits."""

from dataclasses import replace
import math

import pytest
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.func import jacrev

from mnist_experiment.fisher_memory_grid_registration.curvature import (
    SIGMA2, fd_values, gaussian_nll, jvp_values, output_jacobians, reference_fisher,
)
from mnist_experiment.fisher_memory_grid_registration.model import (
    build_model, flat_prediction, memory_indices,
)
from src.parameters import ParameterLayout


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def case(*, channels=(4, 6), moved=True):
    model, layout = build_model(17, channels=channels, dtype=torch.float64, factor_seed=57)
    theta = layout.flatten_module(model, detach=True)
    generator = torch.Generator().manual_seed(91)
    if moved:
        for spec in layout.specs:
            if spec.name.endswith("lora_B"):
                theta[spec.start:spec.stop] = .15 * torch.randn(spec.numel, generator=generator,
                                                              dtype=theta.dtype)
    x = torch.randn(2, 2, 84, 84, generator=generator, dtype=theta.dtype)
    ltm, stm = memory_indices(layout)
    probes = (2 * torch.randint(2, (2, len(stm)), generator=generator) - 1).double()
    return model, layout, theta.requires_grad_(), x, ltm, stm, probes


@pytest.mark.parametrize("channels,counts", [((4, 6), (3268, 344)), ((4, 8), (4314, 384))])
def test_capacity_counts_lora_blocks_dense_head_and_rng_pairing(channels, counts):
    rng = torch.random.get_rng_state().clone()
    model, layout = build_model(11, channels=channels, factor_seed=42, dtype=torch.float64)
    twin, _ = build_model(11, channels=channels, factor_seed=42, dtype=torch.float64)
    other, _ = build_model(11, channels=channels, factor_seed=43, dtype=torch.float64)
    assert torch.equal(rng, torch.random.get_rng_state())
    ltm, stm = memory_indices(layout)
    assert (len(ltm), len(stm)) == counts and layout.total_numel == sum(counts)
    assert torch.equal(torch.cat((ltm, stm)).sort().values, torch.arange(sum(counts)))
    assert model.classifier.in_features == 9 * channels[1] >= 54
    assert model.classifier.out_features == model.output_dim == 54
    assert model.classifier.weight.shape == (54, 9 * channels[1])
    assert torch.linalg.matrix_rank(model.classifier.weight) == 54
    assert model.features[0].rank == model.features[3].rank == model.classifier.rank == 2
    assert model.features[2].kernel_size == 2 and model.features[5].kernel_size == 14
    theta = layout.flatten_module(model)
    assert torch.equal(theta, layout.flatten_module(twin))
    other_theta = layout.flatten_module(other)
    assert torch.equal(theta[ltm], other_theta[ltm])
    assert not torch.equal(theta[stm], other_theta[stm])
    x = torch.ones(2, 2, 84, 84, dtype=theta.dtype)
    prediction = flat_prediction(model, layout, theta, x)
    assert prediction.shape == (2, 54)
    torch.testing.assert_close(prediction, other(x), rtol=0, atol=0)
    torch.testing.assert_close(prediction, model(x, stm_scale=0), rtol=0, atol=0)
    changed = theta.detach().clone()
    bias = next(spec for spec in layout.specs if spec.name == "classifier.bias")
    changed[bias.start:bias.stop] += 100
    torch.testing.assert_close(flat_prediction(model, layout, changed, x) - prediction,
                               torch.full_like(prediction, 100))
    assert all(parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(ValueError, match="84, 84"):
        model(x[:, :, :28, :28])


def test_pooling_preserves_channel_row_column_layout():
    model, _ = build_model(1)
    cells = torch.arange(9).float().reshape(1, 1, 3, 3).repeat_interleave(14, 2).repeat_interleave(14, 3)
    cells = torch.cat((cells, cells + 10), dim=1)
    pooled = model.features[5](cells)
    assert pooled.shape == (1, 2, 3, 3)
    torch.testing.assert_close(pooled.flatten(1), torch.tensor([[0., 1., 2., 3., 4., 5., 6., 7., 8.,
                                                             10., 11., 12., 13., 14., 15., 16., 17., 18.]]))


def test_chart_rejects_legacy_architectures_offsets_and_dimensions():
    from mnist_experiment.fisher_memory_registration.model import build_model as old_model

    with pytest.raises(ValueError, match="54"):
        memory_indices(old_model(1)[1])
    _, layout = build_model(1)
    damaged = replace(layout.specs[0], stop=layout.specs[0].stop + 1)
    with pytest.raises(ValueError, match="offsets"):
        memory_indices(replace(layout, specs=(damaged,) + layout.specs[1:]))
    with pytest.raises(ValueError, match="chart"):
        memory_indices(replace(layout, specs=tuple(reversed(layout.specs))))
    with pytest.raises(ValueError, match="count"):
        memory_indices(replace(layout, total_numel=layout.total_numel + 1))
    for kwargs in ({"output_dim": 6}, {"output_dim": True}, {"channels": (6, 8)},
                   {"channels": (4, 5)}, {"channels": (4,)}, {"seed": True},
                   {"factor_seed": -1}, {"dtype": torch.float16}):
        with pytest.raises(ValueError):
            build_model(**({"seed": 1} | kwargs))


def test_random_a_zero_b_initial_gradients_and_ltm_only_pretraining():
    model, layout, theta, x, ltm, stm, _ = case(moved=False)
    y = torch.full((len(x), 54), 2., dtype=x.dtype)
    gradient = torch.autograd.grad(gaussian_nll(flat_prediction(model, layout, theta, x), y).mean(), theta)[0]
    assert gradient[ltm].norm() > 0 and gradient[stm].norm() > 0
    for spec in layout.specs:
        if spec.name.endswith("lora_A"):
            assert theta[spec.start:spec.stop].count_nonzero() == spec.numel
            assert gradient[spec.start:spec.stop].count_nonzero() == 0
        elif spec.name.endswith("lora_B"):
            assert theta[spec.start:spec.stop].count_nonzero() == 0
            assert gradient[spec.start:spec.stop].norm() > 0
    loss = gaussian_nll(flat_prediction(model, layout, theta, x, stm_scale=0), y).mean()
    gradient = torch.autograd.grad(loss, theta)[0]
    assert gradient[stm].count_nonzero() == 0 and gradient[ltm].norm() > 0


def test_gaussian_nll_sums_54_coordinates_and_gradient_is_negative_score():
    prediction = torch.linspace(-2., 3., 108, dtype=torch.float64).reshape(2, 54).requires_grad_()
    y = torch.linspace(1., -1., 108, dtype=torch.float64).reshape(2, 54)
    normal = Normal(prediction, 2.)
    nll = gaussian_nll(prediction, y)
    expected = (prediction - y).square().sum(-1) / 8 + 27 * math.log(8 * math.pi)
    torch.testing.assert_close(nll, expected)
    torch.testing.assert_close(nll, -normal.log_prob(y).sum(-1))
    gradient = torch.autograd.grad(nll.sum(), prediction, retain_graph=True)[0]
    score = torch.autograd.grad(normal.log_prob(y).sum(), prediction)[0]
    torch.testing.assert_close(gradient, -score)
    torch.testing.assert_close(gradient, (prediction - y) / 4)
    blocks = sum(gaussian_nll(prediction[:, start:start + 6], y[:, start:start + 6])
                 for start in range(0, 54, 6))
    torch.testing.assert_close(nll, blocks)


class LinearPixels(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 54
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.register_buffer("coordinate_scale", torch.linspace(.5, 1.5, 54, dtype=torch.float64))

    def forward(self, x, stm_scale=1.):
        return (x[:, :, 0, 0] @ self.weight + self.bias)[:, None] * self.coordinate_scale


def linear_case():
    model = LinearPixels()
    layout = ParameterLayout.from_module(model)
    theta = layout.flatten_module(model, detach=True)
    x = torch.zeros(3, 2, 84, 84, dtype=theta.dtype)
    x[:, :, 0, 0] = torch.tensor([[1., 2.], [-2., .5], [3., -1.]])
    return model, layout, theta, x


def test_analytic_gaussian_fisher_chunked_output_rows_and_exact_target_integration():
    model, layout, theta, x = linear_case()
    jacobian = jacrev(lambda p: flat_prediction(model, layout, p, x))(theta)
    expected = torch.einsum("ndp,ndq->pq", jacobian, jacobian) / (len(x) * SIGMA2)
    fisher = reference_fisher(model, layout, theta, x, chunk_size=2, output_chunk_size=7)
    assert fisher.dtype == torch.float64 and not fisher.requires_grad
    assert torch.equal(fisher, fisher.mT)
    torch.testing.assert_close(fisher, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(reference_fisher(model, layout, theta, x, chunk_size=1,
                                               output_chunk_size=54), expected)
    selected = torch.tensor([2, 0])
    torch.testing.assert_close(reference_fisher(model, layout, theta, x, indices=selected),
                               expected[selected][:, selected])
    iterator = output_jacobians(model, layout, theta, x, chunk_size=2, output_chunk_size=7)
    first = next(iterator)
    assert torch.is_grad_enabled()
    pieces = [first, *iterator]
    assert all(not piece.requires_grad and piece.dtype == torch.float64 for piece in pieces)
    assert all(piece.shape[0] <= 2 and piece.shape[1] <= 7 and piece.shape[2] == 3 for piece in pieces)
    torch.testing.assert_close(torch.cat(pieces[:8], dim=1), jacobian[:2])
    torch.testing.assert_close(torch.cat(pieces[8:], dim=1), jacobian[2:])
    per_grid = torch.einsum("ndp,ndq->npq", jacobian, jacobian) / 4
    torch.testing.assert_close(per_grid.mean(0), fisher)
    per_cell = sum(torch.einsum("ndp,ndq->pq", jacobian[:, j:j + 6], jacobian[:, j:j + 6])
                   for j in range(0, 54, 6)) / (len(x) * 4)
    torch.testing.assert_close(per_cell, fisher)
    wrong = jacobian.mean(0).T @ jacobian.mean(0) / 4
    assert not torch.allclose(fisher, wrong)
    means = flat_prediction(model, layout, theta, x).detach()
    observed = jacrev(lambda p: gaussian_nll(flat_prediction(model, layout, p, x), means))(theta)
    assert observed.count_nonzero() == 0 and fisher.norm() > 0
    # The +/- coordinate-axis targets integrate the Gaussian's second moments.
    integrated = torch.zeros_like(fisher)
    for axis in torch.eye(54, dtype=theta.dtype):
        residual = 2 * math.sqrt(54) * axis
        gradients = torch.einsum("ndp,d->np", jacobian, residual) / 4
        integrated += gradients.T @ gradients / (54 * len(x))
    torch.testing.assert_close(integrated, fisher)


def test_real_grid_cnn_bias_fisher_has_no_extra_output_or_cell_normalization():
    model, layout, theta, x, _, _, _ = case()
    bias = next(spec for spec in layout.specs if spec.name == "classifier.bias")
    indices = torch.arange(bias.start, bias.stop)
    fisher = reference_fisher(model, layout, theta, x[:1], indices=indices, output_chunk_size=18)
    torch.testing.assert_close(fisher, torch.eye(54, dtype=torch.float64) / 4, rtol=0, atol=0)


@pytest.mark.parametrize("channels", [(4, 6), (4, 8)])
def test_fd_pixel_kl_jvp_values_and_gradients_reach_both_blocks(channels):
    model, layout, theta, x, ltm, stm, probes = case(channels=channels)
    x = x[:1]
    step = 1e-4
    actual = fd_values(model, layout, theta, x, probes, step)
    exact = jvp_values(model, layout, theta, x, probes)
    assert actual.shape == (2, 1) and actual.dtype == torch.float64
    torch.testing.assert_close(actual, exact, rtol=1e-6, atol=1e-10)
    embedded = theta.new_zeros(len(probes), len(theta)).index_copy(1, stm, probes)
    torch.testing.assert_close(actual, fd_values(model, layout, theta, x, embedded, step))
    kl_step = .01
    values = fd_values(model, layout, theta, x, probes, kl_step)
    for k, direction in enumerate(embedded):
        plus = Normal(flat_prediction(model, layout, theta + kl_step / 2 * direction, x), 2.)
        minus = Normal(flat_prediction(model, layout, theta - kl_step / 2 * direction, x), 2.)
        expected = (kl_divergence(plus, minus) + kl_divergence(minus, plus)).sum(-1) / kl_step**2
        torch.testing.assert_close(values[k], expected, rtol=1e-7, atol=1e-10)
    gradient = torch.autograd.grad(actual.mean() / len(stm), theta, retain_graph=True)[0]
    exact_gradient = torch.autograd.grad(exact.mean() / len(stm), theta)[0]
    assert gradient[ltm].norm() > 0 and gradient[stm].norm() > 0
    torch.testing.assert_close(gradient, exact_gradient, rtol=2e-5, atol=1e-10)
    for spec in layout.specs:
        if spec.name == "classifier.bias":
            assert gradient[spec.start:spec.stop].count_nonzero() == 0
        else:
            assert gradient[spec.start:spec.stop].norm() > 0
    detached_probes = probes.clone().requires_grad_()
    penalty = fd_values(model, layout, theta, x, detached_probes, .01).mean() / len(stm)
    assert torch.autograd.grad(penalty, detached_probes, allow_unused=True)[0] is None


def test_curvature_validation_and_cooperative_stop():
    model, layout, theta, x = linear_case()
    for kwargs in ({"chunk_size": 0}, {"output_chunk_size": True}, {"indices": []},
                   {"indices": torch.tensor([0, 0])}, {"indices": torch.tensor([-1])},
                   {"indices": torch.tensor([.5])}):
        with pytest.raises(ValueError):
            reference_fisher(model, layout, theta, x, **kwargs)
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 5

    from mnist_experiment.fisher_memory.learner import StopRequested
    with pytest.raises(StopRequested):
        reference_fisher(model, layout, theta, x, check_stop=stop)
    model, layout, theta, x, ltm, stm, probes = case()
    for bad in (0., -1., float("nan"), True):
        with pytest.raises(ValueError, match="step"):
            fd_values(model, layout, theta, x, probes, bad)
    contaminated = torch.zeros(1, len(theta), dtype=theta.dtype)
    contaminated[:, ltm[0]] = 1.
    with pytest.raises(ValueError, match="zero in LTM"):
        fd_values(model, layout, theta, x, contaminated, .01)
    with pytest.raises(ValueError, match="84, 84"):
        reference_fisher(model, layout, theta, x[:, :, :28, :28], indices=stm[:2])
