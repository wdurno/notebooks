"""Bounded CPU contracts for the real Plan 18 chart and Gaussian derivatives."""

from dataclasses import replace
import math

import pytest
import torch
from torch import nn
from torch.func import jacrev

from mnist_experiment.combined_memory import curvature, model
from mnist_experiment.fisher_memory_stm_capacity import model as inherited


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def case(dtype=torch.float64, count=1):
    net, layout = model.build_model(17, dtype=dtype, factor_seed=57)
    theta = layout.flatten_module(net, detach=True)
    generator = torch.Generator().manual_seed(91)
    _, stm = model.memory_indices(layout)
    theta[stm[-54:]] = .1 * torch.randn(54, generator=generator, dtype=dtype)
    x = torch.rand(count, 2, 120, 120, generator=generator, dtype=dtype)
    return net, layout, theta, x, generator


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_chart_pooling_rng_and_inherited_initialization(dtype):
    state = torch.get_rng_state().clone()
    net, layout = model.build_model(11, dtype=dtype, factor_seed=42)
    assert torch.equal(state, torch.get_rng_state())
    old, old_layout = inherited.build_model(11, dtype=dtype, factor_seed=42, stm_config="head_r1")
    assert layout.metadata() == old_layout.metadata()
    torch.testing.assert_close(layout.flatten_module(net), old_layout.flatten_module(old), rtol=0, atol=0)
    ltm, stm = model.memory_indices(layout)
    assert (len(ltm), len(stm), layout.total_numel) == (3268, 108, 3376)
    assert torch.equal(ltm, torch.arange(3268))
    assert torch.equal(stm, torch.arange(3268, 3376))
    assert all(p.requires_grad for p in net.parameters())
    assert isinstance(net.features[0], nn.Conv2d) and isinstance(net.features[3], nn.Conv2d)
    assert net.features[5].kernel_size == net.features[5].stride == 20
    assert (net.classifier.rank, net.classifier.lora_alpha) == (1, 1.)
    assert net.classifier.lora_B.count_nonzero() == 0
    assert net.classifier.lora_A.count_nonzero() == 54
    x = torch.ones(1, 2, 120, 120, dtype=dtype)
    theta = layout.flatten_module(net)
    pred = model.flat_prediction(net, layout, theta, x)
    assert pred.shape == (1, 54)
    assert net.features(x).shape == (1, 6, 3, 3)
    torch.testing.assert_close(pred, net(x, stm_scale=0), rtol=0, atol=0)
    with pytest.raises(ValueError, match="120,120"):
        net(torch.zeros(1, 2, 84, 84, dtype=dtype))


def test_chart_validation_and_seed_errors():
    _, layout = model.build_model(1)
    with pytest.raises(ValueError, match="chart/order"):
        model.memory_indices(replace(layout, specs=tuple(reversed(layout.specs))))
    bad = replace(layout.specs[0], stop=layout.specs[0].stop + 1)
    with pytest.raises(ValueError, match="offsets"):
        model.memory_indices(replace(layout, specs=(bad,) + layout.specs[1:]))
    with pytest.raises(ValueError, match="count"):
        model.memory_indices(replace(layout, total_numel=3377))
    for kwargs in (dict(seed=True), dict(seed=-1), dict(seed=1, factor_seed=True),
                   dict(seed=1, dtype=torch.float16)):
        with pytest.raises(ValueError):
            model.build_model(**kwargs)


def test_gaussian_normalization_positive_scores_and_per_grid_gradients():
    net, layout, theta, x, generator = case(count=2)
    variable = theta.clone().requires_grad_()
    pred = model.flat_prediction(net, layout, variable, x)
    y = torch.randn(2, 54, generator=generator, dtype=theta.dtype)
    values = model.gaussian_nll(pred, y)
    torch.testing.assert_close(values, -torch.distributions.Normal(pred, 2).log_prob(y).sum(-1))
    assert values.shape == (2,)
    torch.testing.assert_close(model.gaussian_nll(pred, pred),
                               pred.new_full((2,), 27 * math.log(8 * math.pi)))
    scores = curvature.observed_scores(net, layout, theta, x, y, chunk_size=1)
    grads = torch.stack([torch.autograd.grad(v, variable, retain_graph=True)[0] for v in values])
    torch.testing.assert_close(scores, -grads, rtol=1e-10, atol=1e-12)
    assert scores.shape == (2, 3376) and not scores.requires_grad
    selected = torch.tensor([3370, 0, 300, 3214, 3268])
    actual = curvature.observed_scores(net, layout, theta, x, y, indices=selected, chunk_size=2)
    torch.testing.assert_close(actual, scores[:, selected], rtol=1e-10, atol=1e-12)
    direction = torch.randn(theta.shape, generator=generator, dtype=theta.dtype)
    direction /= direction.norm()
    epsilon = 1e-5
    plus = model.gaussian_nll(model.flat_prediction(net, layout, theta + epsilon * direction, x), y)
    minus = model.gaussian_nll(model.flat_prediction(net, layout, theta - epsilon * direction, x), y)
    torch.testing.assert_close(scores @ direction, -(plus - minus) / (2 * epsilon),
                               rtol=1e-6, atol=1e-10)


def test_separate_merged_model_prediction_score_and_explicit_ltm_order():
    net, layout, theta, x, _ = case(count=2)
    original = layout.flatten_module(net, detach=True)
    state = torch.get_rng_state().clone()
    merged, merged_layout, merged_theta = model.merge_parameters(net, layout, theta)
    assert torch.equal(state, torch.get_rng_state())
    assert merged is not net and merged_layout.total_numel == 3268
    assert merged_layout.names == layout.names[:-2]
    ltm, stm = model.memory_indices(merged_layout)
    assert len(ltm) == 3268 and not len(stm)
    assert not any("lora" in name for name, _ in merged.named_parameters())
    for source, target, mapping in zip(layout.specs, merged_layout.specs, merged.ltm_mapping):
        assert source.name == target.name == mapping["name"]
        assert (mapping["source_start"], mapping["source_stop"]) == (source.start, source.stop)
        assert (mapping["merged_start"], mapping["merged_stop"]) == (target.start, target.stop)
    expected = model.flat_prediction(net, layout, theta, x)
    torch.testing.assert_close(model.flat_prediction(merged, merged_layout, merged_theta, x), expected,
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(merged(x), expected, rtol=1e-12, atol=1e-12)
    y = torch.ones_like(expected)
    scores = curvature.observed_scores(net, layout, theta, x, y, indices=ltm)
    merged_scores = curvature.observed_scores(merged, merged_layout, merged_theta, x, y)
    torch.testing.assert_close(scores, merged_scores, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(layout.flatten_module(net), original, rtol=0, atol=0)


def test_functional_double_derivatives_leave_float_training_state_untouched(monkeypatch):
    net, layout, theta, x, _ = case(dtype=torch.float32)
    before = layout.flatten_module(net, detach=True)
    seen = []
    predict = curvature.flat_prediction

    def audited(net, layout, parameter, images, **kwargs):
        seen.append((parameter.dtype, images.dtype))
        return predict(net, layout, parameter, images, **kwargs)

    monkeypatch.setattr(curvature, "flat_prediction", audited)
    selected = torch.tensor([0, 100, 3214, 3268])
    z = curvature.fisher_diagonal(net, layout, theta, x, indices=selected, output_chunk_size=9)
    scores = curvature.observed_scores(net, layout, theta, x, torch.ones(1, 54), indices=selected)
    merged, merged_layout, merged_theta = model.merge_parameters(net, layout, theta.double())
    assert merged_theta.dtype == next(merged.parameters()).dtype == torch.float64
    assert seen and set(seen) == {(torch.float64, torch.float64)}
    assert z.dtype == scores.dtype == torch.float64
    assert next(net.parameters()).dtype == torch.float32
    torch.testing.assert_close(layout.flatten_module(net), before, rtol=0, atol=0)


def test_diagonal_output_sum_chunk_invariance_and_label_integration():
    net, layout, theta, x, _ = case(count=2)
    selected = torch.tensor([0, 72, 100, 300, 3214, 3268, 3375])

    def reduced(value):
        full = theta.index_copy(0, selected, value)
        return model.flat_prediction(net, layout, full, x)

    jacobian = jacrev(reduced, chunk_size=3)(theta[selected])
    expected = jacobian.square().sum((0, 1)) / (4 * len(x))
    calls = []
    actual = curvature.fisher_diagonal(net, layout, theta, x, indices=selected,
        chunk_size=1, output_chunk_size=5, check=lambda: calls.append(1))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert len(calls) >= 2 * math.ceil(54 / 5)
    alternate = curvature.fisher_diagonal(net, layout, theta, x, indices=selected,
                                          chunk_size=2, output_chunk_size=7)
    torch.testing.assert_close(actual, alternate, rtol=1e-10, atol=1e-12)
    assert actual[4] == pytest.approx(.25)
    y = model.flat_prediction(net, layout, theta, x).detach()
    scores = curvature.observed_scores(net, layout, theta, x, y, indices=selected)
    assert scores.count_nonzero() == 0 and actual.sum() > 0
    assert actual.ndim == 1 and not actual.requires_grad


def test_joint_endpoint_product_direct_difference_zero_and_head_constant_corrections():
    net, layout, before, x, generator = case()
    selected = torch.tensor([0, 76, 100, 300, 3214, 3268, 3375])
    after = before + .02 * torch.randn(before.shape, generator=generator, dtype=before.dtype)
    delta, zplus, info = curvature.paired_change(net, layout, before, after, x, indices=selected)
    zminus = curvature.fisher_diagonal(net, layout, before, x, indices=selected)
    expected_plus = curvature.fisher_diagonal(net, layout, after, x, indices=selected)
    torch.testing.assert_close(zplus, expected_plus, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(delta, zplus - zminus, rtol=1e-10, atol=1e-12)
    assert info["derivative_dtype"] == "float64" and info["direct_difference_error"] < 1e-12
    assert not info["zero_move"] and delta.abs().max() > 0
    zero, same, diagnostics = curvature.paired_change(net, layout, before, before, x, indices=selected)
    assert torch.equal(zero, torch.zeros_like(zero)) and diagnostics["zero_move"]
    torch.testing.assert_close(same, zminus, rtol=0, atol=0)
    head_move = before.clone()
    head_move[298:3268] += .2
    head_move[3268:] += .1
    head_indices = torch.tensor([298, 400, 3213, 3214, 3267])
    constant, _, _ = curvature.paired_change(net, layout, before, head_move, x, indices=head_indices)
    assert torch.equal(constant, torch.zeros_like(constant))
    reverse, _, _ = curvature.paired_change(net, layout, after, before, x, indices=selected)
    torch.testing.assert_close(reverse, -delta, rtol=0, atol=0)


def test_small_step_paired_change_is_a_finite_change_not_a_normalized_derivative():
    net, layout, theta, x, generator = case()
    direction = torch.randn(theta.shape, generator=generator, dtype=theta.dtype)
    direction /= direction.norm()
    selected = torch.tensor([0, 100, 300, 3268, 3375])
    epsilon = 1e-4
    delta, _, _ = curvature.paired_change(net, layout, theta, theta + epsilon * direction, x,
                                         indices=selected)
    twice, _, _ = curvature.paired_change(net, layout, theta, theta + 2 * epsilon * direction, x,
                                         indices=selected)
    torch.testing.assert_close(twice, 2 * delta, rtol=1e-3, atol=1e-10)
    zplus = curvature.fisher_diagonal(net, layout, theta + epsilon * direction, x, indices=selected)
    zminus = curvature.fisher_diagonal(net, layout, theta - epsilon * direction, x, indices=selected)
    torch.testing.assert_close(delta / epsilon, (zplus - zminus) / (2 * epsilon),
                               rtol=1e-3, atol=1e-10)


def test_fd_values_and_all_block_gradients_match_jvp_at_three_steps():
    net, layout, theta, x, generator = case()
    theta.requires_grad_()
    probes = (2 * torch.randint(2, (2, 108), generator=generator) - 1).to(theta)
    exact = curvature.jvp_values(net, layout, theta, x, probes)
    gradient = torch.autograd.grad(exact.sum(), theta)[0]
    assert exact.shape == (2, 1) and exact.min() > 0
    assert gradient[:298].norm() > 0 and gradient[3268:].norm() > 0
    torch.testing.assert_close(gradient[298:3268], torch.zeros_like(gradient[298:3268]), atol=1e-10, rtol=0)
    for step in (.005, .01, .02):
        actual = curvature.fd_values(net, layout, theta, x, probes, step)
        actual_gradient = torch.autograd.grad(actual.sum(), theta)[0]
        torch.testing.assert_close(actual, exact.detach(), rtol=1e-6, atol=1e-10)
        torch.testing.assert_close(actual_gradient, gradient, rtol=1e-6, atol=1e-10)
    full = theta.new_zeros(2, 3376)
    full[:, 3268:] = probes
    torch.testing.assert_close(curvature.fd_values(net, layout, theta, x, full, .01),
                               curvature.fd_values(net, layout, theta, x, probes, .01), rtol=0, atol=0)


def test_fd_double_conversion_preserves_gradient_to_float_optimizer_variable():
    net, layout, theta, x, _ = case(dtype=torch.float32)
    theta.requires_grad_()
    probes = torch.ones(1, 108)
    values = curvature.fd_values(net, layout, theta.double(), x.double(), probes, .01)
    gradient = torch.autograd.grad(values.sum(), theta)[0]
    assert gradient.dtype == torch.float32 and bool(torch.isfinite(gradient).all())
    assert gradient[:298].norm() > 0 and gradient[3268:].norm() > 0


def test_invalid_derivative_settings_and_chunk_interruptions():
    net, layout, theta, x, _ = case()
    for kwargs in (dict(chunk_size=0), dict(output_chunk_size=0), dict(indices=[]),
                   dict(indices=[-1]), dict(indices=[3376]), dict(indices=[1, 1]),
                   dict(indices=[1.5])):
        with pytest.raises(ValueError):
            curvature.fisher_diagonal(net, layout, theta, x, **kwargs)
    with pytest.raises(ValueError):
        curvature.fd_values(net, layout, theta, x, torch.ones(1, 3376), .01)
    with pytest.raises(ValueError):
        curvature.fd_values(net, layout, theta, x, torch.ones(1, 108), 0)

    class Interrupted(Exception):
        pass

    def stop():
        raise Interrupted

    with pytest.raises(Interrupted):
        curvature.fisher_diagonal(net, layout, theta, x, check=stop)
    with pytest.raises(Interrupted):
        curvature.paired_change(net, layout, theta, theta, x, check=stop)
    with pytest.raises(Interrupted):
        curvature.observed_scores(net, layout, theta, x, torch.ones(1, 54), check=stop)
