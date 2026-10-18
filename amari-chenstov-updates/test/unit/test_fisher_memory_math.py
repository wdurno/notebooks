"""Small CPU audits of the Plan 15 model chart and curvature derivatives."""

import pytest
import torch
from torch.nn import functional as F

from mnist_experiment.fisher_memory.curvature import (
    exact_trace,
    jvp_values,
    kl_values,
    reference_fisher,
    step_kl_values,
)
from mnist_experiment.fisher_memory.model import (
    AFFINE_NAMES,
    build_model,
    effective_residuals,
    flat_logits,
    memory_indices,
)
from mnist_experiment.rotated_mnist.plan12.gauge import build_gauge_fixed_model


@pytest.fixture(scope="module", autouse=True)
def single_threaded_math():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _case(*, moved=True, dtype=torch.float64):
    model, layout = build_model(17, dtype=dtype)
    generator = torch.Generator().manual_seed(93)
    if moved:
        with torch.no_grad():
            for name, value in model.named_parameters():
                if name.endswith(".lora_B"):
                    value.copy_(0.15 * torch.randn(value.shape, generator=generator, dtype=dtype))
    theta = layout.flatten_module(model, detach=True).requires_grad_()
    x = torch.randn(2, 1, 28, 28, generator=generator, dtype=dtype)
    ltm, stm = memory_indices(layout)
    probes = (2 * torch.randint(2, (2, stm.numel()), generator=generator) - 1).to(dtype)
    return model, layout, theta, x, ltm, stm, probes


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_parameter_counts_chart_and_bias_ownership(rank):
    model, layout = build_model(3, rank=rank)
    ltm, stm = memory_indices(layout)
    assert ltm.numel() == 487
    assert stm.numel() == 88 * rank
    assert layout.total_numel == 487 + 88 * rank
    if rank == 2:
        assert (ltm.numel(), stm.numel(), layout.total_numel) == (487, 176, 663)
    assert ltm.dtype == stm.dtype == torch.int64
    assert ltm.device.type == stm.device.type == "cpu"
    assert torch.equal(torch.cat((ltm, stm)).sort().values, torch.arange(layout.total_numel))
    for spec in layout.specs:
        if spec.name.endswith(".bias"):
            assert set(range(spec.start, spec.stop)) <= set(ltm.tolist())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_seed_pairing_initial_base_equivalence_and_helmert_chart():
    rng = torch.random.get_rng_state().clone()
    model, layout = build_model(4, dtype=torch.float64)
    twin, _ = build_model(4, dtype=torch.float64)
    base, _ = build_gauge_fixed_model(4, dtype=torch.float64)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.factor_seed == twin.factor_seed
    assert torch.equal(layout.flatten_module(model), layout.flatten_module(twin))
    x = torch.randn(2, 1, 28, 28, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    torch.testing.assert_close(model(x), base(x), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(model(x, stm_scale=0), base(x), rtol=1e-14, atol=1e-14)
    contrast = model.classifier.contrast
    torch.testing.assert_close(contrast.mT @ contrast, torch.eye(9, dtype=torch.float64))
    torch.testing.assert_close(model(x).sum(-1), torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
    for name, value in base.named_parameters():
        torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_build_model_preserves_initialized_cpu_and_cuda_rng_states(device):
    torch.cuda.init()
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(310)
        torch.rand(3)
        for index in devices:
            torch.rand(3, device=f"cuda:{index}")
        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
        build_model(4, factor_seed=71, device=device)
        assert torch.equal(cpu_before, torch.random.get_rng_state())
        cuda_after = torch.cuda.get_rng_state_all()
        assert len(cuda_before) == len(cuda_after)
        assert all(torch.equal(before, after) for before, after in zip(cuda_before, cuda_after))


def test_initialization_can_learn_without_zero_a_and_b_dead_branch():
    model, layout, theta, x, ltm, stm, _ = _case(moved=False)
    for name, value in model.named_parameters():
        if name.endswith(".lora_A"):
            assert torch.count_nonzero(value) == value.numel()
            bound = value.shape[1] ** -0.5
            assert value.abs().max() <= bound
        if name.endswith(".lora_B"):
            assert torch.count_nonzero(value) == 0
    labels = torch.tensor([1, 9])
    loss = F.cross_entropy(flat_logits(model, layout, theta, x), labels)
    gradient = torch.autograd.grad(loss, theta)[0]
    assert gradient[ltm].norm() > 0
    assert gradient[stm].norm() > 0
    for spec in layout.specs:
        if spec.name.endswith(".lora_B"):
            assert gradient[spec.start:spec.stop].norm() > 0
        if spec.name.endswith(".lora_A"):
            assert gradient[spec.start:spec.stop].count_nonzero() == 0
    updated = (theta - 0.1 * gradient).detach().requires_grad_()
    next_loss = F.cross_entropy(flat_logits(model, layout, updated, x), labels)
    assert next_loss < loss
    next_gradient = torch.autograd.grad(next_loss, updated)[0]
    for spec in layout.specs:
        if spec.name.endswith(".lora_A"):
            assert next_gradient[spec.start:spec.stop].norm() > 0


def test_explicit_factor_seed_is_independent_and_ltm_only_pretraining_has_no_stm_gradient():
    first, layout = build_model(4, factor_seed=71, dtype=torch.float64)
    second, _ = build_model(4, factor_seed=72, dtype=torch.float64)
    ltm, stm = memory_indices(layout)
    left = layout.flatten_module(first, detach=True).requires_grad_()
    right = layout.flatten_module(second, detach=True)
    assert first.factor_seed == 71 and second.factor_seed == 72
    torch.testing.assert_close(left[ltm], right[ltm], rtol=0, atol=0)
    assert not torch.equal(left[stm], right[stm])
    x = torch.randn(2, 1, 28, 28, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    torch.testing.assert_close(first(x), second(x), rtol=0, atol=0)
    loss = F.cross_entropy(flat_logits(first, layout, left, x, stm_scale=0), torch.tensor([1, 9]))
    gradient = torch.autograd.grad(loss, left)[0]
    assert gradient[ltm].norm() > 0 and gradient[stm].count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_pooling_matches_base_forward_and_ltm_input_backward(dtype):
    model, _ = build_model(4, dtype=dtype)
    base, _ = build_gauge_fixed_model(4, dtype=dtype)
    assert isinstance(model.features[5], torch.nn.AvgPool2d)
    assert model.features[5].kernel_size == model.features[5].stride == 7
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(2, 1, 28, 28, generator=generator, dtype=dtype, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    expected, actual = base(other_x), model(x)
    torch.testing.assert_close(actual, expected)
    targets = torch.tensor([1, 9])
    F.cross_entropy(actual, targets).backward()
    F.cross_entropy(expected, targets).backward()
    torch.testing.assert_close(x.grad, other_x.grad)
    parameters = dict(model.named_parameters())
    for name, parameter in base.named_parameters():
        torch.testing.assert_close(parameters[name].grad, parameter.grad)
    with pytest.raises(ValueError, match="28, 28"):
        model(torch.zeros(2, 1, 30, 30, dtype=dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_strict_deterministic_pooling_forward_backward_and_curvature(dtype, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        model, layout = build_model(4, device="cuda", dtype=dtype)
        theta = layout.flatten_module(model, detach=True).requires_grad_()
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(2, 1, 28, 28, generator=generator, dtype=dtype).cuda()
        targets = torch.tensor([1, 9], device="cuda")
        _, stm = memory_indices(layout)
        probes = torch.ones(2, len(stm), dtype=dtype, device="cuda")
        first = None
        for _ in range(2):
            logits = flat_logits(model, layout, theta, x)
            objective = F.cross_entropy(logits, targets) + jvp_values(model, layout, theta, x, probes).mean()
            gradient = torch.autograd.grad(objective, theta)[0]
            assert torch.isfinite(gradient).all() and gradient.norm() > 0
            if first is None:
                first = logits.detach().clone(), gradient.detach().clone()
            else:
                assert torch.equal(logits, first[0])
                assert torch.equal(gradient, first[1])
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


def test_functional_logits_scale_all_three_residuals_without_mutation():
    model, layout, theta, x, _, _, _ = _case()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch.testing.assert_close(flat_logits(model, layout, theta, x), model(x))
    assert not torch.allclose(model(x), model(x, stm_scale=0))
    zero_residual = theta.detach().clone()
    for spec in layout.specs:
        if spec.name.endswith(".lora_B"):
            zero_residual[spec.start:spec.stop] = 0
    torch.testing.assert_close(
        flat_logits(model, layout, theta, x, stm_scale=0),
        flat_logits(model, layout, zero_residual, x), rtol=0, atol=0,
    )
    for scale in (0.0, 0.5, 0.9, 1.0):
        gradient = torch.autograd.grad(flat_logits(model, layout, theta, x, stm_scale=scale).square().sum(), theta)[0]
        assert torch.isfinite(gradient).all()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_jvp_matches_exact_per_sample_model_label_quadratic_forms():
    model, layout, theta, x, _, stm, probes = _case()
    actual = jvp_values(model, layout, theta, x, probes)
    assert actual.shape == (2, 2)
    expected = []
    for image in x:
        fisher = reference_fisher(model, layout, theta, image[None], indices=stm)
        expected.append(torch.einsum("ki,ij,kj->k", probes, fisher, probes))
        assert torch.linalg.eigvalsh(fisher).min() > -1e-12
    torch.testing.assert_close(actual, torch.stack(expected, dim=1), rtol=1e-10, atol=1e-12)
    full_probes = theta.new_zeros(2, theta.numel()).index_copy(1, stm, probes)
    torch.testing.assert_close(jvp_values(model, layout, theta, x, full_probes), actual, rtol=0, atol=0)


def test_reference_is_weighted_model_label_mean_not_observed_label_outer_product():
    model, layout, theta, x, _, stm, _ = _case()
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64)
    reference = reference_fisher(model, layout, theta, x, indices=stm, weights=weights, chunk_size=1)
    expected = torch.zeros_like(reference)
    for i, image in enumerate(x):
        logp = flat_logits(model, layout, theta, image[None])[0].log_softmax(-1)
        for label in range(10):
            score = torch.autograd.grad(logp[label], theta, retain_graph=True)[0][stm]
            expected += (weights[i] / weights.sum()) * logp[label].exp().detach() * torch.outer(score, score)
    torch.testing.assert_close(reference, expected, rtol=1e-10, atol=1e-12)
    assert reference.dtype == torch.float64 and not reference.requires_grad
    traces = exact_trace(model, layout, theta, x, stm, chunk_size=1)
    torch.testing.assert_close((traces * weights / weights.sum()).sum(), reference.trace())
    torch.testing.assert_close(
        reference_fisher(model, layout, theta, x, indices=stm, weights=weights, chunk_size=2),
        reference, rtol=1e-10, atol=1e-12,
    )


def test_trace_matches_full_coordinate_diagonal_and_coordinate_probe_sum():
    model, layout, theta, x, ltm, stm, _ = _case()
    x = x[:1]
    fisher = reference_fisher(model, layout, theta, x)
    torch.testing.assert_close(exact_trace(model, layout, theta, x)[0], fisher.trace())
    for indices in (ltm, stm):
        torch.testing.assert_close(
            exact_trace(model, layout, theta, x, indices)[0], fisher.diagonal()[indices].sum()
        )
    # A complete coordinate basis audits trace without a noisy Monte Carlo test.
    values = jvp_values(model, layout, theta, x, torch.eye(stm.numel(), dtype=theta.dtype))
    torch.testing.assert_close(values.sum(0), exact_trace(model, layout, theta, x, stm))


def test_symmetric_kl_matches_declared_stencil_and_converges_to_jvp():
    model, layout, theta, x, _, stm, probes = _case()
    h = 0.02
    full = theta.new_zeros(theta.numel()).index_copy(0, stm, probes[0])
    plus = flat_logits(model, layout, theta + h * full / 2, x).log_softmax(-1)
    minus = flat_logits(model, layout, theta - h * full / 2, x).log_softmax(-1)
    declared = ((plus.exp() * (plus - minus)).sum(-1)
                + (minus.exp() * (minus - plus)).sum(-1)) / h**2
    torch.testing.assert_close(kl_values(model, layout, theta, x, probes, h)[0], declared, rtol=1e-8, atol=1e-11)
    exact = jvp_values(model, layout, theta, x, probes)
    coarse = kl_values(model, layout, theta, x, probes, 0.02)
    fine = kl_values(model, layout, theta, x, probes, 0.01)
    assert (fine - exact).norm() < 0.3 * (coarse - exact).norm()
    torch.testing.assert_close(kl_values(model, layout, theta, x, probes, 0.001), exact, rtol=2e-4, atol=1e-9)


@pytest.mark.parametrize("kind", ["jvp", "kl", "step", "trace"])
def test_penalty_gradients_into_both_blocks_agree_with_finite_differences(kind):
    model, layout, theta, x, ltm, stm, probes = _case()
    delta = 0.02 * probes[0] / probes[0].norm()

    def objective(parameter):
        if kind == "jvp":
            return jvp_values(model, layout, parameter, x, probes).mean()
        if kind == "kl":
            return kl_values(model, layout, parameter, x, probes, 0.01).mean()
        if kind == "step":
            return step_kl_values(model, layout, parameter, x, delta).mean()
        return exact_trace(model, layout, parameter, x, stm).mean()

    gradient = torch.autograd.grad(objective(theta), theta)[0]
    assert torch.isfinite(gradient).all()
    for indices in (ltm, stm):
        assert gradient[indices].norm() > 1e-8
        direction = torch.zeros_like(theta)
        direction[indices] = gradient[indices] / gradient[indices].norm()
        epsilon = 2e-5
        numerical = (objective(theta.detach() + epsilon * direction)
                     - objective(theta.detach() - epsilon * direction)) / (2 * epsilon)
        torch.testing.assert_close(gradient @ direction, numerical, rtol=2e-4, atol=2e-8)


def test_kl_gradient_converges_to_jvp_gradient_in_both_blocks():
    model, layout, theta, x, ltm, stm, probes = _case()
    expected = torch.autograd.grad(jvp_values(model, layout, theta, x, probes).mean(), theta)[0]
    actual = torch.autograd.grad(kl_values(model, layout, theta, x, probes, 0.001).mean(), theta)[0]
    for indices in (ltm, stm):
        assert (actual[indices] - expected[indices]).norm() / expected[indices].norm() < 1e-4


def test_step_kl_detaches_direction_but_not_either_predictive_branch():
    model, layout, theta, x, ltm, stm, probes = _case()
    delta = (0.02 * probes[0] / probes[0].norm()).requires_grad_()
    values = step_kl_values(model, layout, theta, x, delta)
    gradient, delta_gradient = torch.autograd.grad(values.mean(), (theta, delta), allow_unused=True)
    assert delta_gradient is None
    assert gradient[ltm].norm() > 0 and gradient[stm].norm() > 0
    full = theta.new_zeros(theta.numel()).index_copy(0, stm, delta.detach())
    base = flat_logits(model, layout, theta, x).log_softmax(-1)
    moved = flat_logits(model, layout, theta + full, x).log_softmax(-1)
    declared = 2 * (base.exp() * (base - moved)).sum(-1) / delta.detach().square().sum()
    torch.testing.assert_close(values, declared, rtol=1e-8, atol=1e-11)
    torch.testing.assert_close(step_kl_values(model, layout, theta, x, full), values)
    wrong = 2 * (base.detach().exp() * (base.detach() - moved)).sum(-1) / delta.detach().square().sum()
    wrong_gradient = torch.autograd.grad(wrong.mean(), theta)[0]
    assert (wrong_gradient - gradient).norm() > gradient.norm()


def test_step_kl_small_direction_matches_normalized_fisher_and_zero_rule():
    model, layout, theta, x, _, stm, probes = _case()
    direction = probes[0] / probes[0].norm()
    expected = jvp_values(model, layout, theta, x, direction[None])[0]
    actual = step_kl_values(model, layout, theta, x, 0.001 * direction)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-8)
    for delta in (theta.new_zeros(stm.numel()), 1e-14 * direction):
        zero = step_kl_values(model, layout, theta, x, delta)
        assert torch.equal(zero, torch.zeros_like(zero))
        assert torch.count_nonzero(torch.autograd.grad(zero.sum(), theta)[0]) == 0


def test_tiny_nonzero_step_kl_does_not_amplify_roundoff_into_huge_curvature():
    model, layout, theta, x, _, _, probes = _case()
    direction = probes[0] / probes[0].norm()
    expected = jvp_values(model, layout, theta, x, direction[None])[0]
    for norm in (1e-6, 1e-8, 1e-10):
        actual = step_kl_values(model, layout, theta, x, norm * direction)
        assert (actual >= 0).all()
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-8)


def test_factor_rescaling_changes_trace_but_not_effective_weights_or_reliance():
    model, layout, theta, x, _, stm, _ = _case()
    scaled = theta.detach().clone()
    for spec in layout.specs:
        if spec.name.endswith(".lora_A"):
            scaled[spec.start:spec.stop] *= 3
        if spec.name.endswith(".lora_B"):
            scaled[spec.start:spec.stop] /= 3
    residuals = effective_residuals(model, layout, theta)
    other = effective_residuals(model, layout, scaled)
    assert tuple(residuals) == AFFINE_NAMES
    assert [tuple(value.shape) for value in residuals.values()] == [(4, 9), (6, 36), (9, 24)]
    for name in AFFINE_NAMES:
        torch.testing.assert_close(residuals[name], other[name], rtol=1e-12, atol=1e-14)
    for scale in (0.0, 0.5, 0.9, 1.0):
        torch.testing.assert_close(
            flat_logits(model, layout, theta, x, stm_scale=scale),
            flat_logits(model, layout, scaled, x, stm_scale=scale), rtol=1e-12, atol=1e-14,
        )
    original_trace = exact_trace(model, layout, theta, x, stm)
    scaled_trace = exact_trace(model, layout, scaled, x, stm)
    assert not torch.allclose(original_trace, scaled_trace, rtol=0.1, atol=0)


def test_float32_model_uses_double_accumulation_and_remains_differentiable():
    model, layout, theta, x, ltm, stm, probes = _case(dtype=torch.float32)
    for value in (
        jvp_values(model, layout, theta, x, probes),
        kl_values(model, layout, theta, x, probes, 0.01),
        step_kl_values(model, layout, theta, x, 0.01 * probes[0]),
        exact_trace(model, layout, theta, x, stm),
    ):
        assert value.dtype == torch.float64 and torch.isfinite(value).all()
        gradient = torch.autograd.grad(value.mean(), theta)[0]
        assert gradient.dtype == torch.float32
        assert gradient[ltm].norm() > 0 and gradient[stm].norm() > 0
    fisher = reference_fisher(model, layout, theta, x, indices=stm)
    assert fisher.dtype == torch.float64 and torch.isfinite(fisher).all()
    assert torch.isfinite(flat_logits(model, layout, theta.double(), x.double())).all()


@pytest.mark.parametrize("rank", [0, -1, True, 1.5])
def test_invalid_rank_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        build_model(0, rank=rank)


def test_invalid_directions_indices_weights_and_step_rejected():
    model, layout, theta, x, ltm, stm, probes = _case()
    full = theta.new_zeros(1, theta.numel())
    full[0, ltm[0]] = 1
    with pytest.raises(ValueError, match="zero in LTM"):
        jvp_values(model, layout, theta, x, full)
    with pytest.raises(ValueError, match="zero in LTM"):
        step_kl_values(model, layout, theta, x, full[0])
    with pytest.raises(ValueError, match="STM-only or full"):
        jvp_values(model, layout, theta, x, probes[:, :-1])
    with pytest.raises(ValueError, match="nonempty"):
        jvp_values(model, layout, theta, x, probes[:0])
    with pytest.raises(ValueError, match="finite"):
        jvp_values(model, layout, theta, x, torch.full_like(probes, float("nan")))
    for h in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="coordinate step"):
            kl_values(model, layout, theta, x, probes, h)
    for indices in (stm[:2].double(), stm[:1].repeat(2), torch.tensor([-1])):
        with pytest.raises(ValueError, match="indices"):
            exact_trace(model, layout, theta, x, indices)
    for weights in ([-1.0, 1.0], [0.0, 0.0], [1.0], [float("nan"), 1.0]):
        with pytest.raises(ValueError, match="weights"):
            reference_fisher(model, layout, theta, x, indices=stm, weights=weights)
    with pytest.raises(ValueError, match="nonempty"):
        exact_trace(model, layout, theta, x[:0], stm)
    with pytest.raises(ValueError, match="chart"):
        memory_indices(build_gauge_fixed_model(0)[1])
