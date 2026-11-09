"""Bounded CPU acceptance fixtures for the Plan 19 v1 numerical contract."""

import math

import pytest
import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal, kl_divergence

from src.continual import picar_math as pm


DTYPE = torch.float64


def tensor(values, **kwargs):
    return torch.tensor(values, dtype=DTYPE, **kwargs)


def design(batch=3, dtype=DTYPE, **overrides):
    lower = torch.eye(7, dtype=dtype)
    lower[1, 0] = 0.25
    lower[3, 2] = -0.15
    values = dict(bin_values=torch.tensor([-0.8, 0.0, 0.9], dtype=dtype),
                  baseline=torch.linspace(-0.2, 0.3, batch, dtype=dtype),
                  temperature=1.3, sigma_action=lower @ lower.T)
    values.update(overrides)
    return pm.LikelihoodDesign(**values)


class TinyModel(nn.Module):
    """Shared nonlinear features make both-bank FD gradients nontrivial."""

    def __init__(self, dtype=DTYPE):
        super().__init__()
        self.ltm = nn.Parameter(torch.tensor([0.3, -0.2], dtype=dtype))
        self.stm = nn.Parameter(torch.tensor([0.4, 0.15], dtype=dtype))
        self.register_buffer("loadings", torch.linspace(-0.7, 0.8, 7, dtype=dtype))

    def forward(self, histories):
        x = histories["features"]
        hidden = torch.tanh(x @ self.ltm + (x @ self.stm).square())
        return {"return_logits": torch.stack((hidden, 0.4 * hidden + self.stm[0],
                                               -0.7 * hidden), dim=1),
                "action_mean": (hidden[:, None] * self.loadings
                                + self.ltm[1] * self.stm[1])}


def fixture(dtype=DTYPE):
    model = TinyModel(dtype)
    histories = {"features": torch.tensor([[0.4, 0.8], [-0.5, 0.7], [0.9, -0.2]], dtype=dtype)}
    predict = lambda params: pm.functional_predict(model, params, (histories,))
    return model, dict(model.named_parameters()), predict, design(dtype=dtype)


def test_nll_matches_normalized_distributions_and_detaches_all_fixed_inputs():
    model, params, predict, fixed = fixture()
    bins = fixed.bin_values.clone().requires_grad_()
    baseline = fixed.baseline.clone().requires_grad_()
    temperature = tensor(1.3, requires_grad=True)
    covariance = fixed.sigma_action.clone().requires_grad_()
    fixed = design(bin_values=bins, baseline=baseline, temperature=temperature,
                   sigma_action=covariance)
    output = predict(params)
    labels = torch.tensor([0, 2, 1])
    actions = torch.linspace(-0.5, 0.8, 21, dtype=DTYPE).reshape(3, 7).requires_grad_()
    weights = fixed.weights[torch.arange(3), labels]
    reference = (-Categorical(logits=output["return_logits"]).log_prob(labels)
                 - MultivariateNormal(output["action_mean"], covariance_matrix=(
                     fixed.sigma_action / weights[:, None, None])).log_prob(actions))
    actual = pm.joint_nll(output, labels, actions, fixed)
    torch.testing.assert_close(actual, reference, atol=2e-14, rtol=2e-14)
    actual.mean().backward()
    assert all(value.grad is None for value in (bins, baseline, temperature, covariance, actions))
    assert all(value.grad is not None for value in params.values())
    before = fixed.weights.clone()
    with torch.no_grad():
        baseline.add_(3)
        bins.add_(1)
        covariance.mul_(2)
    torch.testing.assert_close(fixed.weights, before)
    assert not fixed.weights.requires_grad
    assert model.ltm.grad.abs().sum() > 0


def test_exact_directed_kl_matches_conditional_distribution_sum():
    _, params, predict, fixed = fixture()
    left = predict(params)
    right = predict({name: value + 0.4 for name, value in params.items()})
    categorical_left = Categorical(logits=left["return_logits"])
    categorical_right = Categorical(logits=right["return_logits"])
    reference = kl_divergence(categorical_left, categorical_right)
    for j in range(3):
        covariance = fixed.sigma_action / fixed.weights[:, j, None, None]
        reference = reference + categorical_left.probs[:, j] * kl_divergence(
            MultivariateNormal(left["action_mean"], covariance_matrix=covariance),
            MultivariateNormal(right["action_mean"], covariance_matrix=covariance))
    actual = pm.directed_kl(left, right, fixed)
    torch.testing.assert_close(actual, reference, atol=1e-14, rtol=1e-13)
    reverse = pm.directed_kl(right, left, fixed)
    assert not torch.allclose(actual, reverse)
    torch.testing.assert_close(pm.symmetric_kl(left, right, fixed), actual + reverse)
    torch.testing.assert_close(pm.directed_kl(left, left, fixed), torch.zeros(3, dtype=DTYPE))


def test_model_sampling_is_detached_reproducible_and_does_not_touch_global_rng():
    _, params, predict, fixed = fixture()
    output = predict(params)
    global_state = torch.random.get_rng_state().clone()
    first = pm.sample_model_labels(output, fixed, generator=torch.Generator().manual_seed(42))
    second = pm.sample_model_labels(output, fixed, generator=torch.Generator().manual_seed(42))
    assert torch.equal(global_state, torch.random.get_rng_state())
    for a, b in zip(first, second):
        assert not a.requires_grad
        torch.testing.assert_close(a, b)


def test_functional_predict_snapshots_even_unspecified_buffers():
    class BufferModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", tensor(0.))
            self.register_buffer("offset", tensor(1.))

        def forward(self):
            self.counter.add_(1)
            return self.counter + self.offset

    model = BufferModel()
    result = pm.functional_predict(model, {}, buffers={"offset": tensor(3.)})
    assert result == 4
    assert model.counter == 0 and model.offset == 1
    with pytest.raises(ValueError, match="buffer"):
        pm.functional_predict(model, {}, buffers={"missing": tensor(3.)})


def test_directed_kl_matches_model_sample_log_ratio():
    # Vectorized label sampling is cheap and does not differentiate a large batch.
    count = 12000
    fixed = design(batch=count)
    source = {"return_logits": tensor([0.1, -0.4, 0.8]).expand(count, -1),
              "action_mean": torch.zeros(count, 7, dtype=DTYPE)}
    target = {"return_logits": tensor([-0.3, 0.2, 0.4]).expand(count, -1),
              "action_mean": torch.full((count, 7), 0.2, dtype=DTYPE)}
    labels, actions = pm.sample_model_labels(source, fixed, generator=torch.Generator().manual_seed(18))
    log_ratio = (pm.joint_nll(target, labels, actions, fixed)
                 - pm.joint_nll(source, labels, actions, fixed))
    assert abs(log_ratio.mean() - pm.directed_kl(source, target, fixed).mean()) < 0.025


def test_per_record_scores_have_negative_nll_sign_and_not_batchmean_square():
    params = {"mu": tensor(0.0, requires_grad=True)}
    fixed = design(batch=2, bin_values=tensor([-1.0, 0.0, 1.0]),
                   baseline=torch.zeros(2, dtype=DTYPE), temperature=1.0,
                   sigma_action=torch.eye(7, dtype=DTYPE))

    def predict(p):
        return {"return_logits": torch.zeros(2, 3, dtype=DTYPE),
                "action_mean": p["mu"] * tensor([1, 0, 0, 0, 0, 0, 0]).expand(2, -1)}

    actions = torch.zeros(2, 7, dtype=DTYPE)
    actions[:, 0] = tensor([1, -1])
    labels = torch.ones(2, dtype=torch.long)
    scores = pm.per_record_scores(predict, params, labels, actions, fixed)["mu"]
    torch.testing.assert_close(scores, tensor([1, -1]))
    assert scores.square().mean() == 1
    assert scores.mean().square() == 0
    nll = pm.joint_nll(predict(params), labels, actions, fixed)
    for i in range(2):
        gradient, = torch.autograd.grad(nll[i], params["mu"], retain_graph=True)
        torch.testing.assert_close(gradient, -scores[i])
    assert params["mu"].grad is None


def test_exact_fisher_equals_tiny_local_kl_hessian_and_supports_ltm_only():
    _, params, predict, fixed = fixture()
    source = {name: value.detach() for name, value in predict(params).items()}
    flat = torch.cat([params["ltm"], params["stm"]]).detach()

    def kl_from_vector(vector):
        return pm.directed_kl(source, predict({"ltm": vector[:2], "stm": vector[2:]}), fixed).mean()

    # Four coordinates only: this is a reference, never a production Hessian.
    reference = torch.autograd.functional.hessian(kl_from_vector, flat).diagonal()
    exact = pm.model_fisher_diagonal(predict, params, fixed)
    torch.testing.assert_close(torch.cat(list(exact.values())), reference, atol=1e-13, rtol=1e-12)
    subset = pm.model_fisher_diagonal(predict, {"ltm": params["ltm"]}, fixed)
    assert set(subset) == {"ltm"}
    torch.testing.assert_close(subset["ltm"], exact["ltm"])
    assert all(not value.requires_grad for value in exact.values())
    assert all(value.grad is None for value in params.values())


def test_sampled_fisher_agrees_with_exact_and_streams_per_record_squares():
    _, params, predict, fixed = fixture()
    exact = pm.model_fisher_diagonal(predict, params, fixed)
    sampled = pm.model_fisher_diagonal(predict, params, fixed, method="sampled", draws=512,
                                       generator=torch.Generator().manual_seed(71))
    for name in params:
        torch.testing.assert_close(sampled[name], exact[name], rtol=0.15, atol=0.008)
    # One draw must equal the explicit mean of recordwise score squares.
    labels, actions = pm.sample_model_labels(predict(params), fixed,
                                             generator=torch.Generator().manual_seed(7))
    scores = pm.per_record_scores(predict, params, labels, actions, fixed)
    one = pm.model_fisher_diagonal(predict, params, fixed, method="sampled",
                                   generator=torch.Generator().manual_seed(7))
    for name in params:
        torch.testing.assert_close(one[name], scores[name].square().mean(0))


@pytest.mark.parametrize("draws", [1, 2, 4])
def test_sampled_fisher_averages_all_draws_and_records(draws):
    _, params, predict, fixed = fixture()
    generator = torch.Generator().manual_seed(29)
    expected = {name: torch.zeros_like(value) for name, value in params.items()}
    for _ in range(draws):
        labels, actions = pm.sample_model_labels(predict(params), fixed, generator=generator)
        scores = pm.per_record_scores(predict, params, labels, actions, fixed)
        for name in params:
            expected[name] += scores[name].square().mean(0) / draws
    actual = pm.model_fisher_diagonal(predict, params, fixed, method="sampled", draws=draws,
                                      generator=torch.Generator().manual_seed(29))
    for name in params:
        torch.testing.assert_close(actual[name], expected[name])


def test_unused_bank_parameters_have_zero_scores_and_fisher():
    params = {"unused": tensor([1.], requires_grad=True)}
    fixed = design(batch=1)
    predict = lambda _: {"return_logits": tensor([[0., 0., 0.]]),
                         "action_mean": torch.zeros(1, 7, dtype=DTYPE)}
    scores = pm.per_record_scores(predict, params, torch.tensor([1]),
                                  torch.zeros(1, 7, dtype=DTYPE), fixed)
    assert scores["unused"].shape == (1, 1)
    assert scores["unused"].item() == 0
    for method in ("exact", "sampled"):
        diagonal = pm.model_fisher_diagonal(predict, params, fixed, method=method,
                                            generator=torch.Generator().manual_seed(21))
        assert diagonal["unused"].item() == 0


def probes():
    return tuple({"stm": tensor(v)} for v in ([1, 1], [1, -1], [-1, 1], [-1, -1]))


def test_fd_trace_normalization_age_weighting_and_step_convergence():
    _, params, predict, fixed = fixture()
    exact = pm.model_fisher_diagonal(predict, {"stm": params["stm"]}, fixed)["stm"].mean()
    penalties = [pm.stm_fd_penalty(predict, params, fixed, ["stm"], probes(),
                                   torch.ones(3, dtype=DTYPE), step)
                 for step in (0.02, 0.01, 0.005)]
    errors = [abs(penalty.detach() - exact).item() for penalty in penalties]
    assert errors[2] < errors[1] < errors[0]
    torch.testing.assert_close(penalties[-1], exact, rtol=2e-5, atol=1e-7)
    half_age = pm.stm_fd_penalty(predict, params, fixed, ["stm"], probes(),
                                torch.full((3,), 0.5, dtype=DTYPE), 0.005)
    torch.testing.assert_close(half_age, penalties[-1] / 4)
    zero = pm.stm_fd_penalty(predict, params, fixed, ["stm"], probes(),
                            torch.zeros(3, dtype=DTYPE), 0.005)
    zero.backward()
    assert zero == 0
    assert all(torch.equal(value.grad, torch.zeros_like(value)) for value in params.values())


def test_fd_nonuniform_ages_use_record_mean_not_normalized_age_weights():
    _, params, predict, fixed = fixture()
    ages = tensor([0., 0.2, 1.])
    expected = tensor(0.)
    for index in range(3):
        one_design = pm.LikelihoodDesign(fixed.bin_values[index:index + 1],
            fixed.baseline[index:index + 1], fixed.temperature[index:index + 1],
            fixed.sigma_action[index:index + 1])

        def one_predict(p):
            return {name: value[index:index + 1] for name, value in predict(p).items()}

        fisher = pm.model_fisher_diagonal(one_predict, {"stm": params["stm"]}, one_design)
        expected += ages[index].square() * fisher["stm"].mean() / 3
    actual = pm.stm_fd_penalty(predict, params, fixed, ["stm"], probes(), ages, 0.002)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)


def test_fd_both_endpoint_and_ltm_gradients_match_numerical_derivatives():
    model, params, predict, fixed = fixture()
    ages = tensor([0.2, 0.6, 1.0], requires_grad=True)
    frozen_probes = ({"stm": tensor([1.0, -1.0], requires_grad=True)},)
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def penalty(p):
        return pm.stm_fd_penalty(predict, p, fixed, ["stm"], frozen_probes, ages, 0.015)

    actual = penalty(params)
    actual.backward()
    for name, value in params.items():
        reference = torch.zeros_like(value)
        for index in range(value.numel()):
            delta = torch.zeros_like(value)
            delta[index] = 2e-5
            plus, minus = dict(params), dict(params)
            plus[name], minus[name] = value + delta, value - delta
            reference[index] = (penalty(plus) - penalty(minus)).detach() / 4e-5
        assert reference.abs().sum() > 1e-3
        torch.testing.assert_close(value.grad, reference, rtol=2e-5, atol=2e-7)
    assert ages.grad is None and frozen_probes[0]["stm"].grad is None
    for name, before in state_before.items():
        torch.testing.assert_close(model.state_dict()[name], before)


def test_probe_rng_and_age_clock_contract():
    _, params, _, _ = fixture()
    global_state = torch.random.get_rng_state().clone()
    first = pm.rademacher_probes(params, ["stm"], 4, generator=torch.Generator().manual_seed(3))
    second = pm.rademacher_probes(params, ["stm"], 4, generator=torch.Generator().manual_seed(3))
    for a, b in zip(first, second):
        torch.testing.assert_close(a["stm"], b["stm"])
        assert bool((a["stm"].abs() == 1).all())
        assert a["stm"].square().sum() == 2  # Not normalized to a unit vector.
    assert torch.equal(global_state, torch.random.get_rng_state())
    torch.testing.assert_close(pm.replay_ages(12, torch.tensor([12, 10, 0]), 4), tensor([0, 0.5, 1]))


def objective_inputs():
    return dict(params={"ltm": tensor([2., 4.], requires_grad=True),
                        "stm": tensor([3.], requires_grad=True)},
                ltm_names=["ltm"], stm_names=["stm"],
                ltm_anchor={"ltm": tensor([1., 2.], requires_grad=True)},
                stm_anchor={"stm": tensor([1.], requires_grad=True)},
                fisher_diagonal={"ltm": tensor([2., 6.], requires_grad=True)},
                pi=0.25, ridge_coefficient=0.1, numerical_floor=0.5,
                new_nll=tensor([1., 3.]), replay_nll=tensor([5., 5., 5.]),
                fd_penalty=tensor(2., requires_grad=True), fd_strength=0.2)


def test_objective_exact_values_gradients_and_frozen_anchors():
    values = objective_inputs()
    result = pm.regularized_objective(**values)
    expected = dict(data=3.5, ewc=39.0, ridge=5.4, fd=0.4, fisher_scale=4., kappa=0.4, total=48.3)
    for name, value in expected.items():
        assert result[name].item() == pytest.approx(value)
    result["total"].backward()
    torch.testing.assert_close(values["params"]["ltm"].grad, tensor([7.2, 38.4]))
    torch.testing.assert_close(values["params"]["stm"].grad, tensor([2.4]))
    assert values["fd_penalty"].grad.item() == pytest.approx(0.2)
    for name in ("ltm_anchor", "stm_anchor", "fisher_diagonal"):
        assert all(value.grad is None for value in values[name].values())


def test_complete_likelihood_fd_ewc_objective_gradient_reference():
    """Learner-wrapper reference: captured histories, bank maps, frozen event state."""
    _, params, predict, fixed = fixture()
    ltm_anchor = {"ltm": params["ltm"].detach().clone() - 0.05}
    stm_anchor = {"stm": params["stm"].detach().clone()}
    diagonal = pm.model_fisher_diagonal(predict, {"ltm": params["ltm"]}, fixed)
    event_probes = pm.rademacher_probes(params, ["stm"], 2,
                                       generator=torch.Generator().manual_seed(12))
    ages = pm.replay_ages(6, torch.tensor([0, 2, 5]), 6)
    actions = torch.linspace(-0.3, 0.4, 21, dtype=DTYPE).reshape(3, 7)

    def objective(p):
        nll = pm.joint_nll(predict(p), torch.tensor([0, 1, 2]), actions, fixed)
        fd = pm.stm_fd_penalty(predict, p, fixed, ["stm"], event_probes, ages, 0.02)
        return pm.regularized_objective(p, ltm_names=["ltm"], stm_names=["stm"],
            ltm_anchor=ltm_anchor, stm_anchor=stm_anchor, fisher_diagonal=diagonal,
            replay_nll=nll, pi=0.025, ridge_coefficient=0.01,
            numerical_floor=1e-6, fd_penalty=fd, fd_strength=0.3)["total"]

    objective(params).backward()
    for name, parameter in params.items():
        for index in range(parameter.numel()):
            delta = torch.zeros_like(parameter)
            delta[index] = 1e-5
            plus, minus = dict(params), dict(params)
            plus[name], minus[name] = parameter + delta, parameter - delta
            reference = (objective(plus) - objective(minus)).detach() / 2e-5
            torch.testing.assert_close(parameter.grad[index], reference, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("which, expected", [("new_nll", 5.), ("replay_nll", 2.)])
def test_objective_single_available_batch_and_numerical_floor(which, expected):
    values = objective_inputs()
    values[which] = None
    values["fisher_diagonal"] = {"ltm": tensor([0., 0.])}
    result = pm.regularized_objective(**values)
    assert result["data"].item() == pytest.approx(expected)
    assert result["ewc"].item() == 0
    assert result["kappa"].item() == pytest.approx(0.05)
    values["ridge_coefficient"] = 0
    assert pm.regularized_objective(**values)["ridge"].item() == 0


def test_eviction_batch_ema_no_admission_and_pi_one():
    diagonal, anchor, accepted, evicted = ({"ltm": tensor([2., 4.])}, {"ltm": tensor([1., 1.])},
                                         {"ltm": tensor([5., 6.], requires_grad=True)},
                                         {"ltm": tensor([6., 8.], requires_grad=True)})
    no_d, no_a, eta = pm.eviction_ema(diagonal, anchor, accepted, None, retired_count=0)
    assert eta == 0
    torch.testing.assert_close(no_d["ltm"], diagonal["ltm"])
    torch.testing.assert_close(no_a["ltm"], anchor["ltm"])
    no_d["ltm"].add_(1)
    torch.testing.assert_close(diagonal["ltm"], tensor([2., 4.]))
    new_d, new_a, eta = pm.eviction_ema(diagonal, anchor, accepted, evicted,
                                      retired_count=2, pi=0.25)
    assert eta == pytest.approx(0.4375)
    torch.testing.assert_close(new_d["ltm"], tensor([3.75, 5.75]))
    torch.testing.assert_close(new_a["ltm"], accepted["ltm"])
    assert not new_a["ltm"].requires_grad and not new_d["ltm"].requires_grad
    direct, _, eta = pm.eviction_ema(diagonal, anchor, accepted, evicted, retired_count=1, pi=1)
    assert eta == 1
    torch.testing.assert_close(direct["ltm"], evicted["ltm"])
    _, _, small_eta = pm.eviction_ema(diagonal, anchor, accepted, evicted,
                                     retired_count=1, pi=1e-18)
    assert small_eta == pytest.approx(1e-18, rel=1e-12, abs=0)


def test_discount_product_returns_segmented_terminals_and_potential_telescoping():
    discounts = pm.time_discounts(tensor([0.5, 1.0, 0.25]), gamma0=0.8)
    torch.testing.assert_close(discounts, tensor([0.8, 0.64, math.sqrt(0.8)]))
    base = tensor([-0.02, -0.04, 10.0])
    potential = tensor([0.2, 0.6, 0.5, 0.0])
    terminal = torch.tensor([False, False, True])
    shaped = base + discounts * potential[1:] - potential[:-1]
    base_return = pm.discounted_returns(base, discounts, terminal)
    shaped_return = pm.discounted_returns(shaped, discounts, terminal)
    torch.testing.assert_close(base_return, tensor([5.068, 6.36, 10.0]))
    torch.testing.assert_close(shaped_return, base_return - potential[:-1])
    segment = pm.discounted_returns(tensor([1., 2., 3., 4.]), tensor([0.5] * 4),
                                    torch.tensor([False, True, False, True]))
    torch.testing.assert_close(segment, tensor([2., 2., 5., 4.]))


def test_binning_boundaries_midpoints_per_item_and_quantization_bound():
    edges = tensor([-2., -1., 0., 1., 2.])
    returns = tensor([-2., -1., -0.25, 0., 1., 2.])
    labels = pm.bin_returns(returns, edges)
    torch.testing.assert_close(labels, torch.tensor([0, 1, 1, 2, 3, 3]))
    representatives = pm.return_bin_values(edges)
    assert bool(((representatives[labels] - returns).abs() <= 0.5).all())
    per_item = tensor([[0., 1., 2.], [3., 5., 7.]])
    torch.testing.assert_close(pm.bin_returns(tensor([1., 5.]), per_item), torch.tensor([1, 1]))
    torch.testing.assert_close(pm.return_bin_values(per_item), tensor([[0.5, 1.5], [4., 6.]]))


def test_teacher_floor_is_applied_to_executed_distribution_before_reference_logits():
    original = tensor([[1, 0, 0, 0, 0, 0, 0, 0], [0.125] * 8], requires_grad=True)
    executed, logits = pm.teacher_finite_logits(original, 0.01)
    recovered = torch.cat((logits, torch.zeros(2, 1, dtype=DTYPE)), dim=-1).softmax(-1)
    torch.testing.assert_close(recovered, executed)
    torch.testing.assert_close(executed[0], tensor([1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) / 1.07)
    assert executed[0, 0] < 1 and original[0, 0] == 1
    assert not logits.requires_grad
    torch.testing.assert_close(logits[1], torch.zeros(7, dtype=DTYPE))


def test_efficiency_fixtures_and_negative_savings_are_not_hidden():
    result = pm.efficiency_metrics(100, 50, 10, 20)
    assert result == dict(f=0.5, c=2.0, S=0.25)
    assert pm.efficiency_metrics(100, 120, 10, 20)["S"] == pytest.approx(-0.1)
    control = pm.budget_averaged_failure(tensor([0, 40, 100]), tensor([0.2, 0.4, 0.6]))
    fd = pm.budget_averaged_failure(tensor([0, 40, 100]), tensor([0.2, 0.6, 0.8]))
    assert control.item() == pytest.approx(0.58)
    assert fd.item() == pytest.approx(0.42)
    assert (control - fd).item() == pytest.approx(0.16)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_accumulation_dtype_and_reduced_precision_output_gradients(dtype):
    _, params, predict, fixed = fixture(torch.float32)

    def lower_precision(p):
        return {name: value.to(dtype) for name, value in predict(p).items()}

    labels = torch.tensor([0, 1, 2])
    actions = torch.zeros(3, 7)
    nll = pm.joint_nll(lower_precision(params), labels, actions, fixed)
    assert nll.dtype == torch.float32
    nll.mean().backward()
    assert all(torch.isfinite(value.grad).all() for value in params.values())
    fisher = pm.model_fisher_diagonal(lower_precision, params, fixed)
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in fisher.values())


def test_float32_nll_fisher_and_fd_agree_with_float64_references():
    results = []
    for dtype in (torch.float32, torch.float64):
        _, params, predict, fixed = fixture(dtype)
        nll = pm.joint_nll(predict(params), torch.tensor([0, 1, 2]),
                           torch.zeros(3, 7, dtype=dtype), fixed)
        fisher = pm.model_fisher_diagonal(predict, params, fixed)
        fd = pm.stm_fd_penalty(predict, params, fixed, ["stm"], probes(),
                               torch.ones(3, dtype=dtype), 0.02)
        results.append((nll, fisher, fd))
    torch.testing.assert_close(results[0][0].double(), results[1][0], rtol=2e-6, atol=1e-6)
    for name in ("ltm", "stm"):
        torch.testing.assert_close(results[0][1][name].double(), results[1][1][name],
                                   rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(results[0][2].double(), results[1][2], rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("override", [
    {"temperature": 0}, {"temperature": float("nan")},
    {"bin_values": tensor([0., 0., 1.])},
    {"baseline": tensor([])}, {"baseline": tensor([float("inf")])},
    {"sigma_action": -torch.eye(7, dtype=DTYPE)},
    {"sigma_action": torch.ones(7, 7, dtype=DTYPE)},
    {"sigma_action": torch.eye(6, dtype=DTYPE)},
    {"sigma_action": torch.tril(torch.ones(7, 7, dtype=DTYPE))},
    {"baseline": tensor([-1000., -1000., -1000.])},
    {"baseline": tensor([1000., 1000., 1000.])},
    {"log_weight_cap": float("nan")},
])
def test_invalid_design_fails(override):
    with pytest.raises(ValueError):
        design(**override)


def test_explicit_upper_log_weight_cap_reports_hits_without_silent_clipping():
    fixed = design(bin_values=tensor([-1., 0., 1.]), baseline=torch.zeros(3, dtype=DTYPE),
                   temperature=1., log_weight_cap=0.5)
    torch.testing.assert_close(fixed.weights[0], tensor([-1., 0., 0.5]).exp())
    assert fixed.cap_hit_rate.item() == pytest.approx(1 / 3)


@pytest.mark.parametrize("case", ["output_shape", "output_nan", "output_keys", "label_dtype",
                                  "label_range", "action_shape", "action_nan"])
def test_likelihood_failure_cases(case):
    _, params, predict, fixed = fixture()
    output, labels, actions = predict(params), torch.tensor([0, 1, 2]), torch.zeros(3, 7, dtype=DTYPE)
    if case == "output_shape":
        output["action_mean"] = actions[:, :6]
    elif case == "output_nan":
        output["return_logits"] = torch.full((3, 3), float("nan"))
    elif case == "output_keys":
        output = {}
    elif case == "label_dtype":
        labels = labels.float()
    elif case == "label_range":
        labels[0] = 3
    elif case == "action_shape":
        actions = actions[:2]
    elif case == "action_nan":
        actions[0, 0] = float("nan")
    with pytest.raises(ValueError):
        pm.joint_nll(output, labels, actions, fixed)


@pytest.mark.parametrize("case", ["no_rng", "bad_method", "zero_draws", "no_params", "unknown_name"])
def test_fisher_failure_cases(case):
    _, params, predict, fixed = fixture()
    kwargs = {}
    if case == "no_rng":
        kwargs["method"] = "sampled"
    elif case == "bad_method":
        kwargs["method"] = "empirical"
    elif case == "zero_draws":
        kwargs["draws"] = 0
    elif case == "no_params":
        params = {}
    elif case == "unknown_name":
        params = {"missing": tensor(1.)}
    with pytest.raises(ValueError):
        pm.model_fisher_diagonal(predict, params, fixed, **kwargs)


@pytest.mark.parametrize("case", ["epsilon", "empty", "probe_keys", "probe_shape", "future_age",
                                  "duplicate_names", "arrival", "capacity"])
def test_fd_and_age_failure_cases(case):
    _, params, predict, fixed = fixture()
    epsilon, probe, ages, names = 0.01, probes(), torch.ones(3, dtype=DTYPE), ["stm"]
    with pytest.raises(ValueError):
        if case == "arrival":
            pm.replay_ages(3, torch.tensor([4]), 5)
        elif case == "capacity":
            pm.replay_ages(3, torch.tensor([1]), 0)
        else:
            if case == "epsilon":
                epsilon = 0
            elif case == "empty":
                probe = ()
            elif case == "probe_keys":
                probe = ({"ltm": tensor([1., 1.])},)
            elif case == "probe_shape":
                probe = ({"stm": tensor([1.])},)
            elif case == "future_age":
                ages[0] = -1
            elif case == "duplicate_names":
                names = ["stm", "stm"]
            pm.stm_fd_penalty(predict, params, fixed, names, probe, ages, epsilon)


@pytest.mark.parametrize("case", ["no_data", "empty_data", "overlap", "negative_fisher", "pi",
                                  "missing_fd", "negative_ridge", "negative_floor"])
def test_objective_failure_cases(case):
    values = objective_inputs()
    if case == "no_data":
        values.update(new_nll=None, replay_nll=None)
    elif case == "empty_data":
        values["new_nll"] = tensor([])
    elif case == "overlap":
        values["stm_names"] = ["ltm"]
    elif case == "negative_fisher":
        values["fisher_diagonal"] = {"ltm": tensor([-1., 1.])}
    elif case == "pi":
        values["pi"] = 0
    elif case == "missing_fd":
        values["fd_penalty"] = None
    elif case == "negative_ridge":
        values["ridge_coefficient"] = -1
    elif case == "negative_floor":
        values["numerical_floor"] = -1
    with pytest.raises(ValueError):
        pm.regularized_objective(**values)


@pytest.mark.parametrize("count, fisher", [(0, {"x": tensor(1.)}), (1, None),
                                         (-1, None), (1, {"x": tensor(-1.)})])
def test_eviction_failure_cases(count, fisher):
    with pytest.raises(ValueError):
        pm.eviction_ema({"x": tensor(1.)}, {"x": tensor(0.)}, {"x": tensor(2.)},
                        fisher, retired_count=count)


@pytest.mark.parametrize("case", ["unterminated", "truncated", "discount", "flags", "reward_nan"])
def test_return_failure_cases(case):
    rewards, discounts, terminated = tensor([1., 2.]), tensor([0.9, 0.9]), torch.tensor([False, True])
    truncated = None
    if case == "unterminated":
        terminated[-1] = False
    elif case == "truncated":
        truncated = torch.tensor([True, False])
    elif case == "discount":
        discounts[0] = 1.1
    elif case == "flags":
        terminated = terminated.long()
    elif case == "reward_nan":
        rewards[0] = float("nan")
    with pytest.raises(ValueError):
        pm.discounted_returns(rewards, discounts, terminated, truncated)


@pytest.mark.parametrize("bad", [tensor([-2.01]), tensor([2.01]), tensor([float("nan")])])
def test_bins_do_not_clip_or_hide_nonfinite_returns(bad):
    with pytest.raises(ValueError):
        pm.bin_returns(bad, tensor([-2., 0., 2.]))


@pytest.mark.parametrize("probabilities, floor", [(tensor([1.] * 8), 0.01),
    (tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 0), (tensor([0.125] * 8), 0.125),
    (tensor([-0.1, 1.1, 0, 0, 0, 0, 0, 0]), 0.01), (tensor([1., 0.]), 0.01)])
def test_teacher_failure_cases(probabilities, floor):
    with pytest.raises(ValueError):
        pm.teacher_finite_logits(probabilities, floor)


@pytest.mark.parametrize("values", [(0, 50, 10, 20), (100, 50, 0, 20),
                                    (100, 50, 10, 0), (100, -1, 10, 20),
                                    (100, 50, 10, float("nan")),
                                    (100, 50, 1e300, 1e-300),
                                    (100, 50, 1e-300, 1e300)])
def test_efficiency_failure_cases(values):
    with pytest.raises(ValueError):
        pm.efficiency_metrics(*values)


def test_budget_area_requires_observed_zero_and_increasing_checkpoints():
    for samples, successes in (([1, 2], [0.1, 0.2]), ([0, 2, 1], [0.1, 0.2, 0.3]),
                               ([0, 1], [0.1, 1.1])):
        with pytest.raises(ValueError):
            pm.budget_averaged_failure(tensor(samples), tensor(successes))
