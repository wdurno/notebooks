"""Small CPU contracts for the Phase 8 scalar Gaussian model and learner."""

import copy
import dataclasses
import io
import math

import pytest
import torch
from torch.distributions import Normal, kl_divergence
from torch.func import jacrev

from mnist_experiment.fisher_memory_regression import learner
from mnist_experiment.fisher_memory_regression.curvature import (
    fd_values, gaussian_nll, jvp_values, reference_fisher,
)
from mnist_experiment.fisher_memory_regression.model import (
    ANGLE_SCALE_DEGREES, build_model, flat_prediction, memory_indices,
)
from src.representations import representation_from_artifact
from src.seeding import derive_component_seed


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    else:
        assert left == right


def case(*, moved=True, dtype=torch.float64):
    model, layout = build_model(17, dtype=dtype, factor_seed=57)
    theta = layout.flatten_module(model, detach=True)
    generator = torch.Generator().manual_seed(91)
    if moved:
        for spec in layout.specs:
            if spec.name.endswith("lora_B"):
                theta[spec.start:spec.stop] = .15 * torch.randn(
                    spec.numel, generator=generator, dtype=dtype
                )
    x = torch.randn(3, 1, 28, 28, generator=generator, dtype=dtype)
    ltm, stm = memory_indices(layout)
    probes = (2 * torch.randint(2, (2, len(stm)), generator=generator) - 1).to(dtype)
    return model, layout, theta.requires_grad_(), x, ltm, stm, probes


@pytest.fixture
def fit_inputs():
    model, layout, theta, x, ltm, _, _ = case()
    return dict(model=model, layout=layout, theta=theta.detach(), new_x=x[:1],
                new_y=torch.tensor([35.], dtype=x.dtype), replay_x=x[1:],
                replay_y=torch.tensor([-4., 8.], dtype=x.dtype),
                ages=torch.tensor([.25, 1.], dtype=x.dtype),
                fisher=.03 * torch.eye(len(ltm), dtype=x.dtype),
                anchor=theta[ltm].detach().clone(), seed=321)


def small_state(capacity=2):
    return learner.initial_state(
        torch.tensor([1., 2., 3.], dtype=torch.float64), torch.tensor([0, 2]),
        torch.tensor([[2., .1], [.1, 1.]], dtype=torch.float64),
        capacity=capacity, rank=2, seed=7,
    )


def dense(artifact):
    return representation_from_artifact(artifact).to_dense()


def test_model_counts_chart_rng_pairing_and_scalar_output():
    before = torch.get_rng_state().clone()
    first, layout = build_model(11, factor_seed=42, dtype=torch.float64)
    twin, _ = build_model(11, factor_seed=42, dtype=torch.float64)
    other_factors, _ = build_model(11, factor_seed=43, dtype=torch.float64)
    assert torch.equal(before, torch.get_rng_state())
    ltm, stm = memory_indices(layout)
    assert (len(ltm), len(stm), layout.total_numel) == (287, 135, 422)
    assert torch.equal(torch.cat((ltm, stm)).sort().values, torch.arange(422))
    assert first.features[0].rank == first.features[3].rank == 2
    assert first.classifier.rank == 1
    theta = layout.flatten_module(first)
    assert torch.equal(theta, layout.flatten_module(twin))
    second = layout.flatten_module(other_factors)
    assert torch.equal(theta[ltm], second[ltm])
    assert not torch.equal(theta[stm], second[stm])
    x = torch.ones(2, 1, 28, 28, dtype=torch.float64)
    prediction = flat_prediction(first, layout, theta, x)
    assert prediction.shape == (2,)
    torch.testing.assert_close(prediction, other_factors(x), rtol=0, atol=0)
    torch.testing.assert_close(prediction, first(x, stm_scale=0), rtol=0, atol=0)
    assert not any("contrast" in name for name, _ in first.named_buffers())
    assert all(value.requires_grad for value in first.parameters())
    for spec in layout.specs:
        if spec.name.endswith(".bias"):
            assert set(range(spec.start, spec.stop)) <= set(ltm.tolist())
    with pytest.raises(ValueError, match="28, 28"):
        first(x[:, :, :20])


def test_initial_adapters_learn_and_pretraining_can_disable_them():
    model, layout, theta, x, ltm, stm, _ = case(moved=False)
    labels = torch.tensor([30., -25., 12.], dtype=torch.float64)
    loss = gaussian_nll(flat_prediction(model, layout, theta, x), labels).mean()
    gradient = torch.autograd.grad(loss, theta)[0]
    assert gradient[ltm].norm() > 0 and gradient[stm].norm() > 0
    for spec in layout.specs:
        if spec.name.endswith("lora_A"):
            assert theta[spec.start:spec.stop].count_nonzero() == spec.numel
            assert gradient[spec.start:spec.stop].count_nonzero() == 0
        elif spec.name.endswith("lora_B"):
            assert theta[spec.start:spec.stop].count_nonzero() == 0
            assert gradient[spec.start:spec.stop].norm() > 0
    loss = gaussian_nll(flat_prediction(model, layout, theta, x, stm_scale=0), labels).mean()
    gradient = torch.autograd.grad(loss, theta)[0]
    assert gradient[stm].count_nonzero() == 0 and gradient[ltm].norm() > 0
    modified = theta.detach().clone()
    bias = next(spec for spec in layout.specs if spec.name == "classifier.bias")
    modified[bias.start] += 100
    assert (45 * flat_prediction(model, layout, modified, x) > 90).all()


def test_gaussian_degree_nll_mse_and_score_sign():
    model, layout, theta, x, _, _, _ = case()
    y = torch.tensor([-12., 40., 7.], dtype=torch.float64)
    prediction = flat_prediction(model, layout, theta, x)
    nll = gaussian_nll(prediction, y)
    normal = Normal(45 * prediction, 45)
    torch.testing.assert_close(nll, -normal.log_prob(y))
    torch.testing.assert_close(
        nll.mean(), (45 * prediction - y).square().mean() / (2 * 45**2)
        + math.log(45 * math.sqrt(2 * math.pi)),
    )
    nll_gradient = torch.autograd.grad(nll.sum(), theta, retain_graph=True)[0]
    score = torch.autograd.grad(normal.log_prob(y).sum(), theta)[0]
    torch.testing.assert_close(nll_gradient, -score)
    with pytest.raises(ValueError, match="matching vectors"):
        gaussian_nll(prediction[:, None], y)
    with pytest.raises(ValueError, match="positive"):
        gaussian_nll(prediction, y, scale=0)


def test_exact_fisher_integrates_model_targets_not_observed_residuals():
    model, layout, theta, x, ltm, _, _ = case()
    prediction = flat_prediction(model, layout, theta, x)
    jacobian = jacrev(lambda p: flat_prediction(model, layout, p, x))(theta)
    expected = jacobian.T @ jacobian / len(x)
    fisher = reference_fisher(model, layout, theta, x, chunk_size=2)
    assert fisher.dtype == torch.float64 and not fisher.requires_grad
    torch.testing.assert_close(fisher, expected, rtol=1e-12, atol=1e-14)
    block = reference_fisher(model, layout, theta, x, indices=ltm, chunk_size=1)
    torch.testing.assert_close(block, fisher[ltm][:, ltm], rtol=1e-12, atol=1e-14)
    assert not torch.allclose(fisher, jacobian.mean(0).outer(jacobian.mean(0)), atol=1e-12)
    # Residual +/-1 has exactly the first two standard-normal moments needed
    # for the score outer product, so this is a deterministic integration audit.
    for sign in (-1, 1):
        targets = 45 * (prediction.detach() + sign)
        scores = -jacrev(lambda p: gaussian_nll(flat_prediction(model, layout, p, x), targets))(theta)
        torch.testing.assert_close(scores.T @ scores / len(x), fisher, rtol=1e-12, atol=1e-14)
    zero_residuals = jacrev(lambda p: gaussian_nll(
        flat_prediction(model, layout, p, x), 45 * prediction.detach()
    ))(theta)
    assert zero_residuals.abs().max() < 1e-15
    assert fisher.norm() > 0


def test_reference_fisher_checks_stop_and_indices():
    model, layout, theta, x, _, _, _ = case()
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(learner.StopRequested):
        reference_fisher(model, layout, theta, x, chunk_size=1, check_stop=stop)
    assert calls == 3
    with pytest.raises(ValueError, match="unique integer"):
        reference_fisher(model, layout, theta, x, indices=torch.tensor([1, 1]))


def test_gaussian_fd_matches_symmetric_kl_and_jvp_values_and_gradients():
    model, layout, theta, x, ltm, stm, probes = case()
    step = 1e-4
    actual = fd_values(model, layout, theta, x, probes, step)
    reference = jvp_values(model, layout, theta, x, probes)
    assert actual.shape == (2, 3) and actual.dtype == torch.float64
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-10)
    embedded = theta.new_zeros(len(probes), len(theta))
    embedded[:, stm] = probes
    for k, direction in enumerate(embedded):
        plus = Normal(45 * flat_prediction(model, layout, theta + step / 2 * direction, x), 45)
        minus = Normal(45 * flat_prediction(model, layout, theta - step / 2 * direction, x), 45)
        expected = (kl_divergence(plus, minus) + kl_divergence(minus, plus)) / step**2
        torch.testing.assert_close(actual[k], expected, rtol=1e-6, atol=1e-8)
    grad_fd = torch.autograd.grad(actual.sum(), theta)[0]
    grad_jvp = torch.autograd.grad(reference.sum(), theta)[0]
    torch.testing.assert_close(grad_fd, grad_jvp, rtol=2e-5, atol=1e-8)
    assert grad_fd[ltm].norm() > 0 and grad_fd[stm].norm() > 0
    embedded[:, ltm[0]] = 1
    with pytest.raises(ValueError, match="zero in LTM"):
        fd_values(model, layout, theta, x, embedded, step)


def test_fd_both_branches_remain_differentiable_and_probes_are_fixed():
    model, layout, theta, x, _, stm, probes = case()
    probes.requires_grad_()
    chosen = torch.tensor([layout.specs[0].start, layout.specs[2].start,
                           layout.specs[3].start, layout.specs[-1].start])

    def function(values):
        parameter = theta.detach().index_copy(0, chosen, values)
        return fd_values(model, layout, parameter, x[:1], probes, .01)

    assert torch.autograd.gradcheck(function, (theta.detach()[chosen].requires_grad_(),),
                                    eps=1e-5, atol=1e-6, rtol=1e-4)
    actual = fd_values(model, layout, theta, x, probes, .01).sum()
    assert torch.autograd.grad(actual, probes, allow_unused=True)[0] is None
    # A detached prediction branch would fail the explicit central-difference
    # gradient audit above even though its forward curvature values look right.
    full = theta.new_zeros(len(probes), len(theta))
    full[:, stm] = probes.detach()
    torch.testing.assert_close(fd_values(model, layout, theta, x, full, .01),
                               fd_values(model, layout, theta, x, probes, .01))


def test_zero_capacity_retires_immediately_with_nominal_four_exposure_unit():
    state = small_state(capacity=0)
    original = copy.deepcopy(state)
    staged = learner.stage_arrival(state, ["a"], [12])
    assert not staged["replay"] and len(staged["evicted"]) == 1
    assert staged["ages"].numel() == 0
    assert learner.replay_indices(state, 0, 13).numel() == 0
    fresh = torch.eye(2, dtype=torch.float64)
    result, stats = learner.commit_arrival(state, state["theta"] + .1, ["a"], [12], fresh,
                                          pi=.025, arrival_size=4, rank=2)
    eta = 1 - .975**.25
    assert stats["eta"] == pytest.approx(eta)
    torch.testing.assert_close(dense(result["fisher_L"]),
                               (1 - eta) * dense(state["fisher_L"]) + eta * fresh)
    assert result["replay"] == [] and result["consolidated_ids"] == ["a"]
    assert result["archive_updates"] == result["accepted_arrivals"] == 1
    torch.testing.assert_close(result["anchor_L"], result["theta"][[0, 2]])
    assert_same(state, original)
    with pytest.raises(ValueError, match="already"):
        learner.stage_arrival(result, ["a"], [12])
    with pytest.raises(ValueError, match="nominal value 4"):
        learner.commit_arrival(state, state["theta"], ["a"], [12], fresh,
                               pi=.025, arrival_size=1)


def test_fifo_exactly_once_ignores_replay_subsampling_and_preserves_non_eviction_archive():
    initial = small_state(capacity=4)
    live, stats = learner.commit_arrival(initial, initial["theta"] + .1,
                                         [0, 1, 2, 3], [10, 11, 12, 13], pi=.025)
    assert stats["eta"] == 0
    assert_same(initial["anchor_L"], live["anchor_L"])
    assert_same(initial["fisher_L"], live["fisher_L"])
    selected = learner.replay_indices(live, 1, 71)
    assert len(selected) == 1
    staged = learner.stage_arrival(live, [4, 5], [14, 15])
    assert [r["exposure_id"] for r in staged["evicted"]] == [0, 1]
    torch.testing.assert_close(staged["ages"], torch.full((4,), .25, dtype=torch.float64))
    result, stats = learner.commit_arrival(
        live, live["theta"] + .2, [4, 5], [14, 15], torch.eye(2), pi=.025, rank=2,
    )
    assert result["consolidated_ids"] == [0, 1]
    assert stats["eta"] == pytest.approx(1 - .975**.5)
    assert [r["exposure_id"] for r in result["replay"]] == [2, 3, 4, 5]
    with pytest.raises(ValueError, match="already"):
        learner.stage_arrival(result, [0], [10])
    stream = io.BytesIO()
    torch.save(result, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    learner.validate_state(restored)
    assert_same(result, restored)


def test_replay_sampling_has_common_prefixes_and_no_global_rng_effect():
    state = small_state(capacity=4)
    state, _ = learner.commit_arrival(state, state["theta"], [0, 1, 2, 3], [0, 1, 2, 3], pi=.1)
    original, rng = copy.deepcopy(state), torch.get_rng_state().clone()
    one = learner.replay_indices(state, 1, 77)
    three = learner.replay_indices(state, 3, 77)
    all_rows = learner.replay_indices(state, 4, 77)
    assert torch.equal(one, three[:1]) and torch.equal(three, all_rows[:3])
    assert len(all_rows.unique()) == 4
    assert torch.equal(rng, torch.get_rng_state())
    assert_same(state, original)
    with pytest.raises(ValueError, match="1 <= b"):
        learner.replay_indices(state, 0, 77)


def test_state_validation_and_interrupted_commits_are_pure():
    state = small_state(capacity=1)
    original = copy.deepcopy(state)
    with pytest.raises(ValueError, match="eviction requires"):
        learner.commit_arrival(state, state["theta"], [0, 1], [10, 11], pi=.1)
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(learner.StopRequested):
        learner.commit_arrival(state, state["theta"], [0, 1], [10, 11],
                               torch.eye(2), pi=.1, check_stop=stop)
    assert_same(state, original)
    damaged = copy.deepcopy(state)
    damaged["likelihood"] = "categorical"
    with pytest.raises(ValueError, match="schema or likelihood"):
        learner.validate_state(damaged)
    damaged = copy.deepcopy(state)
    damaged["consolidated_ids"] = [1]
    with pytest.raises(ValueError, match="inconsistent"):
        learner.validate_state(damaged)


def test_replay_and_fd_normalization_is_mean_not_subset_size_or_age_mass(fit_inputs):
    args = fit_inputs
    cfg = learner.FitSettings(lambda_f=2, num_probes=2, fd_step=.01, chunk_size=1)
    objective, metadata = learner.prepare_objective(**args, condition="fd_replay_ewc", settings=cfg)
    _, parts = objective(args["theta"])
    new = gaussian_nll(flat_prediction(args["model"], args["layout"], args["theta"], args["new_x"]),
                       args["new_y"]).mean()
    replay = gaussian_nll(flat_prediction(args["model"], args["layout"], args["theta"], args["replay_x"]),
                          args["replay_y"]).mean()
    torch.testing.assert_close(parts["data_loss"], (new + replay) / 2)
    _, stm = memory_indices(args["layout"])
    probe_seed = derive_component_seed(args["seed"], "regression_training_curvature_probes")
    assert metadata["probe_seed"] == probe_seed
    probes = (2 * torch.randint(2, (2, len(stm)), generator=torch.Generator().manual_seed(probe_seed)) - 1).double()
    values = fd_values(args["model"], args["layout"], args["theta"], args["replay_x"], probes, .01)
    expected = (args["ages"] * values.mean(0) / len(stm)).mean()
    torch.testing.assert_close(parts["auxiliary"], expected)
    assert not torch.isclose(parts["auxiliary"], expected * len(args["ages"]) / args["ages"].sum())
    duplicated = {**args, "replay_x": args["replay_x"].repeat(2, 1, 1, 1),
                  "replay_y": args["replay_y"].repeat(2), "ages": args["ages"].repeat(2)}
    larger, _ = learner.prepare_objective(**duplicated, condition="fd_replay_ewc", settings=cfg)
    torch.testing.assert_close(larger(args["theta"])[0], objective(args["theta"])[0])
    control, _ = learner.prepare_objective(**args, condition="replay_ewc", settings=cfg)
    assert control(args["theta"])[1]["weighted_auxiliary"] == 0
    empty = {**args, "replay_x": args["replay_x"][:0], "replay_y": args["replay_y"][:0],
             "ages": args["ages"][:0]}
    boundary, _ = learner.prepare_objective(**empty, condition="ewc_only", settings=cfg)
    torch.testing.assert_close(boundary(args["theta"])[1]["data_loss"], new)
    assert boundary(args["theta"])[1]["auxiliary"] == 0
    with pytest.raises(ValueError, match="cannot receive replay"):
        learner.prepare_objective(**args, condition="ewc_only", settings=cfg)


def test_objective_is_immutable_across_closures_and_routes_gradients(fit_inputs):
    args = fit_inputs
    before = torch.get_rng_state().clone()
    cfg = learner.FitSettings(lambda_f=3, num_probes=2, fd_step=.01)
    objective, _ = learner.prepare_objective(**args, condition="fd_replay_ewc", settings=cfg)
    candidate = args["theta"].clone().requires_grad_()
    original, parts = objective(candidate)
    ltm, stm = memory_indices(args["layout"])
    auxiliary_gradient = torch.autograd.grad(parts["auxiliary"], candidate, retain_graph=True)[0]
    assert auxiliary_gradient[ltm].norm() > 0 and auxiliary_gradient[stm].norm() > 0
    gradient = torch.autograd.grad(original, candidate)[0]
    for key in ("theta", "new_x", "new_y", "replay_x", "replay_y", "ages", "fisher", "anchor"):
        args[key].add_(1)
    again = objective(candidate)[0]
    torch.testing.assert_close(original, again, rtol=0, atol=0)
    torch.testing.assert_close(gradient, torch.autograd.grad(again, candidate)[0], rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("condition", sorted(learner.CONDITIONS))
def test_bounded_fit_observer_pairing_and_no_input_mutation(fit_inputs, condition):
    args = fit_inputs
    if condition == "ewc_only":
        args = {**args, "replay_x": args["replay_x"][:0], "replay_y": args["replay_y"][:0],
                "ages": args["ages"][:0]}
    before = {key: value.detach().clone() for key, value in args.items() if isinstance(value, torch.Tensor)}
    model_before = copy.deepcopy(args["model"].state_dict())
    rng = torch.get_rng_state().clone()
    observed = []

    def observer(accepted, objective, metadata, diagnostics):
        observed.append(float(objective(accepted)[0]))
        assert observed[-1] == pytest.approx(diagnostics["final_objective"])
        assert metadata["gaussian_scale_degrees"] == ANGLE_SCALE_DEGREES
        accepted.add_(20)  # The observer must not mutate the solver's result.

    cfg = learner.FitSettings(lambda_f=.2, num_probes=1, max_iter=2, max_eval=5, fd_step=.01)
    accepted, info = learner.fit_step(**args, condition=condition, settings=cfg,
                                      objective_observer=observer)
    twin, other = learner.fit_step(**args, condition=condition, settings=cfg)
    torch.testing.assert_close(accepted, twin, rtol=0, atol=0)
    assert len(observed) == 1
    assert info["final_objective"] == other["final_objective"]
    assert not accepted.requires_grad and not info["solver_failed"]
    assert info["closure_calls"] <= 5 and info["objective_evaluations"] <= 5
    assert info["final_objective"] <= info["initial_objective"]
    assert info["new_presentations"] == info["objective_evaluations"] * len(args["new_y"])
    assert info["replay_presentations"] == info["objective_evaluations"] * len(args["replay_y"])
    expected_fd = 2 * cfg.num_probes * info["replay_presentations"] if condition == "fd_replay_ewc" else 0
    assert info["fd_branch_presentations"] == expected_fd
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before.items():
        assert_same(value, args[key])
    assert_same(model_before, args["model"].state_dict())


def test_interrupted_fit_preserves_inputs_and_budget_never_accepts_wolfe_trial(fit_inputs):
    original = copy.deepcopy(fit_inputs["theta"])
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 7

    cfg = learner.FitSettings(lambda_f=1, max_iter=3, max_eval=5)
    with pytest.raises(learner.StopRequested):
        learner.fit_step(**fit_inputs, condition="fd_replay_ewc", settings=cfg, check_stop=stop)
    assert_same(original, fit_inputs["theta"])
    accepted, info = learner.fit_step(**fit_inputs, condition="replay_ewc",
                                      settings=dataclasses.replace(cfg, max_eval=1))
    assert_same(original, accepted)
    assert info["solver_reason"] == "max_evaluations"


def test_regression_never_calls_categorical_loss_or_fisher(monkeypatch, fit_inputs):
    from mnist_experiment.fisher_memory import curvature as categorical

    def forbidden(*args, **kwargs):
        raise AssertionError("categorical computation in the regression core")

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", forbidden)
    monkeypatch.setattr(categorical, "reference_fisher", forbidden)
    args = fit_inputs
    learner.fit_step(**args, condition="fd_replay_ewc", settings=learner.FitSettings(
        num_probes=1, lambda_f=.1, max_iter=1, max_eval=2,
    ))
    ltm, _ = memory_indices(args["layout"])
    fisher = reference_fisher(args["model"], args["layout"], args["theta"], args["new_x"], indices=ltm)
    assert torch.isfinite(fisher).all()
