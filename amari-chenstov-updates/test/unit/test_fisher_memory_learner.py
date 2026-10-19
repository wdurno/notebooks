"""Lifecycle, gradient routing, and bounded optimizer contracts for Plan 15."""

import copy
import dataclasses
import io
import math

import pytest
import torch
from torch.nn import functional as F

from mnist_experiment.fisher_memory import learner
from mnist_experiment.fisher_memory.model import build_model, flat_logits, memory_indices
from src.representations import representation_from_artifact


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b, strict=True):
            assert_same(first, second)
    else:
        assert a == b


def state(capacity=2):
    return learner.initial_state(torch.tensor([1., 2., 3.], dtype=torch.float64),
                                 torch.tensor([0, 2]),
                                 torch.tensor([[2., .1], [.1, 1.]], dtype=torch.float64),
                                 capacity=capacity, seed=17)


def matrix(artifact):
    return representation_from_artifact(artifact).to_dense()


def test_age_clock_is_clipped_squared_and_not_normalized():
    weights = learner.age_weights([0, 8, 12, 16], 16, 16)
    torch.testing.assert_close(weights, torch.tensor([1., .25, .0625, 0.], dtype=torch.float64))
    assert learner.age_weights([0], 100, 16).item() == 1
    assert learner.age_weights([], 0, 16).numel() == 0
    with pytest.raises(ValueError):
        learner.age_weights([4], 3, 2)


def test_fifo_stage_and_no_ewc_before_eviction():
    original = state()
    before = copy.deepcopy(original)
    proposal = learner.stage_arrival(original, ["a", "b"], [10, 10])
    assert proposal["evicted"] == []
    assert [r["insertion_count"] for r in proposal["new"]] == [2, 2]
    accepted, diagnostics = learner.commit_arrival(
        original, original["theta"] + 1, ["a", "b"], [10, 10], pi=.05, arrival_size=2)
    assert_same(original, before)
    assert_same(accepted["anchor_L"], original["anchor_L"])
    assert_same(accepted["fisher_L"], original["fisher_L"])
    assert accepted["consolidated_ids"] == []
    assert diagnostics["eta"] == 0
    assert accepted["cumulative_count"] == 2
    next_stage = learner.stage_arrival(accepted, ["c"], [11])
    assert [r["exposure_id"] for r in next_stage["evicted"]] == ["a"]
    assert [r["exposure_id"] for r in next_stage["replay"]] == ["b", "c"]
    torch.testing.assert_close(next_stage["ages"], torch.full((2,), .25, dtype=torch.float64))


def test_exactly_once_eviction_eta_and_anchor_then_roundtrip():
    first = state()
    live, _ = learner.commit_arrival(first, first["theta"], [0, 1], [20, 21],
                                     pi=.05, arrival_size=2)
    old = copy.deepcopy(live)
    fresh = torch.tensor([[1., .2], [.2, 3.]], dtype=torch.float64)
    result, diagnostic = learner.commit_arrival(
        live, live["theta"] + .2, [2], [22], fresh, pi=.05, arrival_size=2, seed=18)
    assert_same(live, old)
    eta = 1 - math.sqrt(.95)
    assert diagnostic["eta"] == pytest.approx(eta)
    torch.testing.assert_close(matrix(result["fisher_L"]),
                               (1 - eta) * matrix(live["fisher_L"]) + eta * fresh)
    torch.testing.assert_close(result["anchor_L"], (live["theta"] + .2)[[0, 2]])
    assert result["consolidated_ids"] == [0]
    with pytest.raises(ValueError, match="already"):
        learner.stage_arrival(result, [0], [20])
    with pytest.raises(ValueError, match="already"):
        learner.stage_arrival(result, [1], [21])
    with pytest.raises(ValueError, match="already consolidated"):
        learner.consolidate(result["fisher_L"], result["anchor_L"], result["anchor_L"],
                            fresh, evicted_ids=[0], consolidated_ids=[0], pi=.05, arrival_size=2)
    stream = io.BytesIO()
    torch.save(result, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    learner.validate_state(restored)
    assert_same(restored, result)


def test_large_arrival_eviction_and_failed_transaction_are_pure():
    first = state(capacity=1)
    original = copy.deepcopy(first)
    proposed = learner.stage_arrival(first, [0, 1, 2], [10, 11, 12])
    assert [r["exposure_id"] for r in proposed["evicted"]] == [0, 1]
    with pytest.raises(ValueError, match="eviction requires"):
        learner.commit_arrival(first, first["theta"], [0, 1, 2], [10, 11, 12],
                                pi=.1, arrival_size=3)
    with pytest.raises(learner.StopRequested):
        learner.commit_arrival(first, first["theta"], [0], [10], pi=.1,
                                arrival_size=1, check_stop=lambda: True)
    assert_same(first, original)


def test_compression_preserves_rng_psd_and_no_ridge():
    generator = torch.Generator().manual_seed(13)
    factor = torch.randn(12, 5, generator=generator, dtype=torch.float64)
    source = factor @ factor.T + .5 * torch.eye(12)
    rng = torch.get_rng_state().clone()
    result, diagnostics = learner.compress_fisher(source, rank=3, seed=99)
    assert torch.equal(rng, torch.get_rng_state())
    represented = matrix(result)
    assert result["factor"].shape[1] <= 3
    assert torch.linalg.eigvalsh(represented).min() >= -1e-12
    torch.testing.assert_close(represented.diagonal(), source.diagonal())
    assert diagnostics["source_trace"] == pytest.approx(float(source.trace()))
    zeros, _ = learner.compress_fisher(torch.zeros(3, 3, dtype=torch.float64))
    assert matrix(zeros).count_nonzero() == 0
    with pytest.raises(ValueError, match="indefinite"):
        learner.compress_fisher(torch.tensor([[1., 2.], [2., 1.]]))


def test_compression_cancellation_is_not_retried_as_numerical_breakdown():
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        return calls == 4

    with pytest.raises(learner.StopRequested):
        learner.compress_fisher(torch.diag(torch.arange(1., 10.)), check_stop=check)
    assert calls == 4


@pytest.fixture
def fit_inputs():
    model, layout = build_model(42, device="cpu", dtype=torch.float64)
    theta = layout.flatten_module(model, detach=True)
    generator = torch.Generator().manual_seed(33)
    for spec in layout.specs:
        if spec.name.endswith("lora_B"):
            theta[spec.start:spec.stop] = .05 * torch.randn(spec.numel, generator=generator,
                                                          dtype=theta.dtype)
    x = torch.randn(4, 1, 28, 28, generator=generator, dtype=theta.dtype)
    ltm, stm = memory_indices(layout)
    return dict(model=model, layout=layout, theta=theta, new_x=x[:2],
                new_y=torch.tensor([1, 9]), replay_x=x[2:], replay_y=torch.tensor([3, 4]),
                ages=torch.tensor([.25, 1.], dtype=theta.dtype),
                fisher=.03 * torch.eye(len(ltm), dtype=theta.dtype),
                anchor=theta[ltm].clone(), seed=912)


def test_retention_is_ltm_only_and_ridge_is_step_anchored():
    theta = torch.tensor([2., 7., 4.], dtype=torch.float64, requires_grad=True)
    ltm, stm = torch.tensor([0, 2]), torch.tensor([1])
    fisher = torch.diag(torch.tensor([2., 4.], dtype=torch.float64))
    ewc, ridge, kappa = learner.retention_terms(
        theta, ltm, stm, fisher, torch.tensor([1., 2.]), torch.tensor([5.]), pi=.2)
    assert kappa == pytest.approx(.3)
    assert ewc.item() == pytest.approx(36.)
    assert ridge.item() == pytest.approx(5.4)
    gradient = torch.autograd.grad(ewc, theta, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.tensor([8., 0., 32.], dtype=theta.dtype))
    assert torch.autograd.grad(ridge, theta)[0][1] != 0


def test_data_objective_normalization_empty_replay_and_zero_strength(fit_inputs):
    args = fit_inputs
    cfg = learner.FitSettings(lambda_f=0, num_probes=1)
    objective, _ = learner.prepare_objective(**args, condition="replay_ewc", settings=cfg)
    _, parts = objective(args["theta"])
    new = F.cross_entropy(flat_logits(args["model"], args["layout"], args["theta"], args["new_x"]), args["new_y"])
    replay = F.cross_entropy(flat_logits(args["model"], args["layout"], args["theta"], args["replay_x"]), args["replay_y"])
    torch.testing.assert_close(parts["data_loss"], (new + replay) / 2)
    trace, _ = learner.prepare_objective(**args, condition="jvp", settings=cfg)
    torch.testing.assert_close(objective(args["theta"])[0], trace(args["theta"])[0])
    empty = {**args, "replay_x": args["replay_x"][:0], "replay_y": args["replay_y"][:0],
             "ages": args["ages"][:0]}
    objective, _ = learner.prepare_objective(**empty, condition="distillation", settings=cfg)
    _, parts = objective(args["theta"])
    torch.testing.assert_close(parts["data_loss"], new)
    assert parts["auxiliary"] == 0


@pytest.mark.parametrize("condition", ["jvp", "finite_difference", "directional"])
def test_curvature_gradient_reaches_both_blocks_and_is_fixed(fit_inputs, condition):
    args = fit_inputs
    ltm, stm = memory_indices(args["layout"])
    cfg = learner.FitSettings(lambda_f=1., num_probes=2, fd_step=.01, chunk_size=1)
    direction = torch.full((len(stm),), .001, dtype=torch.float64, requires_grad=True)
    objective, _ = learner.prepare_objective(**args, condition=condition, settings=cfg,
                                             direction=direction)
    theta = args["theta"].clone().requires_grad_()
    loss, parts = objective(theta)
    gradient = torch.autograd.grad(parts["auxiliary"], theta, retain_graph=True)[0]
    assert gradient[ltm].norm() > 1e-10
    assert gradient[stm].norm() > 1e-10
    assert torch.autograd.grad(loss, direction, allow_unused=True)[0] is None
    again = objective(theta)[1]["auxiliary"]
    torch.testing.assert_close(again, parts["auxiliary"], rtol=0, atol=0)


def test_curvature_uses_replay_count_not_age_mass(fit_inputs):
    from mnist_experiment.fisher_memory.curvature import jvp_values
    from src.seeding import derive_component_seed

    args = fit_inputs
    _, stm = memory_indices(args["layout"])
    cfg = learner.FitSettings(lambda_f=1., num_probes=2)
    objective, _ = learner.prepare_objective(**args, condition="jvp", settings=cfg)
    gen = torch.Generator().manual_seed(derive_component_seed(args["seed"], "training_curvature_probes"))
    probes = (torch.randint(2, (2, len(stm)), generator=gen) * 2 - 1).double()
    values = jvp_values(args["model"], args["layout"], args["theta"], args["replay_x"], probes).mean(0) / len(stm)
    torch.testing.assert_close(objective(args["theta"])[1]["auxiliary"],
                               (args["ages"] * values).mean())


def test_distillation_auxiliary_only_updates_ltm_and_teacher_is_frozen(fit_inputs):
    args = fit_inputs
    ltm, stm = memory_indices(args["layout"])
    cfg = learner.FitSettings(lambda_d=1., chunk_size=1)
    objective, metadata = learner.prepare_objective(**args, condition="distillation", settings=cfg)
    theta = args["theta"].clone().requires_grad_()
    _, parts = objective(theta)
    gradient = torch.autograd.grad(parts["auxiliary"], theta, retain_graph=True)[0]
    assert gradient[ltm].norm() > 0
    assert gradient[stm].count_nonzero() == 0
    data_gradient = torch.autograd.grad(parts["data_loss"], theta)[0]
    assert data_gradient[stm].norm() > 0
    assert metadata["teacher_detached"]
    changed = theta.detach().clone()
    changed[stm] += .1
    torch.testing.assert_close(objective(changed)[1]["auxiliary"], parts["auxiliary"])


@pytest.mark.parametrize("solver", ["lbfgs", "first_order"])
def test_fit_is_paired_bounded_and_does_not_mutate_model_or_rng(fit_inputs, solver):
    args = fit_inputs
    original = copy.deepcopy(args)
    rng = torch.get_rng_state().clone()
    cfg = learner.FitSettings(solver=solver, max_iter=3, max_eval=7)
    theta, diagnostics = learner.fit_step(**args, condition="replay_ewc", settings=cfg)
    again, repeat = learner.fit_step(**args, condition="replay_ewc", settings=cfg)
    torch.testing.assert_close(theta, again, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert diagnostics["closure_calls"] <= 7
    assert diagnostics["objective_evaluations"] <= 7
    assert diagnostics["final_objective"] <= diagnostics["initial_objective"]
    assert diagnostics["final_objective"] == repeat["final_objective"]
    assert_same(args["theta"], original["theta"])
    assert_same(args["fisher"], original["fisher"])
    assert_same(args["anchor"], original["anchor"])
    assert_same(args["model"].state_dict(), original["model"].state_dict())


def test_directional_shadow_is_bounded_detached_and_not_committed(fit_inputs):
    args = fit_inputs
    cfg = learner.FitSettings(lambda_f=.2, max_iter=1, max_eval=1,
                              shadow_max_eval=5, direction_max_norm=.001)
    theta, diagnostics = learner.fit_step(**args, condition="directional", settings=cfg)
    torch.testing.assert_close(theta, args["theta"], rtol=0, atol=0)
    assert diagnostics["shadow_closure_calls"] <= 5
    assert diagnostics["closure_calls"] <= 1
    assert diagnostics["direction_norm"] <= .0010000001
    cfg = dataclasses.replace(cfg, shadow_max_eval=1)
    theta, diagnostics = learner.fit_step(**args, condition="directional", settings=cfg)
    assert diagnostics["unsupported_direction"]
    assert diagnostics["final_auxiliary"] == 0.


@pytest.mark.parametrize("condition", sorted(learner.CONDITIONS))
def test_all_conditions_complete_a_real_bounded_solve(fit_inputs, condition):
    args = fit_inputs
    cfg = learner.FitSettings(lambda_f=.1, lambda_d=.1, num_probes=1,
                              max_iter=2, max_eval=5)
    theta, diagnostic = learner.fit_step(**args, condition=condition, settings=cfg)
    assert torch.isfinite(theta).all()
    assert not theta.requires_grad
    assert not diagnostic["solver_failed"]
    assert diagnostic["closure_calls"] <= 5
    assert diagnostic["final_objective"] <= diagnostic["initial_objective"]
    assert diagnostic["gradient_inf_norm"] is not None


def test_state_rejects_missing_or_overlapping_exposure_records():
    initial = state()
    result, _ = learner.commit_arrival(initial, initial["theta"], [0], [10],
                                       pi=.05, arrival_size=1)
    damaged = copy.deepcopy(result)
    damaged["consolidated_ids"] = [0]
    with pytest.raises(ValueError, match="inconsistent"):
        learner.validate_state(damaged)
    damaged = copy.deepcopy(result)
    damaged["cumulative_count"] = 2
    with pytest.raises(ValueError, match="inconsistent"):
        learner.validate_state(damaged)


def test_budget_does_not_accept_unfinished_line_search_trial():
    start = torch.tensor([0.], dtype=torch.float64)
    def objective(theta):
        value = (theta - 2.).square().sum()
        return value, {"loss": value}
    cfg = learner.FitSettings(max_iter=20, max_eval=1)
    theta, diagnostics = learner._bounded_solve(start, objective, cfg, None)
    torch.testing.assert_close(theta, start)
    assert diagnostics["solver_reason"] == "max_evaluations"
    assert diagnostics["closure_calls"] == 1
    assert diagnostics["final_objective"] == 4.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("material", [False, True])
def test_increasing_iterate_is_rejected_and_roundoff_is_not_solver_failure(monkeypatch, dtype, material):
    start = torch.ones(1, dtype=dtype)
    increase = .01 if material else 4 * torch.finfo(dtype).eps
    def step(optimizer, closure):
        initial = closure()
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].add_(increase)
        closure()
        return initial
    monkeypatch.setattr(torch.optim.LBFGS, "step", step)
    theta, diagnostics = learner._bounded_solve(
        start, lambda value: (value.sum(), {}), learner.FitSettings(max_iter=2), None)
    assert torch.equal(theta, start)
    assert diagnostics["solver_failed"] is material
    assert diagnostics["solver_reason"] == ("objective_increase" if material else "roundoff_stagnation")
    assert diagnostics["rejected_objective_increase"] > 0
    assert diagnostics["final_objective"] == diagnostics["initial_objective"]


def test_cancel_during_objective_rolls_back_every_input(fit_inputs):
    args = fit_inputs
    original = copy.deepcopy(args)
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        return calls == 12
    with pytest.raises(learner.StopRequested):
        learner.fit_step(**args, condition="jvp", settings=learner.FitSettings(
            lambda_f=1., num_probes=1, chunk_size=1), check_stop=check)
    assert calls == 12
    assert_same(args["theta"], original["theta"])
    assert_same(args["model"].state_dict(), original["model"].state_dict())
    assert_same(args["fisher"], original["fisher"])


@pytest.mark.parametrize("settings", [{"max_eval": 0}, {"shadow_max_eval": 6},
                                      {"pi": 0.}, {"replay_weight": 1.},
                                      {"fd_step": 0.}, {"solver": "unknown"}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        learner.FitSettings(**settings).validate()
