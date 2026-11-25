"""Actual tiny CPU head fits: EWC, transactions, resumption and retirement."""

import copy

import pytest
import torch

from numerical_experiment.picar_head_awac import learner as learner_module
from numerical_experiment.picar_head_awac.archive import Archive
from numerical_experiment.picar_head_awac.learner import Learner, Recipe
from numerical_experiment.picar_head_awac.likelihood import data_loss, joint_kl
from numerical_experiment.picar_head_awac.model import Heads


def _records():
    return [
        {"feature": torch.tensor([0.2, -0.1, 0.7]),
         "next_feature": torch.tensor([-0.3, 0.5, 0.4]),
         "action": torch.linspace(-0.4, 0.6, 7),
         "reward": 0.8, "discount": 0.9, "return": 1.3, "arrival": 0},
        {"feature": torch.tensor([-0.4, 0.6, 0.1]),
         "next_feature": torch.tensor([0.1, -0.2, 0.5]),
         "action": torch.linspace(0.5, -0.2, 7),
         "reward": -0.3, "discount": 0.0, "return": -0.4, "arrival": 1},
    ]


def _learner(steps=1, fd_strength=0.4, head_seed=91):
    recipe = Recipe(capacity=4, batch_size=2, rounds=1, steps=steps,
                    integration_draws=2, rank=2, learning_rate=0.003,
                    action_variance=4.0, critic_variance=2.0,
                    target_rate=0.25, fd_strength=fd_strength, fd_step=0.02)
    learner = Learner(Heads(hidden=3, width=2, seed=head_seed).cpu(), recipe,
                      seed=1919, method="td_replay")
    factor = torch.linspace(-0.1, 0.15, len(learner.ltm) * 2).reshape(-1, 2)
    learner.archive = Archive(factor.square().sum(1) + 0.5, factor, rank=2)
    return learner


def _assert_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.device == expected.device
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_exact(left, right)
    else:
        assert actual == expected


def _without_timing(row):
    return {key: value for key, value in row.items() if key != "update_seconds"}


@pytest.mark.parametrize("steps", [1, 3])
def test_first_proposal_uses_positive_ewc_and_target_refreshes_only_on_acceptance(steps, monkeypatch):
    learner = _learner(steps)
    before = copy.deepcopy(learner.state_dict())
    heads_before = copy.deepcopy(learner.heads.state_dict())
    records = _records()
    records_before = copy.deepcopy(records)
    global_rng = torch.get_rng_state().clone()
    proposals = []
    solve = learner.archive.solve

    def observe_solve(rhs, damping, weight=1):
        result = solve(rhs, damping, weight=weight)
        proposals.append((rhs.clone(), damping.clone(), result.clone(), weight))
        return result

    monkeypatch.setattr(learner.archive, "solve", observe_solve)
    checks = []

    def check():
        checks.append(True)
        for key in ("vector", "target", "anchor", "moment"):
            _assert_exact(getattr(learner, key), before[key])
        _assert_exact(learner.heads.state_dict(), heads_before)
        assert learner.version == before["version"]

    row = learner.fit(records, acquired=6, check=check)
    first = row["inner"][0]
    assert first["ewc"] == 0.0
    assert first["ewc_proposal_difference"] > 0.0
    rhs, damping, solution, weight = proposals[0]
    assert weight == pytest.approx((1 - learner.recipe.pi) / learner.recipe.pi)
    free = rhs / damping
    assert float((damping * solution.square()).sum()) < float((damping * free.square()).sum())
    assert len(checks) == steps
    assert row["post_step_ewc"] > 0
    old_anchor_offset = learner.vector[learner.ltm] - before["anchor"]
    expected_post = .5 * weight * learner.archive.quadratic(old_anchor_offset)
    assert row["post_step_ewc"] == pytest.approx(float(expected_post))
    if steps > 1:
        assert all(inner["ewc"] > 0 for inner in row["inner"][1:])
    assert learner.evaluations == steps and learner.version == 1
    assert learner.retired == 0
    _assert_exact(learner.archive.state_dict(), before["archive"])
    _assert_exact(learner.anchor, learner.vector[learner.ltm])
    expected_target = before["target"].clone().lerp_(learner.vector, learner.recipe.target_rate)
    _assert_exact(learner.target, expected_target)
    assert not torch.equal(learner.target, before["target"])
    _assert_exact(learner.coordinates.vector(), learner.vector)
    _assert_exact(torch.get_rng_state(), global_rng)
    _assert_exact(records, records_before)
    for name, value in learner.heads.named_buffers():
        _assert_exact(value, heads_before[name])
    assert all(parameter.grad is None for parameter in learner.heads.parameters())


def test_every_fit_starts_at_zero_ewc_but_has_nonzero_post_fit_old_anchor_penalty():
    learner = _learner(steps=3)
    for acquired in (6, 7, 8):
        old_anchor = learner.anchor.clone()
        row = learner.fit(_records(), acquired=acquired)
        assert row["ewc"] == 0.0
        assert row["post_step_ewc"] > 0
        assert row["ewc_proposal_difference"] > 0
        assert all(inner["ewc"] > 0 for inner in row["inner"][1:])
        offset = learner.vector[learner.ltm] - old_anchor
        expected = .5 * (1 - learner.recipe.pi) / learner.recipe.pi * learner.archive.quadratic(offset)
        assert row["post_step_ewc"] == pytest.approx(float(expected))
        _assert_exact(learner.anchor, learner.vector[learner.ltm])


class _Interrupted(RuntimeError):
    pass


@pytest.mark.parametrize("steps,fail_on", [(1, 1), (3, 1), (3, 3)])
def test_check_exception_restores_exact_learner_head_and_rng_state(steps, fail_on):
    learner = _learner(steps)
    records = _records()
    learner.fit(records, acquired=5)
    learner.retire(records[0])
    learner.choose(records[0]["feature"])
    before = copy.deepcopy(learner.state_dict())
    heads_before = copy.deepcopy(learner.heads.state_dict())
    global_rng = torch.get_rng_state().clone()
    calls = []
    interruption = _Interrupted("stop inside the real inner solve")

    def check():
        calls.append(True)
        _assert_exact(learner.target, before["target"])
        assert learner.version == before["version"]
        if len(calls) == fail_on:
            raise interruption

    with pytest.raises(_Interrupted) as error:
        learner.fit(records, acquired=7, check=check)
    assert error.value is interruption
    assert len(calls) == fail_on
    _assert_exact(learner.state_dict(), before)
    _assert_exact(learner.heads.state_dict(), heads_before)
    _assert_exact(torch.get_rng_state(), global_rng)

    uninterrupted = _learner(steps, head_seed=123)
    uninterrupted.load_state_dict(before)
    retried = learner.fit(records, acquired=7)
    expected = uninterrupted.fit(records, acquired=7)
    _assert_exact(_without_timing(retried), _without_timing(expected))
    _assert_exact(learner.state_dict(), uninterrupted.state_dict())


@pytest.mark.parametrize("steps", [1, 3])
def test_checkpoint_resume_fit_is_identical_to_uninterrupted(steps, tmp_path):
    uninterrupted = _learner(steps)
    records = _records()
    uninterrupted.fit(records, acquired=5)
    uninterrupted.retire(records[0])
    uninterrupted.choose(records[0]["feature"])
    path = tmp_path / "learner.pt"
    torch.save(uninterrupted.state_dict(), path)
    resumed = _learner(steps, head_seed=321)
    resumed.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    _assert_exact(resumed.state_dict(), uninterrupted.state_dict())
    for acquired in (6, 7):
        expected = uninterrupted.fit(records, acquired)
        actual = resumed.fit(records, acquired)
        _assert_exact(_without_timing(actual), _without_timing(expected))
        _assert_exact(resumed.state_dict(), uninterrupted.state_dict())
        _assert_exact(resumed.heads.state_dict(), uninterrupted.heads.state_dict())
        _assert_exact(resumed.choose(records[1]["feature"]),
                      uninterrupted.choose(records[1]["feature"]))
    _assert_exact(resumed.state_dict(), uninterrupted.state_dict())


def test_retirement_generates_one_finite_fisher_factor_and_one_admission(monkeypatch):
    learner = _learner()
    record = _records()[0]
    before = copy.deepcopy(learner.state_dict())
    factor_calls, admissions = [], []
    real_factor = learner_module.fisher_factor
    real_update = learner.archive.update

    def observe_factor(*args, **kwargs):
        result = real_factor(*args, **kwargs)
        factor_calls.append(result.clone())
        return result

    def observe_update(factors, alpha, beta):
        admissions.append((factors.clone(), alpha, beta))
        return real_update(factors, alpha=alpha, beta=beta)

    monkeypatch.setattr(learner_module, "fisher_factor", observe_factor)
    monkeypatch.setattr(learner.archive, "update", observe_update)
    diagnostic = learner.retire(record)
    assert len(factor_calls) == len(admissions) == 1
    assert factor_calls[0].shape == (len(learner.vector), 7 + learner.recipe.integration_draws)
    assert bool(torch.isfinite(factor_calls[0]).all())
    factors, alpha, beta = admissions[0]
    assert factors.shape == (len(learner.ltm), 9)
    assert bool(torch.isfinite(factors).all()) and float(factors.square().sum()) > 0
    _assert_exact(factors, factor_calls[0][learner.ltm])
    assert alpha == 1 - learner.recipe.pi and beta == learner.recipe.pi
    expected_diagonal = alpha * before["archive"]["diagonal"] + beta * factors.square().sum(1)
    _assert_exact(learner.archive.diagonal, expected_diagonal)
    assert bool(torch.isfinite(learner.archive.factor).all())
    assert learner.archive.factor.shape[1] <= learner.recipe.rank
    assert diagnostic["trace"] > 0 and learner.retired == before["retired"] + 1
    after = learner.state_dict()
    for key in ("vector", "target", "anchor", "moment", "evaluations", "version"):
        _assert_exact(after[key], before[key])
    for name in ("exploration", "targets", "fd"):
        _assert_exact(after["rng"][name], before["rng"][name])
    assert not torch.equal(after["rng"]["fisher"], before["rng"]["fisher"])


def test_positive_fd_has_an_actual_gradient_and_enters_the_real_fit(monkeypatch):
    learner = _learner(fd_strength=0.7)
    records = _records()
    before = copy.deepcopy(learner.state_dict())
    data, design, _ = learner.design(records)
    vector = learner.vector.clone().requires_grad_()
    output = learner.coordinates.predict(vector, data["feature"], data["action"])
    parts = data_loss(output, data["action"], design)
    base = (parts["nll"] + parts["disagreement"]).mean()
    draws = torch.randn((2, 2, 7), generator=learner.generators["fd"])
    probe = torch.zeros_like(vector)
    signs = 2 * torch.randint(2, (len(learner.stm),),
                              generator=learner.generators["fd"]) - 1
    probe[learner.stm] = signs.to(probe)
    shift = learner.recipe.fd_step * 0.5 * probe
    symmetric_kl = 0.5 * (
        joint_kl(learner.coordinates, vector + shift, vector - shift, data["feature"], design, draws)
        + joint_kl(learner.coordinates, vector - shift, vector + shift, data["feature"], design, draws)
    )
    # acquired=6 and capacity=4 make both replay ages exactly one.
    fd = learner.recipe.fd_strength * symmetric_kl.mean() / (
        learner.recipe.fd_step**2 * len(learner.stm)
    )
    fd_gradient, = torch.autograd.grad(fd, vector, retain_graph=True)
    expected_gradient, = torch.autograd.grad(base + fd, vector)
    assert float(fd.detach()) > 0
    assert bool(torch.isfinite(fd_gradient).all())
    assert float(fd_gradient[learner.stm].norm()) > 1e-7
    learner.load_state_dict(before)
    seen = []
    adaptive_metric = learner_module.adaptive_metric

    def observe_metric(gradient, *args, **kwargs):
        seen.append(gradient.clone())
        return adaptive_metric(gradient, *args, **kwargs)

    monkeypatch.setattr(learner_module, "adaptive_metric", observe_metric)
    result = learner.fit(records, acquired=6)
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], expected_gradient)
    assert result["fd"] == pytest.approx(float(fd.detach()), rel=1e-6)
    assert result["fd"] > 0
