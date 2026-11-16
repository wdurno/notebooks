"""Actual tiny model banks, functional candidates, and atomic CPU recovery."""

import copy
import json
from dataclasses import replace

import pytest
import torch

from numerical_experiment.picar.config import PicarConfig
from numerical_experiment.picar.learner import MemoryLearner
from numerical_experiment.picar.models import build_tiny_model
from numerical_experiment.picar_solver_repair.learner import SolverLearner
from numerical_experiment.picar_solver_repair.numerics import BETA2, RECIPE_STEPS
from test_picar_learner import batch, history


def make(recipe="proximal_1", *, dtype=torch.float64, cls=SolverLearner, **changes):
    config = PicarConfig()
    learning = replace(config.learning, learning_rate=.003, fd_strength=0., ridge=0.,
                       optimizer_steps=RECIPE_STEPS[recipe])
    learning = replace(learning, **changes)
    model = build_tiny_model(31, hidden_size=4, width=4, return_bins=3, context_length=2).to(dtype=dtype)
    return cls(model, learning, config.seeds(0), recipe=recipe)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def accepted_equal(a, b):
    for name in ("parameters", "optimizer", "diagonal", "anchor", "center", "previous_center",
                 "rng", "version", "retired_count", "update_count", "center_version", "event_count"):
        equal(a[name], b[name])


@pytest.mark.parametrize("recipe", list(RECIPE_STEPS))
def test_neutral_initialization_publishes_once_and_preserves_actor_tensors(recipe):
    live = make(recipe)
    before = live.state_dict()
    references = dict(live.params)
    metrics = live.fit(new=batch(), replay=None, acquired_count=2)
    assert live.version == live.model.encoder.publication == live.center_version == live.event_count == 1
    assert live.update_count == RECIPE_STEPS[recipe]
    assert metrics["gradient_evaluations"] == RECIPE_STEPS[recipe]
    assert metrics["fd_evaluations"] == 0 and metrics["diagnostic_gradient_evaluations"] == 0
    assert "post_total" not in metrics and metrics["pre_step"] == 1
    assert metrics["inner_steps"][0]["ewc"] == metrics["inner_steps"][0]["ridge"] == 0
    for name, p in live.params.items():
        assert p is references[name] and p.grad is None and p.dtype == torch.float64
        equal(live.previous_center[name], before["parameters"][name])
        equal(live.center[name], p.detach())
    assert not any(bool(v.count_nonzero()) for v in live.diagonal.values())
    json.dumps(metrics, allow_nan=False)


def test_actual_first_proximal_step_changes_with_ewc_but_metric_does_not():
    unprotected, protected = make(), make()
    protected.diagonal = {n: torch.full_like(v, 1e4) for n, v in protected.diagonal.items()}
    initial = unprotected.state_dict()
    a = unprotected.fit(new=batch(), replay=None, acquired_count=2)
    b = protected.fit(new=batch(), replay=None, acquired_count=2)
    assert a["ewc"] == b["ewc"] == 0
    assert b["first_step_ewc_proposal_difference_norm"] > 0 and b["first_step_ewc_suppression"] > .1
    assert b["post_ewc"] > 0
    equal(unprotected.optimizer.state_dict(), protected.optimizer.state_dict())
    def distance(live, bank):
        return sum(float((live.params[n].detach() - initial["parameters"][n]).square().sum()) for n in live.banks[bank])
    assert distance(protected, "ltm") < distance(unprotected, "ltm") * .1
    assert distance(protected, "stm") == pytest.approx(distance(unprotected, "stm"))


@pytest.mark.parametrize("recipe", ["proximal_1", "lagged_1"])
def test_ridge_changes_first_stm_step_in_both_recipes(recipe):
    free, ridged = make(recipe), make(recipe, ridge=100.)
    for live in (free, ridged):
        live.diagonal = {n: torch.ones_like(v) for n, v in live.diagonal.items()}
    before = free.state_dict()
    free.fit(new=batch(), replay=None, acquired_count=2)
    metrics = ridged.fit(new=batch(), replay=None, acquired_count=2)
    norm = lambda live: sum(float((live.params[n].detach() - before["parameters"][n]).square().sum()) for n in live.banks["stm"])
    assert norm(ridged) < norm(free) * .1 and metrics["ridge"] == 0 and metrics["post_ridge"] > 0
    equal(free.optimizer.state_dict(), ridged.optimizer.state_dict())


def test_candidate_inner_gradients_and_moments_use_live_objective_only():
    class Observed(SolverLearner):
        def _live(self, params, *args):
            result = super()._live(params, *args)
            gradients = torch.autograd.grad(result[0], tuple(params.values()), retain_graph=True, allow_unused=True)
            self.observed.append({n: torch.zeros_like(p) if g is None else g.detach().clone()
                                  for (n, p), g in zip(params.items(), gradients)})
            return result
    live = make("proximal_3", cls=Observed, ridge=2.)
    live.observed = []
    live.diagonal = {n: torch.full_like(v, 3.) for n, v in live.diagonal.items()}
    metrics = live.fit(new=batch(), replay=None, acquired_count=2)
    assert len(live.observed) == 3 and metrics["inner_steps"][1]["ewc"] > 0
    assert any(not torch.equal(live.observed[0][n], live.observed[1][n]) for n in live.params)
    for name in live.params:
        expected = torch.zeros_like(live.params[name])
        for gradients in live.observed:
            expected = BETA2 * expected + (1-BETA2)*gradients[name].square()
        equal(expected, live.optimizer.second_moments[name])


def test_centers_targets_probes_and_actor_publication_stay_frozen_across_three_steps():
    class Observed(SolverLearner):
        def _live(self, params, new, replay, probes, ages):
            self.records.append((new.labels.clone(), replay.baselines.clone(),
                                 copy.deepcopy(probes), copy.deepcopy(self.center)))
            return super()._live(params, new, replay, probes, ages)
    live = make("proximal_3", cls=Observed, fd_strength=.1, ridge=.1)
    live.records = []
    source_new, source_replay = batch(), batch()
    old = live.state_dict()
    calls = []
    def handoff():
        for name, p in live.params.items():
            equal(p.detach(), old["parameters"][name])
        assert live.version == live.model.encoder.publication == 0
        live.choose_action(history(), explore=False)
        source_new.labels[:] = 0
        source_replay.baselines[:] = 100
        calls.append(True)
    metrics = live.fit(new=source_new, replay=source_replay, acquired_count=4, cooperate=handoff)
    assert len(calls) >= 3 and len(live.records) == 3
    for recorded in live.records[1:]:
        equal(recorded, live.records[0])
    assert metrics["fd_evaluations"] == 3 and live.version == 1


@pytest.mark.parametrize("recipe", list(RECIPE_STEPS))
def test_default_has_no_extra_network_pass_and_diagnostics_are_counted(recipe):
    class Observed(SolverLearner):
        def _live(self, *args):
            self.calls += 1
            return super()._live(*args)
    a, b = make(recipe, cls=Observed), make(recipe, cls=Observed)
    a.calls = b.calls = 0
    ordinary = a.fit(new=batch(), replay=None, acquired_count=2)
    diagnosed = b.fit(new=batch(), replay=None, acquired_count=2, diagnostics=True)
    assert a.calls == RECIPE_STEPS[recipe] and b.calls == a.calls + 1
    assert diagnosed["diagnostic_gradient_evaluations"] == 1
    assert diagnosed["post_total"] >= diagnosed["post_data"]
    assert diagnosed["post_centered_total"] == pytest.approx(diagnosed["post_total"])
    assert "post_total" not in ordinary
    accepted_equal(a.state_dict(), b.state_dict())


def test_lagged_common_centered_diagnostic_matches_direct_autograd_without_extra_pass():
    live = make("lagged_1", ridge=.1)
    live.diagonal = {n: torch.full_like(v, .01) for n, v in live.diagonal.items()}
    source = batch()
    live.fit(new=source, replay=None, acquired_count=2)
    current = {n: p.clone() for n, p in live.center.items()}
    previous = {n: p.clone() for n, p in live.previous_center.items()}
    metrics = live.fit(new=source, replay=None, acquired_count=2, diagnostics=True)
    omega = (1-live.config.pi)/live.config.pi
    scale = sum(v.sum() for v in live.diagonal.values()) / sum(live.params[n].numel() for n in live.diagonal)
    ridge = omega * live.config.ridge * scale.clamp_min(live.config.ridge_floor)
    data = live.nll(live.params, source).mean()
    ewc = .5*omega*sum((live.diagonal[n]*(live.params[n]-current[n]).square()).sum() for n in live.diagonal)
    ridge_loss = .5*ridge*sum((p-current[n]).square().sum() for n, p in live.params.items())
    expected = data+ewc+ridge_loss
    grads = torch.autograd.grad(expected, tuple(live.params.values()))
    norm = float(torch.sqrt(sum(g.double().square().sum() for g in grads)))
    assert metrics["post_centered_total"] == pytest.approx(float(expected.detach()), rel=1e-10)
    assert metrics["post_centered_gradient_residual_norm"] == pytest.approx(norm, rel=1e-10)
    assert metrics["post_total"] != metrics["post_centered_total"]
    assert any(not torch.equal(current[n], previous[n]) for n in current)


def test_retirement_only_updates_fisher_not_either_center_or_publication():
    live = make("proximal_3")
    live.fit(new=batch(), replay=None, acquired_count=2)
    before = live.state_dict()
    for index in range(2):
        metrics = live.retire(batch())
        assert metrics["retirement_eta"] == pytest.approx(1-(1-live.config.pi)**2)
        after = live.state_dict()
        for name in ("center", "previous_center", "anchor", "parameters", "optimizer", "version", "center_version", "event_count"):
            equal(before[name], after[name])
        assert live.retired_count == 2*(index+1)
    assert sum(float(v.sum()) for v in live.diagonal.values()) > 0


def test_legacy_bootstrap_conversion_resets_metric_and_centers_without_mutating_asset():
    live = make()
    legacy = MemoryLearner(live.model, live.config, PicarConfig().seeds(0))
    legacy.fit(new=batch(), replay=None, acquired_count=2)
    legacy.retire(batch())
    old = legacy.state_dict()
    saved = copy.deepcopy(old)
    live.load_state_dict(old)
    assert live.update_count == live.optimizer.gradient_evaluations == live.event_count == 0
    assert live.retired_count == old["retired_count"]
    for n in live.params:
        equal(live.center[n], old["parameters"][n])
        equal(live.previous_center[n], old["parameters"][n])
        assert not bool(live.optimizer.second_moments[n].count_nonzero())
    temporary = MemoryLearner(build_tiny_model(31, hidden_size=4, width=4, return_bins=3, context_length=2).double(),
                              live.config, PicarConfig().seeds(0))
    temporary.load_state_dict(live.legacy_state_dict())
    assert temporary.optimizer.state_dict()["state"] == {}
    equal(old, saved)


@pytest.mark.parametrize("recipe", list(RECIPE_STEPS))
def test_abrupt_resume_before_publication_replays_frozen_event_exactly(recipe):
    reference, interrupted = make(recipe, fd_strength=.1), make(recipe, fd_strength=.1)
    expected = reference.fit(new=batch(), replay=batch(), acquired_count=4)
    saved = []
    def checkpoint_then_interrupt():
        if not saved and interrupted._event["completed_inner_steps"] >= 1:
            saved.append(interrupted.state_dict())
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        interrupted.fit(new=batch(), replay=batch(), acquired_count=4, cooperate=checkpoint_then_interrupt)
    assert saved[0]["pending_event"]["replay_from_inner"] == 0
    assert saved[0]["pending_event"]["completed_inner_steps"] == 1
    resumed = make(recipe, fd_strength=.1)
    resumed.load_state_dict(saved[0])
    actual = resumed.fit(new=batch(), replay=batch(), acquired_count=4)
    accepted_equal(reference.state_dict(), resumed.state_dict())
    for key in ("total", "data", "fd", "post_ewc", "post_ridge"):
        assert actual[key] == expected[key]


def test_nonfinite_gradient_rolls_back_even_if_the_live_value_is_finite():
    class InvalidGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.new_zeros(())

        @staticmethod
        def backward(ctx, gradient):
            return gradient.new_full((), float("nan"))

    class Broken(SolverLearner):
        def _live(self, params, *args):
            live, data, fd = super()._live(params, *args)
            first = next(iter(params.values())).reshape(-1)[0]
            return live + InvalidGradient.apply(first), data, fd

    live = make(cls=Broken)
    before = live.state_dict()
    with pytest.raises(FloatingPointError, match="nonfinite live gradient"):
        live.fit(new=batch(), replay=None, acquired_count=2)
    accepted_equal(before, live.state_dict())


@pytest.mark.parametrize("retired_before_checkpoint", [False, True])
def test_resume_between_publication_and_exactly_once_retirement(retired_before_checkpoint):
    reference, live = make("proximal_3"), make("proximal_3")
    for learner in (reference, live):
        learner.fit(new=batch(), replay=None, acquired_count=2)
    reference.retire(batch())
    if retired_before_checkpoint:
        live.retire(batch())
    # The caller persists this flag/retirement ID atomically with learner state.
    checkpoint = {"learner": live.state_dict(), "retired": retired_before_checkpoint}
    resumed = make("proximal_3")
    resumed.load_state_dict(checkpoint["learner"])
    if not checkpoint["retired"]:
        resumed.retire(batch())
    accepted_equal(reference.state_dict(), resumed.state_dict())


def test_interruption_after_inner_step_rolls_back_but_preserves_actor_rng_progress():
    live, actor = make("proximal_3", fd_strength=.1), make("proximal_3", fd_strength=.1)
    before = live.state_dict()
    calls = []
    def handoff():
        if live.records >= 2:
            live.choose_action(history())
            calls.append(True)
            raise RuntimeError("after inner step")
    original = live._live
    live.records = 0
    def observe(*args):
        live.records += 1
        return original(*args)
    live._live = observe
    with pytest.raises(RuntimeError, match="after inner step"):
        live.fit(new=batch(), replay=batch(), acquired_count=4, cooperate=handoff)
    actor.choose_action(history())
    after = live.state_dict()
    expected_rng = actor.generators["exploration"].get_state()
    assert calls and torch.equal(after["rng"]["exploration"], expected_rng)
    before["rng"]["exploration"] = expected_rng
    accepted_equal(before, after)
    assert live._cooperate is None and not live._active


def test_failure_during_publication_restores_all_accepted_state():
    live = make("proximal_3")
    before = live.state_dict()
    original = live.model.mark_published
    def fail():
        original()
        raise RuntimeError("publication failed")
    live.model.mark_published = fail
    with pytest.raises(RuntimeError, match="publication failed"):
        live.fit(new=batch(), replay=None, acquired_count=2)
    accepted_equal(before, live.state_dict())
    assert live.model.encoder.publication == 0


def test_strict_resume_validation_is_atomic_and_state_does_not_alias():
    live = make()
    live.fit(new=batch(), replay=None, acquired_count=2)
    state = live.state_dict()
    changed = make(learning_rate=.004)
    with pytest.raises(ValueError, match="configuration differs"):
        changed.load_state_dict(state)
    before = live.state_dict()
    broken = copy.deepcopy(state)
    broken["optimizer"]["second_moments"][next(iter(live.params))].fill_(-1)
    with pytest.raises(ValueError, match="second moment"):
        live.load_state_dict(broken)
    accepted_equal(before, live.state_dict())
    child = make()
    child.load_state_dict(state)
    child.fit(new=batch(), replay=None, acquired_count=2)
    accepted_equal(before, state)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_updates_report_realized_actor_dtype_displacement(dtype):
    live = make(dtype=dtype, learning_rate=1e-14)
    before = live.state_dict()
    metrics = live.fit(new=batch(), replay=None, acquired_count=2)
    realized = sum((p.detach().double()-before["parameters"][n].double()).square().sum()
                   for n, p in live.params.items()).sqrt()
    assert metrics["inner_steps"][0]["realized_step_norm"] == pytest.approx(float(realized), abs=1e-25)
    assert metrics["inner_steps"][0]["proposed_step_norm"] > 0
    assert metrics["inner_steps"][0]["rounding_error_norm"] >= 0
    for n, p in live.params.items():
        assert p.dtype == dtype
        equal(live.center[n], p.detach())


def test_invalid_step_override_and_stale_event_id_do_not_mutate_state():
    live = make("proximal_3")
    before = live.state_dict()
    with pytest.raises(ValueError, match="declared count"):
        live.fit(new=batch(), replay=None, acquired_count=2, steps=1)
    with pytest.raises(ValueError, match="event ID"):
        live.fit(new=batch(), replay=None, acquired_count=2, event_id=1)
    equal(before, live.state_dict())
