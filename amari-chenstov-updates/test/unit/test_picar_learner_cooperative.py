import pytest
import torch

from test_picar_learner import batch, history, learner


def test_cooperative_fit_matches_atomic_fit_and_actor_sees_only_old_parameters():
    reference, live = learner(), learner()
    old = {name: p.detach().clone() for name, p in live.params.items()}
    calls = []
    def cooperate():
        for name, parameter in live.model.named_parameters():
            if name in old:
                torch.testing.assert_close(parameter, old[name], atol=0, rtol=0)
        assert live.version == 0
        live.choose_action(history(), explore=False)
        calls.append(True)
    a = reference.fit(new=batch(), replay=batch(), acquired_count=4)
    b = live.fit(new=batch(), replay=batch(), acquired_count=4, cooperate=cooperate)
    assert len(calls) >= 4 and live.version == 1
    for key in a:
        if key != "update_seconds":
            assert b[key] == pytest.approx(a[key], abs=1e-7)
    for name in live.params:
        torch.testing.assert_close(live.params[name], reference.params[name], atol=0, rtol=0)
        torch.testing.assert_close(live.params[name].grad, reference.params[name].grad, atol=0, rtol=0)


def test_failed_yielded_update_keeps_actor_noise_consumption_but_rolls_back_learning():
    live, reference = learner(), learner()
    before = live.state_dict()
    reference.choose_action(history(), explore=True)
    expected = reference.generators["exploration"].get_state().clone()
    def fail_after_actor():
        live.choose_action(history(), explore=True)
        raise RuntimeError("interrupted update")
    with pytest.raises(RuntimeError, match="interrupted"):
        live.fit(new=batch(), replay=None, acquired_count=4, cooperate=fail_after_actor)
    assert torch.equal(live.generators["exploration"].get_state(), expected)
    for name, value in live.params.items():
        torch.testing.assert_close(value.cpu(), before["parameters"][name], atol=0, rtol=0)
    assert live.version == live.update_count == 0 and live._cooperate is None
    assert torch.equal(live.generators["fd_probes"].get_state(), before["rng"]["fd_probes"])


def test_multistep_event_never_yields_after_first_parameter_mutation():
    live = learner()
    original = {name: p.detach().clone() for name, p in live.params.items()}
    calls = []
    def cooperate():
        calls.append(True)
        for name, p in live.params.items():
            torch.testing.assert_close(p, original[name], atol=0, rtol=0)
    live.fit(new=batch(), replay=None, acquired_count=4, steps=2, cooperate=cooperate)
    assert calls and live.version == 1 and live.update_count == 2
