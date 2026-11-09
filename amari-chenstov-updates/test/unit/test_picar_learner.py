from dataclasses import replace

import pytest
import torch

from numerical_experiment.picar.config import PicarConfig
from numerical_experiment.picar.learner import LearningBatch, MemoryLearner, predict_histories
from numerical_experiment.picar.models import build_tiny_model


def history(length=1):
    generator = torch.Generator().manual_seed(7)
    return {"images": torch.rand(1, length, 3, 8, 8, generator=generator),
            "executed_actions": torch.zeros(1, length, 4),
            "elapsed": torch.full((1, length, 1), .1),
            "contexts": [[{"text": "Approach the ball and stop.", "issued_at": 0.}]]}


def batch():
    return LearningBatch([history(1), history(2)], torch.zeros(2),
                         torch.tensor([1, 2]), torch.ones(2, 7), torch.tensor([0, 1]))


def learner(fd=.1):
    config = PicarConfig()
    model = build_tiny_model(31, hidden_size=4, width=4, return_bins=3, context_length=2)
    return MemoryLearner(model, replace(config.learning, fd_strength=fd), config.seeds(0))


def test_variable_history_batch_is_recordwise_equivalent():
    model = build_tiny_model(31, hidden_size=4, width=4, return_bins=3, context_length=2)
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    histories = [history(1), history(2), history(1)]
    grouped = predict_histories(model, params, histories)
    for key in grouped:
        expected = torch.cat([model(h)[key] for h in histories])
        torch.testing.assert_close(grouped[key], expected, rtol=2e-5, atol=2e-6)


def test_learning_retirement_and_restore():
    live = learner()
    initial = live.state_dict()
    metrics = live.fit(new=batch(), replay=batch(), acquired_count=4)
    assert metrics["fd"] >= 0 and metrics["update_seconds"] > 0
    assert live.version == 1
    assert any(not torch.equal(initial["parameters"][name], p.cpu()) for name, p in live.params.items())
    live.retire(batch())
    assert live.retired_count == 2
    assert sum(float(x.sum()) for x in live.diagonal.values()) > 0
    state = live.state_dict()
    restored = learner()
    restored.load_state_dict(state)
    a = live.choose_action(history())
    b = restored.choose_action(history())
    torch.testing.assert_close(a["logits"], b["logits"], rtol=0, atol=0)
    live.fit(new=None, replay=batch(), acquired_count=4)
    restored.fit(new=None, replay=batch(), acquired_count=4)
    for name in live.params:
        torch.testing.assert_close(live.params[name], restored.params[name], rtol=0, atol=0)


def test_failure_rolls_back_parameters_rng_and_counters():
    live = learner()
    before = live.state_dict()
    bad = batch()
    bad.baselines[:] = float("nan")
    with pytest.raises(ValueError):
        live.fit(new=bad, replay=batch(), acquired_count=4)
    after = live.state_dict()
    for group in ("parameters", "rng"):
        for name in before[group]:
            assert torch.equal(before[group][name], after[group][name])
    assert live.version == live.retired_count == live.update_count == 0


def test_control_disables_only_fd():
    live = learner(0)
    metrics = live.fit(new=None, replay=batch(), acquired_count=4)
    assert metrics["fd"] == 0
    assert live.banks["stm"] and live.banks["ltm"]


def test_shared_bootstrap_optimizer_state_does_not_alias_condition():
    import copy
    live = learner()
    live.fit(new=batch(), replay=None, acquired_count=2)
    shared = live.state_dict()
    original = copy.deepcopy(shared["optimizer"])
    child = learner()
    child.load_state_dict(shared)
    child.fit(new=batch(), replay=None, acquired_count=2)
    for key, values in original["state"].items():
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, shared["optimizer"]["state"][key][name])


def test_latest_acquisition_has_zero_replay_age(monkeypatch):
    from src.continual import picar_math as pm
    original = pm.replay_ages
    observed = []
    def ages(count, arrivals, capacity):
        result = original(count, arrivals, capacity)
        observed.append(result)
        return result
    monkeypatch.setattr(pm, "replay_ages", ages)
    live = learner()
    live.fit(new=None, replay=batch(), acquired_count=2)
    assert observed[0][-1] == 0
