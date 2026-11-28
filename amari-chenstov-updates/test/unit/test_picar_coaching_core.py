"""Causal language, reward arithmetic and initialization isolation checks."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from numerical_experiment.picar.models import batch_digest, build_tiny_model, render_context
from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_coaching.config import recipe, selected_recipe
from numerical_experiment.picar_coaching.history import History
from numerical_experiment.picar_coaching.initialization import (
    distill, representation_parameters, validation_episode,
)
from numerical_experiment.picar_coaching.scoring import parse, reward_components, validate_evidence
from src.robotics.actions import Calibration


def observation(t):
    return SimpleNamespace(capture_time_s=t, rgb=bytes(8 * 8 * 3), height=8, width=8,
                           executed_controls=Calibration().neutral())


def test_operator_events_are_causal_and_change_feature_cache_key():
    h = History(SimpleNamespace(model=SimpleNamespace(context_length=4)),
                SimpleNamespace(deadline_s=20., width=8, height=8))
    before = h(observation(1.))
    h.update([{"text": "Turn left", "issued_at": 2., "completed_at": 4.}],
             [{"text": "done", "available_at": 4.}])
    assert batch_digest(h(observation(1.))) == batch_digest(before)
    at_three = h(observation(3.))
    rendered = json.loads(render_context(at_three, 0, 1))
    assert rendered["instructions"][-1]["status"] == "active"
    assert not rendered["status_events"]
    assert batch_digest(at_three) != batch_digest(before)
    restored = History(h.config, h.sim_config, h.snapshot())
    assert batch_digest(restored(observation(3.))) == batch_digest(at_three)


def test_rewards_square_only_task_and_keep_guidance_signed_and_separate():
    parts = reward_components([1., 1., 1.], [2., 2., 2.], [10., -5., -1.], [2., -2., 0.])
    torch.testing.assert_close(parts["task"], torch.tensor([100., -25., -1.], dtype=torch.float64))
    torch.testing.assert_close(parts["encoded"], torch.tensor([10.2, -2.7, -.1], dtype=torch.float64))
    with pytest.raises(ValueError):
        reward_components([1], [1], [0], [4])


def test_evaluator_inputs_reject_geometry_and_out_of_window_controls():
    evidence = {"window_id": "w1", "instructions": [{"text": "Turn left", "issued_at": 1.,
        "completed_at": None, "status": "active"}], "start_time": 1., "end_time": 3.,
        "status": "expired", "controls": [{"start_time": 1.5, "end_time": 2.5,
        "executed": [0., 0., -.4, .2], "fallback": False}]}
    validate_evidence(evidence)
    with pytest.raises(ValueError, match="visible inputs"):
        validate_evidence({**evidence, "position": [1, 2]})
    with pytest.raises(ValueError, match="outside"):
        validate_evidence({**evidence, "controls": [{**evidence["controls"][0], "start_time": .5}]})
    assert parse('{"judgment":"violated","reason":"opposite steering"}')["judgment"] == "violated"
    with pytest.raises(ValueError):
        parse('{"judgment":"adherent","reason":"ok","reward":2}')


def test_recipe_is_same_rank64_selected_contract():
    cfg = recipe()
    assert cfg.rank == 64 and cfg.capacity == 256 and cfg.fd_strength == 12.142951965332031


def test_distillation_changes_only_ltm_and_resumes_sealed_fit(tmp_path):
    model = NormalizedStudent(build_tiny_model(seed=9, hidden_size=4, width=8, context_length=2)).cpu()
    pristine = representation_parameters(model)
    records = []
    i = 0
    while len(records) < 4 or not any(r["validation"] for r in records):
        ep = "test:" + str(i)
        records.append({"episode_id": ep, "semantic": torch.ones(1, 2, 8) * (i + 1),
            "controls": torch.zeros(1, 2, 4), "elapsed": torch.ones(1, 2, 1),
            "teacher_probabilities": torch.tensor([.93, .01, .01, .01, .01, .01, .01, .01]),
            "validation": validation_episode(ep)})
        i += 1
    with RunStore(tmp_path / "fit", {"test": 1}, checkpoint_policy="journal-v1") as store:
        result = distill(model, records, pristine, store, lambda: None)
    assert result["stm_neutral"] and result["initial_pi_used"] is False
    after = representation_parameters(model)
    assert all(torch.equal(value, after[name]) for name, value in pristine.items() if ".stm." in name)
    assert any(not torch.equal(value, after[name]) for name, value in pristine.items() if ".ltm." in name)
    assert result["curves"][-1]["train_kl"] < result["curves"][0]["train_kl"]
    with RunStore(tmp_path / "fit", {"test": 1}, resume=True, checkpoint_policy="journal-v1") as store:
        resumed = distill(model, records, pristine, store, lambda: None)
    assert resumed == result
