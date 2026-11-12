"""Cheap deterministic development scales and reversible probe checks."""

from dataclasses import replace

import pytest
import torch

from numerical_experiment.picar_phase3.calibration import bootstrap_statistics, explore_controls
from numerical_experiment.picar_phase3.config import (Recipe, configuration, likelihood_key,
    DEVELOPMENT_SEEDS, CONFIRMATION_SEEDS, panels)
from src.robotics.actions import Calibration


def records():
    return [{"return": value, "baseline": 0., "logits": torch.zeros(7)}
            for value in [-5.]*24+[9.]*8]


def test_bootstrap_temperature_is_bounded_and_teacher_covariance_not_invented():
    result = bootstrap_statistics(records())
    assert len(result["temperature_rows"]) == 3
    assert result["selected_temperature"] in [r["temperature"] for r in result["temperature_rows"]]
    assert result["teacher_logit_variance_max"] == 0
    assert result["action_variance"] == 1
    assert not result["weight_effective_count_is_iid_sample_size"]
    assert result["new_observations"] == 0


def test_bootstrap_rejects_broken_reward_or_baseline():
    values = records()
    values[0]["return"] = 13
    with pytest.raises(ValueError, match="bound"):
        bootstrap_statistics(values)
    values[0]["return"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        bootstrap_statistics(values)


def test_constant_high_returns_do_not_select_float32_overflow_temperature():
    result = bootstrap_statistics([{**r,"return":12.} for r in records()])
    selected = next(r for r in result["temperature_rows"] if r["temperature"] == result["selected_temperature"])
    assert selected["numeric_weight_safe"]


def test_control_probe_is_named_seed_deterministic_and_not_environment_data():
    means = [[0.,0.,6.9,0.,0.,0.,0.]]
    first = explore_controls(means, Calibration(), seed=123, draws=64)
    assert first == explore_controls(means, Calibration(), seed=123, draws=64)
    assert first != explore_controls(means, Calibration(), seed=124, draws=64)
    assert first["new_observations"] == 0 and first["hypothetical_draws"] == 64
    for row in first["exploration_rows"]:
        assert -.1 <= row["drive_mean_mps"] <= .2
        assert 0 <= row["forward_steering_fraction"] <= 1
    with pytest.raises(ValueError):
        explore_controls([[0.,1.]], Calibration())


def test_treatment_regularizers_and_optimizer_do_not_change_initial_fisher_identity():
    base = Recipe("control")
    variant = replace(base,name="fd",ridge=.03,learning_rate=.003,optimizer_steps=2,fd_strength=100.)
    assert likelihood_key(base) == likelihood_key(variant)
    assert configuration(variant,bootstrap=True).learning.fd_strength == 0
    assert configuration(variant,bootstrap=True).learning.learning_rate == .001
    assert likelihood_key(base) != likelihood_key(replace(base,temperature=2.))
    assert likelihood_key(base) != likelihood_key(replace(base,action_variance=2.))
    assert configuration(variant).learning.optimizer_steps == 2


def test_development_is_disjoint_from_confirmation_and_panels_fixed():
    assert not set(DEVELOPMENT_SEEDS)&set(CONFIRMATION_SEEDS)
    result = panels(Recipe("control"))
    assert [len(p["seeds"]) for p in result] == [20,10]
    assert all(p["config"].realtime for p in result)
    assert result[1]["config"].curriculum_exposure_override == 640
    with pytest.raises(ValueError,match="ramp"):
        configuration(Recipe("too short",transitions=320))
