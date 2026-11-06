"""Conditional risk identities and causal recommendations."""

from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.kalman_pi.discrepancy.policies import Controller, PolicySpec
from mnist_experiment.kalman_pi.policies import Controller as OldController, PolicySpec as OldSpec


def config():
    return SimpleNamespace(observer_floor_absolute=1e-12, observer_floor_relative=1e-6,
        initialization_size=100, noise_half_life=32., cold_start=2, pi_min=.0001, pi_max=.2)


def controller(kind):
    obj = Controller(PolicySpec(kind, .0001), config(), .025)
    obj.initialize(torch.zeros(2), torch.ones(2), torch.tensor([2., 3.]))
    return obj


def test_stationary_population_with_learner_lag_requires_catchup():
    obs = controller("discrepancy")
    obs.state["covariance"].zero_()
    obs.state["k0"].zero_()
    theta = torch.tensor([.5, -.25])
    result = obs.choose(.02, torch.ones(2), 4, 20, theta)
    expected_signal = float(theta.square().sum())
    assert result["motion_energy"] == 0
    assert result["discrepancy_energy"] == expected_signal
    assert result["raw_pi"] == pytest.approx(expected_signal / (expected_signal + 5 / 4))
    assert result["shadow_velocity_pi"] != result["raw_pi"]


def test_predicted_level_keeps_cross_covariance_and_no_extra_q():
    obs = controller("discrepancy")
    obs.state["covariance"][:] = torch.tensor([[.1, -.02], [-.02, .03]], dtype=torch.float64)
    obs.state["mean"][:] = torch.tensor([[.1, .2], [-.1, .3]], dtype=torch.float64)
    theta = torch.tensor([.2, .1])
    first = obs.choose(.01, torch.ones(2), 4, 10, theta)
    second = obs.choose(.2, torch.ones(2), 4, 10, theta)
    variance = 2 * (.1 - .04 + .03) + float(obs.process_covariance()[:, 0, 0].sum())
    assert first["level_uncertainty_energy"] == pytest.approx(variance)
    assert first["raw_pi"] == second["raw_pi"]
    assert first["shadow_velocity_pi"] != second["shadow_velocity_pi"]


def test_velocity_matches_frozen_inherited_formula():
    obs = controller("velocity")
    old = OldController(OldSpec("kalman", .0001), config(), .025, state=obs.state)
    for i in range(8):
        scores = torch.tensor([[.2, -.1], [.1, .3], [.4, -.2], [-.1, .2]]) + i * .01
        obs.assimilate(scores, i)
    actual = obs.choose(.003, torch.tensor([1., .5]), 4, 8, torch.tensor([1., -1.]))
    reference = old.choose(.003, torch.tensor([1., .5]), 4, 8)
    for key in ("raw_pi", "applied_pi", "motion_energy", "uncertainty_energy", "noise_scale"):
        assert actual[key] == pytest.approx(reference[key], rel=1e-14)


def test_risk_minimizer_and_replica_marginal_identity():
    old_errors = torch.tensor([[-.2, .1], [.2, -.1]], dtype=torch.float64)
    movement = torch.tensor([.3, -.4], dtype=torch.float64)
    metric = torch.tensor([2., 3.], dtype=torch.float64)
    discrepancies = movement - old_errors
    signal = float((discrepancies.square() * metric).sum(1).mean())
    marginal = float((movement.square() * metric).sum() + (old_errors.square().mean(0) * metric).sum())
    assert signal == pytest.approx(marginal)
    noise = .7
    pi = signal / (signal + noise)
    risk = lambda p: (1-p)**2 * signal + p**2 * noise
    assert risk(pi) < risk(pi-.01) and risk(pi) < risk(pi+.01)


def test_cold_start_and_actual_learner_required():
    obs = controller("discrepancy")
    assert obs.choose(.01, torch.ones(2), 4, 0, torch.ones(2))["applied_pi"] == .025
    with pytest.raises(ValueError):
        obs.choose(.01, torch.ones(2), 4, 5, torch.ones(3))
    assert PolicySpec.parse("discrepancy_eta0.0001").condition == "discrepancy_eta0.0001"
