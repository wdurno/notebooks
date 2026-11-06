"""Independent quadratic and physical-pixel checks for regression learning."""

import math

import pytest
import torch

from mnist_experiment.kalman_pi.registration import learner


def _linear(model, layout, theta, inputs):
    active = inputs * theta
    return torch.cat((active, active.new_zeros((len(active), 4))), dim=-1)


@pytest.mark.parametrize("pi,ridge", [(.025, 0.), (.1, .1), (.2, 1.)])
def test_regression_solve_matches_exact_quadratic(pi, ridge, monkeypatch):
    monkeypatch.setattr(learner, "predict", _linear)
    x = torch.ones((4, 2), dtype=torch.float64)
    y = torch.tensor([[2., -1., 0., 0., 0., 0.]], dtype=torch.float64).expand(4, -1)
    start = torch.tensor([.4, -.2], dtype=torch.float64)
    diagonal = torch.tensor([.25, .5], dtype=torch.float64)
    theta, metrics = learner.fit_batch(None, None, start, diagonal, x, y,
                                      pi=pi, ridge=ridge, tolerance_grad=1e-10,
                                      tolerance_change=1e-14)
    precision = (1 - pi) / pi * (diagonal + ridge * diagonal.mean())
    expected = (.25 * y[0, :2] + precision * start) / (.25 + precision)
    torch.testing.assert_close(theta, expected, atol=2e-7, rtol=2e-7)
    assert metrics["objective"] <= float(learner.mean_nll(_linear(None, None, start, x), y))
    assert metrics["kappa"] == pytest.approx(ridge * float(diagonal.mean()))
    assert metrics["closure_count"] <= 75


def test_evaluation_is_landmark_distance_and_fixed_class_weighted(monkeypatch):
    monkeypatch.setattr(learner, "predict", lambda m, l, t, x: x)
    x = torch.zeros((10, 6), dtype=torch.float64)
    y = torch.zeros_like(x)
    y[9] = 1.
    weights = [.1] * 10
    metrics = learner.evaluate(None, None, torch.zeros(1), x, y, torch.arange(10), weights)
    assert metrics["risk"] == pytest.approx(.2)
    assert metrics["rms"] == pytest.approx(math.sqrt(.2))
    assert metrics["accuracy"] == pytest.approx(.9)
    assert metrics["nll"] == pytest.approx(3 * .2 / 8 + 3 * math.log(8 * math.pi))
    assert metrics["per_digit_risk"][9] == 2.


def test_stop_is_checked_inside_local_solve(monkeypatch):
    monkeypatch.setattr(learner, "predict", _linear)
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("requested-stop")

    with pytest.raises(RuntimeError, match="requested-stop"):
        learner.fit_batch(None, None, torch.zeros(2), torch.ones(2),
            torch.ones(4, 2), torch.ones(4, 6), pi=.025, ridge=0., check=stop)
