import pytest
import torch
from torch.func import jacrev

from src.parameters import ParameterLayout
from mnist_experiment.diagonal_lfu.curvature import (
    build_model, diagonal_function, logits, observation, observe_update,
)
from mnist_experiment.diagonal_lfu.learner import evaluate, fit_batch


def tiny():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2)).double()
    layout = ParameterLayout.from_module(model)
    theta = layout.flatten_module(model, detach=True)
    return model, layout, theta, torch.randn(3, 2, dtype=torch.float64)


def test_label_sum_matches_scores_and_includes_probability_derivative():
    model, layout, theta, x = tiny()
    measured = observation(model, layout, theta, x)
    result = torch.zeros_like(theta)
    for image in x:
        logp = lambda t: logits(model, layout, t, image[None])[0].log_softmax(-1)
        score = jacrev(logp)(theta)
        result += (logp(theta).exp()[:, None] * score.square()).sum(0) / len(x)
    torch.testing.assert_close(measured, result)
    direction = torch.randn_like(theta) * .01
    _, data = observe_update(model, layout, theta, theta + direction, measured, x,
                            condition="analytic_lfu", c=.05)
    assert data["correction_norm"] > 0


def test_paired_zero_step_and_storm_algebra():
    model, layout, theta, x = tiny()
    old = observation(model, layout, theta, x)
    unchanged, data = observe_update(model, layout, theta, theta, old, x,
                                     condition="paired_fd", c=.1)
    torch.testing.assert_close(unchanged, old)
    assert data["correction_norm"] == 0
    current = theta + .01 * torch.randn_like(theta)
    fresh = observation(model, layout, current, x)
    result, _ = observe_update(model, layout, theta, current, old, x,
                               condition="paired_fd", c=.1)
    torch.testing.assert_close(result, fresh)


def test_analytic_lfu_converges_to_endpoint_difference():
    model, layout, theta, x = tiny()
    old = observation(model, layout, theta, x)
    direction = torch.randn_like(theta)
    errors = []
    for step in (.01, .005):
        current = theta + step * direction
        a, _ = observe_update(model, layout, theta, current, old, x, condition="analytic_lfu", c=.1)
        b, _ = observe_update(model, layout, theta, current, old, x, condition="paired_fd", c=.1)
        errors.append(float((a - b).norm()))
    assert errors[1] < .4 * errors[0]


def test_fixed_pooling_preserves_canonical_model():
    model, layout = build_model(2, dtype=torch.float64)
    x = torch.randn(2, 1, 28, 28, dtype=torch.float64)
    fixed = model(x)
    model.features[-1] = torch.nn.AdaptiveAvgPool2d((2, 2))
    torch.testing.assert_close(fixed, model(x))
    assert layout.total_numel == 487


def test_fit_records_evaluated_objective_and_decreases_it():
    model, layout, theta, x = tiny()
    y = torch.tensor([0, 1, 0])
    diagonal = observation(model, layout, theta, x)
    result, metrics = fit_batch(model, layout, theta, diagonal, x, y, c=.05,
                                ridge_ratio=.1, max_iter=5, max_eval=10)
    assert metrics["closure_count"] <= 10
    initial = torch.nn.functional.cross_entropy(logits(model, layout, theta, x), y)
    assert metrics["objective"] <= float(initial) + 1e-10
    assert torch.isfinite(result).all()
    candidate = result.detach().requires_grad_(True)
    precision = 19 * (diagonal + .1 * diagonal.mean())
    value = (torch.nn.functional.cross_entropy(logits(model, layout, candidate, x), y)
             + .5 * (precision * (candidate - theta).square()).sum())
    gradient = torch.autograd.grad(value, candidate)[0]
    assert metrics["grad_norm"] == pytest.approx(float(gradient.abs().max()), abs=1e-12)
