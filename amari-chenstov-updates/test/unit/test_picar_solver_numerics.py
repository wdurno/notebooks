"""CPU acceptance of the declared solver, not another experimental condition."""

import math

import pytest
import torch

from numerical_experiment.picar_solver_repair.numerics import (
    BETA2, EPSILON, adaptive_metric, diagonal_step,
)


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize("recipe", ["proximal_1", "proximal_3", "lagged_1"])
def test_skewed_diagonal_subproblem_has_zero_analytic_derivative(recipe):
    x, center, previous = tensor([2., -1., .5, 7.]), tensor([1., -2., 3., 7.]), tensor([.5, -3., 2., 9.])
    g, metric, ewc, ridge = tensor([1., -3., 4., .1]), tensor([.001, 1., 3., 1e5]), tensor([1e5, 0., .2, 4.]), tensor(.7)
    delta = diagonal_step(x, g, metric, center, ewc, ridge,
                          recipe=recipe, previous_center=previous)
    if recipe == "lagged_1":
        residual = g + metric * delta + ewc * (x - previous) + ridge * (x + delta - center)
    else:
        residual = g + metric * delta + (ewc + ridge) * (x + delta - center)
    torch.testing.assert_close(residual, torch.zeros_like(g), atol=3e-10, rtol=0)


def test_first_step_ewc_and_stm_ridge_act_even_with_zero_initial_penalty():
    c, g, metric, ewc, ridge = tensor([0., 0.]), tensor([2., -3.]), tensor([1., 1.]), tensor([9., 0.]), tensor(4.)
    delta = diagonal_step(c, g, metric, c, ewc, ridge, recipe="proximal_1")
    torch.testing.assert_close(delta, tensor([-2 / 14, 3 / 5]))
    lag = diagonal_step(c, g, metric, c, ewc, ridge, recipe="lagged_1", previous_center=c)
    torch.testing.assert_close(lag, tensor([-2 / 5, 3 / 5]))
    assert delta[1] == lag[1]  # Identical implicit STM ridge at startup.


def test_historical_explicit_recentered_step_is_blind_to_ewc_and_ridge():
    def old_step(strength):
        x = tensor(2.).requires_grad_()
        center = x.detach().clone()
        loss = .5 * (x - 5).square() + .5 * strength * (x - center).square()
        return x.detach() - .1 * torch.autograd.grad(loss, x)[0]
    assert old_step(0) == old_step(1e8)


def test_frozen_center_inner_quadratic_converges_to_known_solution():
    center, target = tensor([1., -2., 4.]), tensor([-3., 5., 2.])
    hessian, ewc, ridge = tensor([.2, 3., 10.]), tensor([0., 5., 200.]), tensor(.7)
    metric = tensor([1., 8., 20.])
    expected = (hessian * target + (ewc + ridge) * center) / (hessian + ewc + ridge)
    x = center.clone()
    for _ in range(160):
        g = hessian * (x - target)
        x += diagonal_step(x, g, metric, center, ewc, ridge, recipe="proximal_3")
    torch.testing.assert_close(x, expected, atol=1e-12, rtol=1e-12)


def test_recurrent_centered_quadratic_uses_one_publication_per_event():
    pi, center, target, hessian = .2, tensor(1.), tensor(6.), tensor(3.)
    for k in range(1, 6):
        g = hessian * (center - target)
        center += diagonal_step(center, g, hessian, center, (1-pi)/pi*hessian,
                                tensor(0.), recipe="proximal_1")
        torch.testing.assert_close(center, target + (1-pi)**k * (1-target))


@pytest.mark.parametrize("ewc,stable", [(.8, True), (1.5, False)])
def test_lagged_recurrence_equilibrium_and_oscillatory_boundary(ewc, stable):
    g, metric, ridge = tensor(2.), tensor(1.), tensor(.3)
    equilibrium = -g / (metric + ridge + ewc)
    c, previous = tensor(0.), tensor(0.)
    errors = []
    for _ in range(80):
        u = diagonal_step(c, g, metric, c, tensor(ewc), ridge,
                          recipe="lagged_1", previous_center=previous)
        errors.append(float(u - equilibrium))
        previous, c = c.clone(), c + u
    assert errors[0] * errors[1] < 0
    assert errors[1] / errors[0] == pytest.approx(-ewc / 1.3)
    if stable:
        assert abs(errors[-1]) < 1e-12
    else:
        assert abs(errors[-1]) > 1e3


def test_live_gradient_second_moment_bias_correction_and_current_numerator():
    g, v, alpha = tensor([2., -4., 0.]), tensor([0., 0., 0.]), .003
    m, v = adaptive_metric(g, v, 1, alpha)
    torch.testing.assert_close(m, (g.abs() + EPSILON) / alpha)
    next_g = tensor([-1., 1., 0.])
    second, updated = adaptive_metric(next_g, v, 2, alpha)
    expected_v = (1-BETA2) * (BETA2*g.square() + next_g.square())
    torch.testing.assert_close(updated, expected_v)
    torch.testing.assert_close(second, (torch.sqrt(expected_v / (1-BETA2**2)) + EPSILON) / alpha)
    delta = diagonal_step(g*0, next_g, second, g*0, g*0, tensor(0.), recipe="proximal_1")
    assert delta[0] > 0 and delta[1] < 0 and delta[2] == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_metric_accumulates_at_least_float32_and_realized_rounding_is_explicit(dtype):
    x = torch.tensor([1., 2.], dtype=dtype)
    m, _ = adaptive_metric(torch.ones_like(x), torch.zeros_like(x), 1, 1e-10)
    assert m.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    delta = diagonal_step(x, torch.ones_like(x), m, x, torch.zeros_like(x), m.new_tensor(0.), recipe="proximal_1")
    realized = (x.to(delta) + delta).to(dtype).double() - x.double()
    assert bool((delta < 0).all())
    if dtype != torch.float64:
        assert torch.equal(realized, torch.zeros_like(realized))


@pytest.mark.parametrize("evaluation,alpha", [(0, .1), (True, .1), (1, 0.), (1, math.inf)])
def test_invalid_metric_settings_fail(evaluation, alpha):
    with pytest.raises(ValueError):
        adaptive_metric(tensor(1.), tensor(0.), evaluation, alpha)


def test_nonfinite_gradient_and_negative_curvature_rejected():
    with pytest.raises(FloatingPointError):
        adaptive_metric(tensor(float("nan")), tensor(0.), 1, .1)
    with pytest.raises(ValueError):
        diagonal_step(tensor(0.), tensor(1.), tensor(1.), tensor(0.), tensor(-1.), tensor(0.), recipe="proximal_1")
