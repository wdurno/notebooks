import torch
import pytest
from torch import nn

from src.config import OptimizerConfig
from src.ewc import (
    build_optimizer,
    ewc_penalty,
    mixture_ewc_strength,
    take_ewc_proposal,
)
from src.parameters import ParameterLayout
from src.representations import LowRankDiagonalFisher


def test_ewc_penalty_matches_quadratic_form() -> None:
    theta = torch.tensor([2.0, -1.0], dtype=torch.float64)
    anchor = torch.tensor([1.0, 1.0], dtype=torch.float64)
    fisher = torch.tensor([[2.0, 0.5], [0.5, 3.0]], dtype=torch.float64)

    penalty = ewc_penalty(theta, anchor, fisher, strength=4.0)

    displacement = theta - anchor
    expected = 2.0 * displacement @ fisher @ displacement
    torch.testing.assert_close(penalty, expected)


def test_structured_ewc_penalty_matches_explicit_dense_form() -> None:
    theta = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    anchor = torch.tensor([1.0, 1.0, -0.5], dtype=torch.float64)
    factor = torch.tensor(
        [[1.0, 0.0], [0.5, 0.25], [-0.5, 1.0]],
        dtype=torch.float64,
    )
    residual = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    structured = LowRankDiagonalFisher(factor, residual)

    penalty = ewc_penalty(theta, anchor, structured, strength=4.0)
    expected = ewc_penalty(
        theta,
        anchor,
        structured.to_dense(),
        strength=4.0,
    )

    torch.testing.assert_close(penalty, expected)


def test_mixture_ewc_strength_is_old_to_new_evidence_odds() -> None:
    assert mixture_ewc_strength(0.5) == pytest.approx(1.0)
    assert mixture_ewc_strength(0.8, multiplier=2.0) == pytest.approx(0.5)
    assert mixture_ewc_strength(1.0) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        mixture_ewc_strength(0.0)


def test_ewc_proposal_records_the_realized_full_network_move() -> None:
    model = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2], [-0.1]], dtype=torch.float64))
    layout = ParameterLayout.from_module(model)
    fisher = torch.eye(layout.total_numel, dtype=torch.float64)
    inputs = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    targets = torch.tensor([1, 0])
    config = OptimizerConfig(
        name="sgd",
        learning_rate=0.1,
        inner_steps=2,
        ewc_strength=3.0,
    )
    optimizer = build_optimizer(model, config)
    before = layout.flatten_module(model, detach=True)

    result = take_ewc_proposal(
        model,
        layout,
        inputs,
        targets,
        fisher,
        config,
        optimizer,
    )

    after = layout.flatten_module(model, detach=True)
    torch.testing.assert_close(result.displacement, after - before)
    assert result.inner_steps == 2
    assert result.optimizer_iterations == 2
    assert result.optimizer_function_evaluations >= 4
    assert result.displacement_norm > 0
    assert result.fisher_weighted_displacement_norm == result.displacement_norm
    assert result.ewc_penalty_after > 0
    assert result.initial_gradient_norm > 0
    assert result.final_gradient_norm >= 0
    assert result.final_gradient_max_abs >= 0
    assert result.relative_final_gradient_norm >= 0
    assert result.final_gradient_rms >= 0
    assert result.objective_decrease >= 0
    assert result.stopping_reason == "fixed_inner_step_budget"


def test_mixture_ewc_proposal_records_odds_without_post_scaling() -> None:
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    layout = ParameterLayout.from_module(model)
    inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25]], dtype=torch.float64)
    targets = torch.tensor([0, 1])
    fisher = torch.eye(layout.total_numel, dtype=torch.float64)
    config = OptimizerConfig(
        name="sgd",
        learning_rate=0.1,
        inner_steps=2,
        ewc_strength=1.5,
    )
    before = layout.flatten_module(model, detach=True)

    result = take_ewc_proposal(
        model,
        layout,
        inputs,
        targets,
        fisher,
        config,
        build_optimizer(model, config),
        adaptation_weight=0.75,
    )

    assert result.adaptation_weight == pytest.approx(0.75)
    assert result.effective_ewc_strength == pytest.approx(0.5)
    assert (
        result.objective_normalization
        == "mean_new_loss_plus_old_to_new_odds"
    )
    torch.testing.assert_close(
        result.displacement,
        layout.flatten_module(model, detach=True) - before,
    )


def test_ewc_proposal_backtracks_through_high_quadratic_curvature() -> None:
    model = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    layout = ParameterLayout.from_module(model)
    inputs = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    targets = torch.tensor([1, 0])
    fisher = 100.0 * torch.eye(layout.total_numel, dtype=torch.float64)
    config = OptimizerConfig(
        name="sgd",
        learning_rate=0.1,
        inner_steps=3,
        ewc_strength=1.0,
    )
    optimizer = build_optimizer(model, config)

    result = take_ewc_proposal(
        model,
        layout,
        inputs,
        targets,
        fisher,
        config,
        optimizer,
        adaptation_weight=0.05,
    )

    assert torch.isfinite(layout.flatten_module(model)).all()
    assert result.backtracking_rejections > 0
    assert result.maximum_backtracks > 0
    assert result.minimum_learning_rate < config.learning_rate
    assert result.optimization_guard == "monotone_objective_backtracking"
    assert optimizer.param_groups[0]["lr"] == config.learning_rate


def test_lbfgs_ewc_proposal_converges_with_strong_wolfe_line_search() -> None:
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float64)
        )
    layout = ParameterLayout.from_module(model)
    inputs = torch.tensor(
        [[1.0, -1.0], [0.5, 0.25], [-0.75, 0.5]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 1, 0])
    fisher = torch.eye(layout.total_numel, dtype=torch.float64)
    config = OptimizerConfig(
        name="lbfgs",
        learning_rate=1.0,
        inner_steps=100,
        ewc_strength=1.0,
        lbfgs_history_size=10,
        lbfgs_max_eval_factor=1.5,
        lbfgs_tolerance_grad=1e-9,
        lbfgs_tolerance_change=1e-12,
        lbfgs_line_search_fn="strong_wolfe",
    )

    result = take_ewc_proposal(
        model,
        layout,
        inputs,
        targets,
        fisher,
        config,
        build_optimizer(model, config),
        adaptation_weight=0.5,
    )

    assert result.objective_decrease > 0.0
    assert result.relative_final_gradient_norm < 1e-6
    assert result.optimization_guard == "strong_wolfe_line_search_transaction"
    assert result.stopping_reason == "lbfgs_convergence_tolerance"
    assert 0 < result.optimizer_iterations < config.inner_steps
    assert result.optimizer_function_evaluations >= result.optimizer_iterations
    assert result.metrics_mapping()["optimizer_iterations"] == (
        result.optimizer_iterations
    )
