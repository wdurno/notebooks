import torch
from torch import nn

from src.config import OptimizerConfig
from src.ewc import build_optimizer, ewc_penalty, take_ewc_proposal
from src.parameters import ParameterLayout


def test_ewc_penalty_matches_quadratic_form() -> None:
    theta = torch.tensor([2.0, -1.0], dtype=torch.float64)
    anchor = torch.tensor([1.0, 1.0], dtype=torch.float64)
    fisher = torch.tensor([[2.0, 0.5], [0.5, 3.0]], dtype=torch.float64)

    penalty = ewc_penalty(theta, anchor, fisher, strength=4.0)

    displacement = theta - anchor
    expected = 2.0 * displacement @ fisher @ displacement
    torch.testing.assert_close(penalty, expected)


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
    assert result.displacement_norm > 0
    assert result.fisher_weighted_displacement_norm == result.displacement_norm
    assert result.ewc_penalty_after > 0
