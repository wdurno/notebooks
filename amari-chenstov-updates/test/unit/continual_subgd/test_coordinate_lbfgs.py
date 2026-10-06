from __future__ import annotations

import copy

import torch
from torch import nn

from mnist_experiment.continual_subgd.coordinate_lbfgs import (
    coordinate_lbfgs_update,
    coordinate_objective_and_gradient,
    geometry_factor,
    identity_factor,
    selection_factor,
)
from mnist_experiment.continual_subgd.geometry import AdaptationGeometry
from src.config import OptimizerConfig
from src.ewc import build_optimizer, take_ewc_proposal
from src.parameters import ParameterLayout
from src.representations import DenseFisher


def _problem() -> tuple[nn.Module, ParameterLayout, torch.Tensor, torch.Tensor, DenseFisher]:
    torch.manual_seed(7)
    model = nn.Linear(3, 2, bias=True, dtype=torch.float64)
    layout = ParameterLayout.from_module(model)
    inputs = torch.tensor(
        [[1.0, 0.0, -1.0], [0.25, 1.0, 0.5], [-0.5, 0.75, 1.0]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 1, 1])
    diagonal = torch.linspace(0.2, 1.1, layout.total_numel, dtype=torch.float64)
    return model, layout, inputs, targets, DenseFisher(torch.diag(diagonal))


def test_geometry_factor_square_matches_declared_preconditioner() -> None:
    torch.manual_seed(4)
    basis, _ = torch.linalg.qr(torch.randn(7, 3, dtype=torch.float64), mode="reduced")
    geometry = AdaptationGeometry(
        basis,
        torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64),
    )
    factor = geometry_factor(
        geometry,
        alpha=0.65,
        epsilon=0.1,
        projector_only=False,
    )
    dense = factor.dense()
    expected_columns = []
    identity = torch.eye(7, dtype=torch.float64)
    for column in range(7):
        expected_columns.append(
            geometry.precondition(
                identity[:, column],
                alpha=0.65,
                epsilon=0.1,
                projector_only=False,
            )
        )
    expected = torch.stack(expected_columns, dim=1)
    assert torch.allclose(dense @ dense.mT, expected, atol=1e-11, rtol=1e-11)


def test_singular_factors_confine_displacements_exactly() -> None:
    model, layout, inputs, targets, fisher = _problem()
    mask = torch.zeros(layout.total_numel, dtype=torch.float64)
    mask[[0, 3]] = 1
    factor = selection_factor(mask)
    before = layout.flatten_module(model, detach=True)
    result = coordinate_lbfgs_update(
        model,
        layout,
        inputs,
        targets,
        fisher,
        factor,
        strength=2.0,
        inner_steps=10,
        max_eval=15,
    )
    after = layout.flatten_module(model, detach=True)
    assert torch.equal(after - before, result.displacement)
    assert torch.equal(result.displacement[mask == 0], torch.zeros_like(result.displacement[mask == 0]))


def test_coordinate_gradient_matches_finite_difference() -> None:
    model, layout, inputs, targets, fisher = _problem()
    mask = torch.zeros(layout.total_numel, dtype=torch.float64)
    mask[[0, 2, 5]] = 1
    factor = selection_factor(mask)
    coordinates = torch.tensor([0.03, -0.02, 0.01], dtype=torch.float64)
    _, gradient = coordinate_objective_and_gradient(
        model,
        layout,
        coordinates,
        factor,
        inputs,
        targets,
        fisher,
        strength=1.7,
    )
    epsilon = 1e-6
    numerical = []
    for index in range(coordinates.numel()):
        direction = torch.zeros_like(coordinates)
        direction[index] = epsilon
        plus, _ = coordinate_objective_and_gradient(
            model,
            layout,
            coordinates + direction,
            factor,
            inputs,
            targets,
            fisher,
            strength=1.7,
        )
        minus, _ = coordinate_objective_and_gradient(
            model,
            layout,
            coordinates - direction,
            factor,
            inputs,
            targets,
            fisher,
            strength=1.7,
        )
        numerical.append((plus - minus) / (2 * epsilon))
    assert torch.allclose(gradient, torch.stack(numerical), atol=1e-7, rtol=1e-6)


def test_identity_coordinate_solver_matches_direct_parameter_lbfgs() -> None:
    model, layout, inputs, targets, fisher = _problem()
    direct = copy.deepcopy(model)
    direct_layout = ParameterLayout.from_module(direct)
    config = OptimizerConfig(
        name="lbfgs",
        learning_rate=1.0,
        inner_steps=20,
        ewc_strength=1.0,
        lbfgs_history_size=10,
        lbfgs_max_eval_factor=1.5,
        lbfgs_tolerance_grad=1e-7,
        lbfgs_tolerance_change=1e-11,
        lbfgs_line_search_fn="strong_wolfe",
    )
    direct_result = take_ewc_proposal(
        direct,
        direct_layout,
        inputs,
        targets,
        fisher,
        config,
        build_optimizer(direct, config),
        adaptation_weight=0.2,
    )
    coordinate_result = coordinate_lbfgs_update(
        model,
        layout,
        inputs,
        targets,
        fisher,
        identity_factor(
            layout.total_numel,
            device=inputs.device,
            dtype=inputs.dtype,
        ),
        strength=4.0,
        inner_steps=20,
        max_eval=30,
        history_size=10,
        tolerance_grad=1e-7,
        tolerance_change=1e-11,
    )
    assert torch.allclose(
        layout.flatten_module(model, detach=True),
        direct_layout.flatten_module(direct, detach=True),
        atol=2e-7,
        rtol=2e-7,
    )
    assert abs(coordinate_result.objective_after - direct_result.objective_after) < 1e-9
