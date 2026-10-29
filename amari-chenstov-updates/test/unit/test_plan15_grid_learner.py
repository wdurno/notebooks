"""The grid retirement clock and complete objective use observation units."""

import math

import pytest
import torch

from mnist_experiment.fisher_memory.learner import FitSettings, StopRequested, retention_terms
from mnist_experiment.fisher_memory_grid_registration.learner import (
    consolidate, initial_state, stage_arrival, commit_arrival, prepare_objective,
)
from mnist_experiment.fisher_memory_grid_registration.model import build_model, memory_indices
from mnist_experiment.fisher_memory_grid_registration.curvature import gaussian_nll, fd_values


def test_retirement_weights_grids_once_and_keeps_expected_trace():
    source = torch.diag(torch.tensor([1., 3.], dtype=torch.float64))
    state = initial_state(torch.zeros(4, dtype=torch.float64), torch.tensor([0, 1]), source,
                          capacity=1, rank=1)
    staged = stage_arrival(state, [10, 11], [0, 1])
    assert [r["exposure_id"] for r in staged["evicted"]] == [10]
    updated, metadata = commit_arrival(state, state["theta"], [10, 11], [0, 1],
        source * 2, pi=.025, rank=1)
    eta = 1 - (1-.025)**(1/4)
    assert metadata["eta"] == pytest.approx(eta)
    assert updated["consolidated_ids"] == [10]
    assert updated["archive_diagnostics"]["represented_trace"] == pytest.approx(4*(1+eta))
    with pytest.raises(ValueError, match="already"):
        stage_arrival(updated, [10], [0])


def test_no_eviction_keeps_archive_and_immediate_retirement_uses_same_clock():
    matrix = torch.eye(2, dtype=torch.float64)
    state = initial_state(torch.zeros(3, dtype=torch.float64), torch.tensor([0, 1]), matrix,
                          capacity=0, rank=1)
    updated, meta = commit_arrival(state, state["theta"], [0, 1], [0, 1], matrix,
                                  pi=.025, rank=1)
    assert meta["eta"] == pytest.approx(1-(1-.025)**.5)
    assert updated["consolidated_ids"] == [0, 1] and not updated["replay"]
    with pytest.raises(ValueError, match="Fisher update"):
        consolidate(matrix, torch.zeros(2), torch.zeros(2), matrix,
                    evicted_ids=[], consolidated_ids=[], pi=.025, arrival_size=4, rank=1, seed=0)


def test_full_objective_sums_54_coordinates_and_differentiates_both_blocks():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(1181)
    model, layout = build_model(31, dtype=torch.float64)
    theta = layout.flatten_module(model, detach=True).requires_grad_()
    ltm, stm = memory_indices(layout)
    x = torch.rand((2, 2, 84, 84), generator=generator, dtype=torch.float64)
    y = torch.randn((2, 54), generator=generator, dtype=torch.float64)
    archive = torch.eye(len(ltm), dtype=torch.float64)
    settings = FitSettings(lambda_f=2., num_probes=2, fd_step=.01, chunk_size=1)
    objective, metadata = prepare_objective(model, layout, theta, x[:1], y[:1], x[1:], y[1:],
        torch.tensor([.25]), archive, theta[ltm].detach(), "fd_replay_ewc", settings, 111)
    value, parts = objective(theta)
    assert float(parts["ewc"].detach()) == 0 and float(parts["ridge"].detach()) == 0
    assert float(value.detach()) == pytest.approx(float((parts["data_loss"]+2*parts["auxiliary"]).detach()))
    assert metadata["output_dim"] == 54 and metadata["stm_dimension"] == 344
    gradient = torch.autograd.grad(value, theta)[0]
    assert gradient[ltm].norm() > 0 and gradient[stm].norm() > 0
    candidate = (theta.detach()+.001*torch.randn(theta.shape,generator=generator,dtype=theta.dtype)).requires_grad_()
    _, moved = objective(candidate)
    expected_ewc, expected_ridge, _ = retention_terms(candidate,ltm,stm,archive,theta[ltm].detach(),theta[stm].detach(),
        pi=settings.pi,ridge_ratio=settings.ridge_ratio)
    torch.testing.assert_close(moved["ewc"],expected_ewc,rtol=0,atol=0)
    torch.testing.assert_close(moved["ridge"],expected_ridge,rtol=0,atol=0)
    actual_gradient = torch.autograd.grad(moved["ewc"]+moved["ridge"],candidate)[0]
    reference_gradient = torch.autograd.grad(expected_ewc+expected_ridge,candidate)[0]
    torch.testing.assert_close(actual_gradient,reference_gradient,rtol=0,atol=0)
    with pytest.raises(ValueError, match="targets"):
        prepare_objective(model, layout, theta, x[:1, :, :28, :28], y[:1], x[1:], y[1:],
            torch.ones(1), archive, theta[ltm].detach(), "fd_replay_ewc", settings, 111)
