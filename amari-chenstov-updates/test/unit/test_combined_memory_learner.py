"""Focused evidence-admission and physical-objective checks for Plan 18."""

import dataclasses

import pytest
import torch

from mnist_experiment.combined_memory.config import Config, Recipe
from mnist_experiment.combined_memory.learner import archive_update, prepare_objective, fit_step
from mnist_experiment.combined_memory.model import build_model, memory_indices


def test_transport_precedes_eviction_and_does_not_add_weights():
    d = torch.tensor([1., 2.], dtype=torch.float64)
    delta = torch.tensor([-.4, .6], dtype=torch.float64)
    observed, q, diagnostic = archive_update(d, delta, None, pi=.1, evicted_count=0,
        nominal_arrival=4, q_archive=.2, transport=True)
    torch.testing.assert_close(observed, d+delta)
    assert q == .2 and diagnostic["eta"] == 0
    fresh = torch.tensor([3., 4.], dtype=torch.float64)
    observed, q, diagnostic = archive_update(d, delta, fresh, pi=.1, evicted_count=4,
        nominal_arrival=4, q_archive=.2, transport=True)
    torch.testing.assert_close(observed, .9*(d+delta)+.1*fresh)
    assert q == pytest.approx(.9**2*.2+.1**2/4)
    assert diagnostic["eta"] == pytest.approx(.1)


def test_raw_projection_diagnostics_and_no_early_fresh_evidence():
    d = torch.tensor([1., 2.], dtype=torch.float64)
    delta = torch.tensor([-3., .6], dtype=torch.float64)
    observed, q, diagnostic = archive_update(d, delta, None, pi=.1, evicted_count=0,
        nominal_arrival=4, q_archive=.2, transport=True)
    assert observed[0] == 0 and diagnostic["negative_mass"] == pytest.approx(2/3)
    assert diagnostic["negative_fraction"] == .5 and q == .2
    with pytest.raises(ValueError, match="without eviction"):
        archive_update(d, delta, d, pi=.1, evicted_count=0,
                       nominal_arrival=4, q_archive=.2, transport=True)


def test_objective_routes_both_blocks_and_freezes_retention_inputs():
    torch.set_num_threads(2)
    model, layout = build_model(18, dtype=torch.float64)
    theta = layout.flatten_module(model, detach=True)
    ltm, stm = memory_indices(layout)
    generator = torch.Generator().manual_seed(19)
    x = torch.rand(2, 2, 120, 120, generator=generator, dtype=torch.float64)
    y = torch.rand(2, 54, generator=generator, dtype=torch.float64)
    probes = torch.ones(2, len(stm), dtype=torch.float64)
    diagonal = torch.ones(len(ltm), dtype=torch.float64)
    anchor = theta[ltm].clone()
    cfg = Config.smoke_config()
    objective, _ = prepare_objective(model, layout, theta, diagonal, anchor,
        x, y, x, y, torch.ones(2), probes, pi=.025,
        recipe=Recipe("B", strength=.1), cfg=cfg)
    candidate = theta.clone().requires_grad_(True)
    value, parts = objective(candidate)
    gradient = torch.autograd.grad(value, candidate)[0]
    assert gradient[ltm].abs().sum() > 0 and gradient[stm].abs().sum() > 0
    assert parts["auxiliary"] > 0
    diagonal.zero_()
    anchor.add_(2)
    unchanged, _ = objective(candidate)
    torch.testing.assert_close(value, unchanged)
    accepted, solve = fit_step(model, layout, theta, torch.ones(len(ltm)), theta[ltm],
        x, y, x[:0], y[:0], torch.empty(0), probes, pi=.025,
        recipe=Recipe("A"), cfg=dataclasses.replace(cfg, max_iter=2, max_eval=4),
        initial_fisher_scale=1.)
    assert solve["final_objective"] <= solve["initial_objective"]
    assert torch.isfinite(accepted).all()
