from __future__ import annotations

import torch

from mnist_experiment.rotated_mnist.plan12.spectral import (
    effective_sample_size_ema,
    empirical_tail_maxima,
    fit_local_power_law,
    power_law_quadrature,
    solve_deformed_mp_edge,
)


def test_spherical_mp_edge_matches_closed_form() -> None:
    gamma = 0.25
    result = solve_deformed_mp_edge(
        torch.ones(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        gamma=gamma,
        effective_sample_size=10_000,
    )
    assert abs(result.edge - (1 + gamma**0.5) ** 2) < 1e-11
    assert result.regular
    assert result.nearest_pole_distance > 0


def test_power_law_fit_and_discrete_quadrature_are_normalized() -> None:
    ranks = torch.arange(1, 17, dtype=torch.float64)
    eigenvalues = 3.5 * ranks.pow(-1.3)
    fit = fit_local_power_law(eigenvalues, start_rank=3, stop_rank=8)
    assert abs(fit.amplitude - 3.5) < 1e-11
    assert abs(fit.exponent - 1.3) < 1e-11
    values, weights = power_law_quadrature(fit, first_rank=9, parameter_count=487)
    assert values.shape == weights.shape
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert bool((values > 0).all())


def test_empirical_tail_resampling_is_seeded_and_removes_resolved_space() -> None:
    generator = torch.Generator().manual_seed(1)
    scores = torch.randn(30, 5, generator=generator, dtype=torch.float64)
    basis = torch.eye(5, dtype=torch.float64)[:, :2]
    first = empirical_tail_maxima(scores, basis, sample_size=12, resamples=7, seed=91)
    second = empirical_tail_maxima(scores, basis, sample_size=12, resamples=7, seed=91)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert bool((first >= 0).all())
    assert effective_sample_size_ema(batch_size=4, gain=0.025) == 316.0
