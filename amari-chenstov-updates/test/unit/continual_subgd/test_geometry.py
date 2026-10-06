from __future__ import annotations

import torch

from mnist_experiment.continual_subgd.controller import (
    InnovationControllerConfig,
    InnovationControllerState,
)
from mnist_experiment.continual_subgd.geometry import (
    AdaptationGeometry,
    half_life_gain,
    projector_distance,
    random_geometry,
)


def test_full_rank_streaming_update_matches_dense_ema() -> None:
    observations = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 2.0, -0.5], [1.0, 1.0, 0.0], [-1.0, 0.5, 1.0]],
        dtype=torch.float64,
    )
    geometry = AdaptationGeometry.from_observations(observations, rank=3)
    update = torch.tensor([0.5, -1.5, 2.0], dtype=torch.float64)
    beta = 0.2
    expected = (1 - beta) * geometry.dense() + beta * torch.outer(update, update)
    actual = geometry.update(update, beta)
    assert torch.allclose(actual.dense(), expected, atol=1e-11, rtol=1e-11)


def test_rank_truncated_update_matches_dense_leading_eigensystem() -> None:
    basis = torch.eye(4, dtype=torch.float64)[:, :2]
    geometry = AdaptationGeometry(basis, torch.tensor([3.0, 1.0], dtype=torch.float64))
    observation = torch.tensor([0.5, 0.25, 2.0, -1.0], dtype=torch.float64)
    beta = 0.3
    dense = (1 - beta) * geometry.dense() + beta * torch.outer(observation, observation)
    values, vectors = torch.linalg.eigh(dense)
    order = torch.argsort(values, descending=True)[:2]
    expected = (vectors[:, order] * values[order]) @ vectors[:, order].mT
    actual = geometry.update(observation, beta)
    assert torch.allclose(actual.dense(), expected, atol=1e-11, rtol=1e-11)


def test_zero_residual_update_and_preconditioner_gains() -> None:
    basis = torch.eye(3, dtype=torch.float64)[:, :2]
    geometry = AdaptationGeometry(basis, torch.tensor([3.0, 1.0], dtype=torch.float64))
    updated = geometry.update(torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64), 0.25)
    assert torch.allclose(updated.basis.mT @ updated.basis, torch.eye(2, dtype=torch.float64))

    vector = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    result = geometry.precondition(vector, alpha=0.75, epsilon=0.1)
    normalized = geometry.normalized_eigenvalues()
    expected = torch.tensor(
        [
            (0.25 + 0.75 * normalized[0]) * 1.0,
            (0.25 + 0.75 * normalized[1]) * 2.0,
            (0.25 + 0.75 * 0.1) * 4.0,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result, expected)
    assert abs(geometry.innovation(torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)) - 1.0) < 1e-12


def test_random_geometry_and_projector_distance_are_deterministic() -> None:
    eigenvalues = torch.tensor([2.0, 1.0], dtype=torch.float64)
    first = random_geometry(5, eigenvalues, seed=42)
    second = random_geometry(5, eigenvalues, seed=42)
    other = random_geometry(5, eigenvalues, seed=43)
    assert torch.equal(first.basis, second.basis)
    assert projector_distance(first.basis, second.basis) == 0.0
    assert projector_distance(first.basis, other.basis) > 0.0


def test_innovation_controller_relaxes_trust_and_speeds_geometry() -> None:
    config = InnovationControllerConfig(
        innovation_half_life=2.0,
        alpha_scale=4.0,
        beta_min_half_life=16.0,
        beta_max_half_life=2.0,
        beta_scale=4.0,
    )
    state = InnovationControllerState()
    low, state = state.decide(0.0, config)
    high, state = state.decide(1.0, config)
    assert low.alpha == 1.0
    assert high.alpha < low.alpha
    assert high.beta > low.beta
    assert 0 < half_life_gain(16.0) < half_life_gain(2.0) < 1
