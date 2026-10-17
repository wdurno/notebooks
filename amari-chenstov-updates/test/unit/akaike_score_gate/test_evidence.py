import math

import pytest
import torch

from mnist_experiment.akaike_score_gate.evidence import (
    consolidate, effective_sample_size, evidence_diagnostics,
    gradient_noise_covariance, retention_precision, weights,
)


def test_finite_weights_with_unequal_batch_sizes_and_exact_ess():
    actual = weights([2, 1, 3], 1)
    raw = torch.tensor([.25, .25, .5, 1., 1., 1.], dtype=torch.float64)
    torch.testing.assert_close(actual, raw / raw.sum())
    assert effective_sample_size(actual) == pytest.approx(float(raw.sum().square() / raw.square().sum()))
    assert actual.dtype == torch.float64
    assert weights([3], 8).tolist() == [1 / 3] * 3


def test_finite_episode_ess_is_not_stationary_limit():
    beta = 2 ** (-1 / 8)
    finite = effective_sample_size(weights([4] * 32, 8))
    limiting = 4 * (1 + beta) / (1 - beta)
    assert finite < limiting
    assert finite == pytest.approx(limiting * (1 - beta**32) / (1 + beta**32))


def test_covariance_uses_squared_weights_for_common_and_heterogeneous_inputs():
    a = weights([1, 1], 1)
    sigma = torch.tensor([[2., .3], [.3, 1.]], dtype=torch.float64)
    torch.testing.assert_close(gradient_noise_covariance(sigma, a), a.square().sum() * sigma)
    sigmas = torch.stack([sigma, 3 * sigma])
    torch.testing.assert_close(gradient_noise_covariance(sigmas, a), (a[0]**2 + 3*a[1]**2) * sigma)
    assert not torch.allclose(gradient_noise_covariance(sigmas, a), (a[0] + 3*a[1]) * sigma)


def test_evidence_counts_and_ages_do_not_count_synthetic_labels():
    result = evidence_diagnostics([2, 1], 1, observation_ids=[4, 4, 5],
                                  labels=torch.tensor([9, 0, 9]),
                                  expected_probabilities=torch.tensor([.1, .1, .2]))
    assert result["buffer_size"] == 3
    assert result["unique_observations"] == 2
    assert result["realized_digit9_count"] == 2
    assert result["expected_digit9_count"] == pytest.approx(.4)
    assert result["mean_age_arrivals"] == .5
    assert result["arrival_weight_mass"] == [.5, .5]


def test_once_only_consolidation_is_pure_and_ridge_is_not_archived():
    archive = torch.eye(3, dtype=torch.float64)
    fresh = torch.diag(torch.tensor([2., 3., 4.], dtype=torch.float64))
    theta = torch.tensor([.1, .2, .3])
    old_precision, old_kappa = retention_precision(archive, 19, .1)
    result = consolidate(archive, theta, fresh, omega=19, arrivals=4,
                         episode_id="episode0", observation_ids=["a", "b", "c", "d"],
                         kind="dense", ridge_ratio=.1)
    eta = 1 - .95**4
    expected = (1 - eta) * archive + eta * fresh
    assert result["eta"] == pytest.approx(eta)
    torch.testing.assert_close(result["archive"], expected)
    assert result["kappa"] == pytest.approx(.1 * float(expected.trace() / 3))
    torch.testing.assert_close(result["precision"], 19 * (expected + result["kappa"] * archive))
    torch.testing.assert_close(archive, torch.eye(3, dtype=torch.float64))
    assert old_kappa == .1
    torch.testing.assert_close(old_precision, 20.9 * archive)
    theta.add_(1)
    assert not torch.equal(theta, result["anchor"])
    with pytest.raises(ValueError, match="episode already"):
        consolidate(expected, theta, fresh, omega=19, arrivals=4, episode_id="episode0",
                    observation_ids=["e"], ledger=result["ledger"])
    with pytest.raises(ValueError, match="exposure already"):
        consolidate(expected, theta, fresh, omega=19, arrivals=1, episode_id="episode1",
                    observation_ids=["a"], ledger=result["ledger"])
    next_result = consolidate(expected, theta, fresh, omega=0, arrivals=1, episode_id="episode1",
                              observation_ids=["e"], ledger=result["ledger"], kind="dense")
    assert "episode1" not in result["ledger"]
    torch.testing.assert_close(next_result["archive"], fresh)
    assert next_result["eta"] == 1


@pytest.mark.parametrize("sizes,half_life", [([], 1), ([0], 1), ([True], 1), ([2], 0), ([2], math.inf)])
def test_invalid_evidence_rejected(sizes, half_life):
    with pytest.raises(ValueError):
        weights(sizes, half_life)


def test_ess_requires_normalized_weights():
    with pytest.raises(ValueError, match="normalized"):
        effective_sample_size(torch.ones(2))
