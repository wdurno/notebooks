import json
from pathlib import Path

import pytest
import torch

from mnist_experiment.rotated_mnist.config import RotatedConfigError
from mnist_experiment.rotated_mnist.phase7_audit import (
    anchor_decomposition,
    compose_observation_weights,
    risk_recommendations,
    stationary_weight_concentration,
    weight_concentration,
    weight_concentration_update,
)
from mnist_experiment.rotated_mnist.phase7_config import (
    Phase7AuditConfig,
    load_phase7_audit_config,
)
from src.representations import DenseFisher


REPO_ROOT = Path(__file__).parents[3]
CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase7_anchor_audit.json"
)


def test_phase7_config_round_trips_frozen_artifact_contract() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = load_phase7_audit_config(CONFIG)

    assert config.to_mapping() == raw
    assert config == Phase7AuditConfig.from_mapping(raw)
    assert config.conditions == (
        "edr_slowtrend_slowaction",
        "fixed_pi0025",
        "fixed_pi005",
    )
    assert config.ranks == (16, 8)


def test_phase7_config_rejects_a_candidate_pi_grid() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["candidate_pis"] = [0.01, 0.05]

    with pytest.raises(RotatedConfigError, match="unknown fields"):
        Phase7AuditConfig.from_mapping(raw)


def test_observation_weights_produce_q_recursion_and_kish_size() -> None:
    weights = torch.full((10,), 0.1, dtype=torch.float64)
    pi = 0.2
    updated = compose_observation_weights(weights, pi, batch_size=4)
    direct = weight_concentration(updated)
    recursive = weight_concentration_update(
        weight_concentration(weights), pi, batch_size=4
    )

    assert direct == pytest.approx(recursive)
    assert 1.0 / weight_concentration(weights) == pytest.approx(10.0)
    assert float(updated.sum()) == pytest.approx(1.0)


def test_stationary_q_implies_half_action_covariance_recommendation() -> None:
    pi = 0.28
    q = stationary_weight_concentration(pi, batch_size=4)
    recommendation = risk_recommendations(
        q=q,
        old_covariance_shape=3.0,
        new_covariance_shape=3.0,
        deployed_batch_size=4,
        population_signal=0.0,
        conditional_signal=0.0,
    )

    assert weight_concentration_update(q, pi, 4) == pytest.approx(q)
    assert recommendation.covariance_only == pytest.approx(pi / 2.0)


def test_anchor_decomposition_closes_with_cross_term() -> None:
    fisher = DenseFisher(torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64)))
    displacement = torch.tensor([1.0, -2.0], dtype=torch.float64)
    error = torch.tensor([0.5, 1.0], dtype=torch.float64)
    result = anchor_decomposition(displacement, error, fisher)

    expected = float((displacement - error) @ fisher.matrix @ (displacement - error))
    assert result.conditional_signal == pytest.approx(expected)
    assert result.conditional_signal == pytest.approx(
        result.population_signal + result.anchor_error_energy + result.cross_term
    )
    assert result.identity_error == pytest.approx(0.0, abs=1e-12)


def test_centered_marginal_risk_is_mean_conditional_risk() -> None:
    generator = torch.Generator().manual_seed(71)
    count = 100_000
    displacement = torch.tensor([0.4, -0.2], dtype=torch.float64)
    q = 0.08
    errors = q**0.5 * torch.randn(
        count, 2, generator=generator, dtype=torch.float64
    )
    conditional = (displacement - errors).square().sum(dim=1).mean()
    marginal = float(displacement.square().sum()) + q * 2.0

    assert float(conditional) == pytest.approx(marginal, abs=0.003)


def test_nonzero_anchor_mean_changes_marginal_signal() -> None:
    displacement = torch.tensor([0.4, -0.2], dtype=torch.float64)
    mean_error = torch.tensor([0.1, 0.3], dtype=torch.float64)
    q = 0.05
    expected = float((displacement - mean_error).square().sum()) + 2.0 * q
    centered_formula = float(displacement.square().sum()) + 2.0 * q

    assert expected != pytest.approx(centered_formula)
    assert expected == pytest.approx(0.44)


def test_common_fisher_rescaling_leaves_recommendations_unchanged() -> None:
    base = risk_recommendations(
        q=0.04,
        old_covariance_shape=5.0,
        new_covariance_shape=6.0,
        deployed_batch_size=4,
        population_signal=0.3,
        conditional_signal=0.8,
    )
    scaled = risk_recommendations(
        q=0.04,
        old_covariance_shape=50.0,
        new_covariance_shape=60.0,
        deployed_batch_size=4,
        population_signal=3.0,
        conditional_signal=8.0,
    )

    assert scaled.covariance_only == pytest.approx(base.covariance_only)
    assert scaled.centered_marginal == pytest.approx(base.centered_marginal)
    assert scaled.realized_conditional == pytest.approx(base.realized_conditional)
