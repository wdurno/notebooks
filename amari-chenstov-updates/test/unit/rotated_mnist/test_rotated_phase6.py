import json
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.config import RotatedConfigError
from mnist_experiment.rotated_mnist.phase6_config import (
    Phase6DebiasConfig,
    Phase6OracleConfig,
    load_phase6_debias_config,
    load_phase6_oracle_config,
)
from mnist_experiment.rotated_mnist.phase6_debias import (
    reconstruct_debiased_recommendations,
)
from mnist_experiment.rotated_mnist.phase6_analysis import discounted_risk_actions
from mnist_experiment.rotated_mnist.phase6_oracle import (
    approximate_score_fisher,
    covariance_risk,
    estimate_oracle,
    full_reference_angle_union,
    paired_displacement_covariance_risk,
)
from src.representations import DenseFisher

import torch


REPO_ROOT = Path(__file__).parents[3]
CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase6_debias.json"
)
ORACLE_SMOKE_CONFIG = CONFIG.with_name("phase6_oracle_smoke.json")


def _rows(gains=(0.5, 0.5)):
    rows = []
    for step, gain in enumerate(gains):
        rows.append(
            {
                "step": step,
                "angle_degrees": float(step),
                "cumulative_angular_degrees": float(step),
                "leg_id": 0,
                "controller": {
                    "signal_energy": 2.0,
                    "uncertainty_scale_estimate": 4.0,
                    "effective_size": 10.0,
                    "applied_pi": 0.05,
                    "cold_start_active": False,
                },
                "controller_acceptance": {"trend_gain": gain},
            }
        )
    rows.append(
        {
            "step": len(gains),
            "angle_degrees": float(len(gains)),
            "cumulative_angular_degrees": float(len(gains)),
            "leg_id": 0,
            "controller": None,
        }
    )
    return rows


def test_phase6_debias_config_round_trips_and_separates_sample_sizes() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = load_phase6_debias_config(CONFIG)

    assert config.to_mapping() == raw
    assert config == Phase6DebiasConfig.from_mapping(raw)
    assert config.deployed_batch_size == 4
    assert config.variance_scales == (0.5, 1.0, 2.0)


def test_phase6_debias_config_rejects_non_authoritative_source() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["source_run_id"] = "other"
    raw["source_run_path"] = "cache/other"

    with pytest.raises(RotatedConfigError, match="authoritative"):
        Phase6DebiasConfig.from_mapping(raw)


def test_phase6_oracle_config_round_trips_and_keeps_m_distinct() -> None:
    raw = json.loads(ORACLE_SMOKE_CONFIG.read_text(encoding="utf-8"))
    config = load_phase6_oracle_config(ORACLE_SMOKE_CONFIG)

    assert config.to_mapping() == raw
    assert config == Phase6OracleConfig.from_mapping(raw)
    assert config.deployed_batch_size == 4
    assert config.reference.fit_sample_size == 128
    assert config.local_mle.sample_sizes == (32,)


def test_phase6_oracle_config_rejects_candidate_transition_outcome_drift() -> None:
    raw = json.loads(ORACLE_SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["deployed_batch_size"] = 128

    with pytest.raises(RotatedConfigError, match="deployed m"):
        Phase6OracleConfig.from_mapping(raw)


def test_trend_variance_recursion_debiases_quadratic_signal() -> None:
    result = reconstruct_debiased_recommendations(
        _rows(),
        batch_size=4,
        variance_scale=1.0,
        pi_min=0.01,
        pi_max=0.95,
    )

    assert result[0]["trend_variance_energy_before"] == 0.0
    expected_variance = 0.5**2 * (0.1 + 0.25) * 4.0
    assert result[1]["trend_variance_energy_before"] == pytest.approx(
        expected_variance
    )
    assert result[1]["debiased_signal_energy"] == pytest.approx(
        2.0 - expected_variance
    )
    assert result[1]["debiased_unclipped_pi"] < (
        (2.0 + 0.4) / (2.0 + 0.4 + 1.0)
    )


def test_debias_clamps_negative_signal_without_clamping_unclipped_pi() -> None:
    rows = _rows(gains=(1.0, 1.0))
    rows[1]["controller"]["signal_energy"] = 0.01
    result = reconstruct_debiased_recommendations(
        rows,
        batch_size=4,
        variance_scale=2.0,
        pi_min=0.2,
        pi_max=0.9,
    )

    assert result[1]["debiased_signal_energy"] == 0.0
    assert result[1]["debiased_unclipped_pi"] == pytest.approx(0.1 / 0.35)
    assert result[1]["debiased_clipped_pi"] == pytest.approx(0.1 / 0.35)


def test_variable_gain_recursion_uses_each_persisted_gain() -> None:
    result = reconstruct_debiased_recommendations(
        _rows(gains=(0.25, 0.75)),
        batch_size=4,
        variance_scale=1.0,
        pi_min=0.01,
        pi_max=0.95,
    )

    first = 0.25**2 * (0.1 + 0.25) * 4.0
    second = 0.75**2 * (0.1 + 0.25) * 4.0 + 0.25**2 * first
    assert result[1]["trend_variance_energy_before"] == pytest.approx(first)
    # The final propagated state is not emitted because no decision follows it.
    assert second > first


def test_reference_angle_union_canonicalizes_reverse_roundoff() -> None:
    values = full_reference_angle_union(
        {
            "left": (0.0, 0.12121212259505855, 30.0),
            "right": (30.0, 0.12121212259506109, 0.0),
        }
    )

    assert values == (0.0, 0.121212122595, 30.0)


def test_score_fisher_rank_plus_diagonal_preserves_exact_diagonal() -> None:
    scores = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 2.0], [-1.0, 0.0]],
        dtype=torch.float64,
    )
    result = approximate_score_fisher(scores, rank=2, seed=31)

    assert result.representation.to_dense() == pytest.approx(
        scores.mT @ scores / scores.shape[0], abs=1e-10
    )
    assert result.representation.diagonal_vector() == pytest.approx(
        scores.square().mean(dim=0), abs=1e-12
    )


def test_covariance_risk_matches_dense_sample_covariance_trace() -> None:
    parameters = torch.tensor(
        [[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]], dtype=torch.float64
    )
    matrix = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    risk, _ = covariance_risk(
        parameters, DenseFisher(matrix), local_sample_size=10
    )
    covariance = torch.cov(parameters.mT)

    assert risk == pytest.approx(float(10 * torch.trace(matrix @ covariance)))


def test_paired_displacement_risk_preserves_cross_covariance() -> None:
    parameters = torch.tensor(
        [[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]], dtype=torch.float64
    )
    fisher = DenseFisher(torch.eye(2, dtype=torch.float64))

    risk, energies = paired_displacement_covariance_risk(
        parameters,
        parameters + torch.tensor([2.0, -3.0], dtype=torch.float64),
        fisher,
        local_sample_size=10,
    )

    assert risk == pytest.approx(0.0, abs=1e-12)
    assert energies == pytest.approx(torch.zeros(3, dtype=torch.float64))


def test_oracle_recovers_analytic_fixed_batch_minimizer() -> None:
    fisher = DenseFisher(torch.eye(2, dtype=torch.float64))
    local = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    estimate = estimate_oracle(
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([2.0, 0.0], dtype=torch.float64),
        local,
        local,
        fisher,
        reference_sample_size=10_000,
        local_sample_size=2,
        q=0.1,
        deployed_batch_size=4,
    )
    shape, _ = covariance_risk(local, fisher, local_sample_size=2)
    signal = 4.0
    expected = (signal + 0.1 * shape) / (signal + 0.1 * shape + shape / 4)

    assert estimate.pi == pytest.approx(expected)


@pytest.mark.parametrize("local_sample_size", [32, 128])
def test_gaussian_location_oracle_has_stable_scaled_covariance(
    local_sample_size: int,
) -> None:
    generator = torch.Generator().manual_seed(73)
    replicate_count = 20_000
    standard_errors = torch.randn(
        replicate_count, 2, generator=generator, dtype=torch.float64
    )
    shifted_errors = torch.randn(
        replicate_count, 2, generator=generator, dtype=torch.float64
    )
    root_covariance = torch.diag(torch.tensor([1.0, 0.5], dtype=torch.float64))
    local_from = standard_errors @ root_covariance / local_sample_size**0.5
    local_to = (
        torch.tensor([0.4, -0.2], dtype=torch.float64)
        + shifted_errors @ root_covariance / local_sample_size**0.5
    )
    fisher = DenseFisher(torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64)))

    estimate = estimate_oracle(
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([0.4, -0.2], dtype=torch.float64),
        local_from,
        local_to,
        fisher,
        reference_sample_size=10**12,
        local_sample_size=local_sample_size,
        q=0.1,
        deployed_batch_size=4,
    )

    # I @ K is the identity for this regular Gaussian location model.
    assert estimate.old_covariance_shape_risk == pytest.approx(2.0, abs=0.04)
    assert estimate.new_covariance_shape_risk == pytest.approx(2.0, abs=0.04)
    expected_signal = 0.4**2 + 4.0 * 0.2**2
    expected_pi = (expected_signal + 0.2) / (
        expected_signal + 0.2 + 0.5
    )
    assert estimate.pi == pytest.approx(expected_pi, abs=0.015)


def test_discounted_oracle_actions_smooth_coefficients_before_ratio() -> None:
    actions = discounted_risk_actions(
        ((1.0, 3.0), (3.0, 1.0), (3.0, 1.0)),
        half_life_steps=1.0,
        cold_start_steps=1,
        cold_start_pi=0.05,
        pi_min=0.01,
        pi_max=0.95,
    )

    assert actions[0] == 0.05
    assert actions[1] == pytest.approx(7.0 / 12.0)
    assert actions[2] == pytest.approx(19.0 / 28.0)
