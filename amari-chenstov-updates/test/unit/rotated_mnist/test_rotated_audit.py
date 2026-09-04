from pathlib import Path

import pytest
import torch

from mnist_experiment.rotated_mnist.audit import (
    feasibility_gate,
    offline_risk_coefficients,
    reference_path_diagnostics,
    repeated_fit_noise,
)
from mnist_experiment.rotated_mnist.audit_config import GateConfig, load_audit_config


REPO_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase2_smoke.json"
)


def test_audit_config_freezes_reference_design() -> None:
    config = load_audit_config(SMOKE_CONFIG)

    assert config.reference.angles_degrees == (0.0, 7.5, 15.0, 22.5, 30.0)
    assert config.reference.endpoint_repeat_count == 1
    assert config.reference.fisher_matrix_dtype == "float64"
    assert config.run_id.startswith("rotated_mnist_phase2_smoke__")


def test_reference_path_and_repeat_noise_use_fisher_quadratics() -> None:
    angles = (0.0, 1.0, 2.0)
    parameters = (
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 2.0]),
    )
    fishers = (
        torch.diag(torch.tensor([2.0, 3.0])),
        torch.diag(torch.tensor([4.0, 5.0])),
        torch.diag(torch.tensor([6.0, 7.0])),
    )

    path = reference_path_diagnostics(angles, parameters, fishers)
    assert path[0]["left_fisher_quadratic"] == pytest.approx(2.0)
    assert path[0]["right_fisher_quadratic"] == pytest.approx(4.0)
    assert path[1]["left_fisher_quadratic"] == pytest.approx(20.0)

    noise = repeated_fit_noise(
        {
            0.0: (
                ("primary", parameters[0], fishers[0]),
                ("repeat", torch.tensor([0.5, 0.0]), fishers[1]),
            )
        }
    )
    assert noise[0]["primary_fisher_quadratic"] == pytest.approx(0.5)
    assert noise[0]["fisher_frobenius_change"] > 0.0


def test_offline_risk_coefficients_are_diagnostic_only() -> None:
    rows = offline_risk_coefficients(
        [
            {
                "from_angle_degrees": 0.0,
                "to_angle_degrees": 1.0,
                "left_fisher_quadratic": 2.0,
            }
        ],
        parameter_count=4,
        initialization_sample_size=100,
        batch_size=2,
        fixed_pi=0.1,
    )

    assert rows[0]["idealized_old_covariance_risk"] == pytest.approx(0.04)
    assert rows[0]["idealized_new_covariance_risk"] == pytest.approx(2.0)
    assert rows[0]["instantaneous_unclipped_pi"] == pytest.approx(2.04 / 4.04)


def test_feasibility_gate_compares_fisher_change_with_repeat_noise() -> None:
    gate = feasibility_gate(
        zero_shot_environment_accuracy_30=0.7,
        reference_environment_accuracy_30=0.8,
        path_rows=[{"relative_fisher_frobenius_change": 0.3}],
        noise_rows=[{"relative_fisher_frobenius_change": 0.1}],
        config=GateConfig(0.75, 0.03, 1.5),
    )

    assert gate["recommendation"] == "proceed"
    assert gate["fisher_change_signal_to_noise"] == pytest.approx(3.0)
