from __future__ import annotations

import torch

from mnist_experiment.rotated_mnist.plan12.analysis import (
    _identified_logits,
    _target_fit_valid,
    decompose_estimates,
)
from mnist_experiment.rotated_mnist.plan12.refresh_notebook import (
    _phase3_scientific_audit,
)
from mnist_experiment.rotated_mnist.plan12.sandwich import penalized_sandwich
from src.representations import DenseFisher


def test_bias_variance_decomposition_has_known_components() -> None:
    estimates = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float64)
    target = torch.tensor([0.5, 0.5], dtype=torch.float64)
    reference = torch.zeros(2, dtype=torch.float64)
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    fisher = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    result = decompose_estimates(estimates, target, reference, basis, fisher)
    assert result["parameter_variance"] == 2.0
    assert result["fisher_variance"] == 5.0
    assert result["estimator_bias_squared"] == {
        "full": 0.5,
        "resolved": 0.25,
        "unresolved": 0.25,
    }
    assert result["regularization_bias_squared"] == result["estimator_bias_squared"]
    assert result["total_bias_squared"] == {
        "full": 2.0,
        "resolved": 1.0,
        "unresolved": 1.0,
    }
    assert result["fisher_total_bias_squared"] == 5.0
    assert result["fisher_total_mse"] == 10.0


def test_functional_logits_ignore_common_class_shift() -> None:
    logits = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 0.0, 5.0]])
    shift = torch.tensor([[17.0], [-4.0]])
    torch.testing.assert_close(_identified_logits(logits + shift), _identified_logits(logits))


def test_penalized_sandwich_matches_its_dense_formula() -> None:
    gradients = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
        dtype=torch.float64,
    )
    penalty = DenseFisher(torch.diag(torch.tensor([0.25, 0.5], dtype=torch.float64)))
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    diagnostics, solved = penalized_sandwich(gradients, penalty, beta=2.0, resolved_basis=basis)
    meat = gradients.mT @ gradients / gradients.shape[0]
    bread = meat + 2.0 * penalty.matrix
    expected = torch.linalg.solve(
        bread + diagnostics["numerical_damping"] * torch.eye(2, dtype=torch.float64),
        gradients.mT,
    ) / gradients.shape[0]
    torch.testing.assert_close(solved, expected)
    assert abs(diagnostics["covariance_trace"] - float(expected.square().sum())) < 1e-14
    assert abs(
        diagnostics["covariance_trace"]
        - diagnostics["resolved_covariance_trace"]
        - diagnostics["unresolved_covariance_trace"]
    ) < 1e-14


def test_target_validity_uses_the_frozen_convergence_gates() -> None:
    summary = {
        "condition": {"no_update": False},
        "repeat_parameter_distance": 0.009,
        "fit_health": {"relative_final_gradient_norm": 0.049, "objective_decrease": 1.0},
        "repeat_fit_health": {"relative_final_gradient_norm": 0.05, "objective_decrease": 0.0},
    }
    assert _target_fit_valid(summary)
    summary["fit_health"]["relative_final_gradient_norm"] = 0.051
    assert not _target_fit_valid(summary)
    summary["condition"]["no_update"] = True
    assert _target_fit_valid(summary)


def test_phase3_audit_does_not_confuse_predictive_success_with_calibration() -> None:
    effect = {
        "mean": 0.2,
        "ci95_low": 0.1,
        "ci95_high": 0.3,
        "positive_count": 8,
        "replicas": 8,
    }
    analyses = {
        "phase1_analysis": {
            "selection": {
                "isotropic": {
                    "eligible_improving_conditions": ["isotropic_1em01", "isotropic_1ep00"]
                }
            },
            "aggregate": [
                {"condition": "isotropic_1em01", "ridge_ratio": 0.1},
                {"condition": "isotropic_1ep00", "ridge_ratio": 1.0},
            ],
        },
        "phase3_analysis": {
            "selection": {"selected_scale_ratio": 40.0},
            "mean_empirical_to_theoretical_ratio": 1.5,
            "pseudo_tail_pit_values": [0.01, 0.02],
        },
        "phase4_analysis": {
            "conditions": [
                {"name": "gauge_no_ridge"},
                {"name": "spectral_selector"},
            ],
            "paired_against_gauge_no_ridge": [
                {
                    "condition": "spectral_selector",
                    "schedule": "linear",
                    "nll_auc_gain": effect,
                    "accuracy_auc_gain": effect,
                }
            ],
            "rows": [
                {
                    "schedule": "linear",
                    "condition": "gauge_no_ridge",
                    "mean_total_displacement_squared": 0.2,
                },
                {
                    "schedule": "linear",
                    "condition": "spectral_selector",
                    "mean_total_displacement_squared": 0.0001,
                },
            ],
        },
    }

    audit = _phase3_scientific_audit(analyses)

    assert audit is not None
    assert audit["selector_contract_status"] == "failed_not_calibrated_or_deployable"
    assert audit["phase1_useful_isotropic_ratio_interval"] == [0.1, 1.0]
    assert not audit["lands_in_useful_region"]
    assert audit["phase4_predictive_confirmation"][0]["nll_auc_gain"]["mean"] == 0.2
    assert audit["phase4_movement_diagnostic"][0]["spectral_to_no_ridge_displacement_ratio"] == 0.0005
    assert not audit["phase4_no_update_control_present"]
    assert "did not include" in audit["phase4_no_update_limitation"]
    assert "do not retroactively" in audit["interpretation"].lower()
