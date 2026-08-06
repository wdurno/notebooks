import json
from pathlib import Path

import pytest
import torch

from src.artifacts import MANIFEST_SCHEMA_VERSION
from src.config import ExperimentConfig
from src.results_analysis import (
    AnalysisArtifactError,
    load_phase8_controller_run,
    phase8_controller_rows,
    phase8_controller_summaries,
)


CONFIG_PATH = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_controller_smoke.json"
)
CONVERGENCE_CONFIG_PATH = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_convergence.json"
)
CORRECTED_CONFIG_PATH = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_corrected_smoke.json"
)
LBFGS_CONFIG_PATH = (
    Path(__file__).parents[2]
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_k100_plugin.json"
)


def _write_run(tmp_path: Path, *, metric_schema: int = 3) -> Path:
    source = (
        LBFGS_CONFIG_PATH
        if metric_schema == 6
        else CORRECTED_CONFIG_PATH
        if metric_schema == 5
        else CONVERGENCE_CONFIG_PATH
        if metric_schema == 4
        else CONFIG_PATH
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["metric_schema_version"] = metric_schema
    trajectory_schema = {2: 2, 3: 2, 4: 3, 5: 4, 6: 4}[metric_schema]
    state_schema = {2: 1, 3: 2, 4: 3, 5: 4, 6: 4}[metric_schema]
    config = ExperimentConfig.from_mapping(raw)
    run = tmp_path / config.run_id
    run.mkdir()
    (run / "COMPLETED").touch()
    (run / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "run_id": config.run_id,
                "config_hash": config.config_hash,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    methods = ["dense_ridge_full"]
    row = {
        "method": methods[0],
        "step": 0,
        "p": 0.0,
        "same_pi_consumed": True,
        "controller": {
            "applied_pi": 0.5,
            "lower_bound_active": False,
            "upper_bound_active": False,
        },
        (
            "trace_relative_error_to_oracle_residual"
            if metric_schema >= 3
            else "trace_relative_error_to_oracle"
        ): 0.25,
        "relative_frobenius_error": 0.1,
        "before_balanced_accuracy": 0.8,
        "before_nine_accuracy": 0.7,
        "before_non_nine_accuracy": 0.9,
        "before_nine_nll": 0.6,
        "before_non_nine_nll": 0.2,
        "parameter_squared_error_to_oracle": 0.02,
        "proposal": {
            "optimization_guard": "monotone_objective_backtracking",
            "inner_steps": 100,
            "optimizer_iterations": 27,
            "optimizer_function_evaluations": 35,
            "stopping_reason": "lbfgs_convergence_tolerance",
            "backtracking_rejections": 3,
            "maximum_backtracks": 2,
            "minimum_learning_rate": 0.00025,
            "relative_final_gradient_norm": 0.2,
            "final_gradient_rms": 0.01,
        },
    }
    metrics = {
        "phase8_metric_schema_version": metric_schema,
        "run_kind": "unified_controller",
        "methods": methods,
        "policy": "optimal_plugin",
        "unified_pi_contract": True,
        "post_optimization_scaling": False,
        "oracle_path_hash": "a" * 64,
        "condition_steps": [row],
        "path_hashes": {methods[0]: "b" * 64},
    }
    if metric_schema >= 3:
        metrics.update(
            {
                "fisher_inverse_used": False,
                "trace_estimator": "accepted_displacement_residual_moments",
            }
        )
    if metric_schema >= 4:
        metrics.update(
            {
                "oracle_convergence_contract": {},
                "residual_dependence": {methods[0]: {}},
                "references": [],
            }
        )
    if metric_schema >= 5:
        metrics.update(
            {
                "controller_displacement_law": (
                    "u_approx_pi_times_drift_plus_noise"
                ),
                "trend_estimand": "environmental_parameter_displacement",
                "trend_observation": "accepted_displacement_divided_by_pi",
                "covariance_residual": (
                    "u_minus_pi_times_predictable_drift"
                ),
                "oracle_covariance_residual": (
                    "u_minus_pi_times_oracle_drift"
                ),
                "ewc_optimality_diagnostic": "final_objective_gradient",
                "oracle_path_provenance": {
                    "mode": "built_in_run",
                    "content_hash": "a" * 64,
                },
            }
        )
    if metric_schema >= 6:
        metrics.update(
            {
                "optimizer_budget_role": (
                    "fixed_compute_budget_not_convergence_claim"
                ),
                "optimizer_accounting": (
                    "structured_iterations_and_function_evaluations"
                ),
            }
        )
    (run / "phase8_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    torch.save(
        {
            "schema_version": trajectory_schema,
            "p_values": [0.0],
            "conditions": {methods[0]: {}},
        },
        run / "phase8_trajectories.pt",
    )
    torch.save(
        {
            "schema_version": state_schema,
            "conditions": {methods[0]: {}},
        },
        run / "phase8_controller_states.pt",
    )
    torch.save(
        {
            "schema_version": state_schema,
            "path": {"content_hash": "a" * 64},
        },
        run / "phase8_reference_optimum.pt",
    )
    torch.save({"schema_version": 2}, run / "phase8_checkpoints.pt")
    return run


def test_phase8_loader_and_summaries_are_strict_and_lightweight(tmp_path: Path) -> None:
    run = load_phase8_controller_run(_write_run(tmp_path))
    rows = phase8_controller_rows([run])
    summaries = phase8_controller_summaries(rows)

    assert rows[0]["theoretical_status"] == "legacy_attenuated_trend_phase8"
    assert summaries[0]["mean_pi"] == 0.5
    assert summaries[0]["mean_trace_relative_error"] == 0.25
    assert summaries[0]["backtracking_rejections"] == 3
    assert summaries[0]["maximum_backtracks"] == 2
    assert summaries[0]["minimum_learning_rate"] == 0.00025
    assert rows[0]["environment_accuracy"] == 0.9
    assert rows[0]["environment_nll"] == 0.2
    assert summaries[0]["environment_accuracy_auc"] == 0.9
    assert summaries[0]["nine_accuracy_auc"] == 0.7
    assert summaries[0]["non_nine_accuracy_auc"] == 0.9


def test_phase8_loader_retains_legacy_spectral_smoke_label(tmp_path: Path) -> None:
    run = load_phase8_controller_run(_write_run(tmp_path, metric_schema=2))
    rows = phase8_controller_rows([run])

    assert rows[0]["theoretical_status"] == "legacy_spectral_trace_smoke"


def test_phase8_loader_accepts_adaptive_convergence_schema(tmp_path: Path) -> None:
    run = load_phase8_controller_run(_write_run(tmp_path, metric_schema=4))
    rows = phase8_controller_rows([run])

    assert run.config.schema_version == 8
    assert rows[0]["theoretical_status"] == "legacy_attenuated_trend_phase8"


def test_phase8_loader_accepts_corrected_normalized_drift_schema(
    tmp_path: Path,
) -> None:
    run = load_phase8_controller_run(_write_run(tmp_path, metric_schema=5))
    rows = phase8_controller_rows([run])
    summaries = phase8_controller_summaries(rows)

    assert run.config.schema_version == 9
    assert rows[0]["theoretical_status"] == "corrected_normalized_drift_phase8"
    assert summaries[0]["mean_relative_final_gradient_norm"] == 0.2
    assert summaries[0]["maximum_relative_final_gradient_norm"] == 0.2
    assert summaries[0]["mean_final_gradient_rms"] == 0.01


def test_phase8_loader_accepts_compute_budget_schema(tmp_path: Path) -> None:
    run = load_phase8_controller_run(_write_run(tmp_path, metric_schema=6))
    summaries = phase8_controller_summaries(phase8_controller_rows([run]))

    assert run.config.schema_version == 10
    assert summaries[0]["mean_optimizer_iterations"] == 27
    assert summaries[0]["maximum_optimizer_function_evaluations"] == 35


def test_phase8_loader_rejects_unified_contract_violation(tmp_path: Path) -> None:
    path = _write_run(tmp_path)
    metrics_path = path / "phase8_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["condition_steps"][0]["same_pi_consumed"] = False
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(AnalysisArtifactError, match="unified pi"):
        load_phase8_controller_run(path)
