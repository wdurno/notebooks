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


def _write_run(tmp_path: Path, *, metric_schema: int = 3) -> Path:
    source = CONVERGENCE_CONFIG_PATH if metric_schema == 4 else CONFIG_PATH
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["metric_schema_version"] = metric_schema
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
        "parameter_squared_error_to_oracle": 0.02,
        "proposal": {
            "optimization_guard": "monotone_objective_backtracking",
            "backtracking_rejections": 3,
            "maximum_backtracks": 2,
            "minimum_learning_rate": 0.00025,
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
    (run / "phase8_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    torch.save(
        {
            "schema_version": 3 if metric_schema == 4 else 2,
            "p_values": [0.0],
            "conditions": {methods[0]: {}},
        },
        run / "phase8_trajectories.pt",
    )
    torch.save(
        {
            "schema_version": 3 if metric_schema == 4 else 2 if metric_schema == 3 else 1,
            "conditions": {methods[0]: {}},
        },
        run / "phase8_controller_states.pt",
    )
    torch.save(
        {
            "schema_version": 3 if metric_schema == 4 else 2 if metric_schema == 3 else 1,
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

    assert rows[0]["theoretical_status"] == "inversion_free_phase8"
    assert summaries[0]["mean_pi"] == 0.5
    assert summaries[0]["mean_trace_relative_error"] == 0.25
    assert summaries[0]["backtracking_rejections"] == 3
    assert summaries[0]["maximum_backtracks"] == 2
    assert summaries[0]["minimum_learning_rate"] == 0.00025


def test_phase8_loader_retains_legacy_spectral_smoke_label(tmp_path: Path) -> None:
    run = load_phase8_controller_run(_write_run(tmp_path, metric_schema=2))
    rows = phase8_controller_rows([run])

    assert rows[0]["theoretical_status"] == "legacy_spectral_trace_smoke"


def test_phase8_loader_accepts_adaptive_convergence_schema(tmp_path: Path) -> None:
    run = load_phase8_controller_run(_write_run(tmp_path, metric_schema=4))
    rows = phase8_controller_rows([run])

    assert run.config.schema_version == 8
    assert rows[0]["theoretical_status"] == "inversion_free_phase8"


def test_phase8_loader_rejects_unified_contract_violation(tmp_path: Path) -> None:
    path = _write_run(tmp_path)
    metrics_path = path / "phase8_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["condition_steps"][0]["same_pi_consumed"] = False
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(AnalysisArtifactError, match="unified pi"):
        load_phase8_controller_run(path)
