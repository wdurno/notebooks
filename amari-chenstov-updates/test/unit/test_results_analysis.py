import dataclasses
import json
from pathlib import Path

import pytest
import torch

from src.artifacts import RunStore
from src.config import load_config
from src.results_analysis import (
    AnalysisArtifactError,
    PHASE4_METHODS_BY_SCHEMA,
    PHASE6_METHODS_BY_SCHEMA,
    discover_phase4_runs,
    discover_phase6_runs,
    load_selected_phase3_runs,
    phase4_replica_summaries,
    phase6_condition_rows,
    phase6_replica_summaries,
    select_phase4_condition_evidence,
    select_principal_phase4_runs,
)


REPO_ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = REPO_ROOT / "mnist_experiment" / "configs" / "smoke.json"
PHASE6_SMOKE_CONFIG = (
    REPO_ROOT / "mnist_experiment" / "configs" / "phase6_smoke.json"
)


def _phase4_config(
    experiment: str,
    *,
    schema: int,
    half_life: float | None,
):
    base = load_config(SMOKE_CONFIG)
    estimator = dataclasses.replace(
        base.estimator,
        ridge_half_life_steps=half_life,
        ridge_amplitude_epsilon=(None if half_life is None else 1e-6),
        ridge_coherence_threshold=(None if half_life is None else 0.75),
    )
    return dataclasses.replace(
        base,
        schema_version=schema,
        experiment=experiment,
        estimator=estimator,
    )


def _condition_row(method: str, step: int, p_value: float) -> dict:
    return {
        "method": method,
        "step": step,
        "p": p_value,
        "relative_frobenius_error": 0.1 + step,
        "applied_correction_fro": float(step),
        "projection": {"relative_projection_distance": 0.0},
        "ridge": None,
    }


def _write_phase4_run(
    root: Path,
    experiment: str,
    *,
    metric_schema: int,
    config_schema: int,
    half_life: float | None,
) -> Path:
    config = _phase4_config(
        experiment,
        schema=config_schema,
        half_life=half_life,
    )
    methods = PHASE4_METHODS_BY_SCHEMA[metric_schema]
    p_values = (0.0, 0.5, 1.0)
    session = RunStore(root).begin(config, REPO_ROOT)
    session.write_torch(
        "phase4_trajectory.pt",
        {
            "schema_version": 1,
            "content_hash": "shared-trajectory",
            "p_values": p_values,
            "parameter_layout": {
                "total_numel": 512,
                "parameters": [{"name": "weight", "start": 0, "stop": 512}],
            },
        },
    )
    session.write_json(
        "phase4_metrics.json",
        {
            "phase4_metric_schema_version": metric_schema,
            "replica_bundle_id": "bundle",
            "trajectory_hash": "shared-trajectory",
            "methods": list(methods),
            "common_steps": [{"step": step} for step in range(3)],
            "condition_steps": [
                _condition_row(method, step, p_value)
                for method in methods
                for step, p_value in enumerate(p_values)
            ],
            "driver": {"trajectory_elapsed_seconds": 1.0},
            "replay_elapsed_seconds": 2.0,
            "peak_cuda_memory_bytes": 3,
            "peak_process_rss_bytes": 4,
            "artifact_payload_bytes_before_metrics": 5,
        },
    )
    return session.complete(["phase4_trajectory.pt", "phase4_metrics.json"])


def _write_phase6_run(root: Path) -> Path:
    config = load_config(PHASE6_SMOKE_CONFIG)
    methods = PHASE6_METHODS_BY_SCHEMA[1]
    p_values = (0.0, 0.5, 1.0)
    parameters = torch.tensor(
        [[0.0, 0.0], [0.1, -0.1], [0.2, -0.05]],
        dtype=torch.float64,
    )
    path_hashes = {method: f"path-{method}" for method in methods}
    session = RunStore(root).begin(config, REPO_ROOT)
    session.write_torch(
        "phase6_trajectories.pt",
        {
            "schema_version": 1,
            "stream_plan_hash": "stream",
            "p_values": p_values,
            "observation_indices": ((1, 2), (3, 4), (5, 6)),
            "parameter_layout": {
                "total_numel": 2,
                "parameters": [
                    {
                        "name": "weight",
                        "shape": [2],
                        "start": 0,
                        "stop": 2,
                    }
                ],
            },
            "conditions": {
                method: {
                    "content_hash": path_hashes[method],
                    "parameters": parameters,
                    "displacements": parameters[1:] - parameters[:-1],
                    "optimizer_state": {},
                }
                for method in methods
            },
        },
    )
    condition_rows = []
    reference_rows = []
    for method in methods:
        for step, p_value in enumerate(p_values):
            parameter_hash = f"{method}-{step}"
            proposal = (
                None
                if step == 2
                else {
                    "data_loss_before": 1.0,
                    "data_loss_after": 0.9,
                    "ewc_penalty_after": 0.1,
                    "displacement_norm": 0.2,
                    "accepted_displacement_norm": 0.2,
                    "fisher_weighted_displacement_norm": 0.3,
                    "effective_ewc_strength": 1.0,
                    "post_optimization_scaling_applied": False,
                }
            )
            condition_rows.append(
                {
                    "method": method,
                    "step": step,
                    "p": p_value,
                    "parameter_hash": parameter_hash,
                    "reference_parameter_hash": parameter_hash,
                    "proposal": proposal,
                    "adaptation_weight": 0.5,
                    "effective_ewc_strength": 1.0,
                    "relative_frobenius_error": 0.1,
                    "distance_from_initial": float(step),
                    "before_non_nine_accuracy": 0.8 - 0.1 * step,
                    "before_nine_accuracy": 0.1 * step,
                    "before_balanced_accuracy": 0.5,
                    "before_non_nine_nll": 0.2 + 0.1 * step,
                    "before_nine_nll": 3.0 - 0.5 * step,
                }
            )
            reference_rows.append(
                {
                    "method": method,
                    "step": step,
                    "p": p_value,
                    "parameter_hash": parameter_hash,
                }
            )
    session.write_json(
        "phase6_metrics.json",
        {
            "phase6_metric_schema_version": 1,
            "replica_bundle_id": "bundle",
            "stream_plan_hash": "stream",
            "methods": list(methods),
            "adaptation": {
                "adaptation_weight": 0.5,
                "pi_max": 0.95,
                "ewc_multiplier": 1.0,
                "effective_ewc_strength": 1.0,
                "objective_normalization": (
                    "mean_new_loss_plus_old_to_new_odds"
                ),
                "post_optimization_scaling": False,
            },
            "pairing": {
                "shared_initialization": True,
                "shared_observation_stream": True,
                "path_diverged": True,
            },
            "path_hashes": path_hashes,
            "path_divergence": [
                {
                    "step": step,
                    "p": p_value,
                    "maximum_pairwise_parameter_distance": float(step),
                }
                for step, p_value in enumerate(p_values)
            ],
            "condition_steps": condition_rows,
            "references": reference_rows,
        },
    )
    return session.complete(
        ["phase6_trajectories.pt", "phase6_metrics.json"]
    )


def test_phase6_loader_enforces_coupled_path_contract(tmp_path: Path) -> None:
    path = _write_phase6_run(tmp_path / "phase6")

    runs = discover_phase6_runs(path.parent)
    rows = phase6_condition_rows(runs)
    summaries = phase6_replica_summaries(rows)

    assert len(runs) == 1
    assert len(rows) == 6 * 3
    assert len(summaries) == 6
    assert {row["effective_ewc_strength"] for row in rows} == {1.0}
    assert {row["final_nine_accuracy"] for row in summaries} == {0.2}


def test_phase6_loader_rejects_reference_from_another_path(
    tmp_path: Path,
) -> None:
    path = _write_phase6_run(tmp_path / "phase6")
    metrics_path = path / "phase6_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["references"][0]["parameter_hash"] = "wrong-path"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(AnalysisArtifactError, match="reference is not keyed"):
        discover_phase6_runs(path.parent)


def test_schema_two_supersedes_legacy_without_duplicating_controls(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    legacy = _write_phase4_run(
        root,
        "legacy",
        metric_schema=1,
        config_schema=4,
        half_life=None,
    )
    ridge = _write_phase4_run(
        root,
        "ridge",
        metric_schema=2,
        config_schema=5,
        half_life=8.0,
    )

    runs = discover_phase4_runs(root)
    selected, inventory = select_phase4_condition_evidence(runs)

    assert {row["run_id"] for row in selected} == {ridge.name}
    assert len(selected) == 6 * 3
    statuses = {row["run_id"]: row["selection_status"] for row in inventory}
    assert statuses[legacy.name] == "superseded"
    assert statuses[ridge.name] == "selected"


def test_half_life_runs_share_raw_controls_but_retain_ridge_treatments(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    first = _write_phase4_run(
        root,
        "ridge-h4",
        metric_schema=2,
        config_schema=5,
        half_life=4.0,
    )
    second = _write_phase4_run(
        root,
        "ridge-h16",
        metric_schema=2,
        config_schema=5,
        half_life=16.0,
    )

    selected, inventory = select_phase4_condition_evidence(
        discover_phase4_runs(root)
    )

    raw = [row for row in selected if not row["method"].startswith("ridge_")]
    ridge = [row for row in selected if row["method"].startswith("ridge_")]
    assert len(raw) == 4 * 3
    assert len(ridge) == 2 * 2 * 3
    assert {
        row["ridge_half_life_steps"] for row in ridge
    } == {4.0, 16.0}
    assert {row["selection_status"] for row in inventory} == {
        "partially selected",
        "selected",
    }
    assert {row["run_id"] for row in inventory} == {first.name, second.name}


def test_replica_summary_uses_only_interior_trajectory_points(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    _write_phase4_run(
        root,
        "ridge",
        metric_schema=2,
        config_schema=5,
        half_life=8.0,
    )
    selected, _ = select_phase4_condition_evidence(
        discover_phase4_runs(root)
    )

    summaries = phase4_replica_summaries(selected)

    assert len(summaries) == 6
    assert {row["interior_step_count"] for row in summaries} == {1}
    assert {
        row["mean_relative_frobenius_error"] for row in summaries
    } == {1.1}


def test_unknown_phase4_schema_fails_clearly(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run_path = _write_phase4_run(
        root,
        "ridge",
        metric_schema=2,
        config_schema=5,
        half_life=8.0,
    )
    metrics_path = run_path / "phase4_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["phase4_metric_schema_version"] = 99
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(AnalysisArtifactError, match="unsupported Phase 4"):
        discover_phase4_runs(root)


def test_incomplete_phase4_run_fails_clearly(tmp_path: Path) -> None:
    run_path = tmp_path / "runs" / "incomplete"
    run_path.mkdir(parents=True)
    (run_path / "phase4_metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AnalysisArtifactError, match="incomplete"):
        discover_phase4_runs(tmp_path / "runs")


def test_principal_selection_requires_named_baseline_and_prefixes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    baseline = _write_phase4_run(
        root,
        "accepted-baseline",
        metric_schema=2,
        config_schema=5,
        half_life=8.0,
    )
    pilot = _write_phase4_run(
        root,
        "mnist_lfu_phase5_factor",
        metric_schema=2,
        config_schema=5,
        half_life=8.0,
    )
    _write_phase4_run(
        root,
        "exploratory-failed",
        metric_schema=1,
        config_schema=4,
        half_life=None,
    )
    runs = discover_phase4_runs(root)

    selected = select_principal_phase4_runs(
        runs,
        required_run_ids=[baseline.name],
    )

    assert {run.run_id for run in selected} == {baseline.name, pilot.name}
    with pytest.raises(AnalysisArtifactError, match="baseline runs are missing"):
        select_principal_phase4_runs(
            runs,
            required_run_ids=["does-not-exist"],
        )


def test_phase3_selection_is_explicit_and_checks_coverage(
    tmp_path: Path,
) -> None:
    base = load_config(SMOKE_CONFIG)
    config = dataclasses.replace(
        base,
        experiment="phase3-selected",
        reference=dataclasses.replace(
            base.reference,
            stencil_p_values=[0.25],
        ),
    )
    root = tmp_path / "phase3"
    session = RunStore(root).begin(config, REPO_ROOT)
    session.write_json(
        "phase3_metrics.json",
        {
            "reference_fisher_schema_version": 1,
            "replica_bundle_id": "bundle",
            "device": "cpu",
            "peak_cuda_memory_bytes": None,
            "checkpoints": [{"p": 0.25}],
        },
    )
    path = session.complete(["phase3_metrics.json"])

    runs = load_selected_phase3_runs(
        root,
        [path.name],
        expected_p_values=[0.25],
    )
    assert runs[0].run_id == path.name

    with pytest.raises(AnalysisArtifactError, match="cover exactly"):
        load_selected_phase3_runs(
            root,
            [path.name],
            expected_p_values=[0.5],
        )
