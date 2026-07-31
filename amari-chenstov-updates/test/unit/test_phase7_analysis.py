import json
from pathlib import Path

import pytest

from src.artifacts import RunStore
from src.config import load_config
from src.results_analysis import (
    AnalysisArtifactError,
    load_phase7_coupled_run,
    load_phase7_fixed_run,
    phase7_coupled_rows,
    phase7_coupled_summaries,
    phase7_fixed_rows,
    phase7_fixed_summaries,
)


REPO_ROOT = Path(__file__).parents[2]
FIXED_CONFIG = (
    REPO_ROOT / "mnist_experiment" / "configs" / "phase7_smoke.json"
)
COUPLED_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "configs"
    / "phase7_coupled_smoke.json"
)
LANCZOS_HASH = (
    "643bb7562ad7ad2d087229414d7adf114f575b342f462be619f90be920691634"
)


def _write_fixed_run(root: Path) -> Path:
    config = load_config(FIXED_CONFIG)
    session = RunStore(root).begin(config, REPO_ROOT)
    rows = []
    for rank in (0, 2, 4):
        condition = "diagonal" if rank == 0 else f"low_rank_diagonal_r{rank}"
        for step, p_value in enumerate((0.0, 0.5, 1.0)):
            rows.append(
                {
                    "condition": condition,
                    "representation": (
                        "diagonal" if rank == 0 else "low_rank_diagonal"
                    ),
                    "requested_rank": rank,
                    "realized_rank": rank,
                    "status": "completed",
                    "step": step,
                    "p": p_value,
                    "relative_frobenius_error_to_dense": (
                        2.0 if rank == 4 and step == 2 else 0.2
                    ),
                    "relative_frobenius_error_to_reference": 0.3,
                    "fixed_probe_matvec_relative_error": 0.1,
                    "mean_probe_quadratic_relative_error": 0.1,
                    "leading_eigenvector_alignment": 0.9,
                    "lanczos_represented_diagonal_relative_error": 0.0,
                    "representation_storage_bytes": 8 * (rank + 1),
                    "update_elapsed_seconds": 0.01,
                }
            )
    session.write_json(
        "phase7_metrics.json",
        {
            "phase7_metric_schema_version": 1,
            "run_kind": "fixed_rank_sweep",
            "trajectory_hash": "trajectory",
            "parameter_count": 512,
            "rank_grid": [0, 2, 4],
            "condition_status": {
                "diagonal": {"status": "completed"},
                "low_rank_diagonal_r2": {"status": "completed"},
                "low_rank_diagonal_r4": {"status": "completed"},
            },
            "condition_steps": rows,
            "legacy_lanczos_source_hash": LANCZOS_HASH,
            "expected_legacy_lanczos_source_hash": LANCZOS_HASH,
        },
    )
    for name in (
        "phase7_trajectory.pt",
        "phase7_representations.pt",
        "phase7_dense_checkpoints.pt",
        "phase7_reference_plans.pt",
    ):
        session.write_torch(name, {"schema_version": 1})
    return session.complete(
        [
            "phase7_metrics.json",
            "phase7_trajectory.pt",
            "phase7_representations.pt",
            "phase7_dense_checkpoints.pt",
            "phase7_reference_plans.pt",
        ]
    )


def _write_coupled_run(root: Path) -> Path:
    config = load_config(COUPLED_CONFIG)
    methods = (
        "dense_ridge_full",
        "diagonal_ridge_full",
        "low_rank_diagonal_r4",
    )
    session = RunStore(root).begin(config, REPO_ROOT)
    rows = []
    references = []
    for method in methods:
        for step, p_value in enumerate((0.0, 0.5, 1.0)):
            proposal = (
                None
                if step == 2
                else {
                    "effective_ewc_strength": 1.0,
                    "post_optimization_scaling_applied": False,
                }
            )
            rows.append(
                {
                    "method": method,
                    "step": step,
                    "p": p_value,
                    "proposal": proposal,
                    "representation": (
                        "dense"
                        if method == "dense_ridge_full"
                        else "diagonal"
                        if method == "diagonal_ridge_full"
                        else "low_rank_diagonal"
                    ),
                    "relative_frobenius_error": 0.1 + 0.1 * step,
                    "before_nll": 2.0 - 0.1 * step,
                    "before_nine_nll": 3.0 - 0.2 * step,
                    "before_non_nine_nll": 0.2 + 0.1 * step,
                    "before_balanced_accuracy": 0.5,
                    "distance_from_initial": float(step),
                    "update_elapsed_seconds": 0.01,
                }
            )
            references.append(
                {"method": method, "step": step, "p": p_value}
            )
    session.write_json(
        "phase7_coupled_metrics.json",
        {
            "phase7_coupled_metric_schema_version": 1,
            "run_kind": "structured_coupled",
            "stream_plan_hash": "stream",
            "methods": list(methods),
            "selected_rank": 4,
            "adaptation": {
                "adaptation_weight": 0.5,
                "ewc_multiplier": 1.0,
                "effective_ewc_strength": 1.0,
                "objective_normalization": (
                    "mean_new_loss_plus_old_to_new_odds"
                ),
                "post_optimization_scaling": False,
            },
            "pairing": {"path_diverged": True},
            "path_hashes": {method: f"path-{method}" for method in methods},
            "condition_steps": rows,
            "references": references,
        },
    )
    for name in (
        "phase7_coupled_trajectories.pt",
        "phase7_coupled_checkpoints.pt",
        "phase7_coupled_reference_plans.pt",
    ):
        session.write_torch(name, {"schema_version": 1})
    return session.complete(
        [
            "phase7_coupled_metrics.json",
            "phase7_coupled_trajectories.pt",
            "phase7_coupled_checkpoints.pt",
            "phase7_coupled_reference_plans.pt",
        ]
    )


def test_phase7_fixed_loader_labels_finite_instability(tmp_path: Path) -> None:
    run = load_phase7_fixed_run(_write_fixed_run(tmp_path / "fixed"))
    rows = phase7_fixed_rows([run])
    summaries = phase7_fixed_summaries(rows)
    statuses = {
        row["requested_rank"]: row["numerical_status"] for row in summaries
    }

    assert statuses == {
        0: "completed",
        2: "completed",
        4: "numerically_unstable",
    }


def test_phase7_coupled_loader_summarizes_each_path(tmp_path: Path) -> None:
    run = load_phase7_coupled_run(_write_coupled_run(tmp_path / "coupled"))
    rows = phase7_coupled_rows([run])
    summaries = phase7_coupled_summaries(rows)

    assert len(rows) == 9
    assert len(summaries) == 3
    assert all(
        row["mean_fisher_error"] == pytest.approx(0.25)
        for row in summaries
    )


def test_phase7_loader_rejects_a_changed_lanczos_source_hash(
    tmp_path: Path,
) -> None:
    path = _write_fixed_run(tmp_path / "fixed")
    metrics_path = path / "phase7_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["legacy_lanczos_source_hash"] = "changed"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(AnalysisArtifactError, match="source hash"):
        load_phase7_fixed_run(path)
