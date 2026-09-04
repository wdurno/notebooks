import json
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.analysis import (
    discover_completed_audits,
    discover_completed_runs,
    phase3_paired_contrast,
)
from mnist_experiment.rotated_mnist.artifacts import RotatedIncompleteRunError
from mnist_experiment.validate_results_notebook import (
    load_notebook,
    validate_notebook_source,
)


REPO_ROOT = Path(__file__).parents[3]
NOTEBOOK = REPO_ROOT / "mnist_experiment" / "rotated_mnist" / "results.ipynb"


def test_discovery_refuses_incomplete_runs(tmp_path: Path) -> None:
    incomplete = tmp_path / ".incomplete" / "run"
    incomplete.mkdir(parents=True)
    with pytest.raises(RotatedIncompleteRunError, match="incomplete"):
        discover_completed_runs(tmp_path)


def test_notebook_is_valid_and_artifact_only() -> None:
    notebook = load_notebook(NOTEBOOK)
    validate_notebook_source(notebook)
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook["nbformat"] == 4
    assert "Artifact-only analysis" in source
    assert "load_completed_summaries" in source
    assert "run_experiment" not in source
    assert "download=" not in source
    json.dumps(notebook, allow_nan=False)


def test_notebook_bootstraps_repo_before_package_import() -> None:
    notebook = load_notebook(NOTEBOOK)
    first_code = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "sys.path.insert(0, str(repo_root))" in first_code
    assert first_code.index("sys.path.insert") < first_code.index(
        "from mnist_experiment.rotated_mnist.analysis"
    )


def test_audit_discovery_refuses_incomplete_runs(tmp_path: Path) -> None:
    incomplete = tmp_path / ".incomplete" / "audit"
    incomplete.mkdir(parents=True)
    with pytest.raises(RotatedIncompleteRunError, match="incomplete"):
        discover_completed_audits(tmp_path)


def test_audit_discovery_ignores_empty_lifecycle_directory(tmp_path: Path) -> None:
    (tmp_path / ".incomplete").mkdir()

    assert discover_completed_audits(tmp_path) == ()


def test_phase3_contrast_uses_paired_ewc_minus_current_direction() -> None:
    class Run:
        run_summary = {
            "condition_summaries": {
                "current_only": {
                    "environment_accuracy_auc": 0.6,
                    "environment_nll_auc": 1.2,
                    "final_current_environment_accuracy": 0.5,
                    "final_upright_environment_accuracy": 0.4,
                },
                "ewc_fixed_pi005": {
                    "environment_accuracy_auc": 0.7,
                    "environment_nll_auc": 1.0,
                    "final_current_environment_accuracy": 0.65,
                    "final_upright_environment_accuracy": 0.75,
                },
            }
        }

    assert phase3_paired_contrast(Run()) == pytest.approx(
        {
            "ewc_minus_current_environment_accuracy_auc": 0.1,
            "ewc_minus_current_environment_nll_auc": -0.2,
            "ewc_minus_current_final_environment_accuracy": 0.15,
            "ewc_minus_current_final_upright_accuracy": 0.35,
        }
    )
