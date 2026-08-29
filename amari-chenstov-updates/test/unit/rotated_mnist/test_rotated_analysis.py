import json
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.analysis import discover_completed_runs
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
