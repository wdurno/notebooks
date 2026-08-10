import json
from pathlib import Path

import pytest

from mnist_experiment.validate_results_notebook import (
    load_notebook,
    validate_notebook_source,
)


REPO_ROOT = Path(__file__).parents[2]
NOTEBOOK = REPO_ROOT / "mnist_experiment" / "results.ipynb"
COUPLED_NOTEBOOK = REPO_ROOT / "mnist_experiment" / "coupled_results.ipynb"
REPRESENTATION_NOTEBOOK = (
    REPO_ROOT / "mnist_experiment" / "representation_results.ipynb"
)
CONTROLLER_NOTEBOOK = (
    REPO_ROOT / "mnist_experiment" / "controller_results.ipynb"
)


def test_results_notebook_is_valid_and_artifact_only() -> None:
    notebook = load_notebook(NOTEBOOK)

    validate_notebook_source(notebook)

    assert notebook["nbformat"] == 4
    assert any(
        "Assumption checks" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert any(
        "Phase 9 command center" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert any(
        "EXPERIMENTAL_CONDITIONS.md" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert any(
        "Paired controller evidence" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_coupled_results_notebook_is_valid_and_artifact_only() -> None:
    notebook = load_notebook(COUPLED_NOTEBOOK)

    validate_notebook_source(notebook)

    assert notebook["nbformat"] == 4
    assert any(
        "Divergent paths" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_representation_results_notebook_is_valid_and_artifact_only() -> None:
    notebook = load_notebook(REPRESENTATION_NOTEBOOK)

    validate_notebook_source(notebook)

    assert notebook["nbformat"] == 4
    assert any(
        "Fixed-trajectory rank frontier" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_controller_results_notebook_is_valid_and_artifact_only() -> None:
    notebook = load_notebook(CONTROLLER_NOTEBOOK)

    validate_notebook_source(notebook)

    assert notebook["nbformat"] == 4
    assert any(
        "Assumption checks" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_results_notebook_validator_rejects_training_imports(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.ipynb"
    path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["from mnist_experiment.run_experiment import main"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="banned operations"):
        validate_notebook_source(load_notebook(path))
