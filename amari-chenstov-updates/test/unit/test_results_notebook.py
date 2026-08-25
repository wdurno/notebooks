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
DEPLOYMENT_NOTEBOOK = (
    REPO_ROOT / "mnist_experiment" / "deployment_results.ipynb"
)
PLAN4_ORACLE_NOTEBOOK = (
    REPO_ROOT / "mnist_experiment" / "plan4_oracle_results.ipynb"
)
PLAN4_FISHER_NOTEBOOK = (
    REPO_ROOT / "mnist_experiment" / "plan4_fisher_results.ipynb"
)
PLAN4_CHALLENGE_NOTEBOOK = (
    REPO_ROOT / "mnist_experiment" / "plan4_challenge_results.ipynb"
)
PLAN4_EDR_NOTEBOOK = (
    REPO_ROOT / "mnist_experiment" / "plan4_edr_results.ipynb"
)
MATHEMATICAL_OVERVIEW = REPO_ROOT / "mathematical_overview.ipynb"
README = REPO_ROOT / "README.md"
MNIST_AGENTS = REPO_ROOT / "mnist_experiment" / "AGENTS.md"


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
    assert any(
        "Plan 2 launch readiness and results" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert any(
        "Phase 4 controller recalibration" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert any(
        "Phase 5 independent confirmation" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert any(
        "instantaneous empirical" in "".join(cell.get("source", []))
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


def test_deployment_results_notebook_has_exposure_contrast() -> None:
    notebook = load_notebook(DEPLOYMENT_NOTEBOOK)

    validate_notebook_source(notebook)

    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "Learning by expected digit-9 exposure" in source
    assert "expected_nines_before_evaluation" in source
    assert "nine_ovr_accuracy" in source


def test_plan4_oracle_notebook_is_artifact_only() -> None:
    notebook = load_notebook(PLAN4_ORACLE_NOTEBOOK)

    validate_notebook_source(notebook)

    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert notebook["nbformat"] == 4
    assert "phase2__*" in source
    assert "unbounded_raw_oracle_pi" in source
    assert "bounded_state_raw_oracle_pi" in source
    assert "run_controller" not in source


def test_plan4_fisher_notebook_is_artifact_only() -> None:
    notebook = load_notebook(PLAN4_FISHER_NOTEBOOK)

    validate_notebook_source(notebook)

    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert notebook["nbformat"] == 4
    assert "fisher_analysis" in source
    assert "Ungated predictable" in source
    assert "contemporaneous_pi" in source
    assert "run_controller" not in source


def test_plan4_challenge_notebook_is_artifact_only() -> None:
    notebook = load_notebook(PLAN4_CHALLENGE_NOTEBOOK)

    validate_notebook_source(notebook)

    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert notebook["nbformat"] == 4
    assert "phase5_actuation__266e6f391abb" in source
    assert "floor_sensitivity__b022b149d154" in source
    assert "Realized Fisher-risk actuation" in source
    assert "Exploratory lower-floor sensitivity" in source
    assert "classification" not in source
    assert "run_controller" not in source


def test_plan4_edr_notebook_is_artifact_only() -> None:
    notebook = load_notebook(PLAN4_EDR_NOTEBOOK)

    validate_notebook_source(notebook)

    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert notebook["nbformat"] == 4
    assert "edr_screen__014849bf996e" in source
    assert "edr_predictive__f83d7cbf0459" in source
    assert "edr_discovery__7b14c111ed7b" in source
    assert "edr_stress__8f56773be109" in source
    assert "exponentially discounted risk control" in source
    assert "Cold-start discovery amendment" in source
    assert "Sigmoid stress diagnostics" in source
    assert "Single-trajectory prequential calibration" in source
    assert "ema_log_calibration_ratio" in source
    assert "run_controller" not in source


def test_mathematical_overview_separates_ideal_risk_from_realization() -> None:
    notebook = load_notebook(MATHEMATICAL_OVERVIEW)
    validate_notebook_source(notebook)

    main = "".join(notebook["cells"][2].get("source", []))
    interpretation = "".join(notebook["cells"][3].get("source", []))
    realization = "".join(notebook["cells"][6].get("source", []))

    assert "Euclidean parameter error and Fisher risk" in main
    assert r"\mathsf M_t=I_{\dim\Theta}" in main
    assert r"\pi_{B,t}^{\mathrm{loc}}" in main
    assert "globally optimal continual-learning policy" in main
    assert "Plug-in trend and covariance estimation" not in main
    assert "fixed composition is generally preferable" in interpretation
    assert "Appendix B: Statistical and numerical realization" in realization
    assert r"G_t:=\widehat{\mathcal I}_{t\mid t-1}" in realization
    assert "Exponentially discounted recommendation" in realization
    assert "Recommendation and closed-loop application" in realization
    assert "Single-trajectory prequential calibration" in realization
    assert "`optimal_plugin`" in realization


def test_phase7_human_documentation_states_conditional_evidence() -> None:
    readme = README.read_text(encoding="utf-8")
    agents = MNIST_AGENTS.read_text(encoding="utf-8")

    assert "Conclusion: Halting progress" not in readme
    assert "numerical_experiment" not in readme
    assert "Fixed $\\pi=.05$ is the incumbent" in readme
    assert "remains open" in readme
    assert "Local-composition recommendation experiments" in agents
    assert "Historical configuration values" in agents
    assert "short-horizon" in agents
    assert "staleness monitor" in agents


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
