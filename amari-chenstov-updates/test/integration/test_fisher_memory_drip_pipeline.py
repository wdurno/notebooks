"""Tiny cached-MNIST accepted-step interruption and resume."""

import json

import nbformat
import pytest

from mnist_experiment.fisher_memory import drip
from mnist_experiment.fisher_memory.config import PACKAGE, ROOT
from mnist_experiment.fisher_memory.data import MNISTSource


def test_drip_pause_resume_cached_mnist(tmp_path):
    data_root = ROOT / "cache/mnist_experiment/datasets"
    try:
        MNISTSource(data_root)
    except RuntimeError as exc:
        if "Dataset not found" in str(exc):
            pytest.skip("cached MNIST required")
        raise
    base = tmp_path / "results.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Existing Plan 15 results")]), base)
    budget = tmp_path / "budget"
    root = budget / "drip_smoke"
    args = ["--config", str(PACKAGE / "configs/pipeline_smoke.json"),
            "--root", str(root), "--budget-root", str(budget),
            "--data-root", str(data_root), "--notebook", str(base),
            "--allocation-hours", "1"]
    assert drip.main(args + ["--smoke-stop-after-step", "1"]) == "paused"
    assert json.loads((root / "status.json").read_text())["status"] == "paused"
    assert drip.main(args + ["--resume"]) == "completed"
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "completed"
    notebook = nbformat.read(base, as_version=4)
    assert "Phase 5B" in notebook.cells[1].source
