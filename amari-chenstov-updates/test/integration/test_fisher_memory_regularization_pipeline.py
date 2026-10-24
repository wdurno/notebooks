"""Cached-MNIST Phase 7 recovery, including a committed parent/pending audit."""

import dataclasses
import json
import subprocess
import sys
import time

import nbformat
import pytest
import torch

from mnist_experiment.fisher_memory import regularization as phase7
from mnist_experiment.fisher_memory.artifacts import Store, iter_units
from mnist_experiment.fisher_memory.config import Config, PACKAGE, ROOT
from mnist_experiment.fisher_memory.data import MNISTSource
from mnist_experiment.fisher_memory.drip_runner import DripRunner


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_phases_resume_pending_audit_and_preserve_scientific_results(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = ROOT / "cache/mnist_experiment/datasets"
    try:
        MNISTSource(source)
    except RuntimeError as error:
        if "Dataset not found" in str(error):
            pytest.skip("cached MNIST required")
        raise
    config = dataclasses.replace(Config.read(PACKAGE / "configs/pipeline_smoke.json"),
                                 device=device, dtype="float32" if device == "cuda" else "float64")
    config_path = tmp_path / "smoke.json"
    config_path.write_text(json.dumps(config.mapping()))
    notebook = tmp_path / "results.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Results")]), notebook)
    budget, root = tmp_path / "budget", tmp_path / "budget/development"
    confirmation = budget / "confirmation"
    common = ["--config", str(config_path), "--budget-root", str(budget),
              "--data-root", str(source), "--notebook", str(notebook), "--allocation-hours", "1"]
    args = common + ["--root", str(root), "--confirmation-root", str(confirmation)]
    assert phase7.main(args + ["--smoke-stop-before-audit"]) == "paused"
    before = list(iter_units(root))
    pending = [row for row in before if row["metadata"].get("kind") == "regularization_audit"]
    assert pending and all(row["status"] == "incomplete" for row in pending)
    assert not confirmation.exists()
    assert phase7.main(args + ["--resume"]) == "completed"
    records = list(iter_units(root, full=True))
    confirmed = list(iter_units(confirmation, full=True))
    assert all(row["status"] == "completed" for row in records + confirmed)
    sealed = {str(row["path"]): row["completion"]["seal_hash"] for row in records + confirmed}
    result = json.loads((confirmation / "units" / phase7.decision_id("results") / "result.json").read_text())
    assert result["complete"] and not result["inference_ready"]  # Smoke is never scientific inference.
    assert phase7.main(args + ["--resume"]) == "completed"
    assert sealed == {str(row["path"]): row["completion"]["seal_hash"]
                      for run in (root, confirmation) for row in iter_units(run)}
    fresh_root, fresh_confirm = budget / "fresh-development", budget / "fresh-confirmation"
    assert phase7.main(common + ["--root", str(fresh_root), "--confirmation-root", str(fresh_confirm)]) == "completed"
    for run, fresh in ((root, fresh_root), (confirmation, fresh_confirm)):
        for record in iter_units(run):
            if record["metadata"].get("kind") != "drip_trajectory":
                continue
            unit = record["unit_id"]
            def checkpoint(path):
                identity = json.loads((path / "identity.json").read_text())["identity"]
                with Store(path, identity, resume=True) as store:
                    return store.load_checkpoint(unit)
            first, second = checkpoint(run), checkpoint(fresh)
            assert torch.equal(first["learner"]["theta"], second["learner"]["theta"])
            assert first["learner"]["consolidated_ids"] == second["learner"]["consolidated_ids"]
            for a, b in zip(first["rows"], second["rows"], strict=True):
                assert {k: v for k, v in a.items() if k != "cumulative_active_seconds"} == {
                    k: v for k, v in b.items() if k != "cumulative_active_seconds"}
    final = nbformat.read(notebook, as_version=4)
    assert len([cell for cell in final.cells if "plan15_phase7" in cell.metadata.get("tags", [])]) <= 8
    assert final.cells[0].source == "# Results"


def test_sigkill_with_pending_audit_preserves_exactly_once_consolidation(tmp_path):
    source = ROOT / "cache/mnist_experiment/datasets"
    try:
        MNISTSource(source)
    except RuntimeError:
        pytest.skip("cached MNIST required")
    config = Config.read(PACKAGE / "configs/pipeline_smoke.json")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.mapping()))
    notebook = tmp_path / "results.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Results")]), notebook)
    budget, root, ready = tmp_path / "budget", tmp_path / "budget/development", tmp_path / "ready"
    args = ["--stage", "development", "--config", str(config_path), "--root", str(root),
            "--budget-root", str(budget), "--data-root", str(source), "--notebook", str(notebook),
            "--allocation-hours", "1", "--max-units", "1"]
    script = ("import sys,time; from pathlib import Path; "
              "from mnist_experiment.fisher_memory import regularization as r; "
              "from mnist_experiment.fisher_memory.regularization_audit import FrozenObjectiveAudit; "
              "FrozenObjectiveAudit._run=lambda self,unit,snapshot: "
              "(Path(sys.argv[1]).write_text(unit),time.sleep(40)); "
              "r.main(sys.argv[2:])")
    process = subprocess.Popen([sys.executable, "-c", script, str(ready), *args],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=ROOT)
    try:
        until = time.monotonic() + 30
        while not ready.exists() and process.poll() is None and time.monotonic() < until:
            time.sleep(.02)
        assert ready.exists(), "child did not reach the committed-parent audit boundary"
        process.kill()
        process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    assert phase7.main(args + ["--resume"]) == "paused"
    identity = json.loads((root / "identity.json").read_text())["identity"]
    records = [row for row in iter_units(root)
               if row["metadata"].get("kind") == "drip_trajectory" and row["status"] == "completed"]
    assert len(records) == 1
    meta, unit = records[0]["metadata"], records[0]["unit_id"]
    with Store(root, identity, resume=True) as store:
        resumed = store.load_checkpoint(unit)
    fresh = budget / "uninterrupted"
    with Store(fresh, identity) as store:
        runner = DripRunner(config, store, MNISTSource(source), phase="7", protocol=phase7.PROTOCOL)
        runner.run_trajectory(meta["family"], meta["replica"], meta["stream"], phase7.Treatment(**meta["treatment"]))
        baseline = store.load_checkpoint(unit)
    assert torch.equal(resumed["learner"]["theta"], baseline["learner"]["theta"])
    assert resumed["learner"]["consolidated_ids"] == baseline["learner"]["consolidated_ids"]
    assert len(set(resumed["learner"]["consolidated_ids"])) == len(resumed["learner"]["consolidated_ids"])
