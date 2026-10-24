"""Cached-MNIST confirmation smoke with exact interrupted/resumed equality."""

import dataclasses
import json

import nbformat
import pytest
import torch

from mnist_experiment.fisher_memory import drip, drip_confirmation as confirm
from mnist_experiment.fisher_memory.artifacts import Store, iter_units
from mnist_experiment.fisher_memory.config import PACKAGE, ROOT, Config
from mnist_experiment.fisher_memory.data import MNISTSource


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_confirmation_pause_resume_cached_mnist(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    data_root = ROOT / "cache/mnist_experiment/datasets"
    try:
        MNISTSource(data_root)
    except RuntimeError as exc:
        if "Dataset not found" in str(exc):
            pytest.skip("cached MNIST required")
        raise
    notebook = tmp_path / "results.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Results")]), notebook)
    budget = tmp_path / "budget"
    development = budget / "development"
    config = dataclasses.replace(Config.read(PACKAGE / "configs/pipeline_smoke.json"),
                                 device=device, dtype="float32" if device == "cuda" else "float64")
    config_path = tmp_path / "smoke.json"
    config_path.write_text(json.dumps(config.mapping()))
    common = ["--config", str(config_path),
              "--budget-root", str(budget), "--data-root", str(data_root),
              "--notebook", str(notebook), "--allocation-hours", "1"]
    assert drip.main(common + ["--root", str(development)]) == "completed"
    root = budget / "confirmation"
    arguments = common + ["--source-root", str(development), "--root", str(root)]
    assert confirm.main(arguments + ["--smoke-stop-after-step", "1"]) == "paused"
    assert json.loads((root / "status.json").read_text())["status"] == "paused"
    assert confirm.main(arguments + ["--resume"]) == "completed"
    records = list(iter_units(root, full=True))
    trajectories = [r for r in records if r["metadata"].get("kind") == "drip_trajectory"]
    assert len(trajectories) == 8
    assert all(r["status"] == "completed" and r["metadata"]["phase"] == "5C" for r in trajectories)
    result = json.loads((root / "units" / confirm.RESULTS / "result.json").read_text())
    assert result["complete"]
    sealed = {r["unit_id"]: r["completion"]["seal_hash"] for r in records}
    assert confirm.main(arguments + ["--resume"]) == "completed"
    assert sealed == {r["unit_id"]: r["completion"]["seal_hash"] for r in iter_units(root)}
    fresh = budget / "uninterrupted"
    assert confirm.main(common + ["--source-root", str(development), "--root", str(fresh)]) == "completed"
    for record in trajectories:
        unit = record["unit_id"]
        def checkpoint(run):
            identity = json.loads((run / "identity.json").read_text())["identity"]
            with Store(run, identity, resume=True) as store:
                return store.load_checkpoint(unit)
        resumed = checkpoint(root)
        uninterrupted = checkpoint(fresh)
        assert torch.equal(resumed["learner"]["theta"], uninterrupted["learner"]["theta"])
        for a, b in zip(resumed["rows"], uninterrupted["rows"], strict=True):
            assert {k: v for k, v in a.items() if k != "cumulative_active_seconds"} == {
                k: v for k, v in b.items() if k != "cumulative_active_seconds"}
