"""Explicit tiny cached-MNIST JVP repair/restart pipeline, never production data."""

import dataclasses
import json

import pytest
import torch

from mnist_experiment.fisher_memory import jvp_repair
from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.config import Config, PACKAGE, ROOT, Treatment, content_hash, provenance
from mnist_experiment.fisher_memory.data import MNISTSource
from mnist_experiment.fisher_memory.jvp_repair_report import collect
from mnist_experiment.fisher_memory.run import dataset_manifest
from mnist_experiment.fisher_memory.runner import Runner


def test_cached_mnist_jvp_repair_pause_then_resume(tmp_path, monkeypatch):
    data_root = ROOT / "cache/mnist_experiment/datasets"
    try:
        source = MNISTSource(data_root)
    except RuntimeError as exc:
        if "Dataset not found" in str(exc):
            pytest.skip("cached MNIST is required")
        raise
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        cfg = dataclasses.replace(Config.read(PACKAGE / "configs/smoke.json"),
                                  smoke=False, device="cpu", dtype="float64", threads=1,
                                  environments=("rotation_sigmoid",), rotation_steps=15,
                                  free_reserve_bytes=0,
                                  max_artifact_bytes=500_000_000)
        cfg.validate()
        config_path = tmp_path / "mini.json"
        config_path.write_text(json.dumps(cfg.mapping()))
        budget = tmp_path / "plan15"
        old_root, new_root = budget / "source", budget / "jvp_repair"
        details = provenance(cfg)
        identity = dict(schema_version=1, config_hash=cfg.scientific_hash,
                        source_hash=details["source_hash"],
                        data_hash=content_hash(dataset_manifest(data_root)))
        baseline = Treatment.default(cfg, "replay_ewc")
        with Store(old_root, identity, max_bytes=cfg.max_artifact_bytes,
                   free_reserve_bytes=0) as store:
            store.write_json(None, "provenance.json", details)
            Runner(cfg, store, source).run_trajectory("confirmation", "phase3", "rotation_sigmoid", 0, baseline)
            store.begin("scheduler-selection-mini", {"name": "selection", "environment": "rotation_sigmoid"})
            store.write_json("scheduler-selection-mini", "result.json", {"result": {
                "shortlist": {"finite_difference": Treatment.default(
                    cfg, "finite_difference", .1).mapping()}}})
            store.complete("scheduler-selection-mini")
        monkeypatch.setattr(jvp_repair, "GRID_FACTORS", (1 / 3, 1.))
        args = ["--config", str(config_path), "--source-root", str(old_root),
                "--root", str(new_root), "--budget-root", str(budget),
                "--data-root", str(data_root), "--notebook", str(tmp_path / "results.ipynb"),
                "--pilot-replicas", "2", "--confirm-replicas", "2", "--allocation-hours", "1"]
        assert jvp_repair.main(args + ["--max-units", "1"]) == "paused"
        assert collect(new_root)["status"]["status"] == "paused"
        assert jvp_repair.main(args + ["--resume"]) == "completed"
        result = collect(new_root)
        assert result["status"]["status"] == "completed"
        assert result["selection"] is not None
        assert len(result["summaries"]) in (10, 12)
        assert (tmp_path / "results.ipynb").exists()
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
