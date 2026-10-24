"""Explicit tiny cached-MNIST extension/restart pipeline, never production data."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.config import Config, PACKAGE, ROOT, Treatment, content_hash, provenance
from mnist_experiment.fisher_memory.data import MNISTSource
from mnist_experiment.fisher_memory.extension import main
from mnist_experiment.fisher_memory.extension_report import collect
from mnist_experiment.fisher_memory.run import dataset_manifest
from mnist_experiment.fisher_memory.runner import Runner


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def seal(store, unit, metadata, result):
    store.begin(unit, metadata)
    store.write_json(unit, "result.json", {"result": result})
    store.complete(unit)


def test_cached_mnist_extension_job_then_resume(tmp_path):
    data_root = ROOT / "cache/mnist_experiment/datasets"
    try:
        source = MNISTSource(data_root)
    except RuntimeError as exc:
        if "Dataset not found" in str(exc):
            pytest.skip("cached MNIST is required")
        raise
    cfg = dataclasses.replace(Config.read(PACKAGE / "configs/smoke.json"),
                              smoke=False, device="cpu", dtype="float64", threads=1,
                              environments=("rotation_linear",), free_reserve_bytes=0,
                              max_artifact_bytes=500_000_000)
    cfg.validate()
    config_path = tmp_path / "mini.json"
    config_path.write_text(json.dumps(cfg.mapping()))
    budget = tmp_path / "plan15"
    old_root, new_root = budget / "source", budget / "extension"
    details = provenance(cfg)
    manifest = dataset_manifest(data_root)
    identity = dict(schema_version=1, config_hash=cfg.scientific_hash,
                    source_hash=details["source_hash"], data_hash=content_hash(manifest))
    baseline = Treatment.default(cfg, "replay_ewc")
    with Store(old_root, identity, max_bytes=cfg.max_artifact_bytes,
               free_reserve_bytes=0) as store:
        store.write_json(None, "provenance.json", details)
        Runner(cfg, store, source).run_trajectory("confirmation", "phase3", "rotation_linear", 0, baseline)
        seal(store, "scheduler-confirmation-ledger-mini", {"name": "confirmation-ledger"},
             dict(replicas=1, evaluation_protocol_hash="mini-evaluation"))
        seal(store, "scheduler-selection-mini", {"name": "selection", "environment": "rotation_linear"},
             dict(shortlist={name: Treatment.default(cfg, name, .01).mapping()
                             for name in ("jvp", "directional")}))
    args = ["--config", str(config_path), "--source-root", str(old_root), "--root", str(new_root),
            "--budget-root", str(budget), "--data-root", str(data_root),
            "--notebook", str(tmp_path / "results.ipynb"), "--methods", "jvp", "directional",
            "--environments", "rotation_linear", "--replicas", "1", "--allocation-hours", "1",
            "--max-units", "1"]
    main(args)
    assert collect(new_root)["status"]["status"] == "paused"
    assert len(collect(new_root)["observed"]) == 1
    main(args + ["--resume"])
    result = collect(new_root)
    assert result["status"]["status"] == "completed"
    assert len(result["observed"]) == 2
    assert (tmp_path / "results.ipynb").exists()
