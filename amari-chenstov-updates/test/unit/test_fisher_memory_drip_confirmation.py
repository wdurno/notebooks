"""Confirmation lineage and protocol guards without training."""

import dataclasses
import json

import pytest

from mnist_experiment.fisher_memory import drip, drip_confirmation as confirm
from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.config import Config, Treatment, provenance
from mnist_experiment.fisher_memory.drip_runner import trajectory_id


def source_root(tmp_path, cfg, *, omit=None):
    root = tmp_path / "development"
    details = provenance(cfg)
    identity = dict(config_hash=cfg.scientific_hash, data_hash="dataset",
                    source_hash=details["source_hash"])
    ledger = drip._ledger(cfg)
    selected = .269
    baseline = Treatment(**ledger["baseline"])
    treatment = drip._treatment(cfg, selected)
    with Store(root, identity) as store:
        store.write_json(None, "provenance.json", details)
        for name, value in ((drip.LEDGER, ledger),
                            (drip.SELECTION, dict(protocol=drip.PROTOCOL, selected_strength=selected))):
            store.begin(name, dict(kind="test"))
            store.write_json(name, "result.json", value)
            store.complete(name)
        for replica in range(ledger["screen_start"], ledger["screen_start"] + ledger["screen_replicas"]):
            for stream in ("mixed", "base"):
                for item in (baseline, treatment):
                    if (stream, item.condition) == omit:
                        continue
                    unit = trajectory_id(drip.FAMILY, replica, stream, item)
                    store.begin(unit, dict(kind="test"))
                    store.write_json(unit, "summary.json", {})
                    store.complete(unit)
    return root


def test_confirmation_freezes_selected_strength_and_fixed_cohort(tmp_path):
    cfg = Config(smoke=True, device="cpu")
    root = source_root(tmp_path, cfg)
    ledger = confirm.reference_ledger(root, cfg, "dataset")
    assert ledger["replicas"] == 2
    assert ledger["family"] == "drip_confirmation"
    assert ledger["treatment"]["strength"] == .269
    assert ledger["baseline"] == drip._ledger(cfg)["baseline"]
    assert ledger["source"]["selection_seal"]
    assert ledger == confirm.reference_ledger(root, cfg, "dataset")
    production = dataclasses.replace(cfg, smoke=False)
    other = source_root(tmp_path / "other", production)
    assert confirm.reference_ledger(other, production, "dataset")["replicas"] == 12


def test_confirmation_rejects_changed_data_configuration_and_missing_pairs(tmp_path):
    cfg = Config(smoke=True, device="cpu")
    root = source_root(tmp_path, cfg)
    with pytest.raises(ValueError, match="dataset differs"):
        confirm.reference_ledger(root, cfg, "other")
    with pytest.raises(ValueError, match="configuration differs"):
        confirm.reference_ledger(root, dataclasses.replace(cfg, seed=99), "dataset")
    missing = source_root(tmp_path / "missing", cfg, omit=("base", "finite_difference"))
    with pytest.raises(ValueError, match="comparison incomplete"):
        confirm.reference_ledger(missing, cfg, "dataset")


def test_confirmation_rejects_changed_numerical_source(tmp_path):
    cfg = Config(smoke=True, device="cpu")
    root = source_root(tmp_path, cfg)
    path = root / "provenance.json"
    value = json.loads(path.read_text())
    value["sources"]["mnist_experiment/fisher_memory/learner.py"] = "changed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="numerical sources changed"):
        confirm.reference_ledger(root, cfg, "dataset")
