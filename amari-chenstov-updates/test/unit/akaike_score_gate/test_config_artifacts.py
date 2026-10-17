import dataclasses
import json
from pathlib import Path

import pytest

from mnist_experiment.akaike_score_gate.config import Study, policies
from mnist_experiment.akaike_score_gate.artifacts import UnitStore, RunLimit, StopExecution, run_lock


def study():
    return Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json")


def test_config_pairing_and_scope():
    config = study()
    assert len(policies()) == 18
    assert len(config.environments()) == 4
    assert len(config.environments(scarcity=True)) == 5
    assert config.seed("phase1:stream") != config.seed("phase5:stream")
    assert config.config_hash == Study(config.mapping()).config_hash
    with pytest.raises(ValueError):
        Study({**config.mapping(), "unknown": 1})


def test_sealing_resume_and_integrity(tmp_path):
    store = UnitStore(tmp_path, study())
    unit = store.unit("phase0", "test", 0)
    required = ("summary.json",)
    session = store.begin(unit, required, resume=False)
    session.write_json("summary.json", {"ok": True})
    restored = store.begin(unit, required, resume=True)
    assert restored.path == session.path
    final = store.finish(restored, required)
    assert store.begin(unit, required, resume=True) is None
    with pytest.raises(RuntimeError):
        store.begin(unit, required, resume=False)
    (final / "summary.json").write_text("{}")
    with pytest.raises(RuntimeError, match="corrupt"):
        UnitStore(tmp_path, study()).completed(unit, required)


def test_lock_and_run_limit(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="writer"):
            with run_lock(tmp_path):
                pass
    limit = RunLimit(max_units=1)
    limit.check()
    limit.completed()
    with pytest.raises(StopExecution):
        limit.check()
