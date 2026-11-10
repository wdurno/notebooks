"""Phase 1 restart/immutability checks with no model or simulator execution."""

import hashlib
from pathlib import Path

import pytest

from numerical_experiment.picar import phase1, phase1_run
from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.storage import RunStore


def test_completed_probe_is_validated_and_skipped_without_launch(tmp_path, monkeypatch):
    recipe = phase1.Recipe()
    identity = phase1._identity(recipe, None, "phase1_resource_probe")
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:8]
    path = tmp_path / "units" / f"probe-{recipe.key}-{digest}"
    with RunStore(path, identity) as store:
        store.checkpoint({"stage": "done"})
        store.complete({"status": "passed"})
    original = (path / "COMPLETED").read_bytes()
    monkeypatch.setattr(phase1.subprocess, "run", lambda *a, **k: pytest.fail("must skip launch"))
    result = phase1.launch_probe(tmp_path, recipe, model_path=None, offline=True, resume=True)
    assert result == {"status": "passed"}
    assert (path / "COMPLETED").read_bytes() == original


def test_stop_is_checkpointed_before_any_gpu_probe(tmp_path, monkeypatch):
    (tmp_path / "STOP").touch()
    monkeypatch.setattr(phase1_run, "scene_unit", lambda *a, **k: {"all_feasible": True})
    monkeypatch.setattr(phase1_run, "_report", lambda: None)
    monkeypatch.setattr(phase1_run, "launch_probe", lambda *a, **k: pytest.fail("must stop before CUDA"))
    with pytest.raises(InterruptedError, match="checkpointed"):
        phase1_run.campaign(tmp_path)
    assert (tmp_path / "orchestration" / "CURRENT.json").is_file()
    assert not (tmp_path / "orchestration" / "COMPLETED").exists()


def test_changed_source_has_a_new_probe_identity(tmp_path, monkeypatch):
    recipe = phase1.Recipe()
    monkeypatch.setattr(phase1, "probe_sources", lambda: {"models.py": "old"})
    first = phase1._identity(recipe, None, "phase1_resource_probe")
    monkeypatch.setattr(phase1, "probe_sources", lambda: {"models.py": "new"})
    second = phase1._identity(recipe, None, "phase1_resource_probe")
    assert first != second
    assert canonical_json(first) != canonical_json(second)
