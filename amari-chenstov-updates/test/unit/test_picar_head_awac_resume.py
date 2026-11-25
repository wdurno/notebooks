import json
import copy

import pytest
import torch

from numerical_experiment.picar_head_resume import recorded_options
from numerical_experiment.picar_head_awac.continuation import (
    Continuation, remaining_training_intervals, validate_identity,
)
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar.runtime import sha256_file


def test_recorded_paths_and_budget_restored_without_changing_scientific_options(tmp_path):
    root=tmp_path/"run"
    (root/"orchestration").mkdir(parents=True)
    config={"numerical_revision":"head-awac-v2","source":"cache/source",
            "model_path":"cache/model","reuse_profile":"cache/profile"}
    (root/"orchestration/config.json").write_text(json.dumps(config))
    options=recorded_options(root)
    assert options["resume"] and options["offline"] and options["max_hours"]==13.
    (root/"runtime_budget.json").write_text(json.dumps({"max_hours":16.}))
    later=recorded_options(root)
    assert later=={**options,"max_hours":16.}
    assert recorded_options(root,max_hours=18.)=={**options,"max_hours":18.}
    assert json.loads((root/"orchestration/config.json").read_text())==config


def _identity(revision="head-awac-v2"):
    return {"numerical_revision": revision, "protocol": {"intervals": 1024},
            "sources": {"learner.py": "unchanged", "trajectory.py": "unchanged",
                        "numerical_experiment/picar_head_awac/campaign.py": revision}}


def test_continuation_rejects_changed_math_data_or_settings():
    original = _identity()
    revised = _identity("head-awac-v3")
    revised["continuation"] = {"source": "original"}
    validate_identity(original, revised)
    for name in ("learner.py", "trajectory.py"):
        bad = copy.deepcopy(revised)
        bad["sources"][name] = "different"
        with pytest.raises(ValueError, match="numerical sources"):
            validate_identity(original, bad)
    bad = copy.deepcopy(revised)
    bad["sources"]["unexpected.py"] = "new"
    with pytest.raises(ValueError, match="numerical sources"):
        validate_identity(original, bad)
    bad = copy.deepcopy(revised)
    bad["protocol"]["intervals"] = 2048
    with pytest.raises(ValueError, match="experimental identity"):
        validate_identity(original, bad)


def test_versioned_continuation_preserves_exact_learner_and_old_artifacts(tmp_path):
    from test_picar_head_awac_learner import _learner, _records, _assert_exact, _without_timing

    root, new = tmp_path/"old", tmp_path/"new"
    identity, revised = _identity(), _identity("head-awac-v3")
    settings = {"method": "td_fd", "seed": 1919, "total": 1024}
    learner = _learner(steps=3)
    learner.fit(_records(), acquired=6)
    learner.retire(_records()[0])
    state = {"learner": learner.state_dict(), "n": 6, "stage": "fit",
             "replay_rng": torch.Generator().manual_seed(82).get_state(),
             "session": {"pending": [2, 3]}, "ready": ["record-id"], "round": 2}
    old_unit = root/"units/path"
    old_config = {**identity, "operation": "trajectory", "settings": settings}
    with RunStore(old_unit, old_config, checkpoint_policy="journal-v1") as store:
        import numpy as np
        ref = store.frames.put(np.zeros((4, 4, 3), dtype=np.uint8))
        state["frame_ref"] = ref
        store.checkpoint(state, progress={"n": 6})
    with RunStore(root/"orchestration", identity, checkpoint_policy="journal-v1") as parent:
        parent.checkpoint({"charged_seconds": 100., "completed": [],
                           "unit_charges": {str(old_unit): 30.}})
    original_digest = sha256_file(old_unit/"CURRENT.json")
    with RunStore(new, {**revised, "operation": "trajectory", "settings": settings},
                  checkpoint_policy="journal-v1") as target:
        continuation = Continuation(root, revised)
        try:
            restored, prior = continuation.restore(target, "trajectory", settings)
            assert restored is None and prior == 30.
            _assert_exact(target.load_checkpoint(trusted=True), state)
            assert target.frames.get(ref).shape == (4, 4, 3)
            inherited = continuation.initial_state()
            assert inherited["charged_seconds"] == 100.
            assert inherited["completed"] == []
            inherited["charged_seconds"] = 200.
            assert continuation.state["charged_seconds"] == 100.
            # An already-committed continuation must never roll back to its source.
            resumed = _learner(steps=3)
            resumed.load_state_dict(target.load_checkpoint(trusted=True)["learner"])
            expected = _without_timing(learner.fit(_records(), acquired=7))
            actual = _without_timing(resumed.fit(_records(), acquired=7))
            _assert_exact(actual, expected)
            _assert_exact(resumed.state_dict(), learner.state_dict())
            advanced = {**state, "n": 7, "learner": resumed.state_dict()}
            target.checkpoint(advanced)
            continuation.restore(target, "trajectory", settings)
            _assert_exact(target.load_checkpoint(trusted=True), advanced)
        finally:
            continuation.close()
    assert sha256_file(old_unit/"CURRENT.json") == original_digest


def test_completed_unit_import_and_matching_uses_full_settings(tmp_path):
    root = tmp_path/"old"
    identity, revised = _identity(), _identity("head-awac-v3")
    source = root/"units/final"
    config = {**identity, "operation": "trajectory", "settings": {"rank": 0}}
    with RunStore(source, config, checkpoint_policy="journal-v1") as old:
        old.checkpoint({"stage": "done", "n": 1024}, progress={"n": 1024})
        old.complete({"operation": "trajectory", "settings": {"rank": 0},
                      "active_seconds": 23., "unit_path": str(source)})
    with RunStore(root/"orchestration", identity, checkpoint_policy="journal-v1") as parent:
        parent.checkpoint({"charged_seconds": 105., "completed": [str(source)]})
    sealed = sha256_file(source/"COMPLETED")
    continuation = Continuation(root, revised)
    try:
        with RunStore(tmp_path/"target", revised, checkpoint_policy="journal-v1") as target:
            assert continuation.restore(target, "trajectory", {"rank": 10}) == (None, 0.)
            summary, prior = continuation.restore(target, "trajectory", {"rank": 0})
            assert prior == 0. and summary["active_seconds"] == 23.
            assert summary["continuation"]["source_completion_sha256"] == sealed
            assert target.load_checkpoint(trusted=True)["n"] == 1024
            target.complete(summary)
    finally:
        continuation.close()
    assert sha256_file(source/"COMPLETED") == sealed


def test_remaining_work_deducts_sealed_and_partial_paths():
    rows = [{"operation": "trajectory", "settings": {"phase": "development", "total": 1024}},
            {"operation": "trajectory", "settings": {"phase": "resource_development", "total": 128}},
            {"operation": "trajectory", "settings": {"phase": "resource_pilot", "total": 128}}]
    assert remaining_training_intervals(rows, {"stage": "fit", "n": 936, "total": 1024}) == (
        11*1024 + 4*128 - 1024 - 128 - 936)


def test_launcher_restores_lineage_and_inherited_budget(tmp_path):
    old, new = tmp_path/"v2", tmp_path/"v3"
    (old/"orchestration").mkdir(parents=True)
    config = {"numerical_revision": "head-awac-v2", "source": "cache/source",
              "model_path": "cache/model", "reuse_profile": "cache/profile"}
    (old/"orchestration/config.json").write_text(json.dumps(config))
    (old/"runtime_budget.json").write_text(json.dumps({"max_hours": 16.}))
    options = recorded_options(new, continue_from=old)
    assert options["output"] == new and options["continue_from"] == old
    assert options["max_hours"] == 16.
    (new/"orchestration").mkdir(parents=True)
    config.update(numerical_revision="head-awac-v3", continuation={"source": str(old)})
    (new/"orchestration/config.json").write_text(json.dumps(config))
    (new/"runtime_budget.json").write_text(json.dumps({"max_hours": 16.}))
    assert recorded_options(new) == options
