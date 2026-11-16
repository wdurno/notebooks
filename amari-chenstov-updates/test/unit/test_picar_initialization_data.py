"""Fast whole-episode isolation, nested data and stationary-collection checks."""

import copy
from dataclasses import replace

import pytest
import torch

from numerical_experiment.picar_initialization import config, data
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_phase3.student import DevelopmentRun
import test_picar_fix3_student as fixtures
from test_picar_fix3_student import setup  # noqa: F401
from test_picar_phase2_student import runtime  # noqa: F401
from test_picar_phase3_student import MemoryStore


def rows():
    return [{"record_id": f"e{i}/{j}", "episode_id": f"e{i}", "arrival": 2*i+j,
             "history": {"same_episode": i}, "logits": torch.zeros(7)}
            for i in range(40) for j in range(2)]


def test_split_whole_episode_stable_fixed_validation_never_leaks():
    all_rows = rows()
    train, validation = data.split_records(all_rows[:40])
    assert train and validation
    assert not {r["episode_id"] for r in train} & {r["episode_id"] for r in validation}
    fixed = [r["record_id"] for r in validation]
    later_train, later_validation = data.split_records(all_rows, fixed_validation_ids=fixed)
    assert later_validation == validation
    assert not set(fixed) & {r["record_id"] for r in later_train}
    assert set(r["record_id"] for r in train) <= set(r["record_id"] for r in later_train)
    with pytest.raises(ValueError, match="duplicate"):
        data.split_records(all_rows + [all_rows[0]])
    with pytest.raises(ValueError, match="fixed validation"):
        data.split_records(all_rows, fixed_validation_ids=[train[0]["record_id"]])


def test_prefix_preserves_complete_episodes_and_nesting():
    small = data.prefix_episodes(rows(), 7)
    large = data.prefix_episodes(rows(), 17)
    assert len(small) == 8 and len(large) == 18
    assert [r["record_id"] for r in large[:len(small)]] == [r["record_id"] for r in small]
    assert small[-1]["episode_id"] != rows()[len(small)]["episode_id"]


def test_namespace_keeps_labels_and_context_unchanged():
    original = {"records": [{"episode_id": "bootstrap:0", "arrival": 3,
        "history": {"tensor": torch.tensor([2.])}, "bin": 4}],
        "episodes": [{"episode_id": "bootstrap:0", "intervals": []}]}
    changed = data.namespace_data(original, "new", arrival_offset=100)
    assert changed["records"][0]["arrival"] == 103
    assert changed["records"][0]["record_id"] == "new/bootstrap:0/3"
    assert changed["records"][0]["history"] is original["records"][0]["history"]
    assert original["records"][0]["episode_id"] == "bootstrap:0"
    assert changed["episodes"][0]["episode_id"] == "new/bootstrap:0"


def test_collection_never_distills_fits_or_archives(tmp_path, runtime, monkeypatch):
    cfg, sim = runtime
    def forbidden(*args, **kwargs):
        pytest.fail("stationary collection entered historical initialization")
    monkeypatch.setattr(data.CollectionRun, "distill", forbidden)
    monkeypatch.setattr(data.CollectionRun, "initial_nll", forbidden)
    monkeypatch.setattr(data.CollectionRun, "drain", forbidden)
    store = MemoryStore(tmp_path / "collection")
    run = data.CollectionRun(fixtures.model(), cfg, sim, store, lambda: None,
        seed=195301, max_transitions=20)
    summary = run.run()
    assert summary["stationary_A"] and not summary["initial_pi_used"]
    assert summary["fitting_updates"] == 0 and summary["records"] > 0
    assert run.s["stage"] == "collection_done"
    assert not run.s["retired"] and not run.s["ready"]
    assert all(ep["reset_exposure"] == 0 for ep in run.s["episodes"])
    assert run.sim_config.domain_override == "A"
    assert run.sim_config.curriculum_exposure_override == 0


def test_collection_resume_does_not_acquire_twice(tmp_path, runtime):
    cfg, sim = runtime
    store = MemoryStore(tmp_path / "collection")
    first = data.CollectionRun(fixtures.model(), cfg, sim, store, lambda: None,
        seed=195301, max_transitions=20)
    first.run()
    acquired = first.s["acquired_count"]
    teacher_calls = fixtures.FakeTeacher.calls
    resumed = data.CollectionRun(fixtures.model(), cfg, sim, store, lambda: None,
        seed=195301, max_transitions=20)
    resumed.run()
    assert resumed.s["acquired_count"] == acquired
    assert fixtures.FakeTeacher.calls == teacher_calls


def test_configuration_preserves_scientific_controls_and_new_limit():
    cfg = config.configuration()
    assert cfg.version == config.VERSION and config.MAX_HOURS == 10
    assert config.protocol()["next_phase_authorized"] is False
    assert cfg.model.top_layers == 1 and cfg.model.stm_rank == 1
    assert cfg.learning.replay_capacity == 8
    assert cfg.learning.action_variance == 16


def test_completed_campaign_resume_never_imports_torch_or_changes_seals(tmp_path, monkeypatch):
    import builtins
    from numerical_experiment.picar_initialization import campaign
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "COMPLETED").write_text("test-only teacher identity")
    monkeypatch.setattr(campaign, "sources", lambda: {})
    identity = campaign.run_identity(teacher)
    output = tmp_path / "repair"
    with RunStore(output / "orchestration", identity, checkpoint_policy="journal-v1") as store:
        store.checkpoint({"charged_seconds": 1.})
        store.complete({"status": "ready_for_checkin", "next_phase_authorized": False})
    marker = (output / "orchestration/COMPLETED").read_bytes()
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("completed resume must not import Torch")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = campaign.run(output, teacher, resume=True)
    assert result["status"] == "ready_for_checkin"
    assert marker == (output / "orchestration/COMPLETED").read_bytes()


def test_ceiling_cannot_be_silently_extended(tmp_path):
    from numerical_experiment.picar_initialization.campaign import run
    with pytest.raises(ValueError, match="maximum ten-hour"):
        run(tmp_path / "no-run", tmp_path / "no-teacher", max_hours=10.1)
    assert not (tmp_path / "no-run").exists()


def test_empty_initial_replay_is_supported_by_frozen_online_runner(tmp_path, runtime):
    _, initial, _ = fixtures.boot(tmp_path / "old-fixture", runtime)
    initial = copy.deepcopy(initial)
    initial["retired"].extend(row["arrival"] for row in initial["eligible"])
    initial["eligible"] = []
    initial["learner"]["retired_count"] = len(initial["retired"])
    cfg, sim = runtime
    store = MemoryStore(tmp_path / "empty-online")
    runner = DevelopmentRun(fixtures.model(), cfg, sim, store, lambda: None,
        seed=195101, max_transitions=20, initial=initial, evaluations=[])
    assert runner.s["eligible"] == []
    summary = runner.run()
    assert summary["pending_updates"] == 0
    assert len(runner.s["retired"]) == len(set(runner.s["retired"]))
    assert not set(runner.s["retired"]) & {row["arrival"] for row in runner.s["eligible"]}


def test_repaired_fit_bulk_fisher_and_online_handoff_cpu_smoke(tmp_path, runtime, monkeypatch):
    from numerical_experiment.picar_initialization import numerics
    from numerical_experiment.picar.learner import MemoryLearner
    cfg, sim = runtime
    cfg = replace(cfg, learning=replace(cfg.learning,
        temperature=2.8375266913896, action_variance=16.))
    monkeypatch.setattr(numerics, "FIT_POLICY", {**numerics.FIT_POLICY,
        "kl_max_passes": 1, "nll_max_passes": 1, "min_passes": 1,
        "patience": 1, "diagnostic_histories": 2, "block_items": 4})
    live = fixtures.model()
    pristine = MemoryLearner(live, cfg.learning, cfg.seeds(0)).state_dict()
    collection = data.CollectionRun(live, cfg, sim, MemoryStore(tmp_path / "data"),
        lambda: None, seed=195301, max_transitions=48)
    collection.run()
    restored = data.namespace_data({"records": collection.s["eligible"],
        "episodes": collection.s["episodes"]}, "smoke")
    train, validation = data.split_records(restored["records"])
    fit_store = MemoryStore(tmp_path / "fit")
    numerics.InitializationFit(live, cfg, train, validation, fit_store, lambda: None,
        pristine=pristine, acquired_count=collection.s["acquired_count"]).run_fit()
    fisher_store = MemoryStore(tmp_path / "fisher")
    fisher = numerics.InitializationFit(live, cfg, train, validation, fisher_store,
        lambda: None, pristine=pristine, acquired_count=collection.s["acquired_count"],
        fitted_state=fit_store.state)
    fisher.run_fisher()
    initial = copy.deepcopy(fisher_store.state)
    assert initial["stage"] == "done" and initial["eligible"] == []
    assert initial["learner"]["retired_count"] == len(train)
    assert not set(initial["retired"]) & {r["arrival"] for r in validation}
    store = MemoryStore(tmp_path / "online")
    pilot = DevelopmentRun(live, cfg, sim, store, lambda: None, seed=195101,
        max_transitions=20, initial=initial, evaluations=[])
    result = pilot.run()
    assert result["pending_updates"] == 0
    assert set(initial["retired"]) <= set(pilot.s["retired"])
    assert pilot.learner.retired_count == len(pilot.s["retired"])
