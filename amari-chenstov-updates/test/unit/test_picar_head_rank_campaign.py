"""Source integrity, bounded selection and passive action instrumentation."""

import copy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_head_rank.campaign import choose_correction, endpoint
from numerical_experiment.picar_head_rank.reuse import receipt, load_state
from numerical_experiment.picar_head_rank.trajectory import RankTrajectory
from src.robotics.actions import Calibration


def source(tmp_path):
    root = tmp_path / "source"
    with RunStore(root, {"test": 1}, checkpoint_policy="journal-v1") as store:
        store.checkpoint({"vector": torch.arange(3.), "counter": 7})
        store.complete({"status": "completed"})
    return root


def test_read_only_source_loading_preserves_every_file(tmp_path):
    root = source(tmp_path)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    identity = receipt(root)
    state = load_state(root)
    torch.testing.assert_close(state["vector"], torch.arange(3.))
    assert state["counter"] == 7 and identity["unit"] == str(root)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", ["config.json", "summary.json", "manifest.json", "CURRENT.json", "state.pt"])
def test_source_corruption_is_rejected_before_use(tmp_path, name):
    root = source(tmp_path)
    if name == "state.pt":
        pointer = json.loads((root / "CURRENT.json").read_text())
        path = root / "transactions" / pointer["commit_id"] / name
    else:
        path = root / name
    if name in ("config.json", "CURRENT.json"):
        value = json.loads(path.read_text())
        value["test" if name == "config.json" else "commit_id"] = "changed"
        path.write_text(json.dumps(value))
    else:
        value = path.read_bytes()
        path.write_bytes(value + b" ")
    with pytest.raises((ValueError, KeyError)):
        load_state(root)


def result(method, retention, acquisition=-10):
    return {"settings": {"method": method}, "evaluations": [
        {"n": 1024, "environment": env, "episodes": [{"raw_return": value}] * 8}
        for env, value in (("initial_A", retention), ("moderate_A", acquisition))]}


def test_retention_selection_uses_full_development_panels_and_targeted_direction():
    assert choose_correction([result("td_replay", 5), result("td_fd", -5)], 0) is None
    selected = choose_correction([result("td_replay", 6), result("td_fd", -6)], 0)
    assert selected["method"] == "td_fd" and selected["ridge"] == .2
    selected = choose_correction([result("td_replay", 7), result("td_fd", -6)], 0)
    assert selected["method"] == "td_replay" and selected["ridge"] == .05
    invalid = result("td_fd", 0)
    invalid["evaluations"][0]["episodes"].pop()
    with pytest.raises(ValueError, match="eight-layout"):
        endpoint(invalid, "initial_A")
    assert choose_correction([result("td_fd", 8, acquisition=-5)], 0, reference_acquisition=-10) is None


def test_action_summary_does_not_change_learner_or_consume_rng():
    runner = object.__new__(RankTrajectory)
    runner.features = SimpleNamespace(clear=lambda: None)
    runner.s = {"current": {"evaluation": False, "episode_id": "train:1:0", "n": 10,
        "simulator_config": {"calibration": asdict(Calibration())},
        "layout": {"curriculum": {"sigma_m": .75}}, "success": False,
        "rows": [{"action": None}, {"action": torch.arange(7.).float()}]},
        "session": {}, "history": {}, "episode_counter": 0}
    rng = torch.random.get_rng_state().clone()
    before = copy.deepcopy(runner.s["current"])
    runner.finish_episode()
    assert runner.s["current"] is None and runner.s["episode_counter"] == 1
    saved = runner.s["action_metrics"][before["episode_id"]]
    assert saved["expected_intervals"] == 2 and saved["complete_history"]
    assert saved["n"] == 10 and saved["sigma_m"] == .75
    torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)


def test_evaluation_actions_are_not_counted_as_training():
    runner = object.__new__(RankTrajectory)
    runner.features = SimpleNamespace(clear=lambda: None)
    runner.s = {"current": {"evaluation": True}, "session": None, "history": None, "episode_counter": 1}
    runner.finish_episode()
    assert "action_metrics" not in runner.s


@pytest.mark.parametrize("stop_commit", [2, 3, 5, 7])
def test_rank_trajectory_resume_keeps_replay_draws_and_unique_retirements(stop_commit):
    # Exercise the real inherited scheduler with the existing tiny causal histories.
    from test_picar_head_awac_trajectory import seeded_work, SchedulerLearner, Features, Pause, exact

    def rank_runner():
        value = seeded_work()
        value.__class__ = RankTrajectory
        value.learner.archive = SimpleNamespace(diagnostics=lambda: {"rank": 64, "trace": 1.})
        value.save()
        return value

    uninterrupted = rank_runner()
    uninterrupted.work()
    interrupted = rank_runner()
    interrupted.store.stop_after = interrupted.store.commits + stop_commit
    with pytest.raises(Pause):
        interrupted.work()
    interrupted.store.stop_after = None
    learner = SchedulerLearner()
    learner.archive = SimpleNamespace(diagnostics=lambda: {"rank": 64, "trace": 1.})
    resumed = RankTrajectory(learner, Features(), None, interrupted.store, lambda: None,
                             seed=91, total=8, evaluation_episodes=1, initial_episodes=1)
    resumed.work()
    exact(resumed.s, uninterrupted.s)
    exact(resumed.learner.state_dict(), uninterrupted.learner.state_dict())


@pytest.mark.parametrize("baseline_healthy", [True, False])
def test_profile_scheduler_seals_units_resumes_and_rejects_missing_baseline(tmp_path, monkeypatch, baseline_healthy):
    import sys
    from types import ModuleType
    from numerical_experiment.picar_head_rank import campaign, numerics
    from numerical_experiment.picar_head_awac.learner import Recipe
    from numerical_experiment.picar_head_rank.reuse import digest

    source_root, output = tmp_path / "source", tmp_path / "output"
    (source_root / "orchestration").mkdir(parents=True)
    (source_root / "orchestration/config.json").write_text(json.dumps({
        "sources": {}, "theory_sha256": digest(campaign.PROJECT / "mathematical_overview.ipynb")}))
    notebook = tmp_path / "report.ipynb"
    notebook.write_text("{}")
    recipe = Recipe(temperature=1., critic_variance=16.)
    units = [{"operation": op, "settings": {}, "unit_path": op} for op in ("features", "initial_fit", "selection")]
    units[-1]["selected"] = {"recipe": asdict(recipe), "fd_strength": 12.14}
    units.append({"operation": "geometry_fisher", "unit_path": "geometry", "settings": {
        "temperature": 1., "action_variance": 16., "critic_variance": 16.}})
    states = {"features": {"train": []}, "initial_fit": {}, "geometry": {"learner": {"archive": {}}}}
    original_load = campaign.load_state
    monkeypatch.setattr(campaign, "source_index", lambda root: (units, []))
    monkeypatch.setattr(campaign, "fingerprints", lambda: {})
    monkeypatch.setattr(campaign, "load_state", lambda path: states[str(path)] if str(path) in states else original_load(path))
    calls = []

    def rebuild(fitted, records, cfg, store, check, **kwargs):
        calls.append("rebuild")
        store.checkpoint({"archives": {}}, progress={"n": 515})
        return {"archives": {}, "diagnostics": {}, "cursor": 515}

    def audit(*args):
        calls.append("audit")
        ranks = [0, 32, 64] if baseline_healthy else [32, 64]
        return {"healthy_ranks": ranks, "timings": [
            {"rank": rank, "method": method, "timing_available": rank in ranks,
             **({"fit_seconds": .03, "retirement_seconds": .006} if rank in ranks else {})}
            for rank in (0, 32, 64) for method in ("td_replay", "td_fd")]}

    monkeypatch.setattr(numerics, "rebuild_initial", rebuild)
    monkeypatch.setattr(numerics, "resource_audit", audit)
    reporting = ModuleType("numerical_experiment.picar_head_rank.report")
    reporting.build_notebook = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, reporting.__name__, reporting)
    options = dict(output=output, source=source_root, notebook=notebook, stage="profile")
    result = campaign.run(**options)
    assert result["status"] == ("profile_complete" if baseline_healthy else "resource_blocked")
    markers = sorted((output / "units").glob("*/COMPLETED"))
    before = {p: p.read_bytes() for p in markers}
    repeated = campaign.run(**options, resume=True)
    assert repeated["status"] == result["status"] and calls == ["rebuild", "audit"]
    assert before == {p: p.read_bytes() for p in markers}
