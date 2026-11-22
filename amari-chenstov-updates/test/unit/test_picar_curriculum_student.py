"""Tiny deterministic control-clock acceptance for the versioned repair."""

import copy
from dataclasses import asdict, replace

import pytest
import torch

from numerical_experiment.picar.learner import MemoryLearner
from numerical_experiment.picar.models import build_tiny_model
from numerical_experiment.picar_reward_repair.config import RewardLearningConfig, configuration
from numerical_experiment.picar_curriculum_repair import student as module
from test_picar_fix3_student import setup  # noqa: F401
from test_picar_phase2_student import runtime, assert_equal  # noqa: F401
from test_picar_phase3_student import MemoryStore
from test_picar_reward_student import FakeWidthScorer
from test_picar_solver_learner import accepted_equal


def runner(store, runtime, monkeypatch, *, strength=0., stopped=lambda: None):
    import numerical_experiment.picar.fix3_student as historical
    monkeypatch.setattr(module, "SimulatorWorker", historical.SimulatorWorker)
    monkeypatch.setattr(module, "ControlSession", historical.ControlSession)
    base, sim = runtime
    model = build_tiny_model(31, hidden_size=4, width=4, return_bins=161, context_length=2).eval()
    cfg = replace(base, image_width=sim.width, image_height=sim.height,
        model=replace(base.model, return_bins=161), learning=RewardLearningConfig(
        **{**asdict(configuration().learning), "replay_capacity": 2, "fd_strength": strength}))
    learner = MemoryLearner(model, cfg.learning, cfg.seeds(0))
    state = learner.state_dict()
    state["diagonal"] = {n: v+1 for n, v in state["diagonal"].items()}
    state["retired_count"] = 2
    initial = {"stage": "done", "learner": state, "eligible": [], "retired": [0, 1],
        "acquired_count": 2, "distill_receipt": {"stage": "test_recorded_KL"}, "distinct_bootstrap_samples": 2}
    return module.CurriculumRun(model, cfg, sim, store, stopped,
        scorer=FakeWidthScorer(), seed=198101, max_transitions=19, initial=initial,
        initial_asset_id="test_asset", checkpoints=(0, 5, 11, 19), evaluations=[
            {"difficulty": d, "seeds": [9], "config": sim} for d in ("A", "B_initial", "B_final")])


@pytest.mark.parametrize("strength", (0., .3))
def test_complete_episode_overshoot_predictions_and_fd(tmp_path, runtime, monkeypatch, strength):
    store = MemoryStore(tmp_path/"run")
    live = runner(store, runtime, monkeypatch, strength=strength)
    result = live.run()
    assert result["acquired_count"] == 20
    assert result["overshoot"] == 1
    assert result["evaluated_targets"] == [0, 5, 11, 19]
    assert all(e["complete"] and not e["truncated"] for e in result["episodes"])
    assert all(m["optimizer_steps"] == 3 for m in result["fit_metrics"])
    assert result["first_step_ewc_events"] == len(result["fit_metrics"])
    assert (result["fd_evaluations"] > 0) == bool(strength)
    for panel in result["evaluations"]:
        for episode in panel["rows"]:
            assert episode["learning"] is episode["explore"] is False
            assert episode["predictive_scoring"] and episode["reward_scoring"]
            assert len(episode["predictive_scores"]) > 0
            for row in episode["predictive_scores"]:
                assert row["joint_nll"] == pytest.approx(row["action_nll"]+row["return_nll"])
    assert len(store.state["retired"]) == len(set(store.state["retired"]))
    assert not set(store.state["retired"]) & {r["arrival"] for r in store.state["eligible"]}


@pytest.mark.parametrize("where", ("training", "labels", "reference", "evaluation_score", "drain"))
def test_resume_at_new_boundaries(tmp_path, runtime, monkeypatch, where):
    baseline, actual = MemoryStore(tmp_path/"base"), MemoryStore(tmp_path/"actual")
    runner(baseline, runtime, monkeypatch, strength=.2).run()
    def halt():
        s = actual.state
        hit = where == "training" and s["acquired_count"] == 7
        hit |= where == "labels" and bool(s["current"] and s["current"].get("label_complete") and s["ready"])
        hit |= where == "drain" and bool(s.get("drain_retire_pending"))
        hit |= where == "reference" and s["stage"] == "checkpoint_evaluation" and bool(s["reference_scores"])
        hit |= where == "evaluation_score" and any(any(
            any(v is not None for v in e.get("visual_scores", [])) and not e.get("predictive_scoring")
            for e in p["rows"]) for p in s["evaluations"])
        if hit:
            raise InterruptedError(where)
    with pytest.raises(InterruptedError, match=where):
        runner(actual, runtime, monkeypatch, strength=.2, stopped=halt).run()
    runner(actual, runtime, monkeypatch, strength=.2).run()
    accepted_equal(baseline.state["learner"], actual.state["learner"])
    for key in ("eligible", "retired", "ready", "replay_rng", "episodes", "evaluations", "evaluated_targets"):
        assert_equal(baseline.state[key], actual.state[key])


def test_reference_collection_never_trains_or_changes_archive(tmp_path, runtime, monkeypatch):
    store = MemoryStore(tmp_path/"reference")
    live = runner(store, runtime, monkeypatch)
    live.reference_mode = True
    before = copy.deepcopy(live.learner.state_dict())
    summary = live.collect_references("A", target_histories=3, min_episodes=2, interval_cap=12)
    assert summary["histories"] > 0 and summary["episodes"] == 2
    assert live.s["acquired_count"] == 0
    assert live.s["eligible"] == live.s["ready"] == []
    after = live.learner.state_dict()
    # Exploration RNG advances by design; parameters/memory/optimizer do not.
    for key in ("parameters", "diagonal", "anchor", "optimizer", "center", "previous_center"):
        assert_equal(before[key], after[key])
    assert len(live.s["reference_records"]) == summary["histories"]
