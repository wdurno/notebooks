"""Label admission and evaluation-score recovery are separate from training."""

from dataclasses import asdict
from types import SimpleNamespace
import copy

import pytest
import torch

from numerical_experiment.picar_reward_repair.config import CONDITIONS, configuration, simulation, BASE_RECIPE
from numerical_experiment.picar_reward_repair.student import (
    RewardDevelopmentRun, _RewardStore, assign_rewards, validate_initial_panels,
)
from numerical_experiment.picar.learner import MemoryLearner
from numerical_experiment.picar.models import build_tiny_model
from numerical_experiment.picar_reward_repair.config import RewardLearningConfig
from dataclasses import replace
from test_picar_fix3_student import setup  # noqa: F401 (shared fake clock fixture)
from test_picar_phase2_student import runtime, assert_equal  # noqa: F401
from test_picar_phase3_student import MemoryStore
from test_picar_solver_learner import accepted_equal


def episode():
    return {"complete": True, "evaluation": True, "truncated": False,
        "intervals": [{"frame_ref": "first", "elapsed_s": .5, "terminated": False, "end_reason": None},
                      {"frame_ref": "last", "elapsed_s": .5, "terminated": True, "end_reason": "success"}]}


def test_labels_retain_raw_and_coded_counterfactuals_without_new_observations():
    value = episode()
    before = copy.deepcopy(value["intervals"])
    result = assign_rewards(value, [{"score": 2.}, {"score": 8.}], CONDITIONS[1],
                            torch.linspace(-6., 12., 162), configuration().learning)
    assert value["raw_stage_rewards"][-1] == pytest.approx(100.3)
    assert value["stage_rewards"][-1] == pytest.approx(10.03)
    assert set(value["counterfactual_rewards"]) == set(CONDITIONS)
    assert value["counterfactual_rewards"][CONDITIONS[0]]["raw_stage_rewards"][-1] == pytest.approx(10.02)
    assert value["intervals"] == before
    assert value["reward_scoring"] and len(result["bins"]) == 2
    with pytest.raises(ValueError, match="all post-action"):
        assign_rewards(episode(), [None, {"score": 8.}], CONDITIONS[0],
                       torch.linspace(-6., 12., 162), configuration().learning)


def fake_runner():
    runner = object.__new__(RewardDevelopmentRun)
    runner.config = configuration()
    runner.reward_condition = CONDITIONS[0]
    runner.learner = SimpleNamespace(edges=torch.linspace(-6., 12., 162))
    runner.store = SimpleNamespace(frames=SimpleNamespace(get=lambda ref: ref))
    row = episode()
    row.update(learning=False, explore=False, reward_scoring=False)
    runner.s = {"stage": "done", "evaluation_scoring_pending": "final",
        "evaluations": [{"checkpoint": "final", "difficulty": "hard", "rows": [row]}],
        "timings": {"scoring_seconds": 0.}, "eligible": [], "retired": [], "acquired_count": 0}
    return runner


def test_final_evaluation_scoring_resumes_without_rescoring_or_training():
    runner = fake_runner()
    calls = []
    runner.scorer = SimpleNamespace(score_rgb=lambda ref: calls.append(ref) or {"score": 3.})
    saved = []
    def stop_after_first_score():
        saved.append(copy.deepcopy(runner.s))
        raise InterruptedError("test halt after first scored frame commit")
    runner.save = stop_after_first_score
    with pytest.raises(InterruptedError):
        runner._finish_evaluation_scores()
    assert calls == ["first"]
    runner.s = saved[-1]
    runner.save = lambda: saved.append(copy.deepcopy(runner.s))
    runner._finish_evaluation_scores()
    assert calls == ["first", "last"]
    assert "evaluation_scoring_pending" not in runner.s
    assert runner.s["evaluations"][0]["rows"][0]["reward_scoring"]
    assert runner.s["eligible"] == runner.s["retired"] == []
    assert runner.s["acquired_count"] == 0


def test_initial_panel_reuse_requires_exact_asset_config_and_scored_outcomes():
    cfg = simulation(BASE_RECIPE)
    specs = [{"difficulty": "easy", "seeds": [9], "config": cfg}]
    row = {**episode(), "seed": 9, "learning": False, "explore": False, "reward_scoring": True}
    panel = {"difficulty": "easy", "checkpoint": "initial", "acquired_count": 0,
        "initial_asset_id": "sealed", "seeds": [9], "config": asdict(cfg), "rows": [row]}
    result = validate_initial_panels([panel], specs, "sealed")
    assert result == [panel] and result[0] is not panel
    with pytest.raises(ValueError):
        validate_initial_panels([panel], specs, "other")
    panel["rows"][0]["reward_scoring"] = False
    with pytest.raises(ValueError, match="scored"):
        validate_initial_panels([panel], specs, "sealed")


def test_constructor_adapter_never_changes_committed_scorer_and_maps_metric_condition():
    old = {"identity": {}, "scorer": {"width": True}}
    stored = []
    base = SimpleNamespace(load_checkpoint=lambda **kwargs: old,
        checkpoint=lambda state, **kwargs: stored.append((state, kwargs)))
    adapter = _RewardStore(base, CONDITIONS[1], {"legacy": True})
    assert adapter.load_checkpoint()["scorer"] == {"legacy": True}
    assert old["scorer"] == {"width": True}
    adapter.constructing = False
    assert adapter.load_checkpoint() is old
    adapter.checkpoint({"evaluations": []}, metrics=[{"condition": "replay_ewc"}], progress={})
    assert stored[0][1]["metrics"][0]["condition"] == CONDITIONS[1]


class FakeWidthScorer:
    provenance = {"kind": "fake-width-unit-test-only"}

    def __init__(self):
        self.last_diagnostic = {}

    def state_dict(self):
        return {"provenance": self.provenance}

    def load_state_dict(self, state):
        assert state == self.state_dict()

    def score_rgb(self, rgb):
        return {"score": 4., "width_percent": 32., "raw_text": '{"width_percent":32,"reward":4}'}

    def score(self, history, pan):
        return self.score_rgb(None)


def integrated_runner(store, runtime, *, stopped=lambda: None):
    base, sim = runtime
    live = build_tiny_model(31, hidden_size=4, width=4, return_bins=161, context_length=2).eval()
    cfg = replace(base, model=replace(base.model, return_bins=161), learning=RewardLearningConfig(
        **{**asdict(configuration().learning), "replay_capacity": 2}))
    learner = MemoryLearner(live, cfg.learning, cfg.seeds(0))
    state = learner.state_dict()
    state["diagonal"] = {n: value+1 for n, value in state["diagonal"].items()}
    state["retired_count"] = 2
    initial = {"stage": "done", "learner": state, "eligible": [], "retired": [0, 1],
        "acquired_count": 2, "distill_receipt": {"stage": "test_recorded_KL"},
        "distinct_bootstrap_samples": 2}
    return RewardDevelopmentRun(live, cfg, sim, store, stopped, condition=CONDITIONS[0],
        scorer=FakeWidthScorer(), seed=197101, max_transitions=20, initial=initial,
        initial_asset_id="test_asset", evaluations=[
            {"difficulty": domain, "seeds": [9], "config": sim} for domain in ("easy", "hard")])


def test_fake_clock_complete_runtime_uses_three_steps_and_scores_all_test_rewards(tmp_path, runtime):
    store = MemoryStore(tmp_path/"live")
    live = integrated_runner(store, runtime)
    result = live.run()
    assert result["condition"] == CONDITIONS[0]
    assert result["fd_strength"] == 0
    assert result["acquired_count"] == 16
    assert result["online_updates"] > 0 and result["sleep_updates"] > 0
    assert all(m["optimizer_steps"] == 3 for m in result["fit_metrics"])
    assert all(m["condition"] == CONDITIONS[0] for m in result["fit_metrics"])
    assert [p["checkpoint"] for p in result["evaluations"]] == [
        c for c in ("initial", "midpoint", "final") for _ in range(2)]
    for panel in result["evaluations"]:
        for row in panel["rows"]:
            assert row["reward_scoring"] and row["learning"] is row["explore"] is False
            assert len(row["raw_returns"]) == len(row["intervals"]) == 4
    assert len(live.s["retired"]) == len(set(live.s["retired"]))
    assert not set(live.s["retired"]) & {r["arrival"] for r in live.s["eligible"]}


@pytest.mark.parametrize("boundary", ("labels", "midpoint_score", "final_score"))
def test_fake_clock_resume_matches_uninterrupted_at_new_reward_boundaries(tmp_path, runtime, boundary):
    reference = MemoryStore(tmp_path/"reference")
    integrated_runner(reference, runtime).run()
    actual = MemoryStore(tmp_path/"interrupted")
    def halt():
        s = actual.state
        pending = s.get("evaluation_scoring_pending")
        hit = boundary == "labels" and bool(s["current"] and s["current"].get("label_complete") and s["ready"])
        if boundary in ("midpoint_score", "final_score") and pending == boundary.split("_")[0]:
            hit = any(p["checkpoint"] == pending and any(
                any(score is not None for score in r.get("visual_scores", [])) and not r.get("reward_scoring")
                for r in p["rows"]) for p in s["evaluations"])
        if hit:
            raise InterruptedError(boundary)
    with pytest.raises(InterruptedError, match=boundary):
        integrated_runner(actual, runtime, stopped=halt).run()
    integrated_runner(actual, runtime).run()
    for key in ("learner", "eligible", "retired", "ready", "replay_rng", "episodes", "evaluations"):
        if key == "learner":
            # Wall-time fields in cached last-event diagnostics are not state equality.
            accepted_equal(reference.state[key], actual.state[key])
        else:
            assert_equal(reference.state[key], actual.state[key])
