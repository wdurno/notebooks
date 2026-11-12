import pytest

from numerical_experiment.picar_phase2.analysis import forecast, hard_gain, observed_trainability, panel_summary
from numerical_experiment.picar_phase2.config import BASE, RESCUES, evaluation_panels, execution_recipe, floor_controls, learner_config, simulator_config
from numerical_experiment.picar_phase2.campaign import audit_initial, language_unit


def evaluations(before=False, after=True):
    return [{"checkpoint": point, "difficulty": "hard", "rows": [
        {"seed": 3, "success": success, "end_reason": "success" if success else "collision",
         "layout": {"curriculum": {"direct_path_blocked": True}}}]}
        for point, success in (("initial", before), ("final", after))]


def test_recipe_freezes_scope_and_fresh_panel_pairing():
    recipe = execution_recipe()
    assert recipe["maximum_rescue_attempts"] == 2
    assert not recipe["fd_comparison"]
    panels = evaluation_panels()
    assert len(panels[0]["seeds"]) == 20 and len(panels[1]["seeds"]) == 10
    assert panels[0]["seeds"][:10] == panels[1]["seeds"]
    assert panels[1]["config"].domain_override == "B"
    assert simulator_config().curriculum_total_intervals*.7 >= 200
    assert learner_config().learning.fd_strength == 0
    assert learner_config().learning.replay_capacity == 8
    assert RESCUES[1].teacher_floor != BASE.teacher_floor


def test_blocked_layouts_remain_in_denominator():
    rows = panel_summary(evaluations())
    assert rows[0]["count"] == 1 and rows[1]["count"] == 0 and rows[2]["count"] == 1


def test_pilot_gain_is_descriptive_not_significance():
    assert hard_gain({"evaluations": evaluations()}) == 1
    assert observed_trainability([{"evaluations": evaluations()}])
    bad = evaluations()
    bad[1]["rows"][0]["seed"] = 4
    with pytest.raises(ValueError, match="same nonempty"):
        hard_gain({"evaluations": bad})


def test_forecast_labels_unmeasured_fd_cost_and_counts_two_runs_per_pair():
    pilot = {"acquired_count": 100, "timings": {"active_seconds": 320, "evaluation_seconds": 20},
             "evaluations": evaluations()}
    result = forecast([pilot], phase2_seconds=400, prior_study_seconds=1000)
    assert result["seconds_per_captured_interval"] == 3
    assert result["scope_rows"][0]["training_processing_hours"] == 8
    assert result["fd_unmeasured"] and not result["next_phase_authorized"]
    assert result["remaining_to_20h_seconds"] == 70600


def initial_fixture():
    return {"stage": "done", "distill_receipt": {"steps": 9},
        "identity": {"config": learner_config().to_dict()}, "retired": [0],
        "eligible": [{"arrival": i, "learned": True} for i in range(1, 9)],
        "learner": {"retired_count": 1}, "acquired_count": 10, "distinct_bootstrap_samples": 9}


def test_admits_only_compatible_disjoint_initialized_asset():
    state = initial_fixture()
    assert audit_initial(state, learner_config())["initial_evidence_disjoint"]
    state["eligible"][0]["arrival"] = 0
    with pytest.raises(ValueError, match="not disjoint"):
        audit_initial(state, learner_config())


def test_rejects_old_likelihood_initialization():
    state = initial_fixture()
    state["identity"]["config"]["learning"]["reward_version"] = "potential-v1"
    with pytest.raises(ValueError, match="incompatible initial likelihood"):
        audit_initial(state, learner_config())


def test_teacher_floor_is_applied_before_physical_controls():
    rows = floor_controls(.01)
    assert all(sum(row["probabilities"]) == pytest.approx(1) for row in rows)
    assert rows[2]["controls"]["drive_mps"] > 0
    assert rows[3]["controls"]["drive_mps"] < 0
    assert rows[7]["controls"]["drive_mps"] < .02
    assert rows[0]["controls"]["steering_rad"] < 0 < rows[1]["controls"]["steering_rad"]


def test_initial_action_likelihood_variance_must_match():
    state = initial_fixture()
    state["identity"]["config"]["learning"]["action_variance"] = 7.
    with pytest.raises(ValueError, match="action_variance"):
        audit_initial(state, learner_config())


def test_language_unit_commits_full_runner_before_and_after_read_only_work(monkeypatch):
    from types import SimpleNamespace
    from numerical_experiment.picar_phase2 import student
    calls = []
    runner = SimpleNamespace(learner=SimpleNamespace(state_dict=lambda: {"version": 1}),
        s={"eligible": []}, save=lambda: calls.append("save"))
    def probe(learner, eligible):
        calls.append("probe")
        return {"rows": []}
    monkeypatch.setattr(student, "language_readiness", probe)
    assert language_unit(runner)["learning_state_unchanged"]
    assert calls == ["save", "probe", "save"]
