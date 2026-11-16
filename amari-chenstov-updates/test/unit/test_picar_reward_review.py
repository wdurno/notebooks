"""Reward summaries are artifact-only and preserve terminal/replica semantics."""

import copy

import pytest

from numerical_experiment import picar_reward_review as review
from numerical_experiment.picar import report as shared
from test_picar_solver_report import pilot


def test_cumulative_rewards_hold_terminal_instead_of_dropping_finished_episodes():
    assert review.cumulative_curve([1., 1.], [-.025, -5.025], [0., .5, 1., 2., 10.]) == pytest.approx(
        [0., 0., -.025, -5.05, -5.05])


def test_replica_weighting_is_not_episode_pooling():
    rows = [{"seed": 1, "curve": [0., 2.]}, {"seed": 1, "curve": [0., 4.]},
            {"seed": 2, "curve": [0., 9.]}]
    assert review._replica_mean(rows, "curve") == [0., 6.]
    assert review._replica_mean([], "curve") is None


def snapshot(tmp_path):
    path = pilot(tmp_path, "replay_ewc")
    unit = shared.read_completed_unit(path)
    scores = [{"visible": True, "proximity": 1., "centering_bonus": .5, "score": 1.5}] * 2
    episode = {"label_complete": True, "complete": True, "truncated": False,
        "reset_exposure": 0., "end_reason": "collision",
        "layout": {"curriculum": {"direct_path_blocked": True}},
        "intervals": [{"elapsed_s": 1.}, {"elapsed_s": 1.}],
        "visual_scores": scores, "terminal_bonuses": [0., -5.], "stage_rewards": [-.025, -5.025]}
    unit["summary"]["episodes"] = [episode]
    return {"root": str(path.parents[1]), "complete_units": [unit], "live": []}


def test_reward_components_exclude_terminal_from_dense_rate_and_do_not_use_test_rows(tmp_path):
    s = snapshot(tmp_path)
    rows = review.training_rows(s)
    assert len(rows) == 1 and rows[0]["blocked"] is True and rows[0]["third"] == 0
    assert rows[0]["curve"][-1] == pytest.approx(-5.05)
    values = review._binned_components(rows, "replay_ewc")
    assert values[0]["proximity"] == 1.
    assert values[0]["centering"] == .5
    assert values[0]["dense_rate"] == pytest.approx(-.025)
    assert values[-1]["dense_rate"] is None
    assert review.summary_rows(rows)[0]["Visible-ball proximity = 1/10"] == "100.0%"
    s["complete_units"][0]["summary"]["episodes"] = []
    assert review.training_rows(s) == []
    assert "test rewards were not recorded" in review.outputs(s)[0]["text"]


@pytest.mark.parametrize("field", ["stage_rewards", "visual_scores", "terminal_bonuses"])
def test_incomplete_or_inconsistent_reward_components_fail_clearly(tmp_path, field):
    s = snapshot(tmp_path)
    s["complete_units"][0]["summary"]["episodes"][0][field] = []
    with pytest.raises(shared.ArtifactError, match="incomplete"):
        review.training_rows(s)


def test_incompatible_reward_scale_is_not_silently_reconstructed(tmp_path):
    s = snapshot(tmp_path)
    s["complete_units"][0]["summary"]["episodes"][0]["stage_rewards"][0] = 0.
    with pytest.raises(shared.ArtifactError, match="scale"):
        review.training_rows(s)


def test_plot_grid_keeps_empty_groups_empty_and_two_method_styles(tmp_path, monkeypatch):
    s = snapshot(tmp_path)
    rows = review.training_rows(s)
    fd = copy.deepcopy(rows[0]); fd.update(condition="stm_fd", seed=2)
    captured = []
    monkeypatch.setattr(review.figures, "_figure", lambda f, d: captured.append((f, d)))
    review._reward_figure(rows + [fd])
    figure = captured[0][0]
    assert len(figure.axes) == 6
    assert sum(len(a.lines) for a in figure.axes) == 2
    assert [l.get_linestyle() for l in figure.axes[3].lines] == ["-", "--"]
    assert not figure.axes[0].lines


def test_scored_training_plot_outputs_and_scope_are_explicit(tmp_path):
    result = review.outputs(snapshot(tmp_path))
    assert sum("image/png" in o.get("data", {}) for o in result) == 2
    assert "not A/B" in review.INTRO
    assert "not initial/mid/final" in review.INTRO
    assert "not one-step Bellman/TD" in result[-1]["text"]
