import pytest

from mnist_experiment.akaike_score_gate.config import policies, canonical_policy

from mnist_experiment.akaike_score_gate.calibration import (
    candidate_grid, balanced_score, select_candidate, preassigned_steps, out_of_fold_selection,
)


def row(env, policy, replica, nll=1., retention=0., time=1., dimension=100):
    return {"environment": env, "policy": policy, "replica_index": replica,
            "nll_auc": nll, "retention_change": retention, "retention_nll": .5 + retention,
            "online_seconds": time, "mean_active_parameters": dimension}


def test_candidates_are_coordinatewise_and_bounded():
    candidates = candidate_grid()
    assert len(candidates) == 8
    for c in candidates:
        assert sum(c[key] != candidates[0][key] for key in c) <= 1
    assert preassigned_steps(120, 6)[-1] < 119


def test_uncalibrated_alias_only_when_calibration_matches():
    for c, count in ((1., 17), (2., 18)):
        resolved = [canonical_policy(p) for p in policies(c=c)]
        assert len({p["name"] for p in resolved}) == count
        assert ("asg_uncalibrated" in {p["name"] for p in resolved}) == (c != 1)
        if c == 1:
            assert resolved[-2] == next(p for p in resolved if p["name"] == "asg_lr_lr")


def test_selection_does_not_reward_freezing_without_predictive_quality():
    scores = {"good": balanced_score([row("linear", "asg", 0, nll=.5)]),
              "frozen": balanced_score([row("linear", "asg", 0, nll=1., time=.01, dimension=1)])}
    assert select_candidate(scores)[0] == "good"


def test_practical_risk_tie_prefers_cheaper_and_retention_is_explicit():
    scores = {"expensive": balanced_score([row("linear", "asg", 0, nll=.5, time=2)]),
              "cheap": balanced_score([row("linear", "asg", 0, nll=.503, time=1)])}
    assert select_candidate(scores)[0] == "cheap"
    scores["cheap"]["admissible"] = False
    assert select_candidate(scores)[0] == "expensive"


def test_condition_weighting_and_fold_pairing():
    rows = [row("linear", "asg", 0, nll=0.)] + [row("sigmoid", "asg", i, nll=2.) for i in range(4)]
    assert balanced_score(rows)["nll_auc"] == pytest.approx(1.)
    candidates = {"a": [row("linear", "asg", i, nll=1.) for i in range(3)],
                  "b": [row("linear", "asg", i, nll=2.) for i in range(3)]}
    assert all(r["selected"] == "a" for r in out_of_fold_selection(candidates))


def test_screening_replicas_never_become_scoring_folds():
    candidates = {"a": [row("linear", "asg", i) for i in range(8)]}
    folds = out_of_fold_selection(candidates, screening_replicas=(0, 1, 2, 3))
    assert [r["held_out_replica"] for r in folds] == [4, 5, 6, 7]
    assert all(r["screening_replicas_excluded_from_scoring"] == [0, 1, 2, 3] for r in folds)
    assert out_of_fold_selection(candidates, screening_replicas=tuple(range(8))) == []
