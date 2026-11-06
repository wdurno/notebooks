"""Fast scalar-only acceptance for Plan 17 Phase 5 selection and inference."""

from copy import deepcopy
import json
import math

import numpy as np
import pytest

from mnist_experiment.kalman_pi import analysis as historical
from mnist_experiment.kalman_pi.retention import analysis


def trajectory(condition="fixed", replica=0, *, stage="development", schedule="main",
               ridge=None, old_gap_pp=0., nine_gap_pp=0., factor=1, completed=True, cost=1.):
    ridge = (.1 if condition == "fixed" else .03) if ridge is None else ridge
    rows = [dict(arrivals=n * factor, step=n, p=.005 if i < 2 else .1,
                 segment="hold" if i < 2 else "rise", expected_nines=.1 * n * factor,
                 observed_nines=n * factor // 2, nine_recall=.2 + .1 * i + nine_gap_pp / 100 * i / 1.3,
                 old_accuracy=.9 - .01 * i + old_gap_pp / 100 * i / 1.3,
                 wall_seconds=n * cost)
            for i, n in enumerate((0, 2, 10))]
    return dict(unit_id=f"{stage}-{schedule}-{condition}-{replica}-{ridge}", stage=stage,
                schedule=schedule, condition=condition, replica=replica, c=.025, ridge=ridge,
                completed=completed, metrics=rows if completed else rows[:2],
                stream_hash=f"stream-{stage}-{schedule}-{replica}",
                asset_unit=f"assets-{stage}-{replica}", initial_unit=f"initial-{stage}-{replica}",
                summary={"wall_seconds": 10 * cost, "online_wall_seconds": 9 * cost,
                         "costs": {"solver_seconds": 3 * cost, "curvature_seconds": 2 * cost,
                                   "controller_seconds": cost, "evaluation_seconds": cost,
                                   "checkpoint_seconds": cost, "calibration_seconds": cost}})


def selection():
    return dict(conditions=list(analysis.CONDITIONS),
                ridges={"main": dict(zip(analysis.CONDITIONS, (.1, .03, .3, 1.))),
                        "stationary": dict(zip(analysis.CONDITIONS, (.1, .01, 0., .1)))},
                development_replicas={"main": 4, "stationary": 2},
                main_replicas=8, stationary_replicas=4, frozen_at="2026-11-02T10:00:00Z")


def fresh_block(replica=0, *, schedule="main", selected=None, **kwargs):
    selected = selection() if selected is None else selected
    return [trajectory(c, replica, stage="fresh", schedule=schedule,
                       ridge=selected["ridges"][schedule][c], **kwargs)
            for c in analysis.CONDITIONS]


def candidate(ridge, *, gap=.1, recall=.5, seconds=6., n_pairs=4, **kwargs):
    return dict(ridge=ridge, n_pairs=n_pairs, old_gap_pp=gap, nine_gap_pp=2.,
                nine_recall=recall, old_accuracy=.8 + gap / 100,
                learning_seconds=seconds, matched=False, **kwargs)


def comparison(result, condition="ema_h16", control="fixed", schedule="main"):
    group = next(g for g in result["groups"] if g["schedule"] == schedule)
    return next(c for c in group["comparisons"] if c["condition"] == condition and c["control"] == control)


@pytest.mark.parametrize("gap,matched", [(-.5, True), (.5, True), (-.500001, False), (.500001, False)])
def test_matching_band_includes_float_conversion_boundary_only(gap, matched):
    rows = [trajectory(), trajectory("covariance", old_gap_pp=gap)]
    table = analysis.candidate_table(rows, "main", "covariance", replicas=1)
    assert table[0]["old_gap_pp"] == pytest.approx(gap)
    assert table[0]["matched"] is matched
    assert analysis.select_candidate(table)["matched"] is matched


def test_table_uses_arrival_trapezoids_pp_and_equal_replica_weights():
    rows = []
    for r, factor in enumerate((1, 10)):
        rows += [trajectory(replica=r, factor=factor),
                 trajectory("covariance", r, factor=factor, nine_gap_pp=2 * (r + 1), old_gap_pp=r + 1)]
    table = analysis.candidate_table(rows, "main", "covariance", replicas=[0, 1])
    assert len(table) == 1
    row = table[0]
    assert row["n_pairs"] == 2
    assert row["nine_gap_pp"] == pytest.approx(3.)
    assert row["old_gap_pp"] == pytest.approx(1.5)
    assert row["nine_recall"] == pytest.approx(.36)
    assert row["old_accuracy"] == pytest.approx(.902)
    assert row["learning_seconds"] == 6.
    assert row["old_gap_pp_ci95"] == pytest.approx([1., 2.])
    assert row["nine_gap_pp_ci95"] == pytest.approx([2., 4.])
    assert row["old_gap_pp"] != pytest.approx((1 + 10 * 2) / 11)
    json.dumps(table, allow_nan=False)


def test_table_requires_the_entire_intended_replica_set_and_never_promotes_prefixes():
    rows = [trajectory(replica=r) for r in range(3)]
    rows += [trajectory("covariance", r, ridge=.01) for r in range(2)]
    rows += [trajectory("covariance", 2, ridge=.01, completed=False)]
    rows += [trajectory("covariance", r, ridge=.03) for r in (0, 2)]
    assert [r["ridge"] for r in analysis.candidate_table(rows, "main", "covariance", replicas=2)] == [.01]
    assert analysis.candidate_table(rows, "main", "covariance", replicas=3) == []
    assert analysis.candidate_table(rows, "main", "covariance", replicas=[0, 2])[0]["ridge"] == .03
    assert [r["n_pairs"] for r in analysis.candidate_table(rows, "main", "covariance")] == [2, 2]
    assert analysis.candidate_table(rows, "main", "covariance", replicas=0) == []


def test_table_stages_schedules_and_policies_remain_separate_without_confirmation_leakage():
    dev = [trajectory(), trajectory("covariance", old_gap_pp=.25)]
    nuisance = [trajectory("covariance", stage="fresh", old_gap_pp=8),
                trajectory("covariance", schedule="stationary", old_gap_pp=-8),
                trajectory("ema_h16", old_gap_pp=3)]
    expected = analysis.candidate_table(dev, "main", "covariance")
    assert analysis.candidate_table(dev + nuisance, "main", "covariance") == expected
    assert analysis.select_candidate(expected) == analysis.select_candidate(
        analysis.candidate_table(dev + nuisance, "main", "covariance"))
    nuisance[0]["metrics"][1]["nine_recall"] = float("nan")
    assert analysis.candidate_table(dev + nuisance, "main", "covariance") == expected


@pytest.mark.parametrize("field,value", [("stream_hash", "other"), ("initial_unit", "other"),
                                         ("asset_unit", "other"), ("asset_unit", None)])
def test_development_rejects_false_pairing(field, value):
    rows = [trajectory(), trajectory("covariance")]
    rows[1][field] = value
    with pytest.raises(ValueError, match=field):
        analysis.candidate_table(rows, "main", "covariance", replicas=1)


def test_development_rejects_different_evaluation_grid_and_initial_model():
    rows = [trajectory(), trajectory("covariance")]
    rows[1]["metrics"][1]["arrivals"] += 1
    with pytest.raises(ValueError, match="arrivals"):
        analysis.candidate_table(rows, "main", "covariance")
    rows = [trajectory(), trajectory("covariance")]
    rows[1]["metrics"][0]["old_accuracy"] -= .01
    with pytest.raises(ValueError, match="initial accuracies"):
        analysis.candidate_table(rows, "main", "covariance")


def test_selection_orders_all_matched_tiebreakers_and_recomputes_status_without_mutation():
    table = [candidate(0., recall=.49, gap=0.), candidate(.01, gap=.3),
             candidate(.03, gap=.2, seconds=7.), candidate(.1, gap=-.2), candidate(.3, gap=.2)]
    before = deepcopy(table)
    selected = analysis.select_candidate(list(reversed(table)))
    assert selected["ridge"] == .1 and selected["matched"] is True
    assert table == before
    selected["old_gap_pp"] = 999
    assert table == before
    # An out-of-band high-recall setting never defeats a matched setting.
    assert analysis.select_candidate(table + [candidate(1., gap=.50001, recall=.9)])["ridge"] == .1


def test_unmatched_selection_uses_closest_retention_then_recall_time_ridge():
    table = [candidate(0., gap=.9, recall=.9), candidate(.01, gap=.6, recall=.4),
             candidate(.03, gap=-.6, recall=.5, seconds=7.), candidate(.1, gap=.6), candidate(.3, gap=-.6)]
    selected = analysis.select_candidate(table)
    assert selected["ridge"] == .1 and selected["matched"] is False
    assert analysis.select_candidate([candidate(0., gap=.51)], tolerance_pp=.6)["matched"] is True


def test_invalid_and_missing_timing_candidates_are_explicit():
    assert analysis.select_candidate([]) is None
    invalid = [candidate(.1, gap=float("nan")), candidate(.3, recall=float("inf")), candidate(1., n_pairs=0)]
    assert analysis.select_candidate(invalid) is None
    assert analysis.refinement_candidate(invalid) is None
    table = [candidate(.01, seconds=None), candidate(.3, seconds=8.)]
    assert analysis.select_candidate(table)["ridge"] == .3
    rows = [trajectory(), trajectory("covariance")]
    rows[1]["summary"]["costs"] = {"solver_seconds": 1.}
    assert analysis.candidate_table(rows, "main", "covariance")[0]["learning_seconds"] is None
    rows[1]["summary"]["learning_seconds"] = 5.
    assert analysis.candidate_table(rows, "main", "covariance")[0]["learning_seconds"] == 5.


def test_refinement_chooses_best_adjacent_sign_bracket_and_ties_by_lower_ridge():
    table = [candidate(0., gap=-4), candidate(.01, gap=2), candidate(.03, gap=-1), candidate(.1, gap=3)]
    assert analysis.refinement_candidate(table) == pytest.approx(math.sqrt(.01 * .03))
    tied = [candidate(.01, gap=-1), candidate(.03, gap=1), candidate(.1, gap=-1)]
    assert analysis.refinement_candidate(tied) == pytest.approx(math.sqrt(.01 * .03))
    assert analysis.refinement_candidate([candidate(0., gap=-1), candidate(.01, gap=1)]) == .005


def test_refinement_without_bracket_uses_closest_incident_neighbor_and_bounded_extrapolation():
    table = [candidate(0., gap=5), candidate(.01, gap=2), candidate(.03, gap=1), candidate(.1, gap=4)]
    assert analysis.refinement_candidate(table) == pytest.approx(math.sqrt(.01 * .03))
    assert analysis.refinement_candidate([candidate(0., gap=1), candidate(.01, gap=2)]) == .005
    assert analysis.refinement_candidate([candidate(.3, gap=-2), candidate(1., gap=-1)]) == 3.
    assert analysis.refinement_candidate([candidate(1., gap=2), candidate(2., gap=1)]) == 3.
    assert analysis.refinement_candidate([candidate(1., gap=2), candidate(3., gap=1)]) == pytest.approx(math.sqrt(3))


def test_refinement_stops_in_band_remeasures_bracket_and_never_duplicates():
    assert analysis.refinement_candidate([candidate(0., gap=-1), candidate(.01, gap=.5)]) is None
    assert analysis.refinement_candidate([candidate(.01, gap=1)]) is None
    table = [candidate(0., gap=-2), candidate(.1, gap=1)]
    first = analysis.refinement_candidate(table)
    table.append(candidate(first, gap=-.8))
    second = analysis.refinement_candidate(table)
    assert first == .05 and second == pytest.approx(math.sqrt(.05 * .1))
    assert second not in [r["ridge"] for r in table]
    # Adjacent representable floats have no distinct midpoint.
    high = float(np.nextafter(.1, math.inf))
    assert analysis.refinement_candidate([candidate(.1, gap=-1), candidate(high, gap=1)]) is None


def test_summary_selected_unequal_ridges_costs_endpoints_segments_and_snapshot_identity():
    rows = fresh_block()
    rows[2] = trajectory("ema_h16", stage="fresh", ridge=.3, old_gap_pp=-.4, nine_gap_pp=3, cost=2.)
    before, selected = deepcopy(rows), selection()
    result = analysis.summarize(rows, selected, bootstrap_draws=20)
    assert result["schema"] == 1 and result["kind"] == "plan17-retention-analysis"
    assert result["trajectories_sha256"] == historical.trajectories_hash(rows)
    assert result["selection"] == selected
    assert result["counts"] == dict(sealed_trajectories=4, complete=4, incomplete=0)
    assert len(result["groups"][0]["comparisons"]) == 5
    effect = comparison(result)
    assert effect["n_pairs"] == 1 and effect["replicas"] == [0]
    assert effect["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(3)
    assert effect["observation_mean_pp"]["old_accuracy"]["mean"] == pytest.approx(-.4)
    assert effect["endpoint_pp"]["nine_recall"]["mean"] == pytest.approx(3 * 2 / 1.3)
    assert effect["segments"]["hold"]["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(3 * .5 / 1.3)
    assert effect["costs"]["learning_seconds"]["difference"]["mean"] == 6.
    assert effect["costs"]["learning_seconds"]["ratio"]["mean"] == 2.
    assert effect["costs"]["online_wall_seconds"]["ratio"]["mean"] == 2.
    assert effect["observation_mean_pp"]["nine_recall"]["ci95"] is None
    assert rows == before and selected == selection()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("condition", analysis.CONDITIONS)
def test_summary_rejects_every_unselected_ridge_including_incomplete_prefixes(condition):
    rows = fresh_block()
    index = list(analysis.CONDITIONS).index(condition)
    rows[index]["ridge"] += .001
    rows[index]["completed"] = False
    with pytest.raises(ValueError, match="selected ridge|fixed reference"):
        analysis.summarize(rows, selection())


@pytest.mark.parametrize("change", ["stream_hash", "asset_unit", "initial_unit", "arrivals", "p",
                                    "segment", "observed_nines", "expected_nines", "step", "initial"])
def test_summary_rejects_false_pairing_even_in_partial_blocks(change):
    rows = fresh_block()[:2]
    if change in ("stream_hash", "asset_unit", "initial_unit"):
        rows[1][change] = "different"
    elif change == "initial":
        rows[1]["metrics"][0]["nine_recall"] += .01
    elif change == "segment":
        rows[1]["metrics"][1][change] = "different"
    elif change == "step":
        rows[1]["metrics"][1][change] += 1
    else:
        rows[1]["metrics"][1][change] += .001
    with pytest.raises(ValueError, match="mismatched"):
        analysis.summarize(rows, selection())


def test_summary_requires_complete_four_arm_blocks_but_preserves_pending_units():
    rows = fresh_block(0) + fresh_block(1)[:2]
    rows += fresh_block(2, schedule="stationary")[:2]
    rows += [trajectory("ema_h16", 1, stage="fresh", ridge=.3, completed=False)]
    result = analysis.summarize(rows, selection(), bootstrap_draws=0)
    main, stationary = result["groups"]
    assert main["eligible_replicas"] == [0] and stationary["eligible_replicas"] == []
    assert len(main["awaiting_full_block_units"]) == 2
    assert len(stationary["awaiting_full_block_units"]) == 2
    assert main["incomplete_units"] == [rows[-1]["unit_id"]]
    assert result["counts"] == dict(sealed_trajectories=9, complete=8, incomplete=1)
    assert all(c["n_pairs"] == 1 for c in main["comparisons"])
    assert all(c["n_pairs"] == 0 for c in stationary["comparisons"])
    assert len(comparison(result)["unpaired_control_units"]) == 1


def test_summary_shared_whole_block_draws_all_endpoints_contrasts_and_costs(monkeypatch):
    rows = []
    for replica in range(4):
        block = fresh_block(replica)
        for i, t in enumerate(block[1:], start=1):
            updated = trajectory(t["condition"], replica, stage="fresh", ridge=t["ridge"],
                                 old_gap_pp=replica * i / 10, nine_gap_pp=replica * i)
            block[i] = updated
        rows.extend(block)
    calls = []
    estimate = historical._estimate

    def tracked(values, **kwargs):
        if len(values):
            calls.append((len(values), kwargs["seed"]))
        return estimate(values, **kwargs)

    monkeypatch.setattr(historical, "_estimate", tracked)
    result = analysis.summarize(rows, selection(), bootstrap_draws=200, seed=7)
    assert len(set(calls)) == 1 and calls[0][0] == 4
    assert result == analysis.summarize(deepcopy(rows), selection(), bootstrap_draws=200, seed=7)
    shuffled = analysis.summarize(list(reversed(rows)), selection(), bootstrap_draws=200, seed=7)
    assert result["groups"] == shuffled["groups"]


def test_summary_development_map_no_fresh_leakage_no_pooling_and_frozen_map_binding():
    dev = [trajectory(replica=r) for r in range(4)]
    dev += [trajectory("covariance", r, old_gap_pp=.2) for r in range(4)]
    selected = selection()
    first = analysis.summarize(dev + fresh_block(), selected, bootstrap_draws=0)
    new_rows = fresh_block(nine_gap_pp=8., old_gap_pp=-4.)
    second = analysis.summarize(dev + new_rows, selected, bootstrap_draws=0)
    assert first["development"] == second["development"]
    assert first["development"]["stationary"]["covariance"] == []
    assert first["development"]["main"]["covariance"][0]["n_pairs"] == 4
    assert first["selection_sha256"] == second["selection_sha256"]
    changed = deepcopy(selected)
    changed["ridges"]["main"]["covariance"] = .3
    with pytest.raises(ValueError, match="selected ridge"):
        analysis.summarize(dev + fresh_block(), changed)


def test_summary_historical_common_ridge_contract_remains_unchanged():
    rows = fresh_block()
    result = historical.summarize(rows, required_conditions=analysis.CONDITIONS)
    assert all(g["eligible_replicas"] == [] for g in result["groups"])
    assert all(c["n_pairs"] == 0 for g in result["groups"] for c in g["comparisons"])
    assert analysis.summarize(rows, selection())["groups"][0]["eligible_replicas"] == [0]


def test_pending_development_is_supported_but_confirmation_needs_frozen_complete_map():
    result = analysis.summarize([], {})
    assert result["groups"] == []
    assert set(result["development"]) == set(analysis.SCHEDULES)
    for edit, match in [(lambda s: s.pop("frozen_at"), "frozen"),
                        (lambda s: s["ridges"]["main"].pop("ema_h16"), "every policy"),
                        (lambda s: s.update(conditions=["fixed", "covariance"]), "four"),
                        (lambda s: s.update(main_replicas=0), "allocation")]:
        selected = selection()
        edit(selected)
        with pytest.raises(ValueError, match=match):
            analysis.summarize(fresh_block(), selected)


@pytest.mark.parametrize("field,value,match", [("c", .05, "c=.025"), ("stage", "old_fresh", "stage"),
                                              ("condition", "ema_h8", "condition"), ("ridge", -.1, "ridge")])
def test_summary_scientific_contract_rejects_wrong_protocol_rows(field, value, match):
    rows = fresh_block()
    rows[1][field] = value
    with pytest.raises(ValueError, match=match):
        analysis.summarize(rows, selection())


def test_bootstrap_limits_and_invalid_controls_fail_clearly():
    assert analysis.summarize([], {}, bootstrap_draws=-2)["bootstrap"]["draws"] == 0
    assert analysis.summarize([], {}, bootstrap_draws=100_000)["bootstrap"]["draws"] == 20_000
    for kwargs in ({"bootstrap_draws": True}, {"seed": True}, {"seed": -1}):
        with pytest.raises(ValueError):
            analysis.summarize([], {}, **kwargs)
    with pytest.raises(ValueError, match="tolerance"):
        analysis.select_candidate([], tolerance_pp=-.1)
    with pytest.raises(ValueError, match="distinct"):
        analysis.candidate_table([], "main", "covariance", replicas=[0, 0])
