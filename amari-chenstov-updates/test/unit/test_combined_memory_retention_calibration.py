"""Pure scalar fixtures for Plan 18 Phase 6; no training or artifact writes."""

from copy import deepcopy
from dataclasses import replace
import itertools
import math
import random

import pytest

from mnist_experiment.combined_memory.config import ARMS, Recipe
from mnist_experiment.combined_memory.retention_match.calibration import (
    EPSILON, STRENGTH, aggregate, initial_recipes, matching, propose_ridge,
    ridge_proposal, select_tuple,
)


SAVED = dict(zip(ARMS, ((2.362342, 4.469951), (2.357051, 4.462540),
                      (2.333694, 4.460042), (2.367923, 4.431826), (2.414720, 4.451717))))


def records_for(recipe, *, havg=2., h100=4., risk=10., seconds=1., members=range(4),
                cohort="development", status="completed"):
    return [dict(unit_id=f"{cohort}-{recipe.key}-{member}", status=status,
        metadata=dict(kind="trajectory", cohort=cohort, replica=member, recipe=recipe.mapping()),
        summary=dict(Havg=havg, H100=h100, novel_auc=risk, learning_seconds=seconds))
        for member in members]


def starting_records(*, saved=False):
    return list(itertools.chain.from_iterable(records_for(recipe,
        havg=SAVED[arm][0] if saved else 2., h100=SAVED[arm][1] if saved else 4.)
        for arm, recipe in initial_recipes().items()))


def means(havg=2., h100=4.):
    return {arm: dict(Havg=havg, H100=h100) for arm in ARMS}


def test_initial_recipes_preserve_the_fixed_policy_and_are_independent():
    recipes = initial_recipes()
    assert list(recipes) == list(ARMS)
    assert [recipes[arm].ridge for arm in ARMS] == [10., 10., 10., 1., .1]
    for arm, recipe in recipes.items():
        assert isinstance(recipe, Recipe)
        assert recipe.strength == (0. if arm == "A" else STRENGTH)
        assert recipe.process_scale == (1e-5 if arm in ("D", "E") else 1e-4)
    recipes.pop("E")
    assert "E" in initial_recipes()
    assert initial_recipes(strength=.25)["B"].strength == .25
    assert initial_recipes(strength=.25)["A"].strength == 0.


def test_aggregation_is_paired_signed_and_averages_already_rooted_values():
    recipe = initial_recipes()["A"]
    rows = records_for(recipe)
    for member, row in enumerate(rows):
        row["summary"].update(Havg=member - 1., H100=member + 1.,
                              novel_auc=2. * member, learning_seconds=member + 1.)
    distractors = records_for(recipe, cohort="fresh", risk=0.)
    distractors += records_for(recipe, cohort="stationary", risk=0.)
    distractors += records_for(recipe, members=(9,), risk=0.)
    before = deepcopy(rows + distractors)
    result = aggregate(iter(rows + distractors), recipe.mapping())
    assert result["members"] == [0, 1, 2, 3] and result["n_members"] == 4
    assert result["Havg"] == .5
    assert result["H100"] == 2.5
    assert result["novel_auc"] == 3.
    assert result["learning_seconds"] == 2.5
    assert result["total_learning_seconds"] == 10.
    assert result["per_member"]["0"]["Havg"] == -1.
    assert rows + distractors == before
    keyed = {str(index): row for index, row in enumerate(rows + distractors)}
    assert aggregate(keyed, recipe, members=(3, 1, 0, 2)) == result


@pytest.mark.parametrize("status", [None, "incomplete", "failed", "running"])
def test_missing_or_noncompleted_member_is_never_replaced_by_fresh_data(status):
    recipe = initial_recipes()["B"]
    rows = records_for(recipe)
    if status is None:
        rows.pop()
    else:
        rows[-1]["status"] = status
    rows += records_for(recipe, members=(3,), cohort="fresh")
    with pytest.raises(ValueError, match="missing or incomplete.*3"):
        aggregate(rows, recipe)


def test_duplicate_members_are_not_averaged_or_resolved_by_input_order():
    recipe = initial_recipes()["B"]
    rows = records_for(recipe)
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate development member 0"):
        aggregate(rows, recipe)


@pytest.mark.parametrize("field", ["Havg", "H100", "novel_auc", "learning_seconds"])
@pytest.mark.parametrize("value", [None, "2.0", True, float("nan"), float("inf"), -float("inf"), []])
def test_missing_nonnumeric_and_nonfinite_metrics_raise(field, value):
    recipe = initial_recipes()["B"]
    rows = records_for(recipe)
    rows[2]["summary"][field] = value
    with pytest.raises(ValueError, match=field):
        aggregate(rows, recipe)
    with pytest.raises(ValueError, match=field):
        select_tuple(starting_records()[:4] + rows)


@pytest.mark.parametrize("field", ["novel_auc", "learning_seconds"])
def test_negative_risk_or_time_is_not_scientific_data(field):
    rows = starting_records()
    rows[0]["summary"][field] = -1.
    with pytest.raises(ValueError, match=field):
        select_tuple(rows)


def test_missing_summary_metric_or_recipe_fields_raise_clearly():
    recipe = initial_recipes()["B"]
    for field in ("Havg", "H100", "novel_auc", "learning_seconds"):
        rows = records_for(recipe)
        del rows[0]["summary"][field]
        with pytest.raises(ValueError, match=field):
            aggregate(rows, recipe)
    rows = records_for(recipe)
    del rows[0]["metadata"]["recipe"]["strength"]
    with pytest.raises(ValueError, match="complete recipe"):
        aggregate(rows, recipe)
    rows = records_for(recipe)
    rows[0]["summary"] = None
    with pytest.raises(ValueError, match="summary mapping"):
        aggregate(rows, recipe)


@pytest.mark.parametrize("members", [(), (0, 0, 1, 2), (True, 1, 2, 3), (0, 1, 2, -1), ("0", 1, 2, 3)])
def test_invalid_member_sets_raise(members):
    with pytest.raises(ValueError, match="development members"):
        aggregate([], initial_recipes()["A"], members)


@pytest.mark.parametrize("replica", [None, True, "0", 0., -1])
def test_invalid_record_member_is_not_silently_dropped(replica):
    rows = records_for(initial_recipes()["A"])
    rows[0]["metadata"]["replica"] = replica
    with pytest.raises(ValueError, match="development replica"):
        select_tuple(rows)


def test_saved_starting_tuple_passes_with_both_recorded_ranges():
    rows = starting_records(saved=True)
    chosen = select_tuple(rows)
    assert chosen is not None
    assert chosen["recipes"] == {arm: recipe.mapping() for arm, recipe in initial_recipes().items()}
    assert chosen["matching"]["passed"] is True
    assert chosen["matching"]["ranges"] == pytest.approx(dict(Havg=.081026, H100=.038125))
    assert chosen["mean_novel_auc"] == 10.
    assert chosen["total_learning_seconds"] == 20.
    assert chosen["diagnostics"]["feasible_tuples"] == 1
    assert all(row["members"] == [0, 1, 2, 3] for row in chosen["calibration"].values())
    assert matching(chosen["calibration"].values()) == chosen["matching"]


def test_phase5_selected_b1000_tuple_is_rejected_without_an_unmatched_fallback():
    recipes = initial_recipes()
    recipes["B"] = replace(recipes["B"], ridge=1000.)
    rows = []
    for arm, recipe in recipes.items():
        havg, h100 = SAVED[arm] if arm != "B" else (.1, .137)
        rows.extend(records_for(recipe, havg=havg, h100=h100))
    assert select_tuple(rows) is None


@pytest.mark.parametrize("field", ["Havg", "H100"])
def test_two_arms_within_point_one_of_A_still_fail_the_full_point_two_range(field):
    rows = means()
    reference = rows["A"][field]
    rows["B"][field], rows["C"][field] = reference - .1, reference + .1
    assert all(abs(row[field] - reference) <= .1 + EPSILON for row in rows.values())
    check = matching(rows)
    assert check["passed"] is False
    assert check["ranges"][field] == pytest.approx(.2)


@pytest.mark.parametrize("field", ["Havg", "H100"])
@pytest.mark.parametrize("slack,passed", [(0., True), (.5e-8, True), (1e-8, True),
                                         (1.1e-8, False), (2e-8, False)])
def test_matching_boundary_uses_no_rounded_bins(field, slack, passed):
    rows = means(havg=0., h100=0.)
    rows["E"][field] = .1 + slack
    assert matching(rows)["passed"] is passed


@pytest.mark.parametrize("field", ["Havg", "H100"])
def test_failure_of_either_statistic_alone_rejects_the_tuple(field):
    rows = starting_records()
    for row in rows[-4:]:
        row["summary"][field] += .11
    assert select_tuple(rows) is None


@pytest.mark.parametrize("tolerance", [-.1, None, "0.1", True, float("nan"), float("inf")])
def test_invalid_tolerance_is_rejected_even_with_no_candidates(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        matching(means(), tolerance)
    with pytest.raises(ValueError, match="tolerance"):
        select_tuple([], tolerance=tolerance)


def test_matching_requires_exactly_one_summary_per_arm():
    with pytest.raises(ValueError, match="all five arms"):
        matching({"A": dict(Havg=0., H100=0.)})
    row = aggregate(records_for(initial_recipes()["A"]), initial_recipes()["A"])
    with pytest.raises(ValueError, match="duplicate matching arm"):
        matching([row, row])
    rows = means()
    rows["A"]["H100"] = "4"
    with pytest.raises(ValueError, match="H100"):
        matching(rows)


def test_incomplete_candidates_and_other_cohorts_cannot_enter_selection():
    rows = starting_records()
    candidate = replace(initial_recipes()["B"], ridge=1.)
    rows += records_for(candidate, members=(0, 1, 2), risk=0.)
    rows += records_for(candidate, members=(3,), cohort="fresh", risk=0.)
    rows += records_for(candidate, members=(3,), status="failed", risk=0.)
    rows += records_for(replace(candidate, ridge=100.), cohort="stationary", risk=0.)
    chosen = select_tuple(rows)
    assert chosen["recipes"]["B"]["ridge"] == 10.
    assert chosen["diagnostics"]["incomplete_candidates"] == {candidate.key: [3]}
    assert select_tuple(rows[:19]) is None


def test_corrupt_completed_member_cannot_hide_in_an_incomplete_candidate():
    rows = starting_records()
    candidate = replace(initial_recipes()["B"], ridge=1.)
    extra = records_for(candidate, members=(0, 1, 2))
    extra[0]["summary"]["novel_auc"] = None
    with pytest.raises(ValueError, match="novel_auc"):
        select_tuple(rows + extra)


def test_selection_fixes_A_and_excludes_other_strength_process_or_out_of_bounds_ridges():
    rows = starting_records()
    candidates = [replace(initial_recipes()["A"], ridge=1.),
                  replace(initial_recipes()["B"], strength=STRENGTH * 10),
                  replace(initial_recipes()["C"], process_scale=1e-5),
                  replace(initial_recipes()["D"], process_scale=1e-4),
                  replace(initial_recipes()["E"], ridge=1001.)]
    for candidate in candidates:
        rows.extend(records_for(candidate, risk=0.))
    chosen = select_tuple(rows)
    assert chosen["recipes"] == {arm: recipe.mapping() for arm, recipe in initial_recipes().items()}
    assert chosen["diagnostics"]["complete_candidates"] == dict.fromkeys(ARMS, 1)
    fallback = select_tuple(rows, reference_ridge=1.)
    assert fallback["recipes"]["A"]["ridge"] == 1.
    assert fallback["reference_ridge"] == 1.
    assert select_tuple(rows, reference_ridge=.1) is None


def test_smoke_strength_is_explicit_and_never_mixed_with_historical_tuning():
    recipes = initial_recipes(strength=.1)
    rows = list(itertools.chain.from_iterable(records_for(recipe) for recipe in recipes.values()))
    assert select_tuple(rows) is None
    for arm in ARMS[1:]:
        historical = replace(initial_recipes()[arm], strength=.005, ridge=1.)
        rows += records_for(historical, risk=0.)
    chosen = select_tuple(rows, tolerance=10., strength=.1)
    assert chosen["recipes"] == {arm: recipe.mapping() for arm, recipe in recipes.items()}
    assert chosen["strength"] == .1
    assert chosen["matching"]["tolerance"] == 10.
    assert chosen["diagnostics"]["complete_candidates"] == dict.fromkeys(ARMS, 1)
    assert propose_ridge("B", rows, recipes["B"], dict(Havg=2., H100=4.), [],
                         tolerance=10., strength=.1) is not None


def test_joint_selection_does_not_independently_choose_unmatchable_arm_optima():
    rows = starting_records()
    for row in rows:
        if row["metadata"]["recipe"]["arm"] == "B":
            row["summary"].update(Havg=1.92, H100=3.92, novel_auc=1.)
    c = replace(initial_recipes()["C"], ridge=1.)
    rows += records_for(c, havg=2.08, h100=4.08, risk=0.)
    b = replace(initial_recipes()["B"], ridge=1.)
    rows += records_for(b, risk=4.)
    chosen = select_tuple(rows)
    assert chosen["recipes"]["B"]["ridge"] == 1.
    assert chosen["recipes"]["C"]["ridge"] == 1.
    assert chosen["mean_novel_auc"] == pytest.approx((10 + 4 + 0 + 10 + 10) / 5)


def test_score_is_equally_weighted_and_then_ties_use_total_learning_seconds():
    rows = starting_records()
    candidate = replace(initial_recipes()["B"], ridge=100.)
    rows += records_for(candidate, risk=10. + 4e-8, seconds=.5)
    chosen = select_tuple(rows)
    assert chosen["recipes"]["B"]["ridge"] == 100.
    assert chosen["mean_novel_auc"] == pytest.approx(10. + 8e-9)
    assert chosen["total_learning_seconds"] == 18.
    # Even an arbitrarily faster tuple cannot beat a score outside the tie.
    outside = replace(candidate, ridge=.1)
    rows += records_for(outside, risk=10. + 6e-8, seconds=0.)
    assert select_tuple(rows)["recipes"]["B"]["ridge"] == 100.


def test_exact_score_epsilon_is_a_tie_without_decimal_rounding():
    rows = starting_records()
    for row in rows:
        row["summary"]["novel_auc"] = 0.
    candidate = replace(initial_recipes()["B"], ridge=100.)
    rows += records_for(candidate, risk=5e-8, seconds=0.)
    chosen = select_tuple(rows)
    assert chosen["mean_novel_auc"] == EPSILON
    assert chosen["recipes"]["B"]["ridge"] == 100.


def test_score_ties_are_anchored_at_global_minimum_not_chained_or_rounded():
    rows = starting_records()
    b = initial_recipes()["B"]
    rows += records_for(replace(b, ridge=1.), risk=10. + 4e-8, seconds=.5)
    rows += records_for(replace(b, ridge=.1), risk=10. + 8e-8, seconds=.1)
    expected = select_tuple(rows)
    assert expected["recipes"]["B"]["ridge"] == 1.
    before = deepcopy(rows)
    for seed in range(8):
        shuffled = deepcopy(rows)
        random.Random(seed).shuffle(shuffled)
        assert select_tuple(iter(shuffled)) == expected
    assert rows == before


def test_ridge_tuple_breaks_exact_score_and_time_ties_in_A_to_E_order():
    rows = starting_records()
    for arm in ARMS[1:]:
        rows += records_for(replace(initial_recipes()[arm], ridge=0.))
    chosen = select_tuple(rows)
    assert chosen["ridge_tuple"] == [10., 0., 0., 0., 0.]
    assert chosen["stable_ids"] == [Recipe(**chosen["recipes"][arm]).key for arm in ARMS]


def test_ridge_proposal_uses_actual_nonmonotone_neighbors_and_records_evidence():
    b = initial_recipes()["B"]
    rows = records_for(replace(b, ridge=1.), havg=1., h100=3.)
    rows += records_for(b, havg=3., h100=5.)
    rows += records_for(replace(b, ridge=100.), havg=1., h100=3.)
    target = dict(Havg=2., H100=4.)
    before = deepcopy(rows)
    proposal = ridge_proposal("B", rows, b.mapping(), target, [1., 10., 100.])
    assert proposal["ridge"] == pytest.approx(math.sqrt(10.))
    assert proposal["bracket"] == [1., 10.]
    assert proposal["reason"] == "observed_two_statistic_bracket"
    assert [row["ridge"] for row in proposal["evidence"]] == [1., 10., 100.]
    assert propose_ridge("B", reversed(rows), b, target, []) == proposal["ridge"]
    next_proposal = ridge_proposal("B", rows, b, target, [proposal["ridge"]])
    assert next_proposal["bracket"] == [10., 100.]
    assert next_proposal["ridge"] == pytest.approx(math.sqrt(1000.))
    assert rows == before


def test_ridge_proposal_never_uses_a_single_statistic_as_a_joint_bracket():
    b = initial_recipes()["B"]
    rows = records_for(replace(b, ridge=1.), havg=1., h100=5.)
    rows += records_for(b, havg=3., h100=6.)
    result = ridge_proposal("B", rows, b, dict(Havg=2., H100=4.), [])
    assert result["reason"] == "adjacent_boundary_decade_probe"
    assert result["ridge"] == .1


def test_ridge_proposal_explores_the_observed_closest_boundary_without_fixed_slope():
    b = initial_recipes()["B"]
    rows = records_for(replace(b, ridge=1.), havg=4., h100=6.)
    rows += records_for(b, havg=5., h100=7.)
    # Increasing ridge increased forgetting here: explore below 1, not above 10.
    result = ridge_proposal("B", rows, b, dict(Havg=2., H100=4.), [])
    assert result["ridge"] == .1
    assert result["reason"] == "adjacent_boundary_decade_probe"


def test_overprotective_point_one_boundary_permits_zero_and_never_repeats_it():
    e = initial_recipes()["E"]
    rows = records_for(e, havg=1., h100=3.)
    target = dict(Havg=2., H100=4.)
    result = ridge_proposal("E", rows, e, target, [])
    assert result["ridge"] == 0.
    assert result["reason"] == "overprotective_lower_boundary_zero_probe"
    assert propose_ridge("E", rows, e, target, [0.]) != 0.


def test_ridge_proposals_are_bounded_do_not_repeat_failed_work_and_require_evidence():
    b = initial_recipes()["B"]
    target = dict(Havg=2., H100=4.)
    assert propose_ridge("B", [], b, target, []) is None
    assert propose_ridge("B", records_for(b, members=(0, 1)), b, target, []) is None
    rows = records_for(replace(b, ridge=1000.), havg=4., h100=6.)
    assert propose_ridge("B", rows, b, target, []) == 100.
    rows += records_for(replace(b, ridge=100.), status="failed")
    assert propose_ridge("B", rows, b, target, []) is None
    rows = records_for(replace(b, ridge=100.), havg=4., h100=6.)
    assert propose_ridge("B", rows, b, target, [10.]) == 1000.
    assert propose_ridge("B", rows, b, target, [10., 1000.]) is None


@pytest.mark.parametrize("used", [[-1.], [1001.], [float("nan")], [True], ["10"]])
def test_invalid_used_ridges_raise(used):
    b = initial_recipes()["B"]
    with pytest.raises(ValueError, match="used ridge"):
        propose_ridge("B", records_for(b), b, dict(Havg=2., H100=4.), used)


def test_A_is_fixed_and_never_admitted_as_a_refinement_arm():
    with pytest.raises(ValueError, match="B--E"):
        propose_ridge("A", [], initial_recipes()["A"], dict(Havg=2., H100=4.), [])
