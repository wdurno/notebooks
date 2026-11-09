"""Pure scalar Phase 7 fixtures: no training, data loading, or GPU work."""

from copy import deepcopy
from dataclasses import replace
import itertools
import math
import random

import pytest

from mnist_experiment.combined_memory.config import ARMS, Recipe
from mnist_experiment.combined_memory.low_ridge import calibration as cal
from mnist_experiment.combined_memory.retention_match import calibration as phase6


def records_for(recipe, *, havg=2., h100=4., risk=10., seconds=1., members=range(4),
                cohort="development", status="completed"):
    return [dict(unit_id=f"{cohort}-{recipe.key}-{member}", status=status,
        metadata=dict(kind="trajectory", cohort=cohort, replica=member, recipe=recipe.mapping()),
        summary=dict(Havg=havg, H100=h100, novel_auc=risk, learning_seconds=seconds))
        for member in members]


def starting_records(strength=cal.STRENGTH):
    return list(itertools.chain.from_iterable(records_for(recipe)
                for recipe in cal.initial_recipes(strength).values()))


def reference_records(**kwargs):
    return records_for(cal.initial_recipes()["B"], **kwargs)


def candidate(arm="C", ridge=1.):
    return replace(cal.initial_recipes()[arm], ridge=ridge)


def test_starting_recipes_freeze_B_but_not_A_and_return_fresh_objects():
    recipes = cal.initial_recipes()
    assert cal.MEMBERS == (0, 1, 2, 3)
    assert cal.STRENGTH == .5691316767656495
    assert list(recipes) == list(ARMS)
    assert [row.ridge for row in recipes.values()] == [1., 1., 1., .1, .1]
    for arm, recipe in recipes.items():
        assert isinstance(recipe, Recipe)
        assert recipe.strength == (0. if arm == "A" else cal.STRENGTH)
        assert recipe.process_scale == (1e-5 if arm in ("D", "E") else 1e-4)
        assert cal.compatible(recipe.mapping())
    recipes.pop("E")
    assert "E" in cal.initial_recipes()
    assert cal.initial_recipes(.25)["B"].strength == .25
    assert cal.initial_recipes(.25)["A"].strength == 0.


def test_aggregate_and_matching_reuse_the_frozen_phase6_implementations():
    assert cal.aggregate is phase6.aggregate
    assert cal.matching is phase6.matching
    recipe = candidate("A")
    rows = records_for(recipe)
    for member, row in enumerate(rows):
        row["summary"].update(Havg=member - 1., H100=member + 1., learning_seconds=member + 1.)
    result = cal.aggregate(rows + records_for(recipe, cohort="fresh"), recipe)
    assert result["Havg"] == .5
    assert result["H100"] == 2.5
    assert result["learning_seconds"] == 2.5
    assert result["total_learning_seconds"] == 10.
    assert result["members"] == [0, 1, 2, 3]


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("ridge", [0., 1e-4, .1, 1., 1000.])
def test_compatible_checks_the_ridge_domain_and_B_lock(arm, ridge):
    assert cal.compatible(candidate(arm, ridge)) is (arm != "B" or ridge == 1.)


@pytest.mark.parametrize("ridge", [1e-5, 1000.001])
def test_out_of_bounds_well_formed_recipes_are_incompatible(ridge):
    assert not cal.compatible(candidate(ridge=ridge))


def test_strength_and_process_are_fixed_in_both_compatibility_and_selection():
    rows = starting_records()
    extras = [replace(candidate(), strength=cal.STRENGTH * 2),
              replace(candidate(), process_scale=1e-5),
              replace(candidate("D"), process_scale=1e-4),
              candidate("E", 1001.), candidate("A", 1e-5)]
    for recipe in extras:
        assert not cal.compatible(recipe)
        rows += records_for(recipe, risk=0.)
    chosen = cal.select_tuple(rows)
    assert chosen["diagnostics"]["complete_candidates"] == dict.fromkeys(ARMS, 1)
    assert chosen["recipes"] == {arm: row.mapping() for arm, row in cal.initial_recipes().items()}
    smoke_rows = starting_records(.1)
    assert cal.select_tuple(smoke_rows) is None
    assert cal.compatible(cal.initial_recipes(.1)["B"], strength=.1)
    assert cal.select_tuple(smoke_rows, strength=.1)["strength"] == .1


def test_selected_tuple_exposes_the_reference_and_all_four_member_summaries():
    chosen = cal.select_tuple(starting_records())
    assert chosen["reference_arm"] == "B"
    assert chosen["reference_ridge"] == 1.
    assert chosen["members"] == [0, 1, 2, 3]
    assert chosen["mean_novel_auc"] == 10.
    assert chosen["total_learning_seconds"] == 20.
    assert chosen["matching"]["passed"]
    assert chosen["matching"]["ranges"] == dict(Havg=0., H100=0.)
    assert all(row["n_members"] == 4 for row in chosen["calibration"].values())
    assert chosen["stable_ids"] == [Recipe(**chosen["recipes"][arm]).key for arm in ARMS]


def test_phase6_tuple_cannot_substitute_for_the_mandatory_B1_reference():
    rows = list(itertools.chain.from_iterable(records_for(recipe)
                for recipe in phase6.initial_recipes().values()))
    assert cal.select_tuple(rows) is None
    rows += reference_records(risk=100.)
    rows += records_for(candidate("B", 1000.), risk=0.)
    chosen = cal.select_tuple(rows)
    assert chosen["recipes"]["B"]["ridge"] == 1.
    assert chosen["calibration"]["B"]["novel_auc"] == 100.


def test_A_is_tunable_and_joint_selection_avoids_unmatchable_individual_optima():
    rows = starting_records()
    for row in rows[:4]:
        row["summary"].update(Havg=1.92, H100=3.92, novel_auc=1.)
    rows += records_for(candidate("A", .1), risk=4.)
    rows += records_for(candidate("C", .1), havg=2.08, h100=4.08, risk=0.)
    chosen = cal.select_tuple(rows)
    assert chosen["recipes"]["A"]["ridge"] == .1
    assert chosen["recipes"]["C"]["ridge"] == .1
    assert chosen["recipes"]["B"]["ridge"] == 1.
    assert chosen["mean_novel_auc"] == pytest.approx((4 + 10 + 0 + 10 + 10) / 5)


@pytest.mark.parametrize("field", ["Havg", "H100"])
def test_all_arm_range_gate_is_stricter_than_each_arms_proximity_to_B(field):
    rows = starting_records()
    for row in rows:
        arm = row["metadata"]["recipe"]["arm"]
        row["summary"][field] += -.09 if arm == "A" else .09 if arm == "C" else 0.
    assert cal.select_tuple(rows) is None


@pytest.mark.parametrize("field", ["Havg", "H100"])
def test_either_failed_retention_statistic_rejects_selection(field):
    rows = starting_records()
    for row in rows[-4:]:
        row["summary"][field] += .11
    assert cal.select_tuple(rows) is None


@pytest.mark.parametrize("status", [None, "failed", "running", "incomplete"])
def test_incomplete_B_and_other_arms_cannot_select_even_with_fresh_substitutes(status):
    for arm in ("B", "E"):
        rows = [row for row in starting_records()
                if not (row["metadata"]["replica"] == 3 and row["metadata"]["recipe"]["arm"] == arm)]
        recipe = cal.initial_recipes()[arm]
        if status is not None:
            rows += records_for(recipe, members=(3,), status=status)
        rows += records_for(recipe, members=(3,), cohort="fresh")
        rows += records_for(recipe, members=(3,), cohort="stationary")
        assert cal.select_tuple(rows) is None


def test_partial_alternative_is_reported_but_not_selected_and_summary_is_validated():
    recipe = candidate("A", .1)
    partial = records_for(recipe, members=(0, 1), risk=0.)
    chosen = cal.select_tuple(starting_records() + partial)
    assert chosen["recipes"]["A"]["ridge"] == 1.
    assert chosen["diagnostics"]["incomplete_candidates"] == {recipe.key: [2, 3]}
    partial[0]["summary"]["Havg"] = float("nan")
    with pytest.raises(ValueError, match="Havg"):
        cal.select_tuple(starting_records() + partial)


def test_risk_ties_use_the_global_epsilon_then_time_and_do_not_chain():
    rows = starting_records()
    rows += records_for(candidate("A", .1), risk=10. + 4e-8, seconds=.5)
    rows += records_for(candidate("A", .01), risk=10. + 8e-8, seconds=.1)
    chosen = cal.select_tuple(rows)
    assert chosen["recipes"]["A"]["ridge"] == .1
    assert chosen["mean_novel_auc"] == pytest.approx(10. + 8e-9)
    assert chosen["total_learning_seconds"] == 18.
    for seed in range(5):
        shuffled = deepcopy(rows)
        random.Random(seed).shuffle(shuffled)
        assert cal.select_tuple(iter(shuffled)) == chosen


def test_exact_epsilon_and_lexicographic_ridge_ties():
    rows = starting_records()
    for row in rows:
        row["summary"]["novel_auc"] = 0.
    rows += records_for(candidate("A", .1), risk=5e-8, seconds=0.)
    chosen = cal.select_tuple(rows)
    assert chosen["mean_novel_auc"] == cal.EPSILON
    assert chosen["recipes"]["A"]["ridge"] == .1
    rows = starting_records()
    for arm in ("A", "C", "D", "E"):
        rows += records_for(candidate(arm, 0.))
    assert cal.select_tuple(rows)["ridge_tuple"] == [0., 1., 0., 0., 0.]


@pytest.mark.parametrize("operation", [cal.select_tuple,
    lambda rows: cal.ridge_proposal("C", rows, [])])
def test_duplicate_development_members_are_rejected(operation):
    rows = starting_records()
    with pytest.raises(ValueError, match="duplicate development member"):
        operation(rows + [deepcopy(rows[0])])


@pytest.mark.parametrize("field", ["Havg", "H100", "novel_auc", "learning_seconds"])
@pytest.mark.parametrize("value", [None, "2", True, float("nan"), float("inf"), []])
def test_completed_malformed_metrics_raise_even_in_partial_evidence(field, value):
    partial = records_for(candidate(), members=(0, 1))
    partial[0]["summary"][field] = value
    for operation in (cal.select_tuple, lambda rows: cal.ridge_proposal("C", rows, [])):
        with pytest.raises(ValueError, match=field):
            operation(reference_records() + partial)


@pytest.mark.parametrize("field", ["novel_auc", "learning_seconds"])
def test_negative_risk_and_time_raise(field):
    rows = starting_records()
    rows[0]["summary"][field] = -1.
    with pytest.raises(ValueError, match=field):
        cal.select_tuple(rows)


@pytest.mark.parametrize("tolerance", [None, True, -.1, "0.1", float("nan"), float("inf")])
def test_invalid_tolerance_raises_without_evidence(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        cal.select_tuple([], tolerance)
    with pytest.raises(ValueError, match="tolerance"):
        cal.ridge_proposal("A", [], [], tolerance)


def test_malformed_recipes_and_member_metadata_raise():
    rows = reference_records()
    del rows[0]["metadata"]["recipe"]["strength"]
    with pytest.raises(ValueError, match="complete recipe"):
        cal.select_tuple(rows)
    with pytest.raises(ValueError, match="complete recipe"):
        cal.compatible({"arm": "B", "ridge": 1.})
    rows = reference_records()
    rows[0]["metadata"]["replica"] = True
    with pytest.raises(ValueError, match="development replica"):
        cal.ridge_proposal("A", rows, [])
    with pytest.raises(ValueError, match="development members"):
        cal.select_tuple([], members=(0, 0))


@pytest.mark.parametrize("arm", ["A", "C", "D", "E"])
def test_midpoints_handle_one_metric_already_in_band_without_exact_bracketing(arm):
    rows = reference_records()
    rows += records_for(candidate(arm, .1), havg=1., h100=4.04)
    rows += records_for(candidate(arm, 10.), havg=3., h100=4.06)
    result = cal.ridge_proposal(arm, rows, [])
    assert result["ridge"] == pytest.approx(1.)
    assert result["bracket"] == [.1, 10.]
    assert result["reason"] == "log_midpoint_predicted_improvement"
    assert result["predicted_retention"] == pytest.approx(dict(Havg=2., H100=4.05))
    assert result["predicted_worst_mismatch"] == pytest.approx(.05)
    assert result["endpoint_worst_mismatch"] == 1.
    assert result["reference_arm"] == "B" and result["reference_ridge"] == 1.
    assert result["covered_members"] == [0, 1, 2, 3]
    assert not result["partial_evidence"]


def test_partial_brackets_and_B_reference_use_exactly_the_same_member_intersection():
    rows = reference_records()
    rows += records_for(candidate(ridge=.1), havg=1., h100=3., members=(0, 1))
    rows += records_for(candidate(ridge=10.), havg=3., h100=5., members=(1, 2))
    for row in rows:
        member = row["metadata"]["replica"]
        row["summary"]["Havg"] += 10 * member
        row["summary"]["H100"] += 10 * member
    before = deepcopy(rows)
    result = cal.ridge_proposal("C", iter(rows), [])
    assert result["ridge"] == pytest.approx(1.)
    assert result["covered_members"] == [1]
    assert result["reference"]["Havg"] == 12.
    assert result["reference"]["H100"] == 14.
    assert result["predicted_retention"] == dict(Havg=12., H100=14.)
    assert result["partial_evidence"]
    assert all(row["members"] == [1] for row in result["evidence"])
    assert result["requested_members"] == [0, 1, 2, 3]
    assert cal.select_tuple(rows) is None
    assert rows == before


def test_partial_B_can_guide_search_but_cannot_certify_and_B10_is_not_a_target():
    rows = reference_records(members=(0, 1))
    rows += records_for(candidate("B", 10.), havg=20., h100=40.)
    rows += records_for(candidate(ridge=.1), havg=1., h100=3.)
    rows += records_for(candidate(ridge=10.), havg=3., h100=5.)
    result = cal.ridge_proposal("C", rows, [])
    assert result["ridge"] == pytest.approx(1.)
    assert result["reference"]["members"] == [0, 1]
    assert result["reference"]["recipe"]["ridge"] == 1.
    assert cal.select_tuple(rows) is None
    assert cal.ridge_proposal("C", rows[2:], []) is None


def test_disjoint_members_cannot_supply_a_bracket_or_unpaired_reference():
    rows = reference_records(members=(0,))
    rows += records_for(candidate(ridge=1.), members=(1,), havg=1., h100=3.)
    rows += records_for(candidate(ridge=10.), members=(2,), havg=3., h100=5.)
    assert cal.ridge_proposal("C", rows, []) is None


def test_neighbors_are_defined_on_the_shared_subset_not_unrelated_middle_evidence():
    rows = reference_records()
    rows += records_for(candidate(ridge=.1), members=(0, 1), havg=1., h100=3.)
    rows += records_for(candidate(ridge=1.), members=(2, 3), havg=20., h100=40.)
    rows += records_for(candidate(ridge=10.), members=(0, 1), havg=3., h100=5.)
    result = cal.ridge_proposal("C", rows, [])
    assert result["bracket"] == [.1, 10.]
    assert result["covered_members"] == [0, 1]
    assert result["missing_members"] == [0, 1]


def test_nonmonotone_midpoints_minimize_predicted_worst_two_metric_mismatch():
    rows = reference_records()
    for ridge, delta in ((.1, -1.), (1., 1.), (10., -.2), (100., .1)):
        rows += records_for(candidate(ridge=ridge), havg=2. + delta, h100=4. + delta)
    result = cal.ridge_proposal("C", rows, [])
    assert result["bracket"] == [.1, 1.]
    assert result["predicted_worst_mismatch"] == pytest.approx(0.)
    assert result["ridge"] == pytest.approx(math.sqrt(.1))
    next_result = cal.ridge_proposal("C", rows, [result["ridge"]])
    assert next_result["bracket"] == [10., 100.]
    assert next_result["predicted_worst_mismatch"] == pytest.approx(.05)
    before = deepcopy(rows)
    for seed in range(5):
        random.Random(seed).shuffle(rows)
        assert cal.ridge_proposal("C", iter(rows), []) == result
    assert sorted(rows, key=lambda row: row["unit_id"]) == sorted(before, key=lambda row: row["unit_id"])


def test_midpoint_score_ties_prefer_four_members_then_stable_recipe_ids():
    rows = reference_records()
    for ridge, delta, members in ((.001, -1., (0, 1)), (.01, 1., (0, 1)),
                                  (1., -1., range(4)), (10., 1., range(4))):
        rows += records_for(candidate(ridge=ridge), havg=2. + delta, h100=4. + delta, members=members)
    result = cal.ridge_proposal("C", rows, [])
    assert result["bracket"] == [1., 10.]
    assert result["n_members"] == 4
    rows = reference_records()
    for ridge, delta in ((.1, -1.), (1., 1.), (10., -1.)):
        rows += records_for(candidate(ridge=ridge), havg=2. + delta, h100=4. + delta)
    pairs = [(.1, 1.), (1., 10.)]
    expected = min(pairs, key=lambda pair: tuple(candidate(ridge=ridge).key for ridge in pair))
    assert cal.ridge_proposal("C", rows, [])["bracket"] == list(expected)


def test_nonimproving_interior_uses_closest_measured_edge_without_extrapolation():
    rows = reference_records()
    rows += records_for(candidate(ridge=1.), havg=4., h100=6.)
    rows += records_for(candidate(ridge=10.), havg=5., h100=7.)
    result = cal.ridge_proposal("C", rows, [])
    assert result["ridge"] == .1
    assert result["bracket"] == [.1, 1.]
    assert result["reason"] == "adjacent_boundary_decade_probe"
    assert result["predicted_retention"] is None
    assert result["predicted_worst_mismatch"] is None
    assert result["endpoint_worst_mismatch"] == 2.


def test_zero_boundary_probe_allows_one_overprotective_metric_and_never_logs_zero():
    rows = reference_records() + records_for(candidate("E", .1), havg=1., h100=4.)
    result = cal.ridge_proposal("E", rows, [])
    assert result["ridge"] == 0.
    assert result["reason"] == "overprotective_lower_boundary_zero_probe"
    assert cal.ridge_proposal("E", rows, [0.])["ridge"] > 0
    rows = reference_records() + records_for(candidate(ridge=0.))
    assert cal.ridge_proposal("C", rows, []) is None
    rows += records_for(candidate(ridge=.1), havg=1., h100=3.)
    rows += records_for(candidate(ridge=10.), havg=3., h100=5.)
    result = cal.ridge_proposal("C", rows, [])
    assert result["ridge"] == pytest.approx(1.)
    assert result["bracket"] == [.1, 10.]


@pytest.mark.parametrize("ridge,expected", [(1e-4, .001), (1000., 100.)])
def test_boundary_decades_respect_positive_domain(ridge, expected):
    rows = reference_records() + records_for(candidate(ridge=ridge), havg=4., h100=6.)
    result = cal.ridge_proposal("C", rows, [])
    assert result["ridge"] == expected
    assert cal.ridge_proposal("C", rows, [expected]) is None


def test_failed_or_explicitly_used_proposals_are_not_retried():
    rows = reference_records() + records_for(candidate(ridge=100.), havg=4., h100=6.)
    rows += records_for(candidate(ridge=10.), status="failed")
    result = cal.ridge_proposal("C", rows, [])
    assert result["ridge"] == 1000.
    assert cal.ridge_proposal("C", rows, [1000.]) is None


def test_proposal_can_complete_a_partial_import_without_counting_it_as_untried_data():
    midpoint = math.sqrt(.1 * 10.)
    rows = reference_records()
    rows += records_for(candidate(ridge=.1), havg=1., h100=3.)
    rows += records_for(candidate(ridge=10.), havg=3., h100=5.)
    rows += records_for(candidate(ridge=midpoint), members=(0, 1))
    result = cal.ridge_proposal("C", rows, [])
    assert result["ridge"] == midpoint
    assert result["missing_members"] == [2, 3]
    assert result["covered_members"] == [0, 1, 2, 3]
    assert midpoint not in result["used_ridges"]


@pytest.mark.parametrize("arm", ["B", "F", None, True])
def test_only_A_C_D_E_can_be_refined(arm):
    with pytest.raises(ValueError, match="A/C/D/E"):
        cal.ridge_proposal(arm, [], [])


@pytest.mark.parametrize("used", [[-1.], [1001.], [1e-5], [float("nan")], [True], ["1"]])
def test_invalid_used_ridges_raise(used):
    with pytest.raises(ValueError, match="used ridge"):
        cal.ridge_proposal("C", [], used)


def test_mapping_records_explicit_smoke_policy_and_members_remain_deterministic():
    recipes = cal.initial_recipes(.1)
    rows = list(itertools.chain.from_iterable(records_for(recipe, members=(0, 1))
                for recipe in recipes.values()))
    keyed = {row["unit_id"]: row for row in rows}
    selected = cal.select_tuple(keyed, members=(1, 0), strength=.1)
    assert selected == cal.select_tuple(iter(rows), members=(0, 1), strength=.1)
    assert selected["members"] == [0, 1]
    rows = records_for(recipes["B"], members=(0, 1))
    rows += records_for(replace(recipes["A"], ridge=.1), havg=1., h100=3., members=(0, 1))
    rows += records_for(replace(recipes["A"], ridge=10.), havg=3., h100=5., members=(0, 1))
    assert cal.ridge_proposal("A", rows, [], members=(0, 1), strength=.1)["ridge"] == pytest.approx(1.)
    assert cal.ridge_proposal("A", rows, [], members=(0, 1)) is None
