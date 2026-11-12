"""Development calibration tests using compact measurements and scalar paths."""

from copy import deepcopy
from dataclasses import replace
import itertools
import math
import random

import pytest

from mnist_experiment.combined_memory.replicated_combination import calibration as cal
from mnist_experiment.fisher_memory_regression.config import Recipe


def records_for(policy, *, ridge=None, havg=.02, h100=.03, risk=10., seconds=1.,
                members=range(4), status="completed", process_scale=1e-4):
    canonical = cal.initial_recipes(process_scale)[policy]
    recipe = Recipe.from_mapping(canonical["recipe"])
    if ridge is not None:
        recipe = replace(recipe, ridge=ridge)
    return [dict(namespace="development", policy=policy, replica=m,
        recipe=recipe.mapping(), representation=canonical["representation"],
        process_scale=process_scale, status=status,
        Havg=havg, H100=h100, novel_auc=risk, learning_seconds=seconds) for m in members]


def starting_records(**kwargs):
    return list(itertools.chain.from_iterable(records_for(p, **kwargs) for p in cal.POLICIES))


def test_starts_and_legacy_recipe_outputs_preserve_fixed_B_R_and_shared_process_scale():
    initial = cal.initial_recipes()
    assert initial["A"]["ridge"] == math.sqrt(10)
    assert all(initial[p]["ridge"] == 1 for p in ("R", "B", "C", "D", "E"))
    assert initial["R"]["representation"] == "low_rank_diagonal"
    assert initial["A"]["recipe"]["strength"] == 0
    selected = cal.select_tuple(starting_records())
    assert selected["selection_ready"] and selected["matching"]["passed"]
    assert set(selected["recipes"]) == set(cal.ARMS)
    assert all(isinstance(Recipe.from_mapping(v), Recipe) for v in selected["recipes"].values())
    assert selected["recipes"]["B"]["ridge"] == selected["representation_reference"]["ridge"] == 1
    assert selected["mean_novel_auc"] == 10
    assert selected["total_learning_seconds"] == 20
    assert selected["representation_reference"]["policy"] == "R"


def test_aggregate_preserves_signed_member_retention_and_sums_time():
    rows = records_for("B")
    for m, row in enumerate(rows):
        row.update(Havg=m - 1., H100=m - 2., learning_seconds=m + 1.)
    result = cal.aggregate(rows, cal.initial_recipes()["B"])
    assert result["Havg"] == .5
    assert result["H100"] == -.5
    assert result["learning_seconds"] == 2.5
    assert result["total_learning_seconds"] == 10
    assert result["members"] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="missing or incomplete"):
        cal.aggregate(rows[:-1], cal.initial_recipes()["B"])


@pytest.mark.parametrize("field", cal.RETENTION)
def test_full_range_gate_is_stricter_than_each_arms_distance_to_B(field):
    rows = starting_records()
    for row in rows:
        row[field] += -.009 if row["policy"] == "A" else .009 if row["policy"] == "C" else 0
    assert cal.select_tuple(rows) is None
    rows += records_for("A", ridge=.1, havg=.029 if field == "Havg" else .02,
                        h100=.039 if field == "H100" else .03, risk=20.)
    chosen = cal.select_tuple(rows)
    assert chosen["recipes"]["A"]["ridge"] == .1
    assert chosen["matching"]["ranges"][field] == pytest.approx(.009)


def test_R_is_excluded_from_matching_risk_and_runtime_ranking_but_requires_coverage():
    rows = starting_records()
    for row in rows:
        if row["policy"] == "R":
            row.update(Havg=50., H100=100., novel_auc=10000., learning_seconds=10000.)
    chosen = cal.select_tuple(rows)
    assert chosen["selection_ready"]
    assert chosen["mean_novel_auc"] == 10
    assert chosen["total_learning_seconds"] == 20
    assert chosen["representation_reference"]["H100"] == 100
    assert cal.select_tuple(rows[1:]) is None


def test_ceiling_is_separate_from_a_passing_range_and_negative_changes_are_allowed():
    assert cal.select_tuple(starting_records(havg=.51, h100=.51)) is None
    result = cal.select_tuple(starting_records(havg=-.05, h100=-.1))
    assert result["selection_ready"]
    assert result["calibration"]["B"]["H100"] == -.1


def test_first_pair_proposals_do_not_certify_four_member_calibration():
    rows = starting_records(members=(0, 1))
    first = cal.select_tuple(rows, replicas=(0, 1))
    assert first["matching"]["passed"] and not first["selection_ready"]
    assert first["members"] == [0, 1]
    assert cal.select_tuple(rows) is None
    rows += starting_records(members=(2, 3))
    assert cal.select_tuple(rows)["selection_ready"]


def test_unmatched_tuple_is_an_explicit_proposal_and_never_confirmation_ready():
    rows = starting_records()
    for row in rows:
        if row["policy"] == "E":
            row["H100"] += .03
    assert cal.select_tuple(rows) is None
    proposed = cal.select_tuple(rows, require_match=False)
    assert proposed["status"] == "proposed_unmatched"
    assert not proposed["selection_ready"]
    assert proposed["matching"]["maximum_violation"] == pytest.approx(.02)
    rows += records_for("E", ridge=10., h100=.045, risk=100.)
    proposed = cal.select_tuple(rows, require_match=False)
    assert proposed["recipes"]["E"]["ridge"] == 10
    assert proposed["matching"]["maximum_violation"] == pytest.approx(.005)


def test_incomplete_low_risk_candidate_cannot_replace_full_coverage():
    rows = starting_records() + records_for("A", ridge=.1, members=(0, 1), risk=0.)
    selected = cal.select_tuple(rows)
    assert selected["recipes"]["A"]["ridge"] == math.sqrt(10)
    assert list(selected["diagnostics"]["incomplete_candidates"].values()) == [[2, 3]]


def test_equal_arm_risk_then_global_epsilon_time_and_stable_ids():
    rows = starting_records()
    rows += records_for("A", ridge=.1, risk=10 + 4e-8, seconds=.5)
    rows += records_for("A", ridge=.01, risk=10 + 8e-8, seconds=.1)
    chosen = cal.select_tuple(rows)
    assert chosen["recipes"]["A"]["ridge"] == .1
    assert chosen["total_learning_seconds"] == 18
    for seed in range(3):
        shuffled = deepcopy(rows)
        random.Random(seed).shuffle(shuffled)
        assert cal.select_tuple(shuffled) == chosen
    identical = starting_records() + records_for("C", ridge=.1)
    expected = min(cal.recipe_id(dict(policy="C", recipe=r["recipe"])) for r in identical if r["policy"] == "C")
    assert cal.select_tuple(identical)["calibration"]["C"]["candidate_id"] == expected


@pytest.mark.parametrize("policy", ["B", "R"])
def test_fixed_ridge_cannot_be_changed_even_if_it_would_improve_results(policy):
    with pytest.raises(ValueError, match="fixed at ridge 1"):
        cal.select_tuple(starting_records() + records_for(policy, ridge=10., risk=0.))


@pytest.mark.parametrize("field,value", [("Havg", float("nan")), ("H100", "0"),
    ("novel_auc", -1.), ("learning_seconds", float("inf")), ("namespace", "fresh"),
    ("replica", True), ("representation", "unsupported"), ("valid", False)])
def test_invalid_development_evidence_raises(field, value):
    rows = starting_records()
    rows[-1][field] = value
    with pytest.raises(ValueError):
        cal.select_tuple(rows)


def test_duplicate_candidate_members_cannot_be_counted_twice():
    rows = starting_records()
    with pytest.raises(ValueError, match="duplicate development"):
        cal.select_tuple(rows + [deepcopy(rows[0])])


def test_ridge_helper_uses_half_and_whole_decades_and_bounded_zero_probe():
    target = dict(Havg=0., H100=0.)
    under = cal.proposals(1., target, dict(Havg=.1, H100=.2))
    assert [r["ridge"] for r in under] == pytest.approx([math.sqrt(10), 10])
    over = cal.proposals(1., target, dict(Havg=-.1, H100=-.2))
    assert [r["ridge"] for r in over] == pytest.approx([1 / math.sqrt(10), .1])
    zero = cal.proposals(1e-4, target, dict(Havg=-.1, H100=-.2))
    assert zero == [dict(ridge=0., reason="overprotective_positive_lower_boundary", predicted_retention=None)]
    assert all(0 <= r["ridge"] <= 1000 for r in cal.proposals(900, target, dict(Havg=.1, H100=.2)))
    assert cal.proposals(1000, target, dict(Havg=.1, H100=.2)) == []


def test_empirical_interpolation_handles_reversed_responses_without_monotonicity_assumption():
    rows = records_for("B", havg=0., h100=0., members=(0, 1))
    rows += records_for("C", ridge=.1, havg=-.03, h100=-.04, members=(0, 1))
    rows += records_for("C", ridge=10., havg=.03, h100=.04, members=(0, 1))
    proposed = cal.ridge_proposal("C", rows)
    assert proposed["ridge"] == pytest.approx(1.)
    assert proposed["reason"] == "empirical_log_interpolation"
    assert proposed["predicted_retention"] == dict(Havg=0., H100=0.)
    assert proposed["covered_members"] == [0, 1]
    assert not proposed["partial_evidence"]
    assert proposed["recipe"]["ridge"] == pytest.approx(1.)


def test_stage_allowances_and_fixed_policy_proposal_prohibition():
    rows = starting_records()
    for row in rows:
        if row["policy"] == "C":
            row.update(Havg=.1, H100=.1)
    assert cal.ridge_proposal("C", rows, proposals_used=2) is None
    assert cal.ridge_proposal("C", rows, stage="repair", replicas=(0, 1, 2, 3), proposals_used=1) is None
    assert cal.ridge_proposal("C", rows)["ridge"] == pytest.approx(math.sqrt(10))
    for p in ("R", "B"):
        with pytest.raises(ValueError):
            cal.ridge_proposal(p, rows)


def test_repair_can_extend_partial_candidate_but_does_not_fabricate_missing_members():
    rows = starting_records()
    for row in rows:
        if row["policy"] == "C":
            row.update(Havg=.1, H100=.1)
    rows += records_for("C", ridge=10., members=(0, 1))
    proposed = cal.ridge_proposal("C", rows, stage="repair", replicas=range(4))
    assert proposed["extension"] and proposed["ridge"] == 10
    assert proposed["covered_members"] == [0, 1]
    assert proposed["missing_members"] == [2, 3]
    assert proposed["partial_evidence"]
    assert cal.ridge_proposal("C", rows, stage="repair") == proposed


def test_nonfinite_partial_candidate_is_not_silently_dropped():
    rows = starting_records() + records_for("C", ridge=10., members=(0,), status="running")
    rows[-1]["Havg"] = float("nan")
    with pytest.raises(ValueError, match="Havg"):
        cal.select_tuple(rows)


def test_metric_summary_roots_correctly_and_is_not_a_pooled_temporal_rms():
    metrics = [dict(n=n, observations=4 * n, base_mse=v, novel_mse=10 - n,
                    learning_seconds=2. * n) for n, v in enumerate([4., 1., 9., 1., 9.])]
    result = cal.summarize_metrics(metrics, N=4)
    assert result == dict(Havg=math.sqrt(5) - 2, H100=1., novel_auc=7.5, learning_seconds=8.)
    assert result["Havg"] != sum(math.sqrt(p["base_mse"]) for p in metrics[1:]) / 4 - 2
    with pytest.raises(ValueError):
        cal.summarize_metrics(metrics[:-1], N=4)


def test_legacy_wrapped_records_are_verified_and_summarized_from_actual_metrics():
    from test_combined_memory_replicated_analysis import record

    rows = [record(p, r, namespace="development") for p in cal.POLICIES for r in range(4)]
    chosen = cal.select_tuple(rows)
    assert chosen["selection_ready"]
    row = rows[0]
    summary = cal.summarize_record(row)
    assert summary["Havg"] == pytest.approx(math.sqrt(4.02) - 2)
    wrapped = [dict(summary={k: v for k, v in r.items() if k != "metrics"}, metrics=r["metrics"]) for r in rows]
    assert cal.select_tuple(wrapped) == chosen
    rows[0]["Havg"] = .4
    with pytest.raises(ValueError, match="summary disagrees"):
        cal.select_tuple(rows)


def test_shared_process_scale_filters_other_closed_loop_candidates():
    rows = starting_records(process_scale=1e-5)
    rows += records_for("D", process_scale=1e-4, risk=0.)
    selected = cal.select_tuple(rows, process_scale=1e-5)
    assert selected["policies"]["D"]["process_scale"] == 1e-5
    assert selected["calibration"]["D"]["novel_auc"] == 10


def test_innovation_selection_is_equal_replica_mean_and_uses_prespecified_tie_break():
    scores = [dict(namespace="development", replica=r, process_scale=s, mean_log_score=-1.)
              for s in cal.PROCESS_SCALES for r in (0, 1)]
    assert cal.select_process_scale(scores)["process_scale"] == 1e-4
    for row in scores:
        if row["process_scale"] == 1e-3:
            row["mean_log_score"] += 2e-8
    assert cal.select_process_scale(scores)["process_scale"] == 1e-3
    with pytest.raises(ValueError, match="both development"):
        cal.select_process_scale(scores[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        cal.select_process_scale(scores + [scores[0]])
