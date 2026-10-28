"""CPU-only Phase 9 selection and inference on small synthetic trajectories."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from mnist_experiment.fisher_memory_regression.analysis import summarize_trajectory, replica_draw_indices
from mnist_experiment.fisher_memory_regression.config import Config as RegressionConfig, METHODS, Recipe, content_hash
from mnist_experiment.fisher_memory_shift import PROTOCOL, analysis, design

FD, REPLAY, EWC = METHODS


def cfg(**changes):
    values = {**asdict(RegressionConfig()), "protocol": PROTOCOL, "reference_capacity": 256,
              "calibration_replicas": 2, "refinement_replicas": 2, "matching_tolerance": .2,
              "retention_margin_degrees": 1., "ridge_grid": (1., 10., 100.),
              "replay_counts": (4, 16), "strength_ratios": (.1, 1., 10.), "refinement_settings": 8}
    return SimpleNamespace(**{**values, **changes})


def record(method, replica=0, *, ridge=10., strength=2., b=4, h=.9, a=None,
           partition="development", context="gamma-test", novel=None, base=None, time=None):
    recipe = Recipe.from_config(cfg(), method, ridge=ridge, replay_count=b,
                                strength=strength if method == FD else 0.)
    n = np.arange(101, dtype=float)
    initial = 300. + 12 * replica
    if novel is None:
        slope = (2. + .15 * replica) if method == FD else (1. + .1 * replica)
        novel = initial - slope * n
        if a is not None:
            novel[1:] = a
    if base is None:
        base = np.r_[9., np.full(100, (3 + h) ** 2)]
    if time is None:
        time = (.4 if method == FD else .25) * n
    metrics = [dict(n=int(i), observations=(10 + replica) * int(i), learning_seconds=float(t),
                    novel_mse=float(v), base_mse=float(o), u=.5 + .005 * i)
               for i, v, o, t in zip(n, novel, base, time)]
    summary = summarize_trajectory(metrics, N=100)
    return dict(summary, protocol=PROTOCOL, family=f"{partition}--{context}", partition=partition,
        target=100, replica=replica, recipe=recipe.mapping(), recipe_key=recipe.key, method=method,
        capacity=recipe.capacity, replay_count=recipe.replay_count, metrics=metrics,
        plan_hash=f"{partition}-plan-{replica}", pairing_id=f"{partition}-plan-{replica}",
        active_seconds=50., peak_vram_bytes=1000, retired_audit=[])


def pair(method, **kwargs):
    return [record(method, r, **kwargs) for r in range(2)]


def refresh(row):
    summary = summarize_trajectory(row["metrics"], N=100)
    row.update({k: summary[k] for k in analysis.SUMMARY_FIELDS})


def cohort():
    from mnist_experiment.fisher_memory_shift.config import Context

    context = Context.from_mapping(dict(key="gamma-test", family="gamma", gamma_max=1.8, blur_max=0.))
    records = [record(m, r, partition="confirmation", context=context.key) for m in METHODS for r in range(12)]
    recipes = [{**next(r["recipe"] for r in records if r["method"] == m),
                "recipe_key": next(r["recipe_key"] for r in records if r["method"] == m)} for m in METHODS]
    manifest = dict(protocol=PROTOCOL, schema_version=1, sealed=True, N=100, replicas=list(range(12)),
        context=context.mapping(), recipes=recipes, delta=1., epsilon=.2, source_hash="sealed-source",
        limits=dict(learning_time_limit=1., memory_ceiling_bytes=2000))
    manifest["manifest_hash"] = content_hash(manifest)
    return records, manifest


def reseal(manifest):
    manifest["manifest_hash"] = content_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})


def test_initial_grid_has_only_frozen_capacity_and_27_distinct_settings():
    recipes = design.initial_grid(cfg(), 2.)
    assert len(recipes) == len({r.key for r in recipes}) == 27
    assert Counter_methods(recipes) == {FD: 18, REPLAY: 6, EWC: 3}
    assert {r.capacity for r in recipes if r.method != EWC} == {256}
    assert {r.strength for r in recipes if r.method == FD} == {.2, 2., 20.}
    assert recipes == design.initial_grid(cfg(), 2.)
    with pytest.raises(ValueError):
        design.initial_grid(cfg(), 0)


def Counter_methods(recipes):
    return {m: sum(r.method == m for r in recipes) for m in METHODS}


def test_no_band_policy_cannot_win_when_jointly_dominated():
    records = pair(REPLAY) + pair(EWC)
    records += pair(FD, strength=2., h=-.1, a=200.)
    records += pair(FD, strength=3., h=-.5, a=150.)
    selected = design.select(design.table(records, cfg()), cfg())[FD]
    assert selected["selected"]["recipe"]["strength"] == 3.
    assert "joint-improvement" in selected["reason"]


def test_table_uses_replica_roots_and_partial_rows_cannot_win():
    records = pair(REPLAY)
    for metric in records[0]["metrics"][1:]:
        metric["base_mse"] = 100.
    refresh(records[0])
    row = design.table(records, cfg())[0]
    assert row["base_harm"] == pytest.approx((7. + .9) / 2)
    assert row["base_harm"] != pytest.approx(math.sqrt((100. + 3.9 ** 2) / 2) - 3.)
    assert design.table(records[:1], cfg()) == []
    partial = design.table(records[:1], cfg(), partial=True)[0]
    assert partial["count"] == 1 and not partial["complete"] and not partial["resource_feasible"]
    with pytest.raises(ValueError, match="complete valid"):
        design.select([partial], cfg())


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(partition="confirmation"),
    lambda r: r.update(pairing_id="unpaired"),
    lambda r: r.update(family="development--other-context"),
    lambda r: r.update(context_key="other-context"),
    lambda r: r.update(novel_auc=0.),
    lambda r: r.update(target=20),
    lambda r: r["metrics"].pop(10),
    lambda r: r.update(learning_seconds=float("nan")),
])
def test_table_rejects_bad_evidence(mutation):
    rows = pair(REPLAY)
    mutation(rows[0])
    with pytest.raises(ValueError):
        design.table(rows, cfg())


def test_pairing_and_failed_replica_are_not_complete_case_filtered():
    records = pair(REPLAY)
    with pytest.raises(ValueError, match="duplicate"):
        design.table(records + [records[0]], cfg())
    records[0]["valid"] = False
    assert not design.table(records, cfg())[0]["valid"]
    rows = pair(REPLAY)
    rows[1].update(plan_hash=rows[0]["plan_hash"], pairing_id=rows[0]["pairing_id"])
    with pytest.raises(ValueError, match="reuse"):
        design.table(rows, cfg())


def test_selection_keeps_joint_improvement_and_labels_diagnostics():
    records = []
    for method in METHODS:
        records += pair(method, ridge=10., h=.9, a=100.)
        records += pair(method, ridge=100., h=.3, a=90.)
    rows = design.table(records, cfg())
    choices = design.select(rows, cfg())
    assert all(c["selected"]["recipe"]["ridge"] == 100. and "joint-improvement" in c["reason"]
               for c in choices.values())
    shortlist = design.shortlist(rows, cfg())
    assert shortlist == design.shortlist(list(reversed(rows)), cfg())
    assert all([r["ridge"] for r in group] == [100., 10.] for group in shortlist.values())
    infeasible = design.select(design.table(records, cfg(), learning_limit=.0001), cfg())
    assert all("resource-infeasible" in c["reason"] for c in infeasible.values())
    unmatched = design.select(design.table([r for m in METHODS for r in pair(m, h=2.)], cfg()), cfg())
    assert all("unmatched" in c["reason"] for c in unmatched.values())


def test_selection_ties_use_cost_then_replay_count_and_numeric_settings():
    records = []
    for m in METHODS:
        records += pair(m, b=16, ridge=10., a=100.)
        if m != EWC:
            records += pair(m, b=4, ridge=10., a=100.)
    rows = design.table(records, cfg())
    assert design.select(rows, cfg())[FD]["selected"]["replay_count"] == 4
    for row in rows:
        if row["replay_count"] == 16:
            row["seconds_per_arrival"] /= 2
    assert design.select(rows, cfg())[FD]["selected"]["replay_count"] == 16


def test_refinement_brackets_first_is_bounded_fair_and_does_not_repeat_seals():
    records = [r for m in METHODS for ridge, h in ((1., 1.5), (100., .2)) for r in pair(m, ridge=ridge, h=h)]
    rows = design.table(records, cfg())
    proposals = design.refinement_proposals(rows, cfg(), [], 2.)
    assert proposals == design.refinement_proposals(list(reversed(rows)), cfg(), [], 2.)
    assert proposals[0]["kind"] == "midpoint"
    assert proposals[0]["recipe"]["ridge"] == pytest.approx(10.)
    assert len(proposals) <= 8
    assert {p["recipe"]["method"] for p in proposals[:3]} == set(METHODS)
    after = design.refinement_proposals(rows, cfg(), [proposals[0]], 2.)
    assert proposals[0]["recipe_key"] not in {p["recipe_key"] for p in after}
    assert design.refinement_proposals(rows, cfg(), proposals * 8, 2.) == []
    two_left = design.refinement_proposals(rows, cfg(refinement_settings=2), [], 2.)
    assert {p["recipe"]["method"] for p in two_left} == {FD, REPLAY}


def test_refinement_does_not_extrapolate_monotonicity_across_observed_reversal():
    records = [r for m in METHODS for ridge, h in ((1., 2.), (10., 3.), (100., 2.5))
               for r in pair(m, ridge=ridge, h=h)]
    proposals = design.refinement_proposals(design.table(records, cfg()), cfg(), [], 2.)
    assert not any(p["kind"] == "edge" for p in proposals)


def test_point_equivalence_actual_time_and_roots_are_within_replica():
    groups = {m: pair(m) for m in METHODS}
    point = analysis.point_equivalence(groups)
    budget = 100 * 1.05 / 2.075
    assert point["matching_budget"] == pytest.approx(budget)
    assert point["learning_time_ratio"] == pytest.approx(.4 * budget / 25.)
    assert point["S"] == pytest.approx(((100 - budget) / 100) / (.4 * budget / 25.))
    assert point["S"] != pytest.approx((1 - budget / 100) / (.4 / .25))
    assert point["fd_observations"] == pytest.approx(10.5 * budget)
    assert point["fd_final_u"] == pytest.approx(.5 + .005 * budget)
    # Non-identical nominal risks distinguish mean roots from root mean.
    for m in METHODS:
        for metric in groups[m][1]["metrics"]:
            metric["base_mse"] *= 4
        refresh(groups[m][1])
    point = analysis.point_equivalence(groups)
    assert point["fd_old_rmse"] == pytest.approx((3.9 + 7.8) / 2)
    assert point["fd_old_rmse"] != pytest.approx(math.sqrt((3.9 ** 2 + 7.8 ** 2) / 2))
    assert point["fd_old_deterioration"] == pytest.approx((.9 + 1.8) / 2)


@pytest.mark.parametrize("curve,target,status", [
    ([3, 4, 5], 3, "target_already_met_at_initialization"),
    ([10, 4, 8], 5, "target_not_reached_at_observed_endpoint"),
    ([10, 0, 0], 9, "crossing_below_first_post_observation_checkpoint"),
    ([10, 8, 5], 5, "valid"),
    ([10, 5, 5, 5], 5, "valid"),
])
def test_crossing_guards(curve, target, status):
    assert analysis.crossing(curve, target)["status"] == status


def test_late_reversal_last_persistent_entry_and_invalid_time():
    point = analysis.crossing([10, 8, 4, 7, 4, 3], 5)
    assert point["downward_crossings"] == pytest.approx([1.75, 3 + 2 / 3])
    assert point["matching_budget"] == pytest.approx(3 + 2 / 3)
    assert point["multiple_crossings"]
    groups = {m: pair(m, time=np.zeros(101)) for m in METHODS}
    assert analysis.point_equivalence(groups)["status"] == "invalid_learning_time_denominator"


def test_context_score_recomputes_context_target_and_does_not_call_final(monkeypatch):
    monkeypatch.setattr(analysis, "analyze", lambda *a, **kw: pytest.fail("point scoring called final inference"))
    records = [r for m in METHODS for r in pair(m)]
    before = deepcopy(records)
    score = design.context_score(records, cfg(), dict(learning_time_limit=1., memory_ceiling_bytes=2000))
    assert score["eligible"] and score["retention_match"] and score["acquisition_gain"] > 0
    assert score["resource_feasible"]
    assert score == design.context_score(records, cfg(), replicas=range(2), learning_limit=1., memory_limit=2000)
    assert set(score["selected"]) == set(METHODS)
    assert score["point"]["S"] is not None and records == before
    alternative = deepcopy(records)
    for r in alternative:
        if r["method"] == REPLAY:
            for metric in r["metrics"][1:]:
                metric["novel_mse"] += 20.
            refresh(r)
    other = design.context_score(alternative, cfg(), {})
    assert other["point"]["target_mse"] == pytest.approx(score["point"]["target_mse"] + 20.)
    assert other["point"]["S"] > score["point"]["S"]
    assert design.context_rank_key(other) < design.context_rank_key(score)


def test_unresolved_or_not_useful_or_harmful_context_cannot_win():
    useful = design.context_score([r for m in METHODS for r in pair(m)], cfg(), {})
    no_crossing = design.context_score([r for m in METHODS for r in pair(m, a=290. if m == FD else 280.)], cfg(), {})
    assert not no_crossing["eligible"] and no_crossing["point"]["S"] is None
    weak_records = [record(m, r, novel=300. + 12 * r - (.02 if m == FD else .01) * np.arange(101))
                    for m in METHODS for r in range(2)]
    weak = design.context_score(weak_records, cfg(), {})
    assert weak["point"]["status"] == "valid" and not weak["eligible"] and not weak["point"]["useful_target"]
    harmful = design.context_score([r for m in METHODS for r in pair(m, h=1.2 if m == FD else .9)], cfg(), {})
    assert not harmful["retention_match"] and not harmful["eligible"]
    overcost = design.context_score([r for m in METHODS for r in pair(m)], cfg(), {"learning_limit": .00001})
    assert not overcost["eligible"]
    assert design.context_rank_key(useful) < design.context_rank_key(no_crossing)


def test_final_analysis_three_controls_joint_bootstrap_and_json_safety():
    records, manifest = cohort()
    original = deepcopy(records)
    result = analysis.analyze(records, manifest, 12, draws=30)
    assert set(result["methods"]) == set(METHODS) and len(result["comparisons"]) == 3
    assert result["G"]["mean"] > 0
    assert result["efficiency"]["retention_qualified"]
    assert result["efficiency"]["bootstrap"]["invalid"] == 0
    assert len(result["methods"][FD]["curves"]["novel"]["mean_mse"]) == 101
    assert result == analysis.analyze(records, manifest, 12, draws=30)
    assert records == original
    json.dumps(result, allow_nan=False)
    sampled = result["efficiency"]["bootstrap"]["draws_detail"][0]
    indices = replica_draw_indices(12, 12, 0)
    # Resampling whole curves jointly recomputes q and both inversion components.
    fd = [r for r in records if r["method"] == FD]
    replay = [r for r in records if r["method"] == REPLAY]
    q = np.mean([replay[i]["metrics"][-1]["novel_mse"] for i in indices])
    expected_n = analysis.crossing(np.mean([[m["novel_mse"] for m in fd[i]["metrics"]] for i in indices], axis=0), q)["matching_budget"]
    assert sampled["target_mse"] == pytest.approx(q)
    assert sampled["matching_budget"] == pytest.approx(expected_n)
    assert sampled["S"] == pytest.approx(sampled["saved_fraction"] / sampled["learning_time_ratio"])


@pytest.mark.parametrize("mutation", [
    lambda r, m: r.pop(),
    lambda r, m: r.append(deepcopy(r[0])),
    lambda r, m: r[0].update(partition="development"),
    lambda r, m: r[0].update(family="confirmation--wrong"),
    lambda r, m: r[0].update(pairing_id="wrong"),
    lambda r, m: r[0].update(valid=False),
    lambda r, m: r[0].update(novel_auc=0.),
    lambda r, m: r[0]["metrics"].pop(20),
    lambda r, m: m.update(delta=100.),
])
def test_final_analysis_rejects_incomplete_or_changed_frozen_design(mutation):
    records, manifest = cohort()
    mutation(records, manifest)
    with pytest.raises(ValueError):
        analysis.analyze(records, manifest, 1, draws=2)


def test_final_rejects_missing_control_undersized_cohort_and_duplicate_plan():
    records, manifest = cohort()
    for count in (7, 9):
        bad = deepcopy(manifest)
        bad["replicas"] = list(range(count))
        reseal(bad)
        with pytest.raises(ValueError, match="8/10/12"):
            analysis.analyze([r for r in records if r["replica"] < count], bad, 1, 2)
    bad = deepcopy(manifest)
    bad["recipes"] = bad["recipes"][:2]
    reseal(bad)
    with pytest.raises(ValueError, match="three"):
        analysis.analyze(records, bad, 1, 2)
    for r in records:
        if r["replica"] == 1:
            r.update(pairing_id="confirmation-plan-0", plan_hash="confirmation-plan-0")
    with pytest.raises(ValueError, match="reuse"):
        analysis.analyze(records, manifest, 1, 2)


def test_explicit_tiny_smoke_parameterizes_N_but_cannot_publish_final_claims():
    records, manifest = cohort()
    records = [r for r in records if r["replica"] < 2]
    for row in records:
        row["metrics"] = row["metrics"][:4]
        row["target"], row["N"] = 3, 3
        summary = summarize_trajectory(row["metrics"], N=3)
        row.update({k: summary[k] for k in analysis.SUMMARY_FIELDS})
    manifest.update(N=3, replicas=[0, 1], smoke=True)
    reseal(manifest)
    result = analysis.analyze(records, manifest, 1, 3)
    assert result["status"] == "smoke_completed" and not result["is_final"]
    assert result["N"] == 3 and result["methods"][FD]["curves"]["n"] == [0, 1, 2, 3]
    assert not result["efficiency"]["evidence_sample_savings"]
    manifest.pop("smoke")
    reseal(manifest)
    with pytest.raises(ValueError, match="100-event"):
        analysis.analyze(records, manifest, 1, 3)


def test_any_invalid_draw_withholds_every_inversion_interval_but_not_full_H(monkeypatch):
    records, manifest = cohort()
    # Point cohort crosses, but a bootstrap containing only replica 0 does not.
    for r in records:
        if r["method"] == FD and r["replica"] == 0:
            for m in r["metrics"][1:]:
                m["novel_mse"] = 299.
            refresh(r)
    monkeypatch.setattr(analysis, "replica_draw_indices", lambda count, seed, d:
        np.zeros(count, dtype=int) if d == 0 else np.arange(count))
    result = analysis.analyze(records, manifest, 3, draws=4)
    efficiency = result["efficiency"]
    assert efficiency["point"]["status"] == "valid"
    assert efficiency["bootstrap"]["invalid"] == 1
    assert efficiency["bootstrap"]["ci"] is None and efficiency["matched_upper_one_sided_95"] is None
    assert not efficiency["retention_qualified"] and not efficiency["evidence_sample_savings"]
    assert result["methods"][FD]["retention"]["ci"] is not None


def test_full_H_certificate_does_not_certify_matched_endpoint_retention():
    records, manifest = cohort()
    for r in records:
        if r["method"] == FD:
            # Low average harm, but a nominal-error spike at the matched budget.
            for m in r["metrics"][1:]:
                m["base_mse"] = 9.
            for m in r["metrics"][48:63]:
                m["base_mse"] = 36.
            refresh(r)
    result = analysis.analyze(records, manifest, 3, draws=20)
    efficiency = result["efficiency"]
    assert efficiency["full_budget_retention_supported"]
    assert not efficiency["matched_budget_retention_supported"] and not efficiency["retention_qualified"]
    assert efficiency["point"]["S"] is not None


def test_full_H_bounds_are_one_sided_and_equivalence_is_not_nonsignificance():
    records, manifest = cohort()
    for r in records:
        if r["method"] == FD:
            h = .1 if r["replica"] < 6 else 1.7
            for m in r["metrics"][1:]:
                m["base_mse"] = (3 + h) ** 2
            refresh(r)
    result = analysis.analyze(records, manifest, 5, draws=100)
    comparison = result["comparisons"][f"{FD}_vs_{REPLAY}"]
    assert comparison["retention_difference"]["ci"][0] < 0 < comparison["retention_difference"]["ci"][1]
    assert not comparison["retention_equivalence_supported"]
    assert not result["methods"][FD]["retention"]["ceiling_supported"]


def test_optional_per_replica_secondary_curves_and_stop_checks():
    records, manifest = cohort()
    for r in records:
        for m in r["metrics"]:
            m["novel_mae"], m["novel_accuracy_10"] = 12., .6
    calls = []
    result = analysis.analyze(records, manifest, 2, draws=101, check=lambda: calls.append(1))
    curves = result["methods"][FD]["curves"]
    assert curves["novel"]["mean_mae"] == [12.] * 101
    assert len(curves["per_replica"]["novel_accuracy_10"]) == 12
    assert len(calls) >= 36 + 2 * 6
    def stop():
        raise InterruptedError("stop requested")
    with pytest.raises(InterruptedError):
        analysis.analyze(records, manifest, 2, draws=3, check=stop)
