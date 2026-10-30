"""Synthetic scalar trajectories only: no MNIST, learners, GPU or integrations."""

from copy import deepcopy
import json
import math

import numpy as np
import pytest

from mnist_experiment.fisher_memory_regression.config import Recipe
from mnist_experiment.fisher_memory_stm_capacity import analysis as a


def record(chart, method, replica=0, partition="confirmation", cost=None, harm=.01):
    recipe = Recipe(method, 32, 2, strength=.5 if method == a.FD else 0.)
    novel = [16., 11., 7., 5., 3.] if method == a.FD else [16., 13., 10., 8., 7.]
    if f"{chart}--{method}" == a.REFERENCE:
        novel = [16., 14., 11., 9., 8.]
    cost = (2. if method == a.FD else 1.) if cost is None else cost
    metrics = [dict(n=n, observations=(10+replica)*n, learning_seconds=cost*(1+.1*replica)*n,
        base_mse=4. if n == 0 else (2+harm)**2, novel_mse=risk*(1+.2*replica),
        u=.5+.1*n, p=.025*n, blur=.6*n,
        base_mae=1., novel_mae=2., base_accuracy1=.4, novel_accuracy1=.3)
        for n, risk in enumerate(novel)]
    return dict(a.summarize_trajectory(metrics, N=4), family=partition+"--grid", partition=partition,
        context_key="grid", output_dim=54, target=4, replica=replica, stm_config=chart,
        arm=f"{chart}--{method}", recipe=recipe.mapping(), recipe_key=recipe.key, method=method,
        capacity=32, replay_count=2, metrics=metrics, plan_hash=f"{partition}-{replica}",
        pairing_id=f"{partition}-{replica}", source_hash="source", config_hash="configuration",
        chart_hash=chart, active_seconds=10., peak_vram_bytes=0, retired_audit=[])


def cohort(partition="confirmation", replicas=2):
    return [record(chart, method, r, partition) for chart in a.STM_CONFIGS
            for method in (a.FD, a.REPLAY) for r in range(replicas)]


def replace_curve(row, field, values):
    for point, value in zip(row["metrics"], values, strict=True):
        point[field] = value
    row.update(a.summarize_trajectory(row["metrics"], N=4))


def run(rows=None, **kwargs):
    return a.analyze(cohort() if rows is None else rows, N=4, draws=kwargs.pop("draws", 80), seed=17, **kwargs)


def test_eight_arms_share_common_reference_and_replay_size_controls_have_headline_scores():
    rows = cohort()
    before = deepcopy(rows)
    result = run(rows)
    assert rows == before
    assert set(result["arms"]) == set(a.ARMS)
    assert result["reference_arm"] == a.REFERENCE
    assert result["incumbent_arm"] == a.INCUMBENT
    point = result["arms"][a.INCUMBENT]["efficiency"]["point"]
    assert point["matching_budget"] == pytest.approx(1.75)
    assert point["saved_fraction"] == pytest.approx(.5625)
    assert point["learning_time_ratio"] == pytest.approx(.875)
    assert point["S"] == pytest.approx(.5625/.875)
    assert point["arm_novel_digit_pairs"] == 9*point["matching_budget"]
    assert point["arm_digit_pairs"] == 9*point["arm_observations"]
    assert point["arm_image_views"] == 18*point["arm_observations"]
    assert point["arm_scalar_labels"] == 54*point["arm_observations"]
    replay = result["arms"][f"all_r1--{a.REPLAY}"]["efficiency"]["point"]
    assert replay["matching_budget"] == pytest.approx(3.)
    assert replay["S"] > 0
    assert result["bootstrap"]["shared_across"] == list(a.ARMS)
    assert not result["is_final"]
    json.dumps(result, allow_nan=False)


def test_noisy_reference_is_zero_by_definition_not_self_inverted():
    rows = cohort()
    for row in rows:
        if row["arm"] == a.REFERENCE:
            replace_curve(row, "novel_mse", np.asarray([16, 6, 10, 7, 8])*(1+.2*row["replica"]))
    result = run(rows)
    reference = result["arms"][a.REFERENCE]["efficiency"]
    assert reference["point"]["reference_by_definition"]
    assert reference["point"]["matching_budget"] == 4
    assert reference["point"]["S"] == reference["point"]["saved_fraction"] == 0
    assert reference["bootstrap"]["ci"]["S"] == [0, 0]
    assert reference["bootstrap"]["invalid"] == 0


def test_last_sustained_crossing_preserves_multiple_entries_without_smoothing():
    rows = cohort()
    for row in rows:
        if row["arm"] == f"head_r1--{a.FD}":
            replace_curve(row, "novel_mse", np.asarray([16, 10, 7, 10, 6])*(1+.2*row["replica"]))
    efficiency = run(rows)["arms"][f"head_r1--{a.FD}"]["efficiency"]
    point = efficiency["point"]
    assert point["matching_budget"] == pytest.approx(3.5)
    assert point["downward_crossings"] == pytest.approx([1+2/3, 3.5])
    assert point["multiple_crossings"]
    assert efficiency["bootstrap"]["multiple_crossing_draws"] == 80
    assert len(efficiency["bootstrap"]["multiple_crossings"]) == 80


@pytest.mark.parametrize("curve,status", [([16, 15, 14, 13, 12], "target_not_reached"),
    ([16, 5, 4, 3, 2], "below_first"), ([16, 7, 12, 8, 9], "target_not_reached")])
def test_inverse_failures_are_not_zero_savings_or_extrapolated(curve, status):
    rows = cohort()
    for row in rows:
        if row["arm"] == f"head_r1--{a.FD}":
            replace_curve(row, "novel_mse", np.asarray(curve)*(1+.2*row["replica"]))
    item = run(rows)["arms"][f"head_r1--{a.FD}"]["efficiency"]
    assert status in item["point"]["status"]
    assert item["point"]["S"] is None
    assert item["bootstrap"]["ci"] is None
    assert item["bootstrap"]["invalid"] == 80


def test_target_already_met_at_initialization_and_weak_target_diagnostic():
    rows = cohort()
    for row in rows:
        if row["arm"] == a.REFERENCE:
            replace_curve(row, "novel_mse", np.asarray([16, 15, 17, 17, 17])*(1+.2*row["replica"]))
    result = run(rows)
    item = result["arms"][a.INCUMBENT]["efficiency"]
    assert item["point"]["status"] == "target_already_met_at_initialization"
    assert item["point"]["useful_target"] is False
    assert result["arms"][a.REFERENCE]["efficiency"]["point"]["S"] == 0
    assert item["bootstrap"]["invalid_draws"][0]["target_mse"] > 0


def test_one_unresolved_arm_withholds_only_its_inverse_intervals_and_affected_differences():
    rows = cohort()
    arm = f"head_r1--{a.FD}"
    row = next(r for r in rows if r["arm"] == arm and r["replica"] == 1)
    replace_curve(row, "novel_mse", [19.2, 17, 15, 13, 11])
    result = run(rows, draws=120)
    efficiency = result["arms"][arm]["efficiency"]
    assert efficiency["point"]["status"] == "valid"
    assert 0 < efficiency["bootstrap"]["invalid"] < 120
    assert efficiency["bootstrap"]["ci"] is None
    assert len(efficiency["bootstrap"]["invalid_draws"]) == efficiency["bootstrap"]["invalid"]
    assert result["incumbent_differences"][arm]["S_difference"]["ci"] is None
    good = f"all_r4--{a.FD}"
    assert result["arms"][good]["efficiency"]["bootstrap"]["ci"] is not None
    assert result["incumbent_differences"][good]["S_difference"]["ci"] is not None
    assert result["arms"][arm]["acquisition"]["ci"] is not None
    assert result["common_reference_comparisons"][arm]["acquisition_gain"]["ci"] is not None
    assert sum(efficiency["bootstrap"]["statuses"].values()) == 120


def test_reference_target_and_S_difference_recomputed_on_identical_draws():
    rows = cohort()
    arm = f"all_r1--{a.FD}"
    for row in rows:
        if row["arm"] == a.REFERENCE and row["replica"] == 1:
            replace_curve(row, "novel_mse", [19.2, 17, 16, 15, 12])
        if row["arm"] == arm:
            values = [16, 11, 8, 6, 4] if row["replica"] == 0 else [19.2, 16, 12, 7, 3]
            replace_curve(row, "novel_mse", values)
    result = run(rows, draws=90)
    expected = []
    for draw in range(90):
        indices = a.replica_draw_indices(2, 17, draw)
        points = {}
        for name in (arm, a.INCUMBENT):
            group = sorted((r for r in rows if r["arm"] == name), key=lambda r: r["replica"])
            reference = sorted((r for r in rows if r["arm"] == a.REFERENCE), key=lambda r: r["replica"])
            target = np.mean([reference[i]["metrics"][-1]["novel_mse"] for i in indices])
            curve = np.mean([[p["novel_mse"] for p in group[i]["metrics"]] for i in indices], axis=0)
            n = a.grid.crossing(curve, target)["matching_budget"]
            seconds = np.mean([[p["learning_seconds"] for p in group[i]["metrics"]] for i in indices], axis=0)
            denominator = np.mean([reference[i]["learning_seconds"] for i in indices])
            points[name] = ((4-n)/4)/(np.interp(n, np.arange(5), seconds)/denominator)
        expected.append(points[arm]-points[a.INCUMBENT])
    assert result["incumbent_differences"][arm]["S_difference"]["ci"] == pytest.approx(np.quantile(expected, [.025, .975]))


def test_full_acquisition_and_within_size_contrasts_keep_their_own_paired_estimands():
    result = run()
    arm = f"head_r1--{a.FD}"
    gain = result["common_reference_comparisons"][arm]["acquisition_gain"]
    within = result["within_size"]["head_r1"]
    assert gain["mean"] != within["acquisition_gain"]["mean"]
    assert gain["favorable_replicas"] == 2
    assert within["acquisition_gain"]["favorable_replicas"] == 2
    assert within["headline_reference"] is False
    assert result["arms"][arm]["per_replica"]["learning_seconds"] == [8., 8.8]
    assert result["arms"][a.REFERENCE]["per_replica"]["learning_seconds"] == [4., 4.4]
    assert result["arms"][arm]["efficiency"]["equal_budget_learning_time_ratio"] == 2


def test_retention_roots_within_replicas_not_after_averaging():
    rows = cohort()
    arm = f"head_r1--{a.FD}"
    for row in rows:
        if row["arm"] == arm:
            replace_curve(row, "base_mse", [4, 1, 9, 16, 4] if row["replica"] == 0 else [4, 4, 16, 36, 8])
    item = run(rows)["arms"][arm]
    expected = (math.sqrt(7.5)+math.sqrt(16))/2-2
    assert item["retention"]["mean"] == pytest.approx(expected)
    assert item["retention"]["mean"] != pytest.approx(math.sqrt((7.5+16)/2)-2)


def test_retention_tolerance_is_one_sided_not_symmetric_equivalence():
    rows = cohort()
    for row in rows:
        if row["method"] == a.FD:
            replace_curve(row, "base_mse", [4, 1.7**2, 1.7**2, 1.7**2, 1.7**2])
    result = run(rows)
    item = result["arms"][a.INCUMBENT]["efficiency"]
    comparison = result["common_reference_comparisons"][a.INCUMBENT]
    assert comparison["retention_difference"]["ci"][0] < -.1
    assert comparison["retention_noninferiority_supported"]
    assert item["full_budget_retention_supported"]
    assert item["matched_budget_retention_supported"]
    assert item["retention_qualified"]


@pytest.mark.parametrize("base,full,matched", [([4, 5.29, 5.29, 3.61, 3.61], True, False),
    ([4, 4.0401, 4.0401, 9, 9], False, True)])
def test_full_and_matched_retention_are_both_required(base, full, matched):
    rows = cohort()
    for row in rows:
        if row["arm"] == a.INCUMBENT:
            replace_curve(row, "base_mse", base)
    item = run(rows)["arms"][a.INCUMBENT]["efficiency"]
    assert item["full_budget_retention_supported"] is full
    assert item["matched_budget_retention_supported"] is matched
    assert not item["retention_qualified"]


def test_ceiling_uses_one_sided_uncertainty_not_only_point_feasibility():
    rows = cohort()
    row = next(r for r in rows if r["arm"] == a.INCUMBENT and r["replica"] == 1)
    replace_curve(row, "base_mse", [4]+[2.7**2]*4)
    item = run(rows)["arms"][a.INCUMBENT]
    assert item["retention"]["point_ceiling_feasible"]
    assert item["retention"]["upper_one_sided_95"] > .5
    assert not item["retention"]["ceiling_supported"]


@pytest.mark.parametrize("field,value,message", [("protocol", "plan15-grid-registration-v1", "protocol"),
    ("arm", a.INCUMBENT, "arm"), ("stm_config", "missing", "chart"), ("output_dim", 6, "output_dim"),
    ("family", "confirmation--vector", "family"), ("partition", "development", "confirmation"),
    ("recipe_key", "wrong", "recipe"), ("chart_hash", "wrong", "chart_hash"),
    ("config_hash", "wrong", "config_hash"), ("source_hash", "wrong", "source_hash"),
    ("stm_parameters", 344, "chart"), ("target", 3, "budget"),
    ("valid", False, "invalid trajectory"), ("novel_auc", 999., "summary")])
def test_incompatible_identity_or_invalid_evidence_fails_without_exclusion(field, value, message):
    rows = cohort()
    rows[0][field] = value
    with pytest.raises(ValueError, match=message):
        run(rows)


def test_missing_duplicate_unplanned_and_changed_policy_replicas_are_rejected():
    rows = cohort()
    for changed in (rows[:-1], rows+[deepcopy(rows[0])]):
        with pytest.raises(ValueError, match="paired|duplicate"):
            run(changed)
    with pytest.raises(ValueError, match="cohort"):
        run(rows, replicas=[0, 2])
    changed = deepcopy(rows)
    recipe = Recipe(a.FD, 32, 2, ridge=10, strength=.5)
    changed[0].update(recipe=recipe.mapping(), recipe_key=recipe.key)
    with pytest.raises(ValueError, match="single-policy"):
        run(changed)


@pytest.mark.parametrize("field", ["plan", "frozen", "observations", "drift"])
def test_cross_chart_pairing_is_checked_not_only_within_chart(field):
    rows = cohort()
    for row in rows:
        if row["stm_config"] != "all_r4":
            continue
        if field == "plan":
            row["plan_hash"] = row["pairing_id"] = "other-"+str(row["replica"])
        elif field == "frozen":
            replace_curve(row, "novel_mse", [row["frozen_novel_mse"]+1]+[p["novel_mse"] for p in row["metrics"][1:]])
        elif field == "observations":
            for point in row["metrics"][1:]:
                point["observations"] += 1
            row["observations"] += 1
        else:
            row["metrics"][2]["u"] += .01
    with pytest.raises(ValueError, match="cross-chart"):
        run(rows)


def test_wrapper_records_development_partition_and_deterministic_input_order():
    rows = cohort("development")
    wrappers = [dict(summary={k: v for k, v in r.items() if k != "metrics"}, metrics=r["metrics"]) for r in rows]
    first = run(wrappers, partition="development")
    assert first == run(list(reversed(rows)), partition="development")
    assert not first["is_final"]


def test_retired_training_cohorts_remain_separate_from_holdout_risk():
    rows = cohort()
    for row in rows:
        row["retired_audit"] = [dict(n=3, exposure=7, squared_error=4+row["replica"])]
    result = run(rows)
    diagnostic = result["arms"][a.INCUMBENT]["retired_training_cohort"]
    assert diagnostic == dict(n=[3], mean_mse=[4.5], n_replicas=[2], held_out=False)
    assert result["arms"][a.INCUMBENT]["curves"]["base"]["mean_mse"][-1] == pytest.approx(2.01**2)


def test_small_results_do_not_store_perdraw_curve_tensors():
    result = run(draws=120)
    for item in result["arms"].values():
        assert len(item["curves"]["novel"]["mean_mse"]) == 5
        assert "draws_detail" not in item["efficiency"]["bootstrap"]
        assert "per_replica" not in item["curves"]
    assert len(json.dumps(result)) < 160_000


def test_checkpointed_chunks_resume_exactly_and_input_or_payload_changes_are_rejected(monkeypatch):
    rows, blocks = cohort(), {}

    def stop(start, block):
        blocks[start] = deepcopy(block)
        raise InterruptedError("safe pause after sealed block")

    with pytest.raises(InterruptedError):
        run(rows, draws=130, save_bootstrap=stop)
    assert len(blocks[0]["values"]) == 100
    original, computed = a.bootstrap_chunk, []

    def compute(*args, **kwargs):
        computed.append(kwargs["start"])
        return original(*args, **kwargs)

    monkeypatch.setattr(a, "bootstrap_chunk", compute)
    resumed = run(rows, draws=130, load_bootstrap=blocks.get,
                  save_bootstrap=lambda start, block: blocks.update({start: deepcopy(block)}))
    assert computed == [100]
    assert resumed == run(rows, draws=130)
    changed = deepcopy(rows)
    changed[0]["active_seconds"] += 1
    with pytest.raises(ValueError, match="checkpoint"):
        run(changed, draws=130, load_bootstrap=blocks.get)
    blocks[0]["values"][0]["arms"][a.INCUMBENT]["S"] = 999
    with pytest.raises(ValueError, match="corrupt"):
        run(rows, draws=130, load_bootstrap=blocks.get)


def test_named_draws_are_independent_of_chunk_boundaries():
    groups, _ = a._cohort(cohort(), 4, "confirmation", None, lambda: None)
    arrays = a.grid._arrays(groups)
    whole = a.bootstrap_chunk(arrays, start=0, stop=8, seed=23)
    split = a.bootstrap_chunk(arrays, start=0, stop=3, seed=23)+a.bootstrap_chunk(arrays, start=3, stop=8, seed=23)
    assert whole == split
    assert all(set(d["arms"]) == set(a.ARMS) for d in whole)


def test_check_callback_can_interrupt_without_launching_any_computation():
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 50:
            raise InterruptedError("paused")

    with pytest.raises(InterruptedError):
        run(check=check)
    assert calls == 50


@pytest.mark.parametrize("kwargs", [dict(draws=1), dict(draws=True), dict(seed=-1), dict(N=True),
    dict(delta=-1), dict(epsilon=float("nan")), dict(partition="unknown"), dict(replicas=[0, 0])])
def test_invalid_analysis_arguments_fail_clearly(kwargs):
    settings = dict(N=4, draws=20, seed=17)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        a.analyze(cohort(), **settings)
