"""Scalar-only six-policy tests; no training, datasets or GPU execution."""

from copy import deepcopy
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from mnist_experiment.combined_memory.replicated_combination import analysis as a
from mnist_experiment.fisher_memory_regression.config import Recipe
from mnist_experiment.fisher_memory_stm_capacity import analysis as old


def record(policy, replica=0, *, N=4, namespace="fresh"):
    method = old.REPLAY if policy == "A" else old.FD
    recipe = Recipe(method, 32, 2, ridge=math.sqrt(10) if policy == "A" else 1.,
                    strength=0. if policy == "A" else a.STRENGTH)
    values = {"R": [16, 12, 8, 6, 4], "A": [16, 14, 11, 9, 8],
              "B": [16, 11, 7, 5, 3], "C": [16, 10, 6, 4, 2],
              "D": [16, 12, 8, 6, 4], "E": [16, 9, 5, 3, 1]}[policy]
    curve = np.interp(np.arange(N + 1), np.linspace(0, N, 5), values) * (1 + .2 * replica)
    metrics = [dict(n=n, observations=4 * n * (replica + 1), novel_mse=float(curve[n]),
                    base_mse=4. if n == 0 else 4.02,
                    learning_seconds=float(n * (1 if policy == "A" else 2) * (1 + .1 * replica)),
                    drift_strength=n / N) for n in range(N + 1)]
    return dict(old.summarize_trajectory(metrics, N=N), policy=policy, replica=replica,
        protocol=old.PROTOCOL if policy == "R" else a.PROTOCOL, namespace=namespace,
        representation="low_rank_diagonal" if policy == "R" else "diagonal",
        family="confirmation--grid", partition="confirmation", context_key="grid", output_dim=54,
        target=N, stm_config="head_r1", arm=f"head_r1--{method}", method=method,
        recipe=recipe.mapping(), recipe_key=recipe.key, capacity=32, replay_count=2,
        metrics=metrics, plan_hash=f"plan-{replica}", pairing_id=f"plan-{replica}",
        chart_hash="head-r1", layout_hash="same-layout", source_hash="source", config_hash="config",
        factor_seeds=dict(init=replica, probes=replica + 8), active_seconds=10., peak_vram_bytes=0,
        retired_audit=[])


def cohort(replicas=2, **kwargs):
    return [record(p, r, **kwargs) for r in range(replicas) for p in a.POLICIES]


def replace_curve(row, field, values):
    for point, value in zip(row["metrics"], values, strict=True):
        point[field] = float(value)
    row.update({k: v for k, v in old.summarize_trajectory(row["metrics"], N=row["target"]).items()
                if k not in ("protocol",)})


def run(rows=None, **kwargs):
    return a.analyze(cohort() if rows is None else rows, draws=kwargs.pop("draws", 30),
                     seed=kwargs.pop("seed", 17), **kwargs)


def test_six_real_policies_are_deterministic_nonmutating_and_share_one_clock():
    rows = cohort()
    before = deepcopy(rows)
    result = run(rows)
    assert rows == before
    assert result == run(list(reversed(rows)))
    assert result["protocol"] == "plan18-replicated-combination-v1"
    assert result["replicas"] == [0, 1]
    assert result["provisional"] and not result["is_final"]
    assert set(result["curves"]) == set(a.POLICIES)
    assert set(result["efficiency_vs_B"]) == {"C", "D", "E"}
    assert result["curves"]["E"]["n"] == list(range(5))
    point = result["efficiency"]["B"]["point"]
    assert point["matching_budget"] == pytest.approx(1.75)
    assert point["saved_fraction"] == pytest.approx(.5625)
    assert point["learning_time_ratio"] == pytest.approx(.875)
    assert point["S"] == pytest.approx(.5625 / .875)
    assert point["arm_observations"] == pytest.approx(10.5)
    assert point["arm_digit_pairs"] == pytest.approx(94.5)
    assert point["arm_image_views"] == pytest.approx(189.)
    assert point["arm_scalar_labels"] == pytest.approx(567.)
    assert result["efficiency"]["A"]["point"]["S"] == 0
    assert result["efficiency"]["A"]["point"]["matching_budget"] == 4
    json.dumps(result, allow_nan=False)


def test_paired_contrasts_interaction_sign_and_bootstrap_are_directly_checkable():
    rows = cohort(3)
    result = run(rows, draws=45)
    vectors = {p: np.asarray([np.mean([m["novel_mse"] for m in r["metrics"][1:]])
                             for r in rows if r["policy"] == p]) for p in a.POLICIES}
    difference = vectors["E"] - vectors["C"] - vectors["D"] + vectors["B"]
    expected = [difference[a.replica_draw_indices(3, 17, draw)].mean() for draw in range(45)]
    actual = result["contrasts"]["interaction"]
    assert actual["acquisition_risk_difference"]["mean"] == pytest.approx(difference.mean())
    assert actual["acquisition_risk_difference"]["ci"] == pytest.approx(np.quantile(expected, [.025, .975]))
    assert set(result["contrasts"]) == {"B-R", "C-B", "D-B", "E-B", "B-A", "E-A", "E-R", "interaction"}
    for key, contrast in result["contrasts"].items():
        if key == "interaction":
            continue
        first, second = key.split("-")
        direct = vectors[first] - vectors[second]
        assert contrast["acquisition_risk_difference"]["mean"] == pytest.approx(direct.mean())
        assert contrast["acquisition_gain"]["mean"] == pytest.approx(-direct.mean())
        assert contrast["favorable_replicas"] == int((direct < 0).sum())
    assert len(result["retention_comparisons"]) == 10
    assert result["bootstrap"]["shared_across"] == list(a.POLICIES)


def test_retention_roots_within_replica_and_secondary_clock_uses_actual_arrivals():
    rows = cohort()
    for row in rows:
        replace_curve(row, "base_mse", [4, 9, 9, 9, 9] if row["replica"] == 0 else [16, 16, 16, 16, 16])
    result = run(rows)
    assert result["summaries"]["B"]["retention"]["mean"] == .5
    assert result["summaries"]["B"]["endpoint_retention"]["mean"] == .5
    expected = []
    for row in rows:
        if row["policy"] != "B":
            continue
        points = row["metrics"]
        expected.append(sum((l["novel_mse"] + r["novel_mse"]) / 2
                            * (r["observations"] - l["observations"])
                            for l, r in zip(points, points[1:])) / points[-1]["observations"])
    assert result["summaries"]["B"]["all_arrival_risk"]["mean"] == pytest.approx(np.mean(expected))
    assert result["summaries"]["B"]["retention"]["mean"] != pytest.approx(math.sqrt(12.5) - math.sqrt(10))


def test_matched_retention_interpolates_mse_before_root_and_full_checks_both_metrics():
    rows = cohort()
    for row in rows:
        if row["policy"] == "B":
            replace_curve(row, "base_mse", [4., 4., 9., 4., 4.])
    result = run(rows)
    item = result["efficiency"]["B"]
    assert item["point"]["arm_old_rmse"] == pytest.approx(math.sqrt(7.75))
    assert not item["full_budget_retention_checks"]["retention"]
    assert item["full_budget_retention_checks"]["endpoint_retention"]
    assert not item["full_budget_retention_supported"]
    assert not item["matched_budget_retention_supported"]


def test_last_sustained_crossing_preserves_multiple_entries_and_never_fits():
    point = a.crossing([16, 7, 10, 5, 3], 8)
    assert point["status"] == "valid"
    assert point["matching_budget"] == pytest.approx(2.4)
    assert point["multiple_crossings"]
    assert point["first_crossing"] == pytest.approx(8 / 9)


@pytest.mark.parametrize("curve,status", [([16, 15, 14, 13, 12], "target_not_reached"),
    ([16, 5, 4, 3, 2], "below_first"), ([16, 16, 16, 16, 16], "target_not_reached")])
def test_unsupported_inverses_withhold_ci_but_keep_prediction_statistics(curve, status):
    rows = cohort()
    for row in rows:
        if row["policy"] == "B":
            replace_curve(row, "novel_mse", np.asarray(curve) * (1 + .2 * row["replica"]))
    result = run(rows)
    item = result["efficiency"]["B"]
    assert status in item["point"]["status"]
    assert item["point"]["S"] is None
    assert item["bootstrap"]["invalid"] == 30
    assert item["bootstrap"]["ci"] is None
    assert item["matched_budget_retention_supported"] is None
    assert result["contrasts"]["B-A"]["acquisition_risk_difference"]["ci"] is not None


def test_one_unsupported_draw_withholds_all_inverse_intervals_without_deletion():
    rows = cohort()
    row = next(r for r in rows if r["policy"] == "B" and r["replica"] == 1)
    replace_curve(row, "novel_mse", [19.2, 17, 15, 13, 11])
    result = run(rows, draws=80)
    item = result["efficiency"]["B"]
    assert item["point"]["status"] == "valid"
    assert 0 < item["bootstrap"]["invalid"] < 80
    assert item["bootstrap"]["ci"] is None
    assert sum(item["bootstrap"]["statuses"].values()) == 80


def test_already_met_and_weak_reference_target_are_visible_without_zero_imputation():
    rows = cohort()
    for row in rows:
        if row["policy"] == "A":
            replace_curve(row, "novel_mse", np.asarray([16, 15, 17, 17, 17]) * (1 + .2 * row["replica"]))
    result = run(rows)
    point = result["efficiency"]["B"]["point"]
    assert point["status"] == "target_already_met_at_initialization"
    assert point["S"] is None and not point["useful_target"]
    assert result["efficiency"]["B"]["bootstrap"]["weak_target_draws"] == 30


def test_zero_learning_times_are_an_explicit_status_not_infinite_ratios():
    rows = cohort()
    for row in rows:
        if row["policy"] == "A":
            replace_curve(row, "learning_seconds", [0] * 5)
    result = run(rows)
    assert result["efficiency"]["B"]["point"]["status"] == "invalid_learning_time_denominator"
    assert result["efficiency"]["B"]["equal_budget_learning_time_ratio"] is None
    json.dumps(result, allow_nan=False)


def test_partial_blocks_require_explicit_opt_in_and_never_invent_an_arm():
    rows = cohort(3)[:-1]
    with pytest.raises(ValueError, match="incomplete paired six-policy"):
        run(rows)
    result = run(rows, partial=True)
    assert result["replicas"] == [0, 1]
    assert result["cohort"]["excluded_blocks"] == [dict(replica=2, missing=["E"], incomplete=[])]
    rows = cohort(3)
    rows[-1]["metrics"].pop()
    result = run(rows, partial=True)
    assert result["cohort"]["excluded_blocks"][0]["incomplete"] == ["E"]
    with pytest.raises(ValueError, match="at least two"):
        run(cohort(2)[:-1], partial=True)
    rows = cohort(3)
    rows[-1]["status"] = "running"
    assert run(rows, partial=True)["cohort"]["excluded_blocks"][0]["incomplete"] == ["E"]


@pytest.mark.parametrize("field,value", [("namespace", "development"), ("replica", True),
    ("valid", False), ("protocol", "wrong"), ("novel_auc", 100.), ("recipe_key", "wrong"),
    ("source_hash", "different"), ("layout_hash", "different"), ("representation", "diagonal"),
    ("frozen", False), ("stm_config", "all_r2"), ("factor_seeds", {"different": 1})])
def test_malformed_evidence_cannot_be_hidden_in_partial_mode(field, value):
    rows = cohort(3)
    rows[0][field] = value
    with pytest.raises(ValueError):
        run(rows, partial=True)


@pytest.mark.parametrize("field,value", [("n", 3), ("observations", 0),
    ("novel_mse", float("nan")), ("base_mse", -1.), ("learning_seconds", -1.)])
def test_bad_metric_prefix_is_rejected_even_when_its_block_would_be_excluded(field, value):
    rows = cohort(3)
    rows[-1]["metrics"].pop()
    rows[-1]["metrics"][1][field] = value
    with pytest.raises(ValueError):
        run(rows, partial=True)


@pytest.mark.parametrize("field", ["plan_hash", "observations", "drift_strength"])
def test_pairing_mismatch_raises(field):
    rows = cohort()
    row = rows[-1]
    if field == "plan_hash":
        row[field] = row["pairing_id"] = "other-plan"
    else:
        for point in row["metrics"][1:]:
            point[field] += 1
        replace_curve(row, "base_mse", [m["base_mse"] for m in row["metrics"]])
    with pytest.raises(ValueError, match="unpaired"):
        run(rows)


def test_duplicate_or_unfrozen_recipe_fails_without_arm_uniqueness_assumption():
    rows = cohort()
    assert rows[0]["recipe_key"] == rows[2]["recipe_key"]
    assert rows[0]["arm"] == rows[2]["arm"]
    run(rows)
    with pytest.raises(ValueError, match="duplicate"):
        run(rows + [deepcopy(rows[0])])
    row = rows[-1]
    recipe = replace(Recipe.from_mapping(row["recipe"]), ridge=10.)
    row.update(recipe=recipe.mapping(), recipe_key=recipe.key)
    with pytest.raises(ValueError, match="one frozen recipe"):
        run(rows)


def test_final_status_requires_eight_full_fresh_blocks_and_smoke_is_always_provisional():
    result = run(cohort(8, N=100), draws=2)
    assert result["is_final"] and not result["provisional"]
    result = run(cohort(8, N=100, namespace="acceptance"), draws=2)
    assert not result["is_final"]
    rows = cohort(2, N=3)
    for row in rows:
        recipe = replace(Recipe.from_mapping(row["recipe"]), capacity=2, replay_count=1)
        row.update(recipe=recipe.mapping(), recipe_key=recipe.key, capacity=2, replay_count=1, smoke=True)
    assert run(rows, draws=2)["provisional"]
    for row in rows:
        row.pop("smoke")
    assert run(rows, draws=2)["provisional"]
    rows = cohort(2, N=100)
    for row in rows:
        recipe = replace(Recipe.from_mapping(row["recipe"]), capacity=2, replay_count=1)
        row.update(recipe=recipe.mapping(), recipe_key=recipe.key, capacity=2, replay_count=1, smoke=True)
    with pytest.raises(ValueError):
        run(rows, draws=2)


def test_bootstrap_chunks_resume_exactly_bind_inputs_and_reject_corruption(monkeypatch):
    blocks = {}

    def save_then_stop(start, block):
        blocks[start] = deepcopy(block)
        raise InterruptedError("sealed chunk")

    with pytest.raises(InterruptedError):
        run(draws=125, save_bootstrap=save_then_stop)
    assert len(blocks[0]["values"]) == 100
    original, computed = a.bootstrap_chunk, []

    def tracked(*args, **kwargs):
        computed.append(kwargs["start"])
        return original(*args, **kwargs)

    monkeypatch.setattr(a, "bootstrap_chunk", tracked)
    resumed = run(draws=125, load_bootstrap=blocks.get,
                  save_bootstrap=lambda start, block: blocks.update({start: deepcopy(block)}))
    assert computed == [100]
    assert resumed == run(draws=125)
    changed = cohort()
    changed[0]["active_seconds"] += 1
    with pytest.raises(ValueError, match="checkpoint"):
        run(changed, draws=125, load_bootstrap=blocks.get)
    blocks[0]["values"][0]["references"]["A"]["B"]["S"] += 1
    with pytest.raises(ValueError, match="corrupt"):
        run(draws=125, load_bootstrap=blocks.get)
    arrays = {p: {f: np.asarray([[m[f] for m in row["metrics"]] for row in cohort() if row["policy"] == p])
                  for f in a.FIELDS} for p in a.POLICIES}
    assert original(arrays, start=0, stop=7, seed=31) == (
        original(arrays, start=0, stop=3, seed=31) + original(arrays, start=3, stop=7, seed=31))


def test_wrapped_records_and_cooperative_cancellation():
    rows = cohort()
    wrapped = [dict(summary={k: v for k, v in r.items() if k != "metrics"}, metrics=r["metrics"]) for r in rows]
    assert run(rows) == run(wrapped)

    def stop():
        raise InterruptedError("paused")

    with pytest.raises(InterruptedError):
        run(check=stop)
    for settings in (dict(draws=1), dict(draws=True), dict(seed=-1), dict(partial="yes")):
        with pytest.raises(ValueError):
            run(**settings)


def test_production_solver_settings_and_cross_policy_nuisance_settings_are_fixed():
    rows = cohort(2, N=100)
    changed = replace(Recipe.from_mapping(rows[-1]["recipe"]), fd_step=.02)
    rows[-1].update(recipe=changed.mapping(), recipe_key=changed.key)
    with pytest.raises(ValueError, match="production solve budget"):
        run(rows, draws=2)
    rows = cohort()
    changed = replace(Recipe.from_mapping(rows[-1]["recipe"]), probes=3)
    rows[-1].update(recipe=changed.mapping(), recipe_key=changed.key)
    with pytest.raises(ValueError, match="incompatible cohort identity"):
        run(rows, draws=2)
