"""Deterministic scalar-only Phase 8 tests: no datasets, models or GPU jobs."""

from copy import deepcopy
import json
import math

import numpy as np
import pytest

from mnist_experiment.combined_memory.reproduction import PROTOCOL, analysis as a
from mnist_experiment.fisher_memory_regression.config import Recipe
from mnist_experiment.fisher_memory_stm_capacity import analysis as old


def record(policy, replica=0, context="blur", namespace="fresh", chart=None, method=None):
    chart, method = (a.POLICIES[policy].split("--") if chart is None else (chart, method))
    recipe = Recipe(method, 32, 2, ridge=1. if method == old.FD else math.sqrt(10),
                    strength=3.34056320604949 if method == old.FD else 0.)
    values = {"R2": [16, 14, 11, 9, 8], "R1": [16, 13, 10, 8, 7], "F1": [16, 11, 7, 5, 3]}[policy]
    if context == "no_blur":
        values = [10, 9, 8, 7, 6] if policy == "R1" else [10, 8, 7, 6, 5]
    exposure = np.asarray([0, 4, 16, 24, 40]) * (replica + 1)
    metrics = [dict(n=n, observations=int(exposure[n]), novel_mse=float(v * (1 + .2 * replica)),
                    base_mse=4. if n == 0 else 2.01**2,
                    learning_seconds=float(n * (2 if method == old.FD else 1) * (1 + .1 * replica)),
                    drift_strength=n / 4) for n, v in enumerate(values)]
    return dict(old.summarize_trajectory(metrics, N=4), policy=policy, replica=replica,
        namespace=namespace, context=context, family="confirmation--grid", partition="confirmation",
        context_key="grid", output_dim=54, target=4, stm_config=chart, arm=f"{chart}--{method}",
        method=method, recipe=recipe.mapping(), recipe_key=recipe.key, capacity=32, replay_count=2,
        metrics=metrics, plan_hash=f"plan-{replica}", pairing_id=f"plan-{replica}",
        chart_hash=chart, source_hash="source", config_hash="config", active_seconds=10.,
        peak_vram_bytes=0, retired_audit=[])


def cohort(replicas=2, contexts=False):
    return [record(p, r, c) for r in range(replicas)
            for c in (("blur", "no_blur") if contexts else ("blur",))
            for p in (a.POLICIES if c == "blur" else ("R1", "F1"))]


def replace_curve(row, field, values):
    for point, value in zip(row["metrics"], values, strict=True):
        point[field] = float(value)
    row.update(old.summarize_trajectory(row["metrics"], N=4))


def run(rows=None, **kwargs):
    return a.analyze(cohort() if rows is None else rows, draws=kwargs.pop("draws", 50),
                     seed=kwargs.pop("seed", 17), **kwargs)


def old_cohort():
    return [record("F1" if m == old.FD else "R2" if c == "all_r2" else "R1",
                   r, namespace="historical", chart=c, method=m)
            for c in old.STM_CONFIGS for m in (old.FD, old.REPLAY) for r in range(2)]


def historical_subset(rows):
    reverse = {v: k for k, v in a.POLICIES.items()}
    return [dict(r, policy=reverse[r["arm"]]) for r in rows if r["arm"] in reverse]


def test_three_real_policies_without_padding_are_deterministic_finite_and_nonmutating():
    rows = cohort()
    before = deepcopy(rows)
    result = run(rows)
    assert rows == before
    assert result == run(list(reversed(rows)))
    assert result["protocol"] == PROTOCOL
    assert result["headline"] == "F1-R2" and result["secondary"] == "F1-R1"
    assert not result["is_final"]
    assert result["provisional"]
    assert result["blur_interaction"]["status"] == "not_supplied"
    assert result["bootstrap"]["resampling_unit"] == "paired_whole_replica"
    assert set(result["efficiency"]) == set(a.POLICIES)
    point = result["efficiency"]["F1"]["point"]
    assert point["matching_budget"] == pytest.approx(1.75)
    assert point["saved_fraction"] == pytest.approx(.5625)
    assert point["learning_time_ratio"] == pytest.approx(.875)
    assert point["S"] == pytest.approx(.5625 / .875)
    assert point["arm_observations"] == pytest.approx(19.5)
    assert point["arm_digit_pairs"] == 9 * point["arm_observations"]
    assert point["arm_novel_digit_pairs"] == 9 * 1.75
    assert point["arm_image_views"] == 18 * point["arm_observations"]
    assert point["arm_scalar_labels"] == 54 * point["arm_observations"]
    assert result["efficiency"]["F1"]["equal_budget_learning_time_ratio"] == 2
    assert result["contrasts"]["F1-R2"]["acquisition_gain"]["favorable_replicas"] == 2
    assert not result["contrasts"]["F1-R1"]["headline_reference"]
    json.dumps(result, allow_nan=False)


def test_independent_subset_matches_frozen_full_eight_arm_analysis_point_and_ci():
    rows = old_cohort()
    seed = a.HISTORICAL_SEED
    frozen = old.analyze(rows, N=4, seed=seed, draws=60)
    result = a.analyze_historical(historical_subset(rows), seed=seed, draws=60)
    for p, name in a.POLICIES.items():
        expected, actual = frozen["arms"][name], result["efficiency"][p]
        for field in old.MATCH_FIELDS:
            assert actual["point"][field] == pytest.approx(expected["efficiency"]["point"][field])
            assert actual["bootstrap"]["ci"][field] == pytest.approx(expected["efficiency"]["bootstrap"]["ci"][field])
        assert result["summaries"]["blur"][p]["acquisition"] == expected["acquisition"]
        assert result["summaries"]["blur"][p]["retention"] == expected["retention"]
    assert result["contrasts"]["F1-R1"]["acquisition_gain"] == frozen["within_size"]["head_r1"]["acquisition_gain"]


def test_historical_audit_verifies_sealed_report_and_is_independent(tmp_path, monkeypatch):
    from mnist_experiment.fisher_memory_stm_capacity.config import Config

    assert a.HISTORICAL_SEED == Config().seed_for("confirmation--grid", 0, "bootstrap")
    rows = old_cohort()
    report = old.analyze(rows, N=4, seed=a.HISTORICAL_SEED, draws=40)
    report["is_final"] = True
    path = tmp_path / "report_snapshot.json"
    path.write_text(json.dumps(dict(analysis=report)))

    def forbidden(*args, **kwargs):
        raise AssertionError("historical headline delegated to frozen arithmetic")

    for module, name in ((old, "analyze"), (old, "_match"), (old.grid, "_point"), (old.grid, "crossing")):
        monkeypatch.setattr(module, name, forbidden)
    result = a.historical_audit(historical_subset(rows), tmp_path)
    assert result["status"] == "passed" and result["checks"] > 100
    report["arms"][a.POLICIES["F1"]]["efficiency"]["bootstrap"]["ci"]["S"][0] += .01
    path.write_text(json.dumps(dict(analysis=report)))
    with pytest.raises(ValueError, match="historical statistics mismatch"):
        a.historical_audit(historical_subset(rows), tmp_path)
    with pytest.raises(ValueError, match="settings/seed"):
        a.historical_audit(historical_subset(rows), tmp_path, seed=1)
    path.unlink()
    with pytest.raises(ValueError, match="historical report"):
        a.historical_audit(historical_subset(rows), tmp_path)


@pytest.mark.parametrize("curve,status", [([16, 15, 14, 13, 12], "target_not_reached"),
    ([16, 5, 4, 3, 2], "below_first")])
def test_unsupported_inverse_never_extrapolates_or_suppresses_risk_intervals(curve, status):
    rows = cohort()
    for row in rows:
        if row["policy"] == "F1":
            replace_curve(row, "novel_mse", np.asarray(curve) * (1 + .2 * row["replica"]))
    result = run(rows)
    item = result["efficiency"]["F1"]
    assert status in item["point"]["status"]
    assert item["point"]["S"] is None and item["bootstrap"]["ci"] is None
    assert item["bootstrap"]["invalid"] == 50
    assert len(item["bootstrap"]["invalid_draws"]) == 50
    assert result["contrasts"]["F1-R2"]["acquisition_gain"]["ci"] is not None
    assert result["efficiency"]["R1"]["bootstrap"]["ci"] is not None


def test_already_met_and_weak_reference_targets_are_retained_not_zero_savings():
    rows = cohort()
    for row in rows:
        if row["policy"] == "R2":
            replace_curve(row, "novel_mse", np.asarray([16, 15, 17, 17, 17]) * (1 + .2 * row["replica"]))
    result = run(rows)
    point = result["efficiency"]["F1"]["point"]
    assert point["status"] == "target_already_met_at_initialization"
    assert point["S"] is None and not point["useful_target"]
    assert result["efficiency"]["F1"]["bootstrap"]["weak_target_draws"] == 50
    assert result["efficiency"]["R2"]["point"]["S"] == 0


def test_zero_matched_learning_time_is_explicit_and_never_nonfinite_json():
    rows = cohort()
    for row in rows:
        if row["policy"] == "F1":
            replace_curve(row, "learning_seconds", [0, 0, 0, 0, 0])
    result = run(rows)
    assert result["efficiency"]["F1"]["point"]["status"] == "invalid_learning_time_denominator"
    assert result["efficiency"]["F1"]["bootstrap"]["ci"] is None
    json.dumps(result, allow_nan=False)


def test_one_invalid_draw_withholds_entire_inverse_ci_without_deleting_draws():
    rows = cohort()
    row = next(r for r in rows if r["policy"] == "F1" and r["replica"] == 1)
    replace_curve(row, "novel_mse", [19.2, 17, 15, 13, 11])
    result = run(rows)
    item = result["efficiency"]["F1"]
    assert item["point"]["status"] == "valid"
    assert 0 < item["bootstrap"]["invalid"] < 50
    assert sum(item["bootstrap"]["statuses"].values()) == 50
    assert item["bootstrap"]["ci"] is None
    assert not item["retention_qualified"]


def test_last_crossing_diagnostics_and_noisy_reference_zero_by_definition():
    rows = cohort()
    for row in rows:
        values = [16, 10, 7, 10, 6] if row["policy"] == "F1" else [16, 6, 10, 7, 8]
        if row["policy"] in ("F1", "R2"):
            replace_curve(row, "novel_mse", np.asarray(values) * (1 + .2 * row["replica"]))
    result = run(rows)
    point = result["efficiency"]["F1"]["point"]
    assert point["matching_budget"] == 3.5
    assert point["first_crossing"] == pytest.approx(1 + 2 / 3)
    assert point["multiple_crossings"]
    reference = result["efficiency"]["R2"]
    assert reference["point"]["reference_by_definition"]
    assert reference["point"]["S"] == reference["point"]["saved_fraction"] == 0
    assert reference["bootstrap"]["ci"]["S"] == [0, 0]


@pytest.mark.parametrize("base,full,matched", [([4, 5.29, 5.29, 3.61, 3.61], True, False),
    ([4, 4.0401, 4.0401, 9, 9], False, True), ([4] + [1.7**2]*4, True, True)])
def test_retention_requires_full_and_matched_one_sided_checks(base, full, matched):
    rows = cohort()
    for row in rows:
        if row["policy"] == "F1":
            replace_curve(row, "base_mse", base)
    item = run(rows)["efficiency"]["F1"]
    assert item["full_budget_retention_supported"] is full
    assert item["matched_budget_retention_supported"] is matched
    assert item["retention_qualified"] is (full and matched)


def test_retention_roots_within_replica_and_uses_upper_uncertainty_ceiling():
    rows = cohort()
    for row in rows:
        if row["policy"] == "F1":
            replace_curve(row, "base_mse", [4, 1, 9, 16, 4] if row["replica"] == 0 else [4, 4, 16, 36, 8])
    summary = run(rows)["summaries"]["blur"]["F1"]
    assert summary["retention"]["mean"] == pytest.approx((math.sqrt(7.5) + 4) / 2 - 2)
    rows = cohort()
    replace_curve(next(r for r in rows if r["policy"] == "F1" and r["replica"] == 1), "base_mse", [4] + [2.7**2]*4)
    retention = run(rows)["summaries"]["blur"]["F1"]["retention"]
    assert retention["point_ceiling_feasible"] and not retention["ceiling_supported"]


def test_four_path_interaction_uses_replica_horizons_native_panels_and_joint_draws():
    rows = cohort(contexts=True)
    result = run(rows)
    interaction = result["blur_interaction"]
    assert interaction["status"] == "available"
    expected = {}
    for c in ("blur", "no_blur"):
        expected[c] = []
        for r in range(2):
            q = {}
            for p in ("F1", "R1"):
                metrics = next(v["metrics"] for v in rows if (v["context"], v["policy"], v["replica"]) == (c, p, r))
                q[p] = sum((right["novel_mse"] + left["novel_mse"]) / 2
                    * (right["observations"] - left["observations"])
                    for left, right in zip(metrics, metrics[1:])) / metrics[-1]["observations"]
            expected[c].append(q["F1"] - q["R1"])
    differences = np.asarray(expected["blur"]) - expected["no_blur"]
    draws = [differences[a.replica_draw_indices(2, 17, i)].mean() for i in range(50)]
    assert interaction["interaction"]["mean"] == pytest.approx(differences.mean())
    assert interaction["interaction"]["ci"] == pytest.approx(np.quantile(draws, [.025, .975]))
    assert result["curves"]["no_blur"]["F1"]["axis"] == "total_integrated_grids"
    assert "n" not in result["curves"]["no_blur"]["F1"]
    assert "no_blur" not in result["efficiency"]
    assert "acquisition" not in result["summaries"]["no_blur"]["F1"]
    assert "saved_fraction" not in json.dumps(result["blur_interaction"])
    assert "S" not in result["summaries"]["no_blur"]["F1"]
    assert "context-native" in interaction["interpretation"]


def test_partial_blocks_are_explicit_and_no_blur_subset_never_creates_interaction():
    rows = cohort(replicas=3)[:-1]
    with pytest.raises(ValueError, match="incomplete paired blur"):
        run(rows)
    result = run(rows, partial=True)
    assert result["replicas"] == [0, 1]
    assert result["cohort"]["excluded_blur_blocks"] == [dict(replica=2, missing=["F1"], incomplete=[])]
    incomplete = cohort(replicas=3)
    incomplete[-1]["metrics"].pop()
    result = run(incomplete, partial=True)
    assert result["cohort"]["excluded_blur_blocks"][0]["incomplete"] == ["F1"]
    rows = cohort(contexts=True)[:-1]
    with pytest.raises(ValueError, match="incomplete paired no_blur"):
        run(rows)
    result = run(rows, partial=True)
    assert result["blur_interaction"]["status"] == "incomplete_paired_blocks"
    assert set(result["curves"]) == {"blur"}
    assert result["cohort"]["pending_no_blur_blocks"] == [dict(replica=1, policies=["F1"])]


@pytest.mark.parametrize("field,value", [("namespace", "historical"), ("replica", True),
    ("arm", a.POLICIES["F1"]), ("valid", False), ("novel_auc", 999.), ("protocol", "wrong"),
    ("chart_hash", "wrong"), ("recipe_key", "wrong"), ("source_hash", "wrong")])
def test_invalid_evidence_is_rejected_even_in_partial_mode(field, value):
    rows = cohort()
    rows[0][field] = value
    with pytest.raises(ValueError):
        run(rows, partial=True)


@pytest.mark.parametrize("field", ["plan_hash", "observations", "drift_strength"])
def test_cross_context_pairing_validation(field):
    rows = cohort(contexts=True)
    for row in rows:
        if row["context"] != "no_blur":
            continue
        if field == "plan_hash":
            row["plan_hash"] = row["pairing_id"] = "new-" + str(row["replica"])
        elif field == "observations":
            for p in row["metrics"][1:]:
                p[field] += 1
            row[field] += 1
        else:
            row["metrics"][1][field] += .01
    with pytest.raises(ValueError, match="cross-context"):
        run(rows)


def test_duplicate_bad_counts_risk_and_time_fail_clearly():
    rows = cohort()
    with pytest.raises(ValueError, match="duplicate"):
        run(rows + [deepcopy(rows[0])])
    for field, value in (("n", 3), ("observations", 0), ("novel_mse", float("nan")), ("learning_seconds", -1)):
        changed = deepcopy(rows)
        changed[0]["metrics"][1][field] = value
        with pytest.raises(ValueError):
            run(changed)


def test_bootstrap_chunks_resume_identically_bind_inputs_and_validate_payload(monkeypatch):
    blocks = {}

    def stop(start, block):
        blocks[start] = deepcopy(block)
        raise InterruptedError("sealed first chunk")

    with pytest.raises(InterruptedError):
        run(draws=125, save_bootstrap=stop)
    assert len(blocks[0]["values"]) == 100
    original, computed = a.bootstrap_chunk, []

    def compute(*args, **kwargs):
        computed.append(kwargs["start"])
        return original(*args, **kwargs)

    monkeypatch.setattr(a, "bootstrap_chunk", compute)
    resumed = run(draws=125, load_bootstrap=blocks.get,
                  save_bootstrap=lambda start, block: blocks.update({start: deepcopy(block)}))
    assert computed == [100]
    assert resumed == run(draws=125)
    changed = cohort()
    changed[0]["active_seconds"] += 1
    with pytest.raises(ValueError, match="checkpoint"):
        run(changed, draws=125, load_bootstrap=blocks.get)
    blocks[0]["values"][0]["arms"]["F1"]["S"] += 1
    with pytest.raises(ValueError, match="corrupt"):
        run(draws=125, load_bootstrap=blocks.get)
    arrays = {p: {f: np.asarray([[m[f] for m in row["metrics"]] for row in cohort() if row["policy"] == p])
                  for f in a.FIELDS} for p in a.POLICIES}
    whole = original(arrays, start=0, stop=7, seed=31)
    split = original(arrays, start=0, stop=3, seed=31) + original(arrays, start=3, stop=7, seed=31)
    assert whole == split
    assert all(v["indices"] == a.replica_draw_indices(2, 31, v["draw"]).tolist() for v in whole)


def test_check_callback_and_invalid_arguments():
    def stop():
        raise InterruptedError("paused")

    with pytest.raises(InterruptedError):
        run(check=stop)
    for settings in (dict(draws=1), dict(draws=True), dict(seed=-1), dict(partial="yes")):
        with pytest.raises(ValueError):
            run(**settings)
