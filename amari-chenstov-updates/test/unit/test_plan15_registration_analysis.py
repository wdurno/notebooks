"""Small CPU-only registration artifacts; no MNIST or learner execution."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from mnist_experiment.fisher_memory_regression.config import METHODS, Recipe, content_hash
from mnist_experiment.fisher_memory_registration import PROTOCOL, analysis, design
from mnist_experiment.fisher_memory_registration.config import Config

FD, REPLAY, EWC = METHODS


def cfg(**changes):
    return replace(Config.smoke_config(), ridge_grid=(.1, 1., math.sqrt(10), 10., 100.),
                   strength_ratios=(.1, 1., 10.), refinement_settings=8, **changes)


def record(method, replica=0, *, task="vector", ridge=1., strength=2., h=.2,
           a=None, novel=None, base=None, cost=None, partition="development", N=3):
    dim = 6 if task == "vector" else 1
    recipe = Recipe.from_config(cfg(), method, ridge=ridge, strength=strength if method == FD else 0.)
    novel = np.asarray(novel if novel is not None else ([16., 8., 4., 2.] if method == FD else [16., 12., 10., 8.]), dtype=float)
    if a is not None:
        novel[1:] = a
    base = np.asarray(base if base is not None else [4., *([(2+h)**2] * N)])
    cost = (.6 if method == FD else .3) if cost is None else cost
    metrics = []
    for n in range(N+1):
        coordinates = [float(base[n]) / 2] * dim if dim == 6 else [float(base[n])]
        novel_coordinates = [float(novel[n]) / 2] * dim if dim == 6 else [float(novel[n])]
        metrics.append(dict(n=n, observations=(10+replica)*n, learning_seconds=cost*(10+replica)*n,
            base_mse=float(base[n]), novel_mse=float(novel[n]), drift_strength=n/N,
            base_common_mse=coordinates[0], novel_common_mse=novel_coordinates[0],
            base_mae=math.sqrt(base[n])*.8, novel_mae=math.sqrt(novel[n])*.8,
            base_accuracy1=.4, novel_accuracy1=.3,
            coordinate_mse={"base": coordinates, "novel": novel_coordinates},
            per_digit_mse={"base": {"0": float(base[n])}, "novel": {"0": float(novel[n])}}))
    summary = analysis.summarize_trajectory(metrics, N=N)
    return dict(summary, family=f"{partition}--{task}", partition=partition, context_key=task,
        output_dim=dim, target=N, replica=replica, recipe=recipe.mapping(), recipe_key=recipe.key,
        method=method, capacity=recipe.capacity, replay_count=recipe.replay_count, metrics=metrics,
        plan_hash=f"{partition}-{replica}", pairing_id=f"{partition}-{replica}",
        active_seconds=30., peak_vram_bytes=1000, retired_audit=[])


def pair(method, **kwargs):
    return [record(method, r, **kwargs) for r in range(2)]


def cohort(task="vector"):
    records = [record(m, r, task=task, partition="confirmation") for m in METHODS for r in range(2)]
    manifest = dict(protocol=PROTOCOL, schema_version=1, sealed=True, N=3, smoke=True, replicas=[0, 1],
        context={"key": task, "output_dim": 6 if task == "vector" else 1},
        recipes=[{**next(r["recipe"] for r in records if r["method"] == m),
                  "recipe_key": next(r["recipe_key"] for r in records if r["method"] == m)} for m in METHODS],
        retention_margin=.5, matching_tolerance=.1, envelope_factor=2.5, source_hash="test-source")
    manifest["manifest_hash"] = content_hash(manifest)
    return records, manifest


def refresh(row):
    row.update({k: v for k, v in analysis.summarize_trajectory(row["metrics"], N=row["target"]).items()
                if k in analysis.SUMMARY_FIELDS})


def test_initial_grid_is_25_paired_recipes_not_a_capacity_search():
    recipes = design.initial_grid(Config(), 2.)
    assert len(recipes) == len({r.key for r in recipes}) == 25
    assert Counter(r.method for r in recipes) == {FD: 15, REPLAY: 5, EWC: 5}
    assert {r.capacity for r in recipes if r.method != EWC} == {256}
    assert {r.replay_count for r in recipes if r.method != EWC} == {16}
    assert {r.strength for r in recipes if r.method == FD} == {.2, 2., 20.}
    assert recipes == design.initial_grid(Config(), 2.)
    with pytest.raises(ValueError):
        design.initial_grid(cfg(), 0)


def test_replay_is_best_acquisition_first_not_a_retention_band_or_fd_cost_enabler():
    records = pair(REPLAY, ridge=1, a=5, h=-.3, cost=.1)
    records += pair(REPLAY, ridge=10, a=6, h=.45, cost=.5)
    records += pair(FD, a=3, h=.1, cost=.3) + pair(EWC, a=4, h=-.5)
    selected = design.select(design.table(records, cfg()), cfg())
    assert selected[REPLAY]["selected"]["recipe"]["ridge"] == 1
    assert selected[EWC]["selected"]["base_harm"] == pytest.approx(-.5)
    assert not selected[FD]["eligible"]
    assert set(selected[FD]["failed_conditions"]) == {"selected_replay_cost", "retention_noninferiority"}


def test_fd_relative_allowance_and_independent_ewc_selection():
    records = pair(REPLAY, h=.1, a=7, cost=.3) + pair(FD, h=.2, a=5, cost=.75)
    records += pair(FD, ridge=10, h=-.2, a=4, cost=.76)
    records += pair(EWC, h=.45, a=3, cost=.05) + pair(EWC, ridge=10, h=.1, a=4)
    choice = design.select(design.table(records, cfg()), cfg())
    assert choice[FD]["eligible"]
    assert choice[FD]["selected"]["recipe"]["ridge"] == 1
    assert choice[EWC]["selected"]["recipe"]["ridge"] == 1
    assert choice[EWC]["selected"]["base_harm"] > choice[REPLAY]["selected"]["base_harm"] + .1


def test_ties_use_cost_then_numeric_recipe_and_diagnostics_minimize_retention_violation():
    records = [r for m in METHODS for ridge, h, a, cost in ((1., .7, 1., .1), (10., .6, 2., .3))
               for r in pair(m, ridge=ridge, h=h, a=a, cost=cost)]
    assert all(s["selected"]["recipe"]["ridge"] == 10 and not s["eligible"]
               for s in design.select(design.table(records, cfg()), cfg()).values())
    records = [r for m in METHODS for ridge in (10., 1.) for r in pair(m, ridge=ridge, a=5, cost=.3)]
    selected = design.select(design.table(records, cfg()), cfg())
    assert all(s["selected"]["recipe"]["ridge"] == 1 for s in selected.values())
    shortlist = design.shortlist(design.table(records, cfg()), cfg())
    assert all([r["ridge"] for r in group] == [1., 10.] for group in shortlist.values())


def test_table_roots_within_replicas_and_partial_rows_cannot_select():
    records = pair(REPLAY)
    for metric in records[0]["metrics"][1:]:
        metric["base_mse"] = 16.
        metric["coordinate_mse"]["base"] = [8.]*6
        metric["base_common_mse"] = 8.
    refresh(records[0])
    row = design.table(records, cfg())[0]
    assert row["H"] == pytest.approx(1.1)
    assert row["H"] != pytest.approx(math.sqrt((16+2.2**2)/2)-2)
    assert design.table(records[:1], cfg()) == []
    partial = design.table(records[:1], cfg(), partial=True)
    assert not partial[0]["complete"] and not partial[0]["resource_feasible"]
    with pytest.raises(ValueError, match="complete valid"):
        design.select(partial, cfg())


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(protocol="plan15-shift-v1"),
    lambda r: r.update(output_dim=1),
    lambda r: r.update(context_key="scalar"),
    lambda r: r.update(family="development--scalar"),
    lambda r: r.update(partition="confirmation"),
    lambda r: r.update(novel_auc=0),
    lambda r: r.update(pairing_id="different"),
    lambda r: r.update(target=2),
    lambda r: r.update(valid="yes"),
    lambda r: r.update(status="running"),
    lambda r: r["metrics"].pop(1),
    lambda r: r["metrics"][1].update(base_mse=float("nan")),
    lambda r: r["metrics"][1]["coordinate_mse"].update(base=[1.]*6),
    lambda r: r["metrics"][1].update(novel_common_mse=200),
])
def test_own_protocol_schema_task_and_mathematical_validation(mutate):
    rows = pair(REPLAY)
    mutate(rows[0])
    with pytest.raises(ValueError):
        design.table(rows, cfg())


def test_generic_wrappers_do_not_mutate_or_relabel_task_and_vector_risk_uses_three():
    rows = pair(REPLAY)
    before = deepcopy(rows)
    wrapped = [{"metadata": {"protocol": PROTOCOL}, "summary": r, "metrics": r["metrics"]} for r in rows]
    assert design.table(wrapped, cfg()) == design.table(rows, cfg())
    assert rows == before
    assert rows[0]["protocol"] == PROTOCOL
    assert rows[0]["metrics"][0]["base_mse"] == sum(rows[0]["metrics"][0]["coordinate_mse"]["base"])/3
    with pytest.raises(ValueError, match="one task"):
        design.table(rows + pair(FD, task="scalar"), cfg())


def test_no_duplicate_or_failed_replica_deletion_or_unpaired_plans():
    rows = pair(REPLAY)
    with pytest.raises(ValueError, match="duplicate"):
        design.table(rows + rows[:1], cfg())
    rows[1]["valid"] = False
    assert not design.table(rows, cfg())[0]["valid"]
    with pytest.raises(ValueError, match="cannot be excluded"):
        analysis.paired_analysis(rows + pair(FD) + pair(EWC), cfg(), partition="development", seed=1, draws=10)
    rows = pair(REPLAY)
    rows[1].update(plan_hash=rows[0]["plan_hash"], pairing_id=rows[0]["pairing_id"])
    with pytest.raises(ValueError, match="reuse"):
        design.table(rows, cfg())


def test_refinement_is_deterministic_bounded_reserved_and_includes_near_cost_exclusions():
    records = [r for m in METHODS for ridge, h, cost in ((.1, .7, .3), (1., .2, .78), (100., .1, .3))
               for r in pair(m, ridge=ridge, h=h, cost=cost if m == FD else .3)]
    rows = design.table(records, cfg())
    proposals = design.refinement_proposals(rows, cfg(), [], 2.)
    assert proposals == design.refinement_proposals(list(reversed(rows)), cfg(), [], 2.)
    assert len(proposals) <= 8
    assert Counter(p["recipe"]["method"] for p in proposals)[FD] >= 2
    assert Counter(p["recipe"]["method"] for p in proposals)[REPLAY] >= 2
    assert any(p["recipe"]["method"] == FD and p["recipe"]["ridge"] == pytest.approx(10.) for p in proposals)
    after = design.refinement_proposals(rows, cfg(), proposals[:1], 2.)
    assert proposals[0]["recipe_key"] not in {p["recipe_key"] for p in after}
    assert len(after) <= 7
    assert design.refinement_proposals(rows, cfg(), proposals * 8, 2.) == []


def test_refinement_does_not_assume_monotonicity_through_reversals():
    rows = design.table([r for m in METHODS for ridge, h in ((.1, .7), (1., .9), (100., .8))
                         for r in pair(m, ridge=ridge, h=h)], cfg())
    assert not any(p["kind"].startswith("ridge-edge") for p in design.refinement_proposals(rows, cfg(), [], 2.))


def test_observed_crossing_keeps_last_entry_and_reports_all_failures():
    value = analysis.crossing([16, 7, 10, 4], 8)
    assert value["matching_budget"] == pytest.approx(2 + 2/6)
    assert value["multiple_crossings"] and len(value["downward_crossings"]) == 2
    assert analysis.crossing([16, 12, 10], 8)["status"] == "target_not_reached_at_observed_endpoint"
    assert analysis.crossing([8, 7, 4], 8)["status"] == "target_already_met_at_initialization"
    assert analysis.crossing([16, 4, 3], 8)["status"] == "crossing_below_first_post_observation_checkpoint"


def test_point_equivalence_interpolates_actual_cost_and_counts_both_views_not_independent_pairs():
    point = analysis.point_equivalence({m: pair(m) for m in METHODS})
    assert point["matching_budget"] == 1
    assert point["saved_fraction"] == pytest.approx(2/3)
    assert point["learning_time_ratio"] == pytest.approx(2/3)
    assert point["S"] == pytest.approx(1.)
    assert point["fd_image_views"] == 2 * point["fd_observations"]
    assert point["fd_scalar_labels"] == 6 * point["fd_observations"]
    assert point["fd_final_drift_strength"] == pytest.approx(1/3)
    assert point["replay_final_drift_strength"] == 1
    bad = {FD: pair(FD), REPLAY: pair(REPLAY)[:1]}
    with pytest.raises(ValueError, match="complete paired"):
        analysis.point_equivalence(bad)


def test_manifest_adapter_and_ducktyped_api_agree_and_json_is_finite():
    rows, manifest = cohort()
    result = analysis.analyze(rows, manifest, seed=7, draws=40)
    direct = analysis.paired_analysis(rows, cfg(), seed=7, draws=40, context_key="vector")
    assert all(result[k] == v for k, v in direct.items())
    assert result["protocol"] == PROTOCOL and not result["is_final"]
    assert result["efficiency"]["point"]["S"] == pytest.approx(1.)
    assert result["G"]["mean"] == pytest.approx(16/3)
    assert result["methods"][FD]["curves"]["novel"]["mean_accuracy1"] == [.3]*4
    json.dumps(result, allow_nan=False)
    keyed = deepcopy(manifest)
    keyed.pop("manifest_hash")
    keyed["selected_recipes"] = {r["method"]: r for r in keyed.pop("recipes")}
    keyed["manifest_hash"] = content_hash(keyed)
    assert analysis.analyze(rows, keyed, seed=7, draws=40)["G"] == result["G"]


def test_manifest_rejects_changed_policies_missing_replicas_and_confirmation_leakage():
    rows, manifest = cohort()
    with pytest.raises(ValueError, match="complete paired"):
        analysis.analyze(rows[:-1], manifest, seed=1, draws=10)
    changed = deepcopy(manifest)
    changed["recipes"][0]["ridge"] *= 10
    changed.pop("manifest_hash")
    changed["recipes"][0].pop("recipe_key")
    changed["recipes"][0]["recipe_key"] = Recipe.from_mapping(changed["recipes"][0]).key
    changed["manifest_hash"] = content_hash(changed)
    with pytest.raises(ValueError, match="frozen recipe"):
        analysis.analyze(rows, changed, seed=1, draws=10)
    changed = deepcopy(manifest)
    changed["sealed"] = False
    with pytest.raises(ValueError, match="unsealed"):
        analysis.analyze(rows, changed, seed=1, draws=10)
    changed = deepcopy(manifest)
    changed["retention_margin"] = .7
    with pytest.raises(ValueError, match="manifest hash"):
        analysis.analyze(rows, changed, seed=1, draws=10)
    changed = deepcopy(manifest)
    changed.pop("source_hash")
    with pytest.raises(ValueError, match="source and manifest hashes"):
        analysis.analyze(rows, changed, seed=1, draws=10)
    rows[0]["partition"] = "development"
    with pytest.raises(ValueError, match="confirmation-only"):
        analysis.analyze(rows, manifest, seed=1, draws=10)


def test_bootstrap_uses_paired_replica_roots_and_withholds_all_inverse_cis_if_one_draw_fails():
    rows, manifest = cohort()
    fd = next(r for r in rows if r["method"] == FD and r["replica"] == 1)
    for metric in fd["metrics"][1:]:
        metric["novel_mse"] = 10.
        metric["novel_common_mse"] = 5.
        metric["coordinate_mse"]["novel"] = [5.]*6
    refresh(fd)
    result = analysis.analyze(rows, manifest, seed=11, draws=80)
    boot = result["efficiency"]["bootstrap"]
    assert 0 < boot["invalid"] < 80
    assert boot["ci"] is None and boot["interval_status"] == "unavailable_unresolved_draws"
    assert result["G"]["ci"] is not None
    assert result["efficiency"]["full_budget_retention_supported"]
    assert not result["efficiency"]["matched_budget_retention_supported"]
    assert not result["efficiency"]["evidence_sample_savings"]


def test_resumable_blocks_are_bound_to_input_not_only_seed_and_indices(monkeypatch):
    rows, manifest = cohort()
    saved = {}
    ordinary = analysis.analyze(rows, manifest, seed=51, draws=210,
        save_bootstrap=lambda start, value: saved.update({start: value}))
    assert sorted(saved) == [0, 100, 200]
    calls, original = [], analysis._point
    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(analysis, "_point", measured)
    assert analysis.analyze(rows, manifest, seed=51, draws=210, load_bootstrap=saved.get) == ordinary
    assert len(calls) == 1
    changed = deepcopy(rows)
    changed[0]["active_seconds"] += 1
    with pytest.raises(ValueError, match="checkpoint"):
        analysis.analyze(changed, manifest, seed=51, draws=210, load_bootstrap=saved.get)


def test_stop_callback_and_cross_task_shared_draws():
    def stop():
        raise InterruptedError("requested stop")
    rows, manifest = cohort()
    with pytest.raises(InterruptedError):
        analysis.analyze(rows, manifest, seed=1, draws=10, check=stop)
    vector = analysis.analyze(rows, manifest, seed=1, draws=30)
    scalar = analysis.analyze(*cohort("scalar"), seed=1, draws=30)
    contrast = analysis.common_coordinate_contrast(vector, scalar)
    assert contrast["secondary"] and contrast["mean"] == pytest.approx(-8/3)
    scalar["bootstrap"]["seed"] = 2
    with pytest.raises(ValueError, match="identical named"):
        analysis.common_coordinate_contrast(vector, scalar)


def test_full_confirmation_is_a_complete_frozen_cohort_not_smoke_evidence():
    rows, manifest = cohort()
    rows = [record(m, r, partition="confirmation", N=100,
                   novel=np.linspace(16., 4. if m == FD else 8., 101))
            for m in METHODS for r in range(8)]
    manifest.update(N=100, smoke=False, replicas=list(range(8)))
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = content_hash(manifest)
    result = analysis.analyze(rows, manifest, seed=7, draws=20)
    assert result["is_final"] and result["efficiency"]["evidence_sample_savings"]
    assert result["efficiency"]["point"]["S"] == pytest.approx(.25)
    with pytest.raises(ValueError, match="complete paired"):
        analysis.analyze(rows[:-1], manifest, seed=7, draws=20)


def test_bootstrap_resume_is_bound_to_frozen_source_and_manifest_hashes():
    rows, manifest = cohort()
    saved = {}
    analysis.analyze(rows, manifest, seed=1, draws=10,
                     save_bootstrap=lambda start, value: saved.update({start: value}))
    changed = {**manifest, "source_hash": "new-source"}
    changed.pop("manifest_hash")
    changed["manifest_hash"] = content_hash(changed)
    with pytest.raises(ValueError, match="source/manifest"):
        analysis.analyze(rows, changed, seed=1, draws=10, load_bootstrap=saved.get)
