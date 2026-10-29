"""Grid-count risk, pairing and inverse uncertainty without model execution."""

from dataclasses import replace
from collections import Counter

import numpy as np
import pytest

from mnist_experiment.fisher_memory_grid_registration import PROTOCOL, analysis, design
from mnist_experiment.fisher_memory_grid_registration.config import Config, Recipe, METHODS, content_hash


def record(method, replica=0, partition="confirmation", cost=1., harm=.01):
    cfg = Config.smoke_config()
    recipe = Recipe.from_config(cfg, method, strength=.5 if method == METHODS[0] else 0)
    novel = [16., 10., 6., 3.] if method == METHODS[0] else [16., 13., 11., 9.]
    metrics = [dict(n=n, observations=(replica+10)*n, learning_seconds=cost*n,
                    base_mse=4. if n == 0 else (2+harm)**2, novel_mse=risk,
                    coordinate_mse={"base": [(4. if n == 0 else (2+harm)**2)/2]*54,
                                    "novel": [risk/2]*54}) for n, risk in enumerate(novel)]
    summary = analysis.summarize_trajectory(metrics, N=3)
    return dict(summary, family=partition+"--grid", partition=partition, context_key="grid",
        output_dim=54, target=3, replica=replica, recipe=recipe.mapping(), recipe_key=recipe.key,
        method=method, capacity=recipe.capacity, replay_count=recipe.replay_count, metrics=metrics,
        plan_hash=f"{partition}-{replica}", pairing_id=f"{partition}-{replica}",
        active_seconds=10., peak_vram_bytes=1000, retired_audit=[])


def cohort():
    rows = [record(m, r) for m in METHODS for r in range(2)]
    manifest = dict(protocol=PROTOCOL, schema_version=1, sealed=True, N=3, smoke=True,
        replicas=[0, 1], context={"key": "grid", "output_dim": 54},
        recipes=[{**next(r["recipe"] for r in rows if r["method"] == m),
                  "recipe_key": next(r["recipe_key"] for r in rows if r["method"] == m)} for m in METHODS],
        retention_margin=.5, matching_tolerance=.1, envelope_factor=2.5, source_hash="fixture")
    manifest["manifest_hash"] = content_hash(manifest)
    return rows, manifest


def test_grid_has_twenty_recipes_and_fixed_whole_grid_memory():
    recipes = design.initial_grid(Config(), 2.)
    assert Counter(r.method for r in recipes) == dict(zip(METHODS, (12, 4, 4)))
    assert len({r.key for r in recipes}) == 20
    assert {r.capacity for r in recipes if r.method != "ewc_only"} == {32}
    assert {r.replay_count for r in recipes if r.method != "ewc_only"} == {2}


def test_risk_is_sum_of_54_coordinate_errors_divided_by_27():
    rows, _ = cohort()
    assert len(analysis.validate_records(rows, 3, "confirmation")) == 6
    rows[0]["metrics"][1]["coordinate_mse"]["novel"] = [10./54]*54
    with pytest.raises(ValueError, match="risk"):
        analysis.validate_records(rows, 3, "confirmation")


def test_analysis_recomputes_S_and_all_sample_units():
    rows, manifest = cohort()
    result = analysis.analyze(rows, manifest, seed=17, draws=50)
    match = result["efficiency"]
    point = match.get("point", match)
    assert point["status"] == "valid"
    assert point["saved_fraction"] == pytest.approx((3-point["matching_budget"])/3)
    assert point["S"] == pytest.approx(point["saved_fraction"]/point["learning_time_ratio"])
    assert point["fd_digit_pairs"] == pytest.approx(9*point["fd_observations"])
    assert point["fd_image_views"] == pytest.approx(18*point["fd_observations"])


def test_incomplete_or_unpaired_cohorts_cannot_be_analyzed():
    rows, manifest = cohort()
    with pytest.raises(ValueError):
        analysis.analyze(rows[:-1], manifest, seed=17, draws=50)
    rows[0]["plan_hash"] = rows[0]["pairing_id"] = "different"
    with pytest.raises(ValueError, match="pair"):
        analysis.analyze(rows, manifest, seed=17, draws=50)


def test_last_sustained_crossing_and_unresolved_endpoint():
    crossing = analysis.crossing([16, 7, 10, 6], 8)
    assert crossing["status"] == "valid" and crossing["multiple_crossings"]
    assert crossing["matching_budget"] == pytest.approx(2.5)
    assert analysis.crossing([16, 12, 10, 9], 8)["status"] != "valid"


def test_policy_selection_never_weakens_replay_to_admit_FD():
    cfg = Config.smoke_config()
    rows = [record(m, r, partition="development", cost=4. if m==METHODS[0] else 1.)
            for m in METHODS for r in range(2)]
    policies = design.select(design.table(rows, cfg), cfg)
    assert policies["replay_ewc"]["eligible"]
    assert "selected_replay_cost" in policies["fd_replay_ewc"]["failed_conditions"]


def test_configuration_scope_and_resume_hash_are_stable():
    cfg = Config()
    cfg.validate()
    assert Config.from_mapping(cfg.mapping()).scientific_hash == cfg.scientific_hash
    for change in ({"output_dims": (6,)}, {"reference_capacity": 256}, {"refinement_settings": 7}):
        with pytest.raises(ValueError):
            replace(cfg, **change).validate()
