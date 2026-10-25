"""Adversarial Phase 8 design checks using scalar artifacts only, never CUDA.

These are contract tests, not expected-failure markers: a decision must not
silently consume malformed/unpaired development evidence or an infeasible
forecast. All fixtures preserve the paired source/initialization contract.
"""

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest

from mnist_experiment.fisher_memory_regression import design
from mnist_experiment.fisher_memory_regression.config import Config, Recipe


FD = "fd_replay_ewc"
REPLAY = "replay_ewc"
BOUNDARY = "ewc_only"


def recipe(method=REPLAY, *, capacity=64, replay_count=16):
    return Recipe(method, 0 if method == BOUNDARY else capacity,
                  0 if method == BOUNDARY else replay_count,
                  strength=1. if method == FD else 0.)


def record(policy, replica, *, risk=None, frozen_base=4., base_auc=4.,
           active_seconds=None, observations=400, family="allocation", valid=True):
    initial = 100. + 2 * replica
    shape = [100., 60., 35., 25., 20.] if policy.method == FD else [100., 80., 65., 50., 40.]
    curve = np.interp(np.arange(101), [0, 25, 50, 75, 100], shape) + 2 * replica
    if risk is not None:
        curve[1:] = risk
    if active_seconds is None:
        active_seconds = {FD: 100., REPLAY: 50., BOUNDARY: 25.}[policy.method]
    learning_seconds = active_seconds * .8
    rows = [dict(n=n, novel_mse=float(value), base_mse=frozen_base if n == 0 else base_auc,
                 observations=observations * n // 100, learning_seconds=learning_seconds * n / 100)
            for n, value in enumerate(curve)]
    return dict(schema_version=1, protocol="plan15-regression-v1", recipe=policy.mapping(),
                recipe_key=policy.key, method=policy.method, capacity=policy.capacity,
                replay_count=policy.replay_count, target=100, N=100, replica=replica,
                family=family, partition="confirmation" if family == "confirmation" else "development",
                plan_hash=f"{family}-shared-{replica}", pairing_id=f"{family}-shared-{replica}",
                frozen_novel_mse=initial, frozen_base_mse=frozen_base,
                novel_auc=float(np.mean(curve[1:])), base_auc=base_auc, observations=observations,
                learning_seconds=learning_seconds, active_seconds=active_seconds,
                peak_vram_bytes=100, valid=valid, metrics=rows)


def setup_design():
    cfg = Config(device="cpu")
    policies = [recipe(FD, capacity=16, replay_count=4),
                recipe(REPLAY, capacity=256, replay_count=64), recipe(BOUNDARY)]
    rows = [record(policy, replica) for policy in policies for replica in range(2)]
    selections = {method: design.choose([row for row in design.aggregate(rows) if row["method"] == method], cfg)
                  for method in (FD, REPLAY)}
    calibration = dict(ridge=1., policy=dict(max_iter=100, max_eval=150),
                       mean_active_seconds={FD: 100., REPLAY: 50., BOUNDARY: 25.},
                       mean_initialization_seconds=20., envelope_seconds_per_arrival=1.,
                       memory_ceiling_bytes=1_000)
    return cfg, policies, rows, selections, calibration


def test_retention_guard_is_mean_of_paired_root_differences_not_worst_replica():
    cfg = Config(device="cpu")
    favored = recipe(FD, capacity=16, replay_count=4)
    alternative = recipe(FD, capacity=64, replay_count=16)
    rows = [record(favored, 0, risk=10., frozen_base=1., base_auc=9.),
            record(favored, 1, risk=10., frozen_base=100., base_auc=81.),
            record(alternative, 0, risk=20., frozen_base=1., base_auc=1.),
            record(alternative, 1, risk=20., frozen_base=100., base_auc=100.)]
    estimates = design.aggregate(rows)
    measured = next(row for row in estimates if row["recipe_key"] == favored.key)
    assert measured["base_harm"] == pytest.approx(.5)  # (+2 degrees - 1 degree) / 2.
    assert measured["base_harm"] != pytest.approx(math.sqrt(45.) - math.sqrt(50.5))
    decision = design.choose(estimates, cfg)
    assert decision["selected"]["recipe_key"] == favored.key
    assert decision["warning"] is None


def test_resource_and_numerical_constraints_dominate_apparent_predictive_winner():
    cfg = Config(device="cpu")
    policies = [recipe(REPLAY, capacity=c, replay_count=4) for c in (16, 32, 64, 128)]
    rows = [record(policy, r, risk=10. if c == 3 else 1.)
            for c, policy in enumerate(policies) for r in range(2)]
    estimates = design.aggregate(rows)
    by_key = {row["recipe_key"]: row for row in estimates}
    by_key[policies[0].key]["valid"] = False
    by_key[policies[1].key]["seconds_per_arrival"] = .10001
    by_key[policies[2].key]["max_vram_bytes"] = 1_001
    decision = design.choose(estimates, cfg, envelope=.1, memory_ceiling=1_000)
    assert decision["selected"]["recipe_key"] == policies[3].key
    assert decision["ranking"] == [policies[3].key]


def test_retention_failure_is_visible_and_cannot_be_rescued_by_lower_new_domain_loss():
    cfg = Config(device="cpu")
    safer, worse = recipe(REPLAY, capacity=16, replay_count=4), recipe(REPLAY)
    rows = [record(safer, r, risk=50., frozen_base=1., base_auc=9.) for r in range(2)]
    rows += [record(worse, r, risk=1., frozen_base=1., base_auc=16.) for r in range(2)]
    decision = design.choose(design.aggregate(rows), cfg)
    assert decision["selected"]["recipe_key"] == safer.key
    assert decision["warning"] and "retention" in decision["warning"]


def test_practical_tie_prefers_cost_then_capacity_then_replay_not_raw_minimum():
    cfg = Config(device="cpu")
    policies = [recipe(REPLAY, capacity=c, replay_count=b) for c, b in ((256, 64), (64, 16), (16, 8), (16, 4))]
    rows = [record(policy, r, risk=10. if index == 0 else 10.8,
                   active_seconds=100. if index == 0 else 50.)
            for index, policy in enumerate(policies) for r in range(2)]
    estimates = design.aggregate(rows)
    selected = design.choose(estimates, cfg)["selected"]
    assert selected["recipe_key"] == policies[-1].key
    assert selected["novel_auc"] > min(row["novel_auc"] for row in estimates)
    assert design.choose(list(reversed(estimates)), cfg)["selected"] == selected


def test_outside_tie_tolerance_does_not_win_by_being_cheap():
    cfg = Config(device="cpu")
    best, cheap = recipe(REPLAY), recipe(REPLAY, capacity=16, replay_count=4)
    rows = [record(best, r, risk=10., active_seconds=100.) for r in range(2)]
    rows += [record(cheap, r, risk=11.02, active_seconds=1.) for r in range(2)]
    assert design.choose(design.aggregate(rows), cfg)["selected"]["recipe_key"] == best.key


def test_control_is_independently_selected_not_copied_from_fd_allocation():
    cfg, _, _, _, calibration = setup_design()
    rows = []
    for method in (FD, REPLAY):
        for c, b in ((16, 4), (256, 64)):
            policy = recipe(method, capacity=c, replay_count=b)
            preferred = c == 16 if method == FD else c == 256
            rows.extend(record(policy, r, risk=10. if preferred else 30.) for r in range(2))
    estimates = design.aggregate(rows)
    selections = {method: design.choose([row for row in estimates if row["method"] == method], cfg)
                  for method in (FD, REPLAY)}
    chosen = {policy.method: policy for policy in design.selected_recipes(cfg, calibration, selections)}
    assert (chosen[FD].capacity, chosen[FD].replay_count) == (16, 4)
    assert (chosen[REPLAY].capacity, chosen[REPLAY].replay_count) == (256, 64)
    assert chosen[BOUNDARY].capacity == chosen[BOUNDARY].replay_count == 0


def test_no_feasible_candidate_is_an_error_not_a_best_invalid_policy():
    rows = design.aggregate([record(recipe(), 0)])
    with pytest.raises(ValueError, match="feasible"):
        design.choose(rows, Config(device="cpu"), envelope=0.)


@pytest.mark.parametrize("defect", ["duplicate", "mixed_recipe", "mixed_target"])
def test_aggregate_rejects_identity_collisions(defect):
    rows = [record(recipe(), r) for r in range(2)]
    if defect == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif defect == "mixed_recipe":
        rows[1]["recipe"]["ridge"] = 10.
    else:
        rows[1]["target"] = 30
    with pytest.raises(ValueError):
        design.aggregate(rows)


def test_grid_masks_impossible_cells_and_can_afford_its_declared_forecast():
    cfg, _, _, _, calibration = setup_design()
    reserve = cfg.confirmation_reserve_seconds + cfg.analysis_reserve_seconds
    full = design.grid(cfg, calibration, seconds_available=reserve + 10_000.)
    expected = {(c, b) for c in cfg.capacities for b in cfg.replay_counts if b <= c}
    assert {tuple(cell) for cell in full["cells"]} == expected
    assert len(full["cells"]) == 8
    assert not full["timing_reduced"]
    assert full["predicted_seconds"] <= 10_000.
    assert full["fit_registered"] is False
    two_cells = 1.5 * cfg.allocation_replicas * (20. + 25. + 2 * (100. + 50.))
    reduced = design.grid(cfg, calibration, seconds_available=reserve + two_cells + 1.)
    assert reduced["predicted_seconds"] <= two_cells + 1.
    assert len({cell[0] for cell in reduced["cells"]}) >= 2
    assert len({cell[1] for cell in reduced["cells"]}) >= 2
    assert reduced["timing_reduced"]
    assert cfg.reference_capacity in [cell[0] for cell in reduced["cells"]]


def test_grid_never_forces_one_cell_when_no_cell_fits():
    cfg, _, _, _, calibration = setup_design()
    reserve = cfg.confirmation_reserve_seconds + cfg.analysis_reserve_seconds
    with pytest.raises(ValueError):
        design.grid(cfg, calibration, seconds_available=reserve + 1.)


def test_grid_forecast_counts_fresh_shared_initialization_once_per_replica():
    cfg, _, _, _, calibration = setup_design()
    calibration["mean_initialization_seconds"] = 2_000.
    reserve = cfg.confirmation_reserve_seconds + cfg.analysis_reserve_seconds
    decision = design.grid(cfg, calibration, seconds_available=reserve + 10_000.)
    minimum_initialization = cfg.allocation_replicas * calibration["mean_initialization_seconds"]
    pair_cost = calibration["mean_active_seconds"][FD] + calibration["mean_active_seconds"][REPLAY]
    minimum_learning = cfg.allocation_replicas * (len(decision["cells"]) * pair_cost +
                                                  calibration["mean_active_seconds"][BOUNDARY])
    assert decision["predicted_seconds"] >= minimum_initialization + minimum_learning


def test_efficiency_registration_uses_paired_means_and_frozen_development_target():
    cfg, policies, rows, _, _ = setup_design()
    original = deepcopy(rows)
    result = design.efficiency_registration(rows, cfg, policies)
    assert rows == original
    assert result["q_dev"] == pytest.approx(41.)
    assert result["n_F"] == 50 and result["n_R"] == 100
    assert result["development_replicas"] == [0, 1]
    assert result["development_supported"]
    assert result["harm_margin"] == 0.
    assert result["fit_registered"] is False
    assert result == design.efficiency_registration(list(reversed(rows)), cfg, policies)


def test_unpaired_extra_replica_cannot_move_registered_target_or_budget():
    cfg, policies, rows, _, _ = setup_design()
    expected = design.efficiency_registration(rows, cfg, policies)
    extra = record(policies[1], 99, risk=100_000.)
    result = design.efficiency_registration([*rows, extra], cfg, policies)
    assert result == expected


def test_no_acquisition_support_retains_explicit_unsupported_75_probe():
    cfg, policies, rows, _, _ = setup_design()
    for row in rows:
        if row["method"] == FD:
            for checkpoint in row["metrics"][1:]:
                checkpoint["novel_mse"] = 110.
            row["novel_auc"] = 110.
    result = design.efficiency_registration(rows, cfg, policies)
    assert result["n_F"] == 75
    assert not result["development_supported"]


def test_beating_replay_without_improving_initialization_does_not_support_savings():
    cfg, policies, rows, _, _ = setup_design()
    for row in rows:
        for checkpoint in row["metrics"][1:]:
            checkpoint["novel_mse"] = 110. if row["method"] == FD else 120.
        row["novel_auc"] = row["metrics"][1]["novel_mse"]
    result = design.efficiency_registration(rows, cfg, policies)
    assert result["n_F"] == 75 and not result["development_supported"]


@pytest.mark.parametrize("defect", ["duplicate_replica", "duplicate_n", "missing_n", "missing_unselected_n",
                                    "bad_target", "unpaired_plan", "unpaired_initialization", "invalid",
                                    "nonfinite", "confirmation_partition"])
def test_registration_rejects_incomplete_or_unpaired_evidence(defect):
    cfg, policies, rows, _, _ = setup_design()
    if defect == "duplicate_replica":
        rows.append(deepcopy(rows[0]))
    elif defect == "duplicate_n":
        rows[0]["metrics"].insert(25, deepcopy(rows[0]["metrics"][25]))
    elif defect in ("missing_n", "missing_unselected_n"):
        rows[0]["metrics"].pop(25 if defect == "missing_n" else 13)
    elif defect == "bad_target":
        rows[0]["target"] = 30
    elif defect == "unpaired_plan":
        rows[0]["plan_hash"] = rows[0]["pairing_id"] = "different-input-stream"
    elif defect == "unpaired_initialization":
        rows[0]["metrics"][0]["novel_mse"] = rows[0]["frozen_novel_mse"] = 110.
    elif defect == "invalid":
        rows[0]["valid"] = False
    elif defect == "nonfinite":
        rows[0]["metrics"][25]["novel_mse"] = float("nan")
    else:
        for row in rows:
            row["family"] = "confirmation"
            row["partition"] = "confirmation"
    with pytest.raises(ValueError):
        design.efficiency_registration(rows, cfg, policies)


def test_registration_requires_both_selected_methods_and_a_paired_replica():
    cfg, policies, rows, _, _ = setup_design()
    without_replay = [row for row in rows if row["method"] != REPLAY]
    with pytest.raises(ValueError, match="paired"):
        design.efficiency_registration(without_replay, cfg, policies)


def test_cohort_size_depends_on_timing_not_apparent_effect_or_significance():
    cfg, _, rows, selections, calibration = setup_design()
    per_replica = 1.5 * (100. + 50. + 25. + 20.)
    available = cfg.analysis_reserve_seconds + 3.1 * per_replica
    before = design.confirmation_manifest(cfg, calibration, selections, rows, available)
    changed = deepcopy(rows)
    for row in changed:
        for checkpoint in row["metrics"][1:]:
            checkpoint["novel_mse"] = .1 if row["method"] == FD else 1_000.
        row["novel_auc"] = row["metrics"][1]["novel_mse"]
    after = design.confirmation_manifest(cfg, calibration, selections, changed, available)
    assert before["replicas"] == after["replicas"] == [0, 1, 2]
    assert before["estimated_seconds"] == after["estimated_seconds"] == 3 * per_replica
    assert before["estimated_seconds"] + cfg.analysis_reserve_seconds <= available
    assert before["precision_warning"]


def test_confirmation_uses_upper_measured_cost_and_shared_initialization():
    cfg, _, rows, selections, calibration = setup_design()
    rows[0]["active_seconds"] = 500.
    per_replica = 1.5 * (500. + 50. + 25. + 20.)
    available = cfg.analysis_reserve_seconds + 2.1 * per_replica
    decision = design.confirmation_manifest(cfg, calibration, selections, rows, available)
    assert decision["replicas"] == [0, 1]
    assert decision["estimated_seconds"] == pytest.approx(2 * per_replica)
    assert decision["estimated_seconds"] + cfg.analysis_reserve_seconds <= available


def test_cohort_never_exceeds_frozen_configured_maximum():
    cfg, _, rows, selections, calibration = setup_design()
    decision = design.confirmation_manifest(cfg, calibration, selections, rows, 1e9)
    assert decision["replicas"] == list(range(cfg.confirmation_replicas))


@pytest.mark.parametrize("available", [0., 100., 1_800.])
def test_no_zero_cohort_or_budget_overrun_when_confirmation_cannot_fit(available):
    cfg, _, rows, selections, calibration = setup_design()
    with pytest.raises(ValueError):
        design.confirmation_manifest(cfg, calibration, selections, rows, available)


def test_one_replica_cannot_be_registered_for_a_two_replica_inference_pipeline():
    cfg, _, rows, selections, calibration = setup_design()
    per_replica = 1.5 * (100. + 50. + 25. + 20.)
    with pytest.raises(ValueError):
        design.confirmation_manifest(cfg, calibration, selections, rows,
                                     cfg.analysis_reserve_seconds + 1.1 * per_replica)


def test_seed_identity_separates_calibration_allocation_and_confirmation():
    cfg = Config(device="cpu")
    components = ("initialization_model", "stream", "probes", "evaluation:0")
    seeds = [cfg.seed_for(family, replica, component)
             for family in ("calibration", "allocation", "confirmation")
             for replica in range(2) for component in components]
    assert len(set(seeds)) == len(seeds)
    assert cfg.seed_for("calibration", 0, "stream") == cfg.seed_for("calibration", 0, "stream")
    assert replace(cfg, seed=cfg.seed + 1).seed_for("calibration", 0, "stream") != seeds[1]


def test_aggregate_cannot_pool_calibration_and_confirmation_outcomes():
    policy = recipe()
    rows = [record(policy, 0, family="calibration"), record(policy, 1, family="confirmation")]
    with pytest.raises(ValueError):
        design.aggregate(rows)


def test_aggregate_cannot_pool_independent_development_families():
    policy = recipe()
    rows = [record(policy, 0, family="calibration"), record(policy, 1, family="allocation")]
    with pytest.raises(ValueError):
        design.aggregate(rows)


def test_invalid_policy_remains_visible_but_cannot_be_selected():
    good, bad = recipe(REPLAY), recipe(REPLAY, capacity=16, replay_count=4)
    rows = [record(good, r, risk=50.) for r in range(2)]
    rows += [record(bad, 0, risk=1.), record(bad, 1, risk=1., valid=False)]
    candidates = design.aggregate(rows)
    assert len(candidates) == 2
    invalid = next(row for row in candidates if row["recipe_key"] == bad.key)
    assert invalid["count"] == 2 and invalid["valid"] is False
    assert design.choose(candidates, Config(device="cpu"))["selected"]["recipe_key"] == good.key


@pytest.mark.parametrize("defect", ["missing_provenance", "wrong_recipe_key", "wrong_method", "missing_initial",
                                    "changed_exposures", "stale_summary", "shared_replica_plan", "missing_partition"])
def test_complete_evidence_validation_covers_identity_and_summary_integrity(defect):
    cfg, policies, rows, _, _ = setup_design()
    if defect == "missing_provenance":
        rows[0].pop("plan_hash")
    elif defect == "wrong_recipe_key":
        rows[0]["recipe_key"] = "invented-policy"
    elif defect == "wrong_method":
        rows[0]["method"] = REPLAY
    elif defect == "missing_initial":
        rows[0]["metrics"].pop(0)
    elif defect == "changed_exposures":
        rows[0]["metrics"][25]["observations"] += 1
    elif defect == "stale_summary":
        rows[0]["novel_auc"] += 1.
    elif defect == "shared_replica_plan":
        for row in rows:
            row["plan_hash"] = row["pairing_id"] = "reused-stream"
    else:
        rows[0].pop("partition")
    with pytest.raises(ValueError):
        design.efficiency_registration(rows, cfg, policies)


def test_selected_control_cannot_be_replaced_by_fd_recipe_under_its_label():
    cfg, _, _, selections, calibration = setup_design()
    selections[REPLAY] = deepcopy(selections[FD])
    with pytest.raises(ValueError):
        design.selected_recipes(cfg, calibration, selections)


@pytest.mark.parametrize("bad_seconds", [float("nan"), float("inf"), -1.])
def test_nonfinite_or_negative_budget_is_not_unlimited_compute(bad_seconds):
    cfg, _, rows, selections, calibration = setup_design()
    with pytest.raises(ValueError):
        design.grid(cfg, calibration, seconds_available=bad_seconds)
    with pytest.raises(ValueError):
        design.confirmation_manifest(cfg, calibration, selections, rows, bad_seconds)


def test_smoke_mode_does_not_disable_budget_admission():
    cfg, _, rows, selections, calibration = setup_design()
    cfg = replace(cfg, smoke=True)
    reserve = cfg.confirmation_reserve_seconds + cfg.analysis_reserve_seconds
    with pytest.raises(ValueError):
        design.grid(cfg, calibration, seconds_available=reserve)
    with pytest.raises(ValueError):
        design.confirmation_manifest(cfg, calibration, selections, rows, cfg.analysis_reserve_seconds)


def test_forecast_accounts_for_shared_assets_once_not_per_grid_cell():
    cfg, _, _, _, calibration = setup_design()
    seconds = cfg.confirmation_reserve_seconds + cfg.analysis_reserve_seconds + 10_000.
    decision = design.grid(cfg, calibration, seconds_available=seconds)
    expected = 1.5 * cfg.allocation_replicas * (
        calibration["mean_initialization_seconds"] + calibration["mean_active_seconds"][BOUNDARY] +
        len(decision["cells"]) * (calibration["mean_active_seconds"][FD] + calibration["mean_active_seconds"][REPLAY]))
    assert decision["predicted_seconds"] == pytest.approx(expected)
