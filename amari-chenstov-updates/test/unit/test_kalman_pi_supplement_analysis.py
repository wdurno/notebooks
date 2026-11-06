"""CPU-only scalar acceptance for independent three-arm Phase 5E inference."""

from copy import deepcopy
import json

import numpy as np
import pytest

from mnist_experiment.kalman_pi import analysis as historical
from mnist_experiment.kalman_pi.retention import analysis as primary
from mnist_experiment.kalman_pi.retention.supplement import analysis
from src.seeding import derive_component_seed


def selection(n=8, factors=(1., 1.25)):
    partition_seed = 171102
    protocol = analysis.PARTITION_PROTOCOL
    return {"schema": 1, "protocol": analysis.PROTOCOL, "frozen": True,
            "frozen_at": "2026-11-02T20:00:00Z", "arms": deepcopy(analysis.ARMS),
            "main_replicas": n, "stationary_replicas": 0, "fresh_trajectory_target": 3 * n,
            "bootstrap_seed": analysis.DEFAULT_SEED, "config_hash": "a" * 64, "sources_hash": "b" * 64,
            "schedule_factors": list(factors),
            "data_partition": {"protocol": protocol, "seed": partition_seed,
                               **{split + "_seed": derive_component_seed(
                                   partition_seed, f"{protocol}:study-partition:0:{split}")
                                  for split in ("train", "test")}}}


def trajectory(arm="fixed_ridge", replica=0, *, old_gap_pp=0., nine_gap_pp=0.,
               completed=True, cost=1., empty=False):
    factor = (1., 1.25)[replica % 2]
    rows = []
    for i, n in enumerate((0, 4, 20)):
        nine = .2 + .1 * i + nine_gap_pp / 100 * i / 1.3
        old = .9 - .01 * i + old_gap_pp / 100 * i / 1.3
        rows.append({"arrivals": n * factor, "step": int(n * factor),
                     "p": .005 if i < 2 else .1, "segment": "hold" if i < 2 else "rise",
                     "expected_nines": .1 * n * factor, "observed_nines": int(n * factor // 4),
                     "nine_recall": nine, "old_accuracy": old, "nll": .5,
                     "class_accuracy": [old] * 9 + [nine], "initial_nines": 3,
                     "frozen_nine_recall": .2, "frozen_old_accuracy": .9, "q": .01,
                     "effective_size": 100., "wall_seconds": n * cost})
    return {"unit_id": f"supplement-main-r{replica:03}-{arm}", "stage": "fresh", "cohort": "fresh",
            "schedule": "main", "replica": replica, "arm": arm, **analysis.ARMS[arm],
            "c": .025, "anchor_refresh": 0, "completed": completed,
            "stream_hash": f"supplement-stream-r{replica}", "asset_unit": f"supplement-assets-r{replica}",
            "initial_unit": f"supplement-init-r{replica}", "schedule_factor": factor,
            "metrics": [] if empty else rows if completed else rows[:2],
            "summary": {"wall_seconds": 20 * cost, "online_wall_seconds": 18 * cost,
                        "costs": {"solver_seconds": 3 * cost, "curvature_seconds": 2 * cost,
                                  "controller_seconds": cost, "evaluation_seconds": 2 * cost,
                                  "checkpoint_seconds": cost}}}


def block(replica=0):
    return [trajectory("fixed_ridge", replica),
            trajectory("fixed_zero", replica, nine_gap_pp=1., old_gap_pp=-.2, cost=.8),
            trajectory("kalman_zero", replica, nine_gap_pp=3., old_gap_pp=.4, cost=2.)]


def contrast(result, arm="kalman_zero", control="fixed_ridge"):
    return next(c for c in result["comparisons"] if c["condition"] == arm and c["control"] == control)


def test_three_contrasts_distinguish_physical_fixed_arms_and_preserve_identity():
    rows, design = block(), selection()
    before, frozen = deepcopy(rows), deepcopy(design)
    result = analysis.summarize(rows, design, bootstrap_draws=20)
    assert result["schema"] == 1 and result["kind"] == "plan17-zero-ridge-analysis"
    assert result["stage"] == "fresh" and result["schedule"] == "main"
    assert result["trajectories_sha256"] == historical.trajectories_hash(rows)
    assert result["selection_sha256"] == historical.trajectories_hash(design)
    assert result["eligible_replicas"] == [0] and result["n_pairs"] == 1
    assert [(c["condition"], c["control"]) for c in result["comparisons"]] == list(analysis.CONTRASTS)
    assert len([r for r in result["replicas"] if r["condition"] == "fixed"]) == 2
    assert {r["arm"] for r in result["replicas"]} == set(analysis.ARMS)
    fixed = contrast(result, "fixed_zero")
    assert fixed["treatment_condition"] == fixed["control_condition"] == "fixed"
    assert fixed["treatment_ridge"] == 0 and fixed["control_ridge"] == .1
    assert contrast(result, control="fixed_zero")["control_ridge"] == 0
    assert result["counts"] == {"sealed_trajectories": 3, "complete": 3, "incomplete": 0,
                                "planned_trajectories": 24, "complete_blocks": 1, "planned_blocks": 8}
    assert rows == before and design == frozen
    json.dumps(result, allow_nan=False)


def test_observation_trapezoids_pp_endpoints_regimes_and_paired_time_ratios():
    result = analysis.summarize(block(), selection(), bootstrap_draws=20)
    effects = contrast(result)
    assert effects["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(3.)
    assert effects["observation_mean_pp"]["old_accuracy"]["mean"] == pytest.approx(.4)
    assert effects["endpoint_pp"]["nine_recall"]["mean"] == pytest.approx(6 / 1.3)
    assert effects["segments"]["hold"]["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(1.5 / 1.3)
    assert effects["segments"]["rise"]["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(4.5 / 1.3)
    assert contrast(result, "fixed_zero")["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(1.)
    assert contrast(result, control="fixed_zero")["observation_mean_pp"]["old_accuracy"]["mean"] == pytest.approx(.6)
    assert effects["costs"]["wall_seconds"]["ratio"]["mean"] == 2.
    assert effects["costs"]["online_wall_seconds"]["ratio"]["mean"] == 2.
    assert effects["costs"]["learning_seconds"]["difference"]["mean"] == 6.
    assert effects["costs"]["learning_seconds"]["ratio"]["mean"] == 2.
    assert effects["observation_mean_pp"]["nine_recall"]["ci95"] is None


def test_each_full_replica_has_equal_weight_despite_different_exposure_lengths():
    rows = block(0) + block(1)
    rows[2] = trajectory("kalman_zero", 0, nine_gap_pp=1., old_gap_pp=-1.)
    rows[5] = trajectory("kalman_zero", 1, nine_gap_pp=3., old_gap_pp=3.)
    result = analysis.summarize(rows, selection(2), bootstrap_draws=2000)
    effects = contrast(result)
    assert effects["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(2.)
    assert effects["observation_mean_pp"]["old_accuracy"]["mean"] == pytest.approx(1.)
    assert effects["observation_mean_pp"]["nine_recall"]["mean"] != pytest.approx((1 + 1.25 * 3) / 2.25)
    assert effects["observation_mean_pp"]["nine_recall"]["ci95"] == pytest.approx([1., 3.])
    assert effects["observation_mean_pp"]["old_accuracy"]["ci95"] == pytest.approx([-1., 3.])


@pytest.mark.parametrize("prefix", (False, True))
def test_partial_blocks_never_enter_any_contrast_but_remain_visible(prefix):
    rows = block(0) + block(1)[:2]
    if prefix:
        rows += [trajectory("kalman_zero", 1, completed=False)]
    result = analysis.summarize(rows, selection())
    assert result["eligible_replicas"] == [0]
    assert len(result["awaiting_full_block_units"]) == 2
    assert all(c["replicas"] == [0] and c["n_pairs"] == 1 for c in result["comparisons"])
    assert contrast(result, "fixed_zero")["unpaired_control_units"] == [rows[3]["unit_id"]]
    assert result["incomplete_units"] == ([rows[-1]["unit_id"]] if prefix else [])
    assert result["counts"]["complete"] == 5 and result["counts"]["incomplete"] == int(prefix)
    without_full_block = analysis.summarize(rows[3:], selection())
    assert all(c["n_pairs"] == 0 for c in without_full_block["comparisons"])


def test_empty_progress_and_empty_prefixes_are_explicit():
    result = analysis.summarize([], selection())
    assert result["eligible_replicas"] == [] and result["n_pairs"] == 0
    assert result["replicas"] == [] and result["complete_units"] == []
    assert all(c["observation_mean_pp"]["old_accuracy"]["mean"] is None for c in result["comparisons"])
    result = analysis.summarize([trajectory(completed=False, empty=True)], selection())
    assert result["counts"]["incomplete"] == 1 and result["n_pairs"] == 0


@pytest.mark.parametrize("field,value", [("c", .05), ("ridge", .01), ("condition", "covariance"),
                                         ("stage", "development"), ("schedule", "stationary"),
                                         ("cohort", "development"), ("replica", 8), ("replica", "0"),
                                         ("arm", "covariance"), ("arm", None), ("anchor_refresh", 64),
                                         ("schedule_factor", 1.25), ("schedule_factor", None)])
def test_reject_wrong_physical_packages_stages_cohorts_replicas_and_assignment(field, value):
    rows = block()
    rows[-1][field] = value
    with pytest.raises(ValueError):
        analysis.summarize(rows, selection())


def test_unselected_settings_are_invalid_even_for_an_incomplete_prefix():
    rows = block()[:2] + [trajectory("kalman_zero", completed=False)]
    rows[-1]["ridge"] = .1
    with pytest.raises(ValueError, match="physical condition/c/ridge"):
        analysis.summarize(rows, selection())


def test_duplicates_cannot_complete_a_missing_arm_or_hide_behind_new_unit_ids():
    rows = block()[:2]
    duplicate = deepcopy(rows[1])
    duplicate["unit_id"] += "-duplicate"
    rows.append(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        analysis.summarize(rows, selection())
    rows = block()
    rows[1]["unit_id"] = rows[0]["unit_id"]
    with pytest.raises(ValueError, match="duplicate"):
        analysis.summarize(rows, selection())


@pytest.mark.parametrize("field", ("asset_unit", "initial_unit", "stream_hash"))
@pytest.mark.parametrize("value", (None, "", "not-the-paired-asset"))
def test_false_pairing_or_missing_identity_is_rejected_in_partial_blocks(field, value):
    rows = block()[:2]
    rows[1][field] = value
    with pytest.raises(ValueError, match=field):
        analysis.summarize(rows, selection())


@pytest.mark.parametrize("field", ("arrivals", "step", "p", "segment", "expected_nines", "observed_nines"))
def test_time_and_exposure_grid_mismatch_is_rejected(field):
    rows = block()
    rows[-1]["metrics"][1][field] = "other" if field == "segment" else rows[-1]["metrics"][1][field] + 1
    with pytest.raises(ValueError):
        analysis.summarize(rows, selection())


@pytest.mark.parametrize("field", ("nine_recall", "old_accuracy", "nll", "class_accuracy", "initial_nines", "q"))
def test_shared_initial_predictive_metrics_and_initial_state_must_match(field):
    rows = block()[:2]
    if field == "class_accuracy":
        rows[1]["metrics"][0][field][0] -= .01
    else:
        rows[1]["metrics"][0][field] += .01
    with pytest.raises(ValueError, match="initial"):
        analysis.summarize(rows, selection())


def test_prefixes_share_the_same_grid_and_an_empty_arm_cannot_mask_false_pairing():
    rows = [trajectory("fixed_ridge"), trajectory("fixed_zero", completed=False),
            trajectory("kalman_zero", completed=False)]
    rows[-1]["metrics"][1]["step"] += 1
    with pytest.raises(ValueError, match="grid"):
        analysis.summarize(rows, selection())
    rows = [trajectory("fixed_ridge"), trajectory("fixed_zero", completed=False, empty=True),
            trajectory("kalman_zero")]
    rows[-1]["metrics"][0]["nine_recall"] += .01
    with pytest.raises(ValueError, match="initial"):
        analysis.summarize(rows, selection())
    rows = [trajectory("fixed_ridge"), trajectory("fixed_zero", completed=False)]
    rows[-1]["metrics"] = rows[-1]["metrics"][1:]
    with pytest.raises(ValueError, match="initial checkpoint"):
        analysis.summarize(rows, selection())


def test_shared_whole_block_draws_cover_all_contrasts_outcomes_regimes_and_costs(monkeypatch):
    rows = [t for r in range(4) for t in block(r)]
    calls = []
    estimate = historical._estimate

    def tracked(values, **kwargs):
        if len(values):
            calls.append((len(values), kwargs["seed"]))
        return estimate(values, **kwargs)

    monkeypatch.setattr(historical, "_estimate", tracked)
    result = analysis.summarize(rows, selection(), bootstrap_draws=200, seed=7)
    assert len(set(calls)) == 1 and calls[0][0] == 4
    assert all(c["n_pairs"] == 4 for c in result["comparisons"])
    assert result == analysis.summarize(deepcopy(rows), selection(), bootstrap_draws=200, seed=7)
    reversed_result = analysis.summarize(list(reversed(rows)), selection(), bootstrap_draws=200, seed=7)
    for name in ("comparisons", "replicas", "eligible_replicas", "awaiting_full_block_units"):
        assert result[name] == reversed_result[name]


def test_cost_subsets_keep_undefined_ratios_explicit_and_shared_subset_draws(monkeypatch):
    rows = block(0) + block(1)
    rows[0]["summary"]["costs"]["controller_seconds"] = 0.
    rows[3]["summary"]["costs"].pop("curvature_seconds")
    result = analysis.summarize(rows, selection(2), bootstrap_draws=200)
    costs = contrast(result)["costs"]
    assert costs["controller_seconds"]["difference"]["n_pairs"] == 2
    assert costs["controller_seconds"]["ratio"]["n_pairs"] == 1
    assert costs["curvature_seconds"]["ratio"]["n_pairs"] == 1
    assert costs["learning_seconds"]["difference"]["n_pairs"] == 1
    assert costs["wall_seconds"]["difference"]["n_pairs"] == 2


@pytest.mark.parametrize("change", ("schema", "protocol", "frozen", "frozen_at", "timezone", "arms",
                                    "ridge", "count", "stationary", "target", "factors", "partition", "seed"))
def test_frozen_design_and_primary_partition_validation(change):
    design = selection()
    if change == "schema":
        design["schema"] = True
    elif change == "protocol":
        design["protocol"] = analysis.PARTITION_PROTOCOL
    elif change == "frozen":
        design["frozen"] = False
    elif change == "frozen_at":
        design["frozen_at"] = "pending"
    elif change == "timezone":
        design["frozen_at"] = "2026-11-02T20:00:00"
    elif change == "arms":
        design["arms"].pop("fixed_zero")
    elif change == "ridge":
        design["arms"]["fixed_zero"]["ridge"] = .1
    elif change == "count":
        design["main_replicas"] = 9
    elif change == "stationary":
        design["stationary_replicas"] = 2
    elif change == "target":
        design["fresh_trajectory_target"] = 3
    elif change == "factors":
        design["schedule_factors"] = [1., 1.5]
    elif change == "partition":
        design["data_partition"]["protocol"] = analysis.PROTOCOL
    else:
        design["data_partition"]["train_seed"] += 1
    with pytest.raises(ValueError):
        analysis.summarize([], design)


def test_incompatible_or_incomplete_scalar_artifacts_are_not_repaired():
    rows = block()
    rows[0]["metrics"] = rows[0]["metrics"][:1]
    with pytest.raises(ValueError, match="initial and final"):
        analysis.summarize(rows, selection())
    rows = block()
    rows[-1]["metrics"][1]["nine_recall"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        analysis.summarize(rows, selection())
    with pytest.raises(ValueError, match="list"):
        analysis.summarize({}, selection())


def test_no_retention_selector_or_primary_summary_is_invoked(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("supplement must never invoke primary selection or inference")

    for name in ("candidate_table", "select_candidate", "refinement_candidate", "summarize"):
        monkeypatch.setattr(primary, name, forbidden)
    result = analysis.summarize(block(), selection())
    assert len(result["comparisons"]) == 3
    assert "development" not in result


def test_bootstrap_controls_are_bounded_and_invalid_inputs_fail():
    assert analysis.summarize([], selection(), bootstrap_draws=-1)["bootstrap"]["draws"] == 0
    assert analysis.summarize([], selection(), bootstrap_draws=100_000)["bootstrap"]["draws"] == 20_000
    for kwargs in ({"bootstrap_draws": True}, {"bootstrap_draws": 3.5}, {"seed": True}, {"seed": -1}):
        with pytest.raises(ValueError):
            analysis.summarize([], selection(), **kwargs)


def test_pp_interval_scaling_uses_replica_differences_not_checkpoint_resampling():
    rows = []
    for r in range(4):
        cohort = block(r)
        cohort[-1] = trajectory("kalman_zero", r, nine_gap_pp=float(r), old_gap_pp=float(r) / 2)
        rows.extend(cohort)
    effect = contrast(analysis.summarize(rows, selection(), bootstrap_draws=2000, seed=43))
    nine = effect["observation_mean_pp"]["nine_recall"]
    old = effect["observation_mean_pp"]["old_accuracy"]
    assert nine["n_pairs"] == 4 and nine["mean"] == pytest.approx(1.5)
    assert old["mean"] == pytest.approx(.75)
    assert np.asarray(old["ci95"]) * 2 == pytest.approx(nine["ci95"])
    assert nine["interval_reliable"] is True
