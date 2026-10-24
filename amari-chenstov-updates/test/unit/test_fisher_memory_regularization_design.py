"""Fast, synthetic-record checks for Phase 7; no data, networks, or artifacts."""

import copy
import dataclasses
import json

import pytest

from mnist_experiment.fisher_memory import regularization_design as design
from mnist_experiment.fisher_memory.config import Config, Treatment
from mnist_experiment.fisher_memory.metrics import paired_summary
from src.seeding import derive_component_seed


def config(smoke=False):
    return Config(smoke=smoke, device="cpu", inner_steps=2, max_evaluations=5,
                  bootstrap_samples=31)


def recipe(ridge=.1, strength=0., *, smoke=False):
    cfg = config(smoke)
    return design.treatment(cfg, "finite_difference" if strength else "replay_ewc",
                            ridge, strength, design.ledger(cfg)["default_policy"])


def record(choice, replica=0, stream="mixed", acquisition=2., base=.4, *, target=100, seconds=10.):
    name = f"trajectory-{design.CONF_FAMILY}-{replica:04d}-{stream}-{choice.key}"
    rows = [dict(paired_rotated_event=event, actual_rotated_events=event if stream == "mixed" else 0,
                 rotated_nll=acquisition if event else 4., base_nll=base if event else .2,
                 rotated_accuracy=.6, base_accuracy=.9)
            for event in range(target + 1)]
    return dict(schema_version=1, unit_id=name, replica=replica, stream=stream,
                condition=choice.condition, strength=choice.strength,
                treatment=choice.mapping(), treatment_id=choice.key, rows=rows,
                event_target=target, rotated_auc=acquisition, base_auc=base,
                final_rotated_nll=acquisition, final_base_nll=base,
                final_rotated_accuracy=.6, final_base_accuracy=.9,
                active_seconds=seconds, closure_calls=15, exposure=1080,
                solver_failures=0, nonfinite=False, under_solved=5)


def ranking_row(ridge=.1, strength=0., acquisition=2., harm=0., seconds=10., valid=True):
    return dict(treatment=recipe(ridge, strength).mapping(), valid=valid,
                mean_acquisition=acquisition, mean_harm=harm, online_seconds=seconds, pairs=[])


def confirmation(smoke=False, same_ridge=False):
    cfg = config(smoke)
    replay, fd = recipe(.3, smoke=smoke), recipe(.3 if same_ridge else 1., .269 if smoke else 160.89, smoke=smoke)
    selection = dict(replay=replay.mapping(), fd=fd.mapping(),
                     policy=design.ledger(cfg)["default_policy"], warnings=["exploratory selection"])
    return design.confirmation_ledger(cfg, selection, {"selection_seal": "sealed"})


def cohort(plan, gains=None):
    gains = [1.] * plan["replicas"] if gains is None else gains
    records = []
    for replica, gain in enumerate(gains):
        for stream in ("mixed", "base"):
            for mapping in plan["treatments"]:
                choice = Treatment(**mapping)
                acquisition, base = 3., .4
                if choice.condition == "finite_difference":
                    acquisition, base = (3. - gain if stream == "mixed" else 2.75), .42
                elif mapping == plan["shared_ridge_baseline"] and mapping != plan["baseline"]:
                    acquisition, base = 2.8, .41
                records.append(record(choice, replica, stream, acquisition, base, target=plan["event_target"]))
    return records


def metadata(row, plan):
    result = {key: row[key] for key in ("schema_version", "replica", "stream", "condition",
                                      "event_target", "treatment", "treatment_id")}
    result.update(kind="drip_trajectory", phase="7", family=design.CONF_FAMILY,
                  protocol=design.PROTOCOL, config_hash=plan["config_hash"],
                  source_hash="scientific-source", asset_id=f"assets-{row['replica']}",
                  plan_hash=f"plan-{row['replica']}")
    return result


def effects(result):
    yield from (result[key] for key in ("primary", "matched_ridge", "retention_harm",
                                      "matched_ridge_retention_harm", "base_only_gain", "interaction"))
    for field in ("frozen_gain", "base_deterioration"):
        for streams in result[field].values():
            yield from streams.values()


def test_frozen_protocol_ledgers_and_named_seeds():
    cfg = config()
    plan = design.ledger(cfg)
    assert design.PROTOCOL == "plan15-regularization-v1"
    assert design.DEV_FAMILY == "regularization_development"
    assert design.CONF_FAMILY == "regularization_confirmation"
    assert design.LAMBDA0 == 160.89
    assert plan["schema_version"] == 1 and plan["protocol"] == design.PROTOCOL
    assert not plan["smoke"]
    assert (plan["development_replicas"], plan["confirmation_replicas"], plan["event_target"]) == (4, 12, 100)
    assert plan["ridge_grid"] == [.03, .1, .3, 1.]
    assert plan["audit_events"] == [10, 50, 90]
    assert plan["default_policy"] == dict(inner_steps=100, max_evaluations=150)
    assert plan["repair_policy"] == dict(inner_steps=200, max_evaluations=300)
    assert plan["audit_repair_threshold_count"] == 12
    assert plan["lambda0"] == 160.89 and plan["retention_margin"] == .01
    assert plan["bootstrap_samples"] == 2000
    assert plan["inference_seed"] == cfg.seed_for(design.CONF_FAMILY, 0, "replica_bootstrap")
    assert plan["inference_seed"] != cfg.seed_for(design.DEV_FAMILY, 0, "replica_bootstrap")
    assert design.decision_id("selection") == "regularization-selection-v1"
    json.dumps(plan, allow_nan=False)


def test_smoke_ledger_scales_only_software_work():
    cfg = config(True)
    plan = design.ledger(cfg)
    assert plan["smoke"]
    assert (plan["development_replicas"], plan["confirmation_replicas"], plan["event_target"]) == (1, 2, 2)
    assert plan["ridge_grid"] == [.1, .3] and plan["audit_events"] == [1, 2]
    assert plan["default_policy"] == dict(inner_steps=2, max_evaluations=5)
    assert plan["repair_policy"] == dict(inner_steps=4, max_evaluations=10)
    assert plan["audit_repair_threshold_count"] == 2 and plan["lambda0"] == .269
    assert plan["bootstrap_samples"] == cfg.bootstrap_samples
    choice = design.treatment(cfg, "finite_difference", .3, .269, plan["repair_policy"])
    assert choice == Treatment.default(cfg, "finite_difference", .269, ridge_multiplier=.3,
                                       inner_steps=4, max_evaluations=10)


def test_table_uses_fixed_reference_mixed_only_and_never_excludes_under_solving():
    reference, candidate = recipe(), recipe(.3, 160.89)
    records = []
    for replica in range(4):
        records.extend([record(reference, replica, acquisition=1., base=.4 + replica),
                        record(candidate, replica, acquisition=2., base=.405 + replica),
                        record(reference, replica, "base", acquisition=999., base=.01, seconds=999.),
                        record(candidate, replica, "base", acquisition=.01, base=999., seconds=999.)])
    table = design.build_table(records, [reference, candidate], reference, 4)
    row = table[1]
    assert row["valid"] and len(row["pairs"]) == 4
    assert row["mean_acquisition"] == 2. and row["online_seconds"] == 10.
    assert row["mean_harm"] == pytest.approx(.005)
    assert design.select([row], .01) == (row, "retention_feasible")
    assert design.build_table(list(reversed(records)), [reference, candidate], reference, [0, 1, 2, 3]) == table
    json.dumps(table, allow_nan=False)


@pytest.mark.parametrize("problem", ["missing", "failure", "nonfinite", "nan", "missing_reference", "time"])
def test_table_requires_every_replica_and_finite_healthy_reference(problem):
    reference, candidate = recipe(), recipe(.3, 160.89)
    records = [record(choice, r, stream) for stream in ("base", "mixed")
               for r in range(4) for choice in (reference, candidate)]
    if problem == "missing":
        records.pop()
    elif problem == "missing_reference":
        records.pop(-2)
    else:
        key, value = {"failure": ("solver_failures", 1), "nonfinite": ("nonfinite", True),
                      "nan": ("rotated_auc", float("nan")), "time": ("active_seconds", float("inf"))}[problem]
        records[-1][key] = value
    row = design.build_table(records, [candidate], reference, 4)[0]
    assert not row["valid"] and len(row["pairs"]) == 4
    with pytest.raises(ValueError, match="all development candidates"):
        design.select([row], .01)
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("arm", ["candidate", "reference"])
@pytest.mark.parametrize("problem", ["missing", "failure", "nonfinite", "nan", "time", "incomplete"])
def test_base_only_candidate_and_reference_gate_validity_without_entering_scores(arm, problem):
    reference, candidate = recipe(), recipe(.3, 160.89)
    records = [record(choice, r, stream) for stream in ("base", "mixed")
               for r in range(4) for choice in (reference, candidate)]
    healthy = design.build_table(records, [reference, candidate], reference, 4)
    choice = candidate if arm == "candidate" else reference
    bad = next(row for row in records if row["replica"] == 3 and row["stream"] == "base"
               and row["treatment_id"] == choice.key)
    if problem == "missing":
        records.remove(bad)
    else:
        key, value = {"failure": ("solver_failures", 1), "nonfinite": ("nonfinite", True),
                      "nan": ("base_auc", float("nan")), "time": ("active_seconds", float("inf")),
                      "incomplete": ("completed", False)}[problem]
        bad[key] = value
    table = design.build_table(records, [reference, candidate], reference, 4)
    row = table[1]
    assert not row["valid"] and not row["pairs"][-1]["valid"]
    assert all(pair["valid"] for pair in row["pairs"][:-1])
    assert table[0]["valid"] == (arm == "candidate")
    for field in ("mean_acquisition", "mean_harm", "online_seconds"):
        assert row[field] == healthy[1][field]
    with pytest.raises(ValueError, match="all development candidates"):
        design.select([row], .01)
    json.dumps(table, allow_nan=False)


def test_duplicates_and_changed_frozen_policies_are_rejected():
    reference = recipe()
    with pytest.raises(ValueError, match="duplicate development trajectory"):
        design.build_table([record(reference), record(reference)], [reference], reference, 4)
    with pytest.raises(ValueError, match="duplicate development recipe"):
        design.build_table([], [reference, reference], reference, 4)
    with pytest.raises(ValueError, match="frozen common policy"):
        design.build_table([], [dataclasses.replace(reference, inner_steps=200)], reference, 4)
    with pytest.raises(ValueError, match="c=0.1"):
        design.build_table([], [reference], recipe(.3), 4)


def test_ranking_margin_fallback_and_exact_lexicographic_ties():
    infeasible = ranking_row(acquisition=.1, harm=.010001)
    feasible = ranking_row(.3, acquisition=5., harm=.01)
    assert design.select([infeasible, feasible], .01) == (feasible, "retention_feasible")
    chosen, status = design.select([infeasible], .01)
    assert chosen is infeasible and status == "retention_warning_acquisition_fallback"
    rows = [ranking_row(.3, 2., seconds=3.), ranking_row(.1, 2., seconds=3.),
            ranking_row(.1, 1., seconds=3.), ranking_row(.03, .5, seconds=4.)]
    assert design.select(rows, .01)[0] is rows[2]
    faster = ranking_row(1., 3., seconds=2.)
    assert design.select([*rows, faster], .01)[0] is faster
    better = ranking_row(1., 3., acquisition=1., seconds=100.)
    assert design.select([faster, better], .01)[0] is better


def test_shortlist_selects_methods_separately_and_preserves_warnings():
    rows = [ranking_row(.3, acquisition=2.), ranking_row(.1, acquisition=3.),
            ranking_row(1., 160.89, acquisition=1., harm=.02)]
    selected = design.shortlist(rows)
    assert selected["replay"] == rows[0]["treatment"]
    assert selected["fd"] == rows[2]["treatment"]
    assert selected["ridges"] == [.3, 1.]
    assert selected["statuses"]["replay"] == "retention_feasible"
    assert selected["warnings"] == ["fd: retention_warning_acquisition_fallback"]
    assert not design.shortlist(rows, .1)["warnings"]
    rows[2]["treatment"]["ridge_multiplier"] = .3
    assert design.shortlist(rows)["ridges"] == [.3]
    with pytest.raises(ValueError, match="all development candidates"):
        design.shortlist(rows[:2])


@pytest.mark.parametrize("winner, expected", [(0, design.LAMBDA0 / 9), (1, None), (2, 9 * design.LAMBDA0)])
def test_bounded_coefficient_extension_uses_same_mixed_ranking(winner, expected):
    rows = [ranking_row(.3, strength, acquisition=2. if i == winner else 3.)
            for i, strength in enumerate((design.LAMBDA0 / 3, design.LAMBDA0, 3 * design.LAMBDA0))]
    assert design.extension(rows, design.LAMBDA0, .01) == expected
    rows[winner]["mean_harm"] = .02
    assert design.extension(rows, design.LAMBDA0, .01) == (None if winner == 0 else design.LAMBDA0 / 9)
    extended = [ranking_row(.3, design.LAMBDA0 / 9, acquisition=1.)]
    assert design.extension(extended, design.LAMBDA0, .01) is None


def test_confirmation_recipe_deduplication_policy_and_defensive_copy():
    assert len(confirmation()["treatments"]) == 3
    same = confirmation(same_ridge=True)
    assert len(same["treatments"]) == 2 and same["baseline"] == same["shared_ridge_baseline"]
    assert same["replicas"] == 12 and same["alpha"] == .05
    assert same["confirmation_replicas"] == same["replicas"]
    assert len(same["trajectory_ids"]) == len(set(same["trajectory_ids"])) == 48
    assert same["source"] == {"selection_seal": "sealed"}
    cfg = config()
    selection = dict(replay=recipe(.3).mapping(), fd=recipe(.3, 160.89).mapping(),
                     policy=design.ledger(cfg)["default_policy"])
    plan = design.confirmation_ledger(cfg, selection, {})
    selection["policy"]["inner_steps"] = 999
    assert plan["policy"]["inner_steps"] == 100
    with pytest.raises(ValueError, match="frozen configuration/policy"):
        design.confirmation_ledger(cfg, selection, {})
    json.dumps(same, allow_nan=False)


def test_complete_confirmation_effect_signs_single_exact_test_and_frozen_comparisons():
    plan = confirmation()
    records = cohort(plan)
    result = design.analyze(records, plan)
    assert result["complete"] and result["inference_ready"]
    assert result["completed"] == result["expected"] == 72 and result["n_pairs"] == 12
    assert result["primary"]["mean_gain"] == 1. and result["primary"]["ci"] == [1., 1.]
    assert result["primary"]["p_value"] == 1 / 4096
    assert result["primary"]["favorable_replicas"] == 12
    assert result["matched_ridge"]["mean_gain"] == pytest.approx(.8)
    assert result["retention_harm"]["mean_gain"] == pytest.approx(.02)
    assert result["matched_ridge_retention_harm"]["mean_gain"] == pytest.approx(.01)
    assert result["base_only_gain"]["mean_gain"] == .25
    assert result["interaction"]["mean_gain"] == .75
    assert all(effect["p_value"] is None for effect in effects(result) if effect is not result["primary"])
    for mapping in plan["treatments"]:
        key = Treatment(**mapping).key
        arm = next(arm for arm in result["arm_summaries"] if arm["treatment_id"] == key and arm["stream"] == "mixed")
        assert result["frozen_gain"][key]["mixed"]["mean_gain"] == pytest.approx(4. - arm["rotated_auc"])
        assert result["base_deterioration"][key]["mixed"]["mean_gain"] == pytest.approx(arm["base_auc"] - .2)
        assert arm["initial_rotated_nll"] == 4. and arm["initial_base_nll"] == pytest.approx(.2)
    assert result == design.analyze(list(reversed(records)), plan)
    assert result["source_lineage"] == plan["source"]
    json.dumps(result, allow_nan=False)


def test_bootstrap_matches_whole_replica_estimator_and_named_analysis_seed():
    plan = confirmation(same_ridge=True)
    gains = [.1] * 11 + [2.]
    result = design.analyze(cohort(plan, gains), plan)
    primary = result["primary"]
    seed = derive_component_seed(plan["inference_seed"], "bootstrap:primary")
    expected = paired_summary(dict(enumerate(gains)), dict.fromkeys(range(12), 0.), seed=seed, resamples=2000)
    assert primary["seed"] == seed and primary["resamples"] == 2000
    assert primary["ci"] == pytest.approx(expected["ci"])
    assert primary["asymmetry_warning"] and primary["sample_skewness"] > 1.
    assert result["matched_ridge_is_primary"]
    assert result["matched_ridge"]["ci"] == primary["ci"]
    assert result["matched_ridge"]["p_value"] is None


@pytest.mark.parametrize("mode", ["missing", "incomplete", "missing_matched", "no_records", "smoke"])
def test_global_inference_gate_even_if_primary_pairs_are_all_present(mode):
    plan = confirmation(smoke=mode == "smoke")
    records = cohort(plan)
    if mode == "missing":
        records.pop()
    elif mode == "incomplete":
        records[-1]["completed"] = False
        records[-1]["rows"] = []
    elif mode == "missing_matched":
        records = [row for row in records if row["treatment"] != plan["shared_ridge_baseline"]]
    elif mode == "no_records":
        records = []
    result = design.analyze(records, plan)
    assert not result["inference_ready"]
    assert result["complete"] == (mode == "smoke")
    assert all(effect["ci"] is None and effect["p_value"] is None for effect in effects(result))
    json.dumps(result, allow_nan=False)


def test_solver_failures_are_disclosed_without_dropping_replicas_or_releasing_inference():
    plan = confirmation()
    records = cohort(plan)
    records[0].update(solver_failures=2, nonfinite=True)
    result = design.analyze(records, plan)
    assert result["complete"] and result["n_pairs"] == 12 and not result["inference_ready"]
    assert result["health"]["solver_failures"] == 2
    assert result["health"]["nonfinite_trajectories"] == 1
    assert result["health"]["under_solved_updates"] == 360
    assert all(effect["ci"] is None and effect["p_value"] is None for effect in effects(result))


@pytest.mark.parametrize("audit_label", [None, False, True])
@pytest.mark.parametrize("residual_count", [0, 5])
def test_optimization_limited_uses_sealed_audit_not_residual_counts(audit_label, residual_count):
    plan = confirmation(smoke=True)
    if audit_label is not None:
        plan["selection"]["optimization_limited"] = audit_label
    records = cohort(plan)
    for row in records:
        row["under_solved"] = residual_count
    result = design.analyze(records, plan)
    assert result["health"]["under_solved_updates"] == len(records) * residual_count
    assert result["health"]["optimization_limited"] is (False if audit_label is None else audit_label)


@pytest.mark.parametrize("mode", ["partial", "smoke", "failed"])
def test_gated_analysis_never_calls_bootstrap_or_sign_flip(monkeypatch, mode):
    plan = confirmation(smoke=mode == "smoke")
    records = cohort(plan)
    if mode == "partial":
        records.pop()
    elif mode == "failed":
        records[0]["solver_failures"] = 1

    def forbidden(*args, **kwargs):
        pytest.fail("inference was executed before its release gate")

    monkeypatch.setattr(design, "paired_summary", forbidden)
    monkeypatch.setattr(design, "_exact_sign_flip", forbidden)
    assert not design.analyze(records, plan)["inference_ready"]


@pytest.mark.parametrize("mutation", ["duplicate", "replica", "treatment", "family", "protocol", "schema",
                                      "events", "actual", "auc", "endpoint", "nan", "initial", "metadata",
                                      "class_counts", "unpaired_assets", "foreign_unit"])
def test_confirmation_rejects_corrupt_or_unplanned_records_before_inference(mutation):
    plan = confirmation()
    records = cohort(plan)
    row = records[0]
    if mutation == "duplicate":
        records.append(copy.deepcopy(row))
    elif mutation == "replica":
        row["replica"] = 12
    elif mutation == "treatment":
        row["treatment"]["ridge_multiplier"] = .03
    elif mutation in ("family", "protocol"):
        row[mutation] = "development"
    elif mutation == "schema":
        row["schema_version"] = 2
    elif mutation == "events":
        row["rows"][-1]["paired_rotated_event"] = 99
    elif mutation == "actual":
        row["rows"][-1]["actual_rotated_events"] = 0
    elif mutation == "auc":
        row["rotated_auc"] += .1
    elif mutation == "endpoint":
        row["final_base_accuracy"] = .1
    elif mutation == "nan":
        row["rows"][0]["base_nll"] = float("nan")
    elif mutation == "initial":
        row["rows"][0]["rotated_nll"] = 8.
    elif mutation == "metadata":
        row["metadata"] = dict(replica=1)
    elif mutation == "class_counts":
        row["per_class_events"] = {str(k): 1 for k in range(10)}
    elif mutation == "unpaired_assets":
        row["metadata"] = dict(asset_id="first")
        records[1]["metadata"] = dict(asset_id="other")
    else:
        row["unit_id"] = "trajectory-from-old-study"
    with pytest.raises(ValueError):
        design.analyze(records, plan)


def test_artifact_wrappers_are_equivalent_to_augmented_summary_records():
    plan = confirmation(smoke=True)
    records = cohort(plan)
    units = []
    for row in records:
        summary = {key: value for key, value in row.items() if key not in ("rows", "treatment", "treatment_id")}
        units.append(dict(summary=summary, metadata=metadata(row, plan), rows=row["rows"], completed=True))
    assert design.analyze(units, plan) == design.analyze(records, plan)
    units[-1].update(summary=None, completed=False, rows=[])
    assert not design.analyze(units, plan)["complete"]


@pytest.mark.parametrize("field,value", [("replicas", 11), ("event_target", 99), ("protocol", "old"),
                                        ("family", design.DEV_FAMILY), ("bootstrap_samples", 100),
                                        ("alpha", .1), ("inference_seed", -1),
                                        ("confirmation_replicas", 11), ("schema_version", True)])
def test_confirmation_rejects_changed_frozen_design(field, value):
    plan = confirmation()
    plan[field] = value
    with pytest.raises(ValueError):
        design.analyze([], plan)


def test_duplicate_confirmation_recipe_is_not_an_extra_arm():
    plan = confirmation(same_ridge=True)
    plan["treatments"].append(plan["baseline"])
    with pytest.raises(ValueError, match="recipe ledger"):
        design.analyze([], plan)


def test_scheduler_style_flat_metadata_and_merged_smoke_ledger():
    cfg = config(True)
    plan = design.ledger(cfg)
    plan.update(confirmation(smoke=True))
    assert plan["replicas"] == plan["confirmation_replicas"] == 2
    assert plan["lambda0"] == .269 and plan["audit_events"] == [1, 2]
    assert plan["development_replicas"] == 1 and plan["ridge_grid"] == [.1, .3]
    records = cohort(plan)
    for row in records:
        row["metadata"] = metadata(row, plan)
    result = design.analyze(records, plan)
    assert result["complete"] and not result["inference_ready"]
    assert result["completed"] == len(plan["trajectory_ids"]) == 12


@pytest.mark.parametrize("field", ["source_hash", "config_hash", "protocol", "phase", "kind", "treatment_id"])
def test_metadata_must_match_frozen_source_and_protocol(field):
    plan = confirmation(smoke=True)
    plan["source"]["identity"] = {"source_hash": "scientific-source", "config_hash": plan["config_hash"]}
    records = cohort(plan)
    for row in records:
        row["metadata"] = metadata(row, plan)
    records[0]["metadata"][field] = "foreign"
    with pytest.raises(ValueError):
        design.analyze(records, plan)


def test_unpaired_metadata_assets_and_source_versions_are_not_pooled():
    plan = confirmation(smoke=True)
    records = cohort(plan)
    for row in records:
        row["metadata"] = metadata(row, plan)
    for field in ("source_hash", "plan_hash", "asset_id"):
        changed = copy.deepcopy(records)
        changed[0]["metadata"][field] = "different"
        with pytest.raises(ValueError):
            design.analyze(changed, plan)


def test_consistent_but_foreign_source_is_not_accepted_as_the_frozen_source():
    plan = confirmation(smoke=True)
    plan["source"]["identity"] = {"source_hash": "expected-source", "config_hash": plan["config_hash"]}
    records = cohort(plan)
    for row in records:
        row["metadata"] = metadata(row, plan)
    with pytest.raises(ValueError, match="frozen protocol/source/configuration"):
        design.analyze(records, plan)


@pytest.mark.parametrize("differences, expected", [([1., 1.], .25), ([-1., -1.], 1.),
                                                   ([0., 0.], 1.), ([1., -1.], .75)])
def test_exact_sign_flip_includes_ties(differences, expected):
    assert design._exact_sign_flip(differences) == expected
