"""Fixed-reference, seed, admission and recovery contracts without training."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mnist_experiment.combined_memory import runtime as original
from mnist_experiment.combined_memory.low_ridge import calibration, runtime
from mnist_experiment.combined_memory.low_ridge.config import Config
from mnist_experiment.combined_memory.low_ridge.run import Scheduler, validate_import_pairing
from mnist_experiment.combined_memory.retention_match.config import Config as HistoryConfig
from mnist_experiment.fisher_memory.artifacts import ArtifactError, Store


def records(cfg):
    return [dict(unit_id=f"{arm}-{member}", status="completed",
        metadata=dict(kind="trajectory", cohort="development", replica=member, recipe=recipe.mapping()),
        summary=dict(Havg=2., H100=4., novel_auc=8., learning_seconds=1., online_wall_seconds=2.))
        for arm, recipe in calibration.initial_recipes(cfg.fd_strength).items() for member in range(4)]


@pytest.fixture
def scheduler(tmp_path):
    cfg = Config.smoke_config()
    with Store(tmp_path / "run", dict(protocol=cfg.protocol, config=cfg.mapping()), free_reserve_bytes=0) as store:
        obj = Scheduler.__new__(Scheduler)
        obj.cfg, obj.store = cfg, store
        obj.budget = SimpleNamespace(check=lambda:None, remaining=100000., used=0., allocation_seconds=100000.)
        obj.records = {r["unit_id"]:r for r in records(cfg)}
        obj.selection, obj.numeric, obj.initial_cache = None, {}, {}
        obj.reference_arm, obj.reference_ridge, obj.current_check, obj.rounds = "B", 1., None, []
        obj.phase, obj.current, obj.calibration_status = "7C: testing", {}, "pending"
        yield obj


def freeze(obj):
    chosen = calibration.select_tuple(list(obj.records.values()), strength=obj.cfg.fd_strength)
    obj.selection = obj.decision("selection", lambda:obj.selection_payload(chosen))
    return obj.selection


def test_production_and_all_four_development_seeds_preserve_scientific_policy():
    cfg, old = Config(), HistoryConfig()
    cfg.validate()
    for member in range(4):
        for component in ("initialization_model", "factors", "stream:transforms", "replay:12", "probes:12"):
            assert cfg.seed_for("development", member, component) == old.seed_for("development", member, component)
            assert cfg.seed_for("fresh", member, component) != old.seed_for("fresh", member, component)
    assert cfg.seed_for("study-partition", 0, "test") == old.seed_for("study-partition", 0, "test")
    for changes in (dict(main_replicas=4), dict(fd_strength=.1), dict(retention_tolerance=.2),
                    dict(refinements_per_arm=2), dict(force_calibration_failure=True), dict(capacity=64)):
        with pytest.raises(ValueError):
            replace(cfg, **changes).validate()


@pytest.mark.parametrize("cohort", ["fresh", "stationary"])
def test_confirmation_requires_a_passing_seal(scheduler, cohort):
    with pytest.raises(ArtifactError, match="sealed passing calibration"):
        scheduler.trajectory(cohort, 0, calibration.initial_recipes(scheduler.cfg.fd_strength)["B"])
    assert not list((scheduler.store.root / "units").glob("path-*"))


def test_reference_roles_are_sealed_and_cannot_swap(scheduler):
    sealed = deepcopy(freeze(scheduler))
    assert sealed["reference_arm"] == "B" and sealed["efficiency_reference_arm"] == "A"
    assert sealed["recipes"]["B"]["ridge"] == 1.
    assert scheduler.validate_selection() == sealed
    scheduler.selection["efficiency_reference_arm"] = "B"
    with pytest.raises(ArtifactError, match="sealed scientific decision"):
        scheduler.validate_selection()
    scheduler.selection = sealed
    scheduler.records.pop("B-3")
    with pytest.raises(ValueError, match="missing or incomplete"):
        scheduler.validate_selection()


def test_b_ridge_and_forced_failure_cannot_be_sealed(scheduler):
    chosen = calibration.select_tuple(list(scheduler.records.values()), strength=scheduler.cfg.fd_strength)
    wrong = deepcopy(chosen)
    wrong["recipes"]["B"]["ridge"] = 10.
    with pytest.raises(ArtifactError, match="B must remain fixed"):
        scheduler.selection_payload(wrong)
    scheduler.cfg = replace(scheduler.cfg, force_calibration_failure=True)
    with pytest.raises(ArtifactError, match="refusing to seal"):
        scheduler.selection_payload(chosen)


def test_confirmation_recipe_must_equal_selected_recipe(scheduler):
    freeze(scheduler)
    wrong = replace(calibration.initial_recipes(scheduler.cfg.fd_strength)["A"], ridge=3.)
    with pytest.raises(ArtifactError, match="differs from its calibration seal"):
        scheduler.trajectory("fresh", 0, wrong)


def test_partially_executed_round_does_not_reset_allowances(scheduler):
    starting = calibration.initial_recipes(scheduler.cfg.fd_strength)
    decision = dict(recipes=[replace(starting[a], ridge=.5).mapping() for a in "ACDE"])
    scheduler.decision("refinement-0", lambda:decision)
    assert scheduler.used_allowances() == dict.fromkeys("ACDE", 1)
    scheduler.store.resume = True
    assert scheduler.decision("refinement-0", lambda:pytest.fail("repeated decision")) == decision
    assert scheduler.used_allowances() == dict.fromkeys("ACDE", 1)


def test_stop_finalization_can_publish_status_without_reentering_stop_check(scheduler):
    scheduler.store.control.stop_requested = True
    scheduler.publish("paused", figures=False)
    import json
    state = json.loads((scheduler.store.root / "status.json").read_text())
    assert state["status"] == "paused"
    assert state["candidate_allowances"] == dict.fromkeys("ACDE", 0)


def test_admission_reserves_entire_confirmation_allocation(scheduler):
    missing = replace(calibration.initial_recipes(scheduler.cfg.fd_strength)["A"], ridge=.5)
    needed = 4*scheduler.path_cost("A")*1.25 + scheduler.confirmation_reserve()
    scheduler.budget.remaining = needed-1e-6
    assert not scheduler.can_admit([missing])
    scheduler.budget.remaining = needed+1e-6
    assert scheduler.can_admit([missing])


def test_remaining_confirmation_reservation_omits_completed_work_and_initialization(scheduler):
    before = scheduler.confirmation_reserve(remaining=True)
    item = deepcopy(scheduler.records["A-0"])
    item["metadata"].update(cohort="fresh", replica=0)
    scheduler.records["fresh-0-A"] = item
    after = scheduler.confirmation_reserve(remaining=True)
    assert before-after == pytest.approx(1.25*(scheduler.path_cost("A") + scheduler.initial_cost()))


def test_progress_counts_admitted_missing_development_members(scheduler):
    assert scheduler.status()["expected_development_paths"] == 20
    scheduler.records.pop("B-3")
    assert scheduler.status()["completed"]["development"] == 19
    assert scheduler.status()["expected_development_paths"] == 20
    recipe = replace(calibration.initial_recipes(scheduler.cfg.fd_strength)["E"], ridge=.01)
    scheduler.rounds.append(dict(recipes=[recipe.mapping()]))
    assert scheduler.status()["expected_development_paths"] == 24


def test_budget_accepts_only_the_sealed_extended_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "authorization", lambda:dict(payload=dict(cumulative_ceiling_seconds=142504.)))
    monkeypatch.setattr(original, "peer_seconds", lambda root:70000.)
    with Store(tmp_path / "budget", dict(protocol="test"), free_reserve_bytes=0) as store:
        store.data_root = tmp_path
        with pytest.raises(ValueError, match="sealed Phase 7"):
            runtime.RuntimeBudget(store, allocation_seconds=200000.)
        budget = runtime.RuntimeBudget(store, other_seconds=70000., allocation_seconds=142504.)
        try:
            budget.check()
            assert budget.allocation_seconds == 142504.
            assert 72000. < budget.remaining < 72504.
        finally:
            budget.close()


@pytest.mark.parametrize("defect", ["asset", "model", "factor", "sign", "numerical", "replay_seed"])
def test_import_rejects_differently_paired_or_solved_history(defect):
    cfg = Config.smoke_config()
    recipe = calibration.initial_recipes(cfg.fd_strength)["B"]
    initial = dict(pairing_asset_hash="same-assets",
        model_seed=cfg.seed_for("development", 2, "initialization_model"),
        factor_seed=cfg.seed_for("development", 2, "factors"))
    assets = dict(asset_hash="same-assets", assignment=dict(global_sign=-1),
        seeds={"replay:0":cfg.seed_for("development", 2, "replay:0")})
    row = dict(metadata=dict(replica=2, recipe=recipe.mapping(), global_sign=-1,
        numerical_policy=dict(fd_dtype=cfg.fd_dtype,max_iter=cfg.max_iter,max_eval=cfg.max_eval), **initial))
    validate_import_pairing(row, assets, initial, cfg)
    if defect == "asset": row["metadata"]["pairing_asset_hash"] = "wrong"
    elif defect == "model": row["metadata"]["model_seed"] += 1
    elif defect == "factor": row["metadata"]["factor_seed"] += 1
    elif defect == "sign": row["metadata"]["global_sign"] = 1
    elif defect == "numerical": row["metadata"]["numerical_policy"]["max_iter"] += 1
    else: assets["seeds"]["replay:0"] += 1
    with pytest.raises(ArtifactError, match="historical"):
        validate_import_pairing(row, assets, initial, cfg)


def test_authorization_baseline_survives_restart_without_budget_reset(tmp_path, monkeypatch):
    family = tmp_path / "family"
    monkeypatch.setattr(runtime, "FAMILY", family)
    monkeypatch.setattr(runtime, "AUTHORIZATION", family / "authorization.json")
    monkeypatch.setattr(original, "peer_seconds", lambda root:100.)
    monkeypatch.setattr(original, "_charged_seconds", lambda root:10.)
    first = runtime.authorize()
    assert first["payload"]["baseline_seconds"] == 110.
    assert first["payload"]["cumulative_ceiling_seconds"] == 72110.
    data = runtime.AUTHORIZATION.read_bytes()
    monkeypatch.setattr(original, "peer_seconds", lambda root:99999.)
    assert runtime.authorize() == first
    assert runtime.AUTHORIZATION.read_bytes() == data
    bad = deepcopy(first)
    bad["payload"]["baseline_seconds"] = 99999.
    runtime.AUTHORIZATION.write_bytes(__import__("json").dumps(bad).encode())
    with pytest.raises(ArtifactError, match="invalid Phase 7 compute authorization"):
        runtime.authorization()
