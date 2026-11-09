"""Selection-gate, seed, and immutable-import checks without model training."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnist_experiment.combined_memory import runtime as base_runtime
from mnist_experiment.combined_memory.broad_drift.config import Config as BroadConfig
from mnist_experiment.combined_memory.config import Recipe
from mnist_experiment.combined_memory.retention_match import calibration, runtime
from mnist_experiment.combined_memory.retention_match.config import Config
from mnist_experiment.combined_memory.retention_match.run import Scheduler
from mnist_experiment.fisher_memory.artifacts import Store, ArtifactError


def records(cfg):
    rows = []
    for arm, recipe in calibration.initial_recipes(cfg.fd_strength).items():
        for member in range(4):
            rows.append(dict(unit_id=f"{arm}-{member}", status="completed",
                metadata=dict(kind="trajectory", cohort="development", replica=member,
                              recipe=recipe.mapping()),
                summary=dict(Havg=2., H100=4., novel_auc=8., learning_seconds=1., online_wall_seconds=2.)))
    return rows


@pytest.fixture
def scheduler(tmp_path):
    cfg = Config.smoke_config()
    identity = dict(protocol=cfg.protocol, config=cfg.mapping())
    with Store(tmp_path / "run", identity, free_reserve_bytes=0) as store:
        obj = Scheduler.__new__(Scheduler)
        obj.cfg, obj.store = cfg, store
        obj.budget = SimpleNamespace(check=lambda: None, remaining=100000., used=0.)
        obj.records = {r["unit_id"]: r for r in records(cfg)}
        obj.selection, obj.numeric, obj.initial_cache = None, {}, {}
        obj.reference_ridge, obj.current_check = 10., None
        yield obj


def freeze(scheduler):
    chosen = calibration.select_tuple(list(scheduler.records.values()),
        tolerance=scheduler.cfg.retention_tolerance, strength=scheduler.cfg.fd_strength)
    scheduler.selection = scheduler.decision("selection", lambda: scheduler.selection_payload(chosen))
    return scheduler.selection


def test_historical_and_new_members_have_the_right_component_seed_identity():
    cfg, old = Config(), BroadConfig()
    for component in ("initialization_model", "stream:transforms", "replay:12", "probes:12"):
        for member in (0, 1):
            assert cfg.seed_for("development", member, component) == old.seed_for("development", member, component)
        for member in (2, 3):
            assert cfg.seed_for("development", member, component) != old.seed_for("development", member, component)
        assert cfg.seed_for("fresh", 0, component) != old.seed_for("fresh", 0, component)
    assert cfg.seed_for("study-partition", 0, "test") == old.seed_for("study-partition", 0, "test")


def test_production_rejects_tolerance_or_policy_changes_and_smoke_failure_flag():
    cfg = Config()
    cfg.validate()
    for changes in (dict(retention_tolerance=.2), dict(fd_strength=.1),
                    dict(force_calibration_failure=True), dict(development_replicas=2)):
        with pytest.raises(ValueError):
            replace(cfg, **changes).validate()


@pytest.mark.parametrize("cohort", ["fresh", "stationary"])
def test_fresh_and_stationary_cannot_start_without_a_passing_seal(scheduler, cohort):
    with pytest.raises(ArtifactError, match="sealed passing calibration"):
        scheduler.trajectory(cohort, 0, calibration.initial_recipes(scheduler.cfg.fd_strength)["B"])
    assert not list((scheduler.store.root / "units").glob("path-*"))


def test_frozen_selection_survives_resume_and_rejects_later_mutation(scheduler):
    sealed = deepcopy(freeze(scheduler))
    assert scheduler.validate_selection() == sealed
    scheduler.initial_cache.clear()  # A completed calibration may resume without in-memory fits.
    assert scheduler.initial_cost() > 0
    scheduler.selection = deepcopy(sealed)
    scheduler.selection["recipes"]["B"]["ridge"] = 1000.
    with pytest.raises(ArtifactError, match="differs from the sealed"):
        scheduler.validate_selection()
    scheduler.selection = deepcopy(sealed)
    missing = scheduler.records.pop("E-3")
    with pytest.raises(ValueError, match="missing or incomplete"):
        scheduler.validate_selection()
    scheduler.records["E-3"] = missing
    assert scheduler.validate_selection() == sealed


def test_unmatched_or_forced_failure_cannot_be_sealed(scheduler):
    chosen = calibration.select_tuple(list(scheduler.records.values()), strength=scheduler.cfg.fd_strength)
    scheduler.cfg = replace(scheduler.cfg, force_calibration_failure=True)
    with pytest.raises(ArtifactError, match="refusing to seal"):
        scheduler.selection_payload(chosen)


def test_balanced_allowances_survive_a_partially_executed_round(scheduler):
    recipes = calibration.initial_recipes(scheduler.cfg.fd_strength)
    mappings = [replace(recipes[a], ridge=3.).mapping() for a in "BCDE"]
    proposal = dict(recipes=mappings, reference_ridge=10., prior_allowances=dict.fromkeys("BCDE", 0))
    scheduler.decision("refinement-0-0", lambda: proposal)
    assert scheduler.used_allowances() == dict.fromkeys("BCDE", 1)
    scheduler.store.resume = True
    again = scheduler.decision("refinement-0-0", lambda: pytest.fail("resume repeated its admission decision"))
    assert again == proposal
    assert scheduler.used_allowances() == dict.fromkeys("BCDE", 1)


def test_confirmation_recipe_must_equal_the_sealed_recipe(scheduler):
    freeze(scheduler)
    wrong = replace(calibration.initial_recipes(scheduler.cfg.fd_strength)["B"], ridge=1000.)
    with pytest.raises(ArtifactError, match="differs from its calibration seal"):
        scheduler.trajectory("fresh", 0, wrong)


def test_scalar_exports_are_verified_without_reading_checkpoint_payloads(tmp_path, monkeypatch):
    cfg = Config.smoke_config()
    with Store(tmp_path / "scalar", dict(protocol=cfg.protocol), free_reserve_bytes=0) as store:
        uid = "trajectory"
        store.begin(uid, dict(kind="trajectory", cohort="development", replica=0,
                             recipe=Recipe("A").mapping()))
        for name, value in (("summary.json", {}), ("metrics.json", []), ("diagnostics.json", [])):
            store.write_json(uid, name, value)
        store.complete(uid)
        before = {p: p.read_bytes() for p in store.unit_path(uid).iterdir() if p.is_file()}
        monkeypatch.setattr(runtime.torch, "load", lambda *a, **kw: pytest.fail("report loaded a checkpoint"))
        exported, _ = runtime.read_exported_records(store.root)
        assert len(exported) == 1
        assert all(p.read_bytes() == v for p, v in before.items())
        (store.unit_path(uid) / "metrics.json").write_text("[1]")
        with pytest.raises(ArtifactError, match="checksum mismatch"):
            runtime.read_exported_records(store.root)


def test_import_rejects_historical_confirmation_role(tmp_path, monkeypatch):
    cfg = Config.smoke_config()
    history = tmp_path / "history"
    with Store(history, dict(protocol="plan18-broad-drift-v1"), free_reserve_bytes=0) as old:
        old.begin("fresh", dict(kind="trajectory", cohort="fresh", replica=0, recipe=Recipe("A").mapping()))
        for name, value in (("summary.json", {}), ("metrics.json", []), ("diagnostics.json", [])):
            old.write_json("fresh", name, value)
        old.complete("fresh")
        _, seals = runtime.scalar_record(history, "fresh")
        identity_hash = old.identity_hash
    monkeypatch.setattr(runtime, "REPO", tmp_path)
    with Store(tmp_path / "new", dict(protocol=cfg.protocol), free_reserve_bytes=0) as store:
        store.begin("import", dict(kind="historical_development_import"))
        store.write_json("import", "references.json", dict(root="history", identity_hash=identity_hash,
            units=[dict(unit_id="fresh", seals=seals)]))
        store.complete("import")
        with pytest.raises(ArtifactError, match="confirmation is not development"):
            runtime.read_exported_records(store.root)
