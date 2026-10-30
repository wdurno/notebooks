"""Bounded Phase 12 scheduling, paired selection, and immutable chart identity."""

import copy
from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist_experiment import plan15_grid_subspace as subspace
from mnist_experiment.fisher_memory.artifacts import ImmutableError, Store
from mnist_experiment.fisher_memory_grid_registration import (
    curvature as grid_curvature, learner as grid_learner, model as grid_model,
    numerical as grid_numerical, runner as grid_runner,
)
from mnist_experiment.fisher_memory_regression.analysis import replica_draw_indices
from mnist_experiment.fisher_memory_regression.config import Recipe
from mnist_experiment.fisher_memory_regression.runtime import file_digest, read_json
from mnist_experiment.fisher_memory_stm_capacity import PROTOCOL, design, model, run
from mnist_experiment.fisher_memory_stm_capacity.config import (
    CHARTS, FD, INCUMBENT, NEW_CHARTS, REFERENCE, REPLAY, Config, arm, frozen_recipe,
)
from mnist_experiment.fisher_memory_stm_capacity.runner import ChartStore, trajectory_id
from mnist_experiment.plan15_stm_capacity_records import Runner
from src.seeding import derive_component_seed


def record(chart, recipe, replica, *, harm=.125, novel=2., seconds=10., observations=100,
           partition="development", valid=True):
    return dict(stm_config=chart, recipe_key=recipe.key, recipe=recipe.mapping(),
                replica=replica, method=recipe.method, partition=partition, valid=valid,
                base_auc=(1 + harm)**2, frozen_base_mse=1., novel_auc=novel,
                learning_seconds=seconds, observations=observations, active_seconds=seconds)


def screen_rows(cfg):
    records = []
    for chart in CHARTS:
        for method in (REPLAY, FD):
            recipe = (frozen_recipe(cfg, method) if chart == "all_r2" else
                      Recipe.from_config(cfg, method, ridge=1., strength=.5 if method == FD else 0.))
            records.extend(record(chart, recipe, replica) for replica in range(cfg.calibration_replicas))
    return design.table(records, cfg)


def set_harm(row, values, cfg):
    row.update(per_replica_H=list(values), H=float(np.mean(values)),
               retention_upper=design.upper(values, cfg))


def find(rows, chart, method):
    return next(row for row in rows if row["arm"] == arm(chart, method))


def test_production_smoke_roundtrip_hashes_frozen_configuration_and_named_pairing(tmp_path):
    production, smoke = Config(), Config.smoke_config("cpu")
    for cfg in (production, smoke):
        cfg.validate()
        assert Config.from_mapping(cfg.mapping()) == cfg
        path = tmp_path / ("smoke.json" if cfg.smoke else "production.json")
        path.write_text(json.dumps(cfg.mapping()))
        assert Config.read(path) == cfg
        assert cfg.charts == CHARTS == model.CONFIGURATIONS
        assert cfg.archive_rank == 64 and cfg.numerical_policy == subspace.POLICY
        assert (cfg.power_steps, cfg.oversample, cfg.ritz_tol, cfg.cluster_tol) == (4, 16, 1e-4, 1e-6)
    assert production.protocol == PROTOCOL and production.protocol != subspace.Config().protocol
    assert (production.channels, production.output_dims, production.sigma_y) == ((4, 6), (54,), 2.)
    assert (production.init_epochs, production.blur_max, production.novel_count) == (45, 2.4, 100)
    assert (production.reference_capacity, production.reference_replay) == (32, 2)
    assert production.confirmation_replicas == 8 and production.refinement_settings == 4
    assert production.ridge_grid == (1., math.sqrt(10), 10.)
    assert (smoke.device, smoke.dtype, smoke.threads) == ("cpu", "float64", 2)
    assert (smoke.init_size, smoke.reference_size, smoke.novel_count) == (8, 2, 3)
    assert smoke.init_epochs == 1 and smoke.refinement_settings == 0
    assert smoke.scientific_hash != production.scientific_hash
    with pytest.raises(FrozenInstanceError):
        production.seed = 1
    expanded = replace(production, confirmation_replicas=10)
    expanded.validate()
    assert expanded.scientific_hash != production.scientific_hash
    components = ("initialization_model", "factors", "source_sampling", "reference",
                  "replay:0", "probes:0", "compression:0", "selection_bootstrap")
    seeds = []
    for component in components:
        expected = derive_component_seed(production.seed, f"{PROTOCOL}:development:3:{component}")
        assert production.seed_for("development--grid", 3, component) == expected
        assert expanded.seed_for("development--head_r1", 3, component) == expected
        assert production.seed_for("confirmation--grid", 3, component) != expected
        assert production.seed_for("development--grid", 4, component) != expected
        seeds.append(expected)
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("changes", [
    dict(protocol="other"), dict(charts=tuple(reversed(CHARTS))), dict(channels=(4, 8)),
    dict(output_dims=(6,)), dict(sigma_y=1.), dict(init_epochs=30), dict(blur_max=1.8),
    dict(novel_count=99), dict(confirmation_replicas=12), dict(calibration_replicas=3),
    dict(reference_capacity=16), dict(reference_replay=1), dict(ridge_grid=(1., 10.)),
    dict(refinement_settings=5), dict(archive_rank=8), dict(power_steps=8),
    dict(reference_size=300, reference_max_size=200), dict(threads=True),
])
def test_production_read_rejects_changed_scientific_contract(tmp_path, changes):
    path = tmp_path / "incompatible.json"
    path.write_text(json.dumps(Config().mapping() | changes))
    with pytest.raises(ValueError):
        Config.read(path)


@pytest.mark.parametrize("changes", [dict(fd_step=.02), dict(probes=4),
                                     dict(max_iter=5), dict(max_eval=7)])
def test_production_cannot_change_frozen_historical_recipe_controls(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes).validate()


@pytest.mark.parametrize("value", [True, 1.5])
def test_refinement_proposal_count_requires_an_integer(value):
    with pytest.raises(ValueError):
        replace(Config(), refinement_settings=value).validate()


def test_unknown_read_field_fails_and_historical_recipes_are_exact(tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(Config().mapping() | {"unknown_knob": 1}))
    with pytest.raises(TypeError):
        Config.read(path)
    cfg = Config()
    fd, replay = frozen_recipe(cfg, FD), frozen_recipe(cfg, REPLAY)
    assert fd.mapping() == dict(method=FD, capacity=32, replay_count=2, ridge=1.,
                               strength=1.976099143570005, fd_step=.01, probes=2,
                               max_iter=100, max_eval=150)
    assert replay.mapping() == fd.mapping() | dict(method=REPLAY, ridge=math.sqrt(10), strength=0.)
    assert REFERENCE == arm("all_r2", REPLAY) and INCUMBENT == arm("all_r2", FD)
    assert fd.key == Recipe.from_mapping(fd.mapping()).key
    assert len({arm(c, m) for c in CHARTS for m in (REPLAY, FD)}) == 8
    for args in (("all_r3", FD), ("all_r2", "ewc_only")):
        with pytest.raises(ValueError):
            arm(*args)


def test_bootstrap_upper_is_named_one_sided_95th_percentile():
    cfg = replace(Config.smoke_config(), bootstrap_samples=401)
    values = np.array([-.4, .1, .7, 1.9])
    seed = cfg.seed_for("development--grid", 0, "selection_bootstrap")
    draws = [values[replica_draw_indices(len(values), seed, i)].mean()
             for i in range(cfg.bootstrap_samples)]
    assert design.upper(values, cfg) == float(np.quantile(draws, .95))
    assert design.upper(values, cfg) != float(np.quantile(draws, .975))
    assert design.upper([.125, .125], cfg) == .125
    for invalid in ([], [.1], [.1, float("nan")], [float("inf"), .1]):
        with pytest.raises(ValueError, match="finite paired"):
            design.upper(invalid, cfg)


def test_development_table_preserves_complete_pairs_and_filters_confirmation():
    cfg = Config.smoke_config()
    recipe = frozen_recipe(cfg, REPLAY)
    records = [record("head_r1", recipe, 1, harm=.25, seconds=6., observations=20),
               record("head_r1", recipe, 0, harm=.125, seconds=2., observations=10),
               record("head_r1", recipe, 0, novel=-999., partition="confirmation"),
               record("all_r1", recipe, 0), record("all_r4", recipe, 9)]
    original = copy.deepcopy(records)
    rows = design.table(records, cfg)
    assert records == original and rows == design.table(list(reversed(records)), cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row["stm_config"] == "head_r1" and row["replicas"] == [0, 1]
    assert row["per_replica_H"] == [.125, .25] and row["H"] == .1875
    assert row["retention_upper"] == .25 and row["novel_auc"] == 2.
    assert row["seconds_per_arrival"] == pytest.approx(8 / 30)
    records[0]["valid"] = False
    assert not design.table(records, cfg)[0]["valid"]


def test_paired_retention_selection_cancels_shared_replica_variation():
    cfg = Config.smoke_config()
    rows = screen_rows(cfg)
    set_harm(find(rows, "all_r2", REPLAY), [-.2, .4], cfg)
    set_harm(find(rows, "head_r1", REPLAY), [-.15, .45], cfg)
    candidate = find(rows, "head_r1", FD)
    set_harm(candidate, [-.14, .46], cfg)
    chosen = design.select(rows, cfg)
    assert chosen[arm("head_r1", FD)]["eligible"]
    assert chosen[arm("head_r1", FD)]["failures"] == []
    assert chosen[REFERENCE]["selected"]["arm"] == REFERENCE
    assert chosen[INCUMBENT]["frozen_historical_policy"]
    assert chosen[INCUMBENT]["selected"]["recipe"] == frozen_recipe(cfg, FD).mapping()


@pytest.mark.parametrize("failure,reference,replay,fd,cost", [
    ("retention_ceiling", .49, .49, .55, 1.),
    ("retention_common_reference", 0., .2, .25, 1.),
    ("retention_same_chart", .2, 0., .15, 1.),
    ("relative_learning_cost", 0., 0., 0., 2.501),
])
def test_each_fd_admission_requirement_uses_the_common_reference(failure, reference, replay, fd, cost):
    cfg = Config.smoke_config()
    rows = screen_rows(cfg)
    set_harm(find(rows, "all_r2", REPLAY), [reference]*2, cfg)
    set_harm(find(rows, "head_r1", REPLAY), [replay]*2, cfg)
    candidate = find(rows, "head_r1", FD)
    set_harm(candidate, [fd]*2, cfg)
    candidate["seconds_per_arrival"] = cost * find(rows, "all_r2", REPLAY)["seconds_per_arrival"]
    # A cheaper, better-acquiring size-matched control must not redefine B.
    find(rows, "head_r1", REPLAY)["novel_auc"] = .01
    selected = design.select(rows, cfg)[arm("head_r1", FD)]
    assert not selected["eligible"]
    assert selected["failures"] == [failure]


def test_screen_prefers_valid_qualified_acquisition_then_cost_and_recipe_order():
    cfg = Config.smoke_config()
    rows = screen_rows(cfg)
    source = find(rows, "head_r1", FD)
    source["novel_auc"] = 3.
    for ridge, strength, cost in ((2., .5, .2), (1., .7, .2), (1., .6, .2), (1., .1, .3)):
        recipe = Recipe.from_config(cfg, FD, ridge=ridge, strength=strength)
        rows.append({**copy.deepcopy(source), "recipe": recipe.mapping(), "recipe_key": recipe.key,
                     "novel_auc": 1., "seconds_per_arrival": cost})
    invalid = {**copy.deepcopy(source), "novel_auc": -100., "valid": False}
    rejected = {**copy.deepcopy(source), "novel_auc": .01}
    set_harm(rejected, [.9, .9], cfg)
    rows.extend((invalid, rejected))
    chosen = design.select(rows, cfg)[arm("head_r1", FD)]
    assert chosen["eligible"]
    assert chosen["selected"]["recipe"]["ridge"] == 1.
    assert chosen["selected"]["recipe"]["strength"] == .6
    assert chosen == design.select(list(reversed(rows)), cfg)[arm("head_r1", FD)]
    for row in rows:
        if row["arm"] == arm("head_r1", FD):
            row["valid"] = False
    with pytest.raises(ValueError, match="no numerically valid"):
        design.select(rows, cfg)


def test_diagnostic_fallback_orders_one_sided_paired_retention_violation():
    cfg = Config.smoke_config()
    rows = screen_rows(cfg)
    set_harm(find(rows, "all_r2", REPLAY), [0., 0.], cfg)
    set_harm(find(rows, "head_r1", REPLAY), [0., 0.], cfg)
    misleading_mean = find(rows, "head_r1", FD)
    set_harm(misleading_mean, [-.3, .3], cfg)
    safer = copy.deepcopy(misleading_mean)
    safer_recipe = Recipe.from_config(cfg, FD, strength=2.)
    safer.update(recipe=safer_recipe.mapping(), recipe_key=safer_recipe.key, novel_auc=3.)
    set_harm(safer, [.15, .15], cfg)
    rows.append(safer)
    result = design.select(rows, cfg)[arm("head_r1", FD)]
    assert not result["eligible"]
    assert result["selected"]["recipe_key"] == safer_recipe.key
    assert result["failures"] == ["retention_common_reference", "retention_same_chart"]


@pytest.mark.parametrize("optional", [False, True])
def test_screen_has_46_mandatory_trajectories_at_most_four_paired_proposals_and_freezes(optional):
    cfg = replace(Config(device="cpu"), bootstrap_samples=32)
    study = object.__new__(run.Study)
    study.cfg, study.records, study.status = cfg, {}, {}
    study.budget = SimpleNamespace(used=2., remaining=100_000.)
    decisions, completed, audits = {}, [], []
    study.decision = decisions.get
    study.progress = lambda **kwargs: None
    study.optional_fits = lambda count: optional
    study.audit_selected = lambda selected: audits.append(copy.deepcopy(selected))

    def seal(name, value):
        if name in decisions:
            assert decisions[name] == value
        decisions[name] = copy.deepcopy(value)
        return decisions[name]

    study.seal = seal

    def job(chart, replica, recipe, partition="development"):
        unit = trajectory_id(chart, partition + "--grid", replica, recipe, cfg.novel_count)
        if unit not in study.records:
            # Favor c=1 and the lower strength so refinements remain bounded and distinct.
            study.records[unit] = record(chart, recipe, replica,
                novel=recipe.ridge + recipe.strength, seconds=1.)
            completed.append((chart, replica, recipe))
        return study.records[unit]

    study.job = job

    def calibrate():
        for replica in range(cfg.calibration_replicas):
            for method in (REPLAY, FD):
                job("all_r2", replica, frozen_recipe(cfg, method))
            for chart in NEW_CHARTS:
                job(chart, replica, study.recipe(REPLAY, ridge=math.sqrt(10)))
        for chart in NEW_CHARTS:
            seal("numerics-" + chart, dict(lambda0=.5, selected_step=.01))

    study.calibrate = calibrate
    study.screen()
    expected_count = 54 if optional else 46
    assert len(completed) == len(study.records) == expected_count
    assert set(decisions["selection"]) == {arm(c, m) for c in CHARTS for m in (REPLAY, FD)}
    assert len(audits) == 1
    assert len([item for item in completed if item[0] == "all_r2"]) == 4
    for chart in NEW_CHARTS:
        policies = [recipe for c, replica, recipe in completed if c == chart and replica == 0]
        assert {recipe.ridge for recipe in policies if recipe.method == REPLAY} == set(cfg.ridge_grid)
        assert {.05, .5, 5.} <= {recipe.strength for recipe in policies if recipe.method == FD}
        assert any(recipe.method == FD and recipe.ridge == math.sqrt(10) for recipe in policies)
    proposals = [v["stm_config"] for k, v in decisions.items() if k.startswith("refinement-")]
    assert len(proposals) == (4 if optional else 0)
    if optional:
        assert proposals[:3] == list(NEW_CHARTS)
        assert study.status["optional_development"] == 8
    before = copy.deepcopy((completed, decisions))
    study.screen()
    assert (completed, decisions) == before and len(audits) == 1


def test_optional_work_cannot_spend_confirmation_and_analysis_reserve():
    study = object.__new__(run.Study)
    study.cfg = Config(device="cpu")
    study.records = {"slow": {"active_seconds": 10.}, "fast": {"active_seconds": 1.}}
    threshold = 1.25 * (8 * study.cfg.confirmation_replicas * 10 + 1800) + 2 * 10
    study.budget = SimpleNamespace(remaining=threshold - .01)
    assert not study.optional_fits(2)
    study.budget.remaining = threshold
    assert study.optional_fits(2)
    study.cfg = Config.smoke_config()
    assert not study.optional_fits(2)


def test_chart_store_stamps_checkpoints_summaries_rejects_mismatch_and_seals(tmp_path):
    root = tmp_path / "budget" / "study"
    with Store(root, {"scheduler_test": 1}, free_reserve_bytes=0) as base:
        store = ChartStore(base)
        with model.use_chart("head_r1"):
            store.begin("trajectory", dict(kind="grid_registration_trajectory", replica=0))
            store.save_checkpoint("trajectory", {"theta": torch.arange(3, dtype=torch.float64)})
            store.write_json("trajectory", "summary.json", dict(method=FD, valid=True))
            state = store.load_checkpoint("trajectory")
            assert state["stm_config"] == "head_r1"
            assert read_json(store.unit_path("trajectory") / "manifest.json")["metadata"]["stm_config"] == "head_r1"
            summary = read_json(store.unit_path("trajectory") / "summary.json")
            assert summary["arm"] == arm("head_r1", FD) and summary["protocol"] == PROTOCOL
            store.complete("trajectory")
            seal = file_digest(store.unit_path("trajectory") / "COMPLETED")
            with pytest.raises(ImmutableError):
                store.save_checkpoint("trajectory", state)
        with model.use_chart("all_r2"), pytest.raises(ValueError, match="checkpoint STM chart"):
            store.load_checkpoint("trajectory")
        assert file_digest(store.unit_path("trajectory") / "COMPLETED") == seal


def test_unique_chart_ids_and_nested_run_chart_restore_inherited_id(monkeypatch):
    cfg = Config.smoke_config()
    recipe = frozen_recipe(cfg, REPLAY)
    identifiers = {trajectory_id(chart, family, replica, recipe, target)
                   for chart in CHARTS for family in ("development--grid", "confirmation--grid")
                   for replica in range(2) for target in (1, 3)}
    assert len(identifiers) == 32
    original = grid_runner.trajectory_id
    runner = object.__new__(Runner)
    runner.store = SimpleNamespace(chart_identity={})
    cached = {}
    runner.record = lambda unit: cached[unit]
    nested = []

    def inherited(self, family, replica, partition, policy, **kwargs):
        current = model.CHART.get()
        unit = grid_runner.trajectory_id(family, replica, policy, cfg.novel_count)
        assert unit == trajectory_id(current, family, replica, policy, cfg.novel_count)
        if current == "head_r1":
            outer = grid_runner.trajectory_id
            nested.append(self.run_chart("all_r2", family, replica, partition, policy))
            assert grid_runner.trajectory_id is outer and model.CHART.get() == "head_r1"
        if kwargs.get("fail"):
            raise RuntimeError("interrupted chart")
        result = dict(unit_id=unit, method=policy.method, stm_config=current,
                      arm=arm(current, policy.method), protocol=PROTOCOL)
        cached[unit] = result
        return result

    monkeypatch.setattr(grid_runner.Runner, "run", inherited)
    result = runner.run_chart("head_r1", "development--grid", 0, "development", recipe)
    assert result["arm"] == arm("head_r1", REPLAY)
    assert nested[0]["arm"] == REFERENCE
    assert grid_runner.trajectory_id is original and model.CHART.get() == "all_r2"
    with pytest.raises(RuntimeError, match="interrupted chart"):
        runner.run_chart("all_r4", "development--grid", 0, "development", recipe, fail=True)
    assert grid_runner.trajectory_id is original and model.CHART.get() == "all_r2"


def test_fresh_and_completed_runner_records_have_identical_chart_metadata(monkeypatch, tmp_path):
    cfg = Config.smoke_config()
    policy = frozen_recipe(cfg, REPLAY)
    with Store(tmp_path / "budget" / "study", {"runner_schema_test": 1}, free_reserve_bytes=0) as base:
        runner = object.__new__(Runner)
        runner.store = ChartStore(base)
        metadata = dict(chart_hash="chart-hash", layout_hash="layout-hash",
                        factor_mapping={"schema_version": 1}, factor_seeds={"named_factor_seed": 17})
        runner.store.chart_identity = metadata

        def inherited(self, family, replica, partition, recipe, **kwargs):
            unit = grid_runner.trajectory_id(family, replica, recipe, cfg.novel_count)
            if self.store.is_complete(unit):
                return self.record(unit)
            summary = dict(unit_id=unit, method=recipe.method, protocol=PROTOCOL)
            self.store.begin(unit, dict(kind="grid_registration_trajectory"))
            self.store.write_json(unit, "summary.json", summary)
            self.store.write_json(unit, "metrics.json", [])
            self.store.write_json(unit, "retirement.json", {"audit": []})
            self.store.complete(unit)
            return {**summary, "metrics": [], "retired_audit": []}

        monkeypatch.setattr(grid_runner.Runner, "run", inherited)
        fresh = runner.run_chart("head_r1", "development--grid", 0, "development", policy)
        completed = runner.run_chart("head_r1", "development--grid", 0, "development", policy)
        assert fresh == completed
        assert {key: fresh[key] for key in metadata} == metadata


def test_versioned_record_bindings_restore_frozen_entry_points():
    from mnist_experiment import plan15_stm_capacity_records as version
    original = run.Runner, run.Study, run.sources, run.process_alive
    original_sources = run.sources()
    with version.bindings():
        assert run.Runner is version.Runner and run.Study is version.Study
        assert original_sources.items() <= run.sources().items()
        assert "mnist_experiment/plan15_stm_capacity_records.py" in run.sources()
    assert (run.Runner,run.Study,run.sources,run.process_alive) == original


def test_decisions_are_append_only_and_incompatible_store_identity_fails(tmp_path):
    root = tmp_path / "budget" / "study"
    identity = dict(protocol=PROTOCOL, config_hash=Config.smoke_config().scientific_hash)
    with Store(root, identity, free_reserve_bytes=0) as store:
        study = object.__new__(run.Study)
        study.store = store
        decision = dict(reference=REFERENCE, recipes={INCUMBENT: frozen_recipe(Config(), FD).mapping()})
        study.seal("selection", decision)
        path = store.unit_path("decision-selection") / "COMPLETED"
        before = file_digest(path)
        assert study.seal("selection", copy.deepcopy(decision)) == decision
        with pytest.raises(ValueError, match="sealed decision changed"):
            study.seal("selection", {**decision, "reference": arm("all_r1", REPLAY)})
        assert file_digest(path) == before
    with pytest.raises(ImmutableError, match="identity changed"):
        with Store(root, {**identity, "config_hash": "different"}, resume=True, free_reserve_bytes=0):
            pass


def test_source_inventory_preserves_parent_and_bindings_restore_on_nested_failure():
    parent = subspace.sources()
    sources = run.sources()
    assert {path: sources[path] for path in parent} == parent
    for name in ("config.py", "design.py", "model.py", "core.py", "runner.py", "run.py", "numerical.py"):
        path = Path(run.__file__).parent / name
        assert sources[str(path.relative_to(run.ROOT))] == file_digest(path)
    assert "mnist_experiment/fisher_memory_stm_capacity/report.py" not in sources
    targets = [(grid_model, "build_model"), (grid_runner, "trajectory_id"),
               (grid_learner, "compress_fisher")]
    targets += [(module, name) for module in (grid_model, grid_curvature, grid_learner, grid_numerical)
                for name in ("memory_indices", "flat_prediction")]

    def snapshot():
        return [getattr(module, name) for module, name in targets]

    previous = snapshot()
    with pytest.raises(RuntimeError, match="binding failure"):
        with run.bindings(), model.use_chart("head_r1"):
            assert grid_model.build_model is model.build_model
            assert grid_learner.compress_fisher is subspace.compress_fisher
            assert grid_numerical.memory_indices is model.memory_indices
            outer = snapshot()
            with run.bindings(), model.use_chart("all_r4"):
                assert snapshot() == outer
            assert model.CHART.get() == "head_r1"
            raise RuntimeError("binding failure")
    assert snapshot() == previous and model.CHART.get() == "all_r2"
