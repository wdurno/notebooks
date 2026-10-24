"""Fast JVP-repair protocol and artifact tests; no production trajectories."""

import dataclasses
import json

import nbformat
import pytest

from mnist_experiment.fisher_memory import jvp_repair
from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.config import Config, Treatment, content_hash
from mnist_experiment.fisher_memory.jvp_repair_report import build_notebook


def ledger(cfg, *, pilot_replicas=2, confirm_replicas=2):
    baseline = Treatment.default(cfg, "replay_ewc", inner_steps=100, max_evaluations=150)
    strengths = (2.689, 26.89)
    grid = [dataclasses.replace(baseline, condition="jvp", strength=s).mapping() for s in strengths]
    return dict(schema_version=1, protocol=jvp_repair.PROTOCOL,
                environment=jvp_repair.ENVIRONMENT, pilot_family=jvp_repair.PILOT_FAMILY,
                confirm_family=jvp_repair.CONFIRM_FAMILY, pilot_replicas=pilot_replicas,
                confirm_replicas=confirm_replicas, baseline=baseline.mapping(), grid=grid,
                matched_strength=26.89, matched_treatment=grid[1], retention_margin=.01)


def seal(store, unit, config, family, replica, current, historical, seconds):
    store.begin(unit, dict(kind="trajectory", family=family, replica=replica,
                           environment=jvp_repair.ENVIRONMENT, config=config))
    store.write_json(unit, "summary.json", dict(unit_id=unit, current_auc=current,
                     historical_auc=historical, active_seconds=seconds,
                     nonfinite=False, solver_failures=0, under_solved_fraction=.25))
    store.write_json(unit, "metrics.json", [dict(split="curvature", exposure=100,
                                                exact_trace_stm=3.5)])
    store.complete(unit)


def test_frozen_grid_has_exact_fd_anchor_and_independent_families(tmp_path, monkeypatch):
    cfg = Config(device="cpu")
    fd = Treatment.default(cfg, "finite_difference", 26.88965972439316,
                           inner_steps=100, max_evaluations=150)
    monkeypatch.setattr(jvp_repair, "source_policy", lambda *args: dict(
        treatment=fd.mapping(), source_hash="old", selection_seal="selection",
        control_seal="control"))
    result = jvp_repair.build_ledger(tmp_path, cfg, "data", 3, 12)
    strengths = [item["strength"] for item in result["grid"]]
    assert strengths == sorted(strengths)
    assert strengths[3] == pytest.approx(26.88965972439316)
    assert strengths[0] < 1 and strengths[-1] > 80
    assert result["pilot_family"] != result["confirm_family"]
    assert cfg.seed_for(result["pilot_family"], 0, "rotation:stream") != cfg.seed_for(
        result["confirm_family"], 0, "rotation:stream")
    with pytest.raises(ValueError, match="at least two"):
        jvp_repair.build_ledger(tmp_path, cfg, "data", 1, 12)


def test_selection_prioritizes_nll_and_always_confirms_matched_strength(tmp_path):
    cfg = Config(device="cpu")
    plan = ledger(cfg)
    with Store(tmp_path / "run", {"protocol": jvp_repair.PROTOCOL}) as store:
        baseline = Treatment(**plan["baseline"])
        for replica in range(2):
            family = jvp_repair.PILOT_FAMILY
            seal(store, jvp_repair.unit_id(family, replica, baseline), baseline.mapping(),
                 family, replica, .8, .9, 100.)
            low, high = (Treatment(**item) for item in plan["grid"])
            # The lower coefficient is one second cheaper but materially worse.
            seal(store, jvp_repair.unit_id(family, replica, low), low.mapping(),
                 family, replica, .70, .80, 299.)
            seal(store, jvp_repair.unit_id(family, replica, high), high.mapping(),
                 family, replica, .65, .78, 300.)
        selected = jvp_repair.select_pilot(store, plan)
        assert selected["selected"]["strength"] == 26.89
        assert len(selected["confirm"]) == 1

        assert store.is_complete(jvp_repair.unit_id(family, 0, low))
    with Store(tmp_path / "other", {"protocol": jvp_repair.PROTOCOL}) as store:
        baseline = Treatment(**plan["baseline"])
        low, high = (Treatment(**item) for item in plan["grid"])
        for replica in range(2):
            family = jvp_repair.PILOT_FAMILY
            for treatment, current, historical in ((baseline, .8, .9), (low, .6, .8),
                                                    (high, .7, .8)):
                seal(store, jvp_repair.unit_id(family, replica, treatment), treatment.mapping(),
                     family, replica, current, historical, 300.)
        selected = jvp_repair.select_pilot(store, plan)
        assert selected["selected"]["strength"] == 2.689
        assert [row["strength"] for row in selected["confirm"]] == [2.689, 26.89]


def test_live_notebook_discloses_partial_progress_by_strength(tmp_path):
    cfg = Config(device="cpu")
    plan = ledger(cfg)
    root = tmp_path / "run"
    with Store(root, {"protocol": jvp_repair.PROTOCOL}) as store:
        store.begin(jvp_repair.LEDGER_UNIT, {"kind": "repair_ledger"})
        store.write_json(jvp_repair.LEDGER_UNIT, "result.json", plan)
        store.complete(jvp_repair.LEDGER_UNIT)
        baseline = Treatment(**plan["baseline"])
        low = Treatment(**plan["grid"][0])
        seal(store, jvp_repair.unit_id(jvp_repair.PILOT_FAMILY, 0, baseline), baseline.mapping(),
             jvp_repair.PILOT_FAMILY, 0, .8, .9, 100.)
        seal(store, jvp_repair.unit_id(jvp_repair.PILOT_FAMILY, 0, low), low.mapping(),
             jvp_repair.PILOT_FAMILY, 0, .7, .8, 300.)
        store.write_json(None, "status.json", dict(status="running", active_seconds=30., current_job=None))
    notebook = build_notebook(root)
    nbformat.validate(notebook)
    assert "pilot 2/6" in notebook.cells[0].source
    assert "26.89" in notebook.cells[1].source
    assert "+0.1000" in notebook.cells[1].source
    assert "Pending completion" in notebook.cells[-2].source


def test_entrypoint_resumes_without_mutating_completed_units(tmp_path, monkeypatch):
    cfg = Config(device="cpu", free_reserve_bytes=0)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg.mapping()))
    budget = tmp_path / "budget"
    source, root = budget / "source", budget / "repair"
    source.mkdir(parents=True)
    notebook = tmp_path / "jvp.ipynb"
    monkeypatch.setattr(jvp_repair, "build_ledger", lambda _source, config, _hash, pilot, confirm:
                        ledger(config, pilot_replicas=pilot, confirm_replicas=confirm))
    monkeypatch.setattr(jvp_repair, "MNISTSource", lambda *args: object())
    monkeypatch.setattr(jvp_repair, "dataset_manifest", lambda *args: {"cached": True})
    monkeypatch.setattr(jvp_repair, "external_bytes", lambda *args: 0)
    monkeypatch.setattr(jvp_repair, "provenance", lambda *args: {"source_hash": "new"})

    class FakeRunner:
        calls = []

        def __init__(self, cfg, store, source, *, on_progress):
            self.store, self.on_progress = store, on_progress
            self.completed_this_invocation = 0

        def run_trajectory(self, family, phase, env, replica, treatment):
            unit = jvp_repair.unit_id(family, replica, treatment)
            assert not self.store.is_complete(unit)
            self.calls.append(unit)
            value = .8 if treatment.condition == "replay_ewc" else .7 + .002 * treatment.strength
            seal(self.store, unit, treatment.mapping(), family, replica, value, value, 1.)
            self.completed_this_invocation += 1
            self.on_progress()
            return dict(unit_id=unit, active_seconds=1.)

    monkeypatch.setattr(jvp_repair, "Runner", FakeRunner)
    args = ["--config", str(config), "--source-root", str(source), "--root", str(root),
            "--budget-root", str(budget), "--notebook", str(notebook),
            "--pilot-replicas", "2", "--confirm-replicas", "2", "--allocation-hours", "1"]
    assert jvp_repair.main(args + ["--max-units", "2"]) == "paused"
    assert json.loads((root / "status.json").read_text())["completed_units"] == 2
    assert jvp_repair.main(args + ["--resume"]) == "completed"
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["completed_units"] == 12  # six pilot and six confirmation jobs
    assert len(FakeRunner.calls) == len(set(FakeRunner.calls)) == 12
    assert notebook.exists()
    with pytest.raises(Exception, match="identity changed"):
        jvp_repair.main(args + ["--resume", "--confirm-replicas", "3"])
