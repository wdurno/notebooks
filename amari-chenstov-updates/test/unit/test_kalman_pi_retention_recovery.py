"""Store-backed scheduler recovery without datasets, models, or GPU work."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnist_experiment.fisher_memory.artifacts import ArtifactError, Store
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.kalman_pi.retention import analysis, run, runtime
from mnist_experiment.kalman_pi.retention.config import Config, CONDITIONS, content_hash
from mnist_experiment.kalman_pi.runtime import read_json


IDENTITY = {"test": "plan17-retention-recovery"}


@pytest.fixture
def cfg():
    return Config.from_json(Path("mnist_experiment/kalman_pi/retention/configs/smoke.json"))


@pytest.fixture(autouse=True)
def no_data_or_publication(monkeypatch):
    monkeypatch.setattr(run.original, "load_data", lambda _: {})
    monkeypatch.setattr(run.Scheduler, "pulse", lambda *args, **kwargs: None)
    monkeypatch.setattr(run.Scheduler, "publish", lambda *args, **kwargs: None)


def budget(remaining=72_000.):
    return SimpleNamespace(used=0., remaining=remaining, allocation_seconds=72_000.,
                           check=lambda: None, pulse=lambda **kwargs: None)


def store(root, *, resume=False):
    return Store(root, IDENTITY, resume=resume, free_reserve_bytes=0,
                 finalization_reserve_bytes=4096, max_bytes=8 * 1024**2)


def frozen(cfg):
    return {"conditions": list(CONDITIONS), "common_c": .025, "reference_ridge": .1,
            "tolerance_pp": .5, "development_replicas": {"main": 1, "stationary": 1},
            "initial_ridges": list(cfg.ridges), "refinement_jobs": [],
            "ridges": {s: dict(zip(CONDITIONS, (.1, .03, .3, 0.))) for s in ("main", "stationary")},
            "development_selection": {}, "bootstrap_seed": cfg.seed_for("analysis", 0, "bootstrap"),
            "main_replicas": 1, "stationary_replicas": 1, "fresh_trajectory_target": 8,
            "frozen_at": "2026-11-02T12:00:00Z"}


def write_decision(storage, decision, *, complete=True):
    storage.write_json(None, "selection.json", decision)
    storage.begin("frozen-selection", {"kind": "selection", "selection_hash": content_hash(decision)})
    storage.write_json("frozen-selection", "selection.json", decision)
    if complete:
        storage.complete("frozen-selection")


@pytest.mark.parametrize("corruption", ("ridge", "freeze_flag", "empty", "deleted"))
def test_sealed_selection_is_authoritative_before_recovery_or_tuning(tmp_path, cfg, corruption):
    decision = frozen(cfg)
    root = tmp_path / "study"
    with store(root) as storage:
        write_decision(storage, decision)
        changed = deepcopy(decision)
        if corruption == "ridge":
            changed["ridges"]["main"]["covariance"] = 1.
        elif corruption == "freeze_flag":
            changed.pop("frozen_at")
        elif corruption == "empty":
            changed = {}
        if corruption == "deleted":
            (root / "selection.json").unlink()
        else:
            storage.write_json(None, "selection.json", changed)
    with store(root, resume=True) as storage:
        try:
            scheduler = run.Scheduler(cfg, storage, budget(), tmp_path)
        except (ValueError, ArtifactError):
            return
        # Recovery may restore the seal, but must never accept a changed plan.
        assert scheduler.selection == decision, "mutable index bypassed the sealed decision"


def test_frozen_resume_skips_all_tuning_preserves_zero_and_never_reseals(tmp_path, cfg, monkeypatch):
    root, decision = tmp_path / "study", frozen(cfg)
    with store(root) as storage:
        write_decision(storage, decision)
    with store(root, resume=True) as storage:
        scheduler = run.Scheduler(cfg, storage, budget(), tmp_path)
        sealed_before = {p.name: p.read_bytes() for p in storage.unit_path("frozen-selection").iterdir()}
        monkeypatch.setattr(scheduler, "run_development_candidate", lambda *args: pytest.fail("resume entered development"))
        monkeypatch.setattr(scheduler, "freeze_selection", lambda: pytest.fail("resume retuned selection"))
        monkeypatch.setattr(scheduler, "count_unit", lambda: pytest.fail("resume resealed the decision"))
        calls = []
        monkeypatch.setattr(scheduler, "trajectory", lambda *args: calls.append(args))
        scheduler.run()
        assert len(calls) == 8
        assert all(cohort == "fresh" and ridge == decision["ridges"][schedule][condition]
                   for cohort, _, schedule, condition, ridge in calls)
        assert any(call[-1] == 0. for call in calls)
        assert scheduler.selection == read_json(root / "selection.json") == decision
        assert sealed_before == {p.name: p.read_bytes() for p in storage.unit_path("frozen-selection").iterdir()}


@pytest.mark.parametrize("started_seal", (False, True))
def test_crash_after_freeze_before_seal_resumes_without_reselection(tmp_path, cfg, monkeypatch, started_seal):
    root, decision = tmp_path / "study", frozen(cfg)
    with store(root) as storage:
        if started_seal:
            write_decision(storage, decision, complete=False)
        else:
            storage.write_json(None, "selection.json", decision)
    with store(root, resume=True) as storage:
        scheduler = run.Scheduler(cfg, storage, budget(), tmp_path)
        monkeypatch.setattr(scheduler, "run_development_candidate", lambda *args: pytest.fail("freeze lost on resume"))
        monkeypatch.setattr(scheduler, "freeze_selection", lambda: pytest.fail("selection changed on resume"))
        monkeypatch.setattr(scheduler, "trajectory", lambda *args: None)
        scheduler.run()
        assert storage.is_complete("frozen-selection")
        assert read_json(storage.unit_path("frozen-selection") / "selection.json") == decision


def fake_candidate_table(trajectories, schedule, condition, replicas=None):
    return [{"schedule": schedule, "condition": condition}]


def proposal(table, **kwargs):
    return .05 if (table[0]["schedule"], table[0]["condition"]) == ("main", "covariance") else None


def finish_without_training(scheduler, cfg):
    decision = frozen(cfg)
    decision["refinement_jobs"] = deepcopy(scheduler.selection["refinement_jobs"])
    decision.update(main_replicas=0, stationary_replicas=0, fresh_trajectory_target=0)
    scheduler.selection = decision
    scheduler.save_selection()


def test_refinement_decision_is_persisted_before_launch_and_replayed_on_resume(tmp_path, cfg, monkeypatch):
    cfg = replace(cfg, max_refinements=1)
    root = tmp_path / "study"
    monkeypatch.setattr(analysis, "candidate_table", fake_candidate_table)
    monkeypatch.setattr(analysis, "refinement_candidate", proposal)
    with store(root) as storage:
        scheduler = run.Scheduler(cfg, storage, budget(), tmp_path)

        def interrupt(condition, ridge, schedule=None):
            if schedule == "main" and condition == "covariance" and ridge == .05:
                jobs = read_json(root / "selection.json")["refinement_jobs"]
                assert len(jobs) == 1 and jobs[0]["ridge"] == .05
                raise StopRequested("after decision persistence, before refinement completion")

        monkeypatch.setattr(scheduler, "run_development_candidate", interrupt)
        with pytest.raises(StopRequested, match="decision persistence"):
            scheduler.run()
        persisted = read_json(root / "selection.json")["refinement_jobs"]
    with store(root, resume=True) as storage:
        scheduler = run.Scheduler(cfg, storage, budget(), tmp_path)
        attempts = []
        monkeypatch.setattr(scheduler, "run_development_candidate", lambda *args: attempts.append(args))
        monkeypatch.setattr(scheduler, "freeze_selection", lambda: finish_without_training(scheduler, cfg))
        scheduler.run()
        assert attempts.count(("covariance", .05, "main")) == 1
        assert scheduler.selection["refinement_jobs"] == persisted


def test_failed_refinement_must_not_consume_second_allowance_with_duplicate_ridge(tmp_path, cfg, monkeypatch):
    cfg = replace(cfg, max_refinements=2)
    with store(tmp_path / "study") as storage:
        scheduler = run.Scheduler(cfg, storage, budget(), tmp_path)
        job = {"schedule": "main", "condition": "covariance", "ridge": .05,
               "ordinal": 1, "decided_at": "2026-11-02T12:00:00Z"}
        scheduler.selection["refinement_jobs"] = [job]
        scheduler.save_selection()
        monkeypatch.setattr(analysis, "candidate_table", fake_candidate_table)
        monkeypatch.setattr(analysis, "refinement_candidate", proposal)
        # A failed replica leaves this ridge out of the complete-paired table.
        monkeypatch.setattr(scheduler, "run_development_candidate", lambda *args: None)
        monkeypatch.setattr(scheduler, "freeze_selection", lambda: finish_without_training(scheduler, cfg))
        scheduler.run()
        jobs = scheduler.selection["refinement_jobs"]
        identities = {(j["schedule"], j["condition"], j["ridge"]) for j in jobs}
        assert len(identities) == len(jobs), "failed refinement was persisted as a second identical job"


def test_zero_ridge_is_forwarded_as_treatment_without_changing_shared_config(tmp_path, cfg, monkeypatch):
    with store(tmp_path / "study") as storage:
        scheduler = run.Scheduler(cfg, storage, budget(), tmp_path)
        initial = {"theta": "shared-theta", "diagonal": "shared-initial-fisher"}
        assets = {"schedule_factor": 1., "metadata": {"streams": {"stream": {"stream_hash": "stream"}}}}
        monkeypatch.setattr(scheduler, "initial", lambda *args: initial)
        monkeypatch.setattr(scheduler, "assets", lambda *args: assets)
        monkeypatch.setattr(run.original, "stream_view", lambda *args: "shared-stream")
        scheduler.data = {"train_x": None, "train_y": None}

        def record(storage, fake_budget, resolved, uid, meta, same_initial, same_assets, *args, **kwargs):
            assert meta["ridge"] == 0. and resolved.ridge_ratio == .1
            assert same_initial is initial and same_assets is assets
            storage.begin(uid, meta)
            storage.write_json(uid, "metrics.json", [])
            storage.write_json(uid, "summary.json", {"wall_seconds": 0., "final_nine_recall": .2, "final_old_accuracy": .8})
            storage.complete(uid)

        monkeypatch.setattr(run, "run_trajectory", record)
        result = scheduler.trajectory("development", 0, "main", "covariance", 0.)
        assert result["ridge"] == 0. and cfg.ridge_ratio == .1
        assert initial == {"theta": "shared-theta", "diagonal": "shared-initial-fisher"}


def test_allocation_reserves_selected_package_cost_not_grid_median(tmp_path, cfg, monkeypatch):
    cfg = replace(cfg, ridges=(0., .01, .03, .1, .3, 1.), main_replicas=2, stationary_replicas=2)
    with store(tmp_path / "study") as storage:
        scheduler = run.Scheduler(cfg, storage, budget(remaining=5000.), tmp_path)
        scheduler.initial_times = {"initial-development-r000": 10.}
        scheduler.complete = {
            f"{c}-{ridge}": {"condition": c, "ridge": ridge, "stage": "development", "schedule": "main",
                            "schedule_factor": 1., "summary": {"wall_seconds": 1000. if ridge == 1. and c != "fixed" else 10.}}
            for c in CONDITIONS for ridge in ((.1,) if c == "fixed" else cfg.ridges)}
        monkeypatch.setattr(analysis, "candidate_table", lambda *args, **kwargs: [{"ridge": 1.}])
        monkeypatch.setattr(analysis, "select_candidate", lambda table, **kwargs: table[0])
        scheduler.freeze_selection()
        count = scheduler.selection["main_replicas"] + scheduler.selection["stationary_replicas"]
        selected_online_seconds = count * (10. + 3 * 1000.)
        assert 1.25 * selected_online_seconds + 1800 <= scheduler.budget.remaining, (
            "frozen allocation ignores the measured slow selected ridge")


def test_phase5_source_identity_includes_helpers_that_determine_selection(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.original, "provenance", lambda *args: {"sources": runtime.original.source_manifest()})
    sources = runtime.provenance(cfg, tmp_path)["sources"]
    assert "mnist_experiment/kalman_pi/analysis.py" in sources
    assert "mnist_experiment/diagonal_lfu/analysis.py" in sources
