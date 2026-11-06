"""Tiny CPU acceptance: full scheduler and actual process-death continuation."""

from __future__ import annotations

import os
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import Store, iter_units
from mnist_experiment.kalman_pi.discrepancy.policies import PolicySpec
from mnist_experiment.kalman_pi.discrepancy.adapters import BatchReader, stream_view
from mnist_experiment.kalman_pi.discrepancy.config import smoke_config
from mnist_experiment.kalman_pi.discrepancy.data import make_assets
from mnist_experiment.kalman_pi.discrepancy.model import build_model
from mnist_experiment.kalman_pi.discrepancy.run import Scheduler
from mnist_experiment.kalman_pi.discrepancy.trajectory import run_trajectory
from mnist_experiment.kalman_pi.discrepancy import trajectory as trajectory_module
from mnist_experiment.fisher_memory.control import StopRequested


def raw_data():
    labels = torch.arange(10).repeat_interleave(150)
    test_labels = torch.arange(10).repeat_interleave(6)
    x = torch.zeros(len(labels), 1, 28, 28)
    x[:, :, 6:22, 11:17] = .8
    tx = torch.zeros(len(test_labels), 1, 28, 28)
    tx[:, :, 7:21, 10:18] = .6
    return dict(train_x=x, train_y=labels, test_x=tx, test_y=test_labels)


class Budget:
    allocation_seconds = 72000
    historical_seconds = 0.

    def __init__(self, store):
        self.store = store

    @property
    def used(self):
        return self.store.control.active_seconds

    @property
    def remaining(self):
        return self.allocation_seconds - self.used

    def check(self):
        self.store.check()

    def pulse(self, **kw):
        self.store.heartbeat(force=True)


def death_worker(root, terminate):
    torch.set_num_threads(1)
    torch.manual_seed(762)
    cfg = smoke_config(main_replicas=1, stationary_replicas=0)
    raw = raw_data()
    assets = make_assets(cfg, 0, "development", raw)
    model, layout = build_model(182)
    initial = {"theta": layout.flatten_module(model, detach=True),
               "diagonal": torch.full((448,), .05, dtype=torch.float64)}
    with Store(Path(root), {"kind": "discrepancy-process-death-acceptance"},
               resume=True, free_reserve_bytes=0) as store:
        budget = Budget(store)
        reader = BatchReader(cfg, assets, raw, budget.check)

        def pulse(**info):
            if terminate and info.get("step") == 2:
                os._exit(41)

        run_trajectory(store, budget, cfg, "path", {"condition": "discrepancy_eta0.0001",
            "ridge": 0., "schedule": "main", "hold_steps": 2, "ramp_steps": 2},
            initial, reader, stream_view(assets, "main"), assets["class_probabilities"],
            calibration={"variance": torch.full((448,), .1, dtype=torch.float64)},
            pulse=pulse)


def test_actual_process_death_preserves_accepted_state(tmp_path):
    reference, interrupted = tmp_path / "reference", tmp_path / "interrupted"
    death_worker(reference, False)
    program = "import runpy; ns=runpy.run_path(__import__('sys').argv[1]); ns['death_worker'](__import__('sys').argv[2], True)"
    result = subprocess.run([sys.executable, "-c", program, str(Path(__file__).resolve()), str(interrupted)],
                            check=False, timeout=90, capture_output=True, text=True)
    assert result.returncode == 41, result.stdout + result.stderr
    with Store(interrupted, {"kind": "discrepancy-process-death-acceptance"},
               resume=True, free_reserve_bytes=0) as store:
        checkpoint = store.load_checkpoint("path")
        assert checkpoint["step"] == 2
        assert checkpoint["controller"]["assimilations"] == 2
    death_worker(interrupted, False)
    with Store(reference, {"kind": "discrepancy-process-death-acceptance"}, resume=True,
               free_reserve_bytes=0) as a, Store(interrupted,
               {"kind": "discrepancy-process-death-acceptance"}, resume=True, free_reserve_bytes=0) as b:
        x, y = a.load_checkpoint("path"), b.load_checkpoint("path")
        for key in ("theta", "diagonal"):
            torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)
        assert x["q"] == y["q"] and x["step"] == y["step"] == 8
        for key in ("mean", "covariance", "raw_variance"):
            torch.testing.assert_close(x["controller"][key], y["controller"][key], rtol=0, atol=0)
        torch.testing.assert_close(x["rng"]["cpu"], y["rng"]["cpu"], rtol=0, atol=0)
        for first, second in zip(x["metrics"], y["metrics"], strict=True):
            for key in ("old_risk", "shifted_risk", "old_accuracy", "shifted_accuracy", "applied_pi", "q"):
                assert first.get(key) == second.get(key)


def test_action_precedes_batch_and_first_fisher_is_blended(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg, raw = smoke_config(), raw_data()
    assets = make_assets(cfg, 0, "development", raw)
    model, layout = build_model(182)
    initial = {"theta": layout.flatten_module(model, detach=True),
               "diagonal": torch.full((448,), .05, dtype=torch.float64)}
    events = []
    original_choose = trajectory_module.Controller.choose

    def choose(self, *args):
        events.append("choose")
        return original_choose(self, *args)

    monkeypatch.setattr(trajectory_module.Controller, "choose", choose)
    monkeypatch.setattr(trajectory_module, "observation", lambda *a, **kw: torch.full((448,), .08, dtype=torch.float64))

    def fit(*args, **kw):
        events.append("fit")
        return args[2], {"solver_seconds": 0., "closure_count": 1}

    monkeypatch.setattr(trajectory_module, "fit_batch", fit)
    with Store(tmp_path / "causal", {"test": "causality"}, resume=True, free_reserve_bytes=0) as store:
        budget = Budget(store)
        reader = BatchReader(cfg, assets, raw, budget.check)

        def batch(role, positions=None):
            if role == "main":
                events.append("batch")
            return reader(role, positions)

        def stop(**info):
            if info.get("step") == 1:
                raise StopRequested("accepted first arrival")

        with pytest.raises(StopRequested):
            run_trajectory(store, budget, cfg, "path", {"condition": "fixed", "ridge": 0., "schedule": "main"},
                initial, batch, stream_view(assets, "main"), assets["class_probabilities"], pulse=stop)
        state = store.load_checkpoint("path")
        assert events[:3] == ["choose", "batch", "fit"]
        assert state["step"] == 1
        torch.testing.assert_close(state["diagonal"], initial["diagonal"] * .975 + .025 * .08)


def test_full_tiny_scheduler_report_and_completed_preservation(tmp_path):
    torch.set_num_threads(1)
    cfg = smoke_config(main_replicas=1, stationary_replicas=1, dev_replicas=2,
                       process_scales=(.0001,), ridge_candidates=(.1,),
                       candidates_per_family=1)
    root = tmp_path / "study"
    identity = {"kind": "discrepancy-scheduler-acceptance"}
    with Store(root, identity, resume=True, free_reserve_bytes=0) as store:
        scheduler = Scheduler(cfg, store, Budget(store), raw_data())
        scheduler.run()
        assert (root / "study_completed.json").exists()
        assert (root / "smoke_results.ipynb").exists()
        assert scheduler.selection["frozen"]
        assert sum(t["stage"] == "fresh" for t in scheduler.complete.values()) >= 8
        audited = scheduler.diagnostics["solver_audit"]
        assert len(audited) == 4
        assert sum(row["key"][1] > 0 for row in audited) == 3
        assert all(scheduler.complete[row["key"][0]]["ridge"] == scheduler.selection["common_ridge"]
                   for row in audited)
    seals = {p.relative_to(root): p.read_bytes() for p in root.glob("units/*/COMPLETED")}
    list(iter_units(root, full=True))
    with Store(root, identity, resume=True, free_reserve_bytes=0) as store:
        scheduler = Scheduler(cfg, store, Budget(store), raw_data())
        assert scheduler.selection["frozen"]
        assert len(scheduler.complete) >= 8
    assert seals == {p.relative_to(root): p.read_bytes() for p in root.glob("units/*/COMPLETED")}


def test_pretraining_commits_every_optimizer_batch(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = smoke_config(initialization_size=32, initial_batch_size=8)
    raw = raw_data()
    assets = make_assets(cfg, 0, "development", raw)
    offsets = []
    original = Store.save_checkpoint

    def save(self, uid, state, **kwargs):
        if state.get("phase") == "fit" and state.get("offset", 0):
            offsets.append(state["offset"])
        return original(self, uid, state, **kwargs)

    monkeypatch.setattr(Store, "save_checkpoint", save)
    with Store(tmp_path / "initialization", {"test": "batch-checkpoint"}, resume=True, free_reserve_bytes=0) as store:
        reader = BatchReader(cfg, assets, raw, store.check)
        trajectory_module.initialize(store, Budget(store), cfg, "initial", {
            "model_seed": 18, "order_seed": 19}, reader)
    assert offsets == [8, 16, 24, 32]


def test_graceful_stop_publication_can_roll_back_to_durable_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = smoke_config(main_replicas=1, stationary_replicas=0, dev_replicas=2,
        process_scales=(.0001,), ridge_candidates=(.1,),
        candidates_per_family=1, eval_every=1)
    root, identity = tmp_path / "publication-recovery", {"kind": "discrepancy-publication-recovery"}
    original = Store.save_checkpoint
    interrupted = []

    def save(store, uid, payload, **kwargs):
        if uid.startswith("fresh-") and payload.get("step") == 2 and not interrupted:
            interrupted.append(uid)
            raise StopRequested("interrupt before checkpoint publication")
        return original(store, uid, payload, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Store, "save_checkpoint", save)
        with Store(root, identity, resume=True, free_reserve_bytes=0) as store:
            scheduler = Scheduler(cfg, store, Budget(store), raw_data())
            with pytest.raises(StopRequested):
                scheduler.run()
            assert store.load_checkpoint(interrupted[0])["step"] == 1
            scheduler.publish(status="paused")
    paused = json.loads((root / "report_snapshot.json").read_text())
    preview = next(t for t in paused["trajectories"] if t["unit_id"] == interrupted[0])
    assert not preview["completed"] and preview["metrics"][-1]["step"] == 2
    seals = {p: p.read_bytes() for p in root.glob("units/*/COMPLETED")}
    with Store(root, identity, resume=True, free_reserve_bytes=0) as store:
        scheduler = Scheduler(cfg, store, Budget(store), raw_data())
        scheduler.run()
        assert store.load_checkpoint(interrupted[0])["step"] == cfg.num_p_steps
    assert (root / "study_completed.json").exists()
    assert all(p.read_bytes() == value for p, value in seals.items())
    list(iter_units(root, full=True))


def test_both_amplitudes_retained_without_efficacy_selection():
    from types import SimpleNamespace
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cfg = SimpleNamespace(amplitudes=(25.,45.), dev_replicas=2)
    scheduler.readiness = []
    def capability(replica,angle):
        return {"replica":replica,"angle_max":angle,"readiness_flags":False}
    scheduler.capability=capability
    scheduler.decision=lambda name,produce:produce()
    context=scheduler.choose_context()
    assert context["amplitudes"]==[25.,45.]
    assert len(context["readiness"])==4


if __name__ == "__main__":
    death_worker(sys.argv[1], sys.argv[2] == "terminate")

