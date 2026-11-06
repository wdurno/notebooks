"""Phase 5 identity, horizon balance, and decision-recovery checks."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.kalman_pi import runtime as original_runtime
from mnist_experiment.kalman_pi.retention.config import Config, CONDITIONS
from mnist_experiment.kalman_pi.retention.data import make_assets
from mnist_experiment.kalman_pi.retention.run import Scheduler


def smoke():
    return Config.from_json(Path("mnist_experiment/kalman_pi/retention/configs/smoke.json"))


def test_config_freezes_reference_observers_band_and_grid():
    cfg = Config()
    assert cfg.control_cs == (.025,) and cfg.ridge_ratio == .1
    assert cfg.tolerance_pp == .5 and len(cfg.ridges) == 6
    for changes in ({"tolerance_pp": .75}, {"control_cs": (.01,)}, {"anchor_refresh": 16},
                    {"ridges": (.1,)}, {"development_stationary": 3}, {"main_replicas": 7}):
        with pytest.raises(ValueError):
            replace(cfg, **changes)
    assert Config.from_mapping(cfg.mapping()).scientific_hash == cfg.scientific_hash
    assert replace(cfg, seed=cfg.seed + 1).scientific_hash != cfg.scientific_hash
    assert replace(cfg, main_replicas=6).scientific_hash != cfg.scientific_hash


def test_balanced_development_and_fresh_horizons_and_isolated_roles():
    cfg = replace(smoke(), schedule_factors=(1., 1.25))
    g = torch.Generator().manual_seed(13)
    labels = torch.arange(2000) % 10
    data = {"train_x": torch.rand(2000, 1, 28, 28, generator=g), "train_y": labels,
            "test_x": torch.rand(100, 1, 28, 28, generator=g), "test_y": torch.arange(100) % 10}
    for cohort in ("development", "fresh"):
        a, b = (make_assets(cfg, r, cohort, data) for r in (0, 1))
        assert (a["schedule_factor"], b["schedule_factor"]) == (1., 1.25)
        assert a["stream_ids"].numel() == cfg.num_p_steps * cfg.samples_per_step
        assert len(b["stream_ids"]) > len(a["stream_ids"])
        again = make_assets(cfg, 0, cohort, data)
        torch.testing.assert_close(a["stream_ids"], again["stream_ids"], atol=0, rtol=0)
        assert set(a["calibration_ids"].tolist()) <= set(a["init_ids"].tolist())
        assert not set(a["eval_ids"].tolist()) & set(a["init_ids"].tolist())
    dev = make_assets(cfg, 0, "development", data)
    fresh = make_assets(cfg, 0, "fresh", data)
    assert not set(dev["eval_ids"].tolist()) & set(fresh["eval_ids"].tolist())


def test_nested_new_modules_leave_historical_scientific_manifest_unchanged():
    sources = original_runtime.source_manifest()
    assert not any("/retention/" in p for p in sources)
    assert "mnist_experiment/kalman_pi/run.py" in sources


def test_development_target_counts_include_persisted_refinement_jobs():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cfg = Config()
    scheduler.selection = {"development_replicas": {"main": 4, "stationary": 2},
                           "refinement_jobs": []}
    assert scheduler.planned_development() == 114
    scheduler.selection["refinement_jobs"] += [{"schedule": "main"}, {"schedule": "stationary"}]
    assert scheduler.planned_development() == 120


def test_completed_and_failed_units_are_not_retrained():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.complete, scheduler.failures = {}, {}
    from mnist_experiment.kalman_pi.retention.config import content_hash
    meta = {"kind": "trajectory", "stage": "development", "cohort": "development", "replica": 0,
            "schedule": "main", "condition": "covariance", "c": .025, "ridge": .03, "anchor_refresh": 0}
    uid = f"development-main-r000-covariance-{content_hash(meta)[:10]}"
    scheduler.complete[uid] = {"unit_id": uid, "completed": True}
    scheduler.initial = lambda *_: pytest.fail("completed units must not initialize or retrain")
    assert scheduler.trajectory("development", 0, "main", "covariance", .03)["completed"]
    scheduler.complete.clear()
    scheduler.failures[uid] = {"reason": "nonfinite objective"}
    assert scheduler.trajectory("development", 0, "main", "covariance", .03) is None


def test_refinement_forecast_reserves_slow_tested_ridge_and_pending_horizons():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cfg = replace(smoke(), schedule_factors=(1., 1.25), main_replicas=2, stationary_replicas=2)
    scheduler.selection = {"ridges": {}}
    scheduler.initial_times = {"initial-development-r000": 10.}
    scheduler.complete = {f"{c}-{r}": {"stage": "development", "condition": c, "ridge": r,
        "schedule": "main", "schedule_factor": 1., "summary": {"wall_seconds": 1000. if r == 1. else 10.}}
        for c in CONDITIONS for r in (.1, 1.)}
    assert scheduler.fresh_forecast(conservative=True) >= 4 * 1.25 * 4000
    assert scheduler.fresh_forecast(conservative=True) > scheduler.fresh_forecast()
    scheduler.selection = {"ridges": {s: {c: 1. for c in CONDITIONS} for s in ("main", "stationary")},
        "fresh_trajectory_target": 16, "main_replicas": 2, "stationary_replicas": 2}
    assert scheduler.remaining_forecast() == pytest.approx(2 * (1 + 1.25) * 4000 + 2 * 11)
