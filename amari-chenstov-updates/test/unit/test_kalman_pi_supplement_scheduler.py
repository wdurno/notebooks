"""Bounded Phase 5E controls, partition integrity and immutable design recovery."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.kalman_pi.retention.config import Config as PrimaryConfig
from mnist_experiment.kalman_pi.retention.data import make_assets
from mnist_experiment.kalman_pi.retention.supplement.config import Config, ARMS
from mnist_experiment.kalman_pi.retention.supplement import run
from mnist_experiment.kalman_pi.runtime import read_json


def smoke():
    return Config.from_json("mnist_experiment/kalman_pi/retention/supplement/configs/smoke.json")


def budget():
    return SimpleNamespace(used=0., remaining=72_000., allocation_seconds=72_000.,
                           check=lambda: None, pulse=lambda **kw: None)


def test_production_settings_freeze_three_arms_and_full_cohort():
    cfg = Config()
    assert cfg.main_replicas == 8 and cfg.stationary_replicas == 0
    assert [v["ridge"] for v in ARMS.values()] == [.1, 0, 0]
    assert [v["condition"] for v in ARMS.values()] == ["fixed", "fixed", "kalman_eta0.0001"]
    for changes in ({"main_replicas": 6}, {"seed": 171104}, {"partition_seed": 13},
                    {"ridge_ratio": 0}, {"control_cs": (.05,)}, {"max_iter": 30},
                    {"stationary_replicas": 2}, {"schedule_factors": (1.,)}):
        with pytest.raises(ValueError):
            replace(cfg, **changes)
    assert Config.from_mapping(cfg.mapping()).scientific_hash == cfg.scientific_hash


def test_primary_partitions_preserved_with_new_component_seeds():
    primary, cfg = PrimaryConfig(), Config()
    for name in ("train", "test"):
        assert cfg.seed_for("study-partition", 0, name) == primary.seed_for("study-partition", 0, name)
    for component in ("initialization", "initial_model", "stream", "evaluation", "noise_calibration"):
        assert cfg.seed_for("fresh", 0, component) != primary.seed_for("fresh", 0, component)
    assert cfg.seed_for("fresh", 0, "stream") != cfg.seed_for("fresh", 1, "stream")


def test_nested_supplement_is_outside_primary_source_glob():
    package = Path(run.__file__).parents[1]
    assert not any("supplement" in str(p.relative_to(package)) for p in package.glob("*.py"))


def test_new_assets_respect_primary_role_isolation_and_horizon_balance():
    cfg = smoke()
    generator = torch.Generator().manual_seed(19)
    data = {"train_x": torch.rand(2000, 1, 28, 28, generator=generator),
            "train_y": torch.arange(2000) % 10,
            "test_x": torch.rand(200, 1, 28, 28, generator=generator), "test_y": torch.arange(200) % 10}
    primary = PrimaryConfig.from_json("mnist_experiment/kalman_pi/retention/configs/smoke.json")
    primary = replace(primary, seed=cfg.partition_seed)
    dev = make_assets(primary, 0, "development", data)
    a, b = (make_assets(cfg, r, "fresh", data) for r in (0, 1))
    assert (a["schedule_factor"], b["schedule_factor"]) == (1., 1.25)
    assert not set(a["eval_ids"].tolist()) & set(dev["eval_ids"].tolist())
    assert set(a["calibration_ids"].tolist()) <= set(a["init_ids"].tolist())
    assert not set(a["stream_ids"].flatten().tolist()) & set(a["init_ids"].tolist())
    assert not torch.equal(a["stream_ids"], make_assets(primary, 0, "fresh", data)["stream_ids"])


@pytest.mark.parametrize("corruption", ("arm", "frozen", "empty", "deleted"))
def test_sealed_design_is_authoritative_on_resume(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(run.original, "load_data", lambda _: {})
    root, cfg = tmp_path / "study", smoke()
    with Store(root, {"test": "supplement"}, free_reserve_bytes=0) as storage:
        scheduler = run.Scheduler(cfg, storage, budget(), tmp_path)
        scheduler.freeze_design()
        sealed = deepcopy(scheduler.selection)
        changed = deepcopy(sealed)
        if corruption == "arm":
            changed["arms"]["fixed_zero"]["ridge"] = .1
        elif corruption == "frozen":
            changed["frozen"] = False
        elif corruption == "empty":
            changed = {}
        if corruption == "deleted":
            (root / "selection.json").unlink()
        else:
            storage.write_json(None, "selection.json", changed)
    with Store(root, {"test": "supplement"}, resume=True, free_reserve_bytes=0) as storage:
        if corruption != "deleted":
            with pytest.raises(ValueError, match="immutable seal"):
                run.Scheduler(cfg, storage, budget(), tmp_path)
        else:
            scheduler = run.Scheduler(cfg, storage, budget(), tmp_path)
            assert scheduler.selection == sealed == read_json(root / "selection.json")
            before = {p.name: p.read_bytes() for p in storage.unit_path("frozen-selection").iterdir()}
            scheduler.freeze_design()
            assert before == {p.name: p.read_bytes() for p in storage.unit_path("frozen-selection").iterdir()}


def test_rotating_arm_order_preserves_complete_blocks_without_tuning(tmp_path, monkeypatch):
    monkeypatch.setattr(run.original, "load_data", lambda _: {})
    monkeypatch.setattr(run.Scheduler, "pulse", lambda *a, **k: None)
    monkeypatch.setattr(run.Scheduler, "publish", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(run.Scheduler, "trajectory", lambda self, r, arm: calls.append((r, arm)))
    with Store(tmp_path / "study", {"test": "order"}, free_reserve_bytes=0) as storage:
        scheduler = run.Scheduler(smoke(), storage, budget(), tmp_path)
        scheduler.run()
        assert calls == [(0, "fixed_ridge"), (0, "fixed_zero"), (0, "kalman_zero"),
                         (1, "fixed_zero"), (1, "kalman_zero"), (1, "fixed_ridge")]
        assert storage.is_complete("frozen-selection")
        assert (storage.root / "study_completed.json").exists()
