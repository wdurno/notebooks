"""Fast CPU-only checks of the bounded Plan 17 config/data/runtime layer."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.diagonal_lfu import config as frozen_config, data as frozen_data
from mnist_experiment.fisher_memory import artifacts, control as frozen_control
from mnist_experiment.fisher_memory.artifacts import ArtifactError, ImmutableError
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested
from mnist_experiment.fisher_memory_regression import runtime as frozen_runtime
from mnist_experiment.kalman_pi import control, data, runtime
from mnist_experiment.kalman_pi.config import Config, PROTOCOL, content_hash


CONFIGS = Path(__file__).resolve().parents[2] / "mnist_experiment/kalman_pi/configs"


@pytest.fixture
def cfg():
    return Config.from_json(CONFIGS / "smoke.json")


@pytest.fixture(scope="module")
def source():
    # Unequal per-class counts exercise raw rather than partition frequencies.
    train_y = torch.cat([torch.full((140 + 3 * k,), k, dtype=torch.long) for k in range(10)])
    test_y = torch.arange(10).repeat_interleave(20)
    return {"train_y": train_y, "test_y": test_y,
            "train_x": torch.zeros(len(train_y), 1, 28, 28),
            "test_x": torch.zeros(len(test_y), 1, 28, 28)}


class Clock:
    now = 10_000.

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    for module in (runtime, frozen_runtime, artifacts, frozen_control):
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=clock.monotonic))
    return clock


@pytest.fixture
def environment(tmp_path, monkeypatch, clock):
    repo = tmp_path / "repo"
    family = repo / "cache/mnist_experiment/plan17"
    family.mkdir(parents=True)
    dataset = repo / "cache/datasets"
    dataset.mkdir(parents=True)
    monkeypatch.setattr(runtime, "REPO", repo)
    monkeypatch.setattr(runtime, "FAMILY", family)
    monkeypatch.setattr(runtime, "setup", lambda cfg: None)

    def identity(cfg, data_root, check=lambda: None):
        check()
        return {"allocation_family": "plan17", "protocol": cfg.protocol,
                "config": cfg.mapping(), "config_hash": cfg.scientific_hash,
                "sources": {"frozen-dependency": "fixed-hash"},
                "git_commit": "commit-at-creation", "git_dirty_at_creation": True}

    monkeypatch.setattr(runtime, "provenance", identity)
    # This is only a mocked filesystem headroom, never relaxed in production.
    monkeypatch.setattr(artifacts.shutil, "disk_usage", lambda root: SimpleNamespace(free=10**12))
    return SimpleNamespace(repo=repo, family=family, dataset=dataset, root=family / "current")


def open_run(cfg, environment, **kwargs):
    options = dict(resume=False, clear_stop=False, allocation_hours=20., max_wall_seconds=None)
    options.update(kwargs)
    return runtime.open_store(cfg, environment.root, environment.dataset, **options)


def make_peer(environment, name, seconds, *, lease=None, allocation_family="plan17"):
    path = environment.family / name
    path.mkdir()
    (path / "identity.json").write_text(json.dumps({"identity": {"allocation_family": allocation_family}}))
    (path / "runtime.json").write_text(json.dumps({"cumulative_active_seconds": seconds}))
    if lease is not None:
        (path / "timing_lease.json").write_text(json.dumps({"running": True, "charged_until": lease}))
    return path


def test_defaults_and_frozen_shared_contract():
    cfg = Config()
    baseline = frozen_config.Config()
    assert cfg.protocol == PROTOCOL == "plan17-kalman-pi-v1"
    assert cfg.seed == 171101
    for field in dataclasses.fields(baseline):
        if field.name not in {"protocol", "seed"}:
            assert getattr(cfg, field.name) == getattr(baseline, field.name), field.name
    assert (cfg.pi_min, cfg.pi_max, cfg.cold_start, cfg.noise_half_life, cfg.anchor_refresh) == (
        1e-4, .2, 16, 32., 64)
    assert (cfg.observer_floor_relative, cfg.observer_floor_absolute) == (1e-6, 1e-12)
    assert (cfg.ema_half_lives, cfg.process_scales, cfg.schedule_factors) == (
        (8., 16., 32.), (1e-4, 1e-3, 1e-2), (1., 1.25))
    assert (cfg.calibration_size, cfg.shadow_replicas, cfg.candidates_per_family) == (256, 1, 2)
    assert Config.from_json(CONFIGS / "default.json") == cfg
    assert isinstance(cfg, frozen_config.Config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 0


def test_json_roundtrip_and_identity_boundaries(cfg, tmp_path):
    mapping = json.loads(json.dumps(cfg.mapping(), allow_nan=False))
    assert Config.from_mapping(mapping) == cfg
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(mapping))
    assert Config.from_json(path) == cfg
    assert content_hash(mapping) == content_hash(cfg.mapping())
    more_replicas = dataclasses.replace(cfg, main_replicas=100, stationary_replicas=30,
                                       dev_replicas=5, shadow_replicas=2, threads=2, device="cuda")
    assert more_replicas.scientific_hash == cfg.scientific_hash
    for changes in ({"seed": 2}, {"pi_min": .001}, {"anchor_refresh": 0},
                    {"noise_half_life": 64.}, {"schedule_factors": (1.,)},
                    {"calibration_size": 8}, {"cold_start": 1}, {"process_scales": (.01,)}):
        assert dataclasses.replace(cfg, **changes).scientific_hash != cfg.scientific_hash
    with pytest.raises(ValueError, match="unknown"):
        Config.from_mapping({"unrecognized": 0})
    with pytest.raises(ValueError, match="mapping"):
        Config.from_mapping([])


@pytest.mark.parametrize("changes", [
    {"protocol": frozen_config.PROTOCOL}, {"schema_version": 2}, {"seed": True},
    {"dtype": "float16"}, {"fisher_dtype": "float32"}, {"max_eval": 1},
    {"pi_min": 0}, {"pi_max": 1.}, {"pi_min": .3}, {"pi_max": float("nan")},
    {"pi_min": True}, {"cold_start": 0}, {"cold_start": True}, {"cold_start": -1},
    {"cold_start": 16}, {"noise_half_life": 0.}, {"noise_half_life": float("inf")},
    {"anchor_refresh": -1}, {"anchor_refresh": True}, {"calibration_size": 1},
    {"calibration_size": 65}, {"calibration_size": True}, {"observer_floor_relative": -1.},
    {"observer_floor_absolute": 0.}, {"ema_half_lives": []}, {"ema_half_lives": ()},
    {"ema_half_lives": (2., 2.)}, {"ema_half_lives": (float("nan"),)},
    {"process_scales": (0.,)}, {"process_scales": (True,)}, {"schedule_factors": (.5,)},
    {"shadow_replicas": 0}, {"shadow_replicas": 3}, {"candidates_per_family": 3},
])
def test_invalid_settings_rejected(cfg, changes):
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, **changes)


def test_production_timescales_and_fixed_anchor_ablation():
    assert dataclasses.replace(Config(), anchor_refresh=0).anchor_refresh == 0
    # Cold start/fixed control precede adaptive clipping even after a bound repair.
    assert dataclasses.replace(Config(), pi_max=.025).control_cs[-1] == .05
    for changes in ({"hold_steps": 320}, {"anchor_refresh": 400},
                    {"ema_half_lives": (100.,)}, {"noise_half_life": 100.}):
        with pytest.raises(ValueError):
            dataclasses.replace(Config(), **changes)


@pytest.mark.parametrize("field", ["control_cs", "ema_half_lives", "process_scales", "schedule_factors"])
def test_tuple_mapping_fields_reject_strings(cfg, field):
    with pytest.raises(ValueError, match="sequence"):
        Config.from_mapping({**cfg.mapping(), field: "invalid"})


def test_cpu_cuda_smoke_configs_have_post_cold_start_updates():
    cpu = Config.from_json(CONFIGS / "smoke.json")
    cuda = Config.from_json(CONFIGS / "smoke_cuda.json")
    for cfg in (cpu, cuda):
        assert cfg.smoke and cfg.cold_start == 2 and cfg.anchor_refresh == 4
        assert (cfg.hold_steps, cfg.ramp_steps) == (4, 2)
        assert cfg.num_p_steps > cfg.cold_start + cfg.anchor_refresh
        assert cfg.calibration_size <= cfg.initialization_size
    assert cpu.device == "cpu" and cuda.device == "cuda"


def test_named_plan17_seeds_are_separate_from_plan16_and_treatment(cfg):
    names = ("initialization_model", "initialization_order", "stream:mixture", "reference:inputs:0",
             "evaluation:0", "schedule_variation", "noise_calibration", "numerical", "bootstrap")
    seeds = {(cohort, replica, name): cfg.seed_for(cohort, replica, name)
             for cohort in ("development", "main", "stationary") for replica in (0, 1) for name in names}
    assert len(set(seeds.values())) == len(seeds)
    tuned = dataclasses.replace(cfg, pi_min=.001, anchor_refresh=0, process_scales=(.1,))
    assert seeds == {key: tuned.seed_for(*key) for key in seeds}
    baseline = frozen_config.Config(seed=cfg.seed)
    assert all(value != baseline.seed_for(*key) for key, value in seeds.items())
    for replica in (-1, True):
        with pytest.raises(ValueError, match="replica"):
            cfg.seed_for("main", replica, "stream")


def test_assets_reuse_frozen_sampler_and_never_mutate_global_rng(cfg, source, monkeypatch):
    original = frozen_data.make_assets
    calls = []

    def record(resolved, replica, cohort, values):
        calls.append(resolved)
        return original(resolved, replica, cohort, values)

    monkeypatch.setattr(frozen_data, "make_assets", record)
    state = torch.random.get_rng_state().clone()
    assets = data.make_assets(cfg, 0, "development", source)
    tuned = data.make_assets(dataclasses.replace(cfg, pi_min=.001, anchor_refresh=0), 0, "development", source)
    assert torch.equal(state, torch.random.get_rng_state())
    assert len(calls) == 2 and all(isinstance(c, Config) for c in calls)
    for key, value in assets.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, tuned[key]), key
            assert value.device.type == "cpu"
    assert assets["metadata"] == tuned["metadata"]
    assert assets["non9_weights"] is assets["non9_probabilities"]
    assert len(set(assets["seeds"].values())) == len(assets["seeds"])
    assert assets["metadata"]["protocol"] == PROTOCOL
    assert assets["seeds"]["noise_calibration"] == cfg.seed_for("development", 0, "noise_calibration")


def test_global_roles_eval_halves_and_initialization_calibration_are_disjoint(cfg, source):
    dev = data.make_assets(cfg, 0, "development", source)
    fresh = data.make_assets(cfg, 1, "main", source)
    role_sets = {name: set(ids.tolist()) for name, ids in dev["partitions"].items()}
    for name, ids in role_sets.items():
        assert torch.equal(dev["partitions"][name], fresh["partitions"][name])
        assert all(ids.isdisjoint(other) for key, other in role_sets.items() if key != name)
    assert set.union(*role_sets.values()) == set(range(len(source["train_y"]) + len(source["test_y"])))
    assert set(dev["eval_ids"].tolist()).isdisjoint(fresh["eval_ids"].tolist())
    for assets in (dev, fresh):
        calibration = set(assets["calibration_ids"].tolist())
        assert len(calibration) == cfg.calibration_size
        assert calibration <= set(assets["init_ids"].tolist())
        for name in ("reference", "validation", "eval", "stream", "stationary_stream"):
            assert calibration.isdisjoint(set(assets[f"{name}_ids"].flatten().tolist()))
        for suffix in ("ids", "x", "y"):
            assert torch.equal(assets[f"calibration_{suffix}"],
                               assets[f"init_{suffix}"][assets["calibration_indices"]])
        assert assets["metadata"]["calibration"]["additional_labels"] == 0
        json.dumps(assets["metadata"], allow_nan=False)


@pytest.mark.parametrize("cohort", ["main", "fresh", "stationary"])
def test_fresh_schedule_factor_is_seeded_before_sampling_and_shared_across_arms(cfg, source, cohort):
    cfg = dataclasses.replace(cfg, schedule_factors=(1.25,))
    assets = data.make_assets(cfg, 2, cohort, source)
    assert assets["schedule_factor"] == 1.25
    meta = assets["metadata"]["schedule_variation"]
    assert meta["seed"] == cfg.seed_for(cohort, 2, "schedule_variation")
    assert (meta["hold_steps"], meta["ramp_steps"], meta["num_p_steps"]) == (5, 3, 21)
    assert meta["resolved_before_learning"]
    for prefix in ("stream", "stationary_stream"):
        labels, p = assets[f"{prefix}_y"], assets[f"{prefix}_p"]
        assert labels.shape == (21, cfg.samples_per_step)
        assert assets["metadata"]["streams"][prefix]["schedule_hash"] == content_hash(p.tolist())
        torch.testing.assert_close(assets[f"{prefix}_expected_cumulative_9s"],
                                   (p * cfg.samples_per_step).cumsum(0))
        assert int(assets[f"{prefix}_realized_cumulative_9s"][-1]) == int((labels == 9).sum())
    assert assets["stationary_stream_segments"][0]["end"] == 21
    assert [segment["arrivals"] for segment in assets["stream_segments"]] == [5, 3, 5, 3, 5]
    dev = data.make_assets(cfg, 2, "development", source)
    assert dev["schedule_factor"] == 1. and len(dev["stream_p"]) == cfg.num_p_steps


def test_schedule_factor_uses_named_generator_not_replica_order(cfg, source):
    for replica in (3, 0):
        seed = cfg.seed_for("main", replica, "schedule_variation")
        generator = torch.Generator().manual_seed(seed)
        expected = cfg.schedule_factors[int(torch.randint(2, (), generator=generator))]
        assets = data.make_assets(cfg, replica, "main", source)
        assert assets["schedule_factor"] == expected
        assert assets["seeds"]["schedule_variation"] == seed


def test_cached_loader_is_the_unchanged_no_download_loader():
    assert data.load_data is frozen_data.load_data


def test_source_manifest_includes_frozen_dependencies_excludes_reporting():
    sources = runtime.source_manifest()
    for name in ("config", "data", "trajectory", "curvature", "learner"):
        path = f"mnist_experiment/diagonal_lfu/{name}.py"
        assert sources[path] == hashlib.sha256((runtime.REPO / path).read_bytes()).hexdigest()
    for name in ("src/parameters.py", "src/seeding.py", "src/mnist_data.py",
                 "mnist_experiment/rotated_mnist/plan12/gauge.py",
                 "mnist_experiment/fisher_memory/artifacts.py",
                 "mnist_experiment/fisher_memory_regression/runtime.py"):
        assert name in sources
    for package in ("diagonal_lfu", "kalman_pi"):
        assert all(f"mnist_experiment/{package}/{name}.py" not in sources
                   for name in ("analysis", "report", "control"))


def test_frozen_learner_keeps_all_487_coordinates_trainable():
    from mnist_experiment.diagonal_lfu.curvature import build_model

    model, layout = build_model(42, device="cpu", dtype=torch.float64)
    assert layout.total_numel == 487
    assert sum(parameter.numel() for parameter in model.parameters()) == 487
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_provenance_hashes_cached_dataset_without_training_or_downloads(cfg, tmp_path):
    raw = tmp_path / "MNIST/raw"
    raw.mkdir(parents=True)
    names = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    for name in names:
        (raw / name).write_bytes(name.encode("ascii"))
    info = runtime.provenance(cfg, tmp_path)
    assert info["config_hash"] == cfg.scientific_hash
    assert info["allocation_family"] == "plan17" and info["fisher_dtype"] == "float64"
    assert info["metric_schema_version"] == info["artifact_schema_version"] == 1
    assert info["git_commit"] and type(info["git_dirty_at_creation"]) is bool
    for name in names:
        assert info["dataset"][name] == {"sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
                                         "bytes": len(name)}
    (raw / names[0]).unlink()
    with pytest.raises(FileNotFoundError, match="no automatic download"):
        runtime.provenance(cfg, tmp_path)


def test_store_bounds_immutable_units_and_explicit_resume(cfg, environment, clock):
    store, budget = open_run(cfg, environment)
    assert isinstance(store, artifacts.Store) and isinstance(budget, frozen_runtime.RuntimeBudget)
    try:
        assert (store.max_bytes, store.free_reserve_bytes) == (500_000_000_000, 20_000_000_000)
        assert budget.allocation_seconds == 72_000.
        store.begin("replica-0", {"kind": "test"})
        store.write_json("replica-0", "metrics.json", {"step": 1})
        store.complete("replica-0")
        sealed = {p.name: p.read_bytes() for p in store.unit_path("replica-0").iterdir()}
        with pytest.raises(ImmutableError, match="immutable"):
            store.write_json("replica-0", "metrics.json", {})
        clock.advance(7.)
        budget.close()
    finally:
        store.close()
    assert store.plan17_allocation_lock.closed
    with pytest.raises(ImmutableError, match="resume=True"):
        open_run(cfg, environment)
    store, budget = open_run(cfg, environment, resume=True)
    try:
        assert budget.used == 7.
        assert store.begin("replica-0", {"kind": "test"}) is None
        assert sealed == {p.name: p.read_bytes() for p in store.unit_path("replica-0").iterdir()}
        budget.close()
    finally:
        store.close()
    with pytest.raises(ImmutableError, match="new root"):
        open_run(dataclasses.replace(cfg, seed=999), environment, resume=True)


def test_family_lock_excludes_other_versions_and_releases_on_close(cfg, environment):
    store, budget = open_run(cfg, environment)
    other = SimpleNamespace(**vars(environment))
    other.root = environment.family / "new-version"
    try:
        with pytest.raises(ArtifactError, match="Plan 17 process"):
            open_run(cfg, other)
        budget.close()
    finally:
        store.close()
    second, second_budget = open_run(cfg, other)
    second_budget.close()
    second.close()


def test_all_versions_and_acceptance_share_plan17_not_plan16_time(environment):
    make_peer(environment, "production-v1", 100., lease=120.)
    make_peer(environment, "production-v2", 30.)
    make_peer(environment, "cpu-smoke", 2.)
    make_peer(environment, "foreign-allocation", 50_000., allocation_family="plan16")
    (environment.family / "acceptance_compute.json").write_text(json.dumps({"entries": [{"seconds": 3.}]}))
    assert runtime.peer_seconds(environment.root) == 155.
    assert runtime.peer_seconds(environment.family / "production-v1") == 35.


def test_crash_reservation_survives_resume_without_offline_charge(cfg, environment, clock):
    store, budget = open_run(cfg, environment)
    clock.advance(12.)
    budget.check()
    assert runtime.read_json(store.root / "timing_lease.json")["charged_until"] == 72.
    store._release_lock()  # Hard kill: no clean heartbeat or budget.close.
    clock.advance(10_000.)
    resumed, resumed_budget = open_run(cfg, environment, resume=True)
    try:
        assert resumed_budget.used == 72.
        resumed_budget.close()
    finally:
        resumed.close()


def test_clean_lease_survives_interrupt_between_lease_and_runtime_flush(cfg, environment, clock):
    store, budget = open_run(cfg, environment)
    clock.advance(7.)
    store.write_json(None, "timing_lease.json", {
        "running": False, "charged_until": 7., "reservation_seconds": 0.,
    }, finalizing=True)
    assert runtime.read_json(store.root / "runtime.json")["cumulative_active_seconds"] == 0.
    store._release_lock()
    clock.advance(500.)
    resumed, resumed_budget = open_run(cfg, environment, resume=True)
    try:
        assert resumed_budget.used == 7.
        resumed_budget.close()
    finally:
        resumed.close()


def test_clean_resume_and_wall_bound_cap_crash_reservation(cfg, environment, clock):
    store, budget = open_run(cfg, environment, max_wall_seconds=3.)
    try:
        lease = runtime.read_json(store.root / "timing_lease.json")
        assert lease["charged_until"] == lease["reservation_seconds"] == 3.
        clock.advance(2.)
        budget.close()
    finally:
        store.close()
    clock.advance(1_000.)
    resumed, resumed_budget = open_run(cfg, environment, resume=True, max_wall_seconds=5.)
    try:
        assert resumed_budget.used == 2.
        assert runtime.read_json(resumed.root / "timing_lease.json")["charged_until"] == 7.
        clock.advance(5.)
        with pytest.raises(ResourceLimit, match="wall-clock"):
            resumed_budget.check()
        resumed_budget.close()
    finally:
        resumed.close()


def test_startup_hashing_time_is_charged_and_reduces_invocation_headroom(cfg, environment, clock, monkeypatch):
    prior = runtime.provenance

    def timed(*args):
        clock.advance(3.)
        return prior(*args)

    monkeypatch.setattr(runtime, "provenance", timed)
    store, budget = open_run(cfg, environment, max_wall_seconds=5.)
    try:
        assert budget.used == 3.
        assert store.control.max_wall_seconds == 2.
        assert runtime.read_json(store.root / "timing_lease.json")["charged_until"] == 5.
        clock.advance(2.)
        with pytest.raises(ResourceLimit, match="wall-clock"):
            budget.check()
        budget.close()
    finally:
        store.close()


def test_cumulative_bound_counts_peers_caps_lease_and_cannot_reset(cfg, environment, clock):
    make_peer(environment, "old-version", 71_990.)
    store, budget = open_run(cfg, environment)
    try:
        assert budget.remaining == 10.
        assert runtime.read_json(store.root / "timing_lease.json")["charged_until"] == 10.
        clock.advance(10.)
        with pytest.raises(ResourceLimit):
            budget.check()
        budget.close()
    finally:
        store.close()
    with pytest.raises(ResourceLimit, match="Plan 17"):
        open_run(cfg, environment, resume=True)


def test_exhausted_crash_reservation_is_rejected_before_rewriting_store(cfg, environment):
    make_peer(environment, "old-version", 71_950.)
    store, budget = open_run(cfg, environment)
    assert runtime.read_json(store.root / "timing_lease.json")["charged_until"] == 50.
    store._release_lock()
    before = {p.name: p.read_bytes() for p in environment.root.iterdir() if p.is_file()}
    with pytest.raises(ResourceLimit, match="Plan 17"):
        open_run(cfg, environment, resume=True)
    assert before == {p.name: p.read_bytes() for p in environment.root.iterdir() if p.is_file()}


def test_smaller_resume_allocation_cannot_reset_consumed_time(cfg, environment, clock):
    store, budget = open_run(cfg, environment, allocation_hours=1.)
    clock.advance(1800.)
    budget.close()
    store.close()
    with pytest.raises(ResourceLimit, match="Plan 17"):
        open_run(cfg, environment, resume=True, allocation_hours=.5)
    resumed, resumed_budget = open_run(cfg, environment, resume=True, allocation_hours=1.)
    assert resumed_budget.remaining == 1800.
    resumed_budget.close()
    resumed.close()


def test_changed_scientific_dependency_rejects_resume_and_releases_family_lock(cfg, environment, monkeypatch):
    store, budget = open_run(cfg, environment)
    budget.close()
    store.close()
    prior = runtime.provenance

    def changed(*args):
        result = prior(*args)
        result["sources"]["frozen-dependency"] = "changed-hash"
        return result

    monkeypatch.setattr(runtime, "provenance", changed)
    with pytest.raises(ImmutableError, match="new root"):
        open_run(cfg, environment, resume=True)
    monkeypatch.setattr(runtime, "provenance", prior)
    resumed, resumed_budget = open_run(cfg, environment, resume=True)
    resumed_budget.close()
    resumed.close()


@pytest.mark.parametrize("hours", [0., -1., 20.001, float("nan"), float("inf"), True])
def test_allocation_cannot_exceed_authorization_or_accept_invalid_bounds(cfg, environment, hours):
    with pytest.raises(ValueError, match="allocation_hours"):
        open_run(cfg, environment, allocation_hours=hours)
    assert not environment.root.exists()


@pytest.mark.parametrize("bound", [-1., float("nan"), float("inf"), True])
def test_invalid_wall_bounds(cfg, environment, bound):
    with pytest.raises(ValueError, match="max_wall_seconds"):
        open_run(cfg, environment, max_wall_seconds=bound)


@pytest.mark.parametrize("seconds", [72_001., float("nan"), -1., True])
def test_runtime_budget_itself_cannot_exceed_twenty_hours(tmp_path, seconds):
    with runtime.Store(tmp_path / "run", {"test": 17}, free_reserve_bytes=0) as store:
        with pytest.raises(ValueError, match="allocation_seconds"):
            runtime.RuntimeBudget(store, allocation_seconds=seconds)


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), True])
def test_corrupt_time_ledgers_fail_closed(environment, value):
    make_peer(environment, "invalid", value)
    with pytest.raises(ArtifactError):
        runtime.peer_seconds(environment.root)


@pytest.mark.parametrize("payload", [{}, [], {"running": "yes", "charged_until": 1.}])
def test_malformed_crash_ledger_fails_closed(environment, payload):
    peer = make_peer(environment, "invalid", 1.)
    (peer / "timing_lease.json").write_text(json.dumps(payload))
    with pytest.raises(ArtifactError, match="ledger|reservation"):
        runtime.peer_seconds(environment.root)


def test_stop_status_and_clear_stop_are_explicit_and_read_only(cfg, environment, capsys):
    store, budget = open_run(cfg, environment)
    store.write_json(None, "progress.json", {"phase": "tiny-test"})
    try:
        assert control.main(["status", "--root", str(store.root)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["writer_active"] and result["progress"]["phase"] == "tiny-test"
        control.main(["stop", "--root", str(store.root)])
        assert json.loads(capsys.readouterr().out)["stop_requested"]
        with pytest.raises(StopRequested):
            budget.check()
        budget.close()
    finally:
        store.close()
    with pytest.raises(StopRequested):
        open_run(cfg, environment, resume=True)
    assert (environment.root / "STOP").exists()
    with pytest.raises(ValueError, match="resume=True"):
        open_run(cfg, environment, clear_stop=True)
    resumed, resumed_budget = open_run(cfg, environment, resume=True, clear_stop=True)
    assert not (environment.root / "STOP").exists()
    resumed_budget.close()
    resumed.close()


def test_storage_counts_nested_dataset_once_and_refreshes_peer_growth(cfg, environment, clock):
    # Nesting under a counted family must not charge the dataset a second time.
    dataset = environment.family / "shared-dataset"
    dataset.mkdir()
    (dataset / "dataset.bin").write_bytes(b"d" * 101)
    old = environment.repo / "cache/mnist_experiment/plan16"
    old.mkdir(parents=True)
    (old / "checkpoint.bin").write_bytes(b"p" * 37)
    environment.dataset = dataset
    assert runtime.external_bytes(environment.root, dataset) == 138
    store, budget = open_run(cfg, environment)
    try:
        assert store.external_bytes == 138
        (old / "more.bin").write_bytes(b"m" * 17)
        clock.advance(runtime.HEARTBEAT_SECONDS)
        budget.check()
        assert store.external_bytes == 155
        budget.close()
    finally:
        store.close()


def test_storage_ceiling_includes_plan16_and_twenty_gb_free_reserve(cfg, environment, monkeypatch):
    monkeypatch.setattr(runtime, "external_bytes", lambda *args: runtime.MAX_BYTES + 1)
    with pytest.raises(ResourceLimit, match="byte allocation"):
        open_run(cfg, environment)
    monkeypatch.setattr(runtime, "external_bytes", lambda *args: 0)
    monkeypatch.setattr(artifacts.shutil, "disk_usage", lambda path: SimpleNamespace(free=19_999_999_999))
    with pytest.raises(ResourceLimit, match="filesystem space"):
        open_run(cfg, environment)


def test_storage_symlinks_are_rejected(environment):
    link = environment.repo / "cache/link"
    link.symlink_to(environment.dataset, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        runtime.external_bytes(environment.root, link)


def test_setup_refuses_silent_cuda_fallback(cfg, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        runtime.setup(dataclasses.replace(cfg, device="cuda"))
