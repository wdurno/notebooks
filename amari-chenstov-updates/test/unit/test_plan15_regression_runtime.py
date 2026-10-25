"""Small config/accounting/immutability checks, without data, GPU, or sleeps."""

import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnist_experiment.fisher_memory import artifacts, control
from mnist_experiment.fisher_memory.artifacts import ArtifactError, ImmutableError, Store
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested
from mnist_experiment.fisher_memory_regression import PROTOCOL, runtime
from mnist_experiment.fisher_memory_regression.config import Config, Recipe, content_hash


def identity(**overrides):
    return {"allocation_family": "plan15-phase8", "role": "production", "protocol": PROTOCOL,
            "source_hash": "frozen-source", "config_hash": Config().scientific_hash, **overrides}


def store_at(root, *, run_identity=None, **kwargs):
    return Store(root, identity() if run_identity is None else run_identity,
                 max_bytes=2_000_000, free_reserve_bytes=0,
                 finalization_reserve_bytes=4096, **kwargs)


class Clock:
    def __init__(self):
        self.now = 10_000.

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    # Patch only experiment modules, not pytest's own elapsed-time clock.
    fake_time = SimpleNamespace(monotonic=value.monotonic)
    for module in (runtime, control, artifacts):
        monkeypatch.setattr(module, "time", fake_time)
    return value


def test_config_defaults_json_roundtrip_hash_and_frozen_fields(tmp_path):
    cfg = Config()
    cfg.validate()
    mapped = cfg.mapping()
    assert mapped == json.loads(json.dumps(mapped, allow_nan=False))
    assert isinstance(mapped["capacities"], list)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(mapped))
    restored = Config.read(path)
    assert cfg == restored
    assert cfg.scientific_hash == restored.scientific_hash == content_hash(mapped)
    assert content_hash({"b": 2, "a": 1}) == content_hash({"a": 1, "b": 2})
    assert cfg.scientific_hash != dataclasses.replace(cfg, seed=cfg.seed + 1).scientific_hash
    assert cfg.scientific_hash != dataclasses.replace(cfg, ridge_grid=(.1, 1., 20.)).scientific_hash
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1
    with pytest.raises(ValueError):
        content_hash({"invalid": float("nan")})
    path.write_text(json.dumps({"unknown_option": 1}))
    with pytest.raises(TypeError):
        Config.read(path)


@pytest.mark.parametrize("changes", [
    {"protocol": "different"}, {"schema_version": 999}, {"device": "automatic"},
    {"dtype": "float16"}, {"smoke": 1}, {"seed": 0}, {"threads": True},
    {"init_size": -1}, {"init_epochs": 0}, {"reference_size": 0}, {"eval_per_class": 0},
    {"refinement_replicas": -1}, {"free_reserve_bytes": -1},
    {"init_learning_rate": float("nan")}, {"fd_step": float("inf")},
    {"analysis_reserve_seconds": 0.}, {"confirmation_reserve_seconds": -1.},
    {"angle_scale": 90.}, {"nominal_batch": 2}, {"pi": 1.}, {"pi": float("nan")},
    {"novel_probability": 0.}, {"ramp_groups": 1}, {"screen_novel_count": 100},
    {"reference_replay": 65}, {"capacities": ()}, {"capacities": (16, 16)},
    {"replay_counts": (True,)}, {"smaller_budgets": (100,)},
    {"ridge_grid": (0., 1.)}, {"strength_ratios": (float("inf"),)},
    {"strength_ratios": (.1, .1)}, {"max_artifact_bytes": 500_000_000_001},
    {"novel_count": 101}, {"novel_probability": .2}, {"ramp_groups": 41},
])
def test_config_rejects_invalid_and_unversioned_scientific_settings(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(Config(), **changes).validate()


def test_smoke_can_downscale_data_but_not_units_or_storage_ceiling():
    small = dataclasses.replace(Config(), smoke=True, novel_count=4, screen_novel_count=1,
                                smaller_budgets=(1, 2), novel_probability=.5, ramp_groups=2,
                                init_size=10, init_validation_size=5, reference_size=5,
                                init_epochs=1, eval_per_class=1)
    small.validate()
    for changes in ({"angle_scale": 1.}, {"nominal_batch": 1},
                    {"max_artifact_bytes": 500_000_000_001}):
        with pytest.raises(ValueError):
            dataclasses.replace(small, **changes).validate()


def test_named_seeds_are_reproducible_and_separate_across_components_and_cohorts():
    cfg = Config()
    components = ("initialization_model", "factors", "epoch:0", "epoch:1", "replay", "probes",
                  "compression", "stream_sources", "stream_digits", "stream_domains", "stream_angles_base")
    seeds = {(family, replica, component): cfg.seed_for(family, replica, component)
             for family in ("development", "confirmation") for replica in (0, 1)
             for component in components}
    assert len(set(seeds.values())) == len(seeds)
    assert all(0 <= seed < 2 ** 63 for seed in seeds.values())
    assert seeds == {key: cfg.seed_for(*key) for key in seeds}
    # Tuning treatment settings cannot change a replica's common random inputs.
    tuned = dataclasses.replace(cfg, reference_replay=8, ridge_grid=(.2, 2.))
    assert seeds == {key: tuned.seed_for(*key) for key in seeds}
    other = dataclasses.replace(cfg, seed=cfg.seed + 1)
    assert all(seed != other.seed_for(*key) for key, seed in seeds.items())


def test_config_rejects_seeds_outside_component_generator_range():
    cfg = dataclasses.replace(Config(), seed=2 ** 63)
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize("changes", [
    {"method": "unknown"}, {"capacity": True}, {"replay_count": 1.5},
    {"capacity": 0}, {"replay_count": 0}, {"replay_count": 65},
    {"ridge": -1.}, {"ridge": float("nan")}, {"strength": -1.},
    {"strength": float("inf")}, {"strength": 1.}, {"fd_step": 0.},
    {"fd_step": float("nan")}, {"probes": True}, {"max_iter": 0}, {"max_eval": -1},
    {"method": "ewc_only"}, {"method": "fd_replay_ewc"},
])
def test_recipe_validation(changes):
    with pytest.raises(ValueError):
        Recipe(**{"method": "replay_ewc", "capacity": 64, "replay_count": 16, **changes})


def test_recipe_construction_boundaries_overrides_and_stable_identity():
    cfg = Config()
    replay = Recipe.from_config(cfg, "replay_ewc")
    assert replay == Recipe("replay_ewc", 64, 16)
    assert Recipe(**replay.mapping()) == replay
    assert replay.key == "replay_ewc-" + content_hash(replay.mapping())[:16]
    assert dataclasses.replace(replay, replay_count=8).key != replay.key
    fd = Recipe.from_config(cfg, "fd_replay_ewc", capacity=16, replay_count=4, ridge=.1,
                            strength=3., fd_step=.005, policy={"max_iter": 7, "max_eval": 11})
    assert fd == Recipe("fd_replay_ewc", 16, 4, .1, 3., .005, cfg.probes, 7, 11)
    boundary = Recipe.from_config(cfg, "ewc_only")
    assert boundary.capacity == boundary.replay_count == boundary.strength == 0
    with pytest.raises(ValueError, match="control"):
        Recipe.from_config(cfg, "ewc_only", strength=1.)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fd.strength = 2.


def test_hard_crash_restores_last_reserved_lease_and_not_offline_time(tmp_path, clock):
    root = tmp_path / "production"
    crashed = store_at(root)
    try:
        budget = runtime.RuntimeBudget(crashed)
        assert budget.used == 0.
        assert runtime.read_json(root / "timing_lease.json")["charged_until"] == 60.
        clock.advance(12.)
        budget.check()
        lease = runtime.read_json(root / "timing_lease.json")
        assert lease["running"] and lease["charged_until"] == 72.
        assert runtime.read_json(root / "runtime.json")["cumulative_active_seconds"] == 12.
        clock.advance(3.)
    finally:
        # A hard kill releases the OS lock without close/heartbeat or lease cleanup.
        crashed._release_lock()
    clock.advance(20_000.)
    with store_at(root, resume=True) as reopened:
        resumed = runtime.RuntimeBudget(reopened)
        assert resumed.used == 72.
        assert resumed.remaining == 72_000. - 72.
        clock.advance(2.)
        assert resumed.used == 74.
        resumed.close()
    assert runtime.read_json(root / "timing_lease.json")["running"] is False
    assert runtime.read_json(root / "runtime.json")["cumulative_active_seconds"] == 74.


def test_clean_close_resume_does_not_charge_an_extra_minute(tmp_path, clock):
    root = tmp_path / "production"
    with store_at(root) as store:
        budget = runtime.RuntimeBudget(store)
        clock.advance(17.)
        budget.close()
        lease = runtime.read_json(root / "timing_lease.json")
        assert lease["running"] is False and lease["reservation_seconds"] == 0.
        assert lease["charged_until"] == 17.
    clock.advance(10_000.)
    with store_at(root, resume=True) as store:
        budget = runtime.RuntimeBudget(store)
        assert budget.used == 17.
        clock.advance(3.)
        budget.close()
    with store_at(root, resume=True) as store:
        budget = runtime.RuntimeBudget(store)
        assert budget.used == 20.
        budget.close()


def test_crash_reservation_does_not_decrease_a_newer_runtime_counter(tmp_path, clock):
    root = tmp_path / "production"
    with store_at(root) as store:
        runtime.RuntimeBudget(store)
        clock.advance(75.)
        store.heartbeat(force=True)
    with store_at(root, resume=True) as store:
        budget = runtime.RuntimeBudget(store)
        assert budget.used == 75.  # The surviving lease only reserved through 60.
        budget.close()


def test_lease_heartbeat_is_bounded_and_throttled(tmp_path, clock):
    with store_at(tmp_path / "production") as store:
        budget = runtime.RuntimeBudget(store)
        path = store.root / "timing_lease.json"
        first = path.read_bytes()
        clock.advance(runtime.HEARTBEAT_SECONDS - 1)
        budget.check()
        assert path.read_bytes() == first
        clock.advance(1)
        budget.check()
        assert runtime.read_json(path)["charged_until"] == runtime.HEARTBEAT_SECONDS + runtime.LEASE_SECONDS
        assert runtime.read_json(path)["reservation_seconds"] == 60.
        budget.close()


def test_other_compute_includes_superseded_production_and_acceptance_not_smoke(tmp_path, clock):
    own = tmp_path / "current"

    def peer(name, seconds, **overrides):
        with store_at(tmp_path / name, run_identity=identity(**overrides)) as store:
            store.control.restore_active_seconds(seconds)

    peer("current", 999.)
    peer("production-v1", 120., protocol="superseded-protocol", source_hash="old-source")
    peer("production-v2", 30.)
    peer("acceptance-v1", 12., role="acceptance", protocol="old-acceptance")
    peer("acceptance-v2", 8., role="acceptance")
    peer("cpu-smoke", 6., role="smoke")
    peer("cuda-smoke", 6., role="smoke")
    peer("other-experiment", 20_000., allocation_family="plan15-phase7")
    (tmp_path / "not-a-study").mkdir()
    (tmp_path / "README.txt").write_text("not a peer root")
    assert runtime.other_compute_seconds(tmp_path, own) == 170.


def test_peer_crash_lease_is_counted_once_and_clean_close_releases_it(tmp_path, clock):
    root = tmp_path / "peer"
    own = tmp_path / "current"
    store = store_at(root)
    try:
        store.control.restore_active_seconds(100.)
        budget = runtime.RuntimeBudget(store)
        assert runtime.other_compute_seconds(tmp_path, own) == 160.
        clock.advance(5.)
        store.heartbeat(force=True)
        assert runtime.other_compute_seconds(tmp_path, own) == 160.
        budget.close()
        assert runtime.other_compute_seconds(tmp_path, own) == 105.
    finally:
        store.close()


def test_twenty_hour_limit_aggregates_other_roots_and_survives_resume(tmp_path, clock):
    with store_at(tmp_path / "old", run_identity=identity(protocol="old")) as old:
        old.control.restore_active_seconds(71_990.)
    root = tmp_path / "current"
    other = runtime.other_compute_seconds(tmp_path, root)
    with store_at(root) as store:
        budget = runtime.RuntimeBudget(store, other_seconds=other)
        assert budget.allocation_seconds == 20 * 60 * 60
        assert budget.remaining == 10.
        clock.advance(9.)
        budget.check()
        assert budget.remaining == 1.
        clock.advance(1.)
        with pytest.raises(ResourceLimit, match="aggregate Phase 8"):
            budget.check()
        assert budget.remaining == 0.
        budget.close()
    clock.advance(50_000.)
    with store_at(root, resume=True) as store:
        budget = runtime.RuntimeBudget(store, other_seconds=runtime.other_compute_seconds(tmp_path, root))
        assert budget.used == 72_000.
        with pytest.raises(ResourceLimit, match="aggregate Phase 8"):
            budget.check()
        budget.close()


@pytest.mark.parametrize("stop_kind", ["marker", "flag", "wall", "active"])
def test_budget_propagates_store_stop_and_resource_limits_and_can_finalize(tmp_path, clock, stop_kind):
    options = {"max_wall_seconds": 3.} if stop_kind == "wall" else {}
    if stop_kind == "active":
        options["max_active_seconds"] = 3.
    with store_at(tmp_path / "production", **options) as store:
        budget = runtime.RuntimeBudget(store)
        if stop_kind == "marker":
            control.request_stop(store.root)
        elif stop_kind == "flag":
            store.control.stop_requested = True
        else:
            clock.advance(3.)
        expected = StopRequested if stop_kind in ("marker", "flag") else ResourceLimit
        with pytest.raises(expected):
            budget.check()
        budget.close()
        assert runtime.read_json(store.root / "timing_lease.json")["running"] is False


@pytest.mark.parametrize("field,value", [
    ("other_seconds", -1.), ("other_seconds", float("nan")), ("other_seconds", float("inf")),
    ("allocation_seconds", -1.), ("allocation_seconds", float("nan")),
    ("allocation_seconds", float("inf")),
])
def test_runtime_budget_rejects_invalid_accounting_inputs(tmp_path, clock, field, value):
    with store_at(tmp_path / "production") as store:
        with pytest.raises(ValueError):
            runtime.RuntimeBudget(store, **{field: value})


def test_completed_units_and_identity_collisions_require_new_root(tmp_path, clock):
    root = tmp_path / "regression-v1"
    metadata = {"family": "confirmation", "replica": 0, "recipe": "replay"}
    with store_at(root) as store:
        unit = store.begin("replica-0", metadata)
        store.write_json("replica-0", "metrics.json", {"novel_count": 1, "mse": 25.})
        store.complete("replica-0")
        sealed = {path.name: path.read_bytes() for path in unit.iterdir()}
        with pytest.raises(ImmutableError, match="immutable"):
            store.write_json("replica-0", "metrics.json", {"mse": 0.})
    with pytest.raises(ImmutableError, match="resume=True"):
        store_at(root)
    with store_at(root, resume=True) as store:
        assert store.begin("replica-0", metadata) is None
        assert store.is_complete("replica-0", full=True)
        with pytest.raises(ImmutableError, match="metadata changed"):
            store.begin("replica-0", {**metadata, "replica": 1})
        assert sealed == {path.name: path.read_bytes() for path in unit.iterdir()}
    for changed in (identity(source_hash="new-source"), identity(config_hash="new-config")):
        with pytest.raises(ImmutableError, match="use a new root"):
            store_at(root, run_identity=changed, resume=True)
    with store_at(tmp_path / "regression-v2", run_identity=identity(source_hash="new-source")) as store:
        assert store.begin("replica-0", metadata) is not None
    assert sealed == {path.name: path.read_bytes() for path in unit.iterdir()}


@pytest.mark.parametrize("value", [-1., float("inf"), float("nan")])
def test_invalid_peer_time_ledgers_fail_closed(tmp_path, clock, value):
    root = tmp_path / "peer"
    with store_at(root):
        pass
    (root / "runtime.json").write_text(json.dumps({"cumulative_active_seconds": value}))
    with pytest.raises(ArtifactError):
        runtime.other_compute_seconds(tmp_path, tmp_path / "current")


def test_invalid_own_crash_reservation_is_not_silently_ignored(tmp_path, clock):
    root = tmp_path / "production"
    with store_at(root) as store:
        store.write_json(None, "timing_lease.json", {"running": True, "charged_until": -1.})
    with store_at(root, resume=True) as store:
        with pytest.raises(ArtifactError, match="crash-time reservation"):
            runtime.RuntimeBudget(store)


def test_file_bytes_counts_nested_files_and_prunes_excluded_subtree(tmp_path, monkeypatch):
    root = tmp_path / "budget"
    root.mkdir()
    (root / "top.bin").write_bytes(b"123")
    nested = root / "nested"
    nested.mkdir()
    (nested / "payload.bin").write_bytes(b"45678")
    excluded = nested / "own-root"
    excluded.mkdir()
    (excluded / "not-counted.bin").write_bytes(b"x" * 100)
    assert runtime.file_bytes(root) == 108
    scanned = []
    scandir = runtime.os.scandir

    def record(path):
        scanned.append(Path(path))
        return scandir(path)

    monkeypatch.setattr(runtime.os, "scandir", record)
    assert runtime.file_bytes(root, exclude=excluded) == 8
    assert excluded not in scanned
    assert runtime.file_bytes(root, excluded) == 8
    assert runtime.file_bytes(root, exclude=root) == 0
    assert runtime.file_bytes(tmp_path / "absent") == 0
    assert runtime.file_bytes(root / "top.bin") == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    assert runtime.file_bytes(empty) == 0


@pytest.mark.parametrize("kind", ["file", "directory", "dangling", "root", "excluded-directory"])
def test_file_bytes_rejects_file_and_directory_symlinks(tmp_path, kind):
    root = tmp_path / "budget"
    root.mkdir()
    target = tmp_path / "target"
    if kind == "file":
        target.write_bytes(b"payload")
    elif kind != "dangling":
        target.mkdir()
        (target / "payload.bin").write_bytes(b"payload")
    link = tmp_path / "root-link" if kind == "root" else root / "link"
    link.symlink_to(target, target_is_directory=kind in ("directory", "root", "excluded-directory"))
    kwargs = {"exclude": link} if kind == "excluded-directory" else {}
    with pytest.raises(ValueError, match="must not be a symlink"):
        runtime.file_bytes(link if kind == "root" else root, **kwargs)


def test_file_bytes_stop_is_checked_before_listing_and_within_flat_directory(tmp_path, monkeypatch):
    for index in range(20):
        (tmp_path / f"{index}.bin").write_bytes(b"x")
    calls = []

    def stop():
        calls.append(None)
        if len(calls) == 5:
            raise StopRequested("stop within directory")

    with pytest.raises(StopRequested, match="within directory"):
        runtime.file_bytes(tmp_path, check=stop)
    assert len(calls) == 5

    def stop_immediately():
        raise StopRequested("before listing")

    def forbidden_listing(*args, **kwargs):
        pytest.fail("directory was listed before the first stop check")

    monkeypatch.setattr(runtime.os, "scandir", forbidden_listing)
    with pytest.raises(StopRequested, match="before listing"):
        runtime.file_bytes(tmp_path, check=stop_immediately)


def test_file_bytes_does_not_silently_ignore_enumeration_failure(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("unreadable budget dependency")

    monkeypatch.setattr(runtime.os, "scandir", denied)
    with pytest.raises(PermissionError, match="unreadable"):
        runtime.file_bytes(tmp_path)


def test_scientific_sources_cover_direct_reused_imports_and_compression_implementation(monkeypatch):
    sources = runtime.scientific_sources()
    for name in ("mnist_experiment/fisher_memory/initialize.py", "src/lanczos.py"):
        assert sources[name] == hashlib.sha256((runtime.ROOT / name).read_bytes()).hexdigest()
    assert "mnist_experiment/fisher_memory_regression/report.py" not in sources
    assert "mnist_experiment/fisher_memory_regression/acceptance.py" not in sources

    for filename in runtime.PACKAGE.glob("*.py"):
        if filename.name in ("report.py", "acceptance.py"):
            continue
        for node in ast.walk(ast.parse(filename.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                if not module.startswith(("src.", "mnist_experiment.")):
                    continue
                path = module.replace(".", "/") + ".py"
                if (runtime.ROOT / path).is_file():
                    assert path in sources, f"unhashed direct scientific import in {filename.name}: {module}"

    original_digest = runtime.file_digest
    for dependency in ("mnist_experiment/fisher_memory/initialize.py", "src/lanczos.py"):
        def changed_digest(path, dependency=dependency):
            return "changed-scientific-source" if path == runtime.ROOT / dependency else original_digest(path)

        monkeypatch.setattr(runtime, "file_digest", changed_digest)
        assert content_hash(runtime.scientific_sources()) != content_hash(sources)
