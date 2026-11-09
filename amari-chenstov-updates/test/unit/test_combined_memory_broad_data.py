"""Bounded Phase 5 data/config checks, with synthetic sources and no training."""

from dataclasses import replace
import hashlib
import io
import itertools
import json
from pathlib import Path

import pytest
import torch

from mnist_experiment.combined_memory import data as original
from mnist_experiment.combined_memory.config import Config as OriginalConfig
from mnist_experiment.combined_memory.broad_drift import PROTOCOL, SCHEMA_VERSION
from mnist_experiment.combined_memory.broad_drift import data
from mnist_experiment.combined_memory.broad_drift.config import Config, content_hash
from src.seeding import derive_component_seed


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def raw():
    return dict(train_y=torch.cat([torch.full((40 + 5 * digit,), digit, dtype=torch.long)
                                   for digit in range(10)]),
                test_y=torch.arange(10).repeat_interleave(30))


@pytest.fixture(scope="module")
def cfg():
    return Config.smoke_config()


@pytest.fixture(scope="module")
def assets(raw, cfg):
    return data.make_assets(cfg, 0, "development", raw)


def test_config_defaults_projection_and_versioned_files(cfg):
    production = Config()
    production.validate()
    assert production.protocol == PROTOCOL == "plan18-broad-drift-v1"
    assert production.schema_version == SCHEMA_VERSION == 1
    assert production.seed == 18110705
    assert production.analysis_novel_count == production.outbound_budget == 1216
    assert production.compatibility_projection().mapping() == OriginalConfig(seed=18110705).mapping()
    assert Config.from_mapping(production.mapping()) == production
    assert cfg.analysis_novel_count == cfg.ramp_steps * cfg.samples_per_step == 6
    smaller = replace(cfg, ramp_steps=2, analysis_novel_count=None)
    assert smaller.analysis_novel_count == 4
    smaller.validate()
    directory = Path(data.__file__).parent / "configs"
    assert Config.read(directory / "broad_drift.json") == production
    for device in ("cpu", "cuda"):
        assert Config.read(directory / f"smoke_{device}.json") == Config.smoke_config(device)
    assert cfg.scientific_hash != OriginalConfig.smoke_config().scientific_hash
    replace(production, main_replicas=4).validate()


@pytest.mark.parametrize("change", [
    dict(protocol="plan18-combined-memory-v1"), dict(schema_version=2),
    dict(samples_per_step=2, analysis_novel_count=None), dict(nominal_arrival=2), dict(hold_steps=159),
    dict(ramp_steps=303, analysis_novel_count=None), dict(novel_probability=.25), dict(angle_max=46.),
    dict(angle_std=7.), dict(angle_limit=71.), dict(capacity=31), dict(replay_count=3),
    dict(probes=3), dict(fd_step=.02), dict(analysis_novel_count=100),
    dict(init_size=2299, initialization_size=2299), dict(main_replicas=5),
    dict(stationary_replicas=3), dict(smoke="true"), dict(seed=2**63),
])
def test_production_validation_never_bypasses_original_constraints(change):
    with pytest.raises(ValueError):
        replace(Config(), **change).validate()


def test_new_component_seeds_and_only_legacy_partition_exception(cfg):
    for component in ("initialization_model", "training_sources", "stream:transforms", "test"):
        expected = derive_component_seed(cfg.seed, f"{PROTOCOL}:development:0:{component}")
        assert cfg.seed_for("development", 0, component) == expected
        assert expected != OriginalConfig.smoke_config().seed_for("development", 0, component)
    partition = OriginalConfig(seed=181106).seed_for("study-partition", 0, "test")
    assert cfg.seed_for("study-partition", 0, "test") == partition
    assert replace(cfg, seed=99).seed_for("study-partition", 0, "test") == partition
    assert cfg.seed_for("study-partition", 1, "test") != partition
    assert cfg.seed_for("study-partition", 0, "other") != partition
    assert cfg.seed_for("main", 0, "training_sources") != cfg.seed_for("development", 0, "training_sources")
    for replica in (-1, True, 1.5):
        with pytest.raises(ValueError):
            cfg.seed_for("development", replica, "training_sources")


def test_production_schedule_unchanged_speed_quiet_intervals_and_boundaries():
    production = Config()
    h, segments = data.make_schedule(production)
    old_h, old_segments = original.make_schedule(OriginalConfig())
    torch.testing.assert_close(h, old_h, rtol=0, atol=0)
    assert segments == old_segments
    assert len(h) == 928 and production.stream_size == 3712
    assert [(s["start"], s["end"]) for s in segments] == [(0, 160), (160, 464), (464, 768), (768, 928)]
    assert not h[:160].any() and not h[768:].any()
    assert h[160] > 0 and h[463] == 1 and h[464] < 1 and h[767] == 0
    assert int((h == 1).sum()) == 1
    speed = float((h.diff() * production.angle_max).abs().max())
    assert speed == pytest.approx(.232516, abs=1e-5)
    assert production.ramp_steps * production.samples_per_step == 1216


def test_whole_stream_law_and_nominal_quiet_observations(cfg, assets):
    stream = assets["stream"]
    h = stream["h"].repeat_interleave(cfg.samples_per_step)
    means = h[:, None] * cfg.angle_max * assets["signs"]
    expected, info = data.sample_transforms(means.flatten(), assets["seeds"]["stream:transforms"],
        angle_std=cfg.angle_std, angle_limit=cfg.angle_limit,
        uniforms=stream["uniforms"].reshape(-1, 6))
    torch.testing.assert_close(stream["transforms"], expected.reshape(-1, 9, 6), rtol=0, atol=0)
    assert torch.equal(stream["p"], (stream["h"] > 0).double())
    assert not torch.equal(stream["p"].bool(), stream["acquisition"].reshape(-1, cfg.samples_per_step).any(1))
    quiet = h == 0
    nominal, _ = data.sample_transforms(torch.zeros(int(quiet.sum()) * 9),
        assets["seeds"]["stream:transforms"], angle_std=cfg.angle_std, angle_limit=cfg.angle_limit,
        uniforms=stream["uniforms"][quiet].reshape(-1, 6))
    torch.testing.assert_close(stream["transforms"][quiet], nominal.reshape(-1, 9, 6), rtol=0, atol=0)
    assert assets["metadata"]["transform_diagnostics"]["stream"]["parameter_mean"] == info["parameter_mean"]
    assert not assets["metadata"]["stream_law"]["bernoulli_component"]
    for panel in (stream, assets["panels"]["current"]):
        assert "mixture_uniforms" not in panel
    assert not any("mixture" in name for name in assets["seeds"])
    assert data.render_panel is original.render_panel
    assert data.render_batch is original.render_batch
    assert data.make_schedule is original.make_schedule
    assert data.sample_transforms is original.sample_transforms


def test_legacy_probability_cannot_change_resolved_law_or_acquisition(raw, cfg, assets):
    changed = data.make_assets(replace(cfg, novel_probability=.9), 0, "development", raw)
    assert changed["seeds"] == assets["seeds"]
    assert changed["metadata"]["source_allocation_hash"] == assets["metadata"]["source_allocation_hash"]
    assert changed["metadata"]["schedule"] == assets["metadata"]["schedule"]
    for key in ("transforms", "uniforms", "acquisition", "p", "cumulative_acquisition"):
        torch.testing.assert_close(changed["stream"][key], assets["stream"][key], rtol=0, atol=0)
    assert changed["stream"]["hash"] == assets["stream"]["hash"]


def test_outbound_acquisition_endpoint_and_no_return_contamination(cfg, assets):
    stream = assets["stream"]
    start, end = cfg.hold_steps, cfg.hold_steps + cfg.ramp_steps
    flags = stream["acquisition"].reshape(cfg.num_p_steps, cfg.samples_per_step)
    assert not flags[:start].any() and flags[start:end].all() and not flags[end:].any()
    assert torch.equal(stream["novel"], stream["acquisition"])
    counts = stream["cumulative_acquisition"]
    assert counts[start - 1] == 0
    torch.testing.assert_close(counts[start:end], torch.arange(1, cfg.ramp_steps + 1) * cfg.samples_per_step)
    assert (counts[end - 1:] == cfg.analysis_novel_count).all()
    assert stream["outbound_count"] == cfg.outbound_budget
    assert stream["p"][end] == 1 and not flags[end].any()
    torch.testing.assert_close(counts, stream["cumulative_novel"], rtol=0, atol=0)
    torch.testing.assert_close(counts.double(), stream["expected_cumulative_novel"], rtol=0, atol=0)
    metadata = assets["metadata"]
    assert metadata["acquisition"]["zero_accepted_step"] == start
    assert metadata["acquisition"]["endpoint_accepted_step"] == end
    assert metadata["acquisition"]["applicable"]
    assert not metadata["acquisition"]["return_advances_clock"]
    assert metadata["schedule"]["outbound_count"] == cfg.outbound_budget
    assert [s["acquisition_count"] for s in stream["segments"]] == [0, cfg.outbound_budget, 0, 0]
    assert metadata["schedule"]["expected_novel"] == metadata["schedule"]["realized_novel"] == cfg.outbound_budget
    assert "trajectory.novel_count" in metadata["legacy_aliases"]


def test_stationary_nominal_law_full_budget_and_no_acquisition(raw, cfg):
    assets = data.make_assets(cfg, 0, "stationary", raw)
    stream = assets["stream"]
    assert len(stream["ids"]) == cfg.stream_size
    for field in ("h", "p", "novel", "acquisition", "cumulative_acquisition", "cumulative_novel"):
        assert not stream[field].any()
    assert stream["outbound_count"] == 0
    assert not assets["metadata"]["acquisition"]["applicable"]
    assert [s["acquisition_count"] for s in stream["segments"]] == [0] * 4
    assert assets["metadata"]["schedule"]["expected_novel"] == 0
    means = torch.zeros(cfg.stream_size * 9)
    expected, _ = data.sample_transforms(means, assets["seeds"]["stream:transforms"],
        uniforms=stream["uniforms"].reshape(-1, 6))
    torch.testing.assert_close(stream["transforms"], expected.reshape(-1, 9, 6), rtol=0, atol=0)
    basis = assets["panels"]["current"]
    for step in (0, cfg.hold_steps + cfg.ramp_steps, cfg.num_p_steps):
        panel = data.current_panel(assets, step, cfg)
        torch.testing.assert_close(panel["transforms"], basis["transforms"], rtol=0, atol=0)


def test_current_whole_law_frozen_uniforms_and_no_mutation(cfg, assets):
    basis = assets["panels"]["current"]
    before = basis["transforms"].clone()
    endpoint = cfg.hold_steps + cfg.ramp_steps
    for step in (0, cfg.hold_steps, cfg.hold_steps + 1, endpoint, endpoint + 1, cfg.num_p_steps):
        panel = data.current_panel(assets, step, cfg)
        h = 0. if step == 0 else float(assets["stream"]["h"][step - 1])
        means = (h * cfg.angle_max * assets["signs"]).expand(cfg.eval_size, 9)
        expected, _ = data.sample_transforms(means.flatten(), assets["seeds"]["current:transforms"],
            angle_std=cfg.angle_std, angle_limit=cfg.angle_limit,
            uniforms=basis["uniforms"].reshape(-1, 6))
        torch.testing.assert_close(panel["transforms"], expected.reshape(-1, 9, 6), rtol=0, atol=0)
        for key in ("ids", "digits", "uniforms"):
            torch.testing.assert_close(panel[key], basis[key], rtol=0, atol=0)
        torch.testing.assert_close(panel["transforms"][:, :, 1:], before[:, :, 1:], rtol=0, atol=0)
        assert panel["h"] == h and panel["p"] == float(h > 0)
        assert "novel" not in panel and "mixture_uniforms" not in panel
        assert panel["hash"] == data.current_panel(assets, step, cfg)["hash"]
    torch.testing.assert_close(basis["transforms"], before, rtol=0, atol=0)
    final = data.current_panel(assets, cfg.num_p_steps, cfg)
    torch.testing.assert_close(final["transforms"], before, rtol=0, atol=0)
    peak = data.current_panel(assets, endpoint, cfg)
    assert peak["p"] == 1 and peak["h"] == 1
    assert not torch.equal(peak["transforms"][:, :, 0], before[:, :, 0])
    for invalid in (-1, cfg.num_p_steps + 1, True, 1.):
        with pytest.raises(ValueError):
            data.current_panel(assets, invalid, cfg)


def test_disjoint_uniform_sources_and_original_test_partition_fixed(raw, cfg, assets):
    roles = {**{role: assets[role]["ids"] for role in data.TRAIN_ROLES},
             **{role: assets["panels"][role]["ids"] for role in data.PANEL_ROLES}}
    for first, second in itertools.combinations(roles.values(), 2):
        assert not set(first.flatten().tolist()) & set(second.flatten().tolist())
    training = torch.cat([roles[role].flatten() for role in data.TRAIN_ROLES])
    generator = torch.Generator().manual_seed(assets["seeds"]["training_sources"])
    assert torch.equal(training, torch.randperm(len(raw["train_y"]), generator=generator)[:len(training)])
    assert training.unique().numel() == training.numel()
    legacy = original.make_assets(OriginalConfig.smoke_config(), 0, "development", raw)
    partition_seed = legacy["seeds"]["evaluation_partition"]
    order = torch.randperm(len(raw["test_y"]), generator=torch.Generator().manual_seed(partition_seed))
    old_dev = set(order[:len(order) // 2].tolist())
    old_confirmation = set(order[len(order) // 2:].tolist())
    fresh = data.make_assets(replace(cfg, seed=99), 1, "confirmation", raw)
    for item, pool in ((assets, old_dev), (fresh, old_confirmation)):
        assert item["seeds"]["evaluation_partition"] == partition_seed
        assert item["metadata"]["test_partition_hash"] == legacy["metadata"]["test_partition_hash"]
        for role in data.PANEL_ROLES:
            panel = item["panels"][role]
            ids = panel["ids"] - len(raw["train_y"])
            assert set(ids.flatten().tolist()) <= pool
            torch.testing.assert_close(panel["digits"], raw["test_y"][ids], rtol=0, atol=0)
    assert assets["metadata"]["evaluation_partition"]["preserved_from_original_study"]


def test_deterministic_hashes_named_seeds_and_coherent_provenance(raw, cfg, assets):
    state = torch.get_rng_state().clone()
    twin = data.make_assets(cfg, 0, "development", raw)
    assert torch.equal(state, torch.get_rng_state())
    for key in ("asset_hash", "source_allocation_hash", "source_hash", "config_hash"):
        assert twin["metadata"][key] == assets["metadata"][key]
    assert twin["seeds"] == assets["seeds"]
    for name, seed in assets["seeds"].items():
        if name != "evaluation_partition":
            assert seed == cfg.seed_for("development", 0, name)
    assert len(set(assets["seeds"].values())) == len(assets["seeds"])
    other = data.make_assets(cfg, 1, "development", raw)
    assert other["metadata"]["asset_hash"] != assets["metadata"]["asset_hash"]
    assert assets["metadata"]["assignment"]["global_sign"] == -1
    assert other["metadata"]["assignment"]["global_sign"] == 1
    assert int((assets["signs"] == 1).sum()) == 5
    assert int((other["signs"] == 1).sum()) == 4
    assert assets["metadata"]["config_hash"] == cfg.scientific_hash
    assert assets["metadata"]["source_hash"] == content_hash(assets["metadata"]["source_hashes"])
    root = Path(data.__file__).resolve().parents[3]
    for name, digest in assets["metadata"]["source_hashes"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert "mnist_experiment/combined_memory/broad_drift/data.py" in assets["metadata"]["source_hashes"]
    for panel in [assets[role] for role in data.TRAIN_ROLES] + list(assets["panels"].values()):
        assert panel["hash"] == data._panel_hash(panel)
    stream = assets["stream"]
    assert data._panel_hash(dict(stream, outbound_count=stream["outbound_count"] + 1)) != stream["hash"]
    json.dumps(assets["metadata"], allow_nan=False)
    memory = io.BytesIO()
    torch.save(assets, memory)
    assert memory.tell() < 200_000
    memory.seek(0)
    restored = torch.load(memory, weights_only=True)
    assert restored["metadata"]["asset_hash"] == assets["metadata"]["asset_hash"]


def test_asset_validation_remains_bounded(raw, cfg):
    with pytest.raises(ValueError, match="disjoint training sources"):
        data.make_assets(replace(cfg, init_size=1000, initialization_size=1000), 0, "development", raw)
    with pytest.raises(ValueError, match="disjoint test sources"):
        data.make_assets(replace(cfg, eval_size=7), 0, "development", raw)
    with pytest.raises(ValueError, match="cohort"):
        data.make_assets(cfg, 0, "unknown", raw)
