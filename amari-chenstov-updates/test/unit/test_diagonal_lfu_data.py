"""Fast CPU contracts for Plan 16 configuration and paired data assets."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.diagonal_lfu import data as module
from mnist_experiment.diagonal_lfu.config import Config, PROTOCOL


CONFIGS = Path(__file__).resolve().parents[2] / "mnist_experiment/diagonal_lfu/configs"


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg():
    return Config.from_json(CONFIGS / "smoke.json")


@pytest.fixture
def source():
    # Unequal class sizes deliberately distinguish raw frequencies from pool frequencies.
    train_y = torch.repeat_interleave(torch.arange(10), torch.arange(120, 150, 3))
    test_y = torch.arange(10).repeat_interleave(32)

    def images(n):
        return (torch.arange(n).float() / n)[:, None, None, None].expand(-1, 1, 28, 28).clone()

    return dict(train_x=images(len(train_y)), train_y=train_y,
                test_x=images(len(test_y)), test_y=test_y)


def test_default_config_roundtrip_and_scientific_identity(cfg, tmp_path):
    default = Config.from_json(CONFIGS / "default.json")
    assert default == Config()
    assert default.protocol == PROTOCOL and default.num_p_steps == 1312
    assert (default.hold_steps, default.ramp_steps, default.samples_per_step) == (352, 128, 4)
    assert default.control_cs == (.01, .025, .05)
    assert default.fisher_dtype == "float64"
    assert (cfg.hold_steps, cfg.ramp_steps, cfg.samples_per_step) == (2, 2, 2)
    assert (cfg.initialization_size, cfg.reference_size, cfg.eval_per_class) == (64, 16, 8)
    assert cfg.smoke and cfg.device == "cpu" and cfg.num_p_steps == 10
    assert default.scientific_hash != cfg.scientific_hash
    assert Config.from_mapping(json.loads(json.dumps(cfg.mapping()))) == cfg
    path = tmp_path / "config.json"
    path.write_text(json.dumps(default.mapping()))
    assert Config.from_json(path).scientific_hash == default.scientific_hash
    assert default.scientific_hash == replace(default, device="cpu", threads=1, main_replicas=9).scientific_hash
    assert default.scientific_hash != replace(default, dtype="float64").scientific_hash
    assert default.scientific_hash != replace(default, samples_per_step=8).scientific_hash
    assert not any(name in default.mapping() for name in ("root", "allocation_hours"))
    with pytest.raises(FrozenInstanceError):
        cfg.seed = 5


@pytest.mark.parametrize("changes", [
    {"schema_version": True}, {"schema_version": 0}, {"protocol": "old"},
    {"seed": -1}, {"seed": True}, {"seed": 2**63}, {"smoke": 1},
    {"samples_per_step": 0}, {"hold_steps": True}, {"ramp_steps": 1},
    {"initialization_size": 0}, {"validation_size": -1}, {"reference_size": 1.5},
    {"eval_per_class": 0}, {"eval_every": 0}, {"bootstrap_draws": 0},
    {"initial_lr": 0}, {"initial_lr": True}, {"learning_rate": float("inf")},
    {"tolerance_grad": 0}, {"tolerance_change": -1}, {"ridge_ratio": -.1},
    {"p_low": .2, "p_high": .1}, {"p_high": 1.1}, {"p_low": float("nan")},
    {"control_cs": ()}, {"control_cs": (.1, .1)}, {"control_cs": (0.,)},
    {"control_cs": (1.,)}, {"control_cs": (float("nan"),)}, {"control_cs": (True,)},
    {"device": "gpu"}, {"dtype": "float16"}, {"fisher_dtype": "float32"},
    {"max_iter": 10, "max_eval": 5}, {"dev_replicas": -1}, {"threads": 0},
])
def test_config_rejects_invalid_values(cfg, changes):
    with pytest.raises(ValueError):
        replace(cfg, **changes)


def test_config_rejects_runtime_fields_and_insufficient_production_holds():
    for values in ({"root": "cache/x"}, {"allocation_hours": 20}, {"control_cs": ".05"}, []):
        with pytest.raises(ValueError):
            Config.from_mapping(values)
    with pytest.raises(ValueError, match="five half-lives"):
        replace(Config(), control_cs=(.005,))
    replace(Config(), control_cs=(.005,), hold_steps=700)


def test_schedule_exact_lengths_endpoints_and_half_cosine(cfg):
    for candidate in (cfg, Config()):
        p, segments = module.make_schedule(candidate)
        stationary, stationary_segments = module.make_schedule(candidate, stationary=True)
        assert p.shape == stationary.shape == (candidate.num_p_steps,)
        assert p.dtype == torch.float64 and p.device.type == "cpu"
        assert torch.equal(stationary, torch.full_like(p, candidate.p_low))
        assert stationary_segments[0]["end"] == candidate.num_p_steps
        lengths = [s["end"] - s["start"] for s in segments]
        assert lengths == [candidate.hold_steps, candidate.ramp_steps, candidate.hold_steps,
                           candidate.ramp_steps, candidate.hold_steps]
        assert [s["start"] for s in segments] == [0, *[s["end"] for s in segments[:-1]]]
        rise = p[segments[1]["start"]:segments[1]["end"]]
        fall = p[segments[3]["start"]:segments[3]["end"]]
        assert rise[0] == candidate.p_low and rise[-1] == candidate.p_high
        assert fall[0] == candidate.p_high and fall[-1] == candidate.p_low
        assert bool((rise.diff() > 0).all()) and torch.equal(fall, rise.flip(0))
        phases = torch.linspace(0, 1, candidate.ramp_steps, dtype=torch.float64)
        torch.testing.assert_close(rise, candidate.p_low + (candidate.p_high - candidate.p_low)
                                   * (1 - torch.cos(torch.pi * phases)) / 2)


def test_global_partitions_and_all_selected_roles_are_disjoint(cfg, source):
    dev = module.make_assets(cfg, 0, "development", source)
    fresh = module.make_assets(cfg, 1, "main", source)
    role_sets = {role: set(ids.tolist()) for role, ids in dev["partitions"].items()}
    for role, ids in role_sets.items():
        assert torch.equal(dev["partitions"][role], fresh["partitions"][role])
        for other, other_ids in role_sets.items():
            if role != other:
                assert ids.isdisjoint(other_ids)
    assert set.union(*role_sets.values()) == set(range(len(source["train_y"]) + len(source["test_y"])))
    for assets, eval_role in ((dev, "evaluation_development"), (fresh, "evaluation_fresh")):
        for prefix, role in (("init", "initialization"), ("validation", "validation"),
                             ("reference", "reference"), ("stream", "online"),
                             ("stationary_stream", "online"), ("eval", eval_role)):
            ids = assets[f"{prefix}_ids"].flatten()
            assert set(ids.tolist()) <= role_sets[role]
            if prefix not in ("stream", "stationary_stream"):
                assert len(ids.unique()) == len(ids)
        assert torch.equal(assets["eval_y"].bincount(minlength=10), torch.full((10,), cfg.eval_per_class))
        for name in ("init", "validation", "reference"):
            assert torch.equal(assets[f"{name}_x"], source["train_x"][assets[f"{name}_ids"]])
        assert torch.equal(assets["eval_x"], source["test_x"][assets["eval_source_ids"]])
    assert set(dev["eval_ids"].tolist()).isdisjoint(fresh["eval_ids"].tolist())


def test_deterministic_pairing_component_seeds_and_no_global_rng_mutation(cfg, source):
    state = torch.random.get_rng_state().clone()
    first = module.make_assets(cfg, 0, "development", source)
    second = module.make_assets(replace(cfg, control_cs=(.01,), ridge_ratio=0., main_replicas=20),
                                0, "development", source)
    other = module.make_assets(cfg, 1, "development", source)
    assert torch.equal(state, torch.random.get_rng_state())
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, second[key]), key
            assert value.device.type == "cpu"
    assert first["metadata"] == second["metadata"]
    assert len(set(first["seeds"].values())) == len(first["seeds"])
    assert first["seeds"] != other["seeds"]
    assert not torch.equal(first["stream_ids"], other["stream_ids"])
    assert first["metadata"]["partition_hash"] == other["metadata"]["partition_hash"]
    assert first["seeds"]["stream:mixture"] != first["seeds"]["stationary_stream:mixture"]
    assert first["seeds"]["initialization_model"] == cfg.seed_for("development", 0, "initialization_model")


def test_frozen_raw_non9_frequencies_and_exposure_accounting(cfg, source):
    assets = module.make_assets(cfg, 0, "development", source)
    counts = source["train_y"].bincount(minlength=10)
    torch.testing.assert_close(assets["non9_probabilities"], counts[:9].double() / counts[:9].sum())
    pool_counts = source["train_y"][assets["partitions"]["online"]].bincount(minlength=10)
    assert not torch.equal(assets["non9_probabilities"], pool_counts[:9].double() / pool_counts[:9].sum())
    assert assets["metadata"]["raw_train_class_counts"] == counts.tolist()
    json.dumps(assets["metadata"], allow_nan=False)
    for prefix in ("stream", "stationary_stream"):
        ids, labels, p = (assets[f"{prefix}_{name}"] for name in ("ids", "y", "p"))
        meta = assets["metadata"]["streams"][prefix]
        assert ids.shape == labels.shape == (cfg.num_p_steps, cfg.samples_per_step)
        assert torch.equal(labels, source["train_y"][ids])
        torch.testing.assert_close(assets[f"{prefix}_expected_cumulative_9s"], (p * cfg.samples_per_step).cumsum(0))
        assert torch.equal(assets[f"{prefix}_realized_cumulative_9s"], (labels == 9).sum(1).cumsum(0))
        assert meta["arrival_count"] == cfg.num_p_steps
        assert meta["observation_count"] == ids.numel()
        assert meta["unique_id_count"] == len(ids.unique())
        assert meta["realized_9s"] == int((labels == 9).sum())
        assert meta["expected_9s"] == pytest.approx(float(p.sum()) * cfg.samples_per_step)
        for t in range(cfg.num_p_steps):
            assert assets[f"{prefix}_cumulative_unique_ids"][t] == len(ids[:t + 1].unique())
        assert sum(s["realized_9s"] for s in assets[f"{prefix}_segments"]) == meta["realized_9s"]
    assert assets["metadata"]["role_class_counts"]["init"][9] == int((assets["init_y"] == 9).sum())


def test_online_replacement_counted_and_sampling_failure_is_explicit(cfg, source):
    repeated = module.make_assets(replace(cfg, samples_per_step=200), 0, "development", source)
    assert repeated["metadata"]["streams"]["stream"]["observation_count"] > repeated["metadata"]["streams"]["stream"]["unique_id_count"]
    with pytest.raises(ValueError, match="insufficient digit"):
        module.make_assets(replace(cfg, initialization_size=10000), 0, "development", source)
    with pytest.raises(ValueError, match="eval_per_class"):
        module.make_assets(replace(cfg, eval_per_class=17), 0, "development", source)
    with pytest.raises(ValueError, match="cohort"):
        module.make_assets(cfg, 0, "unknown", source)
    with pytest.raises(ValueError, match="replica"):
        module.make_assets(cfg, True, "development", source)


def test_cached_loader_never_downloads_and_uses_fixed_normalization(monkeypatch, tmp_path):
    calls = []

    def cached(root, *, download):
        calls.append((root, download))
        dataset = SimpleNamespace(data=torch.full((2, 28, 28), 255, dtype=torch.uint8),
                                  targets=torch.tensor([0, 9]))
        return dataset, dataset

    monkeypatch.setattr(module, "load_mnist_datasets", cached)
    result = module.load_data(tmp_path)
    assert calls == [(tmp_path, False)]
    assert set(result) == {"train_x", "train_y", "test_x", "test_y"}
    assert result["train_x"].shape == (2, 1, 28, 28)
    assert result["train_x"].dtype == torch.float32 and bool((result["train_x"] == 1).all())
    assert result["train_y"].dtype == torch.long


def test_bad_in_memory_data_rejected(cfg, source):
    broken = dict(source, train_y=source["train_y"].float())
    with pytest.raises(ValueError, match="long vectors"):
        module.make_assets(cfg, 0, "development", broken)
    broken = dict(source, test_x=source["test_x"] + 1)
    with pytest.raises(ValueError, match="normalized"):
        module.make_assets(cfg, 0, "development", broken)


def test_image_validation_uses_bounded_chunks_and_checks_the_tail(cfg, source, monkeypatch):
    finite = torch.isfinite
    sizes = []

    def bounded(values):
        sizes.append(len(values))
        assert len(values) <= 512
        return finite(values)

    monkeypatch.setattr(torch, "isfinite", bounded)
    module.make_assets(cfg, 0, "development", source)
    assert len(sizes) >= 4
    source["train_x"][-1, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module.make_assets(cfg, 0, "development", source)
