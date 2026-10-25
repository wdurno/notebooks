"""CPU-only data-contract tests using tiny in-memory cached MNIST stand-ins."""

import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist_experiment.fisher_memory import data as old_data
from mnist_experiment.fisher_memory_regression import data
from mnist_experiment.fisher_memory_regression.config import Config
from mnist_experiment.rotated_mnist.transform import rotate_mnist_batch


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class CachedSource(data.MNISTSource):
    def __init__(self):
        self.train_labels = torch.repeat_interleave(torch.arange(10), torch.arange(100, 200, 10))
        self.test_labels = torch.arange(10).repeat_interleave(24)
        self.train = SimpleNamespace(data=self._pixels(len(self.train_labels)), targets=self.train_labels)
        self.test = SimpleNamespace(data=self._pixels(len(self.test_labels)), targets=self.test_labels)

    @staticmethod
    def _pixels(count):
        pixels = torch.zeros((count, 28, 28), dtype=torch.uint8)
        pixels[:, 8:18, 16:21] = 180
        pixels[:, 0, :] = 255
        pixels[:, 12, 20] = torch.arange(count).remainder(256).to(torch.uint8)
        return pixels


@pytest.fixture
def source():
    return CachedSource()


@pytest.fixture
def cfg():
    return dataclasses.replace(Config(), smoke=True, device="cpu", init_size=40,
                               init_validation_size=12, reference_size=20, init_epochs=2,
                               eval_per_class=4, novel_count=12, ramp_groups=3,
                               novel_probability=.5)


@pytest.mark.parametrize("domain", data.DOMAINS)
def test_angle_sampling_is_seeded_float32_and_global_rng_independent(domain):
    state = torch.random.get_rng_state().clone()
    first = data.sample_angles(2_000, domain, 13)
    assert first.shape == (2_000,)
    assert first.dtype == torch.float32 and first.device.type == "cpu"
    assert torch.equal(first, data.sample_angles(2_000, domain, 13))
    assert torch.equal(first[:71], data.sample_angles(71, domain, 13))
    assert torch.equal(first[:3], data.sample_angles(3, domain, 13))
    assert not torch.equal(first, data.sample_angles(2_000, domain, 14))
    assert torch.equal(state, torch.random.get_rng_state())
    assert data.sample_angles(0, domain, 13).shape == (0,)
    bound = 10 if domain == "base" else 90
    assert (first.abs() < bound).all()
    assert .45 < (first > 0).double().mean() < .55


def test_novel_rejection_not_clipping_and_sign_independence(monkeypatch):
    def fake_normal(count, **kwargs):
        result = torch.zeros(count, dtype=kwargs["dtype"])
        result[::2] = 4.  # 40 * exp(1) > 90: these must be rejected.
        return result

    monkeypatch.setattr(data.torch, "randn", fake_normal)
    angles = data.sample_angles(101, "novel", 123)
    assert torch.equal(angles.abs(), torch.full((101,), 40.))
    assert (angles > 0).any() and (angles < 0).any()


def test_analytic_densities_normalize_are_symmetric_and_match_sampler():
    x = np.linspace(-100, 100, 40_001)
    pdf = data.angle_density(x, "novel")
    integral = torch.trapezoid(torch.from_numpy(pdf), torch.from_numpy(x)).item()
    assert integral == pytest.approx(1., abs=2e-6)
    np.testing.assert_allclose(pdf, pdf[::-1], atol=1e-15)
    assert np.all(pdf[np.abs(x) > 90] == 0)
    assert data.angle_density(0., "novel") == 0
    assert data.angle_density(90., "novel") > 0
    mode = 40 * math.exp(-.25 ** 2)
    positive_x = x[x > 0]
    assert positive_x[np.argmax(pdf[x > 0])] == pytest.approx(mode, abs=.005)
    base_x = torch.linspace(-10, 10, 2_001, dtype=torch.float64)
    base_pdf = data.angle_density(base_x, "base")
    assert torch.trapezoid(torch.from_numpy(base_pdf), base_x).item() == pytest.approx(1.)
    np.testing.assert_array_equal(data.angle_density([-11, -10, 0, 10, 11], "base"),
                                  [0., .05, .05, .05, 0.])
    draws = data.sample_angles(100_000, "novel", 42)
    assert draws.mean().item() == pytest.approx(0., abs=.5)
    quantiles = torch.quantile(draws.abs(), torch.tensor([.025, .5, .975]))
    torch.testing.assert_close(quantiles, torch.tensor([24.5, 40., 65.1]), rtol=0, atol=.5)
    assert not (draws.abs() == 90).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_renderer_matches_existing_rotation_including_fill_and_sign(dtype):
    images = CachedSource._pixels(8).unsqueeze(1).to(dtype).div(255)
    angles = torch.tensor([0., 7., -7., 30., -45., 89.5, 90., -90.], dtype=torch.float64)
    actual = data.rotate_batch(images, angles)
    expected = torch.cat([rotate_mnist_batch(image[None], float(angle), old_data.ROTATION)
                          for image, angle in zip(images, angles, strict=True)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6 if dtype == torch.float32 else 1e-13)
    assert torch.equal(actual, data.rotate_batch(images, angles))
    assert actual.dtype == dtype and actual.shape == images.shape and actual.is_contiguous()
    point = torch.zeros((1, 1, 28, 28), dtype=dtype)
    point[0, 0, 13, 20] = 1
    positive = data.rotate_batch(point, torch.tensor([90.]))
    negative = data.rotate_batch(point, torch.tensor([-90.]))
    assert positive[0, 0, 7, 13].item() == pytest.approx(1., abs=2e-6)
    assert negative[0, 0, 20, 14].item() == pytest.approx(1., abs=2e-6)
    assert data.rotate_batch(images[:0], angles[:0]).shape == images[:0].shape


def test_render_chunking_does_not_change_values_and_stop_is_cooperative(monkeypatch):
    images = CachedSource._pixels(8).unsqueeze(1).float().div(255)
    angles = torch.linspace(-60, 60, 8)
    expected = data.rotate_batch(images, angles)
    monkeypatch.setattr(data, "ROTATION_CHUNK", 2)
    assert torch.equal(expected, data.rotate_batch(images, angles))
    calls = []

    def stop():
        calls.append(1)
        if len(calls) == 3:
            raise InterruptedError("test stop")

    with pytest.raises(InterruptedError, match="test stop"):
        data.rotate_batch(images, angles, check_stop=stop)
    assert len(calls) == 3


def test_plan_is_json_deterministic_disjoint_and_uses_named_seeds(source, cfg):
    state = torch.random.get_rng_state().clone()
    plan = data.plan_data(source, cfg, "arbitrary-family", 3, "development")
    assert plan == data.plan_data(source, cfg, "arbitrary-family", 3, "development")
    assert plan == json.loads(json.dumps(plan, allow_nan=False))
    assert torch.equal(state, torch.random.get_rng_state())
    roles = [plan[f"{role}_ids"] for role in ("initialization", "validation", "reference", "stream")]
    all_ids = [item for ids in roles for item in ids]
    assert len(all_ids) == len(set(all_ids))
    assert [len(ids) for ids in roles[:3]] == [cfg.init_size, cfg.init_validation_size, cfg.reference_size]
    assert plan["stream_labels"] == source.train_labels[plan["stream_ids"]].tolist()
    seeds = plan["seeds"]
    for name in ("initialization_model", "factors", "replay", "probes", "compression"):
        assert seeds[name] == cfg.seed_for("arbitrary-family", 3, name)
    flat_seeds = [seed for value in seeds.values() for seed in (value if isinstance(value, list) else [value])]
    assert len(flat_seeds) == len(set(flat_seeds))
    assert len(seeds["epochs"]) == cfg.init_epochs
    actual = torch.tensor(plan["class_probabilities"], dtype=torch.float64)
    expected = torch.bincount(source.train_labels).double() / len(source.train_labels)
    torch.testing.assert_close(actual, expected)


def test_partition_is_explicit_global_and_not_a_family_allowlist(source, cfg):
    development, confirmation = set(), set()
    for family in ("anything", "confirmation", "unrecognized-confirmation-v2"):
        for replica in (0, 1):
            dev = data.plan_data(source, cfg, family, replica, "development")
            confirm = data.plan_data(source, cfg, family, replica, "confirmation")
            assert dev["stream_ids"] == confirm["stream_ids"]
            assert dev["stream_angles"] == confirm["stream_angles"]
            assert dev["seeds"]["evaluation_partition"] == confirm["seeds"]["evaluation_partition"]
            for plan, pool in ((dev, development), (confirm, confirmation)):
                assert plan["evaluation_ids"]["base"] == plan["evaluation_ids"]["novel"]
                ids = plan["evaluation_ids"]["base"]
                assert len(set(ids)) == cfg.eval_per_class * 10
                assert torch.equal(torch.bincount(source.test_labels[ids]), torch.full((10,), cfg.eval_per_class))
                pool.update(ids)
    assert not development & confirmation
    with pytest.raises(ValueError, match="partition"):
        data.plan_data(source, cfg, "anything", 0, "typo")


@pytest.mark.parametrize("probability", [.5, 1.])
def test_segments_end_at_every_novel_event_and_stop_at_exact_budget(source, cfg, probability):
    cfg = dataclasses.replace(cfg, novel_probability=probability)
    plan = data.plan_data(source, cfg, "segmentation", 2, "development")
    flags = plan["stream_novel"]
    assert all(type(flag) is bool for flag in flags)
    assert sum(flags) == cfg.novel_count and flags[-1]
    assert not any(flags[:cfg.nominal_batch])
    previous_end, observed_novel = 0, 0
    for segment in plan["segments"]:
        start, end = segment["start"], segment["end"]
        assert start == previous_end and 1 <= end - start <= cfg.nominal_batch
        assert start // cfg.nominal_batch == (end - 1) // cfg.nominal_batch == segment["group"] - 1
        local_flags = flags[start:end]
        assert sum(local_flags) <= 1
        assert not any(local_flags[:-1])
        assert local_flags[-1] or end % cfg.nominal_batch == 0
        observed_novel += sum(local_flags)
        assert segment["novel_count"] == observed_novel
        expected_probability = probability * min((segment["group"] - 1) / (cfg.ramp_groups - 1), 1.)
        assert segment["probability"] == expected_probability
        previous_end = end
    assert previous_end == len(plan["stream_ids"])
    assert observed_novel == cfg.novel_count


def test_independent_components_keep_stream_fixed_when_other_seeds_change(source, cfg):
    baseline = data.plan_data(source, cfg, "pair", 0, "development")

    class SeedOverride:
        def __getattr__(self, name):
            return getattr(cfg, name)

        def seed_for(self, family, replica, component):
            seed = cfg.seed_for(family, replica, component)
            if component in ("initialization_model", "factors", "initialization_angles", "replay", "probes"):
                seed += 1
            return seed

    changed = data.plan_data(source, SeedOverride(), "pair", 0, "development")
    for key in ("stream_ids", "stream_labels", "stream_novel", "stream_angles", "segments",
                "evaluation_ids", "evaluation_angles"):
        assert baseline[key] == changed[key]
    assert baseline["initialization_angles"] != changed["initialization_angles"]


def test_source_and_angle_stream_rngs_do_not_advance_domain_or_digit_rngs(source, cfg):
    baseline = data.plan_data(source, cfg, "components", 0, "development")

    class SeedOverride:
        def __getattr__(self, name):
            return getattr(cfg, name)

        def seed_for(self, family, replica, component):
            suffix = ":changed" if component in ("stream_sources", "stream_angles_novel") else ""
            return cfg.seed_for(family, replica, component + suffix)

    changed = data.plan_data(source, SeedOverride(), "components", 0, "development")
    for key in ("stream_labels", "stream_novel", "segments"):
        assert baseline[key] == changed[key]
    assert baseline["stream_ids"] != changed["stream_ids"]
    for old, new, novel in zip(baseline["stream_angles"], changed["stream_angles"],
                               baseline["stream_novel"], strict=True):
        assert (old != new) if novel else (old == new)


def test_longer_budget_preserves_stream_prefix_and_novelty_is_not_angle_threshold(source, cfg, monkeypatch):
    small = data.plan_data(source, cfg, "prefix", 0, "development")
    large = data.plan_data(source, dataclasses.replace(cfg, novel_count=40), "prefix", 0, "development")
    for key in ("stream_ids", "stream_angles", "stream_labels", "stream_novel", "segments"):
        assert small[key] == large[key][:len(small[key])]
    sampler = data.sample_angles

    def zero_novel_angles(count, domain, seed):
        return torch.zeros(count) if domain == "novel" else sampler(count, domain, seed)

    monkeypatch.setattr(data, "sample_angles", zero_novel_angles)
    overlapping = data.plan_data(source, cfg, "prefix", 0, "development")
    assert overlapping["stream_novel"] == small["stream_novel"]
    assert sum(overlapping["stream_novel"]) == cfg.novel_count
    assert all(angle == 0. for angle, flag in zip(overlapping["stream_angles"], overlapping["stream_novel"], strict=True) if flag)


def test_materialization_is_paired_cpu_degrees_and_not_a_persistence_layer(source, cfg):
    plan = data.plan_data(source, cfg, "materialize", 4, "development")
    saved = json.loads(json.dumps(plan))
    first = data.materialize(source, cfg, plan)
    second = data.materialize(source, cfg, saved)
    assert first["plan"] == plan == saved
    assert set(first) == {"plan", "pretrain_x", "pretrain_y", "validation_x", "validation_y",
                          "reference_x", "stream_x", "stream_y", "panels"}
    for key in first:
        if isinstance(first[key], torch.Tensor):
            assert first[key].device.type == "cpu" and first[key].dtype == torch.float32
            assert torch.equal(first[key], second[key])
    assert torch.equal(first["stream_y"], torch.tensor(plan["stream_angles"]))
    assert first["stream_y"].abs().max() > 20  # Still degrees, not divided by 45.
    expected_pretrain = data.rotate_batch(source.images(plan["initialization_ids"]),
                                         torch.tensor(plan["initialization_angles"]))
    assert torch.equal(first["pretrain_x"], expected_pretrain)
    for domain, panel in first["panels"].items():
        assert panel["x"].shape == (cfg.eval_per_class * 10, 1, 28, 28)
        assert torch.equal(panel["y"], torch.tensor(plan["evaluation_angles"][domain]))
        assert panel["weights"].sum().item() == pytest.approx(1.)
        for digit in range(10):
            assert panel["weights"][panel["digits"] == digit].sum().item() == pytest.approx(plan["class_probabilities"][digit])
        assert all(torch.equal(value, second["panels"][domain][key]) for key, value in panel.items())


def test_materialize_stop_precedes_source_access_and_bad_schema_is_rejected(source, cfg, monkeypatch):
    plan = data.plan_data(source, cfg, "stop", 0, "development")

    def unexpected_images(*args, **kwargs):
        pytest.fail("stop should be checked before loading images")

    def stop():
        raise InterruptedError("halt")

    monkeypatch.setattr(source, "images", unexpected_images)
    with pytest.raises(InterruptedError, match="halt"):
        data.materialize(source, cfg, plan, check_stop=stop)
    with pytest.raises(ValueError, match="schema"):
        data.materialize(source, cfg, {**plan, "schema_version": -1})


def test_cached_source_is_reused_and_never_requests_a_download(monkeypatch, tmp_path):
    cached = CachedSource()
    calls = []

    def load(root, *, download):
        calls.append((root, download))
        return cached.train, cached.test

    monkeypatch.setattr(old_data, "load_mnist_datasets", load)
    assert data.MNISTSource is old_data.MNISTSource
    assert data.class_weights is old_data.class_weights
    loaded = data.MNISTSource(tmp_path)
    assert calls == [(tmp_path, False)]
    assert torch.equal(loaded.images([0, 1]), cached.images([0, 1]))


def test_insufficient_sources_and_invalid_inputs_fail_clearly(source, cfg):
    with pytest.raises(ValueError, match="insufficient disjoint"):
        data.plan_data(source, dataclasses.replace(cfg, init_size=100_000), "bad", 0, "development")
    with pytest.raises(ValueError, match="insufficient disjoint"):
        data.plan_data(source, dataclasses.replace(cfg, eval_per_class=13), "bad", 0, "confirmation")
    with pytest.raises(ValueError, match="exhausted a digit"):
        data.plan_data(source, dataclasses.replace(cfg, novel_count=2_000), "bad", 0, "development")
    with pytest.raises(ValueError, match="drip schedule"):
        data.plan_data(source, dataclasses.replace(cfg, novel_probability=0.), "bad", 0, "development")
    with pytest.raises(ValueError, match="domain"):
        data.sample_angles(10, "other", 0)
    with pytest.raises(ValueError, match="domain"):
        data.angle_density([0.], "other")
    with pytest.raises(ValueError, match="finite"):
        data.angle_density([float("nan")], "base")
    with pytest.raises(ValueError, match="nonnegative"):
        data.sample_angles(-1, "base", 0)
    images = torch.zeros(2, 1, 28, 28)
    with pytest.raises(ValueError, match="one finite degree"):
        data.rotate_batch(images, torch.tensor([0.]))
    images[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        data.rotate_batch(images, torch.zeros(2))
