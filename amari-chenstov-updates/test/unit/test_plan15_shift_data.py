"""CPU-only Phase 9 data/config contracts; no download, training, or GPU."""

from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from mnist_experiment.fisher_memory import data as source_data
from mnist_experiment.fisher_memory_regression import data as regression_data
from mnist_experiment.fisher_memory_regression.config import Config as RegressionConfig
from mnist_experiment.fisher_memory_shift import ALLOCATION_FAMILY, PROTOCOL, data
from mnist_experiment.fisher_memory_shift.config import Config, Context, default_contexts


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
        self.train = SimpleNamespace(data=self.pixels(len(self.train_labels)), targets=self.train_labels)
        self.test = SimpleNamespace(data=self.pixels(len(self.test_labels)), targets=self.test_labels)

    @staticmethod
    def pixels(count):
        pixels = torch.zeros(count, 28, 28, dtype=torch.uint8)
        pixels[:, 0, :] = 255
        pixels[:, 8:19, 16:21] = 180
        pixels[:, 12, 20] = torch.arange(count).remainder(256).to(torch.uint8)
        return pixels


@pytest.fixture
def source():
    return CachedSource()


@pytest.fixture
def cfg():
    return Config.smoke_config()


def test_config_identity_defaults_frozen_values_and_inherited_smoke(tmp_path):
    cfg = Config()
    cfg.validate()
    assert isinstance(cfg, RegressionConfig)
    assert PROTOCOL == cfg.protocol == "plan15-continuous-shift-v1"
    assert ALLOCATION_FAMILY == "plan15-phase9"
    assert cfg.seed != RegressionConfig().seed
    assert (cfg.init_epochs, cfg.max_init_epochs, cfg.init_epoch_stages) == (15, 45, (15, 30, 45))
    assert (cfg.calibration_replicas, cfg.finalist_extra_replicas, cfg.confirmation_replicas) == (2, 2, 12)
    assert cfg.capacities == (256,) and cfg.replay_counts == (4, 16)
    assert cfg.ridge_grid == (1., 10., 100.) and cfg.strength_ratios == (.1, 1., 10.)
    assert cfg.retention_margin_degrees == 1 and cfg.matching_tolerance == .2
    assert cfg.historical_strength == 2.6356432540763373
    assert cfg.refinement_settings == 8
    assert cfg.diagnostic_base_size == 512 and cfg.diagnostic_novel_size == 256
    for epochs in (15, 30, 45):
        replace(cfg, init_epochs=epochs).validate()
    for replicas in (8, 10, 12):
        replace(cfg, confirmation_replicas=replicas).validate()
    with pytest.raises(FrozenInstanceError):
        cfg.init_epochs = 30
    for candidate in (cfg, Config.smoke_config(), Config.smoke_config("cuda")):
        candidate.validate()
        resolved = candidate.mapping()
        assert Config.from_mapping(resolved) == candidate
        path = tmp_path / "config.json"
        path.write_text(json.dumps(resolved))
        assert Config.read(path) == candidate
        assert Config.read(path).scientific_hash == candidate.scientific_hash
    smoke = Config.smoke_config()
    assert smoke.smoke and smoke.device == "cpu" and smoke.dtype == "float64"
    assert smoke.max_init_epochs == 1 and smoke.init_epoch_stages == (1,)
    assert len(smoke.contexts) == 2
    assert smoke.max_iter < cfg.max_iter and smoke.init_size < cfg.init_size


@pytest.mark.parametrize("changes", [
    {"protocol": "old"}, {"schema_version": 0}, {"seed": 0}, {"threads": 0},
    {"max_init_epochs": 0}, {"init_epochs": 16}, {"init_epoch_stages": (15, 45, 30)},
    {"init_epoch_stages": (15, 30, 30, 45)}, {"init_epoch_stages": (15, 30)},
    {"diagnostic_base_size": 0}, {"diagnostic_novel_size": True},
    {"diagnostic_epochs": 11}, {"diagnostic_learning_rate": float("nan")},
    {"finalist_extra_replicas": -1}, {"refinement_settings": 9},
    {"matching_tolerance": 1.}, {"historical_strength": float("nan")},
    {"historical_strength": 0.}, {"confirmation_replicas": 11},
    {"ridge_grid": (.1, 1.)}, {"max_artifact_bytes": 500_000_000_001},
    {"contexts": ()}, {"contexts": (default_contexts()[0],) * 4},
])
def test_config_rejects_invalid_or_unversioned_changes(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes).validate()


def test_context_mapping_defaults_and_combination():
    contexts = default_contexts()
    assert [context.key for context in contexts] == ["gamma18", "gamma30", "blur06", "blur12"]
    assert contexts == data.default_contexts()
    assert Context is data.Context
    for context in (*contexts, Context("combo", "composite", 1.8, .6),
                    Context("gamma-blur-combined", "combined", 1.8, .6)):
        assert context == Context.from_mapping(json.loads(json.dumps(context.mapping())))
        with pytest.raises(FrozenInstanceError):
            context.gamma_max = 2.
    replace(Config(), contexts=(*contexts, Context("combo", "composite", 1.8, .6))).validate()
    replace(Config(), contexts=(*contexts, Context("combo", "combined", 1.8, .6))).validate()
    for mapping in (dict(key="", family="gamma", gamma_max=2.),
                    dict(key="x--y", family="gamma", gamma_max=2.),
                    dict(key="x", family="other"),
                    dict(key="x", family="gamma", gamma_max=float("inf")),
                    dict(key="x", family="gamma", gamma_max=2., blur_max=.1),
                    dict(key="x", family="blur", blur_max=-1.),
                    dict(key="x", family="composite", gamma_max=2.)):
        with pytest.raises(ValueError):
            Context.from_mapping(mapping)


def test_context_suffix_is_stripped_at_the_first_separator_for_every_seed(cfg):
    for component in ("initialization_model", "initialization_data", "stream_domains", "factors",
                      "probes", "replay", "epoch:0", "evaluation_strengths"):
        seed = cfg.seed_for("development", 0, component)
        assert seed == cfg.seed_for("development--gamma18", 0, component)
        assert seed == cfg.seed_for("development--gamma18--extra", 0, component)
        assert seed != cfg.seed_for("confirmation--gamma18", 0, component)
        assert seed != cfg.seed_for("development--gamma18", 1, component)
        assert seed != RegressionConfig().seed_for("development", 0, component)


@pytest.mark.parametrize("domain", data.DOMAINS)
def test_uniform_angle_law_has_continuous_support_prefixes_and_no_global_rng_mutation(domain):
    state = torch.random.get_rng_state().clone()
    angles = data.sample_angles(30_000, domain, 13)
    assert angles.dtype == torch.float32 and angles.device.type == "cpu"
    assert bool(((angles >= -60) & (angles <= 60)).all())
    assert float(angles.mean()) == pytest.approx(0., abs=.5)
    assert float(angles.double().square().mean()) == pytest.approx(1200., abs=30)
    assert torch.equal(angles[:73], data.sample_angles(73, domain, 13))
    assert torch.equal(angles, data.sample_angles(len(angles), domain, 13))
    assert not torch.equal(angles, data.sample_angles(len(angles), domain, 14))
    assert data.sample_angles(0, domain, 13).shape == (0,)
    assert torch.equal(state, torch.random.get_rng_state())


def test_drift_schedule_has_smooth_endpoints_and_uses_nominal_groups():
    assert data.drift_strength(1) == .5
    assert data.drift_strength(200) == 1
    assert data.drift_strength(10_000) == 1
    values = [data.drift_strength(group) for group in range(1, 201)]
    assert all(right > left for left, right in zip(values[:-1], values[1:], strict=True))
    assert values[1] - values[0] < .0001 and values[-1] - values[-2] < .0001
    assert data.drift_strength(101) == pytest.approx(.5 + .5 * (3 * (100 / 199) ** 2 - 2 * (100 / 199) ** 3))
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            data.drift_strength(invalid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_renderer_identity_continuity_order_and_no_quantization(dtype):
    raw = CachedSource.pixels(2).unsqueeze(1).to(dtype).div(255)
    rotated = data.rotate_batch(raw, torch.tensor([17., -23.]))
    saved = rotated.clone()
    contexts = (*default_contexts(), Context("combo", "composite", 3., 1.2))
    strengths = torch.tensor([.63, .81], dtype=torch.float64)
    assert data.transform(rotated, None, strengths) is rotated
    for context in contexts:
        assert torch.equal(data.transform(rotated, context, torch.zeros(2)), rotated)
        nominal_limit = data.transform(rotated, context, torch.full((2,), 1e-8, dtype=torch.float64))
        torch.testing.assert_close(nominal_limit, rotated, rtol=0, atol=2e-7)
        output = data.transform(rotated, context, strengths)
        assert output.shape == rotated.shape and output.dtype == dtype and output.is_contiguous()
        assert output.device.type == "cpu" and bool(torch.isfinite(output).all())
        assert float(output.min()) >= 0 and float(output.max()) <= 1 + 1e-6
        assert torch.equal(output, data.transform(rotated, context, strengths))
        nearby = data.transform(rotated, context, strengths + 1e-4)
        difference = float((nearby - output).abs().max())
        assert 0 < difference < .001
    gamma = Context("gamma", "gamma", 3.)
    blur = Context("blur", "blur", blur_max=1.2)
    combo = contexts[-1]
    sequential = data.transform(data.transform(rotated, gamma, strengths), blur, strengths)
    torch.testing.assert_close(data.transform(rotated, combo, strengths), sequential, rtol=0, atol=0)
    reverse = data.transform(data.transform(rotated, blur, strengths), gamma, strengths)
    assert not torch.allclose(sequential, reverse, rtol=0, atol=1e-4)
    transform_before_rotation = data.rotate_batch(data.transform(raw, combo, strengths), torch.tensor([17., -23.]))
    assert not torch.allclose(sequential, transform_before_rotation, rtol=0, atol=1e-4)
    assert torch.equal(rotated, saved)
    assert data.transform(rotated[:0], combo, torch.empty(0)).shape == rotated[:0].shape


def test_blur_matches_fixed_radius_normalized_separable_zero_padded_kernel():
    image = torch.zeros(1, 1, 28, 28, dtype=torch.float64)
    image[0, 0, 14, 14] = 1
    coordinates = torch.arange(-6, 7, dtype=torch.float64)
    sigma = .731
    kernel = torch.exp(-coordinates.square() / (2 * sigma ** 2))
    kernel /= kernel.sum()
    expected = F.conv2d(image, (kernel[:, None] * kernel[None, :]).reshape(1, 1, 13, 13), padding=6)
    actual = data.transform(image, Context("blur", "blur", blur_max=1.), torch.tensor([sigma], dtype=torch.float64))
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-16)
    assert float(actual.sum()) == pytest.approx(1.)
    assert float(actual[0, 0, 14, 20]) > 0 and float(actual[0, 0, 14, 21]) == 0
    image.zero_()
    image[0, 0, 0, 0] = 1
    assert float(data.transform(image, Context("blur", "blur", blur_max=1.), 1.).sum()) < 1
    mixed = image.repeat(2, 1, 1, 1)
    assert torch.equal(data.transform(mixed, Context("blur", "blur", blur_max=1.), [0., 1.])[0], image[0])


def test_renderer_chunk_invariance_and_cooperative_stop(monkeypatch):
    images = CachedSource.pixels(7).unsqueeze(1).float().div(255)
    strengths = torch.linspace(.5, 1, 7, dtype=torch.float64)
    context = Context("combo", "composite", 3., 1.2)
    expected = data.transform(images, context, strengths)
    monkeypatch.setattr(data, "TRANSFORM_CHUNK", 2)
    torch.testing.assert_close(data.transform(images, context, strengths), expected, rtol=0, atol=5e-8)
    calls = []

    def stop():
        calls.append(1)
        if len(calls) == 3:
            raise InterruptedError("halt")

    with pytest.raises(InterruptedError, match="halt"):
        data.transform(images, context, strengths, check_stop=stop)
    assert len(calls) == 3


def test_plan_is_deterministic_json_safe_and_all_source_roles_are_disjoint(source, cfg):
    state = torch.random.get_rng_state().clone()
    plan = data.plan_data(source, cfg, "development--gamma18", 3, "development")
    assert plan == data.plan_data(source, cfg, "development--blur12", 3, "development")
    assert plan == json.loads(json.dumps(plan, allow_nan=False))
    assert plan["protocol"] == PROTOCOL and plan["schema_version"] == data.SCHEMA_VERSION
    assert plan["family"] == "development"
    roles = ("initialization", "validation", "reference", "stream", "diagnostic_base", "diagnostic_novel")
    all_ids = [source_id for role in roles for source_id in plan[f"{role}_ids"]]
    assert len(all_ids) == len(set(all_ids))
    assert len(plan["diagnostic_base_ids"]) == cfg.diagnostic_base_size
    assert len(plan["diagnostic_novel_ids"]) == cfg.diagnostic_novel_size
    assert plan["stream_labels"] == source.train_labels[plan["stream_ids"]].tolist()
    for role in roles:
        angles = torch.tensor(plan[f"{role}_angles"])
        assert len(angles) == len(plan[f"{role}_ids"])
        assert bool(((angles >= -60) & (angles <= 60)).all())
    assert len(plan["seeds"]["epochs"]) == cfg.max_init_epochs
    assert len(plan["seeds"]["diagnostic_epochs"]) == cfg.max_diagnostic_epochs
    flat_seeds = [item for value in plan["seeds"].values() for item in (value if isinstance(value, list) else [value])]
    assert len(flat_seeds) == len(set(flat_seeds))
    assert torch.equal(state, torch.random.get_rng_state())


def test_initialization_stage_and_treatment_config_cannot_resample_plan(source, cfg):
    cfg = replace(cfg, max_init_epochs=3, init_epoch_stages=(1, 2, 3))
    first = data.plan_data(source, cfg, "development--gamma18", 0, "development")
    later = replace(cfg, init_epochs=3, reference_capacity=16, reference_replay=4,
                    ridge_grid=(10.,), strength_ratios=(1.,))
    assert first == data.plan_data(source, later, "development--blur06", 0, "development")
    assert first["seeds"]["epochs"] == [cfg.seed_for("development", 0, f"epoch:{i}") for i in range(3)]


def test_old_source_drip_split_groups_and_exact_novel_event_checkpoints_are_preserved(source, cfg):
    cfg = replace(cfg, novel_count=12, novel_probability=1.)
    plan = data.plan_data(source, cfg, "development", 0, "development")
    old = regression_data.plan_data(source, cfg, "development", 0, "development")
    for key in ("initialization_ids", "validation_ids", "reference_ids", "stream_ids", "stream_labels", "stream_novel"):
        assert plan[key] == old[key]
    assert [{key: value for key, value in segment.items() if key != "strength"}
            for segment in plan["segments"]] == old["segments"]
    assert sum(plan["stream_novel"]) == cfg.novel_count and plan["stream_novel"][-1]
    previous_end, novel_count = 0, 0
    for segment in plan["segments"]:
        start, end, group = segment["start"], segment["end"], segment["group"]
        assert start == previous_end and 1 <= end - start <= cfg.nominal_batch
        assert start // 4 == (end - 1) // 4 == group - 1
        assert segment["probability"] == cfg.novel_probability * min((group - 1) / (cfg.ramp_groups - 1), 1)
        assert segment["strength"] == data.drift_strength(group)
        for index in range(start, end):
            novel = plan["stream_novel"][index]
            assert plan["stream_groups"][index] == group
            assert plan["stream_strengths"][index] == (data.drift_strength(group) if novel else 0.)
            novel_count += int(novel)
        assert segment["novel_count"] == novel_count
        assert sum(plan["stream_novel"][start:end]) <= 1
        previous_end = end
    assert previous_end == len(plan["stream_ids"])
    assert [segment["novel_count"] for segment in plan["segments"]
            if plan["stream_novel"][segment["end"] - 1]] == list(range(1, cfg.novel_count + 1))
    assert len({segment["group"] for segment in plan["segments"]}) < len(plan["segments"])


def test_fixed_test_pools_pair_angles_and_strengths_without_confirmation_leakage(source, cfg):
    development, confirmation = set(), set()
    for family in ("development--gamma18", "arbitrary--blur12"):
        for replica in (0, 1):
            for partition, pool in (("development", development), ("confirmation", confirmation)):
                plan = data.plan_data(source, cfg, family, replica, partition)
                assert plan["evaluation_ids"]["base"] == plan["evaluation_ids"]["novel"]
                assert plan["evaluation_angles"]["base"] == plan["evaluation_angles"]["novel"]
                ids = plan["evaluation_ids"]["base"]
                assert torch.equal(torch.bincount(source.test_labels[ids]), torch.full((10,), cfg.eval_per_class))
                assert plan["evaluation_strengths"]["base"] == [0.] * len(ids)
                assert all(.5 <= strength <= 1 for strength in plan["evaluation_strengths"]["novel"])
                assert all(.5 <= strength <= 1 for strength in plan["diagnostic_novel_strengths"])
                assert plan["seeds"]["evaluation_strengths"] != plan["seeds"]["diagnostic_novel_strengths"]
                pool.update(ids)
    assert not development & confirmation


def test_nominal_materialization_and_context_cache_sharing_leave_targets_unchanged(source, cfg, monkeypatch):
    state = torch.random.get_rng_state().clone()
    plan = data.plan_data(source, cfg, "development--gamma18", 0, "development")
    saved = json.loads(json.dumps(plan))
    base = data.materialize(source, cfg, plan)
    contextual = data.materialize(source, cfg, saved, default_contexts()[1])
    for domain in data.DOMAINS:
        panel = base["panels"][domain]
        assert panel["weights"].sum().item() == pytest.approx(1.)
        for digit in range(10):
            assert panel["weights"][panel["digits"] == digit].sum().item() == pytest.approx(plan["class_probabilities"][digit])
    assert torch.equal(base["panels"]["base"]["x"], base["panels"]["novel"]["x"])
    expected = data.rotate_batch(source.images(plan["stream_ids"]), torch.tensor(plan["stream_angles"]))
    assert torch.equal(base["stream_x"], expected)
    assert torch.equal(base["stream_y"], torch.tensor(plan["stream_angles"]))
    for domain in data.DOMAINS:
        key = f"diagnostic_{domain}"
        assert torch.equal(base[f"{key}_y"], torch.tensor(plan[f"{key}_angles"]))
        assert base[f"{key}_x"].shape == (len(plan[f"{key}_ids"]), 1, 28, 28)

    def no_source_access(*args, **kwargs):
        pytest.fail("contextualize must reuse cached nominal inputs")

    monkeypatch.setattr(source, "images", no_source_access)
    for context in (*default_contexts(), Context("combo", "composite", 1.8, .6)):
        rendered = data.contextualize(base, cfg, context)
        for key in ("plan", "pretrain_x", "pretrain_y", "reference_x", "validation_x", "validation_y",
                    "stream_y", "diagnostic_base_x", "diagnostic_base_y", "diagnostic_novel_y"):
            assert rendered[key] is base[key]
        assert rendered["panels"]["base"] is base["panels"]["base"]
        for key in ("y", "digits", "weights"):
            assert rendered["panels"]["novel"][key] is base["panels"]["novel"][key]
        flags = torch.tensor(plan["stream_novel"])
        assert torch.equal(rendered["stream_x"][~flags], base["stream_x"][~flags])
        assert torch.equal(rendered["stream_x"][flags], data.transform(
            base["stream_x"][flags], context, torch.tensor(plan["stream_strengths"], dtype=torch.float64)[flags]))
        assert torch.equal(rendered["diagnostic_novel_x"], data.transform(base["diagnostic_novel_x"],
            context, plan["diagnostic_novel_strengths"]))
        if context == default_contexts()[1]:
            assert torch.equal(rendered["stream_x"], contextual["stream_x"])
        with pytest.raises(ValueError, match="nominal cache"):
            data.contextualize(rendered, cfg, context)
    assert torch.equal(base["stream_x"], expected)
    assert plan == saved and base["context"] is None
    assert torch.equal(state, torch.random.get_rng_state())


def test_cached_mnist_helpers_are_reused_without_downloading(monkeypatch, tmp_path):
    cached = CachedSource()
    calls = []

    def load(root, *, download):
        calls.append((root, download))
        return cached.train, cached.test

    monkeypatch.setattr(source_data, "load_mnist_datasets", load)
    assert data.MNISTSource is source_data.MNISTSource
    assert data.class_weights is source_data.class_weights
    assert data.rotate_batch is regression_data.rotate_batch
    assert torch.equal(data.MNISTSource(tmp_path).images([0]), cached.images([0]))
    assert calls == [(tmp_path, False)]


def test_stopping_precedes_source_access_and_bad_plan_schema_fails_clearly(source, cfg, monkeypatch):
    plan = data.plan_data(source, cfg, "development", 0, "development")
    base = data.materialize(source, cfg, plan)

    def stop():
        raise InterruptedError("halt")

    def no_source_access(*args, **kwargs):
        pytest.fail("stop must precede source access")

    monkeypatch.setattr(source, "images", no_source_access)
    with pytest.raises(InterruptedError, match="halt"):
        data.materialize(source, cfg, plan, check_stop=stop)
    with pytest.raises(InterruptedError, match="halt"):
        data.contextualize(base, cfg, default_contexts()[0], check_stop=stop)
    for invalid in ({**plan, "schema_version": -1}, {**plan, "protocol": "old"}):
        with pytest.raises(ValueError, match="schema"):
            data.materialize(source, cfg, invalid)
    with pytest.raises(ValueError, match="partition"):
        data.materialize(source, cfg, {**plan, "partition": "typo"})


def test_invalid_sampler_rendering_and_insufficient_disjoint_sources_fail(source, cfg):
    for count, domain in ((-1, "base"), (1, "typo")):
        with pytest.raises(ValueError):
            data.sample_angles(count, domain, 0)
    with pytest.raises(ValueError, match="partition"):
        data.plan_data(source, cfg, "development", 0, "typo")
    with pytest.raises(ValueError, match="insufficient disjoint"):
        data.plan_data(source, replace(cfg, diagnostic_base_size=10_000), "development", 0, "development")
    with pytest.raises(ValueError, match="insufficient disjoint"):
        data.plan_data(source, replace(cfg, eval_per_class=13), "development", 0, "development")
    context = default_contexts()[0]
    images = torch.zeros(2, 1, 28, 28)
    for strengths in ([.5], [-1., 1.], [float("nan"), .5], [1.1, .5]):
        with pytest.raises(ValueError, match="strengths"):
            data.transform(images, context, strengths)
    for invalid in (torch.zeros(2, 28, 28), images.to(torch.uint8), images - 1,
                    images + float("nan"), images + 2):
        with pytest.raises(ValueError):
            data.transform(invalid, context, [.5, .5])
