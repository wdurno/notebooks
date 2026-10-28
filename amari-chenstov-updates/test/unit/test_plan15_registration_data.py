"""Small CPU-only registration data contracts; no MNIST download or training."""

from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from mnist_experiment.fisher_memory import data as source_data
from mnist_experiment.fisher_memory_regression import data as regression_data
from mnist_experiment.fisher_memory_registration import PROTOCOL, data
from mnist_experiment.fisher_memory_registration.config import Config


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class CachedSource(data.MNISTSource):
    def __init__(self):
        self.train_labels = torch.repeat_interleave(torch.arange(10), torch.arange(120, 220, 10))
        self.test_labels = torch.arange(10).repeat_interleave(24)
        self.train = SimpleNamespace(data=self.pixels(torch.arange(len(self.train_labels))), targets=self.train_labels)
        self.test = SimpleNamespace(data=self.pixels(torch.arange(len(self.test_labels))), targets=self.test_labels)

    @staticmethod
    def pixels(ids):
        pixels = torch.zeros(len(ids), 28, 28, dtype=torch.uint8)
        pixels[:, 4:23, 11:15] = 190
        pixels[:, 8:12, 14:23] = 255
        pixels[:, 18:23, 8:17] = 160
        pixels[:, 12:15, 16:19] = (torch.as_tensor(ids) % 256).to(torch.uint8)[:, None, None]
        return pixels


@pytest.fixture
def source():
    return CachedSource()


@pytest.fixture
def cfg():
    return Config.smoke_config()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_affine_parameter_order_and_clockwise_rotation(dtype):
    parameters = torch.tensor([[0., 2., -1., 0., 0., 0.],
                               [90., 0., 0., 0., 0., 0.],
                               [17., 1., 2., math.log(1.1), -math.log(1.1), .07]], dtype=dtype)
    matrix, translation = data.affine_from_parameters(parameters)
    torch.testing.assert_close(matrix[0], torch.eye(2, dtype=dtype))
    torch.testing.assert_close(translation[0], torch.tensor([2., -1.], dtype=dtype))
    torch.testing.assert_close(matrix[1] @ torch.tensor([1., 0.], dtype=dtype),
                               torch.tensor([0., 1.], dtype=dtype), atol=1e-7, rtol=0)
    angle = math.radians(17.)
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                              [math.sin(angle), math.cos(angle)]], dtype=dtype)
    expected = rotation @ torch.diag(torch.tensor([1.1, 1 / 1.1], dtype=dtype))
    expected = expected @ torch.tensor([[1., .07], [0., 1.]], dtype=dtype)
    torch.testing.assert_close(matrix[2], expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_landmarks_encode_decode_identity_translation_and_unconstrained_predictions(dtype):
    identity = torch.eye(2, dtype=dtype)
    zero = torch.zeros(2, dtype=dtype)
    assert torch.equal(data.encode_affine(identity, zero), torch.zeros(6, dtype=dtype))
    translation = torch.tensor([2., -1.], dtype=dtype)
    assert torch.equal(data.encode_affine(identity, translation), translation.repeat(3))
    parameters, _ = data.sample_transforms(20, 34)
    matrix, translation = data.affine_from_parameters(parameters.to(dtype))
    y = data.encode_affine(matrix, translation)
    anchors = torch.tensor(data.ANCHORS, dtype=dtype)
    expected = (anchors @ matrix.transpose(1, 2) + translation[:, None] - anchors).reshape(-1, 6)
    torch.testing.assert_close(y, expected)
    restored_matrix, restored_translation = data.decode_affine(y)
    torch.testing.assert_close(restored_matrix, matrix)
    torch.testing.assert_close(restored_translation, translation)
    # Decoding must not repair singular, orientation-reversing, or extreme maps.
    arbitrary = torch.tensor([[4., -100., 8., 32., 0., 11.],
                              [6., 4., -6., 4., 0., -8.]], dtype=dtype)
    torch.testing.assert_close(data.encode_affine(*data.decode_affine(arbitrary)), arbitrary)
    singular, _ = data.decode_affine(arbitrary[1])
    assert torch.equal(singular, torch.zeros_like(singular))


def test_support_rejection_includes_interpolation_footprint_and_is_geometric():
    matrix = torch.eye(2, dtype=torch.float64).repeat(4, 1, 1)
    translation = torch.tensor([[3.5, 0.], [3.500001, 0.], [4., 0.], [0., -3.500001]], dtype=torch.float64)
    assert data.support_mask(matrix, translation).tolist() == [True, False, False, False]
    # Pixel-center-only corners at +/-8.5 would wrongly accept this translation.
    assert bool((torch.tensor([8.5, 8.5]) + translation[2]).abs().le(13).all())
    parameters = torch.tensor([[20., 2., 2., math.log(1.1), math.log(1.1), .1]])
    assert not bool(data.support_mask(*data.affine_from_parameters(parameters))[0])


def test_sampler_freezes_prefixes_rejections_covariance_and_rng_state():
    state = torch.random.get_rng_state().clone()
    parameters, diagnostics = data.sample_transforms(2048, 13)
    shorter, small_diagnostics = data.sample_transforms(17, 13)
    assert torch.equal(shorter, parameters[:17])
    assert torch.equal(parameters, data.sample_transforms(2048, 13)[0])
    assert not torch.equal(shorter, data.sample_transforms(17, 14)[0])
    assert diagnostics["accepted"] == 2048 and diagnostics["rejected"] > 0
    assert diagnostics["proposed"] == diagnostics["accepted"] + diagnostics["rejected"]
    assert small_diagnostics["rejected"] <= diagnostics["rejected"]
    limits = torch.tensor([20., 2., 2., math.log(1.1), math.log(1.1), .1])
    assert bool((parameters.abs() <= limits).all())
    generator = torch.Generator().manual_seed(13)
    candidates = (2 * torch.rand(1024, 6, generator=generator, dtype=torch.float64) - 1)
    candidates *= torch.tensor([20., 2., 2., math.log(1.1), math.log(1.1), .1], dtype=torch.float64)
    accepted_indices = torch.where(data.support_mask(*data.affine_from_parameters(candidates)))[0]
    assert small_diagnostics["rejected"] == int(accepted_indices[16]) + 1 - 17
    assert torch.equal(shorter, candidates[accepted_indices[:17]])
    matrix, translation = data.affine_from_parameters(parameters)
    assert bool(data.support_mask(matrix, translation).all())
    targets = data.encode_affine(matrix, translation)
    assert int(torch.linalg.matrix_rank(targets - targets.mean(0))) == 6
    torch.testing.assert_close(torch.tensor(diagnostics["target_covariance"], dtype=torch.float64),
                               torch.cov(targets.T))
    torch.testing.assert_close(torch.tensor(diagnostics["target_mean"], dtype=torch.float64), targets.mean(0))
    assert diagnostics == json.loads(json.dumps(diagnostics, allow_nan=False))
    assert data.sample_transforms(0, 0)[0].shape == (0, 6)
    assert torch.equal(state, torch.random.get_rng_state())


def test_schedules_start_nominal_and_are_smooth_on_original_groups():
    assert data.mixture_probability(1) == 0.
    assert data.mixture_probability(40) == data.mixture_probability(10000) == .1
    assert data.mixture_probability(20) == pytest.approx(.1 * (3 * (19 / 39) ** 2 - 2 * (19 / 39) ** 3))
    probabilities = [data.mixture_probability(group) for group in range(1, 41)]
    strengths = [data.drift_strength(group) for group in range(1, 201)]
    assert all(b > a for a, b in zip(probabilities[:-1], probabilities[1:], strict=True))
    assert all(b > a for a, b in zip(strengths[:-1], strengths[1:], strict=True))
    assert probabilities[1] < .0002 and .1 - probabilities[-2] < .0002
    assert strengths[0] == .5 and strengths[-1] == data.drift_strength(10000) == 1.
    assert strengths[1] - .5 < .0001 and 1 - strengths[-2] < .0001
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            data.drift_strength(invalid)
        with pytest.raises(ValueError):
            data.mixture_probability(invalid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pair_renderer_antialiased_centering_forward_translation_and_rotation(dtype):
    images = torch.zeros(5, 1, 28, 28, dtype=dtype)
    images[:, :, 10:13, 18:21] = 1
    parameters = torch.tensor([[0., 0., 0., 0., 0., 0.],
                               [0., 2., -1., 0., 0., 0.],
                               [90., 0., 0., 0., 0., 0.],
                               [0., .5, 0., 0., 0., 0.],
                               [0., 30., 0., 0., 0., 0.]], dtype=dtype)
    saved = images.clone()
    pairs = data.render_pairs(images, *data.affine_from_parameters(parameters))
    reference = F.pad(F.interpolate(images, (18, 18), mode="bilinear", align_corners=False,
                                    antialias=True), (5, 5, 5, 5))
    assert pairs.shape == (5, 2, 28, 28) and pairs.is_contiguous()
    assert pairs.dtype == dtype and pairs.device.type == "cpu"
    assert torch.equal(pairs[:, :1], reference)
    tolerance = 1e-6 if dtype == torch.float32 else 2e-15
    torch.testing.assert_close(pairs[0, 1], reference[0, 0], atol=tolerance, rtol=0)
    shifted = F.pad(reference[1:2, :, 1:, :-2], (2, 0, 0, 1))
    torch.testing.assert_close(pairs[1:2, 1:2], shifted, atol=tolerance, rtol=0)
    clockwise = torch.rot90(reference[2, 0], -1, (-2, -1))
    torch.testing.assert_close(pairs[2, 1], clockwise, atol=tolerance, rtol=0)
    half_pixel = .5 * (reference[3:4] + F.pad(reference[3:4, :, :, :-1], (1, 0)))
    torch.testing.assert_close(pairs[3:4, 1:2], half_pixel, atol=tolerance, rtol=0)
    assert torch.equal(pairs[4, 1], torch.zeros(28, 28, dtype=dtype))
    assert torch.equal(images, saved)
    assert data.render_pairs(images[:0], torch.empty(0, 2, 2), torch.empty(0, 2)).shape == (0, 2, 28, 28)


def test_blur_query_only_matches_fixed_radius_kernel_and_zero_padding():
    pairs = torch.zeros(2, 2, 28, 28, dtype=torch.float64)
    pairs[:, 0, 3:12, 2:9] = .8
    pairs[:, 1, 14, 14] = 1
    saved = pairs.clone()
    sigma = .731
    actual = data.blur_pairs(pairs, [0., sigma], 1.)
    coordinates = torch.arange(-6, 7, dtype=torch.float64)
    kernel = torch.exp(-coordinates.square() / (2 * sigma ** 2))
    kernel /= kernel.sum()
    expected = F.conv2d(pairs[1:2, 1:2], (kernel[:, None] * kernel[None, :]).reshape(1, 1, 13, 13), padding=6)
    torch.testing.assert_close(actual[1:2, 1:2], expected, rtol=1e-14, atol=1e-16)
    assert torch.equal(actual[:, 0], pairs[:, 0]) and torch.equal(actual[0], pairs[0])
    assert float(actual[1, 1].sum()) == pytest.approx(1.)
    assert float(actual[1, 1, 14, 20]) > 0 and float(actual[1, 1, 14, 21]) == 0.
    assert data.blur_pairs(pairs, [1., 1.], 0.) is pairs
    assert data.blur_pairs(pairs, 0., 1.8) is pairs
    empty = pairs[:0]
    assert data.blur_pairs(empty, [], 1.8) is empty
    assert torch.equal(pairs, saved)
    edge = pairs[1:].clone()
    edge[:, 1] = 0
    edge[:, 1, 0, 0] = 1
    assert float(data.blur_pairs(edge, 1., 1.)[:, 1].sum()) < 1


def test_blur_rendering_convergence_and_strength_continuity(source):
    raw = source.images([1, 2])
    parameters, _ = data.sample_transforms(2, 37)
    nominal = data.render_pairs(raw, *data.affine_from_parameters(parameters))
    limit = data.blur_pairs(nominal, 1e-8, 1.8)
    assert torch.equal(limit, nominal)
    output = data.blur_pairs(nominal, [.63, .81], 1.8)
    nearby = data.blur_pairs(nominal, [.6301, .8101], 1.8)
    difference = float((nearby - output).abs().max())
    assert 0 < difference < .001
    assert torch.equal(output[:, 0], nominal[:, 0])


def test_plan_json_pairing_seeds_roles_and_no_angle_planner(source, cfg, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("registration planning cannot call the old angle planner or access images")

    monkeypatch.setattr(regression_data, "plan_data", forbidden)
    monkeypatch.setattr(source, "images", forbidden)
    state = torch.random.get_rng_state().clone()
    plan = data.plan_data(source, cfg, "development--scalar", 3, "development")
    assert plan == data.plan_data(source, cfg, "development--vector--other", 3, "development")
    assert plan == json.loads(json.dumps(plan, allow_nan=False))
    assert plan["protocol"] == PROTOCOL and plan["schema_version"] == data.SCHEMA_VERSION
    assert plan["family"] == "development"
    all_ids = [source_id for role in data.TRAIN_ROLES for source_id in plan[f"{role}_ids"]]
    assert len(all_ids) == len(set(all_ids))
    assert len(plan["reference_ids"]) == cfg.reference_max_size
    assert len(plan["initialization_ids"]) == cfg.init_size
    assert len(plan["validation_ids"]) == cfg.init_validation_size
    assert len(plan["diagnostic_base_ids"]) == cfg.diagnostic_base_size
    assert len(plan["diagnostic_novel_ids"]) == cfg.diagnostic_novel_size
    assert plan["stream_labels"] == source.train_labels[plan["stream_ids"]].tolist()
    assert len(plan["seeds"]["epochs"]) == cfg.max_init_epochs
    assert len(plan["seeds"]["diagnostic_epochs"]) == cfg.max_diagnostic_epochs
    flat_seeds = [item for value in plan["seeds"].values() for item in (value if isinstance(value, list) else [value])]
    assert len(flat_seeds) == len(set(flat_seeds))
    for role in data.TRAIN_ROLES:
        parameters = torch.tensor(plan[f"{role}_transforms"], dtype=torch.float64)
        assert len(parameters) == len(plan[f"{role}_ids"])
        assert bool(data.support_mask(*data.affine_from_parameters(parameters)).all())
        assert plan["transform_diagnostics"][role]["accepted"] == len(parameters)
    assert plan["seeds"]["initialization_model"] == cfg.seed_for("development", 3, "initialization_model")
    assert torch.equal(state, torch.random.get_rng_state())


def test_data_only_config_and_later_epoch_reference_choices_do_not_resample(source, cfg):
    cfg = replace(cfg, max_init_epochs=3, init_epoch_stages=(1, 2, 3), max_diagnostic_epochs=3)
    first = data.plan_data(source, cfg, "development--scalar", 0, "development")
    later = replace(cfg, init_epochs=3, reference_size=cfg.reference_max_size,
                    diagnostic_epochs=3, channels=(6, 8), blur_max=2.4)
    assert first == data.plan_data(source, later, "development--vector", 0, "development")
    fields = ("init_size", "init_validation_size", "reference_size", "reference_max_size",
              "max_init_epochs", "max_diagnostic_epochs", "eval_per_class", "novel_count",
              "nominal_batch", "ramp_groups", "diagnostic_base_size", "diagnostic_novel_size",
              "novel_probability", "seed_for")
    data_only_cfg = SimpleNamespace(**{name: getattr(cfg, name) for name in fields})
    assert first == data.plan_data(source, data_only_cfg, "development--scalar", 0, "development")
    assert first["seeds"]["epochs"] == [cfg.seed_for("development", 0, f"epoch:{i}") for i in range(3)]
    assert first["seeds"]["diagnostic_epochs"] == [cfg.seed_for("development", 0, f"diagnostic_order:{i}") for i in range(3)]


def test_exact_novel_segmentation_preserves_original_groups_and_both_schedules(source, cfg):
    cfg = replace(cfg, novel_count=12, novel_probability=1.)
    plan = data.plan_data(source, cfg, "development", 0, "development")
    assert sum(plan["stream_novel"]) == cfg.novel_count and plan["stream_novel"][-1]
    assert not any(plan["stream_novel"][:cfg.nominal_batch])
    assert plan["segments"][0]["probability"] == 0.
    previous_end, novel_count = 0, 0
    for segment in plan["segments"]:
        start, end, group = segment["start"], segment["end"], segment["group"]
        assert start == previous_end and 1 <= end - start <= cfg.nominal_batch
        assert start // 4 == (end - 1) // 4 == group - 1
        assert segment["probability"] == data.mixture_probability(group, cfg.novel_probability, cfg.ramp_groups)
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


def test_fixed_panels_pair_sources_transforms_and_fresh_partition_pools(source, cfg):
    development, confirmation = set(), set()
    for family in ("development--scalar", "arbitrary--vector"):
        for replica in (0, 1):
            for partition, pool in (("development", development), ("confirmation", confirmation)):
                plan = data.plan_data(source, cfg, family, replica, partition)
                assert plan["evaluation_ids"]["base"] == plan["evaluation_ids"]["novel"]
                assert plan["evaluation_transforms"]["base"] == plan["evaluation_transforms"]["novel"]
                ids = plan["evaluation_ids"]["base"]
                assert torch.equal(torch.bincount(source.test_labels[ids]), torch.full((10,), cfg.eval_per_class))
                assert plan["evaluation_strengths"]["base"] == [0.] * len(ids)
                for key in ("diagnostic_novel_strengths", "diagnostic_validation_strengths"):
                    assert all(.5 <= strength <= 1. for strength in plan[key])
                assert len(plan["diagnostic_validation_strengths"]) == cfg.init_validation_size
                assert all(.5 <= strength <= 1. for strength in plan["evaluation_strengths"]["novel"])
                pool.update(ids)
    assert not development & confirmation


def test_materialize_pair_targets_reference_reserve_and_context_sharing(source, cfg, monkeypatch):
    state = torch.random.get_rng_state().clone()
    plan = data.plan_data(source, cfg, "development--scalar", 0, "development")
    saved = json.loads(json.dumps(plan))
    base = data.materialize(source, cfg, plan)
    assert len(base["reference_x"]) == cfg.reference_max_size > cfg.reference_size
    extended = data.materialize(source, replace(cfg, reference_size=cfg.reference_max_size), plan)
    assert torch.equal(base["reference_x"], extended["reference_x"])
    for role, output in (("initialization", "pretrain"), ("validation", "validation"),
                         ("stream", "stream"), ("diagnostic_base", "diagnostic_base"),
                         ("diagnostic_novel", "diagnostic_novel")):
        parameters = torch.tensor(plan[f"{role}_transforms"], dtype=torch.float64)
        matrix, translation = data.affine_from_parameters(parameters)
        expected = data.render_pairs(source.images(plan[f"{role}_ids"]), matrix, translation)
        assert torch.equal(base[f"{output}_x"], expected)
        assert torch.equal(base[f"{output}_y"], data.encode_affine(matrix, translation).float())
        assert base[f"{output}_y"].shape == (len(parameters), 6)
        assert base[f"{output}_y"].dtype == base[f"{output}_x"].dtype == torch.float32
    assert base["panels"]["base"]["x"] is base["panels"]["novel"]["x"]
    assert base["panels"]["base"]["y"] is base["panels"]["novel"]["y"]
    for panel in base["panels"].values():
        assert panel["weights"].sum().item() == pytest.approx(1.)
        for digit in range(10):
            assert panel["weights"][panel["digits"] == digit].sum().item() == pytest.approx(plan["class_probabilities"][digit])
    contextual = data.materialize(source, cfg, saved, 1.8)

    def forbidden(*args, **kwargs):
        pytest.fail("contextualize must reuse nominal inputs without source access")

    monkeypatch.setattr(source, "images", forbidden)
    for blur_max in (0., 1.2, 1.8, 2.4):
        rendered = data.contextualize(base, cfg, blur_max)
        for key in ("plan", "pretrain_x", "pretrain_y", "reference_x", "validation_x", "validation_y",
                    "stream_y", "diagnostic_base_x", "diagnostic_base_y", "diagnostic_novel_y"):
            assert rendered[key] is base[key]
        assert rendered["panels"]["base"] is base["panels"]["base"]
        for key in ("y", "digits", "weights"):
            assert rendered["panels"]["novel"][key] is base["panels"]["novel"][key]
        flags = torch.tensor(plan["stream_novel"], dtype=torch.bool)
        assert torch.equal(rendered["stream_x"][~flags], base["stream_x"][~flags])
        assert torch.equal(rendered["stream_x"][:, 0], base["stream_x"][:, 0])
        expected = data.blur_pairs(base["stream_x"][flags],
            torch.tensor(plan["stream_strengths"], dtype=torch.float64)[flags], blur_max)
        assert torch.equal(rendered["stream_x"][flags], expected)
        assert torch.equal(rendered["diagnostic_novel_x"], data.blur_pairs(base["diagnostic_novel_x"],
            plan["diagnostic_novel_strengths"], blur_max))
        if blur_max == 1.8:
            assert torch.equal(rendered["stream_x"], contextual["stream_x"])
        if blur_max:
            with pytest.raises(ValueError, match="nominal cache"):
                data.contextualize(rendered, cfg, 1.8)
    assert plan == saved and base["blur_max"] == 0.
    assert torch.equal(state, torch.random.get_rng_state())


def test_rendering_chunk_invariance_and_cooperative_stop(source, monkeypatch):
    raw = source.images(list(range(7)))
    parameters, _ = data.sample_transforms(7, 33)
    matrix, translation = data.affine_from_parameters(parameters)
    expected = data.render_pairs(raw, matrix, translation)
    blurred = data.blur_pairs(expected, .7, 1.8)
    monkeypatch.setattr(data, "TRANSFORM_CHUNK", 2)
    torch.testing.assert_close(data.render_pairs(raw, matrix, translation), expected, rtol=0, atol=0)
    torch.testing.assert_close(data.blur_pairs(expected, .7, 1.8), blurred, rtol=0, atol=1e-7)
    calls = []

    def stop():
        calls.append(1)
        if len(calls) == 3:
            raise InterruptedError("halt")

    with pytest.raises(InterruptedError, match="halt"):
        data.render_pairs(raw, matrix, translation, check_stop=stop)
    calls.clear()
    with pytest.raises(InterruptedError, match="halt"):
        data.blur_pairs(expected, .7, 1.8, check_stop=stop)


def test_cached_source_sampler_helpers_are_reused_without_downloading(source, tmp_path, monkeypatch):
    calls = []

    def load(root, *, download):
        calls.append((root, download))
        return source.train, source.test

    monkeypatch.setattr(source_data, "load_mnist_datasets", load)
    assert data.MNISTSource is source_data.MNISTSource
    assert data.class_weights is source_data.class_weights
    assert data._choose is source_data._choose
    assert torch.equal(data.MNISTSource(tmp_path).images([0]), source.images([0]))
    assert calls == [(tmp_path, False)]


def test_stopping_precedes_source_access_and_wrong_schema_fails(source, cfg, monkeypatch):
    plan = data.plan_data(source, cfg, "development", 0, "development")
    base = data.materialize(source, cfg, plan)

    def stop():
        raise InterruptedError("halt")

    def forbidden(*args, **kwargs):
        pytest.fail("stop must precede source access")

    monkeypatch.setattr(source, "images", forbidden)
    with pytest.raises(InterruptedError, match="halt"):
        data.materialize(source, cfg, plan, check_stop=stop)
    with pytest.raises(InterruptedError, match="halt"):
        data.contextualize(base, cfg, 1.8, check_stop=stop)
    for invalid in ({**plan, "schema_version": -1}, {**plan, "protocol": "angle-regression"}):
        with pytest.raises(ValueError, match="schema"):
            data.materialize(source, cfg, invalid)
    with pytest.raises(ValueError, match="partition"):
        data.materialize(source, cfg, {**plan, "partition": "typo"})


def test_invalid_counts_sources_transforms_and_blur_fail_clearly(source, cfg):
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            data.sample_transforms(invalid, 0)
    with pytest.raises(ValueError, match="partition"):
        data.plan_data(source, cfg, "development", 0, "typo")
    with pytest.raises(ValueError, match="insufficient disjoint"):
        data.plan_data(source, replace(cfg, diagnostic_base_size=10000), "development", 0, "development")
    with pytest.raises(ValueError, match="insufficient disjoint"):
        data.plan_data(source, replace(cfg, eval_per_class=13), "development", 0, "development")
    with pytest.raises(ValueError, match="reference reserve"):
        data.plan_data(source, replace(cfg, reference_size=100), "development", 0, "development")
    with pytest.raises(ValueError, match="invertible"):
        data.render_pairs(torch.zeros(2, 1, 28, 28), torch.zeros(2, 2, 2), torch.zeros(2, 2))
    pairs = torch.zeros(2, 2, 28, 28)
    for invalid in (float("nan"), float("inf"), -1.):
        with pytest.raises(ValueError, match="blur_max"):
            data.blur_pairs(pairs, .7, invalid)
    for invalid in ([.5], [-1., 1.], [float("nan"), .5], [1.1, .5]):
        with pytest.raises(ValueError, match="strengths"):
            data.blur_pairs(pairs, invalid, 1.8)
    for invalid in (torch.zeros(2, 28, 28), pairs.to(torch.uint8), pairs - 1,
                    pairs + float("nan"), pairs + 2):
        with pytest.raises(ValueError):
            data.blur_pairs(invalid, .7, 1.8)
