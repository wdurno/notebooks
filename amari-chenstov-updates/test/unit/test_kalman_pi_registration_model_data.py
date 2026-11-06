"""Lightweight CPU contracts for Plan 17 Phase 6 model and render plans."""

from __future__ import annotations

import dataclasses
import io
import json
import math

import pytest
import torch
from torch.func import jacrev

from mnist_experiment.kalman_pi.registration import data, model
from mnist_experiment.kalman_pi.registration.config import Config, ROLE_COUNTS, smoke_config


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def raw():
    # Unequal source frequencies distinguish empirical weighting from balancing.
    train_y = torch.cat([torch.full((140 + 2 * digit,), digit, dtype=torch.long)
                         for digit in range(10)])
    test_y = torch.arange(10).repeat_interleave(6)
    train_x = torch.zeros(len(train_y), 1, 28, 28)
    train_x[:, :, 6:22, 11:17] = .8
    test_x = torch.zeros(len(test_y), 1, 28, 28)
    test_x[:, :, 7:21, 10:18] = .6
    return dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


@pytest.fixture(scope="module")
def assets(raw):
    return data.make_assets(smoke_config(), 0, "development", raw)


def test_config_defaults_roundtrip_hash_seeds_and_replace(tmp_path):
    cfg = Config()
    assert cfg.role_counts == ROLE_COUNTS
    assert (cfg.initialization_size, cfg.initial_epochs, cfg.initial_max_epochs) == (24000, 15, 30)
    assert cfg.num_p_steps == 1312 and cfg.fixed_pi == .025
    assert cfg.process_scales == (1e-5, 1e-4, 1e-3)
    assert (cfg.reference_size, cfg.reference_max_size, cfg.calibration_size) == (1024, 2048, 2048)
    assert dataclasses.replace(cfg, device="cpu", main_replicas=4).scientific_hash == cfg.scientific_hash
    assert dataclasses.replace(cfg, seed=1).scientific_hash != cfg.scientific_hash
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.mapping()), encoding="utf-8")
    assert Config.from_json(path) == cfg
    seeds = {cfg.seed_for("main", r, name) for r in (0, 1)
             for name in ("initialization_model", "stream:sources", "reference:transforms")}
    assert len(seeds) == 6
    assert cfg.seed_for("main", 0, "stream:sources") != cfg.seed_for("stationary", 0, "stream:sources")
    assert smoke_config().device == "cpu"


@pytest.mark.parametrize("changes", [dict(fisher_dtype="float32"), dict(anchor_refresh=64),
    dict(ramp_steps=1), dict(calibration_size=25000), dict(reference_size=3000),
    dict(angle_std=0), dict(angle_limit=60), dict(fixed_pi=0), dict(max_eval=1),
    dict(initial_max_epochs=14), dict(smoke=1), dict(seed=True), dict(main_replicas=9),
    dict(schedule_factors=[1., 2.]), dict(role_counts=(1, 2)), dict(ema_half_lives=[{}]),
    dict(unknown=3)])
def test_config_rejects_invalid_scientific_contract(changes):
    with pytest.raises(ValueError):
        Config.from_mapping(changes)


def test_model_count_order_functional_means_and_global_rng():
    rng = torch.get_rng_state().clone()
    cnn, layout = model.build_model(17, dtype=torch.float64)
    assert torch.equal(rng, torch.get_rng_state())
    assert layout.total_numel == 448
    assert layout.names == ("features.0.weight", "features.0.bias", "features.3.weight",
                            "features.3.bias", "classifier.weight", "classifier.bias")
    assert all(p.requires_grad for p in cnn.parameters())
    assert not any("lora" in name for name in layout.names)
    theta = layout.flatten_module(cnn, detach=True)
    before = {name: p.clone() for name, p in cnn.named_parameters()}
    x = torch.zeros(2, 2, 28, 28, dtype=torch.float64)
    torch.testing.assert_close(model.predict(cnn, layout, theta, x), cnn(x))
    changed = theta.clone()
    changed[-6:] += 100
    assert (model.predict(cnn, layout, changed, x) > 90).all()
    for name, p in cnn.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    twin, other = model.build_model(17, dtype=torch.float64)
    assert other.metadata() == layout.metadata()
    torch.testing.assert_close(layout.flatten_module(twin), theta, rtol=0, atol=0)


def test_gaussian_nll_coordinate_sum_and_observation_reduction():
    outputs = torch.arange(12, dtype=torch.float64).reshape(2, 6) / 10
    targets = outputs + .5
    exact = -torch.distributions.Normal(outputs, 2).log_prob(targets).sum(-1)
    torch.testing.assert_close(model.nll(outputs, targets, "none"), exact)
    torch.testing.assert_close(model.nll(outputs, targets), exact.mean())
    torch.testing.assert_close(model.nll(outputs, targets, "sum"), exact.sum())
    assert model.nll(outputs, outputs).item() == pytest.approx(3 * math.log(8 * math.pi))
    with pytest.raises(ValueError):
        model.nll(outputs, targets, "coordinate_mean")


def test_positive_scores_negative_loss_gradients_and_finite_difference():
    cnn, layout = model.build_model(3, dtype=torch.float64)
    theta = layout.flatten_module(cnn, detach=True)
    generator = torch.Generator().manual_seed(4)
    x = torch.rand(2, 2, 28, 28, generator=generator, dtype=torch.float64)
    y = torch.rand(2, 6, generator=generator, dtype=torch.float64)
    scores = model.observed_scores(cnn, layout, theta, x, y, chunk_size=1)
    variable = theta.clone().requires_grad_()
    gradient = torch.autograd.grad(model.nll(model.predict(cnn, layout, variable, x), y), variable)[0]
    torch.testing.assert_close(scores.mean(0), -gradient, rtol=1e-10, atol=1e-12)
    jacobian = jacrev(lambda t: model.predict(cnn, layout, t, x))(theta)
    analytic = torch.einsum("nki,nk->ni", jacobian, y - model.predict(cnn, layout, theta, x)) / 4
    torch.testing.assert_close(scores, analytic, rtol=1e-10, atol=1e-12)
    direction = torch.randn(theta.shape, generator=generator, dtype=torch.float64)
    direction /= direction.norm()
    epsilon = 1e-5
    positive = model.nll(model.predict(cnn, layout, theta + epsilon * direction, x), y, "none")
    negative = model.nll(model.predict(cnn, layout, theta - epsilon * direction, x), y, "none")
    torch.testing.assert_close(scores @ direction, -(positive - negative) / (2 * epsilon),
                               rtol=2e-6, atol=1e-10)
    torch.testing.assert_close(model.observed_scores(cnn, layout, theta, x, y, chunk_size=2), scores,
                               rtol=1e-10, atol=1e-12)


def test_model_integrated_diagonal_is_jtj_not_observed_residual_square():
    cnn, layout = model.build_model(5)
    theta = layout.flatten_module(cnn, detach=True)
    generator = torch.Generator().manual_seed(6)
    x = torch.rand(2, 2, 28, 28, generator=generator)
    jacobian = jacrev(lambda t: model.predict(cnn, layout, t, x.double()))(theta.double())
    expected = jacobian.square().sum(1).mean(0) / 4
    calls = []
    actual = model.observation(cnn, layout, theta, x, chunk_size=1, check=lambda: calls.append(1))
    assert len(calls) == 2 and actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual[-6:], torch.full((6,), .25, dtype=torch.float64))
    torch.testing.assert_close(model.observation(cnn, layout, theta, x, chunk_size=2), actual,
                               rtol=1e-10, atol=1e-12)
    y = model.predict(cnn, layout, theta.double(), x.double()).detach()
    scores = model.observed_scores(cnn, layout, theta, x, y)
    assert scores.square().sum() == 0 and actual.sum() > 0
    assert not actual.requires_grad
    with pytest.raises(ValueError):
        model.observation(cnn, layout, theta, x, chunk_size=0)


def test_schedule_continuity_horizons_and_balanced_assignments():
    cfg = Config()
    h, segments = data.make_schedule(cfg)
    assert len(h) == cfg.num_p_steps
    rise = h[cfg.hold_steps:cfg.hold_steps + cfg.ramp_steps]
    torch.testing.assert_close(rise, (1 - torch.cos(torch.pi * torch.linspace(0, 1, 128, dtype=torch.float64))) / 2)
    for left, right in zip(segments[:-1], segments[1:]):
        assert left["end"] == right["start"]
        assert h[left["end"] - 1] == h[right["start"]]
    stationary, still = data.make_schedule(cfg, stationary=True)
    assert not stationary.any() and [(s["start"], s["end"]) for s in still] == [(s["start"], s["end"]) for s in segments]
    extended, extended_segments = data.make_schedule(cfg, factor=1.25, hold_factor=2)
    assert extended_segments[0]["arrivals"] == 880
    assert extended_segments[1]["arrivals"] == 160
    assert len(extended) == 2960
    assignments = [data.replica_assignment(cfg, i, "main") for i in range(8)]
    for sign in (-1, 1):
        for factor in (1., 1.25):
            assert sum(a["sign"] == sign and a["factor"] == factor for a in assignments) == 2
    assert [data.replica_assignment(cfg, i, "development")["sign"] for i in range(2)] == [-1, 1]


def test_transform_law_support_counts_densities_and_forward_conventions():
    means = torch.cat([torch.zeros(32), torch.full((32,), 25.)]).double()
    before = torch.get_rng_state().clone()
    parameters, info = data.sample_transforms(means, 9)
    assert torch.equal(before, torch.get_rng_state())
    repeated, other = data.sample_transforms(means, 9)
    torch.testing.assert_close(parameters, repeated, rtol=0, atol=0)
    assert info["proposed"] == other["proposed"]
    matrix, translation = data.affine_from_parameters(parameters)
    assert data.support_mask(matrix, translation).all()
    assert parameters[:, 0].abs().max() <= 50
    assert parameters[:, 1:3].abs().max() <= 2
    assert parameters[:, 3:5].abs().max() <= math.log(1.1)
    assert parameters[:, 5].abs().max() <= .1
    assert info["accepted"] == 64
    assert info["proposed"] == info["rejected"] + 64 == int(info["proposal_counts"].sum())
    assert sum(info["proposed_angle_counts"]) == info["proposed"]
    assert sum(info["accepted_angle_counts"]) == 64
    assert sum(info["accepted_angle_density"]) * 2.5 == pytest.approx(1)
    assert torch.tensor(info["target_covariance"]).shape == (6, 6)
    # Positive alpha rotates clockwise in downward-positive image coordinates.
    matrix, translation = data.affine_from_parameters([[10., 0., 0., 0., 0., 0.]])
    assert matrix[0, 1, 0] > 0
    targets = data.encode_affine(matrix, translation).reshape(1, 3, 2)
    anchors = torch.tensor(data.ANCHORS, dtype=torch.float64)
    torch.testing.assert_close(targets, anchors @ matrix.transpose(1, 2) - anchors)


def test_study_wide_disjoint_roles_and_test_split(raw, assets):
    cfg = smoke_config()
    other = data.make_assets(dataclasses.replace(cfg, initialization_size=12), 1, "main", raw)
    roles = list(assets["partitions"])
    for i, role in enumerate(roles):
        ids = assets["partitions"][role]
        torch.testing.assert_close(ids, other["partitions"][role], rtol=0, atol=0)
        for second in roles[i + 1:]:
            assert not set(ids.tolist()) & set(assets["partitions"][second].tolist())
    assert sum(len(assets["partitions"][role]) for role in data.TRAIN_ROLES) == len(raw["train_y"])
    assert assets["metadata"]["partition_hash"] == other["metadata"]["partition_hash"]
    assert not set(assets["panels"]["old"]["ids"].tolist()) & set(other["panels"]["old"]["ids"].tolist())
    assert assets["panels"]["old"]["ids"].min() >= len(raw["train_y"])
    assert assets["metadata"]["raw_train_class_counts"] == torch.bincount(raw["train_y"]).tolist()


def test_canonical_allocation_exact_counts_without_loading_images():
    labels = torch.arange(10).repeat_interleave(6000)
    raw_labels = dict(train_y=labels, test_y=torch.arange(10).repeat_interleave(2))
    partitions, _ = data._partitions(raw_labels, Config())
    assert tuple(len(partitions[role]) for role in data.TRAIN_ROLES) == ROLE_COUNTS
    assert torch.cat([partitions[role] for role in data.TRAIN_ROLES]).unique().numel() == 60000


def test_angle_and_hold_extension_preserve_initialization_and_nested_reference(raw, assets):
    cfg = dataclasses.replace(smoke_config(), reference_size=8)
    alternative = data.make_assets(cfg, 0, "development", raw, angle=35., hold_factor=2)
    for role in ("initialization", "validation", "reference", "calibration"):
        for name in ("ids", "transforms"):
            torch.testing.assert_close(assets["panels"][role][name], alternative["panels"][role][name],
                                       rtol=0, atol=0)
    primary = data.render_panel(raw, assets["panels"]["reference"], slice(0, 4))
    reserve = data.render_panel(raw, alternative["panels"]["reference"])
    for left, right in zip(primary, reserve):
        torch.testing.assert_close(left, right[:4], rtol=0, atol=0)
    assert alternative["metadata"]["resolved_hold_steps"] == 4
    assert alternative["metadata"]["resolved_ramp_steps"] == 2


def test_render_plans_pairing_aliases_calibration_and_no_image_copies(raw, assets):
    cfg = smoke_config()
    repeated = data.make_assets(cfg, 0, "development", raw)
    assert assets["panels"]["initialization"] is assets["panels"]["init"]
    positions = assets["calibration_indices"]
    for key in ("ids", "transforms", "digits"):
        torch.testing.assert_close(assets["panels"]["calibration"][key],
            assets["panels"]["initialization"][key][positions], rtol=0, atol=0)
    initial_x, initial_y = data.render_panel(raw, assets["panels"]["initialization"])
    calibration_x, calibration_y = data.render_panel(raw, assets["panels"]["calibration"])
    torch.testing.assert_close(calibration_x, initial_x[positions], rtol=0, atol=0)
    torch.testing.assert_close(calibration_y, initial_y[positions], rtol=0, atol=0)
    assert initial_x.shape == (cfg.initialization_size, 2, 28, 28)
    assert initial_y.shape == (cfg.initialization_size, 6) and initial_y.square().sum() > 0
    assert initial_x.dtype == initial_y.dtype == torch.float32
    assert initial_x.device.type == "cpu"
    for name in ("main", "stationary"):
        stream = assets["streams"][name]
        assert stream["ids"].unique().numel() == stream["ids"].numel()
        assert len(stream["segments"]) == len(stream["ids"])
        assert len(stream["segment_ranges"]) == 5
        assert stream["hash"] == repeated["streams"][name]["hash"]
        x, y = data.render_batch(raw, stream, 0)
        values = data.get_batch(raw, assets, name, torch.arange(cfg.samples_per_step))
        assert set(values) == {"inputs", "targets", "digits", "ids"}
        torch.testing.assert_close(values["inputs"], x, rtol=0, atol=0)
        torch.testing.assert_close(values["targets"], y, rtol=0, atol=0)
    assert assets["streams"]["main"]["ids"] is assets["streams"]["stationary"]["ids"]
    assert not assets["streams"]["stationary"]["h"].any()
    assert assets["streams"]["main"]["h"].max() == 1

    def no_images(value):
        if isinstance(value, torch.Tensor):
            assert not (value.ndim >= 3 and value.shape[-2:] == (28, 28))
        elif isinstance(value, dict):
            for item in value.values():
                no_images(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                no_images(item)

    no_images(assets)


def test_fixed_paired_evaluation_panels_empirical_weights_and_rendering(raw, assets):
    old, shifted = assets["panels"]["old"], assets["panels"]["shifted"]
    assert old["ids"] is shifted["ids"]
    assert not torch.equal(old["transforms"], shifted["transforms"])
    assert old["weights"].sum() == pytest.approx(1.)
    for digit in range(10):
        expected = float((raw["train_y"] == digit).double().mean())
        assert old["weights"][old["digits"] == digit].sum() == pytest.approx(expected)
    x0, y0 = data.render_panel(raw, old)
    x1, y1 = data.render_panel(raw, shifted)
    torch.testing.assert_close(x0[:, 0], x1[:, 0], rtol=0, atol=0)
    assert not torch.equal(y0, y1)
    assert not torch.equal(x0[:, 1], x1[:, 1])
    assert torch.isfinite(x0).all() and 0 <= x0.min() <= x0.max() <= 1
    one, target = data.render_panel(raw, old, 0)
    assert one.shape == (1, 2, 28, 28) and target.shape == (1, 6)
    empty, target = data.render_panel(raw, old, slice(0, 0))
    assert empty.shape == (0, 2, 28, 28) and target.shape == (0, 6)


def test_torch_safe_roundtrip_and_mnist_source_adapter(raw, assets):
    json.dumps(assets["metadata"], allow_nan=False)
    buffer = io.BytesIO()
    torch.save(assets, buffer)
    assert buffer.tell() < 500_000
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert loaded["streams"]["main"]["hash"] == assets["streams"]["main"]["hash"]

    class Source:
        train_labels, test_labels = raw["train_y"], raw["test_y"]

        def images(self, ids, *, test=False):
            return raw["test_x" if test else "train_x"][ids]

    source = Source()
    other = data.make_assets(smoke_config(), 0, "development", source)
    left = data.get_batch(raw, assets, "initialization", torch.arange(3))
    right = data.get_batch(source, other, "initialization", torch.arange(3))
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_insufficient_sources_and_invalid_rendering_fail_clearly(raw, assets):
    with pytest.raises(ValueError, match="disjoint sources"):
        data.make_assets(dataclasses.replace(smoke_config(), initialization_size=1000), 0, "main", raw)
    with pytest.raises(ValueError, match="existing arrival"):
        data.render_batch(raw, assets["streams"]["main"], -1)
    invalid = dict(assets["panels"]["initialization"])
    invalid["transforms"] = invalid["transforms"].clone()
    invalid["transforms"][0, 1] = 20
    with pytest.raises(ValueError, match="geometrically invalid"):
        data.render_panel(raw, invalid)
