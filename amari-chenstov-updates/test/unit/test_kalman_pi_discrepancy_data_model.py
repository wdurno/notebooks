"""CPU contracts for padded drift, paired current-law panels and one Fisher."""

from __future__ import annotations

import dataclasses
import io
import itertools
import json
import math

import pytest
import torch
from torch.func import jacrev

from mnist_experiment.kalman_pi.discrepancy import data, model
from mnist_experiment.kalman_pi.discrepancy.adapters import BatchReader, stream_view
from mnist_experiment.kalman_pi.discrepancy.config import Config, load, smoke_config


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def raw():
    train_y = torch.cat([torch.full((140 + 2 * digit,), digit, dtype=torch.long)
                         for digit in range(10)])
    test_y = torch.arange(10).repeat_interleave(6)
    train_x = torch.zeros(len(train_y), 1, 28, 28)
    train_x[:, :, 6:22, 11:17] = .8
    train_x[:, :, 16:22, 6:17] = .5
    test_x = torch.zeros(len(test_y), 1, 28, 28)
    test_x[:, :, 7:21, 10:18] = .6
    return dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


@pytest.fixture(scope="module")
def assets(raw):
    return data.make_assets(smoke_config(), 0, "development", raw)


def test_default_config_roundtrip_different_protocol_and_seeds(tmp_path):
    cfg = Config()
    assert cfg.seed == 171105 and cfg.protocol == "plan17-discrepancy-pi-v1"
    assert (cfg.hold_steps, cfg.ramp_steps, cfg.num_p_steps) == (352, 304, 1312)
    assert cfg.amplitudes == (25., 45.) and cfg.angle_limit == 70
    assert cfg.fresh_paths == 80
    assert cfg.ema_half_lives == cfg.retention_ridges == ()
    assert dataclasses.replace(cfg, main_replicas=4).fresh_paths == 48
    assert dataclasses.replace(cfg, device="cpu", threads=1).scientific_hash == cfg.scientific_hash
    assert dataclasses.replace(cfg, angle_max=45).scientific_hash != cfg.scientific_hash
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.mapping()), encoding="utf-8")
    assert load(path) == cfg == Config.from_mapping(cfg.mapping())
    assert len({cfg.seed_for(c, r, s) for c in ("main", "development") for r in (0, 1)
                for s in ("initialization_model", "stream:sources", "reference:transforms")}) == 12


@pytest.mark.parametrize("changes", [dict(protocol="plan17-registration-pi-v1"),
    dict(schema_version=True), dict(seed=True), dict(angle_limit=50), dict(amplitudes=[45, 25]),
    dict(angle_max=35), dict(ema_half_lives=[8]), dict(retention_ridges=[0]), dict(unknown=1),
    dict(ramp_steps=1), dict(main_replicas=9), dict(fisher_dtype="float32"),
    dict(initial_max_epochs=14), dict(reference_size=3000), dict(cold_start=2000)])
def test_config_rejects_invalid_protocol_and_unapproved_conditions(changes):
    with pytest.raises(ValueError):
        Config.from_mapping(changes)


def test_schedule_has_two_holds_two_ramps_and_smooth_shared_turning_point():
    cfg = Config()
    h, segments = data.make_schedule(cfg)
    assert [(s["start"], s["end"]) for s in segments] == [(0, 352), (352, 656), (656, 960), (960, 1312)]
    assert [s["name"] for s in segments] == ["low_hold_initial", "rise", "return", "low_hold_final"]
    assert (h[655], h[656]) == (1, 1)
    torch.testing.assert_close(h[352:656], h[656:960].flip(0), rtol=0, atol=0)
    assert float((45 * h.diff()).abs().max()) == pytest.approx(.23328, rel=1e-3)
    longer, long_segments = data.make_schedule(cfg, factor=1.25)
    assert len(longer) == 1640
    assert [s["arrivals"] for s in long_segments] == [440, 380, 380, 440]
    stationary, still_segments = data.make_schedule(cfg, stationary=True)
    assert not stationary.any()
    assert [s["arrivals"] for s in still_segments] == [352, 304, 304, 352]
    assignments = [data.replica_assignment(cfg, i, "main") for i in range(8)]
    assert all(sum(a["sign"] == sign and a["factor"] == length for a in assignments) == 2
               for sign, length in itertools.product((-1, 1), (1., 1.25)))


def test_all_extreme_proposals_fit_support_and_retain_independent_uniform_factors():
    nuisance_bounds = (2., 2., math.log(1.1), math.log(1.1), .1)
    extreme = torch.tensor([(angle, *(s * b for s, b in zip(signs, nuisance_bounds, strict=True)))
        for angle in range(-70, 71) for signs in itertools.product((-1, 1), repeat=5)], dtype=torch.float64)
    matrix, translation = data.affine_from_parameters(extreme)
    assert data.support_mask(matrix, translation).all()
    mapped = matrix.new_tensor(data.SUPPORT_CORNERS) @ matrix.transpose(-1, -2) + translation[:, None]
    bound = 2 + 9.5 * 1.1 * math.sqrt(1.1**2 + 1)
    assert float(mapped.abs().max()) <= bound < data.SUPPORT_BOUND
    rng = torch.get_rng_state().clone()
    low, info0 = data.sample_transforms(25., 13, count=4096)
    high, info1 = data.sample_transforms(45., 13, count=4096)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(low[:, 1:], high[:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(info0["uniforms"], info1["uniforms"], rtol=0, atol=0)
    assert torch.all(high[:, 0] > low[:, 0])
    for info in (info0, info1):
        assert info["rejected"] == 0 and info["proposed"] == info["accepted"] == 4096
        assert info["proposal_counts"].eq(1).all()
        assert sum(info["accepted_angle_density"]) * 3.5 == pytest.approx(1)
    assert info1["untruncated_mass_outside_mean"] == pytest.approx(1.54543e-5, rel=1e-5)
    replay, _ = data.sample_transforms(45., 999, count=4096, uniforms=info0["uniforms"])
    torch.testing.assert_close(replay, high, rtol=0, atol=0)


def test_padded_render_identity_clockwise_convention_and_no_border_clipping(raw):
    images = raw["train_x"][:1]
    parameters = torch.zeros(1, 6, dtype=torch.float64)
    identity = data.render_pairs(images, *data.affine_from_parameters(parameters))
    assert identity.shape == (1, 2, 40, 40)
    torch.testing.assert_close(identity[:, 0], identity[:, 1], rtol=0, atol=1e-6)
    assert identity[:, :, :10].count_nonzero() == 0
    parameters[0, 0] = 90
    turned = data.render_pairs(images, *data.affine_from_parameters(parameters))
    torch.testing.assert_close(turned[:, 1], torch.rot90(turned[:, 0], -1, (-2, -1)), rtol=0, atol=1e-6)
    parameters = torch.tensor([[45., 2., -2., math.log(1.1), math.log(1.1), .1],
                               [-45., -2., 2., math.log(1.1), math.log(1.1), -.1]], dtype=torch.float64)
    rendered = data.render_pairs(torch.ones(2, 1, 28, 28), *data.affine_from_parameters(parameters))
    assert rendered[:, 1].sum() > 0
    assert rendered[:, :, (0, -1), :].count_nonzero() == 0
    assert rendered[:, :, :, (0, -1)].count_nonzero() == 0
    with pytest.raises(ValueError, match="geometrically invalid"):
        data.render_pairs(images, torch.eye(2)[None], torch.tensor([[20., 0.]]))


def test_amplitude_pairing_disjoint_roles_and_shared_initialization(raw, assets):
    larger = data.make_assets(smoke_config(), 0, "development", raw, angle=45.)
    assert larger["metadata"]["partition_hash"] == assets["metadata"]["partition_hash"]
    for role in ("init", "validation", "reference", "calibration"):
        for key in ("ids", "transforms", "uniforms"):
            torch.testing.assert_close(assets["panels"][role][key], larger["panels"][role][key], rtol=0, atol=0)
    for role in ("main", "stationary"):
        for key in ("ids", "uniforms"):
            torch.testing.assert_close(assets["streams"][role][key], larger["streams"][role][key], rtol=0, atol=0)
        torch.testing.assert_close(assets["streams"][role]["transforms"][..., 1:],
                                   larger["streams"][role]["transforms"][..., 1:], rtol=0, atol=0)
        assert assets["streams"][role]["ids"].unique().numel() == assets["streams"][role]["ids"].numel()
    torch.testing.assert_close(assets["streams"]["stationary"]["transforms"],
                               larger["streams"]["stationary"]["transforms"], rtol=0, atol=0)
    pools = list(assets["partitions"].values())
    for i, ids in enumerate(pools):
        for other in pools[i + 1:]:
            assert not set(ids.tolist()) & set(other.tolist())
    fresh = data.make_assets(smoke_config(), 0, "main", raw)
    assert not set(fresh["panels"]["old"]["ids"].tolist()) & set(assets["panels"]["old"]["ids"].tolist())
    indices = assets["calibration_indices"]
    initial = data.get_batch(raw, assets, "initialization", indices)
    calibration = data.get_batch(raw, assets, "calibration")
    for name in initial:
        torch.testing.assert_close(initial[name], calibration[name], rtol=0, atol=0)


def test_current_panels_post_batch_index_shared_uniforms_endpoint_dedup_and_weights(raw):
    cfg = smoke_config(ramp_steps=4)
    assets = data.make_assets(cfg, 0, "development", raw, angle=45)
    old, shifted = assets["panels"]["old"], assets["panels"]["shifted"]
    assert data.current_panel(assets, "main", 0) is old
    assert data.current_panel(assets, "main", 3) is old
    assert data.current_panel(assets, "main", 6) is shifted
    for step in range(cfg.num_p_steps + 1):
        assert data.current_panel(assets, "stationary", step) is old
    panel = data.current_panel(assets, "main", 4)
    assert panel["mean"] == float(assets["streams"]["main"]["mu"][3])
    assert panel["ids"] is old["ids"] and panel["uniforms"] is old["uniforms"]
    assert panel["weights"] is old["weights"]
    torch.testing.assert_close(panel["transforms"][:, 1:], old["transforms"][:, 1:], rtol=0, atol=0)
    assert not torch.equal(panel["transforms"][:, 0], old["transforms"][:, 0])
    batch = data.get_batch(raw, assets, "current:main:4")
    assert batch["inputs"].shape == (10, 2, 40, 40)
    assert batch["weights"].sum() == pytest.approx(1)
    assert batch["ids"].min() >= len(raw["train_y"])
    for digit in range(10):
        assert batch["weights"][batch["digits"] == digit].sum() == pytest.approx(
            float((raw["train_y"] == digit).double().mean()))
    for role in ("current:main:-1", "current:main:999", "current:unknown:0", "current:main:1:2"):
        with pytest.raises(ValueError):
            data.get_batch(raw, assets, role)


def test_reader_bounded_current_cache_and_equivalence_to_direct_rendering(raw):
    cfg = smoke_config(ramp_steps=4)
    assets = data.make_assets(cfg, 0, "development", raw)
    reader = BatchReader(cfg, assets, raw)
    reader.preload_evaluation()
    assert set(reader.cache) == {"old", "shifted"}
    for step in range(cfg.num_p_steps + 1):
        role = f"current:main:{step}"
        direct = data.get_batch(raw, assets, role)
        cached = reader(role)
        for key in direct:
            torch.testing.assert_close(direct[key], cached[key], rtol=0, atol=0)
        assert not any(name.startswith("current:") for name in reader.cache)
        assert len(reader.cache) == 2
        if reader._current_cache is not None:
            assert reader._current_cache[1][0].shape == (10, 2, 40, 40)
    reader.preload_initialization()
    direct = data.get_batch(raw, assets, "initialization", torch.arange(3))
    for key, value in reader("initialization", torch.arange(3)).items():
        torch.testing.assert_close(value, direct[key], rtol=0, atol=0)
    assert reader("old", torch.empty(0, dtype=torch.long))["inputs"].shape == (0, 2, 40, 40)
    assert stream_view(assets, "main") == {"segments": assets["streams"]["main"]["segments"],
                                           "hash": assets["streams"]["main"]["hash"]}


def test_model_dimension_functional_prediction_and_shared_parameter_layout():
    before = torch.get_rng_state().clone()
    cnn, layout = model.build_model(17, dtype=torch.float64)
    assert torch.equal(before, torch.get_rng_state())
    assert layout.total_numel == 448
    assert layout.names == ("features.0.weight", "features.0.bias", "features.3.weight",
                            "features.3.bias", "classifier.weight", "classifier.bias")
    features = torch.randn(2, 6, 20, 20, dtype=torch.float64, requires_grad=True)
    pooled = cnn.features[-1](features)
    reference = torch.nn.functional.adaptive_avg_pool2d(features, (2, 2))
    torch.testing.assert_close(pooled, reference, rtol=1e-14, atol=1e-15)
    weight = torch.arange(pooled.numel(), dtype=torch.float64).reshape_as(pooled)
    actual_grad = torch.autograd.grad((pooled * weight).sum(), features, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((reference * weight).sum(), features)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-14, atol=1e-15)
    theta = layout.flatten_module(cnn, detach=True)
    x = torch.zeros(2, 2, 40, 40, dtype=torch.float64)
    torch.testing.assert_close(model.predict(cnn, layout, theta, x), cnn(x))
    changed = theta.clone()
    changed[-6:] += 100
    assert (model.predict(cnn, layout, changed, x) > 90).all()
    torch.testing.assert_close(layout.flatten_module(cnn), theta, rtol=0, atol=0)
    with pytest.raises(ValueError, match="40,40"):
        cnn(torch.zeros(2, 2, 28, 28, dtype=torch.float64))


def test_nll_scores_and_model_integrated_fisher_match_exact_likelihood():
    cnn, layout = model.build_model(4, dtype=torch.float64)
    theta = layout.flatten_module(cnn, detach=True)
    generator = torch.Generator().manual_seed(9)
    x = torch.rand(2, 2, 40, 40, generator=generator, dtype=torch.float64)
    y = torch.rand(2, 6, generator=generator, dtype=torch.float64)
    outputs = model.predict(cnn, layout, theta, x)
    expected_nll = -torch.distributions.Normal(outputs, 2).log_prob(y).sum(-1)
    torch.testing.assert_close(model.nll(outputs, y, "none"), expected_nll)
    scores = model.observed_scores(cnn, layout, theta, x, y, chunk_size=1)
    variable = theta.clone().requires_grad_()
    gradient = torch.autograd.grad(model.nll(model.predict(cnn, layout, variable, x), y), variable)[0]
    torch.testing.assert_close(scores.mean(0), -gradient, rtol=1e-10, atol=1e-12)
    jacobian = jacrev(lambda t: model.predict(cnn, layout, t, x))(theta)
    torch.testing.assert_close(scores, torch.einsum("nki,nk->ni", jacobian, y - outputs) / 4,
                               rtol=1e-10, atol=1e-12)
    actual = model.observation(cnn, layout, theta, x, chunk_size=1)
    torch.testing.assert_close(actual, jacobian.square().sum(1).mean(0) / 4, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual[-6:], torch.full((6,), .25, dtype=torch.float64))
    torch.testing.assert_close(model.observation(cnn, layout, theta, x, chunk_size=2), actual)
    at_mean_scores = model.observed_scores(cnn, layout, theta, x, outputs.detach())
    assert at_mean_scores.square().sum() == 0 and actual.sum() > 0
    assert actual.dtype == scores.dtype == torch.float64
    assert not actual.requires_grad and not scores.requires_grad


def test_artifacts_have_no_images_and_safe_serialization(raw, assets):
    json.dumps(assets["metadata"], allow_nan=False)
    buffer = io.BytesIO()
    torch.save(assets, buffer)
    buffer.seek(0)
    recovered = torch.load(buffer, weights_only=True)
    assert recovered["streams"]["main"]["hash"] == assets["streams"]["main"]["hash"]
    assert recovered["panels"]["old"]["uniforms"].shape == (10, 6)

    def no_images(value):
        if isinstance(value, torch.Tensor):
            assert not (value.ndim >= 3 and value.shape[-2:] in ((28, 28), (40, 40)))
        elif isinstance(value, dict):
            for item in value.values():
                no_images(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                no_images(item)

    no_images(assets)
    with pytest.raises(ValueError, match="disjoint sources"):
        data.make_assets(dataclasses.replace(smoke_config(), initialization_size=1000), 0, "main", raw)
