"""Plan 18 source budgets, incoming-grid clocks, and padded physical rendering."""

from dataclasses import replace
import io
import itertools
import json
import math

import pytest
import torch
from torch.nn import functional as F

from mnist_experiment.combined_memory import data
from mnist_experiment.combined_memory.config import Config
from mnist_experiment.fisher_memory_registration.data import decode_affine
from mnist_experiment.kalman_pi.discrepancy import data as inherited


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def raw():
    train_y = torch.cat([torch.full((40 + 5 * digit,), digit, dtype=torch.long)
                         for digit in range(10)])
    test_y = torch.arange(10).repeat_interleave(30)
    train_x = torch.zeros(len(train_y), 1, 28, 28)
    train_x[:, :, 4:24, 11:17] = .8
    test_x = torch.zeros(len(test_y), 1, 28, 28)
    test_x[:, :, 5:22, 9:18] = torch.linspace(.2, 1., len(test_y))[:, None, None, None]
    return dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


@pytest.fixture(scope="module")
def assets(raw):
    return data.make_assets(Config.smoke_config(), 0, "development", raw)


def ids_for(assets):
    return {**{role: assets[role]["ids"] for role in data.TRAIN_ROLES},
            **{role: assets["panels"][role]["ids"] for role in data.PANEL_ROLES}}


def test_production_exact_schedule_one_based_ramps_and_expected_novel_budget():
    cfg = Config()
    h, segments = data.make_schedule(cfg)
    assert len(h) == 928 and h.dtype == torch.float64
    assert [(part["name"], part["start"], part["end"]) for part in segments] == [
        ("low_hold_initial", 0, 160), ("rise", 160, 464),
        ("return", 464, 768), ("low_hold_final", 768, 928)]
    j = torch.arange(1, 305, dtype=torch.float64)
    torch.testing.assert_close(h[160:464], (1 - torch.cos(torch.pi * j / 304)) / 2)
    torch.testing.assert_close(h[464:768], (1 + torch.cos(torch.pi * j / 304)) / 2)
    assert not h[:160].any() and not h[768:].any()
    assert h[160] > 0 and h[463] == 1 and h[464] < 1 and h[767] == 0
    assert int((h == 1).sum()) == 1
    assert float(h.sum() * cfg.novel_probability * cfg.samples_per_step) == pytest.approx(152.)
    assert max(abs(float(value)) for value in (h[1:] - h[:-1]) * 45) == pytest.approx(.232516,
                                                                                             abs=1e-5)
    still, boundaries = data.make_schedule(cfg, stationary=True)
    assert not still.any()
    assert [(s["start"], s["end"]) for s in boundaries] == [(s["start"], s["end"]) for s in segments]


def test_all_training_roles_uniform_without_replacement_and_global_test_ids(raw, assets):
    cfg = Config.smoke_config()
    sizes = (cfg.init_size, cfg.validation_size, cfg.reference_size, cfg.diagnostic_size, cfg.stream_size)
    roles = ids_for(assets)
    training = torch.cat([roles[role].flatten() for role in data.TRAIN_ROLES])
    assert training.numel() == training.unique().numel() == sum(sizes) * 9
    generator = torch.Generator().manual_seed(assets["seeds"]["training_sources"])
    torch.testing.assert_close(training, torch.randperm(len(raw["train_y"]), generator=generator)[:len(training)],
                               rtol=0, atol=0)
    for role, size in zip(data.TRAIN_ROLES, sizes):
        panel = assets[role]
        assert panel["ids"].shape == panel["digits"].shape == (size, 9)
        assert panel["transforms"].shape == (size, 9, 6)
        assert not panel["test"]
        torch.testing.assert_close(panel["digits"], raw["train_y"][panel["ids"]])
    for first, second in itertools.combinations(roles, 2):
        assert not set(roles[first].flatten().tolist()) & set(roles[second].flatten().tolist())
    for role in data.PANEL_ROLES:
        panel = assets["panels"][role]
        assert panel["test"] and int(panel["ids"].min()) >= len(raw["train_y"])
        torch.testing.assert_close(panel["digits"], raw["test_y"][panel["ids"] - len(raw["train_y"])])
    assert not assets["metadata"]["source_sampling"]["class_balancing"]
    assert not assets["metadata"]["source_sampling"]["finite_population_independent"]
    assert assets["metadata"]["raw_train_class_counts"] == torch.bincount(raw["train_y"]).tolist()


def test_test_halves_fixed_study_wide_and_independent_across_roles(raw, assets):
    fresh = data.make_assets(Config.smoke_config(), 1, "main", raw)
    assert assets["seeds"]["evaluation_partition"] == fresh["seeds"]["evaluation_partition"]
    assert assets["metadata"]["test_partition_hash"] == fresh["metadata"]["test_partition_hash"]
    dev_ids = torch.cat([assets["panels"][role]["ids"].flatten() for role in data.PANEL_ROLES])
    main_ids = torch.cat([fresh["panels"][role]["ids"].flatten() for role in data.PANEL_ROLES])
    assert dev_ids.unique().numel() == dev_ids.numel()
    assert main_ids.unique().numel() == main_ids.numel()
    assert not set(dev_ids.tolist()) & set(main_ids.tolist())
    assert assets["metadata"]["evaluation_role"] == "development"
    assert fresh["metadata"]["evaluation_role"] == "confirmation"


def test_canonical_source_table_without_loading_or_rendering_images():
    labels = dict(train_y=torch.arange(10).repeat_interleave(6000),
                  test_y=torch.arange(10).repeat_interleave(1000))
    assets = data.make_assets(Config(), 0, "development", labels)
    counts = assets["metadata"]["role_source_counts"]
    assert [counts[role] for role in data.TRAIN_ROLES] == [20700, 1800, 2304, 1152, 33408]
    assert sum(counts[role] for role in data.TRAIN_ROLES) == 59364
    assert [counts[role] for role in data.PANEL_ROLES] == [1152, 1152, 1152]
    assert assets["metadata"]["schedule"]["expected_novel"] == pytest.approx(152.)
    assert assets["stream"]["ids"].unique().numel() == 33408


def test_reproducible_compact_assets_named_seeds_and_no_global_rng(raw, assets):
    state = torch.get_rng_state().clone()
    twin = data.make_assets(Config.smoke_config(), 0, "development", raw)
    assert torch.equal(state, torch.get_rng_state())
    assert twin["metadata"]["asset_hash"] == assets["metadata"]["asset_hash"]
    assert twin["metadata"]["source_allocation_hash"] == assets["metadata"]["source_allocation_hash"]
    for role in data.TRAIN_ROLES:
        for key in ("ids", "transforms", "digits"):
            torch.testing.assert_close(twin[role][key], assets[role][key], rtol=0, atol=0)
    required = {"initialization_model", "initialization_order", "factors", "training_sources",
                "stream:transforms", "stream:mixture", "reference:transforms", "replay", "probes",
                "diagnostics", "bootstrap", "cell_signs", "noise_calibration", "evaluation_sources",
                "current:mixture", "current:transforms", "novel:severity", "evaluation_partition"}
    assert required <= set(assets["seeds"])
    cfg = Config.smoke_config()
    for step in range(cfg.num_p_steps):
        for name in ("replay", "probes"):
            assert assets["seeds"][f"{name}:{step}"] == cfg.seed_for("development", 0, f"{name}:{step}")
    assert len(set(assets["seeds"].values())) == len(assets["seeds"])
    assert all(len(value) == 64 for value in assets["metadata"]["source_hashes"].values())
    json.dumps(assets["metadata"], allow_nan=False)
    memory = io.BytesIO()
    torch.save(assets, memory)
    assert memory.tell() < 200_000
    memory.seek(0)
    restored = torch.load(memory, weights_only=True)
    assert restored["metadata"]["asset_hash"] == assets["metadata"]["asset_hash"]

    def no_images(value):
        if isinstance(value, torch.Tensor):
            assert value.ndim < 3 or value.shape[-2:] not in ((28, 28), (40, 40), (120, 120))
        elif isinstance(value, dict):
            for item in value.values():
                no_images(item)

    no_images(assets)


def test_balanced_global_flip_static_cell_pattern_and_exact_stream_law(raw, assets):
    cfg = Config.smoke_config()
    assert assets["metadata"]["assignment"]["global_sign"] == -1
    signs = assets["signs"]
    assert int((signs == -1).sum()) == 4 and int((signs == 1).sum()) == 5
    other = data.make_assets(cfg, 1, "development", raw)
    assert other["metadata"]["assignment"]["global_sign"] == 1
    assert int((other["signs"] == 1).sum()) == 4
    stream = assets["stream"]
    assert stream["novel"].shape == (cfg.stream_size,) and stream["novel"].dtype == torch.bool
    assert stream["h"].shape == stream["p"].shape == (cfg.num_p_steps,)
    assert torch.equal(stream["p"], cfg.novel_probability * stream["h"])
    assert torch.equal(stream["novel"], stream["mixture_uniforms"] < stream["p"].repeat_interleave(
        cfg.samples_per_step))
    means = stream["novel"].double()[:, None] * stream["h"].repeat_interleave(
        cfg.samples_per_step)[:, None] * cfg.angle_max * signs
    expected, _ = data.sample_transforms(means.flatten(), assets["seeds"]["stream:transforms"],
        angle_std=cfg.angle_std, angle_limit=cfg.angle_limit, uniforms=stream["uniforms"].reshape(-1, 6))
    torch.testing.assert_close(stream["transforms"], expected.reshape(-1, 9, 6), rtol=0, atol=0)
    expected_count = stream["novel"].reshape(-1, cfg.samples_per_step).sum(1).cumsum(0)
    assert torch.equal(stream["cumulative_novel"], expected_count)
    assert torch.equal(stream["cumulative_grids"], torch.arange(1, cfg.num_p_steps + 1) * cfg.samples_per_step)
    assert sum(s["realized_novel"] for s in stream["segments"]) == int(stream["novel"].sum())
    for info in assets["metadata"]["transform_diagnostics"].values():
        assert info["rejected"] == 0 and info["accepted"] == info["proposed"]


def test_stationary_keeps_full_arrival_source_budget_but_never_novel(raw):
    cfg = Config.smoke_config()
    stationary = data.make_assets(cfg, 0, "stationary", raw)
    stream = stationary["stream"]
    assert len(stream["ids"]) == cfg.stream_size
    assert not stream["h"].any() and not stream["p"].any() and not stream["novel"].any()
    assert len(stream["segments"]) == 4
    assert stationary["panels"]["novel"]["h"].min() >= .5
    assert stationary["panels"]["novel"]["h"].max() <= 1
    current = data.current_panel(stationary, cfg.num_p_steps, cfg)
    torch.testing.assert_close(current["transforms"], stationary["panels"]["current"]["transforms"],
                               rtol=0, atol=0)


def test_current_law_uses_own_sources_and_frozen_uniforms_without_mutation(raw, assets):
    cfg = Config.smoke_config()
    basis = assets["panels"]["current"]
    before = basis["transforms"].clone()
    initial = data.current_panel(assets, 0, cfg)
    peak_step = cfg.hold_steps + cfg.ramp_steps
    peak = data.current_panel(assets, peak_step, cfg)
    assert initial["h"] == 0 and peak["h"] == 1
    assert peak["p"] == cfg.novel_probability
    for name in ("ids", "digits", "uniforms", "mixture_uniforms"):
        torch.testing.assert_close(initial[name], peak[name], rtol=0, atol=0)
    for role in ("old", "novel"):
        assert not set(peak["ids"].flatten().tolist()) & set(assets["panels"][role]["ids"].flatten().tolist())
    assert torch.equal(peak["novel"], basis["mixture_uniforms"] < cfg.novel_probability)
    assert peak["novel"].any()
    torch.testing.assert_close(initial["transforms"], before, rtol=0, atol=0)
    torch.testing.assert_close(basis["transforms"], before, rtol=0, atol=0)
    torch.testing.assert_close(initial["transforms"][:, :, 1:], peak["transforms"][:, :, 1:], rtol=0, atol=0)
    end = data.current_panel(assets, cfg.num_p_steps, cfg)
    torch.testing.assert_close(end["transforms"], initial["transforms"], rtol=0, atol=0)
    assert data.current_panel(assets, peak_step, cfg)["hash"] == peak["hash"]


def test_renderer_preserves_cell_and_target_order_split_offset_and_cpu(raw, assets):
    assert data.render_pairs is inherited.render_pairs
    assert data.sample_transforms is inherited.sample_transforms
    assert data.support_mask is inherited.support_mask
    panel = assets["panels"]["old"]
    raw_before = raw["test_x"].clone()
    calls = []
    x, y = data.render_panel(raw, panel, indices=[0, 2], check=lambda: calls.append(1))
    assert len(calls) >= 3
    assert x.shape == (2, 2, 120, 120) and y.shape == (2, 54)
    assert x.dtype == y.dtype == torch.float32 and x.device.type == "cpu"
    assert x.min() >= 0 and x.max() <= 1
    assert x[:, 0].sum() > 0 and x[:, 1].sum() > 0
    selected = panel["ids"][[0, 2]] - len(raw["train_y"])
    matrices, translations = data.affine_from_parameters(panel["transforms"][[0, 2]])
    decoded_matrix, decoded_translation = decode_affine(y.double().reshape(2, 9, 6))
    torch.testing.assert_close(decoded_matrix, matrices, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(decoded_translation, translations, rtol=1e-6, atol=1e-6)
    for row in range(3):
        for col in range(3):
            cell = row * 3 + col
            expected = inherited.render_pairs(raw["test_x"][selected[:, cell]], matrices[:, cell],
                                              translations[:, cell])
            torch.testing.assert_close(x[:, :, row * 40:(row + 1) * 40, col * 40:(col + 1) * 40],
                                       expected, rtol=0, atol=0)
    torch.testing.assert_close(raw["test_x"], raw_before, rtol=0, atol=0)
    assert data.render_panel(raw, panel, 0)[0].shape == (1, 2, 120, 120)
    empty_x, empty_y = data.render_panel(raw, panel, slice(0, 0))
    assert empty_x.shape == (0, 2, 120, 120) and empty_y.shape == (0, 54)


def test_render_batch_preserves_all_four_consecutive_production_grid_positions(raw, assets):
    cfg = Config.smoke_config()
    step = 3
    first = data.render_batch(raw, assets, step)
    second = data.render_panel(raw, assets["stream"], slice(step * cfg.samples_per_step,
                                                           (step + 1) * cfg.samples_per_step))
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    cfg4 = replace(cfg, samples_per_step=4, nominal_arrival=4)
    assets4 = data.make_assets(cfg4, 0, "development", raw)
    four_x, four_y = data.render_batch(raw, assets4, 0)
    assert four_x.shape == (4, 2, 120, 120) and four_y.shape == (4, 54)


def test_nonrejecting_transform_law_matches_fixed_support_inverse_cdf():
    means = torch.tensor([-45., 0., 45.], dtype=torch.float64).repeat_interleave(101)
    uniforms = torch.rand(len(means), 6, generator=torch.Generator().manual_seed(15), dtype=torch.float64)
    uniforms[:, 0] = torch.linspace(0., 1 - 1e-12, len(means), dtype=torch.float64)
    values, info = data.sample_transforms(means, 16, uniforms=uniforms)
    normal = torch.distributions.Normal(torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
    lower, upper = normal.cdf((-70 - means) / 6), normal.cdf((70 - means) / 6)
    quantiles = (lower + uniforms[:, 0] * (upper - lower)).clamp(torch.finfo(torch.float64).eps,
                                                              1 - torch.finfo(torch.float64).eps)
    torch.testing.assert_close(values[:, 0], means + 6 * normal.icdf(quantiles), rtol=0, atol=0)
    assert values[:, 0].abs().max() <= 70 and values[:, 0].max() > 45
    assert values[:, 1:3].abs().max() <= 2
    assert values[:, 3:5].abs().max() <= math.log(1.1)
    assert values[:, 5].abs().max() <= .1
    assert info["rejected"] == 0 and info["proposed"] == len(means)
    assert info["nuisance_independence"].endswith("no rejection")


def test_full_support_extremes_corners_and_no_canvas_clipping():
    angles = (-70., -45., 0., 45., 70.)
    nuisance_corners = itertools.product((-2., 2.), (-2., 2.), (-math.log(1.1), math.log(1.1)),
                                         (-math.log(1.1), math.log(1.1)), (-.1, .1))
    nuisance = list(nuisance_corners)
    parameters = torch.tensor([(angle, *corner) for angle in angles for corner in nuisance], dtype=torch.float64)
    matrix, translation = data.affine_from_parameters(parameters)
    assert data.support_mask(matrix, translation).all()
    corners = torch.tensor(data.SUPPORT_CORNERS, dtype=torch.float64)
    mapped = corners @ matrix.transpose(-1, -2) + translation[:, None]
    assert mapped.abs().max() < 19.
    # An all-one raw image populates every corner of the actual resized support.
    raw = torch.ones(len(parameters), 1, 28, 28)
    pairs = data.render_pairs(raw, matrix, translation)
    assert pairs.shape == (len(parameters), 2, 40, 40)
    assert pairs[:, 1].sum((1, 2)).min() > 150
    assert pairs[:, 1, (0, -1), :].abs().max() == 0
    assert pairs[:, 1, :, (0, -1)].abs().max() == 0
    # Compare against a larger canvas at the same physical center, not a bound alone.
    q = torch.arange(64, dtype=torch.float64) - 31.5
    yy, xx = torch.meshgrid(q, q, indexing="ij")
    pixels = torch.stack((xx, yy), -1).reshape(1, -1, 2)
    grid = ((pixels - translation[:, None]) @ torch.linalg.inv(matrix).transpose(1, 2) / 32).reshape(-1, 64, 64, 2)
    reference = F.pad(torch.ones(len(parameters), 1, 18, 18), (23,) * 4)
    wider = F.grid_sample(reference, grid.float(), mode="bilinear", align_corners=False)
    torch.testing.assert_close(pairs[:, 1], wider[:, 0, 12:52, 12:52], rtol=1e-5, atol=4e-6)
    torch.testing.assert_close(pairs[:, 1].sum((1, 2)), wider[:, 0].sum((1, 2)), rtol=1e-6, atol=1e-4)


def test_identity_translation_and_clockwise_rotation_forward_labels():
    images = torch.zeros(3, 1, 28, 28)
    images[:, :, 5:10, 16:22] = 1
    parameters = torch.tensor([[0., 0., 0., 0., 0., 0.], [0., 2., -2., 0., 0., 0.],
                               [45., 0., 0., 0., 0., 0.]], dtype=torch.float64)
    matrix, translation = data.affine_from_parameters(parameters)
    pairs = data.render_pairs(images, matrix, translation)
    torch.testing.assert_close(pairs[0, 0], pairs[0, 1], rtol=0, atol=2e-6)
    torch.testing.assert_close(pairs[1, 1, :-2, 2:], pairs[1, 0, 2:, :-2], rtol=0, atol=2e-6)
    targets = data.encode_affine(matrix, translation).reshape(3, 3, 2)
    assert torch.equal(targets[0], torch.zeros_like(targets[0]))
    torch.testing.assert_close(targets[1], translation[1].expand(3, 2))
    assert matrix[2, 1, 0] > 0
    anchors = torch.tensor(data.ANCHORS, dtype=torch.float64)
    torch.testing.assert_close(targets[2], anchors @ matrix[2].T - anchors)


def test_data_validation_capacity_wrong_split_and_interruptions(raw, assets):
    cfg = Config.smoke_config()
    with pytest.raises(ValueError, match="disjoint training sources"):
        data.make_assets(replace(cfg, init_size=1000, initialization_size=1000), 0, "development", raw)
    with pytest.raises(ValueError, match="disjoint test sources"):
        data.make_assets(replace(cfg, eval_size=7), 0, "development", raw)
    with pytest.raises(ValueError):
        data.make_assets(cfg, -1, "development", raw)
    with pytest.raises(ValueError):
        data.make_assets(cfg, 0, "unknown", raw)
    with pytest.raises(ValueError, match="declared split"):
        data.render_panel(raw, dict(assets["panels"]["old"], test=False))
    invalid = dict(assets["init"], transforms=assets["init"]["transforms"].clone())
    invalid["transforms"][0, 0, 1] = 100
    with pytest.raises(ValueError, match="geometrically invalid"):
        data.render_panel(raw, invalid)
    with pytest.raises(ValueError):
        data.render_batch(raw, assets, -1)
    with pytest.raises(ValueError):
        data.current_panel(assets, cfg.num_p_steps + 1, cfg)

    class Interrupted(Exception):
        pass

    def stop():
        raise Interrupted

    with pytest.raises(Interrupted):
        data.render_panel(raw, assets["init"], check=stop)
