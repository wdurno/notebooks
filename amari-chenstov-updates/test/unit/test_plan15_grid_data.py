"""Synthetic-source CPU contracts; never download MNIST or train a model."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.fisher_memory_registration import data as cell_data
from mnist_experiment.fisher_memory_grid_registration import PROTOCOL, data
from mnist_experiment.fisher_memory_grid_registration.config import Config


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class CachedSource(data.MNISTSource):
    def __init__(self, train_count=1800, test_count=400):
        self.train_labels = torch.arange(train_count) % 10
        self.test_labels = torch.arange(test_count) % 10

    def images(self, ids, *, test=False):
        images = torch.zeros(len(ids), 1, 28, 28)
        images[:, :, 4:23, 11:15] = .8
        images[:, :, 8:12, 14:23] = 1.
        images[:, :, 18:23, 8:17] = .6
        images[:, :, 12:15, 16:19] = ((torch.tensor(ids) % 250 + 1) / 255)[:, None, None, None]
        return images


@pytest.fixture
def source():
    return CachedSource()


@pytest.fixture
def cfg():
    return Config.smoke_config()


def flattened(rows):
    return [item for row in rows for item in row]


def test_grid_assembly_offsets_forward_direction_and_local_target_order():
    cells = torch.arange(18, dtype=torch.float64).reshape(2, 9, 1, 1, 1).expand(-1, -1, 2, 28, 28)
    grid = data.assemble_grid(cells)
    assert grid.shape == (2, 2, 84, 84) and grid.is_contiguous()
    assert torch.equal(data.split_grid(grid), cells)
    for cell, (x, y) in enumerate(data.CELL_OFFSETS):
        assert torch.equal(grid[:, :, y:y + 28, x:x + 28], cells[:, cell])
    parameters = torch.zeros(2, 9, 6, dtype=torch.float64)
    parameters[:, :, 1] = torch.arange(9)
    parameters[:, :, 2] = -torch.arange(9)
    targets = data.target_tensor(parameters, dtype=torch.float64)
    assert targets.shape == (2, 54)
    expected = parameters[:, :, 1:3, None].transpose(-1, -2).expand(-1, -1, 3, -1)
    torch.testing.assert_close(targets.reshape(2, 9, 3, 2), expected)
    matrix, translation = data.decode_affine(targets)
    torch.testing.assert_close(matrix, torch.eye(2, dtype=torch.float64).expand(2, 9, 2, 2))
    torch.testing.assert_close(translation, parameters[:, :, 1:3])
    assert targets[0, :6].count_nonzero() == 0
    parameters[:] = 0
    parameters[:, :, 0] = 90
    rotation, _ = data.affine_from_parameters(parameters)
    torch.testing.assert_close(rotation[0, 0] @ torch.tensor([1., 0.], dtype=torch.float64),
                               torch.tensor([0., 1.], dtype=torch.float64))
    arbitrary = torch.arange(54, dtype=torch.float64).reshape(1, 54) * 10
    torch.testing.assert_close(data.encode_affine(*data.decode_affine(arbitrary)).reshape(1, 54), arbitrary)
    empty = torch.empty(0, 9, 2, 28, 28)
    assert data.assemble_grid(empty).shape == (0, 2, 84, 84)


def test_joint_targets_have_full_rank_and_independent_cell_draws():
    draws, diagnostics = data._grid_transforms(1024, 7)
    values = torch.tensor(draws, dtype=torch.float64)
    assert values.shape == (1024, 9, 6)
    assert bool(data.support_mask(*data.affine_from_parameters(values)).all())
    assert not torch.equal(values[:, 0], values[:, 1])
    targets = data.target_tensor(values, dtype=torch.float64)
    assert int(torch.linalg.matrix_rank(targets - targets.mean(0))) == 54
    torch.testing.assert_close(torch.tensor(diagnostics["target_covariance"], dtype=torch.float64),
                               torch.cov(targets.T))
    assert diagnostics["accepted"] == diagnostics["accepted_cells"] == 9 * 1024
    assert diagnostics["accepted_grids"] == 1024 and diagnostics["rejected"] > 0
    assert diagnostics["proposed"] == diagnostics["accepted"] + diagnostics["rejected"]
    assert draws[:3] == data._grid_transforms(3, 7)[0]
    assert abs(float(torch.corrcoef(values[:, :2, 1].T)[0, 1])) < .1


def test_plan_pairing_disjoint_roles_seeds_and_immutable_choices(source, cfg, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("planning cannot render, download, or call the historical planner")

    monkeypatch.setattr(source, "images", forbidden)
    monkeypatch.setattr(cell_data, "plan_data", forbidden)
    state = torch.random.get_rng_state().clone()
    plan = data.plan_data(source, cfg, "development--grid", 3, "development")
    assert plan == json.loads(json.dumps(plan, allow_nan=False))
    assert plan["protocol"] == PROTOCOL and plan["schema_version"] == data.SCHEMA_VERSION
    assert plan["family"] == "development"
    all_ids = [item for role in data.TRAIN_ROLES for item in flattened(plan[f"{role}_ids"])]
    assert len(all_ids) == len(set(all_ids))
    reserved = [item for role in data.TRAIN_ROLES if role != "stream"
                for item in flattened(plan[f"{role}_ids"])]
    assert not set(reserved) & set(flattened(plan["stream_pool_ids"]))
    assert plan["stream_ids"] == plan["stream_pool_ids"][:len(plan["stream_ids"])]
    for role in data.TRAIN_ROLES:
        ids = torch.tensor(plan[f"{role}_ids"])
        assert ids.ndim == 2 and ids.shape[1] == 9
        assert plan[f"{role}_labels"] == source.train_labels[ids].tolist()
        assert len(plan[f"{role}_transforms"]) == len(ids)
    assert len(plan["reference_ids"]) == cfg.reference_max_size
    assert len(plan["diagnostic_reserve_ids"]) == cfg.diagnostic_reserve_size
    assert plan["source_sampling"]["class_draw"] == "remaining_source_count_weighted"
    flat_seeds = [item for value in plan["seeds"].values()
                  for item in (value if isinstance(value, list) else [value])]
    assert len(flat_seeds) == len(set(flat_seeds))
    later = replace(cfg, reference_size=cfg.reference_max_size, channels=(4, 8), blur_max=2.4)
    assert plan == data.plan_data(source, later, "development--grid--later-policy", 3, "development")
    assert torch.equal(state, torch.random.get_rng_state())


def test_synthetic_production_source_budget_and_test_panels_without_pixels():
    source = CachedSource(60_000, 10_000)
    cfg = Config()
    plan = data.plan_data(source, cfg, "budget-test", 0, "development")
    counts = [len(plan[f"{role}_ids"]) for role in data.TRAIN_ROLES if role != "stream"]
    assert counts == [3000, 200, 256, 64, 32, 128]
    assert plan["stream_pool_size"] == 2986
    reserved = [item for role in data.TRAIN_ROLES if role != "stream"
                for item in flattened(plan[f"{role}_ids"])] + flattened(plan["stream_pool_ids"])
    assert len(reserved) == len(set(reserved)) == 59_994
    assert all(len(plan["evaluation_ids"][domain]) == 128 for domain in data.DOMAINS)


def test_exhaustion_retains_unfavorable_branches_and_never_exhausts_class_bucket(cfg):
    source = CachedSource(360)
    source.train_labels[:] = 9
    source.train_labels[:9] = torch.arange(9)
    cfg = replace(cfg, novel_count=100, novel_probability=1e-12)
    plan = data.plan_data(source, cfg, "exhaustion", 0, "development")
    assert not plan["stream_complete"] and plan["stream_status"] == "source_pool_exhausted"
    assert plan["stream_novel_count"] == 0
    assert len(plan["stream_ids"]) == plan["stream_pool_size"] == 19
    assert not any(plan["stream_novel"])
    source_ids = [item for role in data.TRAIN_ROLES for item in flattened(plan[f"{role}_ids"])]
    assert len(source_ids) == len(set(source_ids)) == 360
    assert sorted(source.train_labels[source_ids].tolist()) == sorted(source.train_labels.tolist())
    assert plan == data.plan_data(source, cfg, "exhaustion", 0, "development")
    other = data.plan_data(source, replace(cfg, novel_probability=1.), "exhaustion", 0, "development")
    assert other["stream_pool_ids"] == plan["stream_pool_ids"]
    assert other["stream_ids"] == plan["stream_ids"]


def test_exact_novel_segments_and_original_group_schedules(source, cfg):
    plan = data.plan_data(source, cfg, "development", 0, "development")
    assert plan["stream_complete"] and sum(plan["stream_novel"]) == cfg.novel_count
    assert plan["stream_novel"][-1] and not any(plan["stream_novel"][:cfg.nominal_batch])
    previous, novel_count = 0, 0
    for segment in plan["segments"]:
        start, end, group = segment["start"], segment["end"], segment["group"]
        assert previous == start and 1 <= end - start <= cfg.nominal_batch
        assert start // cfg.nominal_batch == (end - 1) // cfg.nominal_batch == group - 1
        assert segment["probability"] == data.mixture_probability(group, cfg.novel_probability, cfg.ramp_groups)
        assert segment["strength"] == data.drift_strength(group)
        assert sum(plan["stream_novel"][start:end]) <= 1
        for i in range(start, end):
            novel = plan["stream_novel"][i]
            assert plan["stream_groups"][i] == group
            assert plan["stream_strengths"][i] == (data.drift_strength(group) if novel else 0.)
            novel_count += int(novel)
        assert segment["novel_count"] == novel_count
        previous = end
    assert previous == len(plan["stream_ids"])
    assert data.mixture_probability(1) == 0 and data.mixture_probability(40) == .1
    assert data.drift_strength(1) == .5 and data.drift_strength(200) == 1.


def test_panels_are_disjoint_with_frozen_test_partitions(source, cfg):
    pools = {partition: set() for partition in ("development", "confirmation")}
    for replica in (0, 1):
        for partition in pools:
            plan = data.plan_data(source, cfg, partition, replica, partition)
            base, novel = (flattened(plan["evaluation_ids"][domain]) for domain in data.DOMAINS)
            assert len(base) == len(set(base)) == 9 * cfg.eval_size
            assert len(novel) == len(set(novel)) == 9 * cfg.eval_size
            assert not set(base) & set(novel)
            pools[partition].update(base + novel)
            assert all(.5 <= strength <= 1 for strength in plan["evaluation_strengths"]["novel"])
    assert not pools["development"] & pools["confirmation"]


def test_materialize_exact_cell_helpers_context_sharing_and_reserve(source, cfg, monkeypatch):
    plan = data.plan_data(source, cfg, "development", 0, "development")
    frozen = json.loads(json.dumps(plan))
    state = torch.random.get_rng_state().clone()
    nominal = data.materialize(source, cfg, plan)
    assert nominal["pretrain_x"] is nominal["initialization_x"]
    assert nominal["pretrain_y"] is nominal["initialization_y"]
    assert len(nominal["reference_x"]) == cfg.reference_max_size
    assert len(nominal["diagnostic_reserve_x"]) == cfg.diagnostic_reserve_size
    for role in data.TRAIN_ROLES:
        parameters = torch.tensor(plan[f"{role}_transforms"], dtype=torch.float64)
        raw = source.images(flattened(plan[f"{role}_ids"]))
        matrix, translation = data.affine_from_parameters(parameters.reshape(-1, 6))
        expected = cell_data.render_pairs(raw, matrix, translation)
        expected = data.assemble_grid(expected.reshape(-1, 9, 2, 28, 28))
        assert torch.equal(nominal[f"{role}_x"], expected)
        assert torch.equal(nominal[f"{role}_y"], data.target_tensor(parameters))
    for panel in nominal["panels"].values():
        assert panel["digits"].shape == (cfg.eval_size, 9)
        assert torch.equal(panel["weights"], torch.full((cfg.eval_size,), 1 / cfg.eval_size, dtype=torch.float64))

    def forbidden(*args, **kwargs):
        pytest.fail("context selection must not access source pixels")

    monkeypatch.setattr(source, "images", forbidden)
    actual = data.contextualize(nominal, cfg, 1.8)
    flags = torch.tensor(plan["stream_novel"], dtype=torch.bool)
    strengths = torch.tensor(plan["stream_strengths"], dtype=torch.float64)
    expected = cell_data.blur_pairs(data.split_grid(nominal["stream_x"])[flags].reshape(-1, 2, 28, 28),
        strengths[flags].repeat_interleave(9), 1.8).reshape(-1, 9, 2, 28, 28)
    assert torch.equal(actual["stream_x"][flags], data.assemble_grid(expected))
    assert torch.equal(actual["stream_x"][~flags], nominal["stream_x"][~flags])
    assert torch.equal(actual["stream_x"][:, 0], nominal["stream_x"][:, 0])
    for role in data.TRAIN_ROLES:
        assert actual[f"{role}_y"] is nominal[f"{role}_y"]
    for key in ("initialization_x", "pretrain_x", "validation_x", "reference_x", "diagnostic_reserve_x"):
        assert actual[key] is nominal[key]
    assert actual["panels"]["base"] is nominal["panels"]["base"]
    assert actual["panels"]["novel"]["y"] is nominal["panels"]["novel"]["y"]
    with pytest.raises(ValueError, match="nominal cache"):
        data.contextualize(actual, cfg, 2.4)
    assert plan == frozen and torch.equal(state, torch.random.get_rng_state())


def test_blur_cannot_leak_across_cell_boundaries():
    cells = torch.zeros(1, 9, 2, 28, 28, dtype=torch.float64)
    cells[0, 0, 1, 14, 27] = 1
    cells[0, :, 0] = .25
    grid = data.assemble_grid(cells)
    result = data.blur_pairs(grid, .8, 1.8)
    blurred = data.split_grid(result)
    assert blurred[0, 0, 1].sum() > 0
    assert blurred[0, 1:, 1].count_nonzero() == 0
    assert torch.equal(result[:, 0], grid[:, 0])
    assert data.blur_pairs(grid, 0., 1.8) is grid
    assert data.blur_pairs(grid, .8, 0.) is grid


def test_geometry_averages_27_landmarks_not_54_coordinates_or_nine_cells():
    prediction = torch.zeros(2, 54, dtype=torch.float64)
    prediction[0] = torch.tensor([3., 4.]).repeat(27)
    prediction[1] = torch.tensor([0., 1.]).repeat(27)
    digits = torch.arange(18).reshape(2, 9) % 10
    result = data.geometry_metrics(prediction, torch.zeros_like(prediction), digits=digits)
    assert result["mse"] == 13 and result["rmse"] == pytest.approx(13 ** .5)
    assert result["mae"] == 3 and result["accuracy1"] == .5
    assert result["per_cell_mse"] == [13.] * 9
    assert result["per_anchor_mse"] == [[13.] * 3] * 9
    assert result["coordinate_mse"] == [4.5, 8.5] * 27
    weighted = data.geometry_metrics(prediction, torch.zeros_like(prediction), weights=torch.tensor([1., 0.]))
    assert weighted["mse"] == 25.


def test_cooperative_stop_chunk_invariance_and_schema_validation(source, cfg, monkeypatch):
    plan = data.plan_data(source, cfg, "development", 0, "development")
    expected = data.materialize(source, cfg, plan)
    monkeypatch.setattr(data, "TRANSFORM_CHUNK", 1)
    actual = data.materialize(source, cfg, plan)
    for role in data.TRAIN_ROLES:
        torch.testing.assert_close(actual[f"{role}_x"], expected[f"{role}_x"], rtol=0, atol=0)

    def stop():
        raise InterruptedError("halt")

    def forbidden(*args, **kwargs):
        pytest.fail("stopping must precede source access")

    monkeypatch.setattr(source, "images", forbidden)
    with pytest.raises(InterruptedError, match="halt"):
        data.materialize(source, cfg, plan, check_stop=stop)
    with pytest.raises(InterruptedError, match="halt"):
        data.contextualize(expected, cfg, 1.8, check_stop=stop)
    with pytest.raises(ValueError, match="schema"):
        data.materialize(source, cfg, dict(plan, protocol="plan15-registration-v1"))
    with pytest.raises(ValueError, match="partition"):
        data.materialize(source, cfg, dict(plan, partition="typo"))


def test_invalid_grid_configuration_and_shapes_fail_clearly(source, cfg):
    with pytest.raises(ValueError, match="insufficient disjoint"):
        data.plan_data(source, replace(cfg, init_size=10_000), "development", 0, "development")
    with pytest.raises(ValueError, match="insufficient disjoint"):
        data.plan_data(source, replace(cfg, eval_size=100), "development", 0, "development")
    with pytest.raises(ValueError, match="reference reserve"):
        data.plan_data(source, replace(cfg, reference_size=100), "development", 0, "development")
    with pytest.raises(ValueError, match="partition"):
        data.plan_data(source, cfg, "development", 0, "typo")
    with pytest.raises(ValueError, match="positive integer"):
        data.plan_data(source, SimpleNamespace(**(vars(cfg) | {"eval_size": True})), "development", 0, "development")
    with pytest.raises(ValueError, match="9,6"):
        data.target_tensor(torch.zeros(3, 6))
    grid = torch.zeros(1, 2, 84, 84)
    for strengths in ([.3, .4], [-1.], [1.1], [float("nan")]):
        with pytest.raises(ValueError, match="strengths"):
            data.blur_pairs(grid, strengths, 1.8)
    for maximum in (-1., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="blur_max"):
            data.blur_pairs(grid, .8, maximum)
