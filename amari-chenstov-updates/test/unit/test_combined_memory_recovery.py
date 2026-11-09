"""Four-arrival real-model recovery, isolated from production stores and budgets."""

from dataclasses import replace
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from mnist_experiment.combined_memory import curvature, trajectory
from mnist_experiment.combined_memory.config import Config, Recipe
from mnist_experiment.combined_memory.data import make_assets, render_panel
from mnist_experiment.combined_memory.model import build_model, memory_indices, merge_parameters
from mnist_experiment.combined_memory.runtime import setup
from mnist_experiment.fisher_memory.artifacts import ImmutableError, Store as BaseStore
from mnist_experiment.fisher_memory.control import StopRequested


UNIT = "tiny-trajectory"
IDENTITY = {"protocol": "toy-plan18-recovery", "schema_version": 1}


class DummyBudget:
    """A local safe-callback stop, without touching production-family accounting."""

    def __init__(self):
        self.stop = False
        self.calls = 0

    def check(self):
        self.calls += 1
        if self.stop:
            raise StopRequested("test stop with exactly one accepted batch")


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


@pytest.fixture(scope="module")
def raw():
    train_y, test_y = torch.arange(10).repeat(60), torch.arange(10).repeat(30)
    train_x = torch.zeros(600, 1, 28, 28)
    test_x = torch.zeros(300, 1, 28, 28)
    train_x[:, :, 5:23, 10:17] = torch.linspace(.25, .95, 600)[:, None, None, None]
    test_x[:, :, 6:22, 9:18] = torch.linspace(.3, 1., 300)[:, None, None, None]
    return dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


@pytest.fixture(scope="module")
def experiment(raw):
    cfg = replace(Config.smoke_config(device=os.environ.get("PLAN18_TEST_DEVICE", "cpu")),
                  hold_steps=1, ramp_steps=1, capacity=2,
                  max_iter=2, max_eval=6, analysis_novel_count=1)
    cfg.validate()
    setup(cfg)
    assets = make_assets(cfg, 0, "development", raw)
    model_seed = assets["seeds"]["initialization_model"]
    factor_seed = assets["seeds"]["factors"]
    net, layout = build_model(model_seed, device="cpu", dtype=torch.float64, factor_seed=factor_seed)
    theta = layout.flatten_module(net, detach=True)
    ltm, _ = memory_indices(layout)
    _, _, merged_theta = merge_parameters(net, layout, theta)
    initial = dict(theta=theta, diagonal=torch.ones(len(ltm), dtype=torch.float64),
                   variance=torch.ones(len(ltm), dtype=torch.float64), merged_anchor=merged_theta)
    metadata = dict(cohort="development", replica=0, model_seed=model_seed, factor_seed=factor_seed)
    assert cfg.num_p_steps == 4 and cfg.init_size == cfg.initialization_size == 8
    return cfg, assets, initial, metadata


def store(root, *, resume=False, identity=None):
    return BaseStore(root, IDENTITY if identity is None else identity, resume=resume,
                     max_bytes=32 * 1024**2, free_reserve_bytes=0,
                     finalization_reserve_bytes=64 * 1024)


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items()
                if not str(key).endswith("_seconds")}
    if isinstance(value, (list, tuple)):
        return type(value)(without_timing(item) for item in value)
    return value


def assert_tree_equal(actual, expected, path="state"):
    assert type(actual) is type(expected), path
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
    elif isinstance(actual, np.ndarray):
        assert np.array_equal(actual, expected), path
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            assert_tree_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (first, second) in enumerate(zip(actual, expected)):
            assert_tree_equal(first, second, f"{path}[{index}]")
    else:
        assert actual == expected, path


def snapshot_unit(path):
    return {file.relative_to(path).as_posix():
            (hashlib.sha256(file.read_bytes()).hexdigest(), file.stat().st_size, file.stat().st_mtime_ns)
            for file in path.rglob("*") if file.is_file()}


def run(storage, cfg, assets, initial, metadata, raw, budget=None, **kwargs):
    budget = budget or DummyBudget()
    reader = trajectory.Reader(raw, assets, cfg, check=budget.check)
    return trajectory.run_trajectory(storage, budget, cfg, UNIT, metadata, initial, reader, **kwargs)


@pytest.mark.parametrize("recipe", [Recipe("A"), Recipe("E", strength=.1)],
                         ids=["fixed-replay", "adaptive-transport"])
@pytest.mark.parametrize("stop_at", ["after_first_commit", "before_second_publish"])
def test_accepted_batch_recovery_matches_uninterrupted_scientific_state(
        tmp_path, monkeypatch, experiment, raw, recipe, stop_at):
    cfg, assets, initial, metadata = experiment
    metadata = dict(metadata, recipe=recipe.mapping())
    initial_before = {name: value.clone() for name, value in initial.items()}
    with store(tmp_path / "uninterrupted") as storage:
        uninterrupted = run(storage, cfg, assets, initial, metadata, raw)
        assert storage.is_complete(UNIT)
        expected_metrics = json.loads((storage.unit_path(UNIT) / "metrics.json").read_text())
        expected_diagnostics = json.loads((storage.unit_path(UNIT) / "diagnostics.json").read_text())

    budget = DummyBudget()
    committed = []
    with store(tmp_path / "resumed") as storage:
        save_checkpoint = storage.save_checkpoint

        def stop_before_second_checkpoint(uid, payload, **kwargs):
            if stop_at == "before_second_publish" and payload["step"] == 2:
                assert committed == [1]
                budget.stop = True
                budget.check()
            return save_checkpoint(uid, payload, **kwargs)

        monkeypatch.setattr(storage, "save_checkpoint", stop_before_second_checkpoint)

        def accepted(**progress):
            committed.append(progress["batch"])
            if stop_at == "after_first_commit" and progress["batch"] == 1:
                budget.stop = True

        with pytest.raises(StopRequested, match="exactly one accepted batch"):
            run(storage, cfg, assets, initial, metadata, raw, budget, pulse=accepted)
        assert committed == [1]
        assert not storage.is_complete(UNIT)
        saved = storage.load_checkpoint(UNIT)
        assert saved["step"] == 1
        assert float((saved["theta"] - initial["theta"]).norm()) > 1e-10
        assert saved["diagnostics"][0]["moved"]
        assert saved["replay"] == [0, 1] and saved["retired_ids"] == []
        assert saved["q_archive"] == 1 / cfg.init_size
        assert len(saved["diagnostics"]) == 1
        assert [row["batch"] for row in saved["diagnostics"]] == [1]
        assert max(row["accepted_step"] for row in saved["metrics"]) == 1
        torch.testing.assert_close(saved["anchor"], initial["theta"][:3268], rtol=0, atol=0)
        assert float((saved["theta"][:3268] - saved["anchor"]).norm()) > 1e-10
        torch.testing.assert_close(saved["curvature_reference"], saved["theta"], rtol=0, atol=0)
        if recipe.adaptive:
            assert saved["controller"]["assimilations"] == 1
            assert saved["controller"]["noise_updates"] == 1
        else:
            assert saved["controller"] is None

    # Reopen the store and construct a fresh reader/budget, as a new process does.
    with store(tmp_path / "resumed", resume=True) as storage:
        resumed = run(storage, cfg, assets, initial, metadata, raw)
        assert storage.is_complete(UNIT)
        persisted = storage.load_checkpoint(UNIT)
        metrics = json.loads((storage.unit_path(UNIT) / "metrics.json").read_text())
        diagnostics = json.loads((storage.unit_path(UNIT) / "diagnostics.json").read_text())
    assert_tree_equal(without_timing(resumed), without_timing(uninterrupted))
    assert_tree_equal(without_timing(persisted), without_timing(uninterrupted))
    assert_tree_equal(without_timing(metrics), without_timing(expected_metrics), "metrics export")
    assert_tree_equal(without_timing(diagnostics), without_timing(expected_diagnostics), "diagnostics export")
    assert_tree_equal(initial, initial_before, "shared initial state")
    assert resumed["step"] == 4
    assert float((resumed["theta"] - initial["theta"]).norm()) > 1e-10
    assert any(row["moved"] for row in resumed["diagnostics"])
    assert resumed["replay"] == [6, 7]
    assert resumed["retired_ids"] == list(range(6))
    assert len(set(resumed["retired_ids"])) == 6
    assert set(resumed["replay"]).isdisjoint(resumed["retired_ids"])
    assert len(resumed["diagnostics"]) == 4
    assert [row["batch"] for row in resumed["diagnostics"]] == [1, 2, 3, 4]
    assert sum(row["evicted_count"] for row in resumed["diagnostics"]) == 6
    assert bool(torch.isfinite(resumed["diagonal"]).all()) and bool((resumed["diagonal"] >= 0).all())
    assert not torch.equal(resumed["diagonal"], initial["diagonal"])
    torch.testing.assert_close(resumed["anchor"], resumed["theta"][:3268], rtol=0, atol=0)
    assert float((resumed["anchor"] - initial["theta"][:3268]).norm()) > 1e-10
    torch.testing.assert_close(resumed["curvature_reference"], resumed["theta"], rtol=0, atol=0)
    if recipe.transport:
        # This nonlinear encoder moves, so transport on retired inputs must be exercised.
        assert any(row["evicted_count"] and row["correction_norm"] > 1e-10
                   for row in resumed["diagnostics"])
    q_ref = q_archive = 1 / cfg.init_size
    for diagnostic in resumed["diagnostics"]:
        pi, eta, retired = diagnostic["applied_pi"], diagnostic["eta"], diagnostic["evicted_count"]
        q_ref = (1 - pi)**2 * q_ref + pi**2 / cfg.samples_per_step
        if retired:
            q_archive = (1 - eta)**2 * q_archive + eta**2 / retired
    assert resumed["q_ref"] == q_ref and resumed["q_archive"] == q_archive
    assert resumed["q_ref"] != resumed["q_archive"]
    if recipe.adaptive:
        assert resumed["controller"]["assimilations"] == resumed["controller"]["noise_updates"] == 4
        assert bool(resumed["controller"]["mean"].abs().sum() > 0)
        assert bool(torch.isfinite(resumed["controller"]["covariance"]).all())
    else:
        assert resumed["controller"] is None


def test_completed_unit_resume_skips_training_and_rejects_identity_or_write_collisions(
        tmp_path, monkeypatch, experiment, raw):
    cfg, assets, initial, metadata = experiment
    metadata = dict(metadata, recipe=Recipe("E", strength=.1).mapping())
    root = tmp_path / "completed"
    with store(root) as storage:
        expected = run(storage, cfg, assets, initial, metadata, raw)
        assert float((expected["theta"] - initial["theta"]).norm()) > 1e-10
        assert any(row["correction_norm"] > 1e-10 for row in expected["diagnostics"])
        unit_path = storage.unit_path(UNIT)
        assert storage.is_complete(UNIT)
        sealed = snapshot_unit(unit_path)
    assert "COMPLETED" in sealed and "latest.json" in sealed

    def forbidden(*args, **kwargs):
        pytest.fail("completed-unit resume attempted scientific work or an artifact write")

    with store(root, resume=True) as storage:
        for name in ("build_model", "merge_parameters", "fit_step", "fisher_diagonal",
                     "paired_change", "observed_scores", "evaluate_panel"):
            monkeypatch.setattr(trajectory, name, forbidden)
        monkeypatch.setattr(trajectory.Reader, "panel", forbidden)
        monkeypatch.setattr(trajectory.Reader, "stream", forbidden)
        monkeypatch.setattr(trajectory.Reader, "current", forbidden)
        with monkeypatch.context() as completed_guard:
            completed_guard.setattr(storage, "save_checkpoint", forbidden)
            completed_guard.setattr(storage, "write_json", forbidden)
            completed_guard.setattr(storage, "complete", forbidden)
            for _ in range(2):
                loaded = run(storage, cfg, assets, initial, metadata, raw)
                assert_tree_equal(loaded, expected)
        assert snapshot_unit(unit_path) == sealed
        assert storage.begin(UNIT, metadata) is None
        with pytest.raises(ImmutableError):
            storage.begin(UNIT, dict(metadata, replica=1))
        with pytest.raises(ImmutableError):
            storage.save_checkpoint(UNIT, {"step": 0})
        with pytest.raises(ImmutableError):
            storage.write_json(UNIT, "metrics.json", [])
        with pytest.raises(ImmutableError):
            storage.complete(UNIT)
        assert snapshot_unit(unit_path) == sealed
    assert snapshot_unit(unit_path) == sealed
    with pytest.raises(ImmutableError, match="already exists"):
        store(root)
    with pytest.raises(ImmutableError, match="identity changed"):
        store(root, resume=True, identity=dict(IDENTITY, source_revision="different"))
    assert snapshot_unit(unit_path) == sealed


@pytest.mark.skipif(os.environ.get("PLAN18_TEST_DEVICE", "cpu") != "cuda",
                    reason="explicit PLAN18_TEST_DEVICE=cuda acceptance only")
def test_float64_curvature_cpu_cuda_parity(experiment, raw):
    assert torch.cuda.is_available(), "CUDA acceptance was requested but no CUDA device is available"
    cfg, assets, initial, metadata = experiment
    cpu_model, layout = build_model(metadata["model_seed"], device="cpu", dtype=torch.float64,
                                    factor_seed=metadata["factor_seed"])
    cuda_model, cuda_layout = build_model(metadata["model_seed"], device="cuda", dtype=torch.float64,
                                          factor_seed=metadata["factor_seed"])
    theta = initial["theta"]
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed_for("development", 0, "numerical"))
    moved = theta + .01 * torch.randn(theta.shape, generator=generator, dtype=torch.float64)
    x, y = (value.double() for value in render_panel(raw, assets["reference"], [0, 1]))
    selected = torch.tensor([0, 75, 100, 297, 298, 3214, 3268, 3375])
    cpu_delta, cpu_fresh, _ = curvature.paired_change(cpu_model, layout, theta, moved, x,
                                                     indices=selected)
    cuda_delta, cuda_fresh, _ = curvature.paired_change(cuda_model, cuda_layout, theta.cuda(),
        moved.cuda(), x.cuda(), indices=selected)
    torch.testing.assert_close(cuda_delta.cpu(), cpu_delta, rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(cuda_fresh.cpu(), cpu_fresh, rtol=1e-6, atol=1e-10)
    cpu_scores = curvature.observed_scores(cpu_model, layout, moved, x, y, indices=selected)
    cuda_scores = curvature.observed_scores(cuda_model, cuda_layout, moved.cuda(), x.cuda(), y.cuda(),
                                             indices=selected)
    torch.testing.assert_close(cuda_scores.cpu(), cpu_scores, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("stop_at", ["after_minibatch", "during_reference"])
def test_initialization_recovery_preserves_adam_and_reference_accumulators(
        tmp_path, monkeypatch, experiment, raw, stop_at):
    cfg, assets, _, metadata = experiment
    uid = "tiny-initialization"
    with store(tmp_path / "whole-initial") as storage:
        expected = trajectory.initial_fit(storage, DummyBudget(), cfg, uid, metadata,
            trajectory.Reader(raw, assets, cfg))
        assert storage.is_complete(uid)

    budget = DummyBudget()
    with store(tmp_path / "resume-initial") as storage:
        def accepted(**progress):
            if ((stop_at == "after_minibatch" and progress["epoch"] == 1
                    and progress["batch"] == cfg.init_batch)
                    or (stop_at == "during_reference" and progress["epoch"] == "Fisher"
                        and progress["batch"] == cfg.fisher_chunk)):
                budget.stop = True

        with pytest.raises(StopRequested):
            trajectory.initial_fit(storage, budget, cfg, uid, metadata,
                trajectory.Reader(raw, assets, cfg, check=budget.check), pulse=accepted)
        assert not storage.is_complete(uid)
        checkpoint = storage.load_checkpoint(uid)
        if stop_at == "after_minibatch":
            assert checkpoint["phase"] == "fit"
            assert checkpoint["epoch"] == 0 and checkpoint["offset"] == cfg.init_batch
            assert checkpoint["optimizer"]["state"]
        else:
            assert checkpoint["phase"] == "fisher"
            assert checkpoint["reference_offset"] == checkpoint["score_count"] == cfg.fisher_chunk
            assert checkpoint["reference_sum"].sum() > 0

    with store(tmp_path / "resume-initial", resume=True) as storage:
        actual = trajectory.initial_fit(storage, DummyBudget(), cfg, uid, metadata,
            trajectory.Reader(raw, assets, cfg))
        assert storage.is_complete(uid)
        assert_tree_equal(without_timing(actual), without_timing(expected), "initialization")
        sealed = snapshot_unit(storage.unit_path(uid))
        monkeypatch.setattr(trajectory, "build_model", lambda *a, **kw:
            pytest.fail("completed initialization must not construct or train a model"))
        cached = trajectory.initial_fit(storage, DummyBudget(), cfg, uid, metadata,
            trajectory.Reader(raw, assets, cfg))
        assert_tree_equal(without_timing(cached), without_timing(actual), "completed initialization")
        assert snapshot_unit(storage.unit_path(uid)) == sealed
