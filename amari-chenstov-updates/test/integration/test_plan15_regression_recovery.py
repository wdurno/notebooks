"""Explicit Phase 8A cached-MNIST acceptance, never production learning."""

import dataclasses
import os
import signal
import time
from pathlib import Path

import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import Store, iter_units
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.fisher_memory_regression.config import Config, PACKAGE, ROOT, Recipe
from mnist_experiment.fisher_memory_regression.data import MNISTSource
from mnist_experiment.fisher_memory_regression.runner import Runner, trajectory_id
from mnist_experiment.fisher_memory_regression.runtime import RuntimeBudget, file_digest, setup_torch


@pytest.fixture(scope="module")
def inputs():
    cfg = Config.read(PACKAGE / "configs/smoke.json")
    cfg = dataclasses.replace(cfg, device=os.environ.get("REGRESSION_ACCEPTANCE_DEVICE", "cpu"),
                              dtype=os.environ.get("REGRESSION_ACCEPTANCE_DTYPE", cfg.dtype))
    setup_torch(cfg)
    data = ROOT / "cache/mnist_experiment/datasets"
    if not (data / "MNIST/raw/train-images-idx3-ubyte").exists():
        pytest.skip("cached MNIST required; no download")
    return cfg, MNISTSource(data)


def execute(root, inputs, *, stop=None, resume=False, prefix=None, after_commit=None):
    cfg, source = inputs
    recipe = Recipe.from_config(cfg, "fd_replay_ewc", capacity=2, replay_count=1, strength=.5)
    with Store(root, {"acceptance": 1, "config_hash": cfg.scientific_hash}, resume=resume,
               free_reserve_bytes=0) as store:
        budget = RuntimeBudget(store)
        try:
            with store.control.signal_handlers():
                runner = Runner(cfg, store, source, budget, stop_at=stop, after_commit=after_commit)
                record = runner.run("acceptance", 0, "development", recipe,
                                    target=prefix or cfg.novel_count)
                if prefix:
                    record = runner.run("acceptance", 0, "development", recipe, parent_target=prefix)
                return record, store.load_checkpoint(record["unit_id"])
        finally:
            budget.close()


def scientific(value):
    if isinstance(value, dict):
        return {key: scientific(item) for key, item in value.items()
                if "seconds" not in key and "elapsed" not in key and key != "peak_vram_bytes"}
    if isinstance(value, list):
        return [scientific(item) for item in value]
    return value


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


@pytest.fixture(scope="module")
def baseline(tmp_path_factory, inputs):
    return execute(tmp_path_factory.mktemp("regression_uninterrupted"), inputs)


@pytest.mark.parametrize("stop", ["pretrain", "mid_solve", "eviction", "accepted"])
def test_interrupted_resume_exactly_matches_uninterrupted(tmp_path, inputs, baseline, stop):
    with pytest.raises(StopRequested):
        execute(tmp_path, inputs, stop=stop)
    record, state = execute(tmp_path, inputs, resume=True)
    equal_tree(scientific(state), scientific(baseline[1]))
    equal_tree(scientific(record["metrics"]), scientific(baseline[0]["metrics"]))
    assert state["learner"]["consolidated_ids"]
    ids = state["learner"]["consolidated_ids"]
    assert len(ids) == len(set(ids))


def test_prefix_continuation_matches_full_and_completion_is_read_only(tmp_path, inputs, baseline, monkeypatch):
    record, state = execute(tmp_path, inputs, prefix=1)
    equal_tree(scientific(state), scientific(baseline[1]))
    cfg, _ = inputs
    assert [row["n"] for row in record["metrics"]] == list(range(cfg.novel_count + 1))
    files = {str(path): file_digest(path) for path in (tmp_path / "units").rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("a completed trajectory must not train or deserialize its asset")

    monkeypatch.setattr(Runner, "assets", forbidden)
    repeated, _ = execute(tmp_path, inputs, resume=True, prefix=1)
    assert repeated == record
    assert files == {str(path): file_digest(path) for path in (tmp_path / "units").rglob("*") if path.is_file()}
    assert all(row["status"] == "completed" for row in iter_units(tmp_path, full=True))


def test_stop_file_after_commit_and_resume(tmp_path, inputs, baseline):
    def request_stop(unit, state):
        (tmp_path / "STOP").touch()

    with pytest.raises(StopRequested):
        execute(tmp_path, inputs, after_commit=request_stop)
    (tmp_path / "STOP").unlink()
    _, state = execute(tmp_path, inputs, resume=True)
    equal_tree(scientific(state), scientific(baseline[1]))


def test_completed_metadata_changes_are_rejected(tmp_path, inputs):
    record, _ = execute(tmp_path, inputs)
    cfg, source = inputs
    recipe = Recipe.from_mapping(record["recipe"])
    with Store(tmp_path, {"acceptance": 1, "config_hash": cfg.scientific_hash},
               resume=True, free_reserve_bytes=0) as store:
        budget = RuntimeBudget(store)
        try:
            runner = Runner(cfg, store, source, budget)
            for partition, audit in (("confirmation", False), ("development", True)):
                with pytest.raises(ValueError, match="metadata"):
                    runner.run("acceptance", 0, partition, recipe, audit=audit)
        finally:
            budget.close()


def test_sigterm_stops_within_one_minute_and_resumes(tmp_path, inputs, baseline):
    requested_at = []

    def request_stop(unit, state):
        requested_at.append(time.monotonic())
        os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(StopRequested):
        execute(tmp_path, inputs, after_commit=request_stop)
    assert time.monotonic() - requested_at[0] < 60
    _, state = execute(tmp_path, inputs, resume=True)
    equal_tree(scientific(state), scientific(baseline[1]))
