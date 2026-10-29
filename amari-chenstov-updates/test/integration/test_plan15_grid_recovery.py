"""Tiny cached-MNIST CPU recovery drills for both Phase 11 registration tasks.

Run explicitly with ~/.venv/bin/python -m pytest -q <this file>. These tests
use Runner/Store directly, never production, CLI source manifests or downloads.
Hard-kill workers run this same file in a standalone Python subprocess.
"""

import dataclasses
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import ImmutableError, Store, iter_units
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.fisher_memory_grid_registration.config import Config, ROOT, Recipe
from mnist_experiment.fisher_memory_grid_registration.data import MNISTSource
from mnist_experiment.fisher_memory_grid_registration.runner import Runner, trajectory_id
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, file_digest, read_json, setup_torch,
)


DATA_ROOT = ROOT / "cache/mnist_experiment/datasets"
RAW_FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
WORKER_CODE = """import runpy
import sys
helpers = runpy.run_path(sys.argv[1])
helpers["_hard_kill_worker"](*sys.argv[2:])
"""


def _smoke_config():
    cfg = Config.smoke_config(device=os.environ.get("GRID_TEST_DEVICE", "cpu"))
    assert cfg.smoke and cfg.dtype == "float64"
    return cfg


def _identity(cfg):
    # Deliberately do not hash files another implementation agent is editing.
    # CLI source/config provenance is covered by the parent's acceptance suite.
    return {"grid_registration_recovery": 1, "config_hash": cfg.scientific_hash}


def _recipe(cfg):
    return Recipe.from_config(cfg, "fd_replay_ewc", capacity=2, replay_count=1, strength=.5)


@pytest.fixture(scope="module")
def inputs():
    missing = [name for name in RAW_FILES if not (DATA_ROOT / "MNIST/raw" / name).is_file()]
    if missing:
        pytest.skip(f"cached MNIST required; no download: {', '.join(missing)}")
    cfg = _smoke_config()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        setup_torch(cfg)
        yield cfg, MNISTSource(DATA_ROOT)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.fixture(scope="module", params=("grid",), ids=("d54",))
def task(request):
    return request.param


def execute(root, inputs, task, *, stop=None, resume=False, prefix=None,
            after_commit=None, on_progress=None, numerical_backend=None):
    cfg, source = inputs
    family = "acceptance--" + task
    recipe = _recipe(cfg)
    with Store(root, _identity(cfg), resume=resume, free_reserve_bytes=0) as store:
        budget = RuntimeBudget(store, allocation_seconds=600.)
        try:
            with store.control.signal_handlers():
                kwargs = {} if on_progress is None else {"on_progress": on_progress}
                runner = Runner(cfg, store, source, budget, stop_at=stop,
                                after_commit=after_commit, **kwargs)
                runner.compression_backend = numerical_backend or (
                    "exact_cuda" if cfg.device == "cuda" else "exact_cpu")
                record = runner.run(family, 0, "development", recipe,
                                    target=cfg.novel_count if prefix is None else prefix)
                if prefix is not None:
                    record = runner.run(family, 0, "development", recipe, parent_target=prefix)
                return record, store.load_checkpoint(record["unit_id"])
        finally:
            budget.close()


def scientific(value):
    """Remove runtime measurements, not fit diagnostics or scientific metrics."""
    if isinstance(value, dict):
        return {key: scientific(item) for key, item in value.items()
                if "seconds" not in key and "elapsed" not in key
                and key not in {"peak_vram_bytes", "peak_host_bytes", "process_peak_host_bytes"}}
    if isinstance(value, (list, tuple)):
        return [scientific(item) for item in value]
    return value


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


def completed_files(root):
    return {str(path.relative_to(root)): file_digest(path)
            for unit in iter_units(root, completed_only=True, full=True)
            for path in unit["path"].rglob("*") if path.is_file()}


def assert_exact_recovery(result, baseline, cfg, task):
    record, state = result
    expected_record, expected_state = baseline
    torch.testing.assert_close(state["learner"]["theta"], expected_state["learner"]["theta"],
                               rtol=0, atol=0)
    assert state["learner"]["consolidated_ids"] == expected_state["learner"]["consolidated_ids"]
    equal_tree(scientific(state), scientific(expected_state))
    equal_tree(scientific(record["metrics"]), scientific(expected_record["metrics"]))
    equal_tree(state["retirements"], expected_state["retirements"])
    assert record["output_dim"] == 54
    assert record["valid"] and record["solver_failures"] == 0
    assert state["n"] == cfg.novel_count
    assert [row["n"] for row in record["metrics"]] == list(range(cfg.novel_count + 1))
    assert all(len(row["coordinate_mse"]["novel"]) == record["output_dim"]
               for row in record["metrics"])
    learner = state["learner"]
    retired = learner["consolidated_ids"]
    live = [row["exposure_id"] for row in learner["replay"]]
    assert retired and retired == [row["exposure"] for row in state["retirements"]]
    assert len(retired) == len(set(retired))
    assert set(retired).isdisjoint(live)
    assert sorted(retired + live) == list(range(learner["cumulative_count"]))
    assert live == list(range(learner["cumulative_count"] - learner["capacity"],
                              learner["cumulative_count"]))


@pytest.fixture(scope="module")
def baseline(tmp_path_factory, inputs, task):
    result = execute(tmp_path_factory.mktemp("registration_uninterrupted_" + task), inputs, task)
    assert_exact_recovery(result, result, inputs[0], task)
    return result


@pytest.mark.parametrize("stop", ("pretrain", "mid_solve", "eviction", "accepted"))
def test_interrupted_resume_exactly_matches_uninterrupted(tmp_path, inputs, task, baseline, stop):
    cfg, _ = inputs
    with pytest.raises(StopRequested):
        execute(tmp_path, inputs, task, stop=stop)
    before = completed_files(tmp_path)
    with Store(tmp_path, _identity(cfg), resume=True, free_reserve_bytes=0) as store:
        if stop == "pretrain":
            asset = next(unit for unit in store.iter_units()
                         if unit["metadata"].get("kind") == "grid_registration_assets")
            checkpoint = store.load_checkpoint(asset["unit_id"])
            assert checkpoint["stage"] == "pretrain" and checkpoint["offset"] > 0
            assert checkpoint["optimizer"] is not None and asset["status"] == "incomplete"
        else:
            unit = trajectory_id("acceptance--" + task, 0, _recipe(cfg), cfg.novel_count)
            checkpoint = store.load_checkpoint(unit)
            assert not store.is_complete(unit)
            if stop in ("mid_solve", "eviction"):
                assert checkpoint["next_segment"] == checkpoint["n"] == 0
                assert checkpoint["learner"]["cumulative_count"] == 0
                assert not checkpoint["learner"]["consolidated_ids"]
            else:
                assert checkpoint["next_segment"] > 0
                assert checkpoint["learner"]["accepted_arrivals"] > 0
    result = execute(tmp_path, inputs, task, resume=True)
    assert_exact_recovery(result, baseline, cfg, task)
    assert before.items() <= completed_files(tmp_path).items()
    assert all(unit["status"] == "completed" for unit in iter_units(tmp_path, full=True))


def test_prefix_continuation_and_completed_store_are_read_only(tmp_path, inputs, task, baseline):
    cfg, _ = inputs
    record, state = execute(tmp_path, inputs, task, prefix=1)
    assert_exact_recovery((record, state), baseline, cfg, task)
    before = completed_files(tmp_path)
    repeated = execute(tmp_path, inputs, task, prefix=1, resume=True)
    equal_tree(repeated[0], record)
    equal_tree(repeated[1], state)
    assert before == completed_files(tmp_path)
    with Store(tmp_path, _identity(cfg), resume=True, free_reserve_bytes=0) as store:
        unit = record["unit_id"]
        metadata = read_json(store.unit_path(unit) / "manifest.json")["metadata"]
        for mutation in (lambda: store.save_checkpoint(unit, state),
                         lambda: store.write_json(unit, "metrics.json", []),
                         lambda: store.begin(unit, dict(metadata, output_dim=99))):
            with pytest.raises(ImmutableError):
                mutation()
    assert before == completed_files(tmp_path)


def test_completed_requests_reject_changed_partition_audit_or_config(tmp_path, inputs, task):
    cfg, source = inputs
    record, _ = execute(tmp_path, inputs, task)
    before = completed_files(tmp_path)
    with Store(tmp_path, _identity(cfg), resume=True, free_reserve_bytes=0) as store:
        budget = RuntimeBudget(store, allocation_seconds=600.)
        try:
            runner = Runner(cfg, store, source, budget)
            for kwargs in ({"partition": "confirmation"}, {"audit": True}, {"capture": True}):
                partition = kwargs.pop("partition", "development")
                with pytest.raises(ValueError, match="metadata"):
                    runner.run("acceptance--" + task, 0, partition,
                               Recipe.from_mapping(record["recipe"]), **kwargs)
        finally:
            budget.close()
    changed = dataclasses.replace(cfg, seed=cfg.seed + 1)
    with pytest.raises(ImmutableError, match="identity changed"):
        with Store(tmp_path, _identity(changed), resume=True, free_reserve_bytes=0):
            pass
    assert before == completed_files(tmp_path)


def test_stop_file_after_accepted_checkpoint_resumes_exactly(tmp_path, inputs, task, baseline):
    def request_stop(unit, state):
        (tmp_path / "STOP").touch()

    with pytest.raises(StopRequested):
        execute(tmp_path, inputs, task, after_commit=request_stop)
    assert (tmp_path / "STOP").is_file()
    with pytest.raises(StopRequested):
        execute(tmp_path, inputs, task, resume=True)
    (tmp_path / "STOP").unlink()
    assert_exact_recovery(execute(tmp_path, inputs, task, resume=True), baseline, inputs[0], task)


def test_optimizer_audit_preserves_accepted_trajectory(tmp_path, inputs, task, baseline):
    cfg,source=inputs
    with Store(tmp_path,_identity(cfg),free_reserve_bytes=0) as store:
        budget=RuntimeBudget(store,allocation_seconds=600.)
        try:
            runner=Runner(cfg,store,source,budget)
            runner.compression_backend = "exact_cuda" if cfg.device == "cuda" else "exact_cpu"
            record=runner.run("acceptance--"+task,0,"development",_recipe(cfg),audit=True)
            state=store.load_checkpoint(record["unit_id"])
            torch.testing.assert_close(state["learner"]["theta"],baseline[1]["learner"]["theta"],rtol=0,atol=0)
            assert state["learner"]["consolidated_ids"]==baseline[1]["learner"]["consolidated_ids"]
            equal_tree(scientific(record["metrics"]),scientific(baseline[0]["metrics"]))
            assert state["audits"] and all(not row["enlarged_failed"] for row in state["audits"])
        finally:
            budget.close()


def _hard_kill_worker(root, task, boundary, ready_path):
    """Pause only after a durable checkpoint so SIGKILL never races the hook."""
    cfg = _smoke_config()
    setup_torch(cfg)
    source = MNISTSource(DATA_ROOT)
    ready_path = Path(ready_path)

    def pause(unit):
        temporary = ready_path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump({"boundary": boundary, "unit_id": unit, "pid": os.getpid()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(ready_path)
        while True:
            time.sleep(1.)

    def progress(**updates):
        if boundary == "pretrain" and updates.get("stage") == "initialization":
            pause(updates["current"])

    def committed(unit, state):
        if boundary == "accepted" or (boundary == "eviction" and state["retirements"]):
            pause(unit)

    execute(Path(root), (cfg, source), task, after_commit=committed, on_progress=progress)
    raise AssertionError(f"worker completed without reaching {boundary}")


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL recovery requires POSIX")
@pytest.mark.parametrize("boundary", ("pretrain", "accepted", "eviction"))
def test_hard_kill_checkpoint_resumes_exactly(tmp_path, inputs, task, baseline, boundary):
    cfg, _ = inputs
    root, ready = tmp_path / "study", tmp_path / "boundary.json"
    log_path = tmp_path / "worker.log"
    environment = dict(os.environ, OMP_NUM_THREADS=str(cfg.threads))
    if cfg.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", WORKER_CODE, str(Path(__file__).resolve()),
             str(root), task, boundary, str(ready)],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while not ready.is_file():
                if process.poll() is not None:
                    pytest.fail(f"worker exited before {boundary}:\n{log_path.read_text()}")
                if time.monotonic() >= deadline:
                    pytest.fail(f"worker did not checkpoint within 60 seconds:\n{log_path.read_text()}")
                time.sleep(.02)
            checkpoint_event = read_json(ready)
            assert checkpoint_event["boundary"] == boundary
            assert checkpoint_event["pid"] == process.pid
            process.kill()
            assert process.wait(timeout=10) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    before = completed_files(root)
    with Store(root, _identity(cfg), resume=True, free_reserve_bytes=0) as store:
        unit = checkpoint_event["unit_id"]
        state = store.load_checkpoint(unit)
        assert not store.is_complete(unit)
        if boundary == "pretrain":
            assert state["stage"] == "pretrain" and state["offset"] > 0
            assert state["optimizer"] is not None
        else:
            assert state["next_segment"] > 0 and state["learner"]["accepted_arrivals"] > 0
            if boundary == "eviction":
                assert state["retirements"] and state["learner"]["consolidated_ids"]
    assert_exact_recovery(execute(root, inputs, task, resume=True), baseline, cfg, task)
    assert before.items() <= completed_files(root).items()
    assert all(unit["status"] == "completed" for unit in iter_units(root, full=True))
