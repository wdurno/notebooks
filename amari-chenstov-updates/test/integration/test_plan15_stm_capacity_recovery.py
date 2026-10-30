"""Explicit tiny cached-MNIST recovery tests for the actual Phase 12 Runner.

No CLI, notebook, download, or production root is used. CPU/float64 is the
default; PLAN15_STM_TEST_DEVICE=cuda requests CUDA/float32 explicitly. The direct
Runner harness controls source identity in-process, independently of CLI seals.
PLAN15_STM_RECOVERY_ROOT selects an existing persistent harness budget root.
Use pytest -k to select a chart, interruption boundary, or completed-unit check.
"""

from dataclasses import replace
import math
import os
from pathlib import Path
import time

import pytest
import torch

from mnist_experiment import plan15_grid_subspace as subspace
from mnist_experiment.fisher_memory.artifacts import ImmutableError, Store, iter_units
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.fisher_memory_grid_registration.data import MNISTSource
from mnist_experiment.fisher_memory_grid_registration import runner as grid_runner
from mnist_experiment.fisher_memory_regression.config import Recipe
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, file_digest, read_json, setup_torch,
)
from mnist_experiment.fisher_memory_stm_capacity import PROTOCOL, model, run
from mnist_experiment.fisher_memory_stm_capacity.config import Config, FD, ROOT, arm
from mnist_experiment.fisher_memory_stm_capacity.runner import ChartStore, Runner, trajectory_id


DATA_ROOT = ROOT / "cache/mnist_experiment/datasets"
RAW_FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
FAMILY = "recovery--grid"
REPLICA = 0


def identity(cfg):
    return dict(protocol=PROTOCOL, role="direct_runner_recovery_test", harness_schema=1,
                source_identity="controlled-in-process-v1", config_hash=cfg.scientific_hash)


def recipe(cfg):
    return Recipe.from_config(cfg, FD, capacity=2, replay_count=1, ridge=1., strength=.5)


def scientific(value):
    """Keep every scientific diagnostic; only measured time/memory is variable."""
    if isinstance(value, dict):
        return {key: scientific(item) for key, item in value.items()
                if "seconds" not in key and "elapsed" not in key
                and key not in ("peak_vram_bytes", "peak_host_bytes", "process_peak_host_bytes")}
    if isinstance(value, (list, tuple)):
        return [scientific(item) for item in value]
    return value


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), (
            f"only_left={sorted(left.keys() - right.keys())}; "
            f"only_right={sorted(right.keys() - left.keys())}")
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


@pytest.fixture(scope="module")
def inputs():
    missing = [name for name in RAW_FILES if not (DATA_ROOT / "MNIST/raw" / name).is_file()]
    if missing:
        pytest.skip("cached MNIST required; no download: " + ", ".join(missing))
    device = os.environ.get("PLAN15_STM_TEST_DEVICE", "cpu")
    dtype = "float32" if device == "cuda" else "float64"
    cfg = replace(Config.smoke_config(device), dtype=dtype,
                  reference_max_size=2, evaluation_chunk=2)
    cfg.validate()
    assert (cfg.novel_count, cfg.init_size, cfg.reference_size, cfg.reference_max_size) == (3, 8, 2, 2)
    assert cfg.threads == 2 and cfg.dtype == dtype
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        setup_torch(cfg)
        yield cfg, MNISTSource(DATA_ROOT)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.fixture(scope="module")
def budget_root(tmp_path_factory):
    # An explicit existing harness root permits adding cases without retraining
    # immutable baselines. Store identity still rejects a different config/root.
    existing = os.environ.get("PLAN15_STM_RECOVERY_ROOT")
    if existing:
        root = Path(existing).resolve()
        if not root.is_dir():
            raise ValueError("PLAN15_STM_RECOVERY_ROOT must be an existing harness budget root")
        return root
    return tmp_path_factory.mktemp("phase12_direct_recovery_budget")


@pytest.fixture(scope="module", params=("head_r1", "all_r2"))
def chart(request):
    return request.param


def execute(root, inputs, chart, *, stop=None, resume=False):
    cfg, source = inputs
    with Store(root, identity(cfg), resume=resume, free_reserve_bytes=0,
               max_bytes=2_000_000_000, max_active_seconds=180., max_wall_seconds=180.) as store:
        budget = RuntimeBudget(store, allocation_seconds=180.)
        try:
            with run.bindings(), store.control.signal_handlers():
                runner = Runner(cfg, store, source, budget, stop_at=stop)
                assert runner.compression_backend == "subspace_" + cfg.device
                result = runner.run_chart(chart, FAMILY, REPLICA, "development", recipe(cfg))
                # Recovery concerns sealed artifacts. Fresh-return schema parity
                # has a separate fast scheduler regression test.
                record = runner.record(result["unit_id"])
                with model.use_chart(chart):
                    state = runner.store.load_checkpoint(record["unit_id"])
                return record, state
        finally:
            budget.close()


def assert_result(result, baseline, cfg, chart):
    record, state = result
    expected_record, expected_state = baseline
    equal_tree(scientific(state), scientific(expected_state))
    equal_tree(scientific(record), scientific(expected_record))
    assert record["stm_config"] == state["stm_config"] == chart
    assert record["arm"] == arm(chart, FD) and record["protocol"] == PROTOCOL
    assert record["valid"] and record["solver_failures"] == 0
    assert record["output_dim"] == 54 and record["archive_rank"] == 64
    assert state["n"] == cfg.novel_count
    assert [row["n"] for row in record["metrics"]] == [0, 1, 2, 3]
    learner = state["learner"]
    assert learner["theta"].numel() == sum(model.COUNTS[chart])
    assert learner["compression_backend"] == "subspace_" + cfg.device
    retired = learner["consolidated_ids"]
    live = [entry["exposure_id"] for entry in learner["replay"]]
    assert retired and len(retired) == len(set(retired))
    assert retired == [entry["exposure"] for entry in state["retirements"]]
    assert set(retired).isdisjoint(live)
    assert sorted(retired + live) == list(range(learner["cumulative_count"]))
    assert live == list(range(learner["cumulative_count"] - 2, learner["cumulative_count"]))
    assert learner["accepted_arrivals"] == state["next_segment"] == len(state["fit_records"])
    compression = learner["archive_diagnostics"]["compression"]
    assert compression["requested_backend"] == "subspace_" + cfg.device
    assert compression["requested_rank"] == 64 and compression["numerical_policy"] == subspace.POLICY
    assert compression.get("warm_start_columns", 0) == 0
    assert len(learner["ltm_indices"]) == 3268
    for key in ("learning_seconds", "input_seconds", "fit_seconds", "eviction_seconds",
                "jacobian_fisher_seconds", "compression_seconds", "evaluation_seconds"):
        assert math.isfinite(state[key]) and state[key] >= 0
    assert state["learning_seconds"] == pytest.approx(
        state["input_seconds"] + state["fit_seconds"] + state["eviction_seconds"])
    assert all(not fit["solver_failed"] and fit["accepted_steps"] > 0 for fit in state["fit_records"])


@pytest.fixture(scope="module")
def baseline(budget_root, inputs, chart):
    root = budget_root / (chart + "-uninterrupted")
    result = execute(root, inputs, chart, resume=(root / "identity.json").exists())
    assert_result(result, result, inputs[0], chart)
    return result


@pytest.mark.parametrize("boundary", ("pretrain", "accepted", "eviction", "mid_solve"))
def test_interrupted_runner_resumes_exact_parameters_fifo_retirements_and_metrics(
        budget_root, inputs, chart, baseline, boundary, record_property):
    started = time.perf_counter()
    cfg, _ = inputs
    root = budget_root / (chart + "-" + boundary)
    original_id, original_chart = grid_runner.trajectory_id, model.CHART.get()
    with pytest.raises(StopRequested):
        execute(root, inputs, chart, stop=boundary)
    assert grid_runner.trajectory_id is original_id and model.CHART.get() == original_chart
    before = completed_files(root)
    with Store(root, identity(cfg), resume=True, free_reserve_bytes=0) as store:
        if boundary == "pretrain":
            asset = next(unit for unit in store.iter_units()
                         if unit["metadata"].get("kind") == "grid_registration_assets")
            checkpoint = store.load_checkpoint(asset["unit_id"])
            assert checkpoint["stage"] == "pretrain" and checkpoint["offset"] > 0
            assert checkpoint["optimizer"] is not None and asset["status"] == "incomplete"
            assert checkpoint["stm_config"] == "all_r2"
        else:
            unit = trajectory_id(chart, FAMILY, REPLICA, recipe(cfg), cfg.novel_count)
            with model.use_chart(chart):
                checkpoint = ChartStore(store).load_checkpoint(unit)
            assert not store.is_complete(unit) and checkpoint["stm_config"] == chart
            if boundary == "accepted":
                assert checkpoint["next_segment"] > 0
                assert checkpoint["learner"]["accepted_arrivals"] > 0
            else:
                # Neither an interrupted fit nor a computed retirement commits.
                assert checkpoint["learner"]["consolidated_ids"] == []
                assert checkpoint["retirements"] == []
                if boundary == "mid_solve":
                    assert checkpoint["next_segment"] == checkpoint["n"] == 0
                    assert checkpoint["learner"]["cumulative_count"] == 0
        partial_seconds = read_json(root / "runtime.json")["cumulative_active_seconds"]
    result = execute(root, inputs, chart, resume=True)
    assert_result(result, baseline, cfg, chart)
    after = completed_files(root)
    assert before.items() <= after.items()
    assert all(unit["status"] == "completed" for unit in iter_units(root, full=True))
    assert read_json(root / "runtime.json")["cumulative_active_seconds"] >= partial_seconds
    repeated = execute(root, inputs, chart, resume=True)
    equal_tree(repeated, result)
    assert completed_files(root) == after
    assert grid_runner.trajectory_id is original_id and model.CHART.get() == original_chart
    record_property("direct_runner_test_seconds", round(time.perf_counter() - started, 6))


def test_completed_unit_is_a_read_only_noop(budget_root, inputs, chart, baseline):
    root = budget_root / (chart + "-uninterrupted")
    before = completed_files(root)
    repeated = execute(root, inputs, chart, resume=True)
    equal_tree(repeated, baseline)
    assert completed_files(root) == before


def test_initial_assets_pair_ltm_fisher_named_plans_and_reject_wrong_chart(
        budget_root, inputs, chart, baseline):
    cfg, _ = inputs
    record, final = baseline
    root = budget_root / (chart + "-uninterrupted")
    before = completed_files(root)
    with Store(root, identity(cfg), resume=True, free_reserve_bytes=0) as store:
        asset_id = record["asset_id"]
        with model.use_chart(chart):
            asset = ChartStore(store).load_checkpoint(asset_id)
        canonical_id = asset_id if chart == "all_r2" else asset["parent_asset"]
        canonical = store.load_checkpoint(canonical_id)
        assert canonical["stm_config"] == "all_r2" and canonical["stage"] == asset["stage"] == "ready"
        assert canonical["plan_hash"] == asset["plan_hash"] == record["plan_hash"]
        dtype = getattr(torch, cfg.dtype)
        current_net, current_layout = model.build_model(1, dtype=dtype, stm_config=chart)
        _, canonical_layout = model.build_model(1, dtype=dtype, stm_config="all_r2")
        current_layout.assert_metadata(asset["layout"])
        ltm, stm = model.memory_indices(current_layout)
        old_ltm, _ = model.memory_indices(canonical_layout)
        torch.testing.assert_close(asset["theta"][ltm], canonical["theta"][old_ltm], rtol=0, atol=0)
        torch.testing.assert_close(asset["learner"]["theta"], asset["theta"], rtol=0, atol=0)
        torch.testing.assert_close(asset["learner"]["ltm_indices"], ltm, rtol=0, atol=0)
        equal_tree(asset["learner"]["fisher_L"], canonical["learner"]["fisher_L"])
        equal_tree(asset["learner"]["anchor_L"], canonical["learner"]["anchor_L"])
        for spec in current_layout.specs:
            if spec.name.endswith("lora_B"):
                assert asset["theta"][spec.start:spec.stop].count_nonzero() == 0
        assert stm.numel() == model.COUNTS[chart][1]
        if chart != "all_r2":
            assert "reference_fisher" not in asset and "training_optimizer" not in asset
            assert asset["parent_seal"] == read_json(store.unit_path(canonical_id) / "COMPLETED")["seal_hash"]
            health = read_json(store.unit_path(asset_id) / "health.json")
            assert health["ltm_identical"] and health["initial_residual_zero"]
        plan = read_json(store.unit_path(canonical_id) / "plan.json")
        assert plan["seeds"] and plan["seeds"]["epochs"]
        assert len(plan["stream_novel"]) >= cfg.novel_count
        wrong = "all_r2" if chart == "head_r1" else "head_r1"
        with model.use_chart(wrong), pytest.raises(ValueError, match="checkpoint STM chart"):
            ChartStore(store).load_checkpoint(record["unit_id"])
        with pytest.raises(ImmutableError):
            store.save_checkpoint(record["unit_id"], final)
        assert all(parameter.requires_grad for parameter in current_net.parameters())
    assert before == completed_files(root)
