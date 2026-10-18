"""Explicit Phase 0 acceptance on cached MNIST; never downloads or runs production.

Run this file explicitly with pytest. Set PLAN15_MNIST_ROOT to override the
repository's cached dataset location. Each case checks a fresh full trajectory
against an accepted-step-4 interruption and a new Store/Runner invocation.
PLAN15_TEST_DEVICE=cuda PLAN15_TEST_DTYPE=float32 selects deterministic CUDA
resume acceptance; process fault drills remain CPU-only. Defaults are CPU/float64.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

TEST_DEVICE = os.environ.get("PLAN15_TEST_DEVICE", "cpu")
if TEST_DEVICE == "cuda":
    # Must be configured before CUDA context creation, including provenance capture.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.config import CONDITIONS, Config, PACKAGE, ROOT, Treatment, provenance
from mnist_experiment.fisher_memory.data import MNISTSource
from mnist_experiment.fisher_memory.runner import Runner


RUNTIME_FIELDS = frozenset({"active_seconds", "cumulative_active_seconds", "elapsed_seconds",
                            "solver_seconds", "shadow_solver_seconds", "fit_seconds",
                            "peak_process_rss_bytes", "peak_cuda_allocated_bytes"})
ENVIRONMENTS = ("rotation_linear", "digit9_mixture")


class InterruptedAfterCommit(RuntimeError):
    """The test's intentional interruption, after an immutable checkpoint commit."""


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.set_num_threads(previous)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic


@pytest.fixture(scope="module")
def smoke_config():
    if TEST_DEVICE == "cuda" and not torch.cuda.is_available():
        pytest.skip("PLAN15_TEST_DEVICE=cuda requires an available CUDA device")
    config = dataclasses.replace(
        Config.read(PACKAGE / "configs" / "smoke.json"),
        device=TEST_DEVICE, dtype=os.environ.get("PLAN15_TEST_DTYPE", "float64"),
        threads=1, environments=ENVIRONMENTS,
        inner_steps=1, max_evaluations=3,
    )
    config.validate()
    assert config.smoke
    return config


@pytest.fixture(scope="module")
def source():
    root = Path(os.environ.get("PLAN15_MNIST_ROOT", ROOT / "cache/mnist_experiment/datasets"))
    try:
        return MNISTSource(root)
    except RuntimeError as error:
        if "Dataset not found" in str(error):
            pytest.skip(f"cached MNIST required at {root}; integration acceptance never downloads")
        raise


@pytest.fixture(scope="module")
def identity(smoke_config):
    # Capture once: mutable wall-clock/progress fields are not scientific identity.
    return provenance(smoke_config)


def assert_scientifically_equal(actual: Any, expected: Any, path="state") -> None:
    """Exact equality; only named measured time/peak-memory fields are excluded."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert torch.equal(actual.cpu(), expected.cpu()), path
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key, value in expected.items():
            if key not in RUNTIME_FIELDS:
                assert_scientifically_equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            assert_scientifically_equal(a, b, f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def scientific_files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted((root / "units").rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def open_store(root, identity, *, resume=False):
    return Store(root, identity, resume=resume, free_reserve_bytes=0,
                 finalization_reserve_bytes=32_000_000, max_bytes=500_000_000)


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_resume_matches_uninterrupted_cached_mnist(
        tmp_path, smoke_config, source, identity, environment, condition, monkeypatch):
    cfg = smoke_config
    strength = .01 if condition in ("jvp", "finite_difference", "directional", "distillation") else 0.
    treatment = Treatment.default(cfg, condition, strength=strength)
    full_root, resume_root = tmp_path / "uninterrupted", tmp_path / "interrupted"
    unit = f"trajectory-acceptance-{environment}-0000-{treatment.key}"

    with open_store(full_root, identity) as store:
        full_summary = Runner(cfg, store, source).run_trajectory("acceptance", "0", environment, 0, treatment)
        full = store.load_checkpoint(unit)
        assert store.is_complete(unit, full=True)
        full_assets = store.load_checkpoint(
            f"assets-acceptance-{'mixture' if environment == 'digit9_mixture' else 'rotation'}-0000")

    def stop_after_four(unit_id, accepted):
        assert unit_id == unit
        if accepted["step"] == 4:
            raise InterruptedAfterCommit("accepted step 4")

    with open_store(resume_root, identity) as store:
        with pytest.raises(InterruptedAfterCommit, match="step 4"):
            Runner(cfg, store, source, after_commit=stop_after_four).run_trajectory(
                "acceptance", "0", environment, 0, treatment)
        prefix = store.load_checkpoint(unit)
        assert prefix["step"] == 4
        assert not store.is_complete(unit)
        assert prefix["learner"]["consolidated_ids"], "must interrupt after actual FIFO eviction"
        progress = json.loads((store.unit_path(unit) / "progress.json").read_text())
        assert progress["sealed"] is True
        assert progress["accepted_step"] == 4
        assert_scientifically_equal(progress["rows"], prefix["rows"], "sealed_progress")

    # Resume must use named seeds/state, not the caller's ambient random state.
    random.seed(917)
    np.random.seed(918)
    torch.manual_seed(919)
    with open_store(resume_root, identity, resume=True) as store:
        resumed_summary = Runner(cfg, store, source).run_trajectory(
            "acceptance", "0", environment, 0, treatment)
        resumed = store.load_checkpoint(unit)
        assert store.is_complete(unit, full=True)
        assert resumed["step"] == len(cfg.schedule(environment))
        assert_scientifically_equal(resumed, full)
        assert_scientifically_equal(resumed_summary, full_summary, "summary")
        assert_scientifically_equal(resumed["rows"][:len(prefix["rows"])], prefix["rows"], "retained_prefix")
        assets = store.load_checkpoint(
            f"assets-acceptance-{'mixture' if environment == 'digit9_mixture' else 'rotation'}-0000")
        assert_scientifically_equal(assets, full_assets, "paired_initialization")
        exported = json.loads((store.unit_path(unit) / "metrics.json").read_text())
        assert_scientifically_equal(exported, resumed["rows"], "final_metrics")
        splits = {row["split"] for row in exported}
        assert {"prequential", "current", "historical", "evicted_train", "dependence", "curvature"} <= splits
        learner = resumed["learner"]
        archived = learner["consolidated_ids"]
        live = [item["exposure_id"] for item in learner["replay"]]
        assert len(archived) == len(set(archived))
        assert not set(archived).intersection(live)
        assert len(archived) + len(live) == learner["cumulative_count"]

    immutable_before = scientific_files(resume_root)

    def unexpectedly_recomputed(*args, **kwargs):
        pytest.fail("completed-unit resume must not fit, evaluate, or rebuild reports")

    runner_module = importlib.import_module("mnist_experiment.fisher_memory.runner")
    initialize_module = importlib.import_module("mnist_experiment.fisher_memory.initialize")
    monkeypatch.setattr(runner_module, "fit_step", unexpectedly_recomputed)
    monkeypatch.setattr(initialize_module, "fit_step", unexpectedly_recomputed)
    monkeypatch.setattr(Runner, "_evaluate", unexpectedly_recomputed)
    with open_store(resume_root, identity, resume=True) as store:
        skipped = Runner(cfg, store, source, on_progress=unexpectedly_recomputed).run_trajectory(
            "acceptance", "0", environment, 0, treatment)
        assert_scientifically_equal(skipped, full_summary, "skipped_summary")
    assert scientific_files(resume_root) == immutable_before


@pytest.mark.parametrize("kind", ("rotation", "mixture"))
@pytest.mark.parametrize("stage", ("pretrain", "reference"))
def test_initialization_resume_matches_fresh(
        tmp_path, smoke_config, source, identity, kind, stage, monkeypatch):
    initialize = importlib.import_module("mnist_experiment.fisher_memory.initialize").initialize
    full_root, resume_root = tmp_path / "full", tmp_path / "resumed"
    unit = f"assets-acceptance-{kind}-0000"
    with open_store(full_root, identity) as store:
        asset_id, expected = initialize(store, source, smoke_config, "acceptance", 0, kind)
        assert asset_id == unit and store.is_complete(unit, full=True)

    save_checkpoint = Store.save_checkpoint
    interruptions = []

    def stop_after_initialization_commit(store, unit_id, accepted, **kwargs):
        path = save_checkpoint(store, unit_id, accepted, **kwargs)
        target = (accepted.get("stage") == stage and (
            accepted.get("offset", 0) > 0 if stage == "pretrain"
            else accepted.get("reference_offset") == smoke_config.fisher_chunk))
        if store.root == resume_root and unit_id == unit and target and not interruptions:
            interruptions.append(stage)
            raise InterruptedAfterCommit(f"initialization {stage}")
        return path

    with open_store(resume_root, identity) as store:
        with monkeypatch.context() as patch:
            patch.setattr(Store, "save_checkpoint", stop_after_initialization_commit)
            with pytest.raises(InterruptedAfterCommit, match=stage):
                initialize(store, source, smoke_config, "acceptance", 0, kind)
        partial = store.load_checkpoint(unit)
        assert partial["stage"] == stage
        assert not store.is_complete(unit)
        if stage == "pretrain":
            assert partial["offset"] == smoke_config.init_size
            assert partial["optimizer"] is not None
        else:
            assert partial["reference_offset"] == smoke_config.fisher_chunk < smoke_config.reference_size
            assert bool(torch.isfinite(partial["fisher_sum"]).all())
    assert interruptions == [stage]

    torch.manual_seed(1031)
    with open_store(resume_root, identity, resume=True) as store:
        asset_id, resumed = initialize(store, source, smoke_config, "acceptance", 0, kind)
        assert asset_id == unit and store.is_complete(unit, full=True)
        assert resumed["stage"] == "ready"
        assert_scientifically_equal(resumed, expected, f"initialization.{kind}.{stage}")
        assert_scientifically_equal(resumed["learner"]["theta"], expected["learner"]["theta"], "initial_theta")
        assert_scientifically_equal(resumed["learner"]["fisher_L"], expected["learner"]["fisher_L"], "initial_Fisher")


@pytest.fixture(scope="module")
def process_drill_assets(tmp_path_factory, smoke_config, source, identity):
    root = tmp_path_factory.mktemp("fisher-memory-process-drills")
    treatment = Treatment.default(smoke_config, "replay_ewc")
    unit = f"trajectory-acceptance-rotation_linear-0000-{treatment.key}"
    with open_store(root / "full", identity) as store:
        Runner(smoke_config, store, source).run_trajectory(
            "acceptance", "0", "rotation_linear", 0, treatment)
        expected = store.load_checkpoint(unit)

    def interrupt(unit_id, accepted):
        if accepted["step"] == 4:
            raise InterruptedAfterCommit()

    partial = root / "partial"
    with open_store(partial, identity) as store:
        with pytest.raises(InterruptedAfterCommit):
            Runner(smoke_config, store, source, after_commit=interrupt).run_trajectory(
                "acceptance", "0", "rotation_linear", 0, treatment)
        prefix = store.load_checkpoint(unit)
    return partial, expected, prefix, unit


PROCESS_DRILL = r'''
import json, os, signal, sys
from pathlib import Path
import torch
from mnist_experiment.fisher_memory import artifacts, learner
from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.fisher_memory.config import Config, Treatment
from mnist_experiment.fisher_memory.data import MNISTSource
from mnist_experiment.fisher_memory.runner import Runner

root, identity, values, data_root, phase, signal_name = sys.argv[1:]
cfg = Config(**json.loads(values))
cfg.validate()
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
source = MNISTSource(data_root)
fired = False

def interrupt():
    global fired
    if not fired:
        fired = True
        print("FAULT_REACHED:" + phase, flush=True)
        os.kill(os.getpid(), getattr(signal, signal_name))

try:
    with Store(Path(root), json.loads(identity), resume=True, free_reserve_bytes=0,
               finalization_reserve_bytes=32_000_000, max_bytes=500_000_000) as store:
        with store.control.signal_handlers():
            if phase == "fit":
                prepare = learner.prepare_objective
                def interrupted_objective(*args, **kwargs):
                    objective, metadata = prepare(*args, **kwargs)
                    def evaluate(theta):
                        result = objective(theta)
                        interrupt()
                        return result
                    return evaluate, metadata
                learner.prepare_objective = interrupted_objective
            elif phase == "write":
                write = artifacts._BoundedWriter.write
                def interrupted_write(writer, data):
                    result = write(writer, data)
                    interrupt()
                    return result
                artifacts._BoundedWriter.write = interrupted_write
            elif phase == "validation":
                file_hash = artifacts._file_hash
                def interrupted_hash(path, check=lambda: None):
                    calls = 0
                    def boundary():
                        nonlocal calls
                        check()
                        calls += 1
                        if calls == 2:
                            interrupt()
                    return file_hash(path, boundary)
                artifacts._file_hash = interrupted_hash
                store.is_complete("assets-acceptance-rotation-0000", full=True)
                raise AssertionError("validation injection was not reached")
            elif phase == "report":
                from mnist_experiment.fisher_memory import report
                plot_cell = report._plot_cell
                def interrupted_plot(*args, **kwargs):
                    result = plot_cell(*args, **kwargs)
                    interrupt()
                    return result
                report._plot_cell = interrupted_plot
                report.refresh(store.root, store.root / "reports" / "acceptance.ipynb",
                               publish=report.reserved_publisher(store))
                raise AssertionError("report injection was not reached")
            Runner(cfg, store, source).run_trajectory(
                "acceptance", "0", "rotation_linear", 0, Treatment.default(cfg, "replay_ewc"))
            print("RESUMED", flush=True)
except StopRequested:
    print("GRACEFUL_STOP", flush=True)
    sys.exit(77)
'''


@pytest.mark.parametrize("phase", ["fit", "write", "validation", "report"])
@pytest.mark.parametrize("signum", [signal.SIGKILL, signal.SIGTERM], ids=["SIGKILL", "SIGTERM"])
@pytest.mark.skipif(TEST_DEVICE != "cpu", reason="subprocess fault drills are CPU-only")
def test_fresh_process_halt_and_resume(
        tmp_path, smoke_config, source, identity, process_drill_assets, phase, signum):
    partial, expected, prefix, unit = process_drill_assets
    root = tmp_path / "study"
    shutil.copytree(partial, root)
    sealed_paths = {p.name for p in (root / "units").iterdir() if (p / "COMPLETED").is_file()}
    immutable_before = {name: digest for name, digest in scientific_files(root).items()
                        if Path(name).parts[1] in sealed_paths}

    def child(stage, signum):
        return subprocess.run(
            [sys.executable, "-c", PROCESS_DRILL, str(root), json.dumps(identity),
             json.dumps(smoke_config.mapping()), str(source.train.root), stage, signal.Signals(signum).name],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )

    halted = child(phase, signum)
    assert f"FAULT_REACHED:{phase}" in halted.stdout, halted.stdout + halted.stderr
    assert halted.returncode == (77 if signum == signal.SIGTERM else -signum), halted.stdout + halted.stderr
    if signum == signal.SIGTERM:
        assert "GRACEFUL_STOP" in halted.stdout
    with open_store(root, identity, resume=True) as store:
        assert_scientifically_equal(store.load_checkpoint(unit), prefix, "last_accepted_checkpoint")
        assert not store.is_complete(unit)
    after_halt = scientific_files(root)
    assert {name: after_halt[name] for name in immutable_before} == immutable_before

    finished = child("resume", signal.SIGKILL)
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert "RESUMED" in finished.stdout
    with open_store(root, identity, resume=True) as store:
        assert store.is_complete(unit, full=True)
        assert_scientifically_equal(store.load_checkpoint(unit), expected, "fresh_process_resume")
    after = scientific_files(root)
    assert {name: after[name] for name in immutable_before} == immutable_before
