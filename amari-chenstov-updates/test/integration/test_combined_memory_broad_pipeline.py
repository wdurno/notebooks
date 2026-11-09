"""Explicit cached-MNIST CPU acceptance of the Phase 5 CLI lifecycle.

Each CLI invocation admits at most the first A trajectory, not the full study.
Run through combined_memory.acceptance so subprocess wall time is charged.
"""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from mnist_experiment.combined_memory.broad_drift.config import Config
from mnist_experiment.combined_memory.broad_drift.runtime import DATA_ROOT, FAMILY, REPO
from mnist_experiment.fisher_memory.artifacts import iter_units


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX SIGTERM acceptance")
PUBLIC_NOTEBOOK = REPO / "mnist_experiment/plan18_results.ipynb"
TIME_FIELDS = {"learning_seconds_100"}


@pytest.fixture(scope="module")
def cli_config(tmp_path_factory):
    raw = DATA_ROOT / "MNIST/raw"
    required = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    if not all((raw / name).is_file() for name in required):
        pytest.skip("the four cached MNIST IDX files are required; no downloads")
    cfg = replace(Config.smoke_config(device="cpu"), threads=1, max_iter=2, max_eval=8)
    cfg.validate()
    path = tmp_path_factory.mktemp("plan18-broad-cli-config") / "smoke.json"
    path.write_text(json.dumps(cfg.mapping(), sort_keys=True))
    return cfg, path


def science(value):
    if isinstance(value, dict):
        return {key: science(item) for key, item in value.items()
                if key != "rng" and not str(key).endswith("_seconds") and key not in TIME_FIELDS}
    if isinstance(value, (list, tuple)):
        return type(value)(science(item) for item in value)
    return value


def assert_same(actual, expected, where="scientific state"):
    assert type(actual) is type(expected), where
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=where)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), where
        for key in actual:
            assert_same(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), where
        for index, (first, second) in enumerate(zip(actual, expected)):
            assert_same(first, second, f"{where}[{index}]")
    else:
        assert actual == expected, where


def fingerprint(path):
    if not path.exists():
        return None
    return (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, path.stat().st_mtime_ns)


def sealed_files(root):
    return {file.relative_to(root).as_posix(): fingerprint(file)
            for unit in (root / "units").glob("*") if (unit / "COMPLETED").exists()
            for file in unit.rglob("*") if file.is_file()}


def checkpoint(unit):
    pointer = json.loads((unit / "latest.json").read_text())
    return torch.load(unit / pointer["current"]["file"], map_location="cpu", weights_only=False)["payload"]


def first_path(root, *, completed):
    paths = [row for row in iter_units(root, full=True) if row["metadata"].get("kind") == "trajectory"]
    assert len(paths) == 1
    row = paths[0]
    assert row["metadata"]["cohort"] == "development" and row["metadata"]["replica"] == 0
    assert row["metadata"]["recipe"]["arm"] == "A"
    assert (row["status"] == "completed") is completed
    return row, checkpoint(row["path"])


@contextmanager
def cli(root, config, log, *extra):
    assert not root.resolve().is_relative_to(FAMILY.resolve())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLCONFIGDIR=str(config.parent / "matplotlib"))
    command = [sys.executable, "-m", "mnist_experiment.combined_memory.broad_drift.run", "--config", str(config),
               "--root", str(root), "--data-root", str(DATA_ROOT), "--max-units", "1", *extra]
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)


def clean_exit(process, log, timeout=180):
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"CLI did not exit within {timeout}s:\n{log.read_text()}")
    output = log.read_text()
    assert code == 0, output
    assert "Traceback" not in output, output
    return output


def run_cli(root, config, log, *extra):
    with cli(root, config, log, *extra) as process:
        return clean_exit(process, log)


def signal_after_first_accepted_batch(process, root, log, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert process.poll() is None, log.read_text()
        for unit in (root / "units").glob("path-development-000-A-*"):
            try:
                state = checkpoint(unit)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                continue
            if state["step"] == 1:
                process.send_signal(signal.SIGTERM)
                return state
            assert state["step"] < 1, "polling missed the first accepted checkpoint"
        time.sleep(.01)
    pytest.fail(f"no accepted step-1 checkpoint within {timeout}s:\n{log.read_text()}")


def test_cli_sigterm_resume_matches_uninterrupted_first_a_and_report_stays_local(tmp_path, cli_config):
    cfg, config = cli_config
    resumed_root, whole_root = tmp_path / "resumed", tmp_path / "whole"
    public_before = fingerprint(PUBLIC_NOTEBOOK)
    stop_log = tmp_path / "sigterm.log"
    with cli(resumed_root, config, stop_log) as process:
        observed = signal_after_first_accepted_batch(process, resumed_root, stop_log)
        output = clean_exit(process, stop_log, timeout=60)
    assert '"status": "paused"' in output and "stop requested" in output
    assert json.loads((resumed_root / "status.json").read_text())["status"] == "paused"
    assert json.loads((resumed_root / "runtime.json").read_text())["stop_requested"]
    paused_record, paused = first_path(resumed_root, completed=False)
    assert paused["step"] == 1
    assert_same(science(paused), science(observed), "accepted SIGTERM checkpoint")
    sealed_before_resume = sealed_files(resumed_root)

    resumed_output = run_cli(resumed_root, config, tmp_path / "resume.log", "--resume")
    assert "requested invocation trajectory count" in resumed_output
    assert all(fingerprint(resumed_root / name) == record for name, record in sealed_before_resume.items())
    resumed_record, resumed = first_path(resumed_root, completed=True)
    assert resumed_record["unit_id"] == paused_record["unit_id"]
    assert resumed["step"] == cfg.num_p_steps
    assert len(resumed["diagnostics"]) == cfg.num_p_steps
    assert [row["batch"] for row in resumed["diagnostics"]] == list(range(1, cfg.num_p_steps + 1))
    assert resumed["retired_ids"] + resumed["replay"] == list(range(cfg.stream_size))
    assert any(row["moved"] for row in resumed["diagnostics"])

    run_cli(whole_root, config, tmp_path / "whole.log")
    whole_record, whole = first_path(whole_root, completed=True)
    assert_same(science(resumed), science(whole))
    assert_same(resumed_record["metadata"], whole_record["metadata"], "trajectory metadata")
    for unit in ("assets-development-000", "initial-development-000"):
        assert_same(science(checkpoint(resumed_root / "units" / unit)),
                    science(checkpoint(whole_root / "units" / unit)), unit)
    for name in ("metrics.json", "diagnostics.json", "summary.json"):
        assert_same(science(json.loads((resumed_record["path"] / name).read_text())),
                    science(json.loads((whole_record["path"] / name).read_text())), name)
    for row in resumed["diagnostics"]:
        assert row["replay_seed"] == cfg.seed_for("development", 0, f"replay:{row['batch']-1}")
        assert row["probe_seed"] == cfg.seed_for("development", 0, f"probes:{row['batch']-1}")

    seals = sealed_files(resumed_root)
    run_cli(resumed_root, config, tmp_path / "report-only.log", "--report-only")
    assert fingerprint(PUBLIC_NOTEBOOK) == public_before
    assert (resumed_root / "smoke_results.ipynb").is_file()
    assert sealed_files(resumed_root) == seals
    report_pointer = json.loads((resumed_root / "report/latest.json").read_text())
    report_generation = resumed_root / "report" / report_pointer["generation"]
    assert (report_generation / "COMPLETED").is_file()
    assert (report_generation / "results.ipynb").read_bytes() == (resumed_root / "smoke_results.ipynb").read_bytes()
    assert not (resumed_root / "study_completed.json").exists()


def test_cli_zero_invocation_wall_budget_is_a_controlled_exit(tmp_path, cli_config):
    _, config = cli_config
    root, log = tmp_path / "wall-limit", tmp_path / "wall-limit.log"
    public_before = fingerprint(PUBLIC_NOTEBOOK)
    started = time.monotonic()
    output = run_cli(root, config, log, "--max-wall-seconds", "0")
    assert time.monotonic() - started < 30
    assert "wall-clock allocation exhausted" in output
    assert '"status": "paused"' in output or '"status": "resource_limited"' in output
    assert not list((root / "units").glob("path-*"))
    assert fingerprint(PUBLIC_NOTEBOOK) == public_before

