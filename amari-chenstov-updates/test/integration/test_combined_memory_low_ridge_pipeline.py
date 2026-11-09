"""Cached-MNIST Phase 7 lifecycle checks, explicitly requested by execution."""

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

from mnist_experiment.combined_memory.low_ridge.config import Config
from mnist_experiment.combined_memory.low_ridge.runtime import FAMILY, REPO

HISTORY = FAMILY / "phase7_history_smoke_cpu_v1"
PUBLIC = REPO / "mnist_experiment/plan18_results.ipynb"
pytestmark = pytest.mark.skipif(not (HISTORY / "study_completed.json").exists(),
    reason="requires completed Phase 6 CPU dependency")


def fingerprints(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "units").rglob("*") if p.is_file() and (p.parent / "COMPLETED").exists()}


def command(config, root):
    return [sys.executable, "-u", "-m", "mnist_experiment.combined_memory.low_ridge.run",
            "--config", str(config), "--root", str(root), "--history-root", str(HISTORY), "--resume"]


def execute(config, root):
    result = subprocess.run(command(config, root), cwd=REPO, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=300)
    assert result.returncode == 0, result.stdout
    assert '"status": "completed"' in result.stdout, result.stdout


def payload(root, arm="B"):
    path = next((root / "units").glob(f"path-fresh-000-{arm}-*"))
    descriptor = json.loads((path / "latest.json").read_text())["current"]
    return torch.load(path / descriptor["file"], map_location="cpu", weights_only=False)["payload"]


def test_sigterm_resume_matches_uninterrupted_scientific_state_and_preserves_history(tmp_path):
    config = tmp_path / "smoke.json"
    config.write_text(json.dumps(Config.smoke_config().mapping()))
    root = tmp_path / "resumed"
    before_public, before_history = PUBLIC.read_bytes(), fingerprints(HISTORY)
    with (tmp_path / "interrupted.log").open("w+") as log:
        process = subprocess.Popen(command(config, root), cwd=REPO, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if list((root / "units").glob("path-*/latest.json")):
                    break
                if process.poll() is not None:
                    log.seek(0)
                    pytest.fail("worker exited before checkpoint: " + log.read())
                time.sleep(.02)
            else:
                pytest.fail("recoverable checkpoint did not appear")
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=60)
            log.seek(0)
            assert process.returncode == 0, log.read()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    frozen = fingerprints(root)
    execute(config, root)
    assert all(fingerprints(root).get(name) == digest for name,digest in frozen.items())
    selected = json.loads((root / "selection.json").read_text())
    assert selected["recipes"]["B"]["ridge"] == 1.
    assert selected["reference_arm"] == "B" and selected["efficiency_reference_arm"] == "A"
    assert selected["matching"]["passed"]
    complete = json.loads((root / "study_completed.json").read_text())
    assert len(complete["integrity"]["complete_paired_blocks"]) == 5
    direct = tmp_path / "uninterrupted"
    execute(config, direct)
    left, right = payload(root), payload(direct)
    for field in ("theta", "diagonal"):
        assert torch.equal(left[field], right[field]), field
    assert left["step"] == right["step"] == Config.smoke_config().num_p_steps
    assert left["retired_ids"] == right["retired_ids"] and left["replay"] == right["replay"]
    assert fingerprints(HISTORY) == before_history
    assert PUBLIC.read_bytes() == before_public


def test_failed_calibration_never_starts_confirmation(tmp_path):
    cfg = replace(Config.smoke_config(), force_calibration_failure=True)
    config, root = tmp_path / "failed.json", tmp_path / "failed"
    config.write_text(json.dumps(cfg.mapping()))
    result = subprocess.run(command(config, root), cwd=REPO, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, result.stdout
    closed = json.loads((root / "calibration_completed.json").read_text())
    assert closed["status"] == "calibration_infeasible" and not closed["launched_confirmation"]
    assert not list((root / "units").glob("path-fresh-*"))
    assert not list((root / "units").glob("path-stationary-*"))
    assert (root / "smoke_results.ipynb").exists()
    assert not (root / "study_completed.json").exists()
