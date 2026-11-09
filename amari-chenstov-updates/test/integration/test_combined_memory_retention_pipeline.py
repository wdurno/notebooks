"""Explicit cached-MNIST Phase 6 lifecycle acceptance; no downloads."""

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

from mnist_experiment.combined_memory.retention_match.config import Config
from mnist_experiment.combined_memory.retention_match.runtime import FAMILY, REPO


HISTORY = FAMILY / "phase6_history_smoke_cpu_v1"
PUBLIC = REPO / "mnist_experiment/plan18_results.ipynb"
pytestmark = pytest.mark.skipif(not (HISTORY / "study_completed.json").exists(),
    reason="requires the explicitly prepared, completed cached CPU history smoke")


def fingerprints(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "units").rglob("*") if p.is_file()
            and (p.parent / "COMPLETED").exists()}


def command(config, root):
    return [sys.executable, "-u", "-m", "mnist_experiment.combined_memory.retention_match.run",
            "--config", str(config), "--root", str(root), "--history-root", str(HISTORY), "--resume"]


def test_sigterm_resume_completes_a_passing_study_without_changing_public_history(tmp_path):
    cfg = Config.smoke_config()
    config = tmp_path / "smoke.json"
    config.write_text(json.dumps(cfg.mapping()))
    root = tmp_path / "resumed"
    before = PUBLIC.read_bytes()
    process = subprocess.Popen(command(config, root), cwd=REPO, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if list((root / "units").glob("path-*/latest.json")):
                break
            if process.poll() is not None:
                pytest.fail("worker exited before its first recoverable trajectory: " + process.communicate()[0])
            time.sleep(.02)
        else:
            pytest.fail("first trajectory checkpoint was not published")
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=30)
        assert process.returncode == 0, output
        assert '"status": "paused"' in output
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    frozen = fingerprints(root)
    result = subprocess.run(command(config, root), cwd=REPO, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, result.stdout
    assert '"status": "completed"' in result.stdout
    selection = json.loads((root / "selection.json").read_text())
    assert selection["matching"]["passed"] is True
    assert selection["members"] == [0, 1, 2, 3]
    final = json.loads((root / "study_completed.json").read_text())
    assert len(final["integrity"]["complete_paired_blocks"]) == 3
    assert all(fingerprints(root).get(name) == digest for name, digest in frozen.items())
    assert PUBLIC.read_bytes() == before


def test_failed_calibration_publishes_diagnostics_and_never_opens_confirmation(tmp_path):
    cfg = replace(Config.smoke_config(), force_calibration_failure=True)
    config = tmp_path / "failure.json"
    config.write_text(json.dumps(cfg.mapping()))
    root = tmp_path / "failed"
    result = subprocess.run(command(config, root), cwd=REPO, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, result.stdout
    closed = json.loads((root / "calibration_completed.json").read_text())
    assert closed["status"] == "calibration_infeasible"
    assert closed["launched_confirmation"] is False
    assert not list((root / "units").glob("path-fresh-*"))
    assert not list((root / "units").glob("path-stationary-*"))
    assert (root / "smoke_results.ipynb").exists()
    assert not (root / "study_completed.json").exists()
