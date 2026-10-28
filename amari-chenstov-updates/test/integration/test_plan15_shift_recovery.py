"""Authorized tiny cached-MNIST Phase 9 scheduler and interruption drills."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import iter_units
from mnist_experiment.fisher_memory_regression.config import ROOT
from mnist_experiment.fisher_memory_regression.runtime import file_digest
from mnist_experiment.fisher_memory_shift import report
from mnist_experiment.fisher_memory_shift.run import main


def args(budget, name, device="cpu", *extra):
    return ["--root",str(budget/name),"--budget-root",str(budget),"--smoke",
            "--device",device,"--allocation-hours","1",*extra]


def seals(root):
    return {str(p.relative_to(root)):file_digest(p) for u in (root/"units").iterdir()
            if (u/"COMPLETED").exists() for p in u.iterdir() if p.is_file()}


@pytest.fixture(autouse=True)
def light_reports(monkeypatch):
    # Real publication is tested separately and in the durable CPU/CUDA smokes.
    monkeypatch.setattr(report, "publish", lambda *args, **kwargs: None)


def test_scheduler_batch_resume_seals_and_stop_file(tmp_path):
    assert main(args(tmp_path,"study","cpu","--max-units","1")) == 2
    root = tmp_path/"study"
    first = seals(root)
    (root/"STOP").touch()
    assert main(args(tmp_path,"study","cpu","--resume")) == 2
    assert main(args(tmp_path,"study","cpu","--resume","--clear-stop")) == 0
    assert first.items() <= seals(root).items()
    complete = seals(root)
    assert main(args(tmp_path,"study","cpu","--resume")) == 0
    assert complete == seals(root)
    assert json.loads((root/"status.json").read_text())["status"] == "completed"
    assert all(u["status"] == "completed" for u in iter_units(root, full=True))


@pytest.mark.parametrize("stop", ["pretrain","mid_solve","eviction","accepted"])
def test_atomic_runner_boundaries_resume_the_same_plan(tmp_path, stop):
    assert main(args(tmp_path,"study","cpu","--smoke-stop-at",stop)) == 2
    root = tmp_path/"study"
    initial = seals(root)
    assert main(args(tmp_path,"study","cpu","--resume")) == 0
    assert initial.items() <= seals(root).items()
    assert all(u["status"] == "completed" for u in iter_units(root, full=True))


def test_hard_kill_initialization_has_bounded_recovery(tmp_path):
    log_path = tmp_path/"child.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([sys.executable,"-m",
            "mnist_experiment.fisher_memory_shift.run",*args(tmp_path,"killed","cpu","--stage","calibrate")],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        root = tmp_path/"killed"
        try:
            deadline = time.monotonic()+60
            while time.monotonic() < deadline:
                if list((root/"units").glob("assets-*/latest.json")):
                    break
                if process.poll() is not None:
                    raise AssertionError(log_path.read_text())
                time.sleep(.02)
            else:
                raise AssertionError("initialization did not checkpoint within one minute")
            process.kill()
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert main(args(tmp_path,"killed","cpu","--resume")) == 0
    assert all(u["status"] == "completed" for u in iter_units(root, full=True))


@pytest.mark.skipif(os.environ.get("SHIFT_ACCEPTANCE_DEVICE") != "cuda",
                    reason="explicit CUDA integration selection")
def test_cuda_resume_and_source_pairing(tmp_path):
    assert torch.cuda.is_available()
    assert main(args(tmp_path,"cuda","cuda","--max-units","1")) == 2
    root = tmp_path/"cuda"
    before = seals(root)
    assert main(args(tmp_path,"cuda","cuda","--resume")) == 0
    assert before.items() <= seals(root).items()
    result = json.loads((root/"units/decision-analysis/result.json").read_text())
    assert set(result["methods"]) == {"fd_replay_ewc","replay_ewc","ewc_only"}
