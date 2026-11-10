"""Actual SIGTERM and resumption of the requested reproduction pipeline."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from mnist_experiment.combined_memory.reproduction.config import REPO, DATA

pytestmark = pytest.mark.skipif(not (DATA/"MNIST/raw/train-images-idx3-ubyte").exists(),
                                reason="cached MNIST required")


def command(root):
    return [sys.executable,"-u","-m","mnist_experiment.combined_memory.reproduction.run",
            "--root",str(root),"--smoke","--device","cpu","--replicas","2",
            "--notebook",str(root/"smoke.ipynb"),"--resume"]


def payload(root):
    unit = next((root/"units").glob("fresh-blur--head_r1-trajectory-*-fd_replay_ewc-*"))
    latest = json.loads((unit/"latest.json").read_text())["current"]
    return torch.load(unit/latest["file"],map_location="cpu",weights_only=False)["payload"]


def test_sigterm_resume_preserves_scientific_state(tmp_path):
    root = tmp_path/"resumed"
    with (tmp_path/"interrupted.log").open("w+") as log:
        worker = subprocess.Popen(command(root),cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            deadline = time.monotonic()+180
            while time.monotonic()<deadline:
                if list((root/"units").glob("historical-blur--*trajectory*/latest.json")):
                    break
                if worker.poll() is not None:
                    log.seek(0)
                    pytest.fail(log.read())
                time.sleep(.05)
            else:
                pytest.fail("no accepted checkpoint before deadline")
            os.killpg(worker.pid,signal.SIGTERM)
            worker.wait(timeout=60)
            assert worker.returncode == 2
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid,signal.SIGKILL)
                worker.wait()
    for study in (root,tmp_path/"direct"):
        result = subprocess.run(command(study),cwd=REPO,text=True,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,timeout=600)
        assert result.returncode == 0,result.stdout
        assert (study/"study_completed.json").exists()
    left,right = payload(root),payload(tmp_path/"direct")
    assert torch.equal(left["learner"]["theta"],right["learner"]["theta"])
    assert left["retirements"] == right["retirements"]
    assert left["learner"]["replay"] == right["learner"]["replay"]
    assert torch.equal(left["learner"]["fisher_L"]["factor"],right["learner"]["fisher_L"]["factor"])
    assert torch.equal(left["learner"]["fisher_L"]["residual_diagonal"],
                       right["learner"]["fisher_L"]["residual_diagonal"])
