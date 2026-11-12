"""Actual K interruption/recovery and forced-pi fidelity on cached MNIST."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest
import torch

from mnist_experiment.combined_memory.adaptive_f1.config import REPO, DATA

pytestmark = pytest.mark.skipif(not (DATA/"MNIST/raw/train-images-idx3-ubyte").exists(), reason="cached MNIST required")


def command(root):
    return [sys.executable, "-u", "-m", "mnist_experiment.combined_memory.adaptive_f1.run",
        "--root", str(root), "--notebook", str(root/"results.ipynb"), "--smoke", "--device", "cpu",
        "--replicas", "2", "--resume"]


def execute(root, extra=(), expected=0):
    result = subprocess.run(command(root)+list(extra), cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=600)
    assert result.returncode == expected, result.stdout


def payload(root, policy, namespace="fresh"):
    unit = next((root/"units").glob(f"{namespace}-{policy}--*0000-*"))
    pointer = json.loads((unit/"latest.json").read_text())["current"]
    return torch.load(unit/pointer["file"], map_location="cpu", weights_only=False)["payload"]


def compare(root, direct):
    for p in ("R", "K"):
        a, b = payload(root, p), payload(direct, p)
        assert torch.equal(a["learner"]["theta"], b["learner"]["theta"])
        for field in ("factor", "residual_diagonal"):
            assert torch.equal(a["learner"]["fisher_L"][field], b["learner"]["fisher_L"][field])
        assert a["retirements"] == b["retirements"]
        assert a["learner"]["replay"] == b["learner"]["replay"]
        if p == "K":
            assert a["q_ref"] == b["q_ref"] and a["q_archive"] == b["q_archive"]
            assert not a["pending_scores"] and a["group_pi"] == b["group_pi"]
            assert a["controller"]["assimilations"] == b["controller"]["assimilations"]
            for field in ("mean", "covariance"):
                assert torch.equal(a["controller"][field], b["controller"][field])


@pytest.fixture(scope="module")
def direct(tmp_path_factory):
    root = tmp_path_factory.mktemp("adaptive-f1-direct")
    execute(root)
    return root


def test_real_sigterm_recovers_K_and_preserves_completed_units(tmp_path, direct):
    root = tmp_path/"signal"
    with (tmp_path/"signal.log").open("w+") as log:
        worker = subprocess.Popen(command(root), cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic()+240
            while time.monotonic() < deadline:
                if list((root/"units").glob("fresh-K--*trajectory*/latest.json")):
                    break
                if worker.poll() is not None:
                    log.seek(0); pytest.fail(log.read())
                time.sleep(.05)
            else:
                pytest.fail("K checkpoint not reached")
            os.killpg(worker.pid, signal.SIGTERM)
            worker.wait(timeout=60)
            log.seek(0)
            assert worker.returncode == 2, log.read()
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
    seals = {p: p.read_bytes() for p in (root/"units").glob("*/COMPLETED")}
    execute(root)
    assert all(p.read_bytes()==value for p, value in seals.items())
    compare(root, direct)


@pytest.mark.parametrize("boundary", ["split_group", "mid_solve", "eviction"])
def test_K_atomic_boundaries_resume_exactly(tmp_path, direct, boundary):
    root = tmp_path/boundary
    execute(root, ("--smoke-stop-at", boundary), expected=2)
    if boundary=="split_group":
        saved = payload(root, "K", "acceptance")
        assert saved["pending_scores"] and saved["group_pi"] is not None
    execute(root)
    compare(root, direct)
