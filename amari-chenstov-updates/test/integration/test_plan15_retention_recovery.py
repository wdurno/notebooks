"""Explicitly authorized Phase 8.1 cached-MNIST scheduler/recovery acceptance."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from mnist_experiment.fisher_memory_retention.run import main
from mnist_experiment.fisher_memory_retention.config import Config
from mnist_experiment.fisher_memory_regression.config import ROOT
from mnist_experiment.fisher_memory_regression.runtime import file_digest
from mnist_experiment.fisher_memory.artifacts import iter_units, Store


def args(budget,name,device="cpu",*extra):
    return ["--root",str(budget/name),"--budget-root",str(budget),"--smoke","--device",device,
            "--allocation-hours","1",*extra]


def seals(root):
    return {str(p.relative_to(root)):file_digest(p) for u in (root/"units").iterdir()
            if (u/"COMPLETED").exists() for p in u.iterdir() if p.is_file()}


def test_scheduler_unit_batch_resume_and_completed_replay_are_immutable(tmp_path):
    assert main(args(tmp_path,"study","cpu","--max-units","1"))==2
    root=tmp_path/"study";before=seals(root)
    assert main(args(tmp_path,"study","cpu","--resume"))==0
    assert before.items()<=seals(root).items()
    completed=seals(root)
    assert main(args(tmp_path,"study","cpu","--resume"))==0
    assert seals(root)==completed
    final=json.loads((root/"status.json").read_text())
    assert final["status"]=="completed"
    assert all(u["status"]=="completed" for u in iter_units(root,full=True))
    result=json.loads((root/"units/decision-analysis/result.json").read_text())
    assert len(result["methods"])==3


def test_hard_kill_during_initialization_recovers_without_modifying_completed_units(tmp_path):
    command=[sys.executable,"-m","mnist_experiment.fisher_memory_retention.run",*args(tmp_path,"killed","cpu","--stage","grid")]
    log=(tmp_path/"child.log").open("w")
    process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    root=tmp_path/"killed"
    try:
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if list((root/"units").glob("assets-*/latest.json")): break
            if process.poll() is not None: raise AssertionError((tmp_path/"child.log").read_text())
            time.sleep(.02)
        else: raise AssertionError("child did not checkpoint initialization within one minute")
        process.kill();process.wait(timeout=10)
    finally:
        if process.poll() is None: process.kill();process.wait(timeout=10)
        log.close()
    assert main(args(tmp_path,"killed","cpu","--resume"))==0
    assert all(u["status"]=="completed" for u in iter_units(root,full=True))


@pytest.mark.parametrize("stop",["pretrain","mid_solve","eviction","accepted"])
def test_new_protocol_recovers_from_atomic_runner_boundaries(tmp_path,stop):
    assert main(args(tmp_path,"study","cpu","--smoke-stop-at",stop))==2
    assert main(args(tmp_path,"study","cpu","--resume"))==0


@pytest.mark.skipif(os.environ.get("RETENTION_ACCEPTANCE_DEVICE")!="cuda",reason="explicit CUDA acceptance only")
def test_cuda_scheduler_resume_and_stop_file(tmp_path):
    assert torch.cuda.is_available()
    assert main(args(tmp_path,"cuda","cuda","--max-units","1"))==2
    root=tmp_path/"cuda";before=seals(root)
    (root/"STOP").touch()
    assert main(args(tmp_path,"cuda","cuda","--resume"))==2
    assert main(args(tmp_path,"cuda","cuda","--resume","--clear-stop"))==0
    assert before.items()<=seals(root).items()
    assert all(u["status"]=="completed" for u in iter_units(root,full=True))
