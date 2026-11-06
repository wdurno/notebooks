"""An actual process death must preserve accepted calibrated-policy arrivals."""

import multiprocessing
import os
import importlib.util
from pathlib import Path

import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.kalman_pi import trajectory

_spec = importlib.util.spec_from_file_location("plan17_resume_helpers",
    Path(__file__).with_name("test_kalman_pi_resume.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
Budget, assert_same, inputs = _helpers.Budget, _helpers.assert_same, _helpers.inputs


def _parameters(ridge):
    cfg, initial, assets, stream, x, y, meta, calibration = inputs("kalman_eta0.001")
    cfg.anchor_refresh = 0
    meta.update(c=.025, ridge=ridge)
    return cfg, initial, assets, stream, x, y, meta, calibration


def _abrupt_worker(root, ridge):
    torch.set_num_threads(1)
    cfg, initial, assets, stream, x, y, meta, calibration = _parameters(ridge)
    with Store(root, {"test": "phase5-process-death", "ridge": ridge}) as store:
        def kill_after_checkpoint(**event):
            if event.get("step") == 3:
                os._exit(73)
        trajectory.run_trajectory(store, Budget(store), cfg, "trajectory", meta,
            initial, assets, stream, x, y, calibration=calibration, pulse=kill_after_checkpoint)


@pytest.mark.parametrize("ridge", (0., .03))
def test_actual_process_death_preserves_unequal_ridge_policy_state(tmp_path, ridge):
    torch.set_num_threads(1)
    cfg, initial, assets, stream, x, y, meta, calibration = _parameters(ridge)
    identity = {"test": "phase5-process-death", "ridge": ridge}
    with Store(tmp_path / "whole", identity) as store:
        whole = trajectory.run_trajectory(store, Budget(store), cfg, "trajectory", meta,
            initial, assets, stream, x, y, calibration=calibration)
    root = tmp_path / "interrupted"
    process = multiprocessing.get_context("spawn").Process(target=_abrupt_worker, args=(root, ridge))
    process.start()
    process.join(30)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail("abrupt-interruption child exceeded its bounded timeout")
    assert process.exitcode == 73
    with Store(root, identity, resume=True) as store:
        accepted = store.load_checkpoint("trajectory")
        assert accepted["step"] == 3
        resumed = trajectory.run_trajectory(store, Budget(store), cfg, "trajectory", meta,
            initial, assets, stream, x, y, calibration=calibration)
        assert_same(whole, resumed)
        assert resumed["step"] == 6
        assert len(resumed["step_metrics"]) == 6
        assert store.is_complete("trajectory", full=True)
