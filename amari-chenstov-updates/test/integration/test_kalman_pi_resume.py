"""Tiny CPU acceptance of the complete observer/learner checkpoint boundary."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.diagonal_lfu.curvature import build_model, observation
from mnist_experiment.fisher_memory.artifacts import Store, ImmutableError
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.kalman_pi import trajectory
from mnist_experiment.kalman_pi.observations import observed_scores


class Budget:
    def __init__(self, store):
        self.store = store

    def check(self):
        self.store.check()


def inputs(condition):
    generator = torch.Generator().manual_seed(41)
    x = torch.rand(30, 1, 28, 28, generator=generator, dtype=torch.float64)
    y = torch.arange(30) % 10
    model, layout = build_model(9, dtype=torch.float64)
    theta = layout.flatten_module(model, detach=True)
    initial = {"theta": theta, "diagonal": observation(model, layout, theta, x[:4]),
               "initial_observed_nines": 1}
    calibration = {"variance": observed_scores(model, layout, theta, x[:8], y[:8]).var(0), "wall_seconds": 0.}
    assets = {"eval_x": x, "eval_y": y, "non9_weights": torch.ones(9) / 9}
    stream = {"ids": torch.arange(12).reshape(6, 2), "p": torch.full((6,), .1),
              "segments": ["hold"] * 6, "expected_cumulative_nines": torch.arange(1, 7) * .2}
    cfg = SimpleNamespace(device="cpu", dtype="float64", samples_per_step=2,
                          initialization_size=64, chunk_size=2, max_iter=2, max_eval=4,
                          eval_every=2, history_size=10, learning_rate=1., anchor_refresh=2,
                          tolerance_grad=1e-7, tolerance_change=1e-9, cold_start=2,
                          pi_min=1e-4, pi_max=.2, noise_half_life=32.,
                          observer_floor_relative=1e-6, observer_floor_absolute=1e-12)
    metadata = dict(stage="development", c=.05, ridge=.1, condition=condition)
    return cfg, initial, assets, stream, x, y, metadata, calibration


def assert_same(a, b):
    for name in ("theta", "diagonal", "previous"):
        torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
    assert a["q"] == b["q"] and a["log_replacement_mass"] == b["log_replacement_mass"]
    for key, value in a["controller"].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, b["controller"][key], atol=0, rtol=0)
        else:
            assert value == b["controller"][key]
    for left, right in zip(a["metrics"], b["metrics"]):
        for key in ("nine_recall", "old_accuracy", "nll", "observed_nines", "q", "replacement_mass"):
            assert left[key] == right[key]


@pytest.mark.parametrize("condition", ("fixed", "covariance", "ema_h2", "kalman_eta0.001"))
def test_cpu_stop_resume_reproduces_full_policy_trajectory(tmp_path, condition):
    torch.set_num_threads(1)
    cfg, initial, assets, stream, x, y, meta, calibration = inputs(condition)
    with Store(tmp_path / "whole", {"test": 17}) as store:
        whole = trajectory.run_trajectory(store, Budget(store), cfg, "trajectory", meta, initial,
                                          assets, stream, x, y, calibration=calibration)
        with pytest.raises(ImmutableError):
            store.write_json("trajectory", "other.json", {})
    root = tmp_path / "resumed"
    with Store(root, {"test": 17}) as store:
        def stop(**event):
            if event.get("step") == 3:
                raise StopRequested("accepted boundary interruption")
        with pytest.raises(StopRequested):
            trajectory.run_trajectory(store, Budget(store), cfg, "trajectory", meta, initial,
                                       assets, stream, x, y, calibration=calibration, pulse=stop)
        assert store.load_checkpoint("trajectory")["step"] == 3
    with Store(root, {"test": 17}, resume=True) as store:
        resumed = trajectory.run_trajectory(store, Budget(store), cfg, "trajectory", meta, initial,
                                             assets, stream, x, y, calibration=calibration)
        assert_same(whole, resumed)


def test_interruption_during_reanchor_rolls_back_entire_uncommitted_transition(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg, initial, assets, stream, x, y, meta, calibration = inputs("kalman_eta0.001")
    with Store(tmp_path / "whole", {"test": 17}) as store:
        whole = trajectory.run_trajectory(store, Budget(store), cfg, "trajectory", meta, initial,
                                          assets, stream, x, y, calibration=calibration)
    original = trajectory.observed_scores
    calls = 0
    def stop_on_refresh(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise StopRequested("mid-reanchor interruption")
        return original(*args, **kwargs)
    root = tmp_path / "rollback"
    with Store(root, {"test": 17}) as store:
        monkeypatch.setattr(trajectory, "observed_scores", stop_on_refresh)
        with pytest.raises(StopRequested):
            trajectory.run_trajectory(store, Budget(store), cfg, "trajectory", meta, initial,
                                       assets, stream, x, y, calibration=calibration)
        checkpoint = store.load_checkpoint("trajectory")
        assert checkpoint["step"] == checkpoint["controller"]["assimilations"] == 1
    monkeypatch.setattr(trajectory, "observed_scores", original)
    with Store(root, {"test": 17}, resume=True) as store:
        resumed = trajectory.run_trajectory(store, Budget(store), cfg, "trajectory", meta, initial,
                                             assets, stream, x, y, calibration=calibration)
        assert_same(whole, resumed)


def test_stop_after_seal_never_rewrites_completed_unit(tmp_path):
    torch.set_num_threads(1)
    cfg, initial, assets, stream, x, y, meta, calibration = inputs("covariance")
    with Store(tmp_path / "sealed", {"test": 17}) as store:
        original = store.complete
        def seal_stop(*args, **kwargs):
            result = original(*args, **kwargs)
            raise StopRequested("after sealing")
        store.complete = seal_stop
        with pytest.raises(StopRequested, match="after sealing"):
            trajectory.run_trajectory(store, Budget(store), cfg, "trajectory", meta, initial,
                                       assets, stream, x, y, calibration=calibration)
        assert store.is_complete("trajectory")
