"""Explicit tiny CPU acceptance: interruption preserves the exact learner path."""

from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.diagonal_lfu.curvature import build_model, observation
from mnist_experiment.diagonal_lfu.trajectory import run_trajectory, initialize
from mnist_experiment.fisher_memory.artifacts import Store, ImmutableError
from mnist_experiment.fisher_memory.control import StopRequested


def inputs():
    generator = torch.Generator().manual_seed(19)
    x = torch.rand(30, 1, 28, 28, generator=generator, dtype=torch.float64)
    y = torch.arange(30) % 10
    model, layout = build_model(4, dtype=torch.float64)
    theta = layout.flatten_module(model, detach=True)
    initial = {"theta": theta, "diagonal": observation(model, layout, theta, x[:4]),
               "initial_observed_nines": 1}
    assets = {"eval_x": x, "eval_y": y, "non9_weights": torch.ones(9) / 9}
    stream = {"ids": torch.arange(12).reshape(6, 2), "p": torch.full((6,), .1),
              "segments": ["hold"] * 6, "expected_cumulative_nines": torch.arange(1, 7) * .2}
    cfg = SimpleNamespace(device="cpu", dtype="float64", samples_per_step=2,
                          initialization_size=64, chunk_size=2, max_iter=2, max_eval=4,
                          eval_every=2, history_size=10, learning_rate=1.,
                          tolerance_grad=1e-7, tolerance_change=1e-9)
    metadata = dict(stage="development", c=.05, ridge=.1, condition="paired_fd")
    return cfg, initial, assets, stream, x, y, metadata


class Budget:
    def __init__(self, store):
        self.store = store

    def check(self):
        self.store.check()


def test_cpu_stop_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(1)
    cfg, initial, assets, stream, x, y, metadata = inputs()
    with Store(tmp_path / "whole", {"test": 16}) as store:
        whole = run_trajectory(store, Budget(store), cfg, "trajectory", metadata,
                               initial, assets, stream, x, y)
        assert store.is_complete("trajectory")
        with pytest.raises(ImmutableError):
            store.write_json("trajectory", "other.json", {})
    root = tmp_path / "resumed"
    with Store(root, {"test": 16}) as store:
        def stop(**event):
            if event.get("step") == 3:
                raise StopRequested("acceptance interruption")
        with pytest.raises(StopRequested):
            run_trajectory(store, Budget(store), cfg, "trajectory", metadata,
                           initial, assets, stream, x, y, pulse=stop)
        assert store.load_checkpoint("trajectory")["step"] == 3
        assert not store.is_complete("trajectory")
    with Store(root, {"test": 16}, resume=True) as store:
        resumed = run_trajectory(store, Budget(store), cfg, "trajectory", metadata,
                                 initial, assets, stream, x, y)
        torch.testing.assert_close(whole["theta"], resumed["theta"], atol=0, rtol=0)
        torch.testing.assert_close(whole["diagonal"], resumed["diagonal"], atol=0, rtol=0)
        assert [v["step"] for v in resumed["metrics"]] == [0, 2, 4, 6]
        assert resumed["q"] == whole["q"]
        for a, b in zip(whole["metrics"], resumed["metrics"]):
            for key in ("nine_recall", "old_accuracy", "nll", "observed_nines"):
                assert a[key] == b[key]


def test_stop_after_sealing_preserves_original_stop_and_completed_artifact(tmp_path):
    torch.set_num_threads(1)
    cfg, initial, assets, stream, x, y, metadata = inputs()
    with Store(tmp_path / "sealed-stop", {"test": 16}) as store:
        complete = store.complete
        def seal_then_stop(*args, **kwargs):
            result = complete(*args, **kwargs)
            raise StopRequested("stop immediately after seal")
        store.complete = seal_then_stop
        with pytest.raises(StopRequested, match="immediately after seal"):
            run_trajectory(store, Budget(store), cfg, "trajectory", metadata,
                           initial, assets, stream, x, y)
        assert store.is_complete("trajectory")


def test_initialization_consumes_the_recorded_model_and_order_seeds(tmp_path, monkeypatch):
    import mnist_experiment.diagonal_lfu.trajectory as module
    torch.set_num_threads(1)
    cfg, _, assets, _, x, y, _ = inputs()
    cfg.initial_lr, cfg.initial_epochs, cfg.initial_batch_size = .003, 1, 8
    assets.update(init_x=x[:16], init_y=y[:16], validation_x=x[16:24],
                  validation_y=y[16:24], reference_x=x[24:28],
                  seeds={"initialization_model": 1234, "initialization_order": 4321})
    seen = []
    original = module.build_model
    def recorded(seed, **kwargs):
        seen.append(seed)
        return original(seed, **kwargs)
    monkeypatch.setattr(module, "build_model", recorded)
    with Store(tmp_path / "seeds", {"test": 16}) as store:
        result = initialize(store, Budget(store), cfg, "initial", assets, 0, "development")
        assert seen == [1234]
        expected = torch.Generator().manual_seed(4321)
        torch.randperm(16, generator=expected)
        torch.testing.assert_close(result["generator"], expected.get_state())
