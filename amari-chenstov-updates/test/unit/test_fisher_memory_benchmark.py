"""Benchmark contract tests with all fits, inference and Fisher work stubbed."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment.fisher_memory import benchmark as module
from mnist_experiment.fisher_memory.config import CONDITIONS, Config


class FakeStore:
    def __init__(self, root):
        self.root = root
        self.identity = {"source_hash": "test-source"}
        self.metadata, self.checkpoints = {}, {}
        self.completed = {"assets-acceptance-rotation-0000"}
        self.unit_path("assets-acceptance-rotation-0000").mkdir(parents=True)

    def unit_path(self, unit):
        return self.root / "units" / unit

    def is_complete(self, unit):
        return unit in self.completed

    def begin(self, unit, metadata):
        self.unit_path(unit).mkdir(parents=True, exist_ok=True)
        self.metadata[unit] = metadata

    def check(self):
        pass

    def load_checkpoint(self, unit):
        return copy.deepcopy(self.checkpoints.get(unit))

    def save_checkpoint(self, unit, state):
        self.checkpoints[unit] = copy.deepcopy(state)

    def write_json(self, unit, name, value):
        (self.unit_path(unit) / name).write_text(json.dumps(value, allow_nan=False))

    def complete(self, unit):
        self.completed.add(unit)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    cfg = Config(smoke=True, device="cpu", dtype="float64", inner_steps=2, max_evaluations=5, probes=1)
    store = FakeStore(tmp_path)
    source = SimpleNamespace(train_labels=torch.arange(1200) % 10, test_labels=torch.arange(1200) % 10)
    source.images = lambda ids, test=False: torch.tensor(ids, dtype=torch.float32)[:, None, None, None].expand(-1, 1, 28, 28)
    theta = torch.zeros(663, dtype=torch.float64)
    shared = {"learner": {"theta": theta, "fisher_L": torch.eye(487), "anchor_L": theta[:487].clone()},
              "assets": {"plan": {"initialization_ids": list(range(30)), "reference_ids": [30, 31],
                                    "stream_ids": list(range(32, 50))}}}
    calls, references, evaluations = [], [], []
    model, layout = object(), object()
    value = SimpleNamespace(cfg=cfg, store=store, source=source, shared=shared, calls=calls,
                            references=references, evaluations=evaluations, asset_calls=0, fail_condition=None)

    def assets(*args):
        value.asset_calls += 1
        return "assets-acceptance-rotation-0000", shared

    def fit(model, layout, start, nx, ny, rx, ry, ages, fisher, anchor, condition, settings, seed, check):
        if condition == value.fail_condition:
            raise RuntimeError("simulated interrupt")
        calls.append(dict(condition=condition, theta=start.clone(), new=nx.clone(), replay=rx.clone(),
                          ages=ages.clone(), fisher=fisher.clone(), anchor=anchor.clone(), settings=settings, seed=seed))
        return start + .01, {"total_closure_calls": settings.max_iter, "solver_failed": False}

    def reference(model, layout, theta, x, *, indices, chunk_size):
        references.append((theta.clone(), x.clone(), indices.clone(), chunk_size))
        return torch.eye(487)

    def logits(model, layout, theta, x):
        evaluations.append((theta.clone(), x.clone()))
        return torch.zeros(len(x), 10)

    value._assets, value._model, value._logits = assets, lambda shared: (model, layout), logits
    monkeypatch.setattr(module, "fit_step", fit)
    monkeypatch.setattr(module, "reference_fisher", reference)
    monkeypatch.setattr(module, "memory_indices", lambda layout: (torch.arange(487), torch.arange(487, 663)))
    monkeypatch.setattr(module, "effective_residuals", lambda model, layout, theta: {"residual": theta[487:]})
    return value


def test_production_dimensions_paired_start_and_timing_only_metadata(runner):
    original = runner.shared["learner"]["theta"].clone()
    result = module.benchmark(runner)
    assert result["kind"] == "benchmark"
    assert result["warmed"]
    assert result["shared_initialization"]["low_fidelity"]
    assert result["shared_initialization"]["reference_samples"] == 2
    assert result["provenance_hash"]
    assert list(result["methods"]) == list(CONDITIONS)
    assert len(runner.calls) == 7
    assert runner.calls[0]["settings"].max_iter == 5
    for call in runner.calls[1:]:
        assert call["new"].shape == (4, 1, 28, 28)
        assert call["replay"].shape == (32, 1, 28, 28)
        assert (call["settings"].max_iter, call["settings"].max_eval, call["settings"].num_probes) == (50, 75, 2)
        assert call["settings"].solver == "lbfgs"
        assert call["new"].dtype == torch.float64
        assert torch.equal(call["theta"], runner.calls[1]["theta"])
        assert torch.equal(call["fisher"], runner.calls[1]["fisher"])
        assert torch.equal(call["anchor"], call["theta"][:487])
        assert call["seed"] == runner.calls[1]["seed"]
        assert torch.allclose(call["ages"], torch.arange(32, 0, -4).repeat_interleave(4).double().div(32).square())
    assert torch.equal(runner.shared["learner"]["theta"], original)
    assert len(set(result["training_ids"])) == 36
    assert not set(result["training_ids"]).intersection(range(50))
    assert torch.bincount(runner.source.test_labels[result["evaluation_ids"]]).tolist() == [100] * 10
    assert runner.references[0][1].shape[0] == 4
    assert runner.references[0][3] == 4
    assert runner.evaluations[0][1].shape[0] == 1000
    assert all(row["seconds"] >= 0 for row in result["methods"].values())
    json.dumps(result, allow_nan=False)


def test_completed_benchmark_is_immutable_and_reused_without_fits(runner):
    first = module.benchmark(runner)
    count = len(runner.calls)
    assert module.benchmark(runner) == first
    assert len(runner.calls) == count
    assert runner.asset_calls == 1


def test_benchmark_cannot_launch_missing_initialization(runner):
    runner.store.completed.clear()
    with pytest.raises(ValueError, match="no pretraining"):
        module.benchmark(runner)
    assert runner.asset_calls == 0
    assert not runner.calls


def test_interruption_resumes_only_remaining_methods(runner):
    runner.fail_condition = "finite_difference"
    with pytest.raises(RuntimeError, match="interrupt"):
        module.benchmark(runner)
    completed_calls = len(runner.calls)
    assert [call["condition"] for call in runner.calls] == ["replay_ewc", "replay_ewc", "jvp"]
    runner.fail_condition = None
    result = module.benchmark(runner)
    assert len(runner.calls) == completed_calls + 4
    assert set(result["methods"]) == set(CONDITIONS)


def test_already_nonzero_stm_does_not_warm_again(runner):
    runner.shared["learner"]["theta"][487:] = .5
    result = module.benchmark(runner)
    assert not result["warmed"]
    assert len(runner.calls) == 6


def test_timing_synchronizes_before_and_after_cuda_operation(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("synchronize"))
    value, seconds = module._timed(torch.device("cuda"), lambda: calls.append("operation") or 7)
    assert calls == ["synchronize", "operation", "synchronize"]
    assert value == 7 and seconds >= 0
