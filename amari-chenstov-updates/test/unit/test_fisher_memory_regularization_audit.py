"""Frozen-objective reconstruction and isolation from the accepted learner."""

import dataclasses
import json

import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import Store, iter_units
from mnist_experiment.fisher_memory.config import Config, Treatment
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.fisher_memory.learner import FitSettings, fit_step
from mnist_experiment.fisher_memory.model import build_model, memory_indices
from mnist_experiment.fisher_memory.regularization_audit import (
    FrozenObjectiveAudit, scientific_state_hash,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def problem():
    model, layout = build_model(53, dtype=torch.float64)
    theta = layout.flatten_module(model, detach=True)
    ltm, _ = memory_indices(layout)
    generator = torch.Generator().manual_seed(88)
    x = torch.randn(4, 1, 28, 28, generator=generator, dtype=torch.float64)
    y = torch.tensor([0, 1, 9, 3])
    settings = FitSettings(max_iter=2, max_eval=5, num_probes=1,
                           lambda_f=.269, fd_step=.01)
    context = dict(unit="parent", family="regularization_development", replica=0,
        stream="mixed", treatment=Treatment("finite_difference", .269).mapping(),
        step=2, previous_events=0, events=1, before=theta,
        fisher=torch.eye(len(ltm), dtype=torch.float64) * .01,
        anchor=theta[ltm], new_x=x[:2], new_y=y[:2], replay_x=x[2:], replay_y=y[2:],
        ages=torch.ones(2), settings=settings, seed=123, new_ids=[2, 3], replay_ids=[0, 1],
        panels={"rotated": (x, y, torch.full((4,), .25)),
                "base": (x, y, torch.full((4,), .25))}, layout=layout.metadata())
    return model, layout, context


def solve(model, layout, context, observer=None):
    return fit_step(model, layout, context["before"], context["new_x"], context["new_y"],
        context["replay_x"], context["replay_y"], context["ages"], context["fisher"],
        context["anchor"], "finite_difference", context["settings"], context["seed"],
        objective_observer=observer)


def test_reconstructed_audit_preserves_main_solution_and_rng(tmp_path):
    cfg = dataclasses.replace(Config(), smoke=True, device="cpu", dtype="float64",
                               inner_steps=2, max_evaluations=5)
    model, layout, context = problem()
    expected, _ = solve(model, layout, context)
    original = scientific_state_hash(context["before"])
    with Store(tmp_path / "study", {"test": "audit"}) as store:
        audit = FrozenObjectiveAudit(cfg, store, [1])
        actual, diagnostics = solve(model, layout, context, audit.observer(context))
        assert torch.equal(expected, actual)
        assert scientific_state_hash(context["before"]) == original
        rng = torch.random.get_rng_state().clone()
        audit.drain("parent", {"step": 1})
        assert not store.is_complete("audit-parent-event-001")
        audit.drain("parent", {"step": 2})
        assert torch.equal(rng, torch.random.get_rng_state())
        assert store.is_complete("audit-parent-event-001")
        result = json.loads((store.unit_path("audit-parent-event-001") / "result.json").read_text())
        assert result["extended_objective"] <= result["ordinary_objective"]
        assert result["prediction_tv"]["rotated"] >= 0
        assert diagnostics["audit_capture_seconds"] > 0
        seals = {row["unit_id"]: row["completion"]["seal_hash"] for row in iter_units(store.root)}
        audit.drain("parent", {"step": 2})
        assert seals == {row["unit_id"]: row["completion"]["seal_hash"] for row in iter_units(store.root)}


def test_pending_audit_resumes_without_reconstructing_anchor_from_accepted_theta(tmp_path):
    cfg = dataclasses.replace(Config(), smoke=True, device="cpu", dtype="float64")
    model, layout, context = problem()
    root = tmp_path / "study"
    with Store(root, {"test": "resume-audit"}) as store:
        audit = FrozenObjectiveAudit(cfg, store, [1])
        solve(model, layout, context, audit.observer(context))
    with Store(root, {"test": "resume-audit"}, resume=True) as store:
        audit = FrozenObjectiveAudit(cfg, store, [1])
        audit.drain("parent", {"step": 2})
        assert store.is_complete("audit-parent-event-001")


def test_state_hash_tracks_tensor_shape_dtype_and_values():
    first = {"theta": torch.tensor([1., 2.]), "ids": [1, 2]}
    assert scientific_state_hash(first) == scientific_state_hash(dict(reversed(list(first.items()))))
    assert scientific_state_hash(first) != scientific_state_hash({**first, "theta": torch.tensor([1., 3.])})


def test_manifest_without_snapshot_is_recovered_before_parent_commit(tmp_path, monkeypatch):
    cfg = dataclasses.replace(Config(), smoke=True, device="cpu", dtype="float64")
    model, layout, context = problem()
    root, identity = tmp_path / "study", {"test": "capture-crash"}
    with Store(root, identity) as store:
        original_save = store.save_checkpoint
        def crash_after_manifest(*args, **kwargs):
            raise StopRequested("interrupted before snapshot publication")
        monkeypatch.setattr(store, "save_checkpoint", crash_after_manifest)
        with pytest.raises(StopRequested):
            solve(model, layout, context, FrozenObjectiveAudit(cfg, store, [1]).observer(context))
        monkeypatch.setattr(store, "save_checkpoint", original_save)
    with Store(root, identity, resume=True) as store:
        resumed = FrozenObjectiveAudit(cfg, store, [1])
        resumed.drain("parent", {"step": 1})
        solve(model, layout, context, resumed.observer(context))
        resumed.drain("parent", {"step": 2})
        assert store.is_complete("audit-parent-event-001")


def test_new_families_use_disjoint_test_partitions():
    from mnist_experiment.fisher_memory.data import plan_data
    class Source:
        train_labels = torch.arange(10_000) % 10
        test_labels = torch.arange(4_000) % 10
    cfg = dataclasses.replace(Config(), smoke=True, init_size=100, reference_size=20, eval_per_class=5)
    dev = plan_data(Source(), cfg, "regularization_development", 0, "drip")
    confirm = plan_data(Source(), cfg, "regularization_confirmation", 0, "drip")
    assert not set(dev["evaluation_ids"]) & set(confirm["evaluation_ids"])
