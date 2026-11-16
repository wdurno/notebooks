"""Scheduler pairing, recorded selection and immutable resume checks."""

import copy
from dataclasses import replace

import pytest

from numerical_experiment.picar_solver_repair.config import BASE_RECIPE, configuration, solver_steps
from numerical_experiment.picar_solver_repair.data import _spaced, manifest
from numerical_experiment.picar_solver_repair.student import _ConstructingStore, SolverDevelopmentRun


def test_recipe_step_budget_is_explicit_and_common_likelihood():
    configs = [configuration(replace(BASE_RECIPE, optimizer_steps=solver_steps(name)))
        for name in ("proximal_1", "lagged_1", "proximal_3")]
    assert [c.learning.optimizer_steps for c in configs] == [1, 1, 3]
    assert len({(c.learning.temperature, c.learning.action_variance, c.learning.pi,
        c.learning.ridge, c.model.context_length) for c in configs}) == 1
    with pytest.raises(ValueError):
        solver_steps("unknown")


def test_recorded_manifest_separates_validation_and_conditional_restart():
    record = lambda i, ep: {"record_id": f"{ep}/{i}", "episode_id": ep,
        "arrival": i, "source_exposure": float(i)}
    segments = {"early": [record(i, "train-a") for i in range(4)],
        "late": [record(i, "train-b") for i in range(10, 14)]}
    validation = {"old": [record(20, "old-validation")], "live": [record(30, "live-validation")]}
    result = manifest({"asset_id": "before-stream"}, segments, validation)
    assert result["new_observations"] == result["teacher_calls"] == 0
    for part in result["segments"].values():
        assert part["start_asset"] == "before-stream"
        assert part["count"] == 4
        assert not set(part["episode_ids"]) & set(result["validation"]["live"]["episode_ids"])
    assert _spaced(list(range(9)), 3) == [0, 4, 8]


def test_validation_keeps_full_frozen_context_without_replay_trimming():
    import torch
    from numerical_experiment.picar_solver_repair.recorded import evaluate
    from test_picar_learner import history
    h = history(2)
    h.update(decision_time=torch.tensor([[.1, .2]]), start_time=torch.tensor([0.]))
    row = {"history": h, "baseline": 0., "bin": 1, "logits": torch.ones(7),
        "arrival": 0, "episode_id": "heldout"}
    class Learner:
        params = {}
        def nll(self, params, batch):
            assert batch.histories[0] is h
            assert batch.histories[0]["images"].shape[1] == 2
            return torch.tensor([3.])
    assert evaluate(Learner(), {"old": [row], "live": [row]}) == {"old_nll": 3., "live_nll": 3.}


def result(solver, condition, *, old_gap=.02, live=20., cost=.3, status="complete"):
    return {"solver_recipe": solver, "condition": condition, "status": status,
        "validation": {"initial": {"old_nll": 10.}, "final": {"old_nll": 10.+old_gap, "live_nll": live}},
        "timing": {"mean_seconds": cost}}


def test_selection_values_retention_and_compute_not_penalty_size():
    from numerical_experiment.picar_solver_repair.recorded import select_solver
    rows = [result(s, c, live=19. if s == "proximal_3" else 20., cost=.9 if s == "proximal_3" else .3)
        for s in ("proximal_1", "lagged_1", "proximal_3") for c in ("replay_ewc", "stm_fd")]
    assert select_solver(rows)["selected_solver"] == "proximal_3"
    for r in rows:
        if r["solver_recipe"] == "proximal_3":
            r["validation"]["final"]["live_nll"] = 19.99
    assert select_solver(rows)["selected_solver"] == "proximal_1"
    for r in rows:
        if r["solver_recipe"] == "lagged_1":
            r["status"] = "numerical_failure"
    assert select_solver(rows)["selected_solver"] != "lagged_1"
    assert select_solver([])["selected_solver"] is None


def test_temporary_legacy_view_never_mutates_solver_checkpoint():
    class Store:
        def load_checkpoint(self, **kwargs):
            return {"learner": {"schema_version": 2, "optimizer": {"kind": "second_moments"},
                "parameters": {"x": 1.}}, "identity": {"version": "new"}}
    adapter = _ConstructingStore(Store(), {"param_groups": [{"lr": .003}], "state": {}})
    view = adapter.load_checkpoint(trusted=True)
    assert view["learner"]["schema_version"] == 1
    assert adapter.saved["learner"]["schema_version"] == 2
    assert adapter.saved["learner"]["optimizer"] == {"kind": "second_moments"}
    adapter.constructing = False
    assert adapter.load_checkpoint(trusted=True)["learner"]["schema_version"] == 2


def test_solver_runtime_resumes_between_fit_and_retirement(tmp_path, monkeypatch):
    import torch
    import test_picar_fix3_student as fixtures
    from test_picar_phase3_student import MemoryStore
    from test_picar_phase2_student import assert_equal
    from numerical_experiment.picar.config import PicarConfig
    from numerical_experiment.picar.learner import MemoryLearner
    from numerical_experiment.picar import fix3_student
    from numerical_experiment.picar_phase2 import student as phase2_student
    from src.robotics.simulator import SimulatorConfig

    monkeypatch.setattr(fix3_student, "CenteringReward", fixtures.FakeReward)
    monkeypatch.setattr(fix3_student, "CenteringTeacher", fixtures.FakeTeacher)
    config = replace(PicarConfig(seed=71), learning=replace(PicarConfig().learning,
        fd_strength=0., optimizer_steps=1, learning_rate=.003, replay_capacity=2))
    model = fixtures.model()
    legacy = MemoryLearner(model, config.learning, config.seeds(0)).state_dict()
    initial = {"stage": "done", "learner": legacy, "eligible": [], "retired": [],
        "acquired_count": 0, "distill_receipt": {"stage": "test"}, "distinct_bootstrap_samples": 0}
    # No simulator work: exercise only event state/checkpoint construction.
    sim = SimulatorConfig(realtime=True)
    store = MemoryStore(tmp_path / "solver-resume")
    run = SolverDevelopmentRun(model, config, sim, store, lambda: None,
        solver_recipe="proximal_1", mode="pilot", seed=196101, max_transitions=640,
        initial=initial, evaluations=[])
    run.save()
    expected = copy.deepcopy(run.learner.state_dict())
    resumed = SolverDevelopmentRun(fixtures.model(), config, sim, store, lambda: None,
        solver_recipe="proximal_1", mode="pilot", seed=196101, max_transitions=640,
        initial=initial, evaluations=[])
    assert_equal(expected, resumed.learner.state_dict())
    assert resumed.s["solver_recipe"] == "proximal_1"


@pytest.mark.parametrize("solver", ["proximal_1", "lagged_1", "proximal_3"])
def test_recorded_sequence_resume_preserves_model_metric_and_evidence(solver):
    import torch
    from test_picar_learner import history
    from test_picar_initialization_numerics import MemoryStore, equal
    from numerical_experiment.picar.config import PicarConfig
    from numerical_experiment.picar.models import build_tiny_model
    from numerical_experiment.picar.learner import MemoryLearner
    from numerical_experiment.picar_solver_repair.recorded import RecordedRun

    torch.set_num_threads(1)
    make = lambda: build_tiny_model(31, hidden_size=4, width=4, return_bins=3, context_length=2)
    cfg = replace(PicarConfig(), learning=replace(PicarConfig().learning,
        fd_strength=.01, optimizer_steps=solver_steps(solver), replay_capacity=2, fisher_draws=1))
    initial = {"learner": MemoryLearner(make(), cfg.learning, cfg.seeds(0)).state_dict()}
    def record(i, episode):
        h = history()
        h.update(decision_time=torch.tensor([[float(i)+.1]]), start_time=torch.tensor([float(i)]))
        return {"history": h, "episode_id": episode, "arrival": i, "baseline": 0.,
            "bin": 1, "logits": torch.ones(7), "learned": False, "source_exposure": float(i)}
    segments = {"early": [record(i, "train") for i in range(10)]}
    validation = {"old": [record(20, "held-old")], "live": [record(21, "held-live")]}
    baseline = MemoryStore()
    reference = RecordedRun(make(), cfg, initial, segments, validation, baseline, lambda: None,
        solver_recipe=solver, condition="stm_fd")
    reference.run()
    partial = MemoryStore()
    def interrupt():
        if partial.state and len(partial.state["rows"]) == 8:
            raise InterruptedError("test committed event block")
    run = RecordedRun(make(), cfg, initial, segments, validation, partial, interrupt,
        solver_recipe=solver, condition="stm_fd")
    with pytest.raises(InterruptedError):
        run.run()
    resumed = RecordedRun(make(), cfg, initial, segments, validation, partial, lambda: None,
        solver_recipe=solver, condition="stm_fd")
    resumed.run()
    assert len(partial.state["rows"]) == 10
    equal(baseline.state["eligible"], partial.state["eligible"])
    for key in ("data", "fd", "ewc", "ridge", "first_step_ewc_suppression"):
        assert [r[key] for r in baseline.state["rows"]] == [r[key] for r in partial.state["rows"]]


@pytest.mark.parametrize("message,numerical", [("model output must be a finite tensor", True),
                                              ("history shape differs", False)])
def test_recorded_nonfinite_retirement_is_candidate_failure_not_schema_failure(monkeypatch, message, numerical):
    import torch
    from test_picar_learner import history
    from test_picar_initialization_numerics import MemoryStore
    from numerical_experiment.picar.config import PicarConfig
    from numerical_experiment.picar_solver_repair import recorded

    class FakeLearner:
        params = {}
        device = torch.device("cpu")
        generators = {}
        def __init__(self, *args, **kwargs):
            self.version = self.update_count = 0
        def load_state_dict(self, state):
            pass
        def state_dict(self):
            return {"version": self.version}
        def nll(self, params, batch):
            return torch.tensor([3.])
        def fit(self, **kwargs):
            self.version += 1
            return {"update_seconds": .1, "data": 3.}
        def retire(self, batch):
            raise ValueError(message)
    monkeypatch.setattr(recorded, "SolverLearner", FakeLearner)
    def item(i):
        h = history()
        h.update(decision_time=torch.tensor([[float(i)+.1]]), start_time=torch.tensor([float(i)]))
        return {"history": h, "arrival": i, "episode_id": "train", "learned": False,
            "baseline": 0., "bin": 1, "logits": torch.ones(7), "source_exposure": float(i)}
    cfg = replace(PicarConfig(), learning=replace(PicarConfig().learning, replay_capacity=1))
    run = recorded.RecordedRun(None, cfg, {"learner": {}}, {"early": [item(0), item(1)]},
        {"old": [item(5)], "live": [item(6)]}, MemoryStore(), lambda: None,
        solver_recipe="lagged_1", condition="replay_ewc")
    if numerical:
        summary = run.run()
        assert summary["status"] == "numerical_failure"
        assert summary["reason"] == message
        assert summary["completed_events"] == 1
    else:
        with pytest.raises(ValueError, match=message):
            run.run()
