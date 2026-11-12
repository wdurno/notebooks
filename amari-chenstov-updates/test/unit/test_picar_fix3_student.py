"""CPU-only numerical/recovery tests; no Bullet clock, VLM, or GPU launches."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from numerical_experiment.picar import fix3_student as student
from numerical_experiment.picar.config import PicarConfig
from numerical_experiment.picar.models import build_tiny_model
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar.teacher import prepare_teacher_distribution
from src.robotics.actions import Calibration
from src.robotics.protocol import Control, RobotObservation, RewardComponents, StepResult
from src.robotics.simulator import SimulatorConfig


class FakeReward:
    calls = 0

    def __init__(self, encoder):
        self.cache = {}

    def score(self, history, pan):
        type(self).calls += 1
        key = student.batch_digest(history)
        self.cache[key] = {"score": 2., "pan_rad": pan, "diagnostic": {"fake": True}}
        return self.cache[key]

    def state_dict(self):
        return copy.deepcopy(self.cache)

    def load_state_dict(self, state):
        self.cache = copy.deepcopy(state)


class FakeTeacher:
    calls = 0

    def __init__(self, encoder, variant):
        self.provenance = {"kind": "unit-test-only", "variant": variant}
        self.last_diagnostic = {}

    def distribution(self, history, probability_floor):
        type(self).calls += 1
        q = torch.full((1, 8), .01)
        q[0, 2] = .93
        return prepare_teacher_distribution(q, probability_floor=probability_floor,
                                            target_kind="hard_engineered")


class FakeWorker:
    """Deterministic test double, not a permitted production clock."""

    def __init__(self, config, run_id):
        self.config = config
        self.seed = None

    def diagnostics(self):
        return {"layout": {"seed": self.seed}}

    def snapshot(self):
        return {"backend": {"test_only": True}}


class FakeSession:
    def __init__(self, worker, access, *, pending_capacity, own_worker, learner=None,
                 history_builder=None, explore=True, infer=None):
        self.worker, self.access = worker, access
        self.learner, self.builder, self.explore, self.infer = learner, history_builder, explore, infer
        self.index = 0
        self.terminal = False
        self.seed = None

    def observation(self, index):
        rgb = np.full((8, 8, 3), 10 + index * 20, dtype=np.uint8)
        return RobotObservation("test", str(self.seed), f"{self.seed}:{index}", index * .1,
            index * .1, index * .1, index * .1, max(0., .4 - index * .1), 8, 8,
            rgb.tobytes(), Control(), "ok", "simulator_seconds")

    def start(self, *, seed, exposure_intervals, training):
        self.seed = self.worker.seed = seed
        self.training = training
        return self.observation(0)

    def next_interval(self, timeout):
        if self.terminal:
            raise StopIteration
        observation = self.observation(self.index)
        action = None
        if self.index:
            action = self.infer(observation, self.access.version) if self.infer else self.learner.choose_action(
                self.builder(observation), explore=self.explore)
        self.index += 1
        self.terminal = self.index == 4
        result = StepResult(self.observation(self.index), None,
            RewardComponents(.1, .2, -1. if self.terminal else 0., self.terminal),
            self.terminal, False, "timeout" if self.terminal else None)
        return SimpleNamespace(index=self.index - 1, observation=observation,
            policy_observation=observation if action else None, action=action,
            model_version=self.access.version if action else None,
            fallback_reason=None if action else "startup", result=result, timing={})

    def update(self, cooperative=False, **kwargs):
        assert cooperative
        result = self.learner.fit(**kwargs)
        self.access.version = self.learner.version
        return result

    def snapshot(self, timeout=10., capture_external=None):
        result = {"index": self.index, "terminal": self.terminal,
            "seed": self.seed, "training": self.training, "pending": [],
            "learner": self.learner.state_dict() if self.learner else None}
        if capture_external:
            result["external"] = capture_external()
        return result

    def restore(self, value):
        self.index, self.terminal, self.seed, self.training = (
            value[k] for k in ("index", "terminal", "seed", "training"))
        self.worker.seed = self.seed
        if self.learner:
            self.learner.load_state_dict(value["learner"])
            self.access.version = self.learner.version

    def resume(self):
        pass

    def pause(self, reason):
        assert self.terminal

    def close(self, timeout=10.):
        pass


@pytest.fixture
def setup(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(student, "CenteringReward", FakeReward)
    monkeypatch.setattr(student, "CenteringTeacher", FakeTeacher)
    monkeypatch.setattr(student, "SimulatorWorker", FakeWorker)
    monkeypatch.setattr(student, "ControlSession", FakeSession)
    monkeypatch.setattr(student._Run, "render_video", lambda *args: [])
    base = PicarConfig()
    config = replace(base, model=replace(base.model, context_length=2, hidden_size=4, return_bins=7),
        learning=replace(base.learning, fd_strength=0., replay_capacity=2, replay_batch=1,
                         return_min=-6., return_max=12., reward_version="dense-visual-centering-v1"))
    sim = SimulatorConfig(width=8, height=8, control_interval_s=.1, deadline_s=.4,
        task_version="ball-contact-v1", layout_version="curriculum-v1")
    yield config, sim
    torch.set_num_threads(old_threads)


def model():
    return build_tiny_model(31, hidden_size=4, width=4, return_bins=7, context_length=2).eval()


def boot(path, setup, stopped=lambda: None, *, resume=False):
    config, sim = setup
    with RunStore(path, {"test": "fix3"}, resume=resume, checkpoint_policy="journal-v1") as store:
        summary = student.bootstrap(model(), config, sim, store, stopped, max_transitions=12)
        return summary, store.load_checkpoint(trusted=True), store.metrics()


def equal_learner(a, b):
    assert (a["version"], a["retired_count"], a["update_count"]) == (
        b["version"], b["retired_count"], b["update_count"])
    for group in ("parameters", "diagonal", "anchor", "rng"):
        for name in a[group]:
            torch.testing.assert_close(a[group][name], b[group][name], rtol=0, atol=0)
    for key, values in a["optimizer"]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][key][name], rtol=0, atol=0)


def test_bootstrap_joint_nll_partition_receipt_and_archived_contexts(tmp_path, setup):
    summary, state, metrics = boot(tmp_path / "boot", setup)
    assert summary["acquired_count"] == 8
    assert len(state["eligible"]) == 2
    assert state["learner"]["retired_count"] == len(state["retired"]) == 4
    assert not set(state["retired"]) & {r["arrival"] for r in state["eligible"]}
    assert state["nll_step"] == state["distill_receipt"]["steps"] == 6
    assert state["distill_receipt"]["batch_size"] == 1
    assert sum(float(d.sum()) for d in state["learner"]["diagonal"].values()) > 0
    assert metrics and metrics[-1]["new_transitions"] == 8
    assert all(m["condition"] == "replay_ewc" for m in metrics)
    policy = [r for r in state["intervals"] if r["probabilities"] is not None]
    assert all(r["context_refs"] and r["contexts"] for r in policy)
    assert all(len(r["context_refs"]) == len(r["context_actions"]) == len(r["context_decision_times"])
               for r in policy)
    from numerical_experiment.picar.storage import FrameStore
    archive = FrameStore(tmp_path / "boot" / "frames", readonly=True)
    assert archive.get(policy[0]["context_refs"][0]).shape == (8, 8, 3)
    assert summary["distinct_bootstrap_samples"] > 0
    assert not state["ready"]


@pytest.mark.parametrize("boundary", ["collection", "scoring", "kl", "nll", "retirement"])
def test_bootstrap_resume_matches_uninterrupted(tmp_path, setup, boundary):
    _, expected, _ = boot(tmp_path / "reference", setup)
    config, sim = setup
    with RunStore(tmp_path / "interrupted", {"test": "fix3"}, checkpoint_policy="journal-v1") as store:
        def stop():
            s = store.load_checkpoint(trusted=True)
            hit = {"collection": s["stage"] == "collect" and s["acquired_count"] == 2,
                "scoring": bool(s["pending"] and s["pending"][0]["score"]),
                "kl": s["stage"] == "distill" and s["kl_step"] == 2,
                "nll": s["stage"] == "initial_nll" and s["nll_step"] == 2,
                "retirement": s["stage"] == "initial_fisher" and len(s["retired"]) == 1}[boundary]
            if hit:
                raise InterruptedError(boundary)
        with pytest.raises(InterruptedError):
            student.bootstrap(model(), config, sim, store, stop, max_transitions=12)
    _, actual, _ = boot(tmp_path / "interrupted", setup, resume=True)
    equal_learner(expected["learner"], actual["learner"])
    assert actual["retired"] == expected["retired"]
    assert actual["kl_losses"] == expected["kl_losses"]
    assert actual["episodes"] == expected["episodes"]
    assert actual["selected_exploration_std"] == expected["selected_exploration_std"]


def test_full_returns_reject_incomplete_and_truncated(setup):
    config, _ = setup
    p = {"elapsed": .1, "bonus": 0., "terminal": False, "truncated": False,
         "score": {"score": 2.}}
    with pytest.raises(ValueError, match="completed"):
        student._completed_labels([p], config.learning, torch.linspace(-6, 12, 8))
    with pytest.raises(ValueError, match="nontruncated"):
        student._completed_labels([{**p, "terminal": True, "truncated": True}],
                                  config.learning, torch.linspace(-6, 12, 8))


def test_pilot_evaluation_isolation_and_shared_initial(tmp_path, setup):
    _, initial, _ = boot(tmp_path / "boot", setup)
    original = copy.deepcopy(initial)
    config, sim = setup
    panel = [{"difficulty": "easy", "config": sim, "seeds": (195803, 195807)}]
    calls = FakeTeacher.calls
    with RunStore(tmp_path / "pilot", {"test": "pilot"}, checkpoint_policy="journal-v1") as store:
        run = student._Run(model(), replace(config, learning=replace(config.learning,
            exploration_std=initial["selected_exploration_std"])), sim, store, lambda: None,
            mode="pilot", seed=42, max_transitions=12, initial=initial, evaluations=panel)
        before = run.learner.state_dict()
        scorer_calls = FakeReward.calls
        run.evaluate("initial")
        equal_learner(before, run.learner.state_dict())
        assert FakeReward.calls == scorer_calls
        assert run.s["acquired_count"] == 0
        result = run.run()
        final = store.load_checkpoint(trusted=True)
    assert FakeTeacher.calls == calls
    equal_learner(original["learner"], initial["learner"])
    assert result["online_updates"] > 0 and result["pending_updates"] == 0
    assert [(x["checkpoint"], len(x["rows"])) for x in result["evaluations"]] == [("initial", 2), ("final", 2)]
    assert len(final["eligible"]) == config.learning.replay_capacity
    assert len(set(final["retired"])) == len(final["retired"])
    assert not set(final["retired"]) & {r["arrival"] for r in final["eligible"]}
    assert all(x["return"] is not None for x in final["eligible"])


def test_calibration_uses_separate_rng_and_reports_every_probe(tmp_path, setup):
    _, initial, _ = boot(tmp_path / "boot", setup)
    config, _ = setup
    learner = student.MemoryLearner(model(), config.learning, config.seeds(0))
    learner.load_state_dict(initial["learner"])
    before = learner.state_dict()
    result = student.calibrate_exploration(learner, initial["eligible"], Calibration(), seed=11)
    equal_learner(before, learner.state_dict())
    assert [x["std"] for x in result["rows"]] == [.1, .5, 1., 2., 4.]
    passing = [x["std"] for x in result["rows"] if x["passes"]]
    assert result["selected_exploration_std"] == (min(passing) if passing else 4.)
    assert result["probes_per_context"] == 256 and result["rollouts"] == 0


def test_suffix_never_bridges_retired_or_another_episode(setup):
    def record(episode, arrival, times):
        h = {"images": torch.zeros(1, len(times), 3, 8, 8),
            "executed_actions": torch.zeros(1, len(times), 4),
            "elapsed": torch.ones(1, len(times), 1), "decision_time": torch.tensor([times]),
            "start_time": torch.tensor([0.]), "contexts": [[]]}
        return {"history": h, "episode_id": episode, "arrival": arrival,
                "baseline": 0., "bin": 0, "logits": torch.zeros(7)}
    old = record("other", 0, [1., 2.])
    new = record("new", 1, [1., 2., 3.])
    assert student._batch(new, [old, new]).histories[0]["images"].shape[1] == 1
    student._trim_eligible([old, new])
    assert new["history"]["decision_time"].tolist() == [[3.]]
