"""Actual simulator recovery, frame synchronization and frozen-head continuation."""

import copy
from dataclasses import asdict, replace
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar.teacher import prepare_teacher_distribution
from numerical_experiment.picar_evaluation_video import assert_unchanged
from numerical_experiment.picar_head_awac.learner import Learner
from numerical_experiment.picar_head_awac.model import Heads
from numerical_experiment.picar_head_awac.config import simulator
from numerical_experiment.picar_coaching.config import recipe
from numerical_experiment.picar_coaching import trajectory
from numerical_experiment.picar_coaching.trajectory import CoachingTrajectory, snapshot_at_capture
from src.robotics.protocol import RobotCommand
from src.robotics.simulator import PyBulletBackend


def test_snapshot_alignment_recovers_original_physical_pose_and_view():
    cfg = replace(simulator(fixed="initial_A"), realtime=False, width=32, height=24)
    with PyBulletBackend(cfg, run_id="sync-test") as sim:
        observation = sim.reset(seed=1)
        controls = cfg.calibration.map_distribution([1., 0., 0., 0., 0., 0., 0., 0.])
        sim.command(RobotCommand(sim.run_id, observation.session_id, 0, observation.observation_id,
                                controls, sim.time_s, .5, sim.time_s + .5))
        result = sim.advance(60, training=False)
        original_camera = sim.render().copy()
        original_topdown = sim.render_topdown().copy()
        at = result.observation.capture_time_s
        sim.advance(7, training=False)
        snapshot = snapshot_at_capture(sim.snapshot(), at)
    with PyBulletBackend(cfg, run_id="sync-test") as reconstructed:
        reconstructed.restore(snapshot)
        assert reconstructed.time_s == pytest.approx(at)
        assert np.array_equal(reconstructed.render(), original_camera)
        assert np.array_equal(reconstructed.render_topdown(), original_topdown)


class Teacher:
    def __init__(self, encoder):
        self.last_diagnostic = {"test_only": True}

    def distribution(self, history, probability_floor):
        return prepare_teacher_distribution(torch.tensor([[0., 0., 1., 0., 0., 0., 0., 0.]]),
                                             probability_floor=probability_floor, target_kind="hard_engineered")


@pytest.mark.parametrize("condition", ["U", "G"])
def test_collection_resume_does_not_fabricate_terminal_returns_or_duplicate_records(tmp_path, monkeypatch, condition):
    monkeypatch.setattr(trajectory, "CenteringTeacher", Teacher)
    real_simulator = trajectory.simulator
    monkeypatch.setattr(trajectory, "simulator", lambda *args, **kwargs: replace(
        real_simulator(*args, **kwargs), deadline_s=1.5, control_interval_s=.5, width=32, height=24))
    cfg = recipe()
    model = SimpleNamespace(encoder=None)
    scorer = SimpleNamespace(score_rgb=lambda rgb: {"score": 1.})
    output = tmp_path / "collect"
    calls = []
    interrupted = [False]

    def check():
        calls.append(1)
        if len(calls) == 2 and not interrupted[0]:
            interrupted[0] = True
            raise InterruptedError("test checkpoint")

    def make(store, callback):
        return CoachingTrajectory(Learner(Heads(), cfg, 2, "td_fd"), None, scorer, store, callback,
            model=model, compliance=SimpleNamespace(score=lambda *a: {"judgment": "uncertain", "reason": "test only"}),
            condition=condition, phase="initialization", seed=2, total=4, collection=True)

    # Media has independent encode/recovery tests; this test targets acquisition admission.
    monkeypatch.setattr("numerical_experiment.picar_coaching.video.render_video", lambda *a, **k: [])
    with RunStore(output, {"test": 1}, checkpoint_policy="journal-v1") as store:
        runner = make(store, check)
        with pytest.raises(InterruptedError, match="test checkpoint"):
            runner.run()
        saved = store.load_checkpoint(trusted=True)
        assert not saved["initial_records"]
        assert saved["current"] is not None and saved["session"] is not None
    with RunStore(output, {"test": 1}, resume=True, checkpoint_policy="journal-v1") as store:
        runner = make(store, lambda: None)
        result = runner.run()
        ids = [r["id"] for r in runner.s["initial_records"]]
        assert ids and len(ids) == len(set(ids))
        assert result["n"] >= 4 and result["retired"] == 0 and not result["metrics"]
        assert all(ep["complete"] and ep["end_reason"] == "deadline" for ep in result["episodes"])
        assert all(ep["guidance_return"] == 0 for ep in result["episodes"])


def test_rank64_fit_and_retirement_are_transactional_and_resume_equivalent():
    learner = Learner(Heads(), recipe(), 3, "td_fd")
    record = {"id": "new:0", "episode_id": "ep:0", "feature": torch.linspace(-1, 1, 64),
        "next_feature": torch.linspace(-.8, .8, 64), "action": torch.zeros(7), "reward": torch.tensor(.2),
        "return": torch.tensor(.2), "discount": torch.tensor(0.), "arrival": 0, "visits": 0}
    before = copy.deepcopy(learner.state_dict())
    def stop():
        raise InterruptedError("inner solve")
    with pytest.raises(InterruptedError):
        learner.fit([record], 255, stop)
    assert_unchanged(before, learner.state_dict())
    expected = Learner(Heads(), recipe(), 3, "td_fd")
    expected.load_state_dict(before)
    a, b = learner.fit([record], 255), expected.fit([record], 255)
    learner.retire(record)
    expected.retire(record)
    assert_unchanged(learner.state_dict(), expected.state_dict())
    assert learner.retired == 1 and learner.archive.rank == 64 and a["fd"] >= 0


def test_actual_label_path_retains_terminal_and_guidance_rewards_on_fallback(tmp_path):
    from numerical_experiment.picar.fix2_data import history_batch, TASK
    from numerical_experiment.picar_head_awac.data import pack_history
    with RunStore(tmp_path / "labels", {"test": True}, checkpoint_policy="journal-v1") as store:
        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        ref = store.frames.put(rgb)
        history = history_batch([{"rgb": rgb, "time": 1., "elapsed": 1., "action": [0., 0., 0., 0.]}],
                                contexts=[{"text": TASK, "issued_at": 0.}], deadline=3., width=32, height=24)
        rows = [{"id": f"r{i}", "frame_ref": ref, "elapsed": 1., "bonus": 10. if i == 2 else 0.,
                 "terminal": i == 2, "truncated": False, "end_time": float(i + 1),
                 "observed_controls": [0., 0., 0., 0.], "action": torch.zeros(7) if i == 1 else None,
                 "feature": torch.zeros(64) if i == 1 else None, "arrival": i} for i in range(3)]
        rows[1]["history"] = pack_history(history, store.frames)
        runner = object.__new__(CoachingTrajectory)
        runner.s = {"current": {"complete": True, "rows": rows, "scores": [{"score": 2.}] * 3,
            "evaluation": False, "episode_id": "ep", "initial_frame_ref": ref, "initial_frame_time": 0.,
            "initial_controls": [0., 0., 0., 0.], "simulator_config": {"deadline_s": 3., "width": 32, "height": 24},
            "compliance": {"w": {"closing_row": "r2", "receipt": {"raw": 2.}}}}}
        runner.coach = SimpleNamespace(windows=[], state_dict=lambda: {"audits": {}},
            policy_context=lambda *a, **k: {"instructions": [], "statuses": []})
        runner.collection, runner.condition, runner.adherence_enabled = False, "G", True
        runner.store, runner.features = store, lambda history: torch.zeros(64)
        runner.cfg = SimpleNamespace(model=SimpleNamespace(context_length=4))
        runner.save, runner._compliance = lambda: None, lambda: None
        records = runner.label()
        assert len(records) == 1 and records[0]["id"] == "r1"
        assert records[0]["physical_rows"] == 2
        assert float(records[0]["reward"]) == pytest.approx(.99**2 * 10.2)
        assert float(records[0]["discount"]) == 0.


def test_trim_keeps_resident_fallback_covariates_with_distinct_camera_clock(tmp_path):
    from numerical_experiment.picar.fix2_data import history_batch, TASK
    from numerical_experiment.picar_head_awac.data import pack_history
    with RunStore(tmp_path / "trim", {"test": True}, checkpoint_policy="journal-v1") as store:
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        def packed(times):
            return pack_history(history_batch([{"rgb": rgb, "time": t, "elapsed": 1., "action": [0., 0., 0., 0.]}
                for t in times], contexts=[{"text": TASK, "issued_at": 0.}], deadline=20., width=8, height=8), store.frames)
        record = {"id": "decision", "episode_id": "ep", "history": packed([1.01]),
                  "next_history": packed([1., 2., 3.]), "feature": torch.zeros(64), "next_feature": torch.zeros(64)}
        runner = object.__new__(CoachingTrajectory)
        runner.s, runner.store = {"replay": [record]}, store
        class Features:
            def __call__(self, history):
                return torch.zeros(64)
            def clear(self):
                pass
        runner.features = Features()
        runner.trim()
        assert record["next_history"]["decision_time"].tolist() == [[2., 3.]]
