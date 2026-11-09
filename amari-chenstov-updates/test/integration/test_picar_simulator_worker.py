"""Real-time CPU process checks, bounded to seconds and fake/simulated only."""

import math
import os
import time

import pytest

pytest.importorskip("pybullet")

from src.robotics import Control, RobotCommand
from src.robotics.simulator import SimulatorConfig
from src.robotics.worker import SimulatorWorker, WorkerError


def test_parent_learning_delay_does_not_freeze_physics_or_replay_motion():
    cfg = SimulatorConfig(width=32, height=24, deadline_s=2.)
    with SimulatorWorker(cfg, run_id="worker-test") as worker:
        assert worker.pid != os.getpid()
        observation = worker.reset(seed=19)
        start = worker.diagnostics()
        command = RobotCommand(observation.run_id, observation.session_id, 0, observation.observation_id,
                               Control(drive_mps=.2), observation.capture_time_s, .5,
                               observation.capture_time_s + .5)
        assert worker.command(command).accepted
        busy_start = time.monotonic()
        time.sleep(.25)  # The parent performs no RPC while learning/inference is busy.
        result = worker.harvest()
        wall_s = time.monotonic() - busy_start
        after = worker.diagnostics()
        assert after["physics_ticks"] - start["physics_ticks"] >= 20
        assert math.dist(after["position_m"][:2], start["position_m"][:2]) > .015
        assert .08 < result.reward_components.elapsed_s < wall_s + .1
        assert after["physics_ticks"] - start["physics_ticks"] <= math.ceil((wall_s + .05) / cfg.physics_dt_s)
        assert result.timing["worker_realtime"] == 1
        assert result.timing["accelerated"] == 0
        assert worker.command(command).duplicate
        time.sleep(.5)
        later = worker.harvest()
        assert later.observation.executed_controls.drive_mps == 0
        assert later.timing["fallback_ticks"] > 0
        assert worker.diagnostics()["geometry"]["speed_mps"] < .02


def test_worker_pause_snapshot_and_new_process_restore_preserve_authoritative_state():
    cfg = SimulatorConfig(width=32, height=24, deadline_s=2.)
    with SimulatorWorker(cfg, run_id="snapshot-worker") as original:
        original.reset(seed=42, exposure_intervals=200.)
        time.sleep(.08)
        original.pause()
        snapshot = original.snapshot()
        time.sleep(.05)
        assert original.snapshot() == snapshot
        with SimulatorWorker(cfg, run_id="snapshot-worker") as restored:
            restored.restore(snapshot)
            assert restored.snapshot() == snapshot
            assert restored.diagnostics()["worker_active"] is False
            assert restored.diagnostics()["position_m"] == original.diagnostics()["position_m"]
            with pytest.raises(WorkerError, match="paused"):
                observation = restored.observe()
                restored.command(RobotCommand(observation.run_id, observation.session_id, 0,
                    observation.observation_id, Control(), observation.capture_time_s, .5,
                    observation.capture_time_s + .5))
            restored.resume(training=False)
            exposure = restored.diagnostics()["exposure_intervals"]
            ticks = restored.diagnostics()["physics_ticks"]
            time.sleep(.06)
            assert restored.diagnostics()["physics_ticks"] > ticks
            assert restored.diagnostics()["exposure_intervals"] == exposure


def test_worker_terminal_is_harvested_once_and_does_not_busy_step_after_end():
    cfg = SimulatorConfig(width=32, height=24, deadline_s=.1)
    with SimulatorWorker(cfg) as worker:
        worker.reset(seed=9)
        time.sleep(.2)
        result = worker.harvest()
        assert result.terminated and not result.truncated and result.end_reason == "deadline"
        assert result.reward_components.terminal_bonus == -1
        assert result.reward_components.elapsed_s == .1
        ticks = worker.diagnostics()["physics_ticks"]
        time.sleep(.06)
        next_result = worker.harvest()
        assert next_result.reward_components.terminal_bonus == 0
        assert next_result.reward_components.elapsed_s == 0
        assert worker.diagnostics()["physics_ticks"] == ticks == 24
    with pytest.raises(WorkerError, match="closed"):
        worker.observe()


def test_worker_refuses_unlabelled_accelerated_operation():
    with pytest.raises(ValueError, match="realtime only"):
        SimulatorWorker(SimulatorConfig(realtime=False))
