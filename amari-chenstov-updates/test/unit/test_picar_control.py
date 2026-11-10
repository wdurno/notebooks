"""Bounded control-session checks; the protocol fake never models vehicle motion."""

import copy
from dataclasses import asdict, replace
import threading
import time

import pytest

from numerical_experiment.picar.control import ControlSession, ControlSessionError
from numerical_experiment.picar.runtime import ModelAccess
from src.robotics import CommandReceipt, Control, RewardComponents, RobotCommand, RobotObservation, StepResult
from src.robotics.simulator import SimulatorConfig


def until(predicate, timeout=2.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.002)
    raise AssertionError("bounded condition did not become true")


class ProtocolWorker:
    """Concurrent clock/receipt fake only. Real Bullet is exercised separately."""

    def __init__(self, deadline_s=.8):
        self.config = SimulatorConfig(physics_dt_s=.005, control_interval_s=.05,
                                      deadline_s=deadline_s, width=8, height=8)
        self._lock = threading.RLock()
        self._done = threading.Event()
        self.active = False
        self.ticks = self.last_harvest = self.frames = self.highwater = 0
        self.controls = Control()
        self.receipts = {}
        self.observations = {}
        self.session_id = "not-reset"
        self.run_id = "fake-control"
        self.pause_count = self.stop_count = 0
        self.command_calls = []
        self.terminal_reported = False
        self._thread = threading.Thread(target=self._clock, daemon=True)
        self._thread.start()

    @property
    def time_s(self):
        return self.ticks * self.config.physics_dt_s

    def _clock(self):
        while not self._done.wait(self.config.physics_dt_s):
            with self._lock:
                if self.active and self.time_s < self.config.deadline_s:
                    self.ticks += 1
                    if self.time_s >= self.expiry - 1e-10:
                        self.controls = self.config.calibration.neutral(self.controls)

    def reset(self, *, seed, exposure_intervals=None, training=True):
        with self._lock:
            self.ticks = self.last_harvest = self.frames = 0
            self.highwater = -1
            self.controls = Control()
            self.expiry = 0.
            self.terminal_reported = False
            self.receipts, self.observations = {}, {}
            self.session_id = f"fake-{seed}"
            self.active = True
            return self.observe()

    def observe(self):
        with self._lock:
            self.frames += 1
            frame = f"{self.session_id}:{self.frames}"
            self.observations[frame] = self.time_s
            return RobotObservation(self.run_id, self.session_id, frame, self.time_s, time.time(),
                                    self.time_s, self.time_s, max(0., self.config.deadline_s - self.time_s),
                                    8, 8, bytes(192), self.controls, "fake", "simulator_seconds")

    def command(self, command):
        with self._lock:
            cached = self.receipts.get(command.sequence_id)
            if cached is not None:
                assert cached[0] == command.to_dict()
                return replace(CommandReceipt.from_dict(cached[1]), duplicate=True)
            assert command.sequence_id > self.highwater
            assert self.active and command.session_id == self.session_id
            assert command.observation_id in self.observations
            self.config.calibration.validate(command.controls)
            accepted = command.expires_at_s > self.time_s
            self.highwater = command.sequence_id
            self.expiry = min(self.time_s + command.duration_s, command.expires_at_s)
            self.controls = command.controls if accepted else Control()
            receipt = CommandReceipt(self.run_id, self.session_id, command.sequence_id,
                                     command.observation_id, self.time_s, self.controls, accepted,
                                     "executing" if accepted else "expired", self.expiry,
                                     self.time_s - self.observations[command.observation_id])
            self.receipts[command.sequence_id] = (command.to_dict(), asdict(receipt))
            self.command_calls.append(command)
            return receipt

    def harvest(self):
        with self._lock:
            elapsed = (self.ticks - self.last_harvest) * self.config.physics_dt_s
            self.last_harvest = self.ticks
            terminal = self.time_s >= self.config.deadline_s - 1e-10
            bonus = -1. if terminal and not self.terminal_reported else 0.
            self.terminal_reported = terminal
            return StepResult(self.observe(), None,
                              RewardComponents(elapsed, elapsed / self.config.control_interval_s, bonus, terminal),
                              terminal, False, "deadline" if terminal else None,
                              {"physical_time_s": self.time_s})

    def snapshot(self):
        with self._lock:
            return {"worker_schema_version": 1, "active": self.active, "training": True,
                    "backend": {"ticks": self.ticks, "last_harvest": self.last_harvest,
                                "frames": self.frames, "highwater": self.highwater,
                                "receipts": [[key, *value] for key, value in self.receipts.items()],
                                "observations": dict(self.observations), "controls": self.controls,
                                "expiry": self.expiry, "session_id": self.session_id,
                                "terminal_reported": self.terminal_reported}}

    def restore(self, state):
        with self._lock:
            backend = copy.deepcopy(state["backend"])
            for name, value in backend.items():
                if name != "receipts":
                    setattr(self, name, value)
            self.receipts = {key: (command, receipt) for key, command, receipt in backend["receipts"]}
            self.active = state["active"]

    def stop(self):
        with self._lock:
            self.controls = self.config.calibration.neutral(self.controls)
            self.stop_count += 1

    def pause(self):
        with self._lock:
            self.stop()
            self.active = False
            self.pause_count += 1

    def resume(self, *, training=True):
        with self._lock:
            self.active = True

    def close(self):
        self._done.set()
        self._thread.join(timeout=1)
        assert not self._thread.is_alive()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_policy_receipts_causal_observations_and_fallback_are_not_gaussian_labels():
    with ProtocolWorker() as worker, ControlSession(worker, ModelAccess(),
            infer=lambda observation, version: Control(drive_mps=.1)) as session:
        session.start(seed=1)
        first = session.next_interval(timeout=1)
        second = session.next_interval(timeout=1)
        assert first.fallback_reason == "startup"
        assert first.action is first.model_version is first.policy_observation is None
        assert first.command.controls.drive_mps == 0
        assert second.fallback_reason is None and second.model_version == 0
        assert second.policy_observation.observation_id == first.observation.observation_id
        assert second.command.observation_id == second.policy_observation.observation_id
        assert second.receipt.accepted and second.action["controls"].drive_mps == .1
        assert second.observation.capture_time_s < second.result.observation.capture_time_s
        assert second.result.reward_components.elapsed_s > 0
        assert second.timing["capture_to_command_s"] > 0
        assert "probabilities" not in second.action  # A callback Control is not fabricated exploration.
        assert worker.pause_count == 0


def test_blocked_inference_keeps_collecting_fallback_and_clock_ticks():
    entered, release = threading.Event(), threading.Event()
    def infer(observation, version):
        entered.set()
        assert release.wait(2)
        return Control(drive_mps=.1)
    with ProtocolWorker() as worker:
        session = ControlSession(worker, ModelAccess(), infer=infer)
        try:
            session.start(seed=2)
            assert entered.wait(1)
            before = worker.ticks
            rows = [session.next_interval(timeout=1) for _ in range(3)]
            assert worker.ticks > before + 10 and worker.pause_count == 0
            assert all(row.command.controls.drive_mps == 0 and row.action is None for row in rows)
            assert any(row.timing["inference_deadline_miss"] == 1 for row in rows)
            with pytest.raises(TimeoutError, match="in-flight"):
                session.snapshot(timeout=.01)
            release.set()
            next_row = session.next_interval(timeout=1)
            assert next_row.action is None  # The formerly blocked prediction missed its own deadline.
        finally:
            release.set()
            session.close()


def test_model_access_prioritizes_waiting_inference_and_never_overlaps_update():
    access = ModelAccess()
    update_entered, release_update = threading.Event(), threading.Event()
    inference_entered, release_inference = threading.Event(), threading.Event()
    second_update = threading.Event()
    order = []
    def infer(observation, version):
        order.append(("infer", version))
        inference_entered.set()
        assert release_inference.wait(2)
        return Control()
    def first_update():
        order.append(("update", 0))
        update_entered.set()
        assert release_update.wait(2)
    with ProtocolWorker() as worker:
        session = ControlSession(worker, access, infer=infer)
        one = threading.Thread(target=lambda: session.update(callback=first_update))
        two = threading.Thread(target=lambda: session.update(callback=lambda: second_update.set()))
        try:
            one.start()
            assert update_entered.wait(1)
            session.start(seed=3)
            until(lambda: access._waiting_inference == 1)
            two.start()
            # Release within the prediction deadline. Expired requests now
            # correctly skip inference rather than consuming priority work.
            until(lambda: worker.ticks > 0)
            assert worker.controls.drive_mps == 0
            release_update.set()
            assert inference_entered.wait(1)
            assert not second_update.is_set()
            assert order[:2] == [("update", 0), ("infer", 1)]
            release_inference.set()
            one.join(timeout=1)
            two.join(timeout=1)
            assert not one.is_alive() and not two.is_alive()
            assert second_update.is_set() and access.version == 2
            assert worker.pause_count == 0
        finally:
            release_update.set()
            release_inference.set()
            one.join(timeout=2)
            if two.ident is not None:
                two.join(timeout=2)
            session.close()


def test_queue_overflow_stops_controls_not_physics_and_stays_incomplete():
    with ProtocolWorker() as worker, ControlSession(worker, ModelAccess(),
            infer=lambda observation, version: Control(), pending_capacity=1) as session:
        session.start(seed=4)
        until(lambda: session.error is not None)
        assert session.pending_count == 1 and "exhausted" in session.error
        assert worker.pause_count == 0 and worker.stop_count > 0 and worker.active
        before = worker.ticks
        time.sleep(.025)
        assert worker.ticks > before
        valid = session.next_interval(timeout=0)
        assert not valid.result.terminated and not valid.result.truncated
        with pytest.raises(ControlSessionError, match="incomplete"):
            session.next_interval(timeout=0)
        snapshot = session.snapshot()
        assert "incomplete" in snapshot["failure"] and snapshot["open"] is not None


def test_snapshot_restore_is_recovery_paused_with_sequence_reconciliation():
    def infer(observation, version):
        return Control(drive_mps=.1)
    with ProtocolWorker() as original_worker:
        with ControlSession(original_worker, ModelAccess(), infer=infer) as original:
            original.start(seed=5)
            original.next_interval(timeout=1)
            original.next_interval(timeout=1)
            snapshot = original.snapshot()
        highwater = snapshot["worker"]["backend"]["highwater"]
        snapshot["next_sequence"] = 0  # Recovery must also reconcile a stale caller counter.
        with ProtocolWorker() as restored_worker, ControlSession(restored_worker, ModelAccess(),
                infer=infer) as restored:
            restored.restore(snapshot)
            frames = restored_worker.frames
            ticks = restored_worker.ticks
            time.sleep(.02)
            assert restored_worker.ticks == ticks and restored_worker.frames == frames
            assert restored_worker.command_calls == []
            assert restored._next_sequence == highwater + 1
            restored.resume()
            restored.next_interval(timeout=1)
            until(lambda: bool(restored_worker.command_calls))
            assert restored_worker.command_calls[0].sequence_id == highwater + 1
            assert all(command.sequence_id > highwater for command in restored_worker.command_calls)


def test_unacknowledged_command_is_not_replayed_or_made_trainable_on_restore():
    with ProtocolWorker() as worker, ControlSession(worker, ModelAccess(),
            infer=lambda observation, version: Control()) as session:
        session.start(seed=6)
        session.next_interval(timeout=1)
        state = session.snapshot()
    command = replace(state["open"]["command"], sequence_id=999)
    state["pending_command"] = command
    state["failure"] = "ambiguous acknowledgment"
    with ProtocolWorker() as worker, ControlSession(worker, ModelAccess(),
            infer=lambda observation, version: Control()) as restored:
        restored.restore(state)
        assert worker.command_calls == [] and restored._next_sequence == 1000
        assert "unacknowledged" in restored.error
        with pytest.raises(ControlSessionError, match="incomplete"):
            restored.resume()


def test_cached_acknowledgment_reconciles_without_repeating_accepted_motion():
    with ProtocolWorker() as worker, ControlSession(worker, ModelAccess(),
            infer=lambda observation, version: Control()) as session:
        session.start(seed=11)
        session.next_interval(timeout=1)
        state = session.snapshot()
    state["pending_command"] = state["open"]["command"]
    with ProtocolWorker() as worker, ControlSession(worker, ModelAccess(),
            infer=lambda observation, version: Control()) as restored:
        restored.restore(state)
        assert restored._pending_command is None
        assert restored._open["receipt"] == state["open"]["receipt"]
        assert worker.command_calls == []


def test_terminal_between_capture_and_command_is_unlabeled_terminal_telemetry():
    class TerminalRaceWorker(ProtocolWorker):
        def command(self, command):
            with self._lock:
                if self.command_calls:
                    self.ticks = round(self.config.deadline_s / self.config.physics_dt_s)
                    return CommandReceipt(self.run_id, self.session_id, command.sequence_id,
                        command.observation_id, self.time_s, Control(), False, "episode_ended", self.time_s)
                return super().command(command)
    with TerminalRaceWorker() as worker, ControlSession(worker, ModelAccess(),
            infer=lambda observation, version: Control(drive_mps=.1)) as session:
        session.start(seed=12)
        first = session.next_interval(timeout=1)
        tail = session.next_interval(timeout=1)
        assert not first.result.terminated and tail.result.terminated
        assert not tail.receipt.accepted and tail.fallback_reason == "episode_ended_before_command"
        assert tail.action is tail.model_version is tail.policy_observation is None
        assert tail.result.reward_components.terminal_bonus == -1
        assert session.error is None


def test_pause_is_explicit_between_episodes_not_for_inference_and_failures_dont_publish():
    access = ModelAccess()
    with ProtocolWorker(deadline_s=.15) as worker, ControlSession(worker, access,
            infer=lambda observation, version: Control()) as session:
        session.start(seed=7)
        with pytest.raises(ControlSessionError, match="never an inference"):
            session.pause(reason="sleep")
        with pytest.raises(ValueError):
            session.pause(reason="inference")
        def fail():
            raise ValueError("failed fit")
        with pytest.raises(ValueError, match="failed fit"):
            session.update(callback=fail)
        assert access.version == 0
        while not session.next_interval(timeout=1).result.terminated:
            pass
        session.pause(reason="sleep")
        ticks = worker.ticks
        time.sleep(.02)
        assert worker.ticks == ticks and worker.pause_count == 1
        session.start(seed=8)
        assert worker.active and session.next_interval(timeout=1).observation.session_id == "fake-8"


class FakeLearner:
    def __init__(self):
        self.version = 0
        self.calls = 0
        self.fits = []

    def choose_action(self, history, *, explore):
        assert set(history) == {"frame"} and explore
        self.calls += 1
        return {"probabilities": (.125,) * 8, "logits": (0.,) * 7,
                "baseline": 1., "model_version": self.version}

    def fit(self, **kwargs):
        self.fits.append(kwargs)
        self.version += 1
        return {"fit": self.version}

    def state_dict(self):
        return copy.deepcopy({"version": self.version, "calls": self.calls, "fits": self.fits})

    def load_state_dict(self, state):
        self.version, self.calls, self.fits = state["version"], state["calls"], copy.deepcopy(state["fits"])


def test_memory_learner_interface_model_version_and_snapshot_state():
    learner, access = FakeLearner(), ModelAccess()
    builder = lambda observation: {"frame": observation.observation_id}
    with ProtocolWorker() as worker, ControlSession(worker, access, learner=learner,
                                                   history_builder=builder) as session:
        session.start(seed=9)
        session.next_interval(timeout=1)
        policy = session.next_interval(timeout=1)
        assert policy.action["baseline"] == 1. and policy.model_version == 0
        assert session.update(new="new-data", replay=None, acquired_count=2) == {"fit": 1}
        assert access.version == learner.version == 1
        state = session.snapshot()
        assert state["learner"]["version"] == 1 and state["learner"]["calls"] > 0
    with ProtocolWorker() as worker, ControlSession(worker, ModelAccess(), learner=FakeLearner(),
                                                   history_builder=builder) as restored:
        restored.restore(state)
        assert restored.learner.state_dict() == state["learner"]
        assert restored.access.version == 1


def test_invalid_inference_stays_incomplete_not_a_reward_or_action_parser_zero():
    with ProtocolWorker() as worker, ControlSession(worker, ModelAccess(),
            infer=lambda observation, version: {"probabilities": (float("nan"),) * 8}) as session:
        session.start(seed=10)
        first = session.next_interval(timeout=1)
        assert first.action is None and not first.result.terminated
        until(lambda: session.error is not None)
        assert "inference failed" in session.error
        with pytest.raises(ControlSessionError, match="inference failed"):
            session.next_interval(timeout=1)


def test_real_worker_physics_continues_while_update_holds_model_lease():
    pytest.importorskip("pybullet")
    from src.robotics.worker import SimulatorWorker
    entered, release = threading.Event(), threading.Event()
    def update():
        entered.set()
        assert release.wait(3)
    config = SimulatorConfig(control_interval_s=.05, width=32, height=24, deadline_s=2.)
    with SimulatorWorker(config, run_id="control-real") as worker:
        access = ModelAccess()
        session = ControlSession(worker, access, infer=lambda observation, version: Control(drive_mps=.1))
        thread = threading.Thread(target=lambda: session.update(callback=update))
        try:
            thread.start()
            assert entered.wait(1)
            session.start(seed=19)
            before = worker.diagnostics()["physics_ticks"]
            rows = [session.next_interval(timeout=1) for _ in range(3)]
            after = worker.diagnostics()["physics_ticks"]
            assert after > before + 10
            assert all(row.action is None and row.command.controls.drive_mps == 0 for row in rows)
            assert any(row.timing["inference_deadline_miss"] for row in rows)
            assert worker.diagnostics()["worker_active"]
            assert all(row.result.reward_components.elapsed_s > 0 for row in rows)
            release.set()
            thread.join(timeout=2)
            assert not thread.is_alive() and access.version == 1
        finally:
            release.set()
            thread.join(timeout=3)
            session.close()
