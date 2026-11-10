"""CPU-only lease and expiry tests: no simulator, model loading, or device API."""

import builtins
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from numerical_experiment.picar.control import ControlSession, ControlSessionError
from numerical_experiment.picar.runtime import ModelAccess
from src.robotics import Calibration, CommandReceipt, Control, RobotObservation


def until(predicate):
    deadline = time.monotonic() + 2.
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.001)
    raise AssertionError("bounded condition did not become true")


@pytest.fixture
def spawn():
    threads, errors = [], []

    def start(callback):
        def run():
            try:
                callback()
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=run, daemon=True)
        threads.append(thread)
        thread.start()
        return thread

    yield start
    for thread in threads:
        thread.join(timeout=3.)
    assert not any(thread.is_alive() for thread in threads), "model lease deadlocked"
    if errors:
        raise errors[0]


class ProtocolWorker:
    """Receipt/snapshot fake; deliberately has neither physics nor a clock thread."""

    def __init__(self):
        self.config = SimpleNamespace(realtime=True, control_interval_s=.1,
                                      physics_dt_s=.01, calibration=Calibration())
        self.commands = []

    def command(self, command):
        self.commands.append(command)
        return CommandReceipt(command.run_id, command.session_id, command.sequence_id,
                              command.observation_id, command.issued_at_s, command.controls,
                              True, "executing", command.expires_at_s, 0.)

    def snapshot(self):
        return {"active": True, "backend": {"highwater": -1, "receipts": []}}

    def restore(self, state):
        assert not state["active"]

    def stop(self):
        pass


def observation():
    return RobotObservation("run", "session", "frame", 0., time.time(), 0., 0., 10.,
                            8, 8, bytes(192), Control(), "fake", "simulator_seconds")


def submit(session, deadline):
    session._deadline = deadline
    session._submit(observation())
    job = session._job
    session._threads()
    return job


def assert_expired_fallback(session, job):
    assert job["done"].wait(2.)
    assert job["error"] is job["outcome"] is None
    assert job["fallback_reason"] == "inference_expired"
    assert session.error is None
    session._quiescing = True
    session._begin(observation())
    opened = session._open
    assert opened["fallback_reason"] == "inference_expired"
    assert opened["action"] is opened["model_version"] is opened["policy_observation"] is None
    assert opened["command"].controls.drive_mps == 0.


def test_yield_drains_serial_inference_but_keeps_writer_transaction(spawn):
    access = ModelAccess()
    first_inference, release_inference = threading.Event(), threading.Event()
    writer_entered = threading.Event()
    order, versions = [], []
    parameters = {"value": 0}

    def infer():
        with access.inference() as version:
            versions.append(version)
            order.append("infer")
            assert parameters["value"] == 0 and version == 0
            assert access.diagnostics()["inference_active"]
            first_inference.set()
            assert release_inference.wait(2.)

    def competing_writer():
        with access.update() as publish:
            writer_entered.set()
            assert parameters["value"] == 1 and access.version == 1
            order.append("writer")
            publish()

    def observe_handoff():
        assert first_inference.wait(2.)
        assert access.diagnostics()["update_active"]
        assert len(versions) == 1  # Other inference requests still cannot enter.
        assert not writer_entered.is_set()
        release_inference.set()

    with access.update() as publish:
        for _ in range(3):
            spawn(infer)
        until(lambda: access.diagnostics()["waiting_inference"] == 3)
        spawn(competing_writer)
        until(lambda: access.diagnostics()["waiting_updates"] == 1)
        spawn(observe_handoff)
        assert access.cooperate(lambda: order.append("barrier"))
        assert order == ["barrier", "infer", "infer", "infer", "barrier"]
        assert versions == [0, 0, 0] and access.version == 0
        assert not writer_entered.is_set()
        parameters["value"] = 1  # Optimizer mutation and publication stay atomic.
        publish()
        with pytest.raises(RuntimeError, match="after publication"):
            access.cooperate()
    assert writer_entered.wait(2.)
    until(lambda: access.version == 2)
    stats = access.diagnostics()
    assert stats["cooperative_yields"] == 1 and stats["inference_leases"] == 3
    assert stats["cooperate_wait_s"] >= stats["max_cooperate_wait_s"] > 0.
    assert stats["inference_wait_s"] >= stats["max_inference_wait_s"] > 0.
    stats["cooperative_yields"] = -1
    assert access.diagnostics()["cooperative_yields"] == 1


@pytest.mark.parametrize("failure_call", [1, 2])
def test_barrier_failure_reclaims_access_without_publishing(failure_call, spawn):
    access = ModelAccess()
    seen, barrier_calls = [], []
    writer_entered = threading.Event()

    def infer():
        with access.inference() as version:
            seen.append(version)

    def writer():
        with access.update() as publish:
            writer_entered.set()
            assert access.version == 0
            publish()

    def barrier():
        barrier_calls.append(1)
        if len(barrier_calls) == failure_call:
            raise ValueError("barrier failed")

    with access.update():
        spawn(infer)
        until(lambda: access.diagnostics()["waiting_inference"] == 1)
        spawn(writer)
        until(lambda: access.diagnostics()["waiting_updates"] == 1)
        with pytest.raises(ValueError, match="barrier failed"):
            access.cooperate(barrier)
        assert access.version == 0 and not writer_entered.is_set()
        if failure_call == 1:
            assert not seen
            assert access.cooperate()
        else:
            assert not access.cooperate()
        assert seen == [0]
    assert writer_entered.wait(2.)


def test_inference_exception_does_not_strand_cooperative_owner(spawn):
    access = ModelAccess()
    finished = threading.Event()

    def fail_inference():
        with pytest.raises(ValueError, match="inference failed"):
            with access.inference() as version:
                assert version == 0
                raise ValueError("inference failed")
        finished.set()

    with pytest.raises(ValueError, match="fit rolled back"):
        with access.update():
            spawn(fail_inference)
            until(lambda: access.diagnostics()["waiting_inference"] == 1)
            assert access.cooperate()
            assert finished.wait(2.)
            raise ValueError("fit rolled back")
    with access.inference() as version:
        assert version == 0
    with access.update() as publish:
        publish()
    assert access.version == 1


def test_interrupted_yield_reclaims_and_synchronizes_before_rollback(monkeypatch, spawn):
    access = ModelAccess()
    original_wait = access._condition.wait_for
    interrupted, barriers, seen = [], [], []

    def wait(predicate, timeout=None):
        if access._cooperating and not access._active and not interrupted:
            interrupted.append(True)
            raise InterruptedError("interrupted yield")
        return original_wait(predicate, timeout)

    def infer():
        with access.inference() as version:
            seen.append(version)

    with access.update() as publish:
        spawn(infer)
        until(lambda: access.diagnostics()["waiting_inference"] == 1)
        monkeypatch.setattr(access._condition, "wait_for", wait)
        with pytest.raises(InterruptedError, match="interrupted yield"):
            access.cooperate(lambda: barriers.append(True))
        assert barriers == [True, True] and seen == [0]
        assert not access.cooperate()
        publish()


def test_cooperate_and_publication_reject_wrong_owner_and_stale_leases(spawn):
    access = ModelAccess()
    with pytest.raises(RuntimeError, match="active update owner"):
        access.cooperate()
    with access.inference():
        with pytest.raises(RuntimeError, match="active update owner"):
            access.cooperate()
        with pytest.raises(RuntimeError, match="nested"):
            with access.inference():
                pass
        with pytest.raises(RuntimeError, match="nested"):
            with access.update():
                pass
    with access.update() as publish:
        assert not access.cooperate(lambda: pytest.fail("idle barrier called"))
        with pytest.raises(TypeError, match="callable"):
            access.cooperate(barrier=1)
        with pytest.raises(RuntimeError, match="nested"):
            with access.update():
                pass
        with pytest.raises(RuntimeError, match="nested"):
            with access.inference():
                pass

        def wrong_owner():
            with pytest.raises(RuntimeError, match="active update owner"):
                access.cooperate()
            with pytest.raises(RuntimeError, match="active update owner"):
                publish()
        thread = spawn(wrong_owner)
        thread.join(timeout=2.)
        assert not thread.is_alive()
        publish()
        with pytest.raises(RuntimeError, match="only once"):
            publish()
    with access.update() as new_publish:
        with pytest.raises(RuntimeError, match="active update owner"):
            publish()
        new_publish()
    assert access.version == 2


def test_barrier_cannot_nest_cooperate_lease_or_publication(spawn):
    access = ModelAccess()

    def infer():
        with access.inference() as version:
            assert version == 0

    with access.update() as publish:
        spawn(infer)
        until(lambda: access.diagnostics()["waiting_inference"] == 1)

        def barrier():
            with pytest.raises(RuntimeError, match="cannot be nested"):
                access.cooperate()
            with pytest.raises(RuntimeError, match="nested"):
                with access.inference():
                    pass
            with pytest.raises(RuntimeError, match="active update owner"):
                publish()
        assert access.cooperate(barrier)
        publish()


def test_noncooperative_update_keeps_legacy_atomic_priority(spawn):
    access = ModelAccess()
    order = []

    def infer():
        with access.inference() as version:
            order.append(("infer", version))

    def writer():
        with access.update() as publish:
            order.append(("writer", access.version))
            publish()

    with access.update() as publish:
        spawn(writer)
        until(lambda: access.diagnostics()["waiting_updates"] == 1)
        spawn(infer)
        until(lambda: access.diagnostics()["waiting_inference"] == 1)
        assert not order
        publish()
        assert not order
    until(lambda: len(order) == 2)
    assert order == [("infer", 1), ("writer", 1)]
    assert access.diagnostics()["cooperative_yields"] == 0


class Learner:
    def __init__(self):
        self.version = 0
        self.calls = 0
        self.fit_kwargs = []

    def choose_action(self, history, *, explore):
        self.calls += 1
        return {"controls": Control(drive_mps=.1), "model_version": self.version}

    def fit(self, **kwargs):
        self.fit_kwargs.append(kwargs)
        if "cooperate" in kwargs:
            assert not kwargs["cooperate"]()
        self.version += 1
        return self.version

    def state_dict(self):
        return {"version": self.version}

    def load_state_dict(self, state):
        self.version = state["version"]


@pytest.mark.parametrize("mode", ["callback", "learner"])
@pytest.mark.parametrize("expired_while_waiting", [False, True])
def test_expiry_skips_expensive_work_and_is_unlabelled_fallback(mode, expired_while_waiting):
    access, worker, learner = ModelAccess(), ProtocolWorker(), Learner()
    calls = []
    kwargs = ({"infer": lambda obs, version: calls.append("infer")}
              if mode == "callback" else
              {"learner": learner, "history_builder": lambda obs: calls.append("history")})
    with ControlSession(worker, access, **kwargs) as session:
        if expired_while_waiting:
            with access.update():
                job = submit(session, time.monotonic() + 60.)
                until(lambda: access.diagnostics()["waiting_inference"] == 1)
                # Advance expiry deterministically while GPU access is blocked.
                job["deadline"] = time.monotonic() - 1.
        else:
            job = submit(session, time.monotonic() - 1.)
        assert_expired_fallback(session, job)
        assert not calls and learner.calls == 0
        assert job["compute_s"] == 0.
        assert access.diagnostics()["inference_leases"] == int(expired_while_waiting)
        if expired_while_waiting:
            assert job["lease_wait_s"] > 0.


def test_expiry_during_history_build_does_not_consume_actor_rng():
    learner, access = Learner(), ModelAccess()

    def history(obs):
        session._job["deadline"] = time.monotonic() - 1.
        return {"frame": obs.observation_id}

    with ControlSession(ProtocolWorker(), access, learner=learner, history_builder=history) as session:
        job = submit(session, time.monotonic() + 60.)
        assert_expired_fallback(session, job)
        assert learner.calls == 0


def test_expired_prediction_keeps_fallback_tag_across_snapshot_restore():
    with ControlSession(ProtocolWorker(), ModelAccess(), infer=lambda obs, version: Control()) as session:
        job = submit(session, time.monotonic() - 1.)
        assert job["done"].wait(2.)
        state = session.snapshot()
    with ControlSession(ProtocolWorker(), ModelAccess(), infer=lambda obs, version: Control()) as restored:
        restored.restore(state)
        assert_expired_fallback(restored, restored._job)


def test_control_update_only_forwards_cooperate_when_opted_in():
    learner, access = Learner(), ModelAccess()
    with ControlSession(ProtocolWorker(), access, learner=learner,
                        history_builder=lambda obs: {}) as session:
        assert session.update(new="one") == 1
        assert learner.fit_kwargs == [{"new": "one"}]
        assert session.update(cooperative=True, new="two") == 2
        assert callable(learner.fit_kwargs[-1]["cooperate"])
        assert learner.fit_kwargs[-1]["new"] == "two"
        assert access.version == learner.version == 2
        with pytest.raises(ValueError, match="boolean"):
            session.update(cooperative=1)
        with pytest.raises(ValueError, match="own cooperate callback"):
            session.update(cooperative=True, cooperate=lambda: None)


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_cooperative_fit_inference_sees_only_last_published_model(device_type, monkeypatch):
    learner, access = Learner(), ModelAccess()
    learner.device = SimpleNamespace(type=device_type, index=1)
    synchronized = []
    original_import = builtins.__import__

    def import_guard(name, *args, **kwargs):
        assert name != "torch", "CPU/no-op cooperative call imported torch"
        return original_import(name, *args, **kwargs)

    def synchronize(device):
        assert device is learner.device
        assert access.diagnostics()["update_active"]
        assert not access.diagnostics()["inference_active"]
        synchronized.append(device)

    def fit(*, cooperate):
        with monkeypatch.context() as guarded:
            guarded.setattr(builtins, "__import__", import_guard)
            assert not cooperate()  # No-op CUDA handoffs must not import/sync.
        job = submit(session, time.monotonic() + 60.)
        until(lambda: access.diagnostics()["waiting_inference"] == 1)
        if device_type == "cuda":
            monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
                cuda=SimpleNamespace(synchronize=synchronize)))
            assert cooperate()
        else:
            with monkeypatch.context() as guarded:
                guarded.setattr(builtins, "__import__", import_guard)
                assert cooperate()
        assert job["done"].wait(2.)
        assert job["error"] is None and job["outcome"]["model_version"] == 0
        assert len(synchronized) == (2 if device_type == "cuda" else 0)
        assert access.version == learner.version == 0
        learner.version += 1
        return "accepted"

    learner.fit = fit
    with ControlSession(ProtocolWorker(), access, learner=learner,
                        history_builder=lambda obs: {}) as session:
        assert session.update(cooperative=True) == "accepted"
        assert learner.calls == 1 and access.version == learner.version == 1


def test_inference_cannot_reenter_control_model_events_during_yield():
    access = ModelAccess()

    def infer(obs, version):
        with pytest.raises(ControlSessionError, match="inside inference"):
            session.update(callback=lambda: None)
        with pytest.raises(ControlSessionError, match="inside inference"):
            session.snapshot()
        return Control()

    with ControlSession(ProtocolWorker(), access, infer=infer) as session:
        def update():
            job = submit(session, time.monotonic() + 60.)
            until(lambda: access.diagnostics()["waiting_inference"] == 1)
            assert access.cooperate()
            assert job["done"].wait(2.) and job["error"] is None
        session.update(callback=update)
        assert access.version == 1


def test_callback_can_yield_manually_and_fail_without_publication():
    access = ModelAccess()
    with ControlSession(ProtocolWorker(), access,
                        infer=lambda obs, version: Control(drive_mps=.1)) as session:
        def fail():
            job = submit(session, time.monotonic() + 60.)
            until(lambda: access.diagnostics()["waiting_inference"] == 1)
            assert access.cooperate()
            assert job["done"].wait(2.) and job["outcome"]["model_version"] == 0
            with pytest.raises(ControlSessionError, match="nested"):
                session.update(callback=lambda: None)
            with pytest.raises(ControlSessionError, match="inside an update"):
                session.snapshot()
            raise ValueError("callback rolled back")
        with pytest.raises(ValueError, match="callback rolled back"):
            session.update(callback=fail, cooperative=True)
        assert access.version == 0 and not session._updating
        assert session.update(callback=lambda: "accepted") == "accepted"
        assert access.version == 1
