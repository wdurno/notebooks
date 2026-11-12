"""Realtime control-session adapter, not a scientific trajectory scheduler.

The worker owns physics. A control thread closes acquisition intervals even
while inference or fitting owns ModelAccess. One inference slot and a bounded
outcome backlog prevent unbounded work. No return/reward/model is defined here.
Callbacks/history builders must use immutable information available at the
supplied observation, never current privileged state or future frames.

ControlInterval is telemetry, not automatic training admission. Safe fallback
has action=None and no policy version; neither it nor queue overflow invents
Gaussian policy labels. Callback Control outputs are not Gaussian samples.
Phase1 owns real context construction, completed returns and eligibility gates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import queue
import threading
import time
from typing import Any, Callable

from src.robotics import Calibration, CommandReceipt, Control, RobotCommand, RobotObservation, StepResult
from .runtime import ModelAccess

CONTROL_SCHEMA_VERSION = 1


class ControlSessionError(RuntimeError):
    """Infrastructure failure: unfinished intervals must not acquire zero returns."""


@dataclass(frozen=True)
class ControlInterval:
    index: int
    observation: RobotObservation
    policy_observation: RobotObservation | None
    command: RobotCommand
    receipt: CommandReceipt
    action: dict[str, Any] | None
    model_version: int | None
    fallback_reason: str | None
    result: StepResult
    timing: dict[str, float]


class ControlSession:
    """Adapter for SimulatorWorker + ModelAccess + optional MemoryLearner.

    Supply either infer(observation, leased_version), returning Control or an
    action dictionary, or learner/history_builder for choose_action(). update()
    runs a callback or learner.fit(**kwargs) and publishes exactly once.

    start() resets an episode and starts clocked acquisition. next_interval()
    consumes complete command/outcome pairs. A startup or missed-deadline
    interval executes explicit safe neutral, never a fabricated policy action.
    Queue exhaustion stops control and marks infrastructure failure; it does
    not pause physics, silently lose evidence or synthesize a terminal label.

    snapshot() quiesces inference, not physics, and includes learner state when
    available. restore() requires a fresh session and leaves recovery paused;
    resume() explicitly restarts the clock. No unacknowledged motion is replayed.
    The caller must persist external language/history state with this payload.
    """

    def __init__(self, worker, access: ModelAccess, *, learner=None,
                 history_builder: Callable[[RobotObservation], dict] | None = None,
                 infer: Callable[[RobotObservation, int], Control | dict] | None = None,
                 explore: bool = True, control_period_s: float | None = None,
                 pending_capacity: int = 32, own_worker: bool = False):
        if infer is None and (learner is None or history_builder is None):
            raise ValueError("provide infer callback or learner and history_builder")
        if infer is not None and (learner is not None or history_builder is not None):
            raise ValueError("callback and learner inference modes are mutually exclusive")
        self.worker, self.access, self.learner = worker, access, learner
        self.history_builder, self.infer, self.explore = history_builder, infer, explore
        cfg = worker.config
        if not cfg.realtime:
            raise ValueError("control sessions require an independent realtime worker")
        self.period_s = cfg.control_interval_s if control_period_s is None else control_period_s
        if (not math.isfinite(self.period_s) or not 0 < self.period_s <= cfg.control_interval_s
                or self.period_s < cfg.physics_dt_s):
            raise ValueError("control period must fit the backend's physics/control limits")
        if type(pending_capacity) is not int or not 1 <= pending_capacity <= 4096:
            raise ValueError("pending capacity must be in [1, 4096]")
        self.calibration: Calibration = cfg.calibration
        self.pending_capacity, self.own_worker = pending_capacity, own_worker
        self._condition = threading.Condition(threading.RLock())
        self._cycle_lock = threading.RLock()
        self._model_events = threading.RLock()
        self._stop = threading.Event()
        self._inference_stop = threading.Event()
        self._inference_queue: queue.Queue = queue.Queue(maxsize=1)
        self._inference_thread = None
        self._control_thread = None
        self._pending: list[ControlInterval] = []
        self._job: dict | None = None
        self._open: dict | None = None
        self._pending_command: RobotCommand | None = None
        self._next_sequence = 0
        self._index = 0
        self._deadline = 0.
        self._started = self._running = self._paused = self._terminal = False
        self._closed = self._quiescing = self._updating = False
        self._failure: str | None = None
        self._recovery = False
        self._frozen_at: float | None = None
        self._training = True
        self._updates_s = 0.
        self._last_update_s = 0.
        self._decision_count = self._fallback_count = 0

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._pending)

    @property
    def error(self) -> str | None:
        with self._condition:
            return self._failure

    @property
    def episode_ended(self) -> bool:
        with self._condition:
            return self._terminal

    def _check(self, *, allow_failed: bool = False) -> None:
        if self._closed:
            raise ControlSessionError("control session is closed")
        if self._failure and not allow_failed:
            raise ControlSessionError(self._failure)

    def _threads(self) -> None:
        if self._inference_thread is None:
            self._inference_thread = threading.Thread(target=self._infer_loop, daemon=True,
                                                       name="picar-control-inference")
            self._inference_thread.start()

    def start(self, *, seed: int, exposure_intervals: float | None = None,
              training: bool = True) -> RobotObservation:
        with self._cycle_lock:
            self._check()
            if self._running or self._started and not self._terminal:
                raise ControlSessionError("an episode is already active; use explicit recovery resume")
            if self._job is not None and not self._job["done"].is_set():
                raise ControlSessionError("finish in-flight inference before resetting an episode")
            if self._control_thread is not None:
                self._control_thread.join(timeout=1.)
                if self._control_thread.is_alive():
                    raise ControlSessionError("previous episode's control thread has not stopped")
            if self.learner is not None and self.learner.version != self.access.version:
                raise ControlSessionError("learner and publication versions differ")
            observation = self.worker.reset(seed=seed, exposure_intervals=exposure_intervals, training=training)
            self._threads()
            self._training = training
            self._started, self._terminal, self._paused = True, False, False
            self._job = None
            self._stop.clear()
            try:
                self._begin(observation, startup=True)
            except BaseException as exc:
                self._fail(exc)
                raise
            self._running = True
            self._control_thread = threading.Thread(target=self._control_loop, daemon=True,
                                                     name="picar-control-clock")
            self._control_thread.start()
            return observation

    def _submit(self, observation: RobotObservation) -> None:
        with self._condition:
            if self._quiescing or self._job is not None:
                return
            job = {"observation": observation, "requested_at": time.monotonic(),
                   "deadline": self._deadline, "completed_at": None, "outcome": None,
                   "error": None, "fallback_reason": None, "done": threading.Event(),
                   "lease_wait_s": 0., "compute_s": 0.}
            self._job = job
            self._inference_queue.put_nowait(job)

    def _infer_loop(self) -> None:
        while not self._inference_stop.is_set():
            try:
                job = self._inference_queue.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                if time.monotonic() >= job["deadline"]:
                    job["fallback_reason"] = "inference_expired"
                    continue
                with self.access.inference() as version:
                    entered = time.monotonic()
                    job["lease_wait_s"] = entered - job["requested_at"]
                    if entered >= job["deadline"]:
                        job["fallback_reason"] = "inference_expired"
                        continue
                    if self.infer is not None:
                        action = self.infer(job["observation"], version)
                    else:
                        if self.learner.version != version:
                            raise ControlSessionError("learner and leased model versions differ")
                        history = self.history_builder(job["observation"])
                        if time.monotonic() >= job["deadline"]:
                            job["compute_s"] = time.monotonic() - entered
                            job["fallback_reason"] = "inference_expired"
                            continue
                        action = self.learner.choose_action(history, explore=self.explore)
                    if isinstance(action, Control):
                        action = {"controls": action, "model_version": version}
                    elif isinstance(action, dict):
                        action = copy.deepcopy(action)
                        if action.get("model_version", version) != version:
                            raise ValueError("action model version differs from its inference lease")
                        action["model_version"] = version
                    else:
                        raise ValueError("inference must return Control or an action dictionary")
                    controls = action.get("controls")
                    if controls is None:
                        controls = self.calibration.map_distribution(action["probabilities"])
                    if not isinstance(controls, Control):
                        raise ValueError("action controls must use the shared Control schema")
                    self.calibration.validate(controls)
                    if "baseline" in action and not math.isfinite(float(action["baseline"])):
                        raise ValueError("baseline must be finite")
                    action["controls"] = controls
                    job["outcome"] = action
                    job["compute_s"] = time.monotonic() - entered
            except Exception as exc:
                job["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                job["completed_at"] = time.monotonic()
                job["done"].set()

    def _begin(self, observation: RobotObservation, *, startup: bool = False) -> None:
        now = time.monotonic()
        action = policy_observation = None
        reason = "startup" if startup else "inference_pending"
        infer_s = lease_s = 0.
        with self._condition:
            job = self._job
            if job is not None and job["done"].is_set():
                self._job = None
                infer_s, lease_s = job["compute_s"], job["lease_wait_s"]
                if job["error"]:
                    raise ControlSessionError(f"inference failed: {job['error']}")
                if job["observation"].session_id != observation.session_id:
                    reason = "stale_session_prediction"
                elif job.get("fallback_reason"):
                    reason = job["fallback_reason"]
                elif job["completed_at"] > job["deadline"]:
                    reason = "inference_expired"
                elif now > job["deadline"] + self.period_s:
                    reason = "prediction_stale"
                else:
                    action, policy_observation, reason = job["outcome"], job["observation"], None
        controls = (self.calibration.neutral(observation.executed_controls) if action is None
                    else action["controls"])
        command = RobotCommand(observation.run_id, observation.session_id, self._next_sequence,
                               (policy_observation or observation).observation_id, controls,
                               observation.capture_time_s, self.period_s,
                               observation.capture_time_s + self.period_s)
        self._next_sequence += 1  # Never reuse an ID after an ambiguous RPC.
        self._pending_command = command
        source = policy_observation or observation
        self._open = {"observation": observation, "policy_observation": policy_observation,
                      "command": command, "receipt": None, "action": action,
                      "model_version": None if action is None else action["model_version"],
                      "fallback_reason": reason, "wall_started": now,
                      "timing": {"decision_wall_s": time.time(), "inference_seconds": infer_s,
                                 "model_lease_wait_s": lease_s,
                                 "capture_to_command_s": 0.,
                                 "capture_to_command_wall_s": max(0., time.time() - source.wall_time_s),
                                 "model_age_versions": 0. if action is None else
                                 float(self.access.version - action["model_version"])}}
        receipt = self.worker.command(command)
        self._open["receipt"] = receipt
        self._open["timing"]["capture_to_command_s"] = receipt.capture_to_command_s or 0.
        self._pending_command = None
        if not receipt.accepted:
            if receipt.safety_status == "episode_ended":
                # A final physics tick can land between harvest and receipt.
                # Preserve the rejected request, never an invented policy label.
                self._open.update(action=None, model_version=None, policy_observation=None,
                                  fallback_reason="episode_ended_before_command")
                self._deadline = time.monotonic()
                self._decision_count += 1
                self._fallback_count += 1
                return
            raise ControlSessionError(f"worker rejected control: {receipt.safety_status}")
        self._decision_count += 1
        self._fallback_count += int(reason is not None)
        self._deadline = time.monotonic() + self.period_s
        self._submit(observation)

    def _control_loop(self) -> None:
        try:
            while not self._stop.wait(max(0., self._deadline - time.monotonic())):
                with self._cycle_lock:
                    if self._stop.is_set() or self._paused:
                        break
                    with self._condition:
                        if len(self._pending) >= self.pending_capacity:
                            raise ControlSessionError("bounded pending interval queue exhausted; episode incomplete")
                    late = max(0., time.monotonic() - self._deadline)
                    result = self.worker.harvest()
                    opened = self._open
                    if opened is None or result.reward_components.elapsed_s <= 0:
                        raise ControlSessionError("worker returned no new physical interval")
                    interval = ControlInterval(
                        self._index, opened["observation"], opened["policy_observation"],
                        opened["command"], opened["receipt"], opened["action"], opened["model_version"],
                        opened["fallback_reason"], result,
                        {**opened["timing"], "elapsed_s": result.reward_components.elapsed_s,
                         "interval_wall_s": time.monotonic() - opened["wall_started"],
                         "harvest_wall_s": time.time(), "control_deadline_lag_s": late,
                         "missed_control_deadlines": float(int(late / self.period_s)),
                         "inference_deadline_miss": float(opened["fallback_reason"] not in (None, "startup")),
                         "acquisition_intervals": result.reward_components.elapsed_s / self.period_s,
                         "update_seconds_total": self._updates_s, "last_update_seconds": self._last_update_s})
                    self._index += 1
                    self._open = None
                    with self._condition:
                        self._pending.append(interval)
                        if result.terminated or result.truncated:
                            self._terminal = True
                            self._running = False
                        self._condition.notify_all()
                    if result.terminated or result.truncated:
                        break
                    self._begin(result.observation)
        except Exception as exc:
            self._fail(exc)
        finally:
            with self._condition:
                self._running = False
                self._condition.notify_all()

    def _fail(self, exc: BaseException) -> None:
        with self._cycle_lock:
            self._stop.set()
            try:
                self.worker.stop()
            except Exception:
                pass  # The worker still independently enforces command expiry.
            with self._condition:
                self._failure = f"{type(exc).__name__}: {exc}"
                self._running = False
                self._condition.notify_all()

    def next_interval(self, timeout: float | None = None) -> ControlInterval:
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        with self._condition:
            if not self._started:
                raise ControlSessionError("start or restore a session before consuming intervals")
            ready = self._condition.wait_for(
                lambda: self._pending or self._failure or self._terminal or self._closed, timeout=timeout)
            if self._pending:
                return self._pending.pop(0)
            self._check()
            if self._terminal:
                raise StopIteration("episode ended; no pending intervals")
            if not ready:
                raise TimeoutError("no completed control interval before timeout")
            raise ControlSessionError("no active control interval")

    def _cooperate_update(self) -> bool:
        device = getattr(self.learner, "device", None)
        if getattr(device, "type", None) == "cuda":
            def barrier() -> None:
                import torch
                torch.cuda.synchronize(device)
            return self.access.cooperate(barrier=barrier)
        return self.access.cooperate()

    def update(self, callback: Callable[[], Any] | None = None, *,
               cooperative: bool = False, **fit_kwargs) -> Any:
        """Publish a successful callback/fit once; never pause for a model lease.

        Callbacks must roll back mutations on failure. With a MemoryLearner,
        callbacks must use its own publication/version contract (e.g. fit()).
        Opt-in cooperative fitting passes a no-argument cooperate callback to
        learner.fit, synchronizing a CUDA learner's device only on handoffs.
        Callbacks can call access.cooperate directly. Only the learner/callback
        can certify graph-safe boundaries before optimizer mutation.
        """
        self._check()
        if callback is None and self.learner is None:
            raise ValueError("callback required without a learner")
        if callback is not None and fit_kwargs:
            raise ValueError("choose callback or learner.fit keyword arguments")
        if type(cooperative) is not bool:
            raise ValueError("cooperative must be a boolean")
        if cooperative and "cooperate" in fit_kwargs:
            raise ValueError("cooperative fitting supplies its own cooperate callback")
        if threading.current_thread() is self._inference_thread:
            raise ControlSessionError("cannot update inside inference")
        with self._model_events:
            if self._updating:
                raise ControlSessionError("nested control updates are not allowed")
            self._updating = True
            started = time.monotonic()
            try:
                with self.access.update() as publish:
                    before = self.access.version
                    if cooperative and callback is None:
                        fit_kwargs["cooperate"] = self._cooperate_update
                    result = callback() if callback is not None else self.learner.fit(**fit_kwargs)
                    if self.learner is not None and self.learner.version != before + 1:
                        raise ControlSessionError("learner update must publish exactly one new model version")
                    publish()
                return result
            finally:
                self._last_update_s = time.monotonic() - started
                self._updates_s += self._last_update_s
                self._updating = False

    def pause(self, *, reason: str) -> None:
        if reason not in ("episode_boundary", "sleep"):
            raise ValueError("pause is only for an explicit episode boundary or sleep")
        with self._cycle_lock:
            self._check()
            if not self._started or not self._terminal:
                raise ControlSessionError("pause requires a completed episode, never an inference wait")
            self.worker.pause()
            self._paused = True
            self._frozen_at = time.monotonic()

    @staticmethod
    def _save_job(job: dict | None, now: float) -> dict | None:
        if job is None:
            return None
        if not job["done"].is_set():
            raise ControlSessionError("snapshot requires completed inference, not an in-flight GPU operation")
        saved = {key: copy.deepcopy(value) for key, value in job.items()
                 if key not in ("done", "requested_at", "deadline", "completed_at")}
        return {**saved, "requested_age_s": now - job["requested_at"],
                "deadline_remaining_s": job["deadline"] - now,
                "completed_age_s": now - job["completed_at"]}

    def snapshot(self, *, timeout: float = 10., capture_external=None) -> dict:
        """Capture optional caller history while inference and control are quiesced."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("snapshot timeout must be finite and positive")
        if capture_external is not None and not callable(capture_external):
            raise TypeError("capture_external must be callable")
        self._check(allow_failed=True)
        if threading.current_thread() is self._inference_thread:
            raise ControlSessionError("cannot snapshot inside inference")
        with self._model_events:
            if self._updating:
                raise ControlSessionError("cannot snapshot inside an update")
            self._quiescing = True
            try:
                with self._condition:
                    job = self._job
                if job is not None and not job["done"].wait(timeout):
                    raise TimeoutError("in-flight inference prevented a consistent snapshot")
                with self.access.inference() as version, self._cycle_lock:
                    self._check(allow_failed=True)
                    backend = self.worker.snapshot()
                    now = time.monotonic()
                    opened = copy.deepcopy(self._open)
                    if opened is not None:
                        opened["wall_age_s"] = now - opened.pop("wall_started")
                    with self._condition:
                        result = {"schema_version": CONTROL_SCHEMA_VERSION, "period_s": self.period_s,
                                "pending_capacity": self.pending_capacity, "worker": backend,
                                "learner": None if self.learner is None else self.learner.state_dict(),
                                "model_version": version, "next_sequence": self._next_sequence,
                                "index": self._index, "open": opened, "pending": copy.deepcopy(self._pending),
                                "pending_command": self._pending_command,
                                "prediction": self._save_job(self._job, now),
                                "deadline_remaining_s": self._deadline - now,
                                "terminal": self._terminal, "paused": self._paused,
                                "failure": self._failure,
                                "training": self._training, "updates_s": self._updates_s,
                                "last_update_s": self._last_update_s,
                                "decision_count": self._decision_count, "fallback_count": self._fallback_count}
                        if capture_external is not None:
                            result["external"] = capture_external()
                        return result
            finally:
                self._quiescing = False

    def restore(self, state: dict) -> None:
        """Fresh-session recovery; reconstruct but do not run physics until resume.

        An ambiguous acknowledgment is recovered only from the backend receipt
        ledger. Missing acknowledgment stays explicitly incomplete, never retried.
        """
        self._check()
        if self._started:
            raise ControlSessionError("restore requires a fresh control session")
        if (state["schema_version"] != CONTROL_SCHEMA_VERSION or state["period_s"] != self.period_s
                or state["pending_capacity"] != self.pending_capacity):
            raise ValueError("control snapshot schema/timing/capacity mismatch")
        if len(state["pending"]) > self.pending_capacity:
            raise ValueError("restored pending queue exceeds capacity")
        with self._model_events, self.access.update():
            if (state["learner"] is None) != (self.learner is None):
                raise ValueError("control snapshot learner mode mismatch")
            if self.learner is not None:
                self.learner.load_state_dict(state["learner"])
                if self.learner.version != state["model_version"]:
                    raise ValueError("restored learner/publication versions differ")
            self.access.version = state["model_version"]
            worker_state = copy.deepcopy(state["worker"])
            worker_state["active"] = False
            self.worker.restore(worker_state)
        now = time.monotonic()
        backend = state["worker"]["backend"]
        self._next_sequence = max(state["next_sequence"], backend["highwater"] + 1)
        self._index = state["index"]
        self._pending = copy.deepcopy(state["pending"])
        self._open = copy.deepcopy(state["open"])
        if self._open is not None:
            self._open["wall_started"] = now - self._open.pop("wall_age_s")
        self._deadline = now + state["deadline_remaining_s"]
        self._job = copy.deepcopy(state["prediction"])
        if self._job is not None:
            self._job["requested_at"] = now - self._job.pop("requested_age_s")
            self._job["deadline"] = now + self._job.pop("deadline_remaining_s")
            self._job["completed_at"] = now - self._job.pop("completed_age_s")
            self._job["done"] = threading.Event()
            self._job["done"].set()
        self._pending_command = state["pending_command"]
        self._failure = state["failure"]
        if self._pending_command is not None:
            command = self._pending_command
            self._next_sequence = max(self._next_sequence, command.sequence_id + 1)
            receipts = {key: (original, receipt) for key, original, receipt in backend["receipts"]}
            cached = receipts.get(command.sequence_id)
            if cached is None or cached[0] != command.to_dict():
                self._failure = "unacknowledged command absent from backend ledger; episode incomplete"
            else:
                # Persist the reconciled acknowledgment, without applying motion.
                self._pending_command = None
                if self._open is not None:
                    self._open["receipt"] = CommandReceipt.from_dict(cached[1])
        self._terminal, self._training = state["terminal"], state["training"]
        self._updates_s, self._last_update_s = state["updates_s"], state["last_update_s"]
        self._decision_count, self._fallback_count = state["decision_count"], state["fallback_count"]
        self._started, self._paused, self._recovery = True, True, True
        self._frozen_at = now
        self._threads()

    def resume(self) -> None:
        with self._cycle_lock:
            self._check()
            if not self._recovery or not self._paused or self._terminal or self._open is None:
                raise ControlSessionError("resume requires a restored unfinished interval; reset at episode boundaries")
            now = time.monotonic()
            gap = now - self._frozen_at
            self._deadline += gap
            self._open["wall_started"] += gap
            if self._job is not None:
                for name in ("requested_at", "deadline", "completed_at"):
                    self._job[name] += gap
            self.worker.resume(training=self._training)
            self._paused, self._recovery, self._running = False, False, True
            self._frozen_at = None
            self._stop.clear()
            self._control_thread = threading.Thread(target=self._control_loop, daemon=True,
                                                     name="picar-control-clock")
            self._control_thread.start()

    def close(self, *, timeout: float = 10.) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("close timeout must be finite and positive")
        if self._closed:
            return
        self._stop.set()
        self._inference_stop.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=timeout)
        try:
            if self._started:
                self.worker.stop()
        finally:
            if self.own_worker:
                self.worker.close()
            if self._inference_thread is not None:
                self._inference_thread.join(timeout=timeout)
            self._closed = True
            with self._condition:
                self._condition.notify_all()
        if ((self._control_thread is not None and self._control_thread.is_alive())
                or (self._inference_thread is not None and self._inference_thread.is_alive())):
            raise TimeoutError("control session stopped motion but an in-flight operation has not finished")

    def __enter__(self) -> ControlSession:
        return self

    def __exit__(self, *args) -> None:
        self.close()
