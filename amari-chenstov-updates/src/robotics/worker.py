"""Independent, clocked simulator owner with bounded local process IPC.

This module never connects to a robot. Physics continues while the parent is
busy with inference/learning. RPC is explicit and serialized; no automatic
motion retry occurs on a timeout. A timed-out command may have been applied,
so use the command sequence's deduplication when reconciling acknowledgments.
"""

from __future__ import annotations

from dataclasses import replace
import math
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
from typing import Any

from .protocol import CommandReceipt, RobotCommand, RobotObservation, StepResult
from .simulator import PyBulletBackend, SimulatorConfig

WORKER_SCHEMA_VERSION = 1
MAX_MESSAGE_BYTES = 64_000_000


class WorkerError(RuntimeError):
    pass


def _encode(value: Any) -> bytes:
    data = pickle.dumps(value, protocol=5)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("worker message exceeds the bounded IPC payload size")
    return data


def _reply(replies, request_id: int, *, result: Any = None, error: str | None = None) -> None:
    try:
        data = _encode({"id": request_id, "result": result, "error": error})
    except Exception as exc:
        data = _encode({"id": request_id, "error": f"serialization failed: {exc}", "result": None})
    try:
        replies.put_nowait(data)
    except queue.Full:
        # A missing/slow parent must never block the physics owner on a reply.
        pass


def _worker_main(config: SimulatorConfig, run_id: str, requests, replies, parent_pid: int) -> None:
    backend = None
    try:
        backend = PyBulletBackend(config, run_id=run_id)
        ready, active, training = False, False, True
        next_tick = time.monotonic()
        missed_ticks = 0
        started = time.monotonic()
        capture_ticks = round(config.control_interval_s / config.physics_dt_s)
        last_capture_tick = 0
        _reply(replies, 0, result={"pid": os.getpid()})
        while os.getppid() == parent_pid:
            running = ready and active and backend.diagnostics()["end_reason"] is None
            if running:
                now = time.monotonic()
                if now >= next_tick:
                    missed_ticks += max(0, int((now - next_tick) / config.physics_dt_s))
                    backend.advance(1, training=training, capture=False)
                    # Schedule from this actual tick, not a historical deadline.
                    # At most one physics tick executes per loop: no catch-up.
                    next_tick = now + config.physics_dt_s
                    ticks = backend.diagnostics()["physics_ticks"]
                    if ticks - last_capture_tick >= capture_ticks:
                        backend.observe()
                        last_capture_tick = ticks
            wait_s = max(0., min(.01, next_tick - time.monotonic())) if running else .01
            try:
                encoded = requests.get(timeout=wait_s)
            except queue.Empty:
                continue
            request = pickle.loads(encoded)  # Trusted, private parent/child IPC only.
            request_id, method, kwargs = request["id"], request["method"], request["kwargs"]
            if time.monotonic() > request["deadline"]:
                _reply(replies, request_id, error="RPC expired before application")
                continue
            try:
                if method == "close":
                    _reply(replies, request_id, result=None)
                    break
                if method == "reset":
                    training = kwargs.pop("training")
                    result = backend.reset(**kwargs)
                    ready, active = True, True
                    last_capture_tick = 0
                    next_tick = time.monotonic() + config.physics_dt_s
                elif method == "restore":
                    snapshot = kwargs["snapshot"]
                    if "worker_schema_version" in snapshot:
                        if snapshot["worker_schema_version"] != WORKER_SCHEMA_VERSION:
                            raise ValueError("incompatible worker snapshot")
                        active, training = snapshot["active"], snapshot["training"]
                        backend.restore(snapshot["backend"])
                    else:
                        backend.restore(snapshot)
                        active = True
                    ready = True
                    last_capture_tick = backend.diagnostics()["physics_ticks"]
                    next_tick = time.monotonic() + config.physics_dt_s
                    result = None  # Restore must not invent an extra camera frame.
                elif not ready:
                    raise RuntimeError("reset or restore the worker before use")
                elif method == "command":
                    if not active:
                        raise RuntimeError("paused simulator rejects commands")
                    result = backend.command(kwargs["command"])
                elif method == "observe":
                    result = backend.observe()
                elif method == "harvest":
                    result = backend.advance(0)
                    result = replace(result, timing={**result.timing,
                                     "worker_wall_elapsed_s": time.monotonic() - started,
                                     "worker_missed_physics_ticks": float(missed_ticks),
                                     "worker_pid": float(os.getpid()), "worker_realtime": 1.})
                elif method == "snapshot":
                    result = {"worker_schema_version": WORKER_SCHEMA_VERSION,
                              "backend": backend.snapshot(), "active": active, "training": training}
                elif method == "diagnostics":
                    result = {**backend.diagnostics(), "worker_pid": os.getpid(),
                              "worker_wall_elapsed_s": time.monotonic() - started,
                              "worker_missed_physics_ticks": missed_ticks, "worker_active": active}
                elif method == "render_topdown":
                    result = backend.render_topdown()
                elif method == "stop":
                    backend.stop()
                    result = None
                elif method == "pause":
                    backend.stop()
                    active = False
                    result = None
                elif method == "resume":
                    training = kwargs["training"]
                    active = True
                    next_tick = time.monotonic() + config.physics_dt_s
                    result = None
                else:
                    raise ValueError("unsupported simulator worker RPC")
                _reply(replies, request_id, result=result)
            except Exception as exc:
                _reply(replies, request_id, error=f"{type(exc).__name__}: {exc}")
    except BaseException as exc:
        _reply(replies, 0, error=f"worker startup/clock failed: {type(exc).__name__}: {exc}")
    finally:
        if backend is not None:
            backend.close()
        # Do not wait forever flushing a response to a dead parent.
        replies.cancel_join_thread()


class SimulatorWorker:
    """Spawned realtime Bullet owner. Construction alone starts nothing.

    reset() starts active physics; harvest() consumes accumulated elapsed time
    and the once-only terminal component, rather than advancing the world.
    pause() is explicit stationary administration for sleep/checkpointing; it
    is not an inference scheduling tool. stop() stops controls but keeps time.
    """

    def __init__(self, config: SimulatorConfig | None = None, *, run_id: str = "simulator-worker",
                 rpc_timeout_s: float = 10., queue_capacity: int = 2):
        self.config = config or SimulatorConfig()
        if not self.config.realtime:
            raise ValueError("SimulatorWorker is realtime only; accelerated smoke uses PyBulletBackend")
        if not math.isfinite(rpc_timeout_s) or not 0 < rpc_timeout_s <= 60:
            raise ValueError("RPC timeout must be finite and in (0, 60]")
        if type(queue_capacity) is not int or not 1 <= queue_capacity <= 8:
            raise ValueError("IPC queue capacity must be in [1, 8]")
        self.run_id, self.rpc_timeout_s = run_id, rpc_timeout_s
        self.queue_capacity = queue_capacity
        self._process = None
        self._requests = self._replies = None
        self._closed = False
        self._request_id = 0
        self._lock = threading.RLock()
        self.pid: int | None = None

    def start(self) -> SimulatorWorker:
        with self._lock:
            if self._closed:
                raise WorkerError("simulator worker is closed")
            if self._process is not None:
                return self
            context = mp.get_context("spawn")
            self._requests = context.Queue(maxsize=self.queue_capacity)
            self._replies = context.Queue(maxsize=self.queue_capacity)
            self._process = context.Process(target=_worker_main,
                args=(self.config, self.run_id, self._requests, self._replies, os.getpid()), daemon=True)
            try:
                self._process.start()
                self.pid = self._receive(0, time.monotonic() + self.rpc_timeout_s)["pid"]
            except BaseException:
                self.close()
                raise
            return self

    def _receive(self, request_id: int, deadline: float):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("simulator RPC timed out; application may need reconciliation")
            try:
                response = pickle.loads(self._replies.get(timeout=min(remaining, .1)))
            except queue.Empty:
                if not self._process.is_alive():
                    raise WorkerError(f"simulator worker exited with code {self._process.exitcode}")
                continue
            if response["id"] != request_id:
                continue  # Discard a response whose caller previously timed out.
            if response.get("error"):
                raise WorkerError(response["error"])
            return response.get("result")

    def _rpc(self, method: str, **kwargs):
        with self._lock:
            self.start()
            self._request_id += 1
            deadline = time.monotonic() + self.rpc_timeout_s
            request = _encode({"id": self._request_id, "method": method,
                               "kwargs": kwargs, "deadline": deadline})
            try:
                self._requests.put(request, timeout=self.rpc_timeout_s)
            except queue.Full as exc:
                raise WorkerError("simulator request queue is full") from exc
            return self._receive(self._request_id, deadline)

    def reset(self, *, seed: int, exposure_intervals: float | None = None,
              training: bool = True) -> RobotObservation:
        return self._rpc("reset", seed=seed, exposure_intervals=exposure_intervals, training=training)

    def command(self, command: RobotCommand) -> CommandReceipt:
        return self._rpc("command", command=command)

    def observe(self) -> RobotObservation:
        return self._rpc("observe")

    def harvest(self) -> StepResult:
        return self._rpc("harvest")

    def snapshot(self) -> dict:
        return self._rpc("snapshot")

    def restore(self, snapshot: dict) -> None:
        self._rpc("restore", snapshot=snapshot)

    def diagnostics(self) -> dict:
        return self._rpc("diagnostics")

    def render_topdown(self):
        return self._rpc("render_topdown")

    def stop(self) -> None:
        self._rpc("stop")

    def pause(self) -> None:
        self._rpc("pause")

    def resume(self, *, training: bool = True) -> None:
        self._rpc("resume", training=training)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            process = self._process
            if process is not None:
                if process.pid is not None and process.is_alive():
                    try:
                        self._rpc("close")
                    except (WorkerError, TimeoutError, OSError, ValueError):
                        pass
                    process.join(timeout=2.)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2.)
                process.close()
            for channel in (self._requests, self._replies):
                if channel is not None:
                    channel.cancel_join_thread()
                    channel.close()
            self._process = None
            self._closed = True

    def __enter__(self) -> SimulatorWorker:
        return self.start()

    def __exit__(self, *args) -> None:
        self.close()
