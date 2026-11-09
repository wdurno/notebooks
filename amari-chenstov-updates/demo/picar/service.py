"""Authenticated loopback robot service using Python's standard library only.

    No physical driver is shipped. Deploy behind an authenticated SSH tunnel;
    an actual hardware adapter requires separate calibration and authorization.
"""

from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
import secrets
import threading
import time
from typing import Callable, Protocol

from src.robotics.actions import Calibration
from src.robotics.protocol import CommandReceipt, Control, RobotCommand, RobotObservation, SCHEMA_VERSION


class Driver(Protocol):
    def apply(self, controls: Control) -> None: ...
    def stop(self) -> None: ...
    def capture_rgb(self) -> tuple[int, int, bytes]: ...


class FakeDriver:
    """In-memory driver. Constructor and import cannot actuate anything."""

    def __init__(self):
        self.controls = Control()
        self.applied: list[Control] = []
        self.stop_count = 0

    def apply(self, controls: Control) -> None:
        self.controls = controls
        self.applied.append(controls)

    def stop(self) -> None:
        self.controls = replace(self.controls, steering_rad=0., drive_mps=0.)
        self.stop_count += 1

    def capture_rgb(self) -> tuple[int, int, bytes]:
        return 8, 8, bytes((30, 90, 60)) * 64


@dataclass(frozen=True)
class ServiceConfig:
    watchdog_s: float = .75
    max_command_duration_s: float = .5
    max_expiry_ahead_s: float = 1.
    request_timeout_s: float = 2.
    max_payload_bytes: int = 8192
    watchdog_poll_s: float = .02
    calibration: Calibration = Calibration()

    def __post_init__(self) -> None:
        values = (self.watchdog_s, self.max_command_duration_s, self.max_expiry_ahead_s,
                  self.request_timeout_s, self.watchdog_poll_s)
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("service time limits must be finite and positive")
        if self.watchdog_poll_s > self.watchdog_s or self.max_command_duration_s > self.max_expiry_ahead_s:
            raise ValueError("inconsistent service time limits")
        if type(self.max_payload_bytes) is not int or not 256 <= self.max_payload_bytes <= 65536:
            raise ValueError("payload limit must be in [256, 65536]")


class RobotService:
    """Safety state machine, injectable monotonic clock and local driver.

    Every transport reconnection opens a NEW session. Neither it, watchdog
    expiry, operator stop nor a driver fault may implicitly re-arm motion.
    Rejected new sequence IDs are consumed too; a delayed retry cannot run later.
    """

    def __init__(self, *, driver: Driver, token: str, config: ServiceConfig | None = None,
                 clock: Callable[[], float] = time.monotonic):
        if not isinstance(token, str) or len(token.encode()) < 32:
            raise ValueError("authentication token must contain at least 32 bytes")
        self.config = config or ServiceConfig()
        self.driver, self._token, self._clock = driver, token, clock
        self._lock = threading.RLock()
        self._run_id = "disconnected"
        self._session_id = "disconnected"
        self._armed = False
        self._status = "disconnected"
        self._control = self.config.calibration.neutral()
        self._deadline = self._clock()
        self._highwater = -1
        self._receipts: OrderedDict[int, tuple[dict, CommandReceipt]] = OrderedDict()
        self._observations: OrderedDict[str, float] = OrderedDict()
        self._frame = 0
        self._started = self._clock()
        self._closed = False
        self._faulted = False
        self.driver.stop()

    def authenticate(self, token: str) -> bool:
        return isinstance(token, str) and hmac.compare_digest(token.encode(), self._token.encode())

    def _authorize(self, token: str) -> None:
        if not self.authenticate(token):
            raise PermissionError("authentication required")
        if self._closed:
            raise RuntimeError("service is closed")

    def _stop(self, reason: str) -> None:
        self._armed = False
        self._control = self.config.calibration.neutral(self._control)
        self._deadline = self._clock()
        self._status = reason
        try:
            self.driver.stop()
        except Exception:
            self._faulted = True
            self._status = "driver_fault"
            raise

    def connect(self, token: str, *, run_id: str) -> dict:
        with self._lock:
            self._authorize(token)
            if not isinstance(run_id, str) or not 0 < len(run_id) <= 100:
                raise ValueError("run_id must be a bounded nonempty string")
            self._stop("disarmed")
            self._run_id, self._session_id = run_id, secrets.token_hex(24)
            self._highwater = -1
            self._receipts.clear()
            self._observations.clear()
            return self.status(token)

    def status(self, token: str) -> dict:
        with self._lock:
            self._authorize(token)
            self.poll_watchdog()
            return {"schema_version": SCHEMA_VERSION, "run_id": self._run_id,
                    "session_id": self._session_id, "armed": self._armed,
                    "safety_status": self._status, "clock_time_s": self._clock(),
                    "clock_domain": "robot_monotonic_seconds", "last_sequence_id": self._highwater,
                    "calibration": asdict(self.config.calibration), "driver": type(self.driver).__name__}

    def arm(self, token: str, *, session_id: str) -> dict:
        with self._lock:
            self._authorize(token)
            self.poll_watchdog()
            if session_id != self._session_id or self._session_id == "disconnected":
                raise ValueError("unknown session")
            if self._faulted:
                raise RuntimeError("driver fault requires local service restart")
            if self._armed:
                return self.status(token)
            self._armed, self._status = True, "armed"
            self._deadline = self._clock() + self.config.watchdog_s
            return self.status(token)

    def stop(self, token: str) -> dict:
        with self._lock:
            self._authorize(token)
            self._stop("operator_stop")
            return self.status(token)

    def poll_watchdog(self) -> None:
        with self._lock:
            if self._armed and self._clock() >= self._deadline:
                self._stop("watchdog_expired")

    def observe(self, token: str) -> RobotObservation:
        with self._lock:
            self._authorize(token)
            self.poll_watchdog()
            captured = self._clock()
            session_id, run_id, control = self._session_id, self._run_id, self._control
        # Camera I/O must not hold the actuator lock or starve local stopping.
        try:
            width, height, rgb = self.driver.capture_rgb()
            if not (0 < width <= 2048 and 0 < height <= 2048):
                raise ValueError("camera dimensions exceed service bounds")
        except Exception:
            with self._lock:
                self._stop("camera_fault")
            raise
        with self._lock:
            self._authorize(token)
            self.poll_watchdog()
            if session_id != self._session_id:
                raise RuntimeError("session changed during capture")
            self._frame += 1
            try:
                observation = RobotObservation(
                    run_id, session_id, f"{session_id}:frame-{self._frame}",
                    captured, time.time(), captured - self._started, None, None,
                    width, height, rgb, control, self._status, "robot_monotonic_seconds")
            except Exception:
                self._stop("camera_fault")
                raise
            self._observations[observation.observation_id] = captured
            if len(self._observations) > 256:
                self._observations.popitem(last=False)
            return observation

    def command(self, token: str, command: RobotCommand) -> CommandReceipt:
        with self._lock:
            self._authorize(token)
            self.poll_watchdog()
            now = self._clock()
            reason = "executing"
            same_session = command.session_id == self._session_id and command.run_id == self._run_id
            if not same_session:
                reason = "wrong_session"
            elif command.sequence_id in self._receipts:
                original, receipt = self._receipts[command.sequence_id]
                if original == command.to_dict():
                    return replace(receipt, duplicate=True)
                reason = "sequence_conflict"
            elif command.sequence_id <= self._highwater:
                reason = "stale_sequence"
            elif not self._armed:
                reason = "disarmed"
            elif command.observation_id not in self._observations:
                reason = "unknown_observation"
            elif command.expires_at_s <= now or command.issued_at_s > now:
                reason = "expired_or_future"
            elif command.duration_s > self.config.max_command_duration_s:
                reason = "duration_exceeded"
            elif command.expires_at_s - now > self.config.max_expiry_ahead_s:
                reason = "expiry_exceeded"
            try:
                self.config.calibration.validate(command.controls)
            except ValueError:
                reason = "bounds"
            accepted = reason == "executing"
            if accepted:
                self._deadline = min(now + command.duration_s, command.expires_at_s,
                                     now + self.config.watchdog_s)
                try:
                    self.driver.apply(command.controls)
                    self._control, self._status = command.controls, "executing"
                except Exception:
                    self._faulted = True
                    self._stop("driver_fault")
                    reason, accepted = "driver_fault", False
            elif same_session and reason in ("bounds", "duration_exceeded", "expiry_exceeded"):
                self._stop("invalid_command")
            captured = self._observations.get(command.observation_id)
            receipt = CommandReceipt(self._run_id, self._session_id, command.sequence_id,
                                     command.observation_id, now, self._control, accepted, reason,
                                     self._deadline, None if captured is None else now - captured)
            if same_session and command.sequence_id > self._highwater:
                self._highwater = command.sequence_id
                self._receipts[command.sequence_id] = (command.to_dict(), receipt)
                if len(self._receipts) > 128:
                    self._receipts.popitem(last=False)
            return receipt

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                try:
                    self._stop("closed")
                finally:
                    self._closed = True


class RobotHTTPServer(HTTPServer):
    """Single bounded control stream plus an independent local watchdog thread."""

    def __init__(self, address: tuple[str, int], service: RobotService):
        if address[0] not in ("127.0.0.1", "localhost"):
            raise ValueError("service binds loopback only; use an authenticated SSH tunnel")
        self.service = service
        self._watchdog_done = threading.Event()
        super().__init__(address, _Handler)
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_done.wait(self.service.config.watchdog_poll_s):
            try:
                self.service.poll_watchdog()
            except Exception:
                # A driver stop failure remains latched; never resume motion.
                pass

    def server_close(self) -> None:
        self._watchdog_done.set()
        self._watchdog.join(timeout=2.)
        try:
            self.service.close()
        finally:
            super().server_close()


class _Handler(BaseHTTPRequestHandler):
    server_version = "PiCarFake/1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.service.config.request_timeout_s)

    def log_message(self, *args) -> None:
        pass  # Do not log bearer tokens, RGB, or commands to an unbounded log.

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _dispatch(self, *, post: bool) -> None:
        service = self.server.service
        auth = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        if not service.authenticate(token):
            self._reply(401, {"error": "authentication required"})
            return
        try:
            payload = {}
            if post:
                if self.headers.get("Transfer-Encoding"):
                    self._reply(400, {"error": "chunked requests are unsupported"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= service.config.max_payload_bytes:
                    self._reply(413, {"error": "payload is empty or too large"})
                    return
                if self.headers.get_content_type() != "application/json":
                    self._reply(415, {"error": "application/json required"})
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
            if not post and self.path == "/health":
                response = service.status(token)
            elif not post and self.path == "/observe":
                observation = service.observe(token)
                response = asdict(observation)
                response["rgb"] = base64.b64encode(observation.rgb).decode("ascii")
            elif post and self.path == "/session":
                response = service.connect(token, **payload)
            elif post and self.path == "/arm":
                response = service.arm(token, **payload)
            elif post and self.path == "/stop":
                if payload:
                    raise ValueError("stop takes an empty object")
                response = service.stop(token)
            elif post and self.path == "/command":
                response = asdict(service.command(token, RobotCommand.from_dict(payload)))
            else:
                self._reply(405 if self.path in ("/arm", "/stop", "/command", "/session") else 404,
                            {"error": "unsupported method or endpoint"})
                return
            self._reply(200, response)
        except PermissionError:
            self._reply(401, {"error": "authentication required"})
        except (ValueError, TypeError, KeyError, OverflowError):
            self._reply(400, {"error": "invalid request"})
        except TimeoutError:
            self._reply(408, {"error": "request timed out"})
        except Exception:
            self._reply(503, {"error": "robot unavailable; inspect local service"})

    def do_GET(self) -> None:
        self._dispatch(post=False)

    def do_POST(self) -> None:
        self._dispatch(post=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Loopback fake-driver PiCar service; no hardware actuation")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fake-driver", action="store_true", required=True)
    args = parser.parse_args()
    token = os.environ.get("PICAR_SERVICE_TOKEN", "")
    service = RobotService(driver=FakeDriver(), token=token)
    with RobotHTTPServer(("127.0.0.1", args.port), service) as server:
        try:
            server.serve_forever(poll_interval=.1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
