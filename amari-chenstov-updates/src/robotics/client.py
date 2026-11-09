"""Explicit-connect loopback client for the authenticated Pi service.

Use an SSH port forward for future deployments. Construction makes no request;
reconnect only creates a fresh DISARMED session. No automatic command retries.
"""

import base64
from dataclasses import asdict
from http.client import HTTPConnection
import json
from urllib.parse import urlsplit

from .protocol import CommandReceipt, Control, RobotCommand, RobotObservation, SCHEMA_VERSION


class RobotClient:
    def __init__(self, *, token: str, url: str = "http://127.0.0.1:8765",
                 timeout_s: float = 2., max_response_bytes: int = 2_000_000):
        parts = urlsplit(url)
        if (parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost")
                or parts.path not in ("", "/") or parts.query or parts.fragment
                or parts.username or parts.password):
            raise ValueError("only loopback HTTP is supported; use an SSH tunnel")
        if len(token.encode()) < 32 or not 0 < timeout_s <= 30:
            raise ValueError("strong token and bounded positive timeout required")
        if type(max_response_bytes) is not int or not 1024 <= max_response_bytes <= 20_000_000:
            raise ValueError("response limit is invalid")
        self._host, self._port = parts.hostname, parts.port or 8765
        self._token, self._timeout_s, self._max_bytes = token, timeout_s, max_response_bytes
        self.run_id = self.session_id = None

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        connection = HTTPConnection(self._host, self._port, timeout=self._timeout_s)
        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        try:
            connection.request(method, path, body=data, headers=headers)
            response = connection.getresponse()
            raw = response.read(self._max_bytes + 1)
            if len(raw) > self._max_bytes:
                raise RuntimeError("robot response exceeds bounded buffer")
            result = json.loads(raw)
            if response.status != 200:
                raise RuntimeError(f"robot service returned HTTP {response.status}")
            if not isinstance(result, dict):
                raise ValueError("robot response must be a JSON object")
            return result
        finally:
            connection.close()

    def connect(self, *, run_id: str) -> dict:
        self.run_id = self.session_id = None
        result = self._request("POST", "/session", {"run_id": run_id})
        if result.get("schema_version") != SCHEMA_VERSION or result.get("armed") is not False:
            raise ValueError("invalid or armed reconnect response")
        self.run_id, self.session_id = result["run_id"], result["session_id"]
        return result

    def arm(self) -> dict:
        if self.session_id is None:
            raise RuntimeError("connect explicitly before arming")
        return self._request("POST", "/arm", {"session_id": self.session_id})

    def observe(self) -> RobotObservation:
        if self.session_id is None:
            raise RuntimeError("connect explicitly before observing")
        value = self._request("GET", "/observe")
        if (value.get("schema_version") != SCHEMA_VERSION or value.get("session_id") != self.session_id
                or value.get("run_id") != self.run_id):
            raise ValueError("observation schema/session mismatch")
        value["rgb"] = base64.b64decode(value["rgb"], validate=True)
        value["executed_controls"] = Control(**value["executed_controls"])
        return RobotObservation(**value)

    def command(self, command: RobotCommand) -> CommandReceipt:
        if command.session_id != self.session_id or command.run_id != self.run_id:
            raise ValueError("command does not belong to the connected session")
        return CommandReceipt.from_dict(self._request("POST", "/command", asdict(command)))

    def stop(self) -> None:
        if self.session_id is not None:
            self._request("POST", "/stop", {})

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.run_id = self.session_id = None
