from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.config import PiCarControlConfig
from env.picar_bridge import PiCarControlClient


def test_capture_image_accepts_json_body_without_json_content_type(monkeypatch):
    client = PiCarControlClient(PiCarControlConfig(host="robot.local:5000"))
    response = SimpleNamespace(
        headers={"Content-Type": "text/plain; charset=utf-8"},
        text="[[[[1, 2, 3], [4, 5, 6]]], 10.0, 20.0, 3.5]",
        content=b"[[[[1, 2, 3], [4, 5, 6]]], 10.0, 20.0, 3.5]",
    )

    monkeypatch.setattr(client, "_requests_get", lambda route, params=None: response)

    image = client.capture_image()

    assert isinstance(image, np.ndarray)
    assert image.shape == (1, 2, 3)
    assert image.dtype == np.uint8
    assert image[0, 1, 2] == 6


def test_requests_get_retries_timeouts(monkeypatch):
    client = PiCarControlClient(
        PiCarControlConfig(
            host="robot.local:5000",
            request_timeout_seconds=1.0,
            request_retry_count=2,
            request_retry_sleep_seconds=0.0,
        )
    )

    class FakeTimeout(Exception):
        pass

    attempts = {"count": 0}

    def fake_get(url, params=None, timeout=None):
        del url, params, timeout
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeTimeout("timed out")
        return SimpleNamespace(status_code=200, text="ok", headers={}, content=b"ok")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_get))
    monkeypatch.setitem(sys.modules, "requests.exceptions", SimpleNamespace(Timeout=FakeTimeout))

    response = client._requests_get("/health")

    assert response.status_code == 200
    assert attempts["count"] == 3


def test_requests_get_raises_clear_error_after_retry_exhaustion(monkeypatch):
    client = PiCarControlClient(
        PiCarControlConfig(
            host="robot.local:5000",
            request_timeout_seconds=1.0,
            request_retry_count=1,
            request_retry_sleep_seconds=0.0,
        )
    )

    class FakeTimeout(Exception):
        pass

    def fake_get(url, params=None, timeout=None):
        del url, params, timeout
        raise FakeTimeout("timed out")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_get))
    monkeypatch.setitem(sys.modules, "requests.exceptions", SimpleNamespace(Timeout=FakeTimeout))

    with pytest.raises(RuntimeError, match=r"/health timed out after 2 attempts"):
        client._requests_get("/health")


def test_requests_get_waits_between_successive_requests(monkeypatch):
    client = PiCarControlClient(
        PiCarControlConfig(
            host="robot.local:5000",
            min_command_interval_seconds=0.5,
            request_retry_count=0,
        )
    )

    monotonic_values = iter([0.0, 0.0, 0.1, 0.6])
    sleep_calls = []

    def fake_monotonic():
        return next(monotonic_values)

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    def fake_get(url, params=None, timeout=None):
        del url, params, timeout
        return SimpleNamespace(status_code=200, text="ok", headers={}, content=b"ok")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_get))
    monkeypatch.setitem(sys.modules, "requests.exceptions", SimpleNamespace(Timeout=TimeoutError))
    monkeypatch.setattr("env.picar_bridge.time.monotonic", fake_monotonic)
    monkeypatch.setattr("env.picar_bridge.time.sleep", fake_sleep)

    client._requests_get("/health")
    client._requests_get("/health")

    assert sleep_calls == [pytest.approx(0.5)]
