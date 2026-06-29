import io

import numpy as np
import pytest
import requests
from PIL import Image

from picar_kl.robot.client import PiCarClient, PiCarClientConfig, PiCarClientError


class FakeResponse:
    def __init__(self, *, text="", content=b"", status_code=200, headers=None, json_payload=None):
        self.text = text
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self._json_payload = json_payload

    def json(self):
        if self._json_payload is None:
            raise ValueError("No JSON payload")
        return self._json_payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params=None, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_apply_vector_posts_validated_vector_params():
    session = FakeSession(
        [
            FakeResponse(
                text='{"status": "ok"}',
                headers={"Content-Type": "application/json"},
                json_payload={"status": "ok"},
            )
        ]
    )
    client = PiCarClient(PiCarClientConfig(host="10.0.0.1:5000"), session=session)

    receipt = client.apply_vector({"pan": 0.0, "tilt": 1.0, "turn": -0.5, "drive": 0.25})

    assert receipt == {"status": "ok"}
    assert session.calls[0]["url"] == "http://10.0.0.1:5000/apply_vector"
    assert session.calls[0]["params"] == {"pan": 0.0, "tilt": 1.0, "turn": -0.5, "drive": 0.25}


def test_perform_action_uses_action_route():
    session = FakeSession([FakeResponse(text="look-up")])
    client = PiCarClient(PiCarClientConfig(host="http://robot.local"), session=session)

    assert client.perform_action("look-up") == "look-up"
    assert session.calls[0]["url"] == "http://robot.local/look-up"


def test_request_retries_timeouts():
    session = FakeSession(
        [
            requests.exceptions.Timeout("slow"),
            FakeResponse(text="PiCar-V API is functional"),
        ]
    )
    client = PiCarClient(
        PiCarClientConfig(host="robot.local", retries=2, retry_sleep_seconds=0.0),
        session=session,
    )

    assert client.health() == "PiCar-V API is functional"
    assert len(session.calls) == 2


def test_request_raises_after_timeout_budget():
    session = FakeSession([requests.exceptions.Timeout("slow")])
    client = PiCarClient(
        PiCarClientConfig(host="robot.local", retries=1, retry_sleep_seconds=0.0),
        session=session,
    )

    with pytest.raises(PiCarClientError, match="Timed out"):
        client.health()


def test_capture_image_decodes_jpeg():
    image = Image.fromarray(np.full((2, 2, 3), 32, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    session = FakeSession(
        [
            FakeResponse(
                content=buffer.getvalue(),
                headers={"Content-Type": "image/jpeg", "X-Ball-X": "7"},
            )
        ]
    )
    client = PiCarClient(PiCarClientConfig(host="robot.local"), session=session)

    result = client.capture_image()

    assert result.image_rgb.shape == (2, 2, 3)
    assert result.ball_x == 7
