"""HTTP client for the Raspberry Pi PiCar service."""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import requests
from PIL import Image

from picar_kl.actions import (
    ActionVector,
    action_name_to_index,
    validate_action_vector,
)


@dataclass(frozen=True)
class PiCarClientConfig:
    host: str
    timeout_seconds: float = 10.0
    retries: int = 5
    retry_sleep_seconds: float = 0.1
    command_spacing_seconds: float = 0.0
    x_resize: int | None = None
    y_resize: int | None = None
    jpeg_quality: int | None = None


@dataclass(frozen=True)
class PiCarImage:
    image_rgb: np.ndarray
    ball_x: int | None = None
    ball_y: int | None = None
    ball_radius: int | None = None
    headers: dict[str, str] = field(default_factory=dict)


class PiCarClientError(RuntimeError):
    pass


class _Session(Protocol):
    def get(self, url: str, *, params: dict[str, Any] | None = None, timeout: float) -> Any:
        ...


class PiCarClient:
    """Small, retrying client for the PiCar Flask server."""

    def __init__(self, config: PiCarClientConfig, *, session: _Session | None = None):
        self.config = config
        self.session = session or requests.Session()
        self._last_command_at: float | None = None

    @property
    def base_url(self) -> str:
        host = self.config.host
        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/")
        return f"http://{host}".rstrip("/")

    def health(self) -> str:
        response = self._get("/health")
        return str(response.text)

    def capture_image(self) -> PiCarImage:
        params: dict[str, Any] = {}
        if self.config.x_resize is not None:
            params["x_resize"] = self.config.x_resize
        if self.config.y_resize is not None:
            params["y_resize"] = self.config.y_resize
        if self.config.jpeg_quality is not None:
            params["jpeg_quality"] = self.config.jpeg_quality

        response = self._get("/img", params=params or None)
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            image, ball_x, ball_y, ball_radius = json.loads(response.text)
            return PiCarImage(
                image_rgb=np.asarray(image, dtype=np.uint8),
                ball_x=int(ball_x),
                ball_y=int(ball_y),
                ball_radius=int(ball_radius),
                headers=dict(response.headers),
            )

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return PiCarImage(
            image_rgb=np.asarray(image, dtype=np.uint8),
            ball_x=_optional_int_header(response.headers, "X-Ball-X"),
            ball_y=_optional_int_header(response.headers, "X-Ball-Y"),
            ball_radius=_optional_int_header(response.headers, "X-Ball-R"),
            headers=dict(response.headers),
        )

    def perform_action(self, action_name: str) -> str:
        action_name_to_index(action_name)
        self._space_commands()
        response = self._get(f"/{action_name}")
        self._last_command_at = time.time()
        return str(response.text)

    def apply_vector(self, action_vector: dict[str, float]) -> dict[str, Any]:
        vector = validate_action_vector(action_vector)
        self._space_commands()
        response = self._get("/apply_vector", params=vector)
        self._last_command_at = time.time()
        try:
            payload = response.json()
        except AttributeError:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise PiCarClientError("PiCar apply_vector response was not JSON") from exc
        if not isinstance(payload, dict):
            raise PiCarClientError("PiCar apply_vector response must be a JSON object")
        return payload

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        attempts_remaining = max(1, int(self.config.retries))
        while True:
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=float(self.config.timeout_seconds),
                )
                break
            except requests.exceptions.Timeout:
                attempts_remaining -= 1
                if attempts_remaining <= 0:
                    raise PiCarClientError(f"Timed out reading {url}") from None
                time.sleep(float(self.config.retry_sleep_seconds))

        status_code = int(getattr(response, "status_code", 0))
        if status_code != 200:
            body = str(getattr(response, "text", "") or "").strip().replace("\n", " ")
            if len(body) > 300:
                body = body[:300] + "..."
            detail = f": {body}" if body else ""
            raise PiCarClientError(f"PiCar request failed with status {status_code}: {url}{detail}")
        return response

    def _space_commands(self) -> None:
        spacing = float(self.config.command_spacing_seconds)
        if spacing <= 0.0 or self._last_command_at is None:
            return
        elapsed = time.time() - self._last_command_at
        if elapsed < spacing:
            time.sleep(spacing - elapsed)


def _optional_int_header(headers: dict[str, Any], key: str) -> int | None:
    value = headers.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
