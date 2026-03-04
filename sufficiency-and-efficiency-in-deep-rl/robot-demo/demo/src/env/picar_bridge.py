from __future__ import annotations

import io
import json
import time
from typing import Any

from .config import PiCarControlConfig


class PiCarControlClient:
    """Thin HTTP adapter around the PiCar API."""

    def __init__(self, config: PiCarControlConfig):
        self.config = config
        self._last_request_at: float | None = None

    def reset(self):
        self.look_forward()
        return self.capture_image()

    def capture_image(self):
        response = self._requests_get("/img", params=self._image_params())
        content_type = response.headers.get("Content-Type", "")

        payload = self._try_parse_image_json(response.text, content_type=content_type)
        if payload is not None:
            return payload

        return self._decode_image_bytes(response.content)

    def apply_vector(self, vector: dict[str, float]) -> Any:
        response = self._requests_get(
            "/apply_vector",
            params={
                "pan": float(vector.get("pan", 0.0)),
                "tilt": float(vector.get("tilt", 0.0)),
                "turn": float(vector.get("turn", 0.0)),
                "drive": float(vector.get("drive", 0.0)),
            },
        )
        try:
            return response.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to parse `/apply_vector` response as JSON: {response.text[:200]}") from exc

    def look_forward(self) -> None:
        self._requests_get("/look-forward")
        return None

    def _image_params(self) -> dict[str, int] | None:
        if self.config.x_resize is None or self.config.y_resize is None:
            return None
        return {
            "x_resize": int(self.config.x_resize),
            "y_resize": int(self.config.y_resize),
        }

    def _requests_get(self, route: str, *, params: dict[str, Any] | None = None):
        try:
            import requests
            from requests.exceptions import Timeout
        except ModuleNotFoundError as exc:
            raise RuntimeError("requests is required to communicate with the PiCar API") from exc

        url = f"http://{self.config.host}{route}"
        retries_remaining = max(0, int(self.config.request_retry_count))
        while True:
            try:
                self._respect_request_interval()
                response = requests.get(url, params=params, timeout=float(self.config.request_timeout_seconds))
                self._last_request_at = time.monotonic()
                break
            except Timeout as exc:
                self._last_request_at = time.monotonic()
                if retries_remaining <= 0:
                    raise RuntimeError(
                        f"{route} timed out after {self.config.request_retry_count + 1} attempts"
                    ) from exc
                retries_remaining -= 1
                time.sleep(float(self.config.request_retry_sleep_seconds))
        if response.status_code != 200:
            raise RuntimeError(f"{route} returned status {response.status_code}: {response.text[:200]}")
        return response

    def _respect_request_interval(self) -> None:
        """Keep all PiCar API requests spaced out for Raspberry Pi stability."""

        min_interval = float(self.config.min_command_interval_seconds)
        if min_interval <= 0.0 or self._last_request_at is None:
            return None
        elapsed = time.monotonic() - self._last_request_at
        remaining = min_interval - elapsed
        if remaining > 0.0:
            time.sleep(remaining)
        return None

    def _decode_image_bytes(self, image_bytes: bytes):
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError("Pillow is required to decode PiCar camera frames") from exc

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            preview = image_bytes[:80]
            raise RuntimeError(
                f"Failed to decode `/img` response as an RGB image. First bytes: {preview!r}"
            ) from exc
        return self._numpy().array(image, dtype=self._numpy().uint8)

    def _try_parse_image_json(self, text: str, *, content_type: str) -> Any | None:
        candidate = (text or "").lstrip()
        if "application/json" not in content_type and not candidate.startswith(("[", "{")):
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, list) and len(payload) >= 1 and isinstance(payload[0], list):
            return self._numpy().array(payload[0], dtype=self._numpy().uint8)
        if isinstance(payload, dict) and "image" in payload:
            return self._numpy().array(payload["image"], dtype=self._numpy().uint8)
        return None

    def _numpy(self):
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("numpy is required to work with PiCar camera frames") from exc
        return np
