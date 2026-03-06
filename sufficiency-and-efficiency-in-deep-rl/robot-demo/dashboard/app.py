#!/usr/bin/env python3
"""Simple web dashboard for controlling a PiCar via car_env.car_client."""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("dashboard")


def _resize_pil_if_requested(image: Image.Image, *, x_resize: int | None, y_resize: int | None) -> Image.Image:
    if x_resize is None or y_resize is None:
        return image
    return image.resize((x_resize, y_resize), resample=Image.BILINEAR)


def _encode_jpeg_from_frame(
    frame: Any, *, x_resize: int | None = None, y_resize: int | None = None
) -> bytes:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required to encode frames returned by car_env.img()") from exc

    if not isinstance(frame, np.ndarray):
        raise TypeError(f"car_env.img() returned unexpected frame type: {type(frame)}")

    if frame.ndim == 3 and frame.shape[2] >= 3:
        # car_env camera frames are BGR; convert to RGB for correct display colors.
        frame = frame[..., [2, 1, 0]]

    image = _resize_pil_if_requested(Image.fromarray(frame), x_resize=x_resize, y_resize=y_resize)
    out = io.BytesIO()
    image.save(out, format="JPEG")
    return out.getvalue()


def _encode_jpeg_from_image_bytes(
    image_bytes: bytes, *, x_resize: int | None = None, y_resize: int | None = None
) -> bytes:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        preview = image_bytes[:80]
        raise RuntimeError(
            f"Failed to decode `/img` response as an image. First bytes: {preview!r}"
        ) from exc

    image = _resize_pil_if_requested(image, x_resize=x_resize, y_resize=y_resize)
    out = io.BytesIO()
    image.save(out, format="JPEG")
    return out.getvalue()


def _parse_json_frame_payload(text: str, *, content_type: str) -> Any | None:
    candidate = (text or "").lstrip()
    if "application/json" not in content_type and not candidate.startswith(("[", "{")):
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    frame_data = None
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        frame_data = payload[0]
    elif isinstance(payload, dict) and "image" in payload:
        frame_data = payload["image"]

    if frame_data is None:
        return None

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required to decode JSON camera payloads") from exc

    return np.array(frame_data, dtype=np.uint8)


@dataclass
class RobotAdapter:
    car_client: Any
    host: str
    x_resize: int | None
    y_resize: int | None
    _prefer_raw_img: bool = False

    @classmethod
    def build(cls) -> "RobotAdapter":
        from car_env import car_client

        host = os.environ.get("CAR_ENV_HOST")
        if not host:
            raise RuntimeError(
                "CAR_ENV_HOST is required. Example: export CAR_ENV_HOST='192.168.1.42:5000'"
            )
        x_resize = _env_int("CAR_ENV_X_RESIZE")
        y_resize = _env_int("CAR_ENV_Y_RESIZE")
        return cls(car_client=car_client, host=host, x_resize=x_resize, y_resize=y_resize)

    def move(self, action: str) -> None:
        if action == "forward":
            self.car_client.drive_forward(self.host)
        elif action == "backward":
            self.car_client.drive_backward(self.host)
        elif action == "left":
            self.car_client.drive_left(self.host)
        elif action == "right":
            self.car_client.drive_right(self.host)
        elif action == "stop":
            if hasattr(self.car_client, "stop"):
                self.car_client.stop(self.host)
            else:
                self._raw_get("/stop")
        else:
            raise ValueError(f"unknown move action: {action}")

    def camera(self, action: str) -> None:
        if action == "up":
            self.car_client.look_up(self.host)
        elif action == "left":
            self.car_client.look_left(self.host)
        elif action == "right":
            self.car_client.look_right(self.host)
        elif action in ("center", "forward"):
            self.car_client.look_forward(self.host)
        else:
            raise ValueError(f"unknown camera action: {action}")

    def frame(self) -> bytes:
        if self._prefer_raw_img:
            return self._frame_from_raw_img()

        try:
            frame, _, _, _ = self.car_client.img(
                self.host, x_resize=self.x_resize, y_resize=self.y_resize
            )
            return _encode_jpeg_from_frame(
                frame,
                x_resize=self.x_resize,
                y_resize=self.y_resize,
            )
        except Exception as exc:
            if not self._prefer_raw_img:
                LOGGER.warning("car_client.img failed, falling back to direct /img request: %s", exc)
            self._prefer_raw_img = True
            return self._frame_from_raw_img()

    def set_resolution(self, x_resize: int | None, y_resize: int | None) -> None:
        self.x_resize = x_resize
        self.y_resize = y_resize

    def get_resolution(self) -> dict[str, int | None]:
        return {
            "x_resize": self.x_resize,
            "y_resize": self.y_resize,
        }

    def _raw_get(self, route: str) -> None:
        url = f"http://{self.host}{route}"
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            raise RuntimeError(f"{route} not available on robot API (status {response.status_code})")

    def _frame_from_raw_img(self) -> bytes:
        params = None
        if self.x_resize is not None and self.y_resize is not None:
            params = {
                "x_resize": self.x_resize,
                "y_resize": self.y_resize,
            }

        last_exc: Exception | None = None
        for _ in range(3):
            try:
                response = requests.get(f"http://{self.host}/img", params=params, timeout=8)
                if response.status_code != 200:
                    raise RuntimeError(f"/img returned status {response.status_code}")

                content_type = response.headers.get("Content-Type", "")
                frame = _parse_json_frame_payload(response.text, content_type=content_type)
                if frame is not None:
                    return _encode_jpeg_from_frame(
                        frame,
                        x_resize=self.x_resize,
                        y_resize=self.y_resize,
                    )
                return _encode_jpeg_from_image_bytes(
                    response.content,
                    x_resize=self.x_resize,
                    y_resize=self.y_resize,
                )
            except Exception as exc:
                last_exc = exc
                time.sleep(0.1)

        raise RuntimeError(f"Failed to retrieve /img after retries: {last_exc}") from last_exc


def _placeholder_jpeg(x_resize: int | None, y_resize: int | None) -> bytes:
    width = x_resize if x_resize is not None else 160
    height = y_resize if y_resize is not None else 120
    image = Image.new("RGB", (width, height), color=(20, 20, 20))
    out = io.BytesIO()
    image.save(out, format="JPEG")
    return out.getvalue()


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _parse_resize_value(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _validate_resolution_pair(x_resize: int | None, y_resize: int | None) -> None:
    if (x_resize is None) != (y_resize is None):
        raise ValueError("x_resize and y_resize must both be set or both be null")


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    adapter = RobotAdapter.build()
    lock = threading.Lock()

    target_hz = float(os.environ.get("DASHBOARD_CAMERA_HZ", "2"))
    if target_hz <= 0 or target_hz > 2:
        target_hz = 2
    min_interval = 1.0 / target_hz

    cache = {
        "jpeg": b"",
        "last_time": 0.0,
        "last_error": "",
    }

    def get_cached_or_refresh() -> bytes:
        now = time.monotonic()
        with lock:
            if cache["jpeg"] and (now - cache["last_time"] < min_interval):
                return cache["jpeg"]

            try:
                jpeg = adapter.frame()
                cache["jpeg"] = jpeg
                cache["last_time"] = now
                cache["last_error"] = ""
                return jpeg
            except Exception as exc:
                cache["last_error"] = str(exc)
                LOGGER.exception("Failed to retrieve frame")
                if cache["jpeg"]:
                    return cache["jpeg"]
                jpeg = _placeholder_jpeg(adapter.x_resize, adapter.y_resize)
                cache["jpeg"] = jpeg
                cache["last_time"] = now
                return jpeg

    @app.get("/")
    def index() -> str:
        resolution = adapter.get_resolution()
        return render_template(
            "index.html",
            camera_hz=target_hz,
            camera_x_resize=resolution["x_resize"],
            camera_y_resize=resolution["y_resize"],
        )

    @app.post("/api/move")
    def api_move() -> Response:
        payload = request.get_json(silent=True) or {}
        action = str(payload.get("action", "")).strip().lower()
        try:
            adapter.move(action)
            return jsonify({"ok": True, "action": action})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "action": action}), 400

    @app.post("/api/camera")
    def api_camera() -> Response:
        payload = request.get_json(silent=True) or {}
        action = str(payload.get("action", "")).strip().lower()
        try:
            adapter.camera(action)
            return jsonify({"ok": True, "action": action})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "action": action}), 400

    @app.get("/api/frame.jpg")
    def api_frame() -> Response:
        try:
            jpeg = get_cached_or_refresh()
            return Response(jpeg, mimetype="image/jpeg")
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/resolution", methods=["GET", "POST"])
    def api_resolution() -> Response:
        if request.method == "GET":
            return jsonify(
                {
                    "ok": True,
                    **adapter.get_resolution(),
                }
            )

        payload = request.get_json(silent=True) or {}
        try:
            x_resize = _parse_resize_value(payload.get("x_resize"), "x_resize")
            y_resize = _parse_resize_value(payload.get("y_resize"), "y_resize")
            _validate_resolution_pair(x_resize, y_resize)
            with lock:
                adapter.set_resolution(x_resize, y_resize)
                cache["last_time"] = 0.0
            return jsonify(
                {
                    "ok": True,
                    "x_resize": x_resize,
                    "y_resize": y_resize,
                }
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/api/health")
    def api_health() -> Response:
        resolution = adapter.get_resolution()
        return jsonify(
            {
                "ok": True,
                "camera_hz": target_hz,
                **resolution,
                "last_frame_age_sec": max(0.0, time.monotonic() - cache["last_time"]),
                "last_error": cache["last_error"],
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    app.run(host=host, port=port, debug=False)
