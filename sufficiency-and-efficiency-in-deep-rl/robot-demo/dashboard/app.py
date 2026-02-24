#!/usr/bin/env python3
"""Simple web dashboard for controlling a PiCar via car_env.car_client."""

from __future__ import annotations

import io
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


def _encode_jpeg_from_frame(frame: Any) -> bytes:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required to encode frames returned by car_env.img()") from exc

    if not isinstance(frame, np.ndarray):
        raise TypeError(f"car_env.img() returned unexpected frame type: {type(frame)}")

    if frame.ndim == 3 and frame.shape[2] >= 3:
        # car_env camera frames are BGR; convert to RGB for correct display colors.
        frame = frame[..., [2, 1, 0]]

    out = io.BytesIO()
    Image.fromarray(frame).save(out, format="JPEG")
    return out.getvalue()


@dataclass
class RobotAdapter:
    car_client: Any
    host: str
    x_resize: int | None
    y_resize: int | None

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
        frame, _, _, _ = self.car_client.img(
            self.host, x_resize=self.x_resize, y_resize=self.y_resize
        )
        return _encode_jpeg_from_frame(frame)

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
                raise

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
