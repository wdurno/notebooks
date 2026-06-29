"""Dependency-injected Flask app factory for the robot-side PiCar service."""

from __future__ import annotations

import json
from typing import Any, Protocol

from picar_kl.actions import ACTION_NAMES, action_vector_for_name, validate_action_vector


class RobotController(Protocol):
    def apply_vector(self, action_vector: dict[str, float]) -> dict[str, Any]:
        ...


class RobotCamera(Protocol):
    def capture_jpeg(
        self,
        *,
        x_resize: int | None = None,
        y_resize: int | None = None,
        jpeg_quality: int | None = None,
    ) -> bytes:
        ...


def create_app(*, controller: RobotController, camera: RobotCamera | None = None) -> Any:
    """Create the Raspberry Pi HTTP app without importing hardware modules."""

    try:
        from flask import Flask, Response, request
    except ImportError as exc:
        raise RuntimeError("Flask is required to create the PiCar robot server") from exc

    app = Flask(__name__)

    @app.route("/")
    @app.route("/health")
    def health() -> tuple[str, int]:
        return "PiCar-V API is functional", 200

    @app.route("/img")
    def img() -> Any:
        if camera is None:
            return json.dumps({"status": "error", "message": "camera unavailable"}), 503
        jpeg_bytes = camera.capture_jpeg(
            x_resize=_optional_int(request.args.get("x_resize")),
            y_resize=_optional_int(request.args.get("y_resize")),
            jpeg_quality=_optional_int(request.args.get("jpeg_quality")),
        )
        return Response(
            jpeg_bytes,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.route("/apply_vector", methods=["GET", "POST"])
    def apply_vector() -> tuple[str, int]:
        payload = request.get_json(silent=True) or {}
        try:
            vector = validate_action_vector(
                {
                    "pan": _float_param("pan", payload, request),
                    "tilt": _float_param("tilt", payload, request),
                    "turn": _float_param("turn", payload, request),
                    "drive": _float_param("drive", payload, request),
                }
            )
            receipt = controller.apply_vector(vector)
        except ValueError as exc:
            return json.dumps({"status": "error", "message": str(exc)}), 400
        return json.dumps(receipt), 200

    def _route_action(action_name: str) -> tuple[str, int]:
        receipt = controller.apply_vector(action_vector_for_name(action_name))
        if isinstance(receipt, dict):
            return json.dumps(receipt), 200
        return action_name, 200

    for action_name in ACTION_NAMES:
        app.add_url_rule(
            f"/{action_name}",
            endpoint=action_name,
            view_func=lambda action_name=action_name: _route_action(action_name),
        )

    return app


def _float_param(name: str, payload: dict[str, Any], request: Any) -> float:
    value = request.args.get(name, payload.get(name, 0.0))
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be a float") from exc


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
