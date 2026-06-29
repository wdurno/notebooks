"""Command-line entry point for the robot-side Flask server."""

from __future__ import annotations

import argparse
import io
import json
from typing import Any

import numpy as np
from PIL import Image

from picar_kl.actions import ActionVector, validate_action_vector
from picar_kl.robot.server import create_app


TURN_ANGLE_CENTER = 90
TURN_ANGLE_DELTA = 45
MOTOR_SPEED = 40
DRIVE_TIME = 0.5
PAN_CENTER = 80
TILT_CENTER = 20
PAN_TILT_DELTA = 30


class LegacyPiCarController:
    """Adapter around the copied robot hardware controls."""

    def __init__(self) -> None:
        from picar_kl.robot.legacy_car_env.car import bw, fw, pan_servo, tilt_servo

        self.bw = bw
        self.fw = fw
        self.pan_servo = pan_servo
        self.tilt_servo = tilt_servo

    def apply_vector(self, action_vector: dict[str, float]) -> dict[str, Any]:
        vector = validate_action_vector(action_vector)
        pan_angle = _map_linear(
            vector["pan"],
            -1.0,
            1.0,
            TURN_ANGLE_CENTER - TURN_ANGLE_DELTA,
            TURN_ANGLE_CENTER + TURN_ANGLE_DELTA,
        )
        tilt_angle = _map_linear(vector["tilt"], 0.0, 1.0, TILT_CENTER, TILT_CENTER + PAN_TILT_DELTA)
        turn_angle = _map_linear(
            vector["turn"],
            -1.0,
            1.0,
            TURN_ANGLE_CENTER - TURN_ANGLE_DELTA,
            TURN_ANGLE_CENTER + TURN_ANGLE_DELTA,
        )
        self._look(pan_angle=pan_angle, tilt_angle=tilt_angle)

        turn_is_non_zero = abs(vector["turn"]) > 1e-6
        drive_is_non_zero = abs(vector["drive"]) > 1e-6
        if turn_is_non_zero:
            self.fw.turn(turn_angle)
            drive_time = DRIVE_TIME * abs(vector["drive"]) if drive_is_non_zero else DRIVE_TIME
            self._drive(drive_time=drive_time, drive_forward=(vector["drive"] >= 0.0))
        elif drive_is_non_zero:
            self.fw.turn(TURN_ANGLE_CENTER)
            drive_time = DRIVE_TIME * abs(vector["drive"])
            self._drive(drive_time=drive_time, drive_forward=(vector["drive"] > 0.0))
        else:
            drive_time = 0.0

        self.fw.turn(TURN_ANGLE_CENTER)
        return {
            "status": "ok",
            "pan_angle": pan_angle,
            "tilt_angle": tilt_angle,
            "turn_angle": turn_angle,
            "drive_time": drive_time,
        }

    def _look(self, *, pan_angle: int, tilt_angle: int) -> None:
        self.pan_servo.write(pan_angle)
        self.tilt_servo.write(tilt_angle)

    def _drive(self, *, drive_time: float, drive_forward: bool) -> None:
        if drive_time <= 0.0:
            return
        import time

        time.sleep(0.1)
        self.bw.speed = MOTOR_SPEED
        if drive_forward:
            self.bw.backward()
        else:
            self.bw.forward()
        time.sleep(drive_time)
        self.bw.stop()


class LegacyPiCarCamera:
    """Adapter around the copied PiCar camera object."""

    def __init__(self) -> None:
        from picar_kl.robot.legacy_car_env.car import img

        self.img = img

    def capture_jpeg(
        self,
        *,
        x_resize: int | None = None,
        y_resize: int | None = None,
        jpeg_quality: int | None = None,
    ) -> bytes:
        ok, frame = self.img.read()
        if not ok:
            raise RuntimeError("PiCar camera read failed")
        image = Image.fromarray(_bgr_to_rgb(frame))
        if x_resize is not None and y_resize is not None and x_resize > 0 and y_resize > 0:
            image = image.resize((int(x_resize), int(y_resize)))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(jpeg_quality or 80))
        return buffer.getvalue()


class DryRunController:
    """Controller used for command wiring tests and no-hardware smoke checks."""

    def apply_vector(self, action_vector: dict[str, float]) -> dict[str, Any]:
        return {"status": "ok", "dry_run": True, "vector": validate_action_vector(action_vector)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PiCar robot Flask server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Start without importing robot hardware modules.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        controller = DryRunController()
        camera = None
    else:
        controller = LegacyPiCarController()
        camera = LegacyPiCarCamera()
    app = create_app(controller=controller, camera=camera)
    app.run(host=args.host, port=int(args.port))
    return 0


def _map_linear(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> int:
    return int(round(out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)))


def _bgr_to_rgb(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3][:, :, ::-1].astype(np.uint8)
    return array.astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
