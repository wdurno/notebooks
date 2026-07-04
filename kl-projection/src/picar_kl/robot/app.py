"""Command-line entry point for the robot-side Flask server."""

from __future__ import annotations

import argparse
import io
import time
from typing import Any

import numpy as np
from PIL import Image

from picar_kl.actions import validate_action_vector
from picar_kl.robot.server import create_app


TURN_ANGLE_CENTER = 90
TURN_ANGLE_DELTA = 45
MOTOR_SPEED = 40
DRIVE_TIME = 0.5
PAN_CENTER = 80
TILT_CENTER = 20
PAN_TILT_DELTA = 30
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120
CAMERA_READ_ATTEMPTS = 3
CAMERA_READ_SLEEP_SECONDS = 0.15


class LegacyPiCarController:
    """Adapter around the copied robot hardware controls."""

    def __init__(self) -> None:
        from picar import back_wheels, front_wheels
        from picar.SunFounder_PCA9685 import Servo
        import picar

        picar.setup()
        self.bw = back_wheels.Back_Wheels()
        self.fw = front_wheels.Front_Wheels()
        self.pan_servo = Servo.Servo(1)
        self.tilt_servo = Servo.Servo(2)

        self.fw.offset = 0
        self.pan_servo.offset = 10
        self.tilt_servo.offset = 0
        self.bw.speed = 0
        self.fw.turn(TURN_ANGLE_CENTER)
        self.pan_servo.write(TURN_ANGLE_CENTER)
        self.tilt_servo.write(TURN_ANGLE_CENTER)

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
    """Adapter around an OpenCV PiCar camera device."""

    def __init__(self, *, camera_index: int | str = "auto") -> None:
        self.camera_index = camera_index
        self._resolved_camera_index: int | None = None
        self._capture = None

    def capture_jpeg(
        self,
        *,
        x_resize: int | None = None,
        y_resize: int | None = None,
        jpeg_quality: int | None = None,
    ) -> bytes:
        capture = self._open_capture()
        frame = self._read_frame(capture)
        if frame is None:
            raise RuntimeError("PiCar camera read failed")
        image = Image.fromarray(_bgr_to_rgb(frame))
        if x_resize is not None and y_resize is not None and x_resize > 0 and y_resize > 0:
            image = image.resize((int(x_resize), int(y_resize)))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(jpeg_quality or 80))
        return buffer.getvalue()

    def _open_capture(self):
        if self._capture is not None and self._capture.isOpened():
            return self._capture
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for PiCar camera capture") from exc

        if self.camera_index == "auto":
            capture, camera_index = _open_first_readable_camera(cv2)
            self._resolved_camera_index = camera_index
            print(f"PiCar camera auto-selected index {camera_index}", flush=True)
        else:
            camera_index = int(self.camera_index)
            capture = _open_camera_index(cv2, camera_index)
            if capture is None:
                raise RuntimeError(f"PiCar camera index {camera_index} did not open")
            self._resolved_camera_index = camera_index
        self._capture = capture
        return capture

    @staticmethod
    def _read_frame(capture: Any) -> Any | None:
        for attempt in range(CAMERA_READ_ATTEMPTS):
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
            if attempt + 1 < CAMERA_READ_ATTEMPTS:
                time.sleep(CAMERA_READ_SLEEP_SECONDS)
        return None


class DryRunController:
    """Controller used for command wiring tests and no-hardware smoke checks."""

    def apply_vector(self, action_vector: dict[str, float]) -> dict[str, Any]:
        return {"status": "ok", "dry_run": True, "vector": validate_action_vector(action_vector)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PiCar robot Flask server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--camera-index", type=_parse_camera_index, default="auto", help="Camera index or auto.")
    parser.add_argument("--no-camera", action="store_true", help="Start hardware controls without exposing /img.")
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
        camera = None if args.no_camera else LegacyPiCarCamera(camera_index=args.camera_index)
    app = create_app(controller=controller, camera=camera)
    app.run(host=args.host, port=int(args.port))
    return 0



def _parse_camera_index(value: Any) -> int | str:
    if str(value).strip().lower() == "auto":
        return "auto"
    try:
        camera_index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("--camera-index must be an integer or auto") from exc
    if camera_index < 0:
        raise ValueError("--camera-index must be non-negative or auto")
    return camera_index


def _open_first_readable_camera(cv2: Any, *, max_index: int = 8) -> tuple[Any, int]:
    failures: list[str] = []
    for camera_index in range(max_index):
        capture = _open_camera_index(cv2, camera_index)
        if capture is None:
            failures.append(f"{camera_index}: open failed")
            continue
        frame = LegacyPiCarCamera._read_frame(capture)
        if frame is not None:
            return capture, camera_index
        failures.append(f"{camera_index}: read failed")
        _release_capture(capture)
    raise RuntimeError("No readable PiCar camera found; tried " + ", ".join(failures))


def _open_camera_index(cv2: Any, camera_index: int) -> Any | None:
    capture = cv2.VideoCapture(int(camera_index))
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not capture.isOpened():
        _release_capture(capture)
        return None
    return capture


def _release_capture(capture: Any) -> None:
    release = getattr(capture, "release", None)
    if callable(release):
        release()
    return None

def _map_linear(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> int:
    return int(round(out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)))


def _bgr_to_rgb(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3][:, :, ::-1].astype(np.uint8)
    return array.astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
