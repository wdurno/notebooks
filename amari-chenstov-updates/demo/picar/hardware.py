"""Explicit, optional SunFounder PiCar-V adapter; importing this module does no I/O."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import json
import math
import os
from pathlib import Path
import threading

from src.robotics.actions import Calibration
from src.robotics.protocol import Control
from .service import FakeDriver, RobotHTTPServer, RobotService, ServiceConfig


@dataclass(frozen=True)
class ServoCalibration:
    """Measured electrical degrees = zero_deg + degrees_per_rad * SI angle.

    The signed slope includes linkage direction and ratio. These parameters
    have deliberately no guessed defaults. Vendor offsets must remain zero.
    """

    zero_deg: float
    degrees_per_rad: float
    min_deg: float
    max_deg: float

    def __post_init__(self):
        if (any(isinstance(x, bool) or not math.isfinite(x) for x in
                (self.zero_deg, self.degrees_per_rad, self.min_deg, self.max_deg))
                or self.degrees_per_rad == 0
                or not 0 <= self.min_deg < self.max_deg <= 180
                or math.ceil(self.min_deg) > math.floor(self.max_deg)
                or not self.min_deg <= self.zero_deg <= self.max_deg):
            raise ValueError("invalid measured servo calibration")

    def degrees(self, radians: float) -> int:
        if isinstance(radians, bool) or not math.isfinite(radians):
            raise ValueError("servo angle must be finite")
        raw = self.zero_deg + self.degrees_per_rad * radians
        bounded = min(self.max_deg, max(self.min_deg, raw))
        # Older vendor PWM implementations require whole electrical degrees.
        return min(math.floor(self.max_deg), max(math.ceil(self.min_deg), round(bounded)))


@dataclass(frozen=True)
class HardwareCalibration:
    pan: ServoCalibration
    tilt: ServoCalibration
    steering: ServoCalibration
    forward_pwm_per_mps: float
    reverse_pwm_per_mps: float
    pwm_limit: int
    drive_sign: int
    control_limits: Calibration = field(default_factory=Calibration)

    def __post_init__(self):
        gains = (self.forward_pwm_per_mps, self.reverse_pwm_per_mps)
        if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in gains):
            raise ValueError("speed gains must be measured finite positive PWM/(m/s)")
        if type(self.pwm_limit) is not int or not 1 <= self.pwm_limit <= 100:
            raise ValueError("SunFounder PWM limit must be an integer in [1, 100]")
        if type(self.drive_sign) is not int or self.drive_sign not in (-1, 1):
            raise ValueError("drive_sign must be -1 or +1")
        c = self.control_limits
        for servo, low, high in (
            (self.pan, -c.pan_limit_rad, c.pan_limit_rad),
            (self.tilt, c.tilt_neutral_rad, c.tilt_neutral_rad + c.tilt_up_range_rad),
            (self.steering, -c.steering_limit_rad, c.steering_limit_rad),
        ):
            for angle in (low, high):
                raw = servo.zero_deg + servo.degrees_per_rad * angle
                if not servo.min_deg <= raw <= servo.max_deg:
                    raise ValueError("SI limits exceed measured servo travel; reduce control_limits")
        if (c.forward_limit_mps * self.forward_pwm_per_mps > self.pwm_limit
                or c.reverse_limit_mps * self.reverse_pwm_per_mps > self.pwm_limit):
            raise ValueError("SI speed limits exceed measured PWM limit")

    @classmethod
    def from_dict(cls, value: dict) -> HardwareCalibration:
        value = dict(value)
        for name in ("pan", "tilt", "steering"):
            value[name] = ServoCalibration(**value[name])
        value["control_limits"] = Calibration(**value.get("control_limits", {}))
        return cls(**value)


class PiCarDriver:
    """Injected vendor wheel/servo/camera ports, implementing service.Driver.

    Construction explicitly initializes ports; no module-level hardware import.
    Actuator operations must be bounded by the installed vendor implementation.
    Camera reads use a separate lock and can never hold the actuator lock.
    """

    def __init__(self, calibration: HardwareCalibration, *, rear, front, pan, tilt,
                 camera, width: int = 160, height: int = 120):
        if any(type(x) is not int or not 0 < x <= 2048 for x in (width, height)):
            raise ValueError("camera dimensions must be integers in [1, 2048]")
        self.calibration = calibration
        self.rear, self.front, self.pan, self.tilt, self.camera = rear, front, pan, tilt, camera
        self.width, self.height = width, height
        self._actuator_lock, self._camera_lock = threading.RLock(), threading.Lock()
        self._closed = self._faulted = False
        self._direction = 0
        try:
            rear.stop()
            rear.speed = 0
            # Calibration is absolute. Never inherit vendor/personal offset files.
            for port in (front.wheel, pan, tilt):
                port.offset = 0
            # Vendor turn() otherwise adds an unrelated +/-45 degree clamp.
            front.turning_max = 90
            self.stop()
            pan.write(calibration.pan.degrees(0.))
            tilt.write(calibration.tilt.degrees(calibration.control_limits.tilt_neutral_rad))
            if not camera.isOpened():
                raise OSError("camera failed to open")
            # OpenCV CAP_PROP_FRAME_WIDTH/HEIGHT; verify actual dimensions on read.
            if not camera.set(3, width) or not camera.set(4, height):
                raise OSError("camera rejected configured resolution")
        except Exception:
            self.close()
            raise

    def _check(self):
        if self._closed or self._faulted:
            raise RuntimeError("physical driver closed or faulted; local restart required")

    def _trip(self):
        self._faulted = True
        try:
            self.stop()
        except Exception:
            pass  # Keep the original fault; failed stopping is not proof of safety.

    def apply(self, controls: Control) -> None:
        with self._actuator_lock:
            self._check()
            try:
                c = self.calibration
                c.control_limits.validate(controls)
                self.pan.write(c.pan.degrees(controls.pan_rad))
                self.tilt.write(c.tilt.degrees(controls.tilt_rad))
                self.front.turn(c.steering.degrees(controls.steering_rad))
                speed = controls.drive_mps
                gain = c.forward_pwm_per_mps if speed >= 0 else c.reverse_pwm_per_mps
                pwm = min(c.pwm_limit, round(abs(speed) * gain))
                direction = (1 if speed > 0 else -1) * c.drive_sign if pwm else 0
                if direction != self._direction or not direction:
                    self.rear.stop()
                    self.rear.speed = 0
                if direction:
                    # Change polarity at zero PWM before enabling the new setpoint.
                    if direction != self._direction:
                        (self.rear.forward if direction > 0 else self.rear.backward)()
                    self.rear.speed = pwm
                self._direction = direction
            except Exception:
                self._trip()
                raise

    def stop(self) -> None:
        """Hold camera, independently attempt motor stop, zero PWM and steering."""
        with self._actuator_lock:
            errors = []
            for operation in (self.rear.stop, lambda: setattr(self.rear, "speed", 0),
                              lambda: self.front.turn(self.calibration.steering.degrees(0.))):
                try:
                    operation()
                except Exception as exc:
                    errors.append(exc)
            self._direction = 0
            if errors:
                self._faulted = True
                raise OSError("physical stop failed; use independent emergency stop") from errors[0]

    def capture_rgb(self) -> tuple[int, int, bytes]:
        self._check()
        try:
            with self._camera_lock:
                self._check()
                ok, bgr = self.camera.read()
                if (not ok or bgr is None or getattr(bgr, "shape", None) !=
                        (self.height, self.width, 3) or str(bgr.dtype) != "uint8"):
                    raise OSError("failed camera read or unexpected BGR uint8 resolution")
                # OpenCV supplies BGR; the shared contract is packed row-major RGB.
                rgb = bgr[:, :, ::-1].tobytes(order="C")
                self._check()
                return self.width, self.height, rgb
        except Exception:
            self._trip()
            raise

    def close(self) -> None:
        release = False
        try:
            with self._actuator_lock:
                if self._closed:
                    return
                self._closed = True
                release = True
                self.stop()
        finally:
            if release:
                # Serialize read/release, but never hold up independent motor stops.
                with self._camera_lock:
                    self.camera.release()


def open_physical_driver(calibration: HardwareCalibration, *, camera_device: int = 0,
                         width: int = 160, height: int = 120) -> PiCarDriver:
    """The sole vendor import/setup boundary. Call only in an authorized Pi launch."""
    if not isinstance(calibration, HardwareCalibration):
        raise ValueError("explicit HardwareCalibration required")
    if any(type(x) is not int or not 0 < x <= 2048 for x in (width, height)):
        raise ValueError("camera dimensions must be integers in [1, 2048]")
    if type(camera_device) is not int or camera_device < 0:
        raise ValueError("camera_device must be a nonnegative local camera index")
    if any(not servo.min_deg <= 90 <= servo.max_deg for servo in
           (calibration.pan, calibration.tilt, calibration.steering)):
        raise ValueError("vendor startup pose at 90 degrees exceeds measured safe servo travel")
    picar = importlib.import_module("picar")
    front_module = importlib.import_module("picar.front_wheels")
    rear_module = importlib.import_module("picar.back_wheels")
    servo_module = importlib.import_module("picar.SunFounder_PCA9685.Servo")
    cv2 = importlib.import_module("cv2")
    rear = camera = None
    try:
        picar.setup()
        rear = rear_module.Back_Wheels(db=os.devnull)
        rear.stop()
        rear.speed = 0
        front = front_module.Front_Wheels(db=os.devnull)
        pan, tilt = servo_module.Servo(1), servo_module.Servo(2)
        camera = cv2.VideoCapture(camera_device)
        # PiCarDriver owns cleanup from this point, including initialization faults.
    except Exception:
        try:
            if rear is not None:
                rear.stop()
                rear.speed = 0
        finally:
            if camera is not None:
                camera.release()
        raise
    return PiCarDriver(calibration, rear=rear, front=front, pan=pan, tilt=tilt,
                       camera=camera, width=width, height=height)


class _PhysicalRobotService(RobotService):
    def close(self) -> None:
        try:
            super().close()
        finally:
            self.driver.close()


def create_service(*, token: str, physical: bool = False,
                   calibration: HardwareCalibration | None = None,
                   driver_factory=None, **camera_options) -> RobotService:
    """Fake by default. Physical use requires explicit calibration and opt-in."""
    if not isinstance(token, str) or len(token.encode()) < 32:
        raise ValueError("authentication token must contain at least 32 bytes")
    if not physical:
        if calibration is not None or camera_options or driver_factory is not None:
            raise ValueError("hardware options require physical=True")
        return RobotService(driver=FakeDriver(), token=token)
    if not isinstance(calibration, HardwareCalibration):
        raise ValueError("physical launch requires explicit HardwareCalibration")
    driver = (driver_factory or open_physical_driver)(calibration, **camera_options)
    try:
        return _PhysicalRobotService(driver=driver, token=token,
                                     config=ServiceConfig(calibration=calibration.control_limits))
    except Exception:
        driver.close()
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Loopback PiCar service; fake unless --physical")
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--camera-device", type=int, default=0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.physical != (args.calibration is not None):
        parser.error("--physical and --calibration must be supplied together")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    calibration = None
    if args.calibration is not None:
        with args.calibration.open() as stream:
            calibration = HardwareCalibration.from_dict(json.load(stream))
    options = dict(camera_device=args.camera_device, width=args.width, height=args.height)
    service = create_service(token=os.environ.get("PICAR_SERVICE_TOKEN", ""),
                             physical=args.physical, calibration=calibration,
                             **(options if args.physical else {}))
    try:
        with RobotHTTPServer(("127.0.0.1", args.port), service) as server:
            try:
                server.serve_forever(poll_interval=.1)
            except KeyboardInterrupt:
                pass
    finally:
        service.close()


if __name__ == "__main__":
    main()
