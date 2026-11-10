from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from demo.picar.hardware import (
    HardwareCalibration, PiCarDriver, ServoCalibration, create_service, open_physical_driver,
)
from demo.picar.service import FakeDriver, RobotService, ServiceConfig
from src.robotics import Control, RobotCommand

TOKEN = "hardware-sidecar-injected-test-token-000000000000"


class Rear:
    def __init__(self):
        self.trace = []
        self._speed = 0
        self.stopped = True
        self.fail_stop = False

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self.trace.append(("speed", value))
        self._speed = value

    def stop(self):
        self.trace.append(("stop",))
        self.stopped = True
        if self.fail_stop:
            raise OSError("injected motor stop failure")

    def forward(self):
        self.trace.append(("forward",))
        self.stopped = False

    def backward(self):
        self.trace.append(("backward",))
        self.stopped = False


class Servo:
    def __init__(self):
        self.writes = []
        self.offset = 17
        self.fail = False

    def write(self, value):
        assert type(value) is int
        if self.fail:
            raise OSError("injected servo fault")
        self.writes.append(value)

    turn = write


class Front:
    def __init__(self):
        self.wheel = Servo()
        self.turning_max = 45
        self.writes = []

    def turn(self, value):
        value = min(90 + self.turning_max, max(90 - self.turning_max, value))
        self.writes.append(value)
        self.wheel.write(value)


class Camera:
    def __init__(self):
        self.frame = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
        self.ok = self.opened = self.accept_size = True
        self.releases = 0
        self.settings = []

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.settings.append((key, value))
        return self.accept_size

    def read(self):
        return self.ok, self.frame

    def release(self):
        self.releases += 1


def calibration():
    # Deliberately synthetic test calibration, NOT measured Pi settings.
    return HardwareCalibration(
        pan=ServoCalibration(90., -180 / math.pi, 0., 180.),
        tilt=ServoCalibration(90., 180 / math.pi, 0., 180.),
        steering=ServoCalibration(90., -180 / math.pi, 0., 180.),
        forward_pwm_per_mps=300., reverse_pwm_per_mps=400., pwm_limit=80, drive_sign=1,
    )


@pytest.fixture
def driver():
    ports = dict(rear=Rear(), front=Front(), pan=Servo(), tilt=Servo(), camera=Camera())
    driver = PiCarDriver(calibration(), **ports, width=2, height=2)
    yield driver
    driver.close()


def test_import_and_fake_factory_never_import_optional_backends():
    code = """
import builtins, importlib
real_import, real_module = builtins.__import__, importlib.import_module
def check(name):
    assert name.split('.')[0] not in {'picar', 'cv2', 'torch', 'faster_whisper', 'huggingface_hub', 'sounddevice'}, name
def guarded(name, *a, **kw):
    check(name)
    return real_import(name, *a, **kw)
def module(name, *a, **kw):
    check(name)
    return real_module(name, *a, **kw)
builtins.__import__, importlib.import_module = guarded, module
from demo.picar.hardware import create_service
from demo.picar.speech import FasterWhisperTranscriber, SpeechBridge
service = create_service(token='test-token-with-more-than-32-bytes-000000000')
assert type(service.driver).__name__ == 'FakeDriver'
service.close()
bridge = SpeechBridge(FasterWhisperTranscriber())
assert bridge.close()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_calibration_signs_units_and_zero_drive_does_not_enable_motor(driver):
    assert driver.front.wheel.offset == driver.pan.offset == driver.tilt.offset == 0
    assert driver.tilt.writes[-1] == pytest.approx(82.)
    driver.apply(Control(math.radians(20), math.radians(10), math.radians(15), .2))
    assert driver.pan.writes[-1] == pytest.approx(70.)
    assert driver.tilt.writes[-1] == pytest.approx(100.)
    assert driver.front.writes[-1] == pytest.approx(75.)
    assert driver.rear.trace[-2:] == [("forward",), ("speed", 60)]
    before = len(driver.rear.trace)
    driver.apply(Control(steering_rad=.2))
    assert driver.rear.stopped and driver.rear.speed == 0
    assert not any(item[0] in ("forward", "backward") for item in driver.rear.trace[before:])


def test_reverse_stops_before_polarity_change_and_uses_separate_gain(driver):
    driver.apply(Control(drive_mps=.1))
    driver.apply(Control(drive_mps=-.1))
    assert driver.rear.trace[-4:] == [("stop",), ("speed", 0), ("backward",), ("speed", 40)]
    driver.calibration = replace(driver.calibration, drive_sign=-1)
    driver.apply(Control(drive_mps=.1))
    assert driver.rear.trace[-1] == ("speed", 30)
    assert driver._direction == -1


def test_calibration_saturation_and_inconsistent_limits_rejected():
    servo = ServoCalibration(90., -60., 10., 170.)
    assert servo.degrees(100.) == 10.
    assert servo.degrees(-100.) == 170.
    assert ServoCalibration(90., 60., 10.2, 170.8).degrees(100.) == 170
    with pytest.raises(ValueError, match="servo travel"):
        replace(calibration(), pan=ServoCalibration(90., 1000., 10., 170.))
    with pytest.raises(ValueError, match="PWM limit"):
        replace(calibration(), pwm_limit=20)
    for value in (0, True, 2):
        with pytest.raises(ValueError, match="drive_sign"):
            replace(calibration(), drive_sign=value)
    assert HardwareCalibration.from_dict(asdict(calibration())) == calibration()


def test_stop_holds_camera_centers_steering_and_faults_latch(driver):
    driver.apply(Control(pan_rad=.1, steering_rad=.2, drive_mps=.1))
    pan, tilt = len(driver.pan.writes), len(driver.tilt.writes)
    driver.stop()
    assert driver.rear.speed == 0 and driver.front.writes[-1] == 90.
    assert (len(driver.pan.writes), len(driver.tilt.writes)) == (pan, tilt)
    driver.pan.fail = True
    with pytest.raises(OSError, match="servo fault"):
        driver.apply(Control(drive_mps=.1))
    assert driver.rear.stopped and driver.rear.speed == 0
    with pytest.raises(RuntimeError, match="restart"):
        driver.apply(Control())


def test_bounds_fail_closed_and_stop_attempts_all_independent_outputs(driver):
    with pytest.raises(ValueError, match="bounds"):
        driver.apply(Control(drive_mps=1.))
    assert driver.rear.stopped
    driver.rear._speed = 60
    driver.rear.fail_stop = True
    with pytest.raises(OSError, match="stop failed"):
        driver.stop()
    assert driver.rear.speed == 0 and driver.front.writes[-1] == 90.
    driver.rear.fail_stop = False


def test_capture_converts_bgr_without_resizing_and_close_is_idempotent(driver):
    assert driver.camera.settings == [(3, 2), (4, 2)]
    width, height, rgb = driver.capture_rgb()
    assert (width, height, rgb[:6]) == (2, 2, bytes([3, 2, 1, 6, 5, 4]))
    assert len(rgb) == 12
    driver.close()
    driver.close()
    assert driver.camera.releases == 1 and driver.rear.speed == 0
    with pytest.raises(RuntimeError, match="closed"):
        driver.capture_rgb()


@pytest.mark.parametrize("failure", ["failed_read", "wrong_size", "wrong_dtype"])
def test_bad_camera_frames_stop_and_require_restart(driver, failure):
    driver.apply(Control(drive_mps=.1))
    if failure == "failed_read":
        driver.camera.ok = False
    elif failure == "wrong_size":
        driver.camera.frame = driver.camera.frame[:1]
    else:
        driver.camera.frame = driver.camera.frame.astype(np.float32)
    with pytest.raises(OSError, match="camera read"):
        driver.capture_rgb()
    assert driver.rear.stopped and driver.rear.speed == 0
    with pytest.raises(RuntimeError, match="restart"):
        driver.apply(Control())


def test_initialization_failure_releases_camera_and_stops_wheels():
    ports = dict(rear=Rear(), front=Front(), pan=Servo(), tilt=Servo(), camera=Camera())
    ports["camera"].opened = False
    with pytest.raises(OSError, match="failed to open"):
        PiCarDriver(calibration(), **ports, width=2, height=2)
    assert ports["rear"].stopped and ports["camera"].releases == 1


def test_service_reconnect_dedup_watchdog_and_camera_fault(driver):
    now = [100.]
    service = RobotService(driver=driver, token=TOKEN, clock=lambda: now[0],
                           config=ServiceConfig(calibration=driver.calibration.control_limits))
    try:
        session = service.connect(TOKEN, run_id="injected")
        observation = service.observe(TOKEN)
        assert observation.capture_time_s == 100. and observation.width == 2
        service.arm(TOKEN, session_id=session["session_id"])
        command = RobotCommand("injected", session["session_id"], 0, observation.observation_id,
                               Control(drive_mps=.1), 100., .2, 100.2)
        assert service.command(TOKEN, command).accepted
        writes = len(driver.rear.trace)
        assert service.command(TOKEN, command).duplicate
        assert len(driver.rear.trace) == writes
        now[0] += .3
        service.poll_watchdog()
        assert driver.rear.speed == 0 and not service.status(TOKEN)["armed"]
        session2 = service.connect(TOKEN, run_id="injected")
        assert session2["session_id"] != session["session_id"] and not session2["armed"]
        assert service.command(TOKEN, command).safety_status == "wrong_session"
        service.arm(TOKEN, session_id=session2["session_id"])
        driver.camera.ok = False
        with pytest.raises(OSError):
            service.observe(TOKEN)
        assert not service.status(TOKEN)["armed"] and driver.rear.speed == 0
    finally:
        service.close()


def test_stalled_camera_does_not_hold_actuator_lock(driver):
    entered, release = threading.Event(), threading.Event()
    original = driver.camera.read
    errors = []
    def read():
        entered.set()
        assert release.wait(2)
        return original()
    def capture():
        try:
            driver.capture_rgb()
        except Exception as exc:
            errors.append(exc)
    driver.camera.read = read
    thread = threading.Thread(target=capture)
    thread.start()
    try:
        assert entered.wait(1)
        driver.stop()
        assert driver.rear.speed == 0
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive() and not errors


def test_factory_auth_opt_in_and_calibration_precede_hardware_loading(monkeypatch, driver):
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError("real hardware import")
    monkeypatch.setattr("demo.picar.hardware.importlib.import_module", forbidden)
    service = create_service(token=TOKEN)
    assert isinstance(service.driver, FakeDriver)
    service.close()
    with pytest.raises(ValueError, match="token"):
        create_service(token="bad", physical=True, calibration=calibration())
    with pytest.raises(ValueError, match="explicit"):
        create_service(token=TOKEN, physical=True)
    with pytest.raises(ValueError, match="dimensions"):
        open_physical_driver(calibration(), width=0)
    narrow = replace(calibration(), steering=ServoCalibration(50., 180 / math.pi, 20., 80.))
    with pytest.raises(ValueError, match="startup pose"):
        open_physical_driver(narrow)
    def injected(c, **options):
        calls.append((c, options))
        return driver
    service = create_service(token=TOKEN, physical=True, calibration=driver.calibration,
                             driver_factory=injected, width=2, height=2)
    assert calls == [(driver.calibration, {"width": 2, "height": 2})]
    assert service.config.calibration == driver.calibration.control_limits
    service.close()
    assert driver.camera.releases == 1


def test_vendor_factory_is_explicit_and_cleans_partial_initialization(monkeypatch):
    from demo.picar import hardware
    ports = dict(rear=Rear(), front=Front(), pan=Servo(), tilt=Servo(), camera=Camera())
    calls, channels, databases = [], [], []
    def servo(channel):
        channels.append(channel)
        return ports["pan" if channel == 1 else "tilt"]
    def wheels(name, **kwargs):
        databases.append((name, kwargs))
        return ports[name]
    modules = {
        "picar": SimpleNamespace(setup=lambda: calls.append("setup")),
        "picar.front_wheels": SimpleNamespace(Front_Wheels=lambda **kw: wheels("front", **kw)),
        "picar.back_wheels": SimpleNamespace(Back_Wheels=lambda **kw: wheels("rear", **kw)),
        "picar.SunFounder_PCA9685.Servo": SimpleNamespace(Servo=servo),
        "cv2": SimpleNamespace(VideoCapture=lambda index: ports["camera"]),
    }
    monkeypatch.setattr(hardware.importlib, "import_module", lambda name: modules[name])
    driver = open_physical_driver(calibration(), width=2, height=2)
    try:
        assert calls == ["setup"] and channels == [1, 2]
        assert databases == [("rear", {"db": os.devnull}), ("front", {"db": os.devnull})]
        assert driver.rear is ports["rear"]
    finally:
        driver.close()
    assert ports["camera"].releases == 1
    def broken_servo(channel):
        raise OSError("injected vendor initialization failure")
    modules["picar.SunFounder_PCA9685.Servo"] = SimpleNamespace(Servo=broken_servo)
    ports["rear"].speed = 60
    with pytest.raises(OSError, match="initialization failure"):
        open_physical_driver(calibration(), width=2, height=2)
    assert ports["rear"].speed == 0 and ports["rear"].stopped


def test_measured_steering_has_no_hidden_vendor_offset_or_clamp():
    ports = dict(rear=Rear(), front=Front(), pan=Servo(), tilt=Servo(), camera=Camera())
    c = replace(calibration(), steering=ServoCalibration(50., -180 / math.pi, 0., 180.))
    driver = PiCarDriver(c, **ports, width=2, height=2)
    try:
        assert ports["front"].wheel.offset == 0
        driver.apply(Control(steering_rad=math.radians(25.)))
        assert ports["front"].wheel.writes[-1] == 25  # Vendor default clamp would produce 45.
        driver.stop()
        assert ports["front"].wheel.writes[-1] == 50
    finally:
        driver.close()


def test_declared_vendor_distribution_uses_exact_upstream_commit():
    from packaging.requirements import Requirement
    source = Path(__file__).resolve().parents[2] / "demo/picar/requirements-robot.txt"
    vendor, = [Requirement(line) for line in source.read_text().splitlines()
               if line.startswith("SunFounder")]
    assert vendor.name == "SunFounder_PiCar"
    assert re.fullmatch(r"git\+https://github.com/sunfounder/SunFounder_PiCar\.git@[0-9a-f]{40}", vendor.url)


def test_cli_physical_launch_uses_loopback_api_and_releases_camera(monkeypatch, tmp_path, driver):
    from demo.picar import hardware
    path = tmp_path / "measured-calibration.json"
    path.write_text(json.dumps(asdict(driver.calibration)))
    monkeypatch.setenv("PICAR_SERVICE_TOKEN", TOKEN)
    launched = []
    def opened(c, **options):
        launched.append((c, options))
        return driver
    class Server:
        def __init__(self, address, service):
            assert address == ("127.0.0.1", 8765)
            self.service = service
        def __enter__(self):
            return self
        def serve_forever(self, **kwargs):
            assert not self.service.status(TOKEN)["armed"]
            raise KeyboardInterrupt
        def __exit__(self, *args):
            self.service.close()
    monkeypatch.setattr(hardware, "open_physical_driver", opened)
    monkeypatch.setattr(hardware, "RobotHTTPServer", Server)
    hardware.main(["--physical", "--calibration", str(path), "--width", "2", "--height", "2"])
    assert launched == [(driver.calibration, {"camera_device": 0, "width": 2, "height": 2})]
    assert driver.camera.releases == 1 and driver.rear.speed == 0


def test_close_releases_camera_even_when_motor_stop_fails(driver):
    driver.rear.fail_stop = True
    with pytest.raises(OSError, match="stop failed"):
        driver.close()
    assert driver.camera.releases == 1
    driver.rear.fail_stop = False


def test_slow_camera_release_does_not_hold_motor_lock(driver):
    entered, release = threading.Event(), threading.Event()
    original = driver.camera.release
    def slow_release():
        entered.set()
        assert release.wait(2)
        original()
    driver.camera.release = slow_release
    thread = threading.Thread(target=driver.close)
    thread.start()
    try:
        assert entered.wait(1)
        driver.stop()
        assert driver.rear.stopped and driver.rear.speed == 0
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive() and driver.camera.releases == 1
