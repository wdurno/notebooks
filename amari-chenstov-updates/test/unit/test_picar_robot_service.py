from dataclasses import replace
from http.client import HTTPConnection
import json
import threading

import pytest

from demo.picar.service import FakeDriver, RobotHTTPServer, RobotService, ServiceConfig
from src.robotics import Control, RobotCommand
from src.robotics.client import RobotClient

TOKEN = "test-token-for-fake-driver-only-000000000000"


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


@pytest.fixture
def robot():
    clock, driver = Clock(), FakeDriver()
    service = RobotService(driver=driver, token=TOKEN, clock=clock)
    session = service.connect(TOKEN, run_id="test")
    yield service, driver, clock, session["session_id"]
    service.close()


def make_command(service, session, *, sequence=0, control=None):
    observation = service.observe(TOKEN)
    return RobotCommand("test", session, sequence, observation.observation_id,
                        control or Control(drive_mps=.2), observation.capture_time_s,
                        .5, observation.capture_time_s + .5)


def test_authentication_and_explicit_arming(robot):
    service, driver, clock, session = robot
    with pytest.raises(PermissionError):
        service.arm("bad-token", session_id=session)
    command = make_command(service, session)
    assert service.command(TOKEN, command).safety_status == "disarmed"
    service.arm(TOKEN, session_id=session)
    # Rejection consumed the sequence; a queued command cannot spring to life.
    assert service.command(TOKEN, command).duplicate
    assert driver.applied == []
    accepted = service.command(TOKEN, make_command(service, session, sequence=1))
    assert accepted.accepted and len(driver.applied) == 1
    assert accepted.capture_to_command_s == 0


def test_expiry_watchdog_dedup_and_reconnect_stay_disarmed(robot):
    service, driver, clock, session = robot
    service.arm(TOKEN, session_id=session)
    command = make_command(service, session)
    assert service.command(TOKEN, command).accepted
    clock.now += .1
    duplicate = service.command(TOKEN, command)
    assert duplicate.accepted and duplicate.duplicate
    assert len(driver.applied) == 1
    clock.now += .4
    service.poll_watchdog()
    assert driver.controls.drive_mps == 0
    assert not service.status(TOKEN)["armed"]
    assert service.command(TOKEN, command).duplicate
    assert len(driver.applied) == 1
    new = service.connect(TOKEN, run_id="test")
    assert new["session_id"] != session and not new["armed"]
    assert service.command(TOKEN, command).safety_status == "wrong_session"
    assert not service.command(TOKEN, make_command(service, new["session_id"])).accepted


def test_expired_out_of_order_conflicting_and_out_of_bounds_commands(robot):
    service, driver, clock, session = robot
    service.arm(TOKEN, session_id=session)
    command = make_command(service, session, sequence=2)
    expired = replace(command, issued_at_s=99., expires_at_s=99.5)
    assert service.command(TOKEN, expired).safety_status == "expired_or_future"
    assert service.command(TOKEN, replace(command, sequence_id=1)).safety_status == "stale_sequence"
    accepted = replace(command, sequence_id=3)
    assert service.command(TOKEN, accepted).accepted
    conflict = replace(accepted, controls=Control(drive_mps=-.1))
    assert service.command(TOKEN, conflict).safety_status == "sequence_conflict"
    bad = replace(command, sequence_id=4, controls=Control(drive_mps=10.))
    assert service.command(TOKEN, bad).safety_status == "bounds"
    assert driver.controls.drive_mps == 0
    assert not service.status(TOKEN)["armed"]
    assert len(driver.applied) == 1


def test_dedup_cache_eviction_does_not_reenable_old_sequences(robot):
    service, driver, clock, session = robot
    service.arm(TOKEN, session_id=session)
    first = make_command(service, session)
    assert service.command(TOKEN, first).accepted
    for i in range(1, 132):
        assert service.command(TOKEN, replace(first, sequence_id=i)).accepted
    assert service.command(TOKEN, first).safety_status == "stale_sequence"
    assert len(driver.applied) == 132


def test_repeated_arm_does_not_extend_command_deadline(robot):
    service, driver, clock, session = robot
    service.arm(TOKEN, session_id=session)
    assert service.command(TOKEN, make_command(service, session)).accepted
    clock.now += .4
    service.arm(TOKEN, session_id=session)
    clock.now += .11
    service.poll_watchdog()
    assert driver.controls.drive_mps == 0
    assert not service.status(TOKEN)["armed"]


def test_invalid_camera_bytes_fail_closed(robot):
    service, driver, clock, session = robot
    service.arm(TOKEN, session_id=session)
    assert service.command(TOKEN, make_command(service, session)).accepted
    driver.capture_rgb = lambda: (8, 8, b"invalid")
    with pytest.raises(ValueError, match="RGB"):
        service.observe(TOKEN)
    assert driver.controls.drive_mps == 0
    assert not service.status(TOKEN)["armed"]


def test_driver_fault_latches_and_operator_stop_is_independent(robot):
    service, driver, clock, session = robot
    service.arm(TOKEN, session_id=session)
    def broken(controls):
        raise OSError("fake actuator failure")
    driver.apply = broken
    assert service.command(TOKEN, make_command(service, session)).safety_status == "driver_fault"
    with pytest.raises(RuntimeError, match="restart"):
        service.arm(TOKEN, session_id=session)
    assert service.stop(TOKEN)["safety_status"] == "operator_stop"
    assert driver.controls.drive_mps == 0


def test_camera_stall_cannot_hold_watchdog_lock(robot):
    service, driver, clock, session = robot
    service.arm(TOKEN, session_id=session)
    command = make_command(service, session)
    assert service.command(TOKEN, command).accepted
    entered, release = threading.Event(), threading.Event()
    capture = driver.capture_rgb
    def blocked():
        entered.set()
        assert release.wait(2)
        return capture()
    driver.capture_rgb = blocked
    thread = threading.Thread(target=lambda: service.observe(TOKEN))
    thread.start()
    try:
        assert entered.wait(1)
        clock.now += .51
        service.poll_watchdog()
        assert driver.controls.drive_mps == 0
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_loopback_http_transport_auth_methods_payload_and_client_reconnect():
    driver = FakeDriver()
    service = RobotService(driver=driver, token=TOKEN)
    with RobotHTTPServer(("127.0.0.1", 0), service) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        thread.start()
        try:
            host, port = server.server_address
            def request(method, path, body=None, headers=None):
                conn = HTTPConnection(host, port, timeout=2)
                conn.request(method, path, body=body, headers=headers or {})
                response = conn.getresponse()
                status, data = response.status, response.read()
                conn.close()
                return status, data
            assert request("GET", "/health")[0] == 401
            auth = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
            assert request("GET", "/command", headers=auth)[0] == 405
            assert request("POST", "/command", "x" * 9000, auth)[0] == 413
            assert request("POST", "/command", "{not-json", auth)[0] == 400
            client = RobotClient(token=TOKEN, url=f"http://{host}:{port}")
            status = client.connect(run_id="test")
            assert not status["armed"]
            client.arm()
            observation = client.observe()
            command = RobotCommand("test", status["session_id"], 0, observation.observation_id,
                                    Control(drive_mps=.1), observation.capture_time_s, .3,
                                    observation.capture_time_s + .3)
            assert client.command(command).accepted
            assert client.command(command).duplicate
            assert len(driver.applied) == 1
            assert not client.connect(run_id="test")["armed"]
            assert driver.controls.drive_mps == 0
            client.close()
        finally:
            server.shutdown()
            thread.join(timeout=2)
        assert not thread.is_alive()


def test_unsafe_bind_token_and_client_remote_url_are_rejected():
    with pytest.raises(ValueError):
        RobotService(driver=FakeDriver(), token="weak")
    with pytest.raises(ValueError):
        RobotClient(token=TOKEN, url="http://robot.example:8765")
    service = RobotService(driver=FakeDriver(), token=TOKEN)
    try:
        with pytest.raises(ValueError, match="loopback"):
            RobotHTTPServer(("0.0.0.0", 0), service)
    finally:
        service.close()
