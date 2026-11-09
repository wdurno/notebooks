"""Bounded CPU Bullet smoke tests, no model download or hardware access."""

from dataclasses import asdict, replace
import json
import math
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pybullet")

from src.robotics import Control, RobotCommand
from src.robotics.simulator import PyBulletBackend, SimulatorConfig, vertical_fov_deg


@pytest.fixture
def sim():
    with PyBulletBackend(SimulatorConfig(realtime=False, width=80, height=60), run_id="test") as backend:
        backend.reset(seed=191)
        yield backend


def test_real_wheel_dynamics_forward_reverse_steering_only_and_braking(sim):
    start = np.array(sim.diagnostics()["position_m"])
    result = sim.step(Control(drive_mps=.2))
    after = np.array(sim.diagnostics()["position_m"])
    assert .07 < np.linalg.norm(after[:2] - start[:2]) < .11
    assert result.reward_components.elapsed_s == .5
    assert result.timing["accelerated"] == 1
    for _ in range(2):
        sim.step(Control())
    stopped = np.array(sim.diagnostics()["position_m"])
    sim.step(Control(steering_rad=.3))
    assert np.linalg.norm(np.array(sim.diagnostics()["position_m"])[:2] - stopped[:2]) < .002
    before = np.array(sim.diagnostics()["position_m"])
    sim.step(Control(drive_mps=-.1))
    delta = np.array(sim.diagnostics()["position_m"]) - before
    assert .03 < np.linalg.norm(delta[:2]) < .06
    assert np.dot(delta[:2], (after - start)[:2]) < 0
    assert sim.diagnostics()["geometry"]["speed_mps"] <= .11


def test_seeded_pairing_camera_pan_tilt_topdown_and_privilege_boundary(sim):
    first_layout = sim.diagnostics()["layout"]
    first_rgb = sim.render()
    reset = sim.reset(seed=191)
    assert sim.diagnostics()["layout"] == first_layout
    np.testing.assert_array_equal(first_rgb, sim.render())
    assert first_rgb.shape == (60, 80, 3) and first_rgb.std() > 10
    assert sim.render_topdown().shape == first_rgb.shape
    assert np.mean(np.abs(sim.render_topdown().astype(float) - first_rgb)) > 10
    sim.step(Control(pan_rad=.4, tilt_rad=.1))
    assert np.mean(np.abs(sim.render().astype(float) - first_rgb)) > 10
    assert vertical_fov_deg(90, 4, 3) == pytest.approx(73.73979529168804)
    public = asdict(reset)
    assert not {"pose", "position_m", "layout", "alpha", "steering_gain", "exposure_intervals"} & public.keys()
    assert len(reset.rgb) == 80 * 60 * 3


def test_drift_changes_within_physics_ticks_and_not_evaluation_reset(sim):
    sim.reset(seed=1, exposure_intervals=200.)
    before = sim.diagnostics()
    sim.advance(1)
    after = sim.diagnostics()
    assert 0 < after["alpha"] - before["alpha"] < .001
    assert after["steering_gain"] < before["steering_gain"]
    sim.advance(240, training=False)
    assert sim.diagnostics()["alpha"] == after["alpha"]
    sim.reset(seed=2)
    assert sim.diagnostics()["alpha"] == after["alpha"]
    # Same seed/geometry, different drift: textures visibly differ.
    a = sim.reset(seed=3, exposure_intervals=0.).rgb
    b = sim.reset(seed=3, exposure_intervals=500.).rgb
    assert a != b
    assert sim.diagnostics()["steering_gain"] == pytest.approx(.65)


def test_expiry_dedup_and_worker_capture_elapsed_accounting(sim):
    obs = sim.observe()
    command = RobotCommand("test", obs.session_id, 3, obs.observation_id,
                            Control(drive_mps=.2), obs.capture_time_s, .1, obs.capture_time_s + .1)
    receipt = sim.command(command)
    assert receipt.accepted and sim.command(command).duplicate
    assert not sim.command(replace(command, sequence_id=2)).accepted
    sim.advance(120, capture=False)
    result = sim.advance(0)
    assert result.reward_components.elapsed_s == .5
    assert result.observation.executed_controls.drive_mps == 0
    assert result.timing["fallback_ticks"] == 96
    assert sim.diagnostics()["geometry"]["speed_mps"] < .02


def test_right_positive_steering_and_b_gain_affect_bullet_joint_dynamics(sim):
    changes = []
    for steer, exposure in ((.4, 0.), (-.4, 0.), (.4, 500.)):
        sim.reset(seed=20, exposure_intervals=exposure)
        before = sim._p.getEulerFromQuaternion(sim.diagnostics()["orientation_xyzw"])[2]
        sim.step(Control(steering_rad=steer, drive_mps=.2), training=False)
        after = sim._p.getEulerFromQuaternion(sim.diagnostics()["orientation_xyzw"])[2]
        changes.append(after - before)
    assert changes[0] < -.05 and changes[1] > .05
    assert abs(changes[2]) < abs(changes[0]) * .85


def _place(sim, xy, yaw=0.):
    p = sim._p
    p.resetBasePositionAndOrientation(sim.vehicle_id, (*xy, .035), p.getQuaternionFromEuler((0, 0, yaw)),
                                      physicsClientId=sim.client_id)
    p.resetBaseVelocity(sim.vehicle_id, (0, 0, 0), (0, 0, 0), physicsClientId=sim.client_id)


def test_actual_contacts_and_one_second_success_at_last_deadline_tick():
    cfg = SimulatorConfig(realtime=False, width=32, height=24, deadline_s=1.)
    with PyBulletBackend(cfg) as sim:
        sim.reset(seed=10)
        tx, ty = sim.diagnostics()["layout"]["target"]
        _place(sim, (tx - .61, ty))
        sim.advance(239, capture=False)
        assert sim.diagnostics()["end_reason"] is None
        result = sim.advance(1)
        assert result.end_reason == "success" and result.terminated and not result.truncated
        assert result.reward_components.terminal_bonus == 10
        assert result.reward_components.elapsed_s == 1
        assert sim.advance(1).reward_components.terminal_bonus == 0
        sim.reset(seed=10)
        _place(sim, sim.diagnostics()["layout"]["obstacle"])
        result = sim.advance(1)
        assert result.end_reason == "collision"
        assert result.reward_components.terminal_bonus == -5
        sim.reset(seed=10)
        result = sim.advance(240)
        assert result.end_reason == "deadline" and result.terminated and not result.truncated
        assert result.reward_components.terminal_bonus == -1


def test_smooth_reset_geometry_screen_includes_mirrored_layouts(sim):
    mirrors = set()
    for seed in range(8):
        sim.reset(seed=seed)
        layout = sim.diagnostics()["layout"]
        assert PyBulletBackend.feasible_layout(layout)
        mirrors.add(layout["mirror"])
    assert mirrors == {-1, 1}


def test_persistent_fresh_process_snapshot_resume_exact_physics(tmp_path, sim):
    sim.reset(seed=19, exposure_intervals=310.)
    sim.step(Control(steering_rad=-.2, drive_mps=.2))
    sim.advance(17, training=False, capture=False)
    payload = sim.snapshot()
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload))
    # Run continuation after dropping the original process-local Bullet IDs.
    code = """
import json, sys
from src.robotics import Control
from src.robotics.simulator import PyBulletBackend
s = PyBulletBackend.from_snapshot(json.load(open(sys.argv[1])))
r = s.step(Control(steering_rad=.1, drive_mps=.15))
print(json.dumps({'diagnostics': s.diagnostics(), 'rgb': r.observation.rgb.hex()}))
s.close()
"""
    continuation = subprocess.run([sys.executable, "-c", code, str(path)],
                                  capture_output=True, text=True, check=True, timeout=20)
    restored = json.loads(continuation.stdout.splitlines()[-1])
    result = sim.step(Control(steering_rad=.1, drive_mps=.15))
    np.testing.assert_array_equal(restored["diagnostics"]["position_m"], sim.diagnostics()["position_m"])
    assert restored["diagnostics"]["exposure_intervals"] == sim.diagnostics()["exposure_intervals"]
    assert bytes.fromhex(restored["rgb"]) == result.observation.rgb
    # Snapshot is detached, checksummed, and rejects incompatible configurations.
    assert payload["events"] != sim.snapshot()["events"]
    damaged = dict(payload, seed=999)
    with pytest.raises(ValueError, match="checksum"):
        sim.restore(damaged)
    with PyBulletBackend(replace(sim.config, width=32), run_id="test") as wrong:
        with pytest.raises(ValueError, match="configuration"):
            wrong.restore(payload)


def test_pending_terminal_and_elapsed_survive_snapshot_before_capture():
    cfg = SimulatorConfig(realtime=False, width=32, height=24, deadline_s=.5)
    with PyBulletBackend(cfg) as sim:
        sim.reset(seed=9)
        sim.advance(120, capture=False)
        with PyBulletBackend.from_snapshot(sim.snapshot()) as restored:
            result = restored.advance(0)
            assert result.reward_components.elapsed_s == .5
            assert result.reward_components.terminal_bonus == -1
            assert result.terminated and result.end_reason == "deadline"


def test_restore_preserves_capture_and_command_identity_exactly(sim):
    observation = sim.observe()
    snapshot = sim.snapshot()
    with PyBulletBackend.from_snapshot(snapshot) as restored:
        assert restored.snapshot() == snapshot
        # Restoring a saved runner observation must not create a phantom frame.
        command = RobotCommand("test", observation.session_id, 0, observation.observation_id,
                                Control(drive_mps=.1), observation.capture_time_s, .5,
                                observation.capture_time_s + .5)
        assert restored.command(command) == sim.command(command)
        left = sim.step(Control())
        right = restored.step(Control())
        assert left.receipt == right.receipt
        assert left.observation.observation_id == right.observation.observation_id
        assert restored.snapshot() == sim.snapshot()
