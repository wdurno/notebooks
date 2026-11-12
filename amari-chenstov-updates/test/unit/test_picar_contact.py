"""Task-version sidecar, contact precedence and bounded CPU Bullet recovery."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from types import SimpleNamespace

import pytest

from src.robotics import Control
from src.robotics.geometry import (
    TerminalGeometry, contact_terminal_event, in_success_region, target_geometry, terminal_event,
)
from src.robotics.simulator import PyBulletBackend, SimulatorConfig


@pytest.mark.parametrize("target,unsafe,expected", [(False, False, None), (True, False, "success"),
                                                    (False, True, "collision"), (True, True, "collision")])
@pytest.mark.parametrize("elapsed", [.1, 20.])
def test_contact_event_precedence_including_last_tick(target, unsafe, expected, elapsed):
    if expected is None and elapsed == 20:
        expected = "deadline"
    assert contact_terminal_event(target, unsafe, elapsed, 20.) == expected


@pytest.mark.parametrize("args", [(1, False, 0., 20.), (False, "false", 0., 20.),
                                  (False, False, math.nan, 20.), (False, False, -.1, 20.),
                                  (False, False, 0., math.inf), (False, False, 0., 0.)])
def test_contact_event_rejects_invalid_flags_and_clocks(args):
    with pytest.raises(ValueError):
        contact_terminal_event(*args)


def test_original_geometry_schema_clearance_hold_and_contact_semantics_are_unchanged():
    geometry = target_geometry((2.29, 0), 0., 0., (2.9, 0), False)
    assert geometry.clearance_m == pytest.approx(.35)
    assert set(asdict(geometry)) == {"clearance_m", "heading_error_rad", "speed_mps", "contact"}
    assert in_success_region(geometry)
    assert terminal_event(geometry, .999, 19.9, 20.) is None
    assert terminal_event(geometry, 1., 20., 20.) == "success"
    assert terminal_event(replace(geometry, contact=True), 1., 20., 20.) == "collision"
    assert terminal_event(geometry, .5, 20., 20.) == "deadline"
    assert TerminalGeometry(**asdict(geometry)) == geometry


def test_configuration_defaults_roundtrip_and_historical_additive_defaults():
    default = SimulatorConfig()
    assert default.task_version == "safe-stop-v1" and default.obstacle_enabled is True
    current = replace(default, task_version="ball-contact-v1", obstacle_enabled=False)
    assert SimulatorConfig.from_dict(json.loads(json.dumps(asdict(current)))) == current
    legacy = asdict(default)
    del legacy["task_version"], legacy["obstacle_enabled"]
    original = deepcopy(legacy)
    assert SimulatorConfig.from_dict(legacy) == default
    assert legacy == original


@pytest.mark.parametrize("changed", [{"task_version": "contact"}, {"task_version": None},
                                     {"obstacle_enabled": 0}, {"obstacle_enabled": "false"}])
def test_configuration_rejects_unknown_task_or_nonboolean_obstacle(changed):
    with pytest.raises(ValueError):
        SimulatorConfig(**changed)


@pytest.mark.parametrize("case", ["unknown_field", "missing_old_field", "calibration_extra"])
def test_serialized_config_does_not_silently_drop_or_default_other_fields(case):
    value = asdict(SimulatorConfig())
    if case == "unknown_field":
        value["future_task_flag"] = True
    elif case == "missing_old_field":
        del value["deadline_s"]
    else:
        value["calibration"]["unexpected"] = True
    with pytest.raises(ValueError):
        SimulatorConfig.from_dict(value)


@pytest.mark.parametrize("body_ids,expected", [([1], (False, False)), ([6], (True, False)),
                                             ([2], (False, True)), ([5], (False, True)),
                                             ([1, 6, 5], (True, True)), ([99], (False, False))])
def test_floor_excluded_chassis_and_wheel_contacts_classified(body_ids, expected):
    backend = object.__new__(PyBulletBackend)
    backend.client_id, backend.vehicle_id = 0, 10
    backend.wall_ids, backend.obstacle_id, backend.target_id = [2, 3, 4], 5, 6
    # Bullet tuple slots 2 and 3 are bodyB and linkA. No vehicle link is excluded.
    def contacts(**kwargs):
        assert kwargs == dict(bodyA=10, physicsClientId=0)
        return [(0, 10, body, link) for body in body_ids for link in (-1, 0, 3, 5)]
    backend._p = SimpleNamespace(getContactPoints=contacts)
    assert backend._contact_types() == expected
    backend.obstacle_id = None
    if body_ids == [5]:
        assert backend._contact_types() == (False, False)


@pytest.fixture
def bullet():
    return pytest.importorskip("pybullet")


def config(**kwargs):
    return SimulatorConfig(realtime=False, width=32, height=24, task_version="ball-contact-v1", **kwargs)


def place(backend, xy):
    p, cid = backend._p, backend.client_id
    p.resetBasePositionAndOrientation(backend.vehicle_id, (*xy, .035),
                                      p.getQuaternionFromEuler((0, 0, 0)), physicsClientId=cid)
    p.resetBaseVelocity(backend.vehicle_id, (0, 0, 0), (0, 0, 0), physicsClientId=cid)


@pytest.mark.parametrize("task", ["safe-stop-v1", "ball-contact-v1"])
@pytest.mark.parametrize("body", ["target", "obstacle", "wall"])
def test_actual_bullet_contact_types_task_outcomes_and_bonus_once(bullet, task, body):
    cfg = replace(config(), task_version=task)
    with PyBulletBackend(cfg) as backend:
        backend.reset(seed=10)
        layout = backend.diagnostics()["layout"]
        tx, ty = layout["target"]
        xy = {"target": (tx - .22, ty), "obstacle": layout["obstacle"], "wall": (.10, 0)}[body]
        place(backend, xy)
        result = backend.advance(1)
        expected = "success" if task == "ball-contact-v1" and body == "target" else "collision"
        assert result.end_reason == expected and result.terminated and not result.truncated
        assert result.reward_components.terminal_bonus == (10. if expected == "success" else -5.)
        assert result.observation.executed_controls.drive_mps == 0
        assert result.reward_components.elapsed_s == pytest.approx(cfg.physics_dt_s)
        diagnostics = backend.diagnostics()
        assert diagnostics["target_contact"] is (body == "target")
        assert diagnostics["unsafe_contact"] is (body != "target")
        assert diagnostics["geometry"]["contact"]
        assert not {"target_contact", "unsafe_contact", "geometry", "layout"} & asdict(result.observation).keys()
        assert not {"target_contact", "unsafe_contact"} & result.timing.keys()
        again = backend.advance(1)
        assert again.reward_components.terminal_bonus == 0
        assert again.reward_components.elapsed_s == 0


def test_actual_bullet_simultaneous_contact_is_failure(bullet):
    with PyBulletBackend(config()) as backend:
        backend.reset(seed=10)
        tx, ty = backend.diagnostics()["layout"]["target"]
        backend._p.resetBasePositionAndOrientation(backend.obstacle_id, (tx - .16, ty, .045),
                                                   (0, 0, 0, 1), physicsClientId=backend.client_id)
        place(backend, (tx - .22, ty))
        result = backend.advance(1)
        assert backend.diagnostics()["target_contact"]
        assert backend.diagnostics()["unsafe_contact"]
        assert result.end_reason == "collision" and result.reward_components.terminal_bonus == -5.


def test_floor_support_and_old_stop_band_do_not_complete_contact_task(bullet):
    with PyBulletBackend(config(deadline_s=1.)) as backend:
        backend.reset(seed=10)
        result = backend.advance(1)
        assert result.end_reason is None
        assert not backend.diagnostics()["target_contact"]
        assert not backend.diagnostics()["unsafe_contact"]
        assert any(c[2] == backend.floor_id for c in bullet.getContactPoints(
            bodyA=backend.vehicle_id, physicsClientId=backend.client_id))
        tx, ty = backend.diagnostics()["layout"]["target"]
        place(backend, (tx - .61, ty))
        result = backend.advance(239)
        assert in_success_region(TerminalGeometry(**backend.diagnostics()["geometry"]))
        assert result.end_reason == "deadline" and result.reward_components.terminal_bonus == -1.


def test_unobstructed_diagnostic_removes_obstacle_without_changing_seeded_layout(bullet):
    with PyBulletBackend(config()) as full, PyBulletBackend(config(obstacle_enabled=False)) as clear:
        full.reset(seed=191)
        observation = clear.reset(seed=191)
        assert full.diagnostics()["layout"] == clear.diagnostics()["layout"]
        assert clear.obstacle_id is None
        assert bullet.getNumBodies(physicsClientId=full.client_id) == bullet.getNumBodies(
            physicsClientId=clear.client_id) + 1
        assert full.render().tobytes() != observation.rgb
        place(clear, clear.diagnostics()["layout"]["obstacle"])
        assert clear.advance(1).end_reason is None
        assert not clear.diagnostics()["unsafe_contact"]


def test_actual_cpu_contact_trajectory_restores_exactly_before_and_after_success(bullet):
    with PyBulletBackend(config(obstacle_enabled=False), run_id="fix2") as original:
        original.reset(seed=10)
        # Privileged fixture controller, not teacher data or a learner input.
        def approach():
            diagnostics = original.diagnostics()
            x, y, _ = diagnostics["position_m"]
            tx, ty = diagnostics["layout"]["target"]
            yaw = bullet.getEulerFromQuaternion(diagnostics["orientation_xyzw"])[2]
            error = math.atan2(ty - y, tx - x) - yaw
            error = math.atan2(math.sin(error), math.cos(error))
            return Control(steering_rad=max(-.35, min(.35, -error)), drive_mps=.2)

        for _ in range(6):
            assert original.step(approach()).end_reason is None
        snapshot = json.loads(json.dumps(original.snapshot()))
        with PyBulletBackend.from_snapshot(snapshot) as restored:
            assert restored.snapshot() == snapshot
            for _ in range(34):
                control = approach()
                left = original.step(control)
                right = restored.step(control)
                assert original.diagnostics() == restored.diagnostics()
                assert left.observation.rgb == right.observation.rgb
                assert left.observation.observation_id == right.observation.observation_id
                assert left.reward_components == right.reward_components
                assert restored.snapshot() == original.snapshot()
                if left.terminated:
                    break
            assert left.end_reason == "success" and right.end_reason == "success"
            assert left.reward_components.terminal_bonus == 10.
            assert left.observation.executed_controls.drive_mps == 0.
            assert left.reward_components.elapsed_s < .5  # Partial final control interval.
        with PyBulletBackend.from_snapshot(original.snapshot()) as after:
            result = after.advance(0)
            assert result.end_reason == "success" and result.reward_components.terminal_bonus == 0.
            assert result.reward_components.elapsed_s == 0.
        assert snapshot["config"]["task_version"] == "ball-contact-v1"
        assert snapshot["config"]["obstacle_enabled"] is False


def test_pending_terminal_bonus_and_partial_elapsed_restore_before_capture(bullet):
    with PyBulletBackend(config(deadline_s=.25)) as original:
        original.reset(seed=9)
        original.advance(60, capture=False)
        payload = original.snapshot()
        with PyBulletBackend.from_snapshot(payload) as restored:
            assert restored.snapshot() == payload
            left, right = original.advance(0), restored.advance(0)
            assert left.reward_components == right.reward_components
            assert right.reward_components.elapsed_s == .25
            assert right.reward_components.terminal_bonus == -1.
            assert restored.advance(0).reward_components.terminal_bonus == 0.


def test_legacy_snapshot_defaults_restore_without_mutating_source(bullet):
    cfg = SimulatorConfig(realtime=False, width=32, height=24)
    with PyBulletBackend(cfg) as original:
        original.reset(seed=19)
        original.step(Control(drive_mps=.2))
        legacy = original.snapshot()
        del legacy["sha256"], legacy["config"]["task_version"], legacy["config"]["obstacle_enabled"]
        legacy["sha256"] = PyBulletBackend._snapshot_hash(legacy)
        frozen = json.dumps(legacy, sort_keys=True)
        with PyBulletBackend.from_snapshot(legacy) as restored:
            assert restored.config == cfg
            assert restored.diagnostics() == original.diagnostics()
            assert restored.step(Control()).observation.rgb == original.step(Control()).observation.rgb
        assert json.dumps(legacy, sort_keys=True) == frozen


@pytest.mark.parametrize("changed", [{"task_version": "safe-stop-v1"}, {"obstacle_enabled": False}])
def test_snapshot_rejects_task_or_obstacle_mismatch_even_with_same_run(bullet, changed):
    with PyBulletBackend(config(), run_id="fix2") as original:
        original.reset(seed=10)
        payload = original.snapshot()
        with PyBulletBackend(replace(original.config, **changed), run_id="fix2") as wrong:
            with pytest.raises(ValueError, match="configuration/run mismatch"):
                wrong.restore(payload)
        damaged = json.loads(json.dumps(payload))
        damaged["config"].update(changed)
        with pytest.raises(ValueError, match="checksum"):
            original.restore(damaged)


def test_worker_forwards_contact_flags_only_through_admin_diagnostics(bullet):
    from src.robotics.worker import SimulatorWorker

    cfg = replace(config(obstacle_enabled=False, deadline_s=.25), realtime=True)
    with SimulatorWorker(cfg, run_id="fix2-diagnostics") as worker:
        worker.reset(seed=10, training=False)
        worker.pause()
        diagnostics = worker.diagnostics()
        assert diagnostics["privileged"] is True
        assert diagnostics["task_version"] == "ball-contact-v1"
        assert diagnostics["obstacle_enabled"] is False
        assert diagnostics["target_contact"] is False
        assert diagnostics["unsafe_contact"] is False
        result = worker.harvest()
        private = {"target_contact", "unsafe_contact", "geometry", "layout"}
        assert not private & asdict(result.observation).keys()
        assert not private & result.timing.keys()
