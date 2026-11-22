"""Deterministic CPU Bullet dynamics with separate policy and administrative APIs.

    advance() never sleeps: a dedicated production worker owns wall-clock pacing.
    step() is a convenience wrapper, paced unless config.realtime is False.
    Durable snapshots replay this episode's seeded physics events, including
    solver/contact history, rather than serializing process-local Bullet handles.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, replace
import hashlib
from importlib.metadata import version
import json
import math
import time
from typing import Any

from .actions import Calibration
from .curriculum import LAYOUT_VERSION, sample_layout
from .drift import drift_alpha, staged_drift_alpha, steering_gain
from .geometry import contact_terminal_event, in_success_region, target_geometry, terminal_event
from .protocol import (
    CommandReceipt, Control, RewardComponents, RobotCommand, RobotObservation, StepResult,
)

SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class SimulatorConfig:
    physics_dt_s: float = 1 / 240
    control_interval_s: float = .5
    deadline_s: float = 20.
    width: int = 448
    height: int = 336
    horizontal_fov_deg: float = 90.
    total_training_intervals: int = 1000
    realtime: bool = True
    calibration: Calibration = Calibration()
    task_version: str = "safe-stop-v1"
    obstacle_enabled: bool = True
    layout_version: str = "legacy-v1"
    barrier_sigma_easy: float = 1.
    barrier_sigma_hard: float = .08
    barrier_bounds: tuple[float, float] = (-.92, .92)
    curriculum_total_intervals: int = 320
    curriculum_exposure_override: float | None = None
    domain_override: str | None = None
    staged_curriculum: bool = False

    def __post_init__(self) -> None:
        if type(self.task_version) is not str or self.task_version not in ("safe-stop-v1", "ball-contact-v1"):
            raise ValueError("unsupported simulator task_version")
        if type(self.obstacle_enabled) is not bool:
            raise ValueError("obstacle_enabled must be boolean")
        for value in (self.physics_dt_s, self.control_interval_s, self.deadline_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("timing must be finite and positive")
        if self.physics_dt_s > self.control_interval_s:
            raise ValueError("physics timestep exceeds control interval")
        for value in (self.control_interval_s, self.deadline_s):
            if not math.isclose(value / self.physics_dt_s, round(value / self.physics_dt_s),
                                abs_tol=1e-8, rel_tol=0):
                raise ValueError("control interval and deadline must be integral physics ticks")
        if type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) < 8:
            raise ValueError("invalid image dimensions")
        if max(self.width, self.height) > 2048 or not 10 <= self.horizontal_fov_deg <= 150:
            raise ValueError("image/FOV exceeds supported bounds")
        if type(self.total_training_intervals) is not int or self.total_training_intervals <= 0:
            raise ValueError("total_training_intervals must be positive")
        if self.layout_version not in ("legacy-v1", LAYOUT_VERSION):
            raise ValueError("unsupported layout_version")
        if (any(not math.isfinite(s) or s <= 0 for s in (self.barrier_sigma_easy, self.barrier_sigma_hard))
                or self.barrier_sigma_easy < self.barrier_sigma_hard):
            raise ValueError("barrier scales must be positive, finite and broad-to-narrow")
        if (not isinstance(self.barrier_bounds, (tuple, list)) or len(self.barrier_bounds) != 2
                or any(not math.isfinite(x) for x in self.barrier_bounds)
                or not -.95 <= self.barrier_bounds[0] <= -.2 < .2 <= self.barrier_bounds[1] <= .95):
            raise ValueError("barrier bounds must fit the walls and enclose all path centers")
        object.__setattr__(self, "barrier_bounds", tuple(self.barrier_bounds))
        if (type(self.curriculum_total_intervals) is not int or self.curriculum_total_intervals <= 0
                or (self.layout_version == LAYOUT_VERSION and not self.staged_curriculum
                    and .7 * self.curriculum_total_intervals < 200)):
            raise ValueError("curriculum ramp must span at least 200 intervals")
        if self.curriculum_exposure_override is not None and (
                not math.isfinite(self.curriculum_exposure_override) or self.curriculum_exposure_override < 0):
            raise ValueError("curriculum override must be finite and nonnegative")
        if self.domain_override not in (None, "A", "B"):
            raise ValueError("domain_override must be A, B or None")
        if type(self.staged_curriculum) is not bool:
            raise ValueError("staged_curriculum must be boolean")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SimulatorConfig:
        value = dict(value)
        names = {field.name for field in fields(cls)}
        # Only these additive fields may be absent in historical snapshots.
        additive = {"task_version", "obstacle_enabled", "layout_version", "barrier_sigma_easy",
                    "barrier_sigma_hard", "barrier_bounds", "curriculum_total_intervals",
                    "curriculum_exposure_override", "domain_override", "staged_curriculum"}
        if value.keys() - names or (names - additive) - value.keys():
            raise ValueError("unsupported or incomplete simulator configuration")
        try:
            value["calibration"] = Calibration(**value["calibration"])
            return cls(**value)
        except TypeError as exc:
            raise ValueError("invalid simulator configuration") from exc


def vertical_fov_deg(horizontal_deg: float, width: int, height: int) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(horizontal_deg) / 2) * height / width))


class PyBulletBackend:
    """One isolated DIRECT client; no hardware imports or network access."""

    def __init__(self, config: SimulatorConfig | None = None, *, run_id: str = "simulator"):
        try:
            import pybullet as bullet
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("PyBullet and numpy are required for the simulator") from exc
        if not isinstance(run_id, str) or not 0 < len(run_id) <= 100:
            raise ValueError("run_id must be a bounded nonempty string")
        self.config = config or SimulatorConfig()
        self.run_id = run_id
        self._p, self._np = bullet, np
        self.client_id = bullet.connect(bullet.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("could not create Bullet DIRECT client")
        self._ready = False
        self._closed = False
        self._clock_ticks = 0
        self.exposure_intervals = 0.
        self._reset_count = 0
        self._events: list[dict[str, Any]] = []
        self._replaying = False

    def _check(self) -> None:
        if self._closed or not self._ready:
            raise RuntimeError("simulator is closed or not reset")

    @property
    def time_s(self) -> float:
        return self._clock_ticks * self.config.physics_dt_s

    @property
    def episode_time_s(self) -> float:
        return self._ticks * self.config.physics_dt_s

    def reset(self, *, seed: int, exposure_intervals: float | None = None) -> RobotObservation:
        if self._closed:
            raise RuntimeError("simulator is closed")
        if type(seed) is not int or seed < 0:
            raise ValueError("reset seed must be a nonnegative integer")
        if exposure_intervals is not None:
            if not math.isfinite(exposure_intervals) or exposure_intervals < 0:
                raise ValueError("exposure must be finite and nonnegative")
            self.exposure_intervals = float(exposure_intervals)
        self._seed = seed
        self._reset_count += 1
        self.session_id = f"{self.run_id}:reset-{self._reset_count}"
        self._episode_clock_start = self._clock_ticks
        self._initial_exposure = self.exposure_intervals
        self._ticks = 0
        self._held_s = 0.
        self._end_reason: str | None = None
        self._last_result_ticks = 0
        self._terminal_reported = False
        self._obs_counter = 0
        self._observations: OrderedDict[str, float] = OrderedDict()
        self._receipts: OrderedDict[int, tuple[dict[str, Any], CommandReceipt]] = OrderedDict()
        self._highwater = -1
        self._control = self.config.calibration.neutral()
        self._command_deadline_s = self.time_s
        self._safety = "stopped"
        self._fallback_ticks = 0
        self._events = []
        self._build_world(seed)
        self._ready = True
        return self.observe()

    def _box(self, half: tuple[float, float, float], position: tuple[float, float, float],
             color: tuple[float, ...], *, collision: bool = True) -> int:
        p, cid = self._p, self.client_id
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid)
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid) if collision else -1
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                 baseVisualShapeIndex=visual, basePosition=position, physicsClientId=cid)

    def _build_world(self, seed: int) -> None:
        p, cid = self._p, self.client_id
        p.resetSimulation(physicsClientId=cid)
        p.setPhysicsEngineParameter(fixedTimeStep=self.config.physics_dt_s, numSolverIterations=80,
                                   deterministicOverlappingPairs=1, numSubSteps=0, physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        if self.config.layout_version == LAYOUT_VERSION:
            self._layout = sample_layout(seed, self.exposure_intervals, self.config)
        else:
            rng = self._np.random.default_rng(seed)
            mirror = 1 if rng.integers(2) else -1
            self._layout = {
                "start": (float(rng.uniform(.4, .6)), mirror * float(rng.uniform(-.15, .15))),
                "yaw": mirror * float(rng.uniform(-math.radians(10), math.radians(10))),
                "target": (2.9, mirror * float(rng.uniform(-.2, .2))),
                "obstacle": (1.575, mirror * float(rng.uniform(-.15, .15))),
                "mirror": mirror,
            }
            if not self.feasible_layout(self._layout):
                raise RuntimeError("reset violates the fixed geometry feasibility screen")
        self.floor_id = self._box((1.8, 1.2, .025), (1.8, 0, -.025), (.6, .6, .6, 1))
        self._tiles = []
        for ix in range(12):
            for iy in range(8):
                tile = self._box((.15, .15, .0001), (.15 + .3 * ix, -1.05 + .3 * iy, .0002),
                                 (.5, .5, .5, 1), collision=False)
                self._tiles.append((tile, (ix + iy) % 2))
        self._last_alpha = None
        self._update_drift()
        self.wall_ids = [
            self._box((.025, 1.225, .175), (-.025, 0, .175), (.45, .5, .55, 1)),
            self._box((.025, 1.225, .175), (3.625, 0, .175), (.45, .5, .55, 1)),
            self._box((1.8, .025, .175), (1.8, -1.225, .175), (.45, .5, .55, 1)),
            self._box((1.8, .025, .175), (1.8, 1.225, .175), (.45, .5, .55, 1)),
        ]
        ox, oy = self._layout["obstacle"]
        self.obstacle_id = (self._box((.125, .25, .045), (ox, oy, .045), (.1, .25, .75, 1))
                            if self.config.obstacle_enabled else None)
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=.1, physicsClientId=cid)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=.1, rgbaColor=(.85, .08, .08, 1), physicsClientId=cid)
        self.target_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                          baseVisualShapeIndex=visual,
                                          basePosition=(*self._layout["target"], .1), physicsClientId=cid)
        self._create_vehicle()
        self._set_motors(self._control)
        # Settling is reset administration, outside physical task/drift clocks.
        for _ in range(120):
            p.stepSimulation(physicsClientId=cid)

    @staticmethod
    def feasible_layout(layout: dict[str, Any]) -> bool:
        """Fixed conservative screen, not a learned reset filter or proof of control.

        B-gain turning radius is .20/tan(.65*25deg). Require each passage
        wider than that radius plus the vehicle's half-width.
        """
        _, oy = layout["obstacle"]
        radius = .20 / math.tan(.65 * math.radians(25))
        passages = (1.2 - (oy + .25), oy - .25 + 1.2)
        sx, sy = layout["start"]
        tx, ty = layout["target"]
        visible_angle = abs(math.atan2(ty - sy, tx - sx) - layout["yaw"])
        # A single-turn passage needs a radius, not a complete U-turn diameter.
        return min(passages) > radius + .09 and visible_angle < math.radians(45)

    def _create_vehicle(self) -> None:
        p, cid = self._p, self.client_id
        chassis_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=(.16, .067, .025),
                                             collisionFramePosition=(0, 0, .04), physicsClientId=cid)
        chassis_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=(.16, .067, .025),
                                          visualFramePosition=(0, 0, .04), rgbaColor=(.1, .65, .3, 1),
                                          physicsClientId=cid)
        wheel_rot = p.getQuaternionFromEuler((math.pi / 2, 0, 0))
        wheel_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=.035, height=.024,
                                           collisionFrameOrientation=wheel_rot, physicsClientId=cid)
        wheel_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=.035, length=.024,
                                        visualFrameOrientation=wheel_rot, rgbaColor=(.07, .07, .07, 1),
                                        physicsClientId=cid)
        # Parent indices are one-based (0 is chassis). Front wheels ride on
        # genuine revolute steering knuckles; only rear wheels receive torque.
        wheel_links = (0, 1, 3, 5)
        self.vehicle_id = p.createMultiBody(
            baseMass=.8, baseCollisionShapeIndex=chassis_col, baseVisualShapeIndex=chassis_vis,
            basePosition=(*self._layout["start"], .045),
            baseOrientation=p.getQuaternionFromEuler((0, 0, self._layout["yaw"])),
            linkMasses=[.04, .04, .01, .04, .01, .04],
            linkCollisionShapeIndices=[wheel_col, wheel_col, -1, wheel_col, -1, wheel_col],
            linkVisualShapeIndices=[wheel_vis, wheel_vis, -1, wheel_vis, -1, wheel_vis],
            linkPositions=[(-.1, .078, 0), (-.1, -.078, 0), (.1, .078, 0), (0, 0, 0),
                           (.1, -.078, 0), (0, 0, 0)],
            linkOrientations=[(0, 0, 0, 1)] * 6,
            linkInertialFramePositions=[(0, 0, 0)] * 6,
            linkInertialFrameOrientations=[(0, 0, 0, 1)] * 6,
            linkParentIndices=[0, 0, 0, 3, 0, 5],
            linkJointTypes=[p.JOINT_REVOLUTE] * 6,
            linkJointAxis=[(0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 1, 0), (0, 0, 1), (0, 1, 0)],
            physicsClientId=cid,
        )
        for wheel in wheel_links:
            p.changeDynamics(self.vehicle_id, wheel, lateralFriction=1.2, rollingFriction=.001,
                             spinningFriction=.001, restitution=0, physicsClientId=cid)
        p.changeDynamics(self.vehicle_id, -1, linearDamping=.02, angularDamping=.05, physicsClientId=cid)

    def _update_drift(self) -> None:
        if self.config.domain_override is not None:
            self.alpha = {"A": 0., "B": 1.}[self.config.domain_override]
        elif self.config.staged_curriculum:
            self.alpha = staged_drift_alpha(self.exposure_intervals, self.config.curriculum_total_intervals)
        else:
            self.alpha = drift_alpha(self.exposure_intervals, self.config.total_training_intervals)
        if self._last_alpha == self.alpha:
            return
        # Two procedural checker textures interpolate intensities, never geometry.
        for tile, parity in self._tiles:
            a = .72 if parity else .58
            b = .30 if parity else .62
            intensity = (1 - self.alpha) * a + self.alpha * b
            self._p.changeVisualShape(tile, -1, rgbaColor=(intensity, intensity, intensity, 1),
                                      physicsClientId=self.client_id)
        self._last_alpha = self.alpha

    def _set_motors(self, controls: Control) -> None:
        p, cid = self._p, self.client_id
        for wheel in (0, 1):
            p.setJointMotorControl2(self.vehicle_id, wheel, p.VELOCITY_CONTROL,
                                   targetVelocity=controls.drive_mps / .035, force=.15, physicsClientId=cid)
        for wheel in (3, 5):
            p.setJointMotorControl2(self.vehicle_id, wheel, p.VELOCITY_CONTROL,
                                   targetVelocity=0, force=0, physicsClientId=cid)
        for steer in (2, 4):
            p.setJointMotorControl2(self.vehicle_id, steer, p.POSITION_CONTROL,
                                   targetPosition=-controls.steering_rad * steering_gain(self.alpha),
                                   force=.5, maxVelocity=2., physicsClientId=cid)

    def command(self, command: RobotCommand) -> CommandReceipt:
        self._check()
        now = self.time_s
        reason = "executing"
        same_session = command.run_id == self.run_id and command.session_id == self.session_id
        if not same_session:
            reason = "wrong_session"
        elif command.sequence_id in self._receipts:
            original, receipt = self._receipts[command.sequence_id]
            if original == command.to_dict():
                return replace(receipt, duplicate=True)
            reason = "sequence_conflict"
        elif command.sequence_id <= self._highwater:
            reason = "stale_sequence"
        elif self._end_reason:
            reason = "episode_ended"
        elif command.observation_id not in self._observations:
            reason = "unknown_observation"
        elif command.expires_at_s <= now or command.issued_at_s > now + 1e-10:
            reason = "expired_or_future"
        elif command.duration_s > self.config.control_interval_s + 1e-10:
            reason = "duration_exceeded"
        try:
            self.config.calibration.validate(command.controls)
        except ValueError:
            reason = "bounds"
        accepted = reason == "executing"
        if accepted:
            self._control = command.controls
            self._command_deadline_s = min(now + command.duration_s, command.expires_at_s)
            self._safety = "executing"
            if not self._replaying:
                self._events.append({"kind": "command", "controls": asdict(self._control),
                                     "deadline_s": self._command_deadline_s})
        captured = self._observations.get(command.observation_id)
        receipt = CommandReceipt(self.run_id, self.session_id, command.sequence_id,
                                 command.observation_id, now, self._control, accepted, reason,
                                 self._command_deadline_s, None if captured is None else now - captured)
        if same_session and command.sequence_id > self._highwater:
            self._highwater = command.sequence_id
            self._receipts[command.sequence_id] = (command.to_dict(), receipt)
            if len(self._receipts) > 128:
                self._receipts.popitem(last=False)
        return receipt

    def stop(self) -> None:
        self._check()
        self._control = self.config.calibration.neutral(self._control)
        self._command_deadline_s = self.time_s
        self._safety = "stopped"
        self._set_motors(self._control)
        if not self._replaying:
            self._events.append({"kind": "stop"})

    def _contact_types(self) -> tuple[bool, bool]:
        contacts = self._p.getContactPoints(bodyA=self.vehicle_id, physicsClientId=self.client_id)
        bodies = {contact[2] for contact in contacts}
        unsafe_ids = set(self.wall_ids)
        if self.obstacle_id is not None:
            unsafe_ids.add(self.obstacle_id)
        return self.target_id in bodies, bool(bodies & unsafe_ids)

    def _geometry(self, *, contacts: tuple[bool, bool] | None = None):
        p, cid = self._p, self.client_id
        position, orientation = p.getBasePositionAndOrientation(self.vehicle_id, physicsClientId=cid)
        velocity, _ = p.getBaseVelocity(self.vehicle_id, physicsClientId=cid)
        yaw = p.getEulerFromQuaternion(orientation)[2]
        contact = any(self._contact_types() if contacts is None else contacts)
        return target_geometry(position[:2], yaw, math.hypot(*velocity[:2]), self._layout["target"], contact)

    def advance(self, physics_steps: int = 1, *, training: bool = True,
                capture: bool = True) -> StepResult | None:
        self._check()
        if type(physics_steps) is not int or not 0 <= physics_steps <= 1_000_000:
            raise ValueError("physics_steps must be a bounded nonnegative integer")
        started = time.perf_counter()
        before = self._ticks
        for _ in range(physics_steps):
            if self._end_reason:
                break
            if self.time_s >= self._command_deadline_s - 1e-10:
                self._control = self.config.calibration.neutral(self._control)
                self._safety = "fallback_expired"
                self._fallback_ticks += 1
            if training:
                self.exposure_intervals += self.config.physics_dt_s / self.config.control_interval_s
            self._update_drift()
            self._set_motors(self._control)
            self._p.stepSimulation(physicsClientId=self.client_id)
            self._ticks += 1
            self._clock_ticks += 1
            contacts = self._contact_types()
            geometry = self._geometry(contacts=contacts)
            self._held_s = self._held_s + self.config.physics_dt_s if in_success_region(geometry) else 0.
            if self.config.task_version == "ball-contact-v1":
                self._end_reason = contact_terminal_event(*contacts, self.episode_time_s,
                                                          self.config.deadline_s)
            else:
                self._end_reason = terminal_event(geometry, self._held_s, self.episode_time_s,
                                                  self.config.deadline_s)
            if self._end_reason:
                self._control = self.config.calibration.neutral(self._control)
                self._safety = "episode_ended"
                self._set_motors(self._control)
        advanced = self._ticks - before
        if not self._replaying and advanced:
            self._events.append({"kind": "advance", "physics_steps": advanced, "training": training})
        if not capture:
            return None
        elapsed = (self._ticks - self._last_result_ticks) * self.config.physics_dt_s
        self._last_result_ticks = self._ticks
        bonus = {"success": 10., "collision": -5., "deadline": -1.}.get(self._end_reason, 0.)
        # A terminal component is emitted once, not again on later observation.
        if self._terminal_reported:
            bonus = 0.
        self._terminal_reported = self._end_reason is not None
        return StepResult(
            self.observe(), None,
            RewardComponents(elapsed, elapsed / self.config.control_interval_s, bonus,
                             self._end_reason is not None),
            self._end_reason is not None, False, self._end_reason,
            {"physics_compute_s": time.perf_counter() - started,
             "physical_time_s": self.episode_time_s, "simulator_time_s": self.time_s,
             "wall_time_s": time.time(), "exposure_intervals": self.exposure_intervals,
             "acquisition_count": float(math.floor(self.exposure_intervals + 1e-9)),
             "fallback_ticks": float(self._fallback_ticks),
             "accelerated": float(not self.config.realtime),
             **({"alpha": self.alpha, "steering_gain": steering_gain(self.alpha)}
                if self.config.staged_curriculum else {})},
        )

    def step(self, action: Control | RobotCommand, *, training: bool = True) -> StepResult:
        self._check()
        started = time.monotonic()
        if isinstance(action, Control):
            if not self._observations:
                self.observe()
            action = RobotCommand(self.run_id, self.session_id, self._highwater + 1,
                                  next(reversed(self._observations)), action, self.time_s,
                                  self.config.control_interval_s,
                                  self.time_s + self.config.control_interval_s)
        receipt = self.command(action)
        ticks = round(self.config.control_interval_s / self.config.physics_dt_s)
        if self.config.realtime:
            advanced = 0
            while advanced < ticks and not self._end_reason:
                self.advance(1, training=training, capture=False)
                advanced += 1
                # Never accelerate catch-up after a slow tick.
                time.sleep(self.config.physics_dt_s)
            result = self.advance(0, training=training)
        else:
            result = self.advance(ticks, training=training)
        return replace(result, receipt=receipt, timing={**result.timing,
                       "step_wall_s": time.monotonic() - started})

    def render(self):
        """On-robot RGB camera only (numpy HWC uint8)."""
        self._check()
        p = self._p
        position, orientation = p.getBasePositionAndOrientation(self.vehicle_id, physicsClientId=self.client_id)
        rotation = self._np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        eye = self._np.asarray(position) + rotation @ self._np.asarray((.06, 0, .18))
        pan, tilt = self._control.pan_rad, self._control.tilt_rad
        direction = rotation @ self._np.asarray((math.cos(tilt) * math.cos(pan),
                                                math.cos(tilt) * math.sin(pan), math.sin(tilt)))
        up = rotation @ self._np.asarray((-math.sin(tilt) * math.cos(pan),
                                         -math.sin(tilt) * math.sin(pan), math.cos(tilt)))
        return self._camera(eye, eye + direction, up)

    def _camera(self, eye, target, up):
        p, cfg = self._p, self.config
        view = p.computeViewMatrix(eye, target, up)
        projection = p.computeProjectionMatrixFOV(vertical_fov_deg(cfg.horizontal_fov_deg, cfg.width, cfg.height),
                                                  cfg.width / cfg.height, .015, 8.)
        rgba = p.getCameraImage(cfg.width, cfg.height, viewMatrix=view, projectionMatrix=projection,
                                renderer=p.ER_TINY_RENDERER, physicsClientId=self.client_id)[2]
        return self._np.asarray(rgba, dtype=self._np.uint8).reshape(cfg.height, cfg.width, 4)[:, :, :3].copy()

    def render_topdown(self):
        """Privileged reporting camera. Never returned by observe()."""
        self._check()
        return self._camera((1.8, 0, 3.8), (1.8, 0, 0), (0, 1, 0))

    def observe(self) -> RobotObservation:
        self._check()
        rgb = self.render()
        self._obs_counter += 1
        observation_id = f"{self.session_id}:frame-{self._obs_counter}"
        self._observations[observation_id] = self.time_s
        if len(self._observations) > 256:
            self._observations.popitem(last=False)
        return RobotObservation(self.run_id, self.session_id, observation_id, self.time_s, time.time(),
                                self.episode_time_s, self.time_s,
                                max(0., self.config.deadline_s - self.episode_time_s),
                                self.config.width, self.config.height, rgb.tobytes(), self._control,
                                self._safety, "simulator_seconds")

    def diagnostics(self) -> dict[str, Any]:
        self._check()
        position, orientation = self._p.getBasePositionAndOrientation(self.vehicle_id, physicsClientId=self.client_id)
        target_contact, unsafe_contact = self._contact_types()
        return {"privileged": True, "position_m": tuple(position), "orientation_xyzw": tuple(orientation),
                "layout": self._layout,
                "geometry": asdict(self._geometry(contacts=(target_contact, unsafe_contact))),
                "task_version": self.config.task_version, "obstacle_enabled": self.config.obstacle_enabled,
                "target_contact": target_contact, "unsafe_contact": unsafe_contact,
                "alpha": self.alpha, "steering_gain": steering_gain(self.alpha),
                "exposure_intervals": self.exposure_intervals, "held_success_s": self._held_s,
                "end_reason": self._end_reason, "physics_ticks": self._ticks,
                "simulator_time_s": self.time_s, "physical_time_s": self.episode_time_s,
                "fallback_ticks": self._fallback_ticks,
                "parameters_physically_validated": self.config.calibration.physically_validated}

    def snapshot(self) -> dict[str, Any]:
        """JSON-compatible durable reconstruction, O(episode ticks) to restore.

        The owner persists this payload transactionally alongside learner state.
        No pickle, temporary files, in-memory Bullet IDs or external assets.
        """
        self._check()
        payload = {
            "schema_version": SNAPSHOT_VERSION, "bullet_api_version": self._p.getAPIVersion(),
            "bullet_package_version": version("pybullet"),
            "config": asdict(self.config), "run_id": self.run_id, "seed": self._seed,
            "reset_count": self._reset_count, "clock_start_ticks": self._episode_clock_start,
            "initial_exposure": self._initial_exposure, "events": self._events,
            "obs_counter": self._obs_counter, "observations": list(self._observations.items()),
            "last_result_ticks": self._last_result_ticks, "terminal_reported": self._terminal_reported,
            "highwater": self._highwater,
            "receipts": [[key, value, asdict(receipt)] for key, (value, receipt) in self._receipts.items()],
        }
        # Detach from mutable live event lists before returning ownership.
        payload = json.loads(json.dumps(payload, allow_nan=False))
        payload["sha256"] = self._snapshot_hash(payload)
        return payload

    @staticmethod
    def _snapshot_hash(value: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()

    def restore(self, snapshot: dict[str, Any], *, capture: bool = False) -> RobotObservation | None:
        """Restore authoritative state without inventing a capture by default.

        Runners restore their saved observation alongside this snapshot. An
        explicit capture=True creates a genuinely new observation after restore.
        """
        if self._closed:
            raise RuntimeError("simulator is closed")
        value = dict(snapshot)
        checksum = value.pop("sha256", None)
        if checksum != self._snapshot_hash(value):
            raise ValueError("snapshot checksum mismatch")
        if (value["schema_version"] != SNAPSHOT_VERSION
                or value["bullet_api_version"] != self._p.getAPIVersion()
                or value["bullet_package_version"] != version("pybullet")):
            raise ValueError("incompatible simulator snapshot schema or Bullet version")
        if SimulatorConfig.from_dict(value["config"]) != self.config or value["run_id"] != self.run_id:
            raise ValueError("snapshot configuration/run mismatch")
        self._clock_ticks = value["clock_start_ticks"]
        self._reset_count = value["reset_count"] - 1
        self.reset(seed=value["seed"], exposure_intervals=value["initial_exposure"])
        self._replaying = True
        try:
            for event in value["events"]:
                if event["kind"] == "command":
                    self._control = Control(**event["controls"])
                    self._command_deadline_s = event["deadline_s"]
                    self._safety = "executing"
                elif event["kind"] == "advance":
                    self.advance(event["physics_steps"], training=event["training"], capture=False)
                elif event["kind"] == "stop":
                    self.stop()
                else:
                    raise ValueError("unsupported snapshot event")
        finally:
            self._replaying = False
        self._events = json.loads(json.dumps(value["events"]))
        self._highwater = value["highwater"]
        self._receipts = OrderedDict((key, (command, CommandReceipt.from_dict(receipt)))
                                     for key, command, receipt in value["receipts"])
        self._obs_counter = value["obs_counter"]
        self._observations = OrderedDict(value["observations"])
        self._last_result_ticks = value["last_result_ticks"]
        self._terminal_reported = value["terminal_reported"]
        return self.observe() if capture else None

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> PyBulletBackend:
        backend = cls(SimulatorConfig.from_dict(snapshot["config"]), run_id=snapshot["run_id"])
        try:
            backend.restore(snapshot)
        except BaseException:
            backend.close()
            raise
        return backend

    def close(self) -> None:
        if not self._closed:
            self._p.disconnect(physicsClientId=self.client_id)
            self._closed = True

    def __enter__(self) -> PyBulletBackend:
        return self

    def __exit__(self, *args) -> None:
        self.close()
