"""Fix 6 privileged text proposals and simulator-time coaching windows.

No planner control is executed here. Only ``policy_context(as_of)`` is a policy
input; proposals, window audits and checkpoint state contain privileged data.
Call advance at every captured acquisition, using *current* diagnostic time.
Delayed receipts can fill their owning window until its reward is claimed.
Never backdate a cue to the image which triggered a later diagnostic query.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any, Callable, Mapping

from demo.picar.speech import InstructionEvent, InstructionLedger
from src.robotics.drift import smoothstep5
from src.robotics.protocol import CommandReceipt, Control
from src.robotics.simulator import SimulatorConfig


SCHEMA_VERSION = 1
SEED_NAMESPACE = "plan19-fix6/coach-gates"


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def guidance_schedule(acquired: float, total: int = 1024, *, condition: str = "G",
                      initialization: bool = False) -> dict:
    """Acquisition clocks, never replay/update counts or elapsed time."""
    _nonnegative(acquired, "acquired")
    if type(total) is not int or total <= 0 or condition not in ("U", "G"):
        raise ValueError("expected positive total and U/G condition")
    x = acquired / total
    d = 0. if initialization else smoothstep5((x - .125) / .75)
    p = .5 if initialization else .5 * smoothstep5((.75 - x) / .625)
    return {"fraction": x, "difficulty": d, "barrier_sigma_m": .5 ** d,
            "coaching_probability": p if condition == "G" else 0.}


def guidance_probability(acquired: float, total: int = 1024, **kwargs) -> float:
    return guidance_schedule(acquired, total, **kwargs)["coaching_probability"]


@dataclass(frozen=True)
class SimulatorGeometry:
    """Versioned declared geometry of the frozen Bullet simulator.

    Kept here because the historical simulator has no geometry-spec API. This
    is shared by all Fix 6 geometry operations; tests check it against Bullet
    collision geometry rather than assuming the old feasibility screen proves
    arbitrary-state safety. Wheel steering enlarges the nominal .18 m width.
    """

    version: str = "plan19-bullet-geometry-v1"
    arena_bounds_m: tuple = (0., 3.6, -1.2, 1.2)
    chassis_half_length_m: float = .16
    chassis_half_width_m: float = .067
    wheelbase_m: float = .20
    wheel_track_half_m: float = .078
    wheel_radius_m: float = .035
    wheel_half_thickness_m: float = .012
    barrier_half_extents_m: tuple = (.125, .25)

    def half_width(self, steering_limit: float, gain: float) -> float:
        angle = min(abs(steering_limit * gain), math.pi / 2)
        peak = min(angle, math.atan2(self.wheel_radius_m, self.wheel_half_thickness_m))
        return max(self.chassis_half_width_m, self.wheel_track_half_m
                   + self.wheel_radius_m * math.sin(peak)
                   + self.wheel_half_thickness_m * math.cos(peak))


SIMULATOR_GEOMETRY = SimulatorGeometry()


@dataclass(frozen=True)
class CoachConfig:
    clearance_margin_m: float = .035
    dynamics_margin_m: float = .015
    control_latency_s: float = .5
    braking_time_s: float = .4
    sweep_step_s: float = .04
    heading_enter_rad: float = math.radians(12)
    heading_exit_rad: float = math.radians(6)
    waypoint_tolerance_m: float = .16

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.heading_exit_rad >= self.heading_enter_rad or self.sweep_step_s > .05:
            raise ValueError("invalid heading hysteresis or sweep resolution")


def chassis_yaw(diagnostics: Mapping) -> float:
    x, y, z, w = (float(x) for x in diagnostics["orientation_xyzw"])
    if not all(math.isfinite(v) for v in (x, y, z, w)):
        raise ValueError("invalid orientation")
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-8:
        raise ValueError("invalid orientation")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def bicycle_step(pose: tuple, controls: Control, seconds: float, *,
                 steering_gain: float, geometry: SimulatorGeometry = SIMULATOR_GEOMETRY) -> tuple:
    """Diagnostic model only: right-positive steering, reverse flips yaw rate."""
    x, y, yaw = pose
    rate = -controls.drive_mps * math.tan(controls.steering_rad * steering_gain) / geometry.wheelbase_m
    if abs(rate) < 1e-10:
        return (x + controls.drive_mps * seconds * math.cos(yaw),
                y + controls.drive_mps * seconds * math.sin(yaw), yaw)
    end = yaw + rate * seconds
    radius = controls.drive_mps / rate
    return x + radius * (math.sin(end) - math.sin(yaw)), y - radius * (math.cos(end) - math.cos(yaw)), _wrap(end)


class GeometricCoach:
    """Bounded, persistent bypass/pursuit teacher, not a navigation guarantee.

    Reuses the feasibility controller's pure-pursuit/right-positive convention,
    but replans from the present pose, checks both passages and inflated swept
    rectangles, and does not inherit its layout-mirror-only side decision.
    Optional executed_controls in diagnostics improves motion direction;
    otherwise current speed and consecutive poses give a conservative bound.
    """

    def __init__(self, config: CoachConfig = CoachConfig(), *,
                 geometry: SimulatorGeometry = SIMULATOR_GEOMETRY):
        self.config, self.geometry = config, geometry
        self.reset()

    def reset(self):
        self._state = {"layout": None, "route_side": None, "route_stage": 0,
                       "last_turn": None, "last_pose": None, "last_time": None,
                       "last_maneuver": None, "reverse_origin": None}

    def state_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "config": asdict(self.config),
                "geometry": asdict(self.geometry), "state": deepcopy(self._state)}

    def load_state_dict(self, value: Mapping):
        # JSON round trips intentionally turn geometry tuples into lists.
        expected = {"schema_version": SCHEMA_VERSION, "config": asdict(self.config),
                    "geometry": asdict(self.geometry)}
        if any(json.dumps(value.get(k), sort_keys=True) != json.dumps(v, sort_keys=True)
               for k, v in expected.items()):
            raise ValueError("incompatible geometric coach checkpoint")
        if set(value.get("state", {})) != set(self._state):
            raise ValueError("incomplete geometric coach state")
        self._state = deepcopy(value["state"])

    def _pose(self, diagnostics):
        x, y = (float(v) for v in diagnostics["position_m"][:2])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("invalid position")
        return x, y, chassis_yaw(diagnostics)

    def _clearance(self, pose, diagnostics, config, *, extra=0.) -> dict:
        """OBB/AABB separating axes, including *all* wheel/chassis extents."""
        x, y, yaw = pose
        g = self.geometry
        gain = float(diagnostics["steering_gain"])
        a = g.chassis_half_length_m
        b = g.half_width(config.calibration.steering_limit_rad, gain)
        c, s = math.cos(yaw), math.sin(yaw)
        margin = self.config.clearance_margin_m + self.config.dynamics_margin_m + extra
        ex, ey = a*abs(c) + b*abs(s), a*abs(s) + b*abs(c)
        xmin, xmax, ymin, ymax = g.arena_bounds_m
        wall = min(x-ex-xmin, xmax-x-ex, y-ey-ymin, ymax-y-ey) - margin
        barrier = None
        if config.obstacle_enabled:
            ox, oy = diagnostics["layout"]["obstacle"]
            hx, hy = g.barrier_half_extents_m
            hx, hy = hx + margin, hy + margin
            dx, dy = x-ox, y-oy
            barrier = max(abs(dx)-ex-hx, abs(dy)-ey-hy,
                          abs(dx*c + dy*s)-a-hx*abs(c)-hy*abs(s),
                          abs(-dx*s + dy*c)-b-hx*abs(s)-hy*abs(c))
        return {"wall_m": wall, "barrier_m": barrier,
                "minimum_m": min(wall, barrier) if barrier is not None else wall}

    def _sweep(self, pose, controls, diagnostics, config, *, latency_s=0., speed_bound=0.) -> dict:
        horizon = max(config.control_interval_s, .75) + self.config.braking_time_s
        gain = float(diagnostics["steering_gain"])
        count = max(1, math.ceil(horizon / min(self.config.sweep_step_s, config.control_interval_s)))
        dt = horizon / count
        radius = math.hypot(self.geometry.chassis_half_length_m,
                            self.geometry.half_width(config.calibration.steering_limit_rad, gain))
        rate = abs(controls.drive_mps * math.tan(controls.steering_rad*gain) / self.geometry.wheelbase_m)
        # Covers inter-sample motion plus unknown heading/motion during latency.
        extra = speed_bound * latency_s + (abs(controls.drive_mps) + radius*rate) * dt / 2
        values = [self._clearance(bicycle_step(pose, controls, i*dt, steering_gain=gain,
                                             geometry=self.geometry), diagnostics, config, extra=extra)
                  for i in range(count+1)]
        wall = min(v["wall_m"] for v in values)
        barriers = [v["barrier_m"] for v in values if v["barrier_m"] is not None]
        barrier = min(barriers) if barriers else None
        minimum = min(wall, barrier) if barrier is not None else wall
        return {"safe": minimum > 0., "wall_m": wall, "barrier_m": barrier,
                "minimum_m": minimum, "horizon_s": horizon, "latency_s": latency_s,
                "inflation_m": extra, "planned_controls": asdict(controls)}

    def _line_safe(self, start, end, diagnostics, config):
        distance = math.dist(start, end)
        yaw = math.atan2(end[1]-start[1], end[0]-start[0])
        count = max(1, math.ceil(distance / .025))
        return all(self._clearance((start[0] + (end[0]-start[0])*i/count,
                                   start[1] + (end[1]-start[1])*i/count, yaw),
                                  diagnostics, config, extra=.0125)["minimum_m"] > 0
                   for i in range(count+1))

    def _route(self, diagnostics, config, side):
        layout, g = diagnostics["layout"], self.geometry
        sx, sy = layout["start"]
        tx, ty = layout["target"]
        ox, oy = layout["obstacle"]
        angle = math.atan2(ty-sy, tx-sx)
        ex, ey = math.cos(angle), math.sin(angle)
        nx, ny = -ey, ex
        hx, hy = g.barrier_half_extents_m
        long = hx*abs(ex) + hy*abs(ey)
        lateral = hx*abs(nx) + hy*abs(ny)
        width = g.half_width(config.calibration.steering_limit_rad, diagnostics["steering_gain"])
        radius = g.wheelbase_m / max(1e-8, math.tan(config.calibration.steering_limit_rad
                                                           * diagnostics["steering_gain"]))
        offset = lateral + width + self.config.clearance_margin_m + .15
        lead = max(.55, radius + g.chassis_half_length_m)
        points = [(ox-ex*lead+side*nx*offset, oy-ey*lead+side*ny*offset),
                  (ox+ex*(long+g.chassis_half_length_m+.1)+side*nx*offset,
                   oy+ey*(long+g.chassis_half_length_m+.1)+side*ny*offset)]
        # Check the entire side corridor, not just center rays/nearest walls.
        feasible = self._line_safe(points[0], points[1], diagnostics, config)
        return points, feasible, (ex, ey), long

    def _choose_turn(self, error):
        old = self._state["last_turn"]
        if abs(error) <= self.config.heading_exit_rad:
            turn = None
        elif abs(error) < self.config.heading_enter_rad:
            turn = old if (old == "left" and error > 0) or (old == "right" and error < 0) else None
        else:
            turn = "left" if error > 0 else "right"
        self._state["last_turn"] = turn
        return turn

    def propose(self, diagnostics: Mapping, config: SimulatorConfig) -> dict:
        """Pure cue dict, plus a privileged diagnostic branch; no action side effects.

        ``text=None`` means abstain. Geometry is NOT an actor/critic/evaluator
        feature. A proposal has no issue time until the window ledger issues it.
        """
        if not isinstance(config, SimulatorConfig):
            raise TypeError("the actual SimulatorConfig is required")
        audit: dict[str, Any] = {"privileged": True, "geometry_version": self.geometry.version,
                                 "prediction": "inflated bicycle sweep; not a dynamics certificate"}

        def result(text, reason, maneuver=None, safety=False):
            self._state["last_maneuver"] = maneuver
            audit.update(route_side=self._state["route_side"], route_stage=self._state["route_stage"])
            return {"text": text, "reason": reason, "maneuver": maneuver,
                    "safety": safety, "diagnostics": audit}

        try:
            pose = self._pose(diagnostics)
            gain = float(diagnostics["steering_gain"])
            if not math.isfinite(gain) or not 0 < gain <= 2:
                raise ValueError("invalid steering gain")
            if config.calibration.steering_limit_rad*gain >= math.pi/2:
                raise ValueError("unsupported effective steering limit")
            for name in ("start", "target", "obstacle"):
                coordinates = diagnostics["layout"][name]
                if len(coordinates) != 2 or not all(math.isfinite(v) for v in coordinates):
                    raise ValueError("invalid layout coordinates")
            layout_key = json.dumps(diagnostics["layout"], sort_keys=True)
            now = float(diagnostics.get("simulator_time_s", diagnostics.get("time_s",
                                     diagnostics.get("physical_time_s", 0.))))
            _nonnegative(now, "diagnostic time")
            if (self._state["layout"] != layout_key or
                    (self._state["last_time"] is not None and now < self._state["last_time"])):
                self.reset()
                self._state["layout"] = layout_key
            speed = _nonnegative(diagnostics["geometry"]["speed_mps"], "speed")
            measured = None
            if self._state["last_time"] is not None and now > self._state["last_time"]:
                old = self._state["last_pose"]
                measured = ((pose[0]-old[0])*math.cos(pose[2]) +
                            (pose[1]-old[1])*math.sin(pose[2])) / (now-self._state["last_time"])
            self._state.update(last_pose=list(pose), last_time=now)
            raw_control = diagnostics.get("executed_controls")
            current = raw_control if isinstance(raw_control, Control) else Control(**raw_control) if raw_control else None
            latency = max(self.config.control_latency_s,
                          _nonnegative(diagnostics.get("control_latency_s", 0.), "control latency"))
            bound = max(speed, abs(measured or 0.))
            audit.update(position_m=list(pose[:2]), chassis_yaw_rad=pose[2], speed_bound_mps=bound,
                         camera_pan_used_for_alignment=False)
            if diagnostics.get("end_reason"):
                return result(None, "episode_ended")
            neutral = config.calibration.neutral()
            calibration = config.calibration
            controls = {"forward": Control(tilt_rad=neutral.tilt_rad, drive_mps=calibration.forward_limit_mps),
                        "left": Control(tilt_rad=neutral.tilt_rad, steering_rad=-calibration.steering_limit_rad,
                                        drive_mps=calibration.forward_limit_mps),
                        "right": Control(tilt_rad=neutral.tilt_rad, steering_rad=calibration.steering_limit_rad,
                                         drive_mps=calibration.forward_limit_mps),
                        "reverse": Control(tilt_rad=neutral.tilt_rad, drive_mps=-calibration.reverse_limit_mps)}
            sweeps = {name: self._sweep(pose, control, diagnostics, config,
                                       latency_s=latency, speed_bound=bound)
                      for name, control in controls.items()}
            audit["sweeps"] = sweeps
            clearance = self._clearance(pose, diagnostics, config)
            audit["current_clearance"] = clearance
            # At unknown direction use both signs; pan/image centering is irrelevant.
            signs = [1 if current.drive_mps >= 0 else -1] if current else (
                [1 if measured >= 0 else -1] if measured is not None and abs(measured) > .01 else [-1, 1])
            motion_sweeps = [self._sweep(pose, Control(tilt_rad=neutral.tilt_rad,
                             drive_mps=sign*bound,
                             steering_rad=current.steering_rad if current else 0.), diagnostics, config)
                             for sign in signs]
            audit["motion_sweeps"] = motion_sweeps
            moving_risk = bound > .02 and any(not v["safe"] for v in motion_sweeps)
            if moving_risk or clearance["minimum_m"] <= 0:
                # A moving vehicle must stop before a reverse instruction can be checked.
                return result("Stop driving now.", "projected_contact" if moving_risk else "clearance_margin",
                              "stop", True)
            if not sweeps["forward"]["safe"]:
                if bound <= .02 and sweeps["reverse"]["safe"]:
                    if self._state["reverse_origin"] is None:
                        self._state["reverse_origin"] = list(pose[:2])
                    return result("Reverse straight briefly, then stop.", "checked_rear_clearance", "reverse", True)
                # A checked turn may avoid a barrier, but walls always precede alignment.
                if sweeps["forward"]["wall_m"] <= 0 or not any(sweeps[s]["safe"] for s in ("left", "right")):
                    return result("Stop driving now.", "no_safe_forward_sweep", "stop", True)

            target = tuple(float(v) for v in diagnostics["layout"]["target"])
            if not all(math.isfinite(v) for v in target):
                raise ValueError("invalid target")
            target_error = _wrap(math.atan2(target[1]-pose[1], target[0]-pose[0])-pose[2])
            audit["target_heading_error_rad"] = target_error
            if (config.task_version == "safe-stop-v1" and diagnostics["geometry"].get("clearance_m", 9.) <= .36
                    and abs(target_error) <= math.radians(15)):
                return result("Stop driving now.", "safe_target_stop", "stop")
            direct = self._line_safe(pose[:2], target, diagnostics, config)
            aim = target
            side = self._state["route_side"]
            if config.obstacle_enabled and (not direct or side is not None):
                routes = {s: self._route(diagnostics, config, s) for s in (-1, 1)}
                audit["passage_feasible"] = {str(s): routes[s][1] for s in routes}
                if side is None or not routes[side][1]:
                    candidates = [s for s in routes if routes[s][1]]
                    if not candidates:
                        return result(None, "no_reliable_passage")
                    # Deterministic nearest corridor; ties prefer chassis-left.
                    side = min(candidates, key=lambda s: (math.dist(pose[:2], routes[s][0][0]), -s))
                    self._state.update(route_side=side, route_stage=0)
                points, _, axis, long = routes[side]
                ox, oy = diagnostics["layout"]["obstacle"]
                along = (pose[0]-ox)*axis[0] + (pose[1]-oy)*axis[1]
                support = (self.geometry.chassis_half_length_m*abs(math.cos(pose[2])*axis[0] + math.sin(pose[2])*axis[1])
                           + self.geometry.half_width(calibration.steering_limit_rad, gain)
                           * abs(-math.sin(pose[2])*axis[0] + math.cos(pose[2])*axis[1]))
                fully_clear = along-support > long+self.config.clearance_margin_m+self.config.dynamics_margin_m
                audit["full_footprint_passed_barrier"] = fully_clear
                if fully_clear:
                    self._state.update(route_side=None, route_stage=0)
                else:
                    stage = self._state["route_stage"]
                    if stage == 0 and (math.dist(pose[:2], points[0]) < self.config.waypoint_tolerance_m
                                       or along >= -max(.55, self.geometry.wheelbase_m /
                                           math.tan(calibration.steering_limit_rad*gain))):
                        stage = 1
                        self._state["route_stage"] = 1
                    aim = points[stage]
            error = _wrap(math.atan2(aim[1]-pose[1], aim[0]-pose[0])-pose[2])
            audit.update(aim_m=list(aim), aim_heading_error_rad=error)
            turn = self._choose_turn(error)
            if turn:
                if not sweeps[turn]["safe"]:
                    return result("Stop driving now.", "required_turn_sweep_blocked", "stop", True)
                passing = self._state["route_side"]
                if passing is not None:
                    route_name = "left" if passing == 1 else "right"
                    return result(f"Pass the barrier on the {route_name}; begin steering {turn} while driving forward.",
                                  "persistent_barrier_passage", f"pass_{route_name}_turn_{turn}")
                return result(f"Steer {turn} toward the ball while driving forward.", "chassis_target_alignment", f"steer_{turn}")
            if not sweeps["forward"]["safe"]:
                return result("Stop driving now.", "straight_sweep_blocked", "stop", True)
            if self._state["route_side"] is not None:
                name = "left" if self._state["route_side"] == 1 else "right"
                return result(f"Continue passing on the {name}; straighten the wheels and drive forward.",
                              "partial_vehicle_passage", f"pass_{name}_straight")
            return result("Straighten the wheels and drive forward toward the ball.", "clear_aligned_approach", "forward")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            audit["invalid_diagnostics"] = str(exc)
            return result(None, "unreliable_geometry")


class CoachingWindows:
    """One named gate and at most one reward receipt per nonoverlapping window.

    All times are in one simulator clock. Call at every acquisition: a window
    boundary strictly between calls uses the previous acquisition count (the
    count changes at the next capture, never by interpolation). Delayed calls
    do not backdate cues. Checkpointing is a pause, NOT finish(terminal=False).
    """

    def __init__(self, seed: int, *, condition: str = "G", total: int = 1024,
                 initialization: bool = False, window_s: float = 4., reaction_s: float = .25,
                 rng_name: str = SEED_NAMESPACE, coach: GeometricCoach | None = None):
        guidance_schedule(0, total, condition=condition, initialization=initialization)
        if type(seed) is not int or seed < 0 or not isinstance(rng_name, str) or not rng_name:
            raise ValueError("expected a nonnegative replica seed and named RNG")
        if window_s not in (4., 6.) or _nonnegative(reaction_s, "reaction_s") >= window_s:
            raise ValueError("freeze a four/six second window and shorter reaction time")
        self.settings = dict(seed=seed, condition=condition, total=total, initialization=initialization,
                             window_s=float(window_s), reaction_s=float(reaction_s), rng_name=rng_name)
        derived = int.from_bytes(hashlib.sha256(f"{rng_name}:{seed}".encode("ascii")).digest()[:8], "big")
        self.seed_manifest = {"replica_seed": seed, "rng_name": rng_name, "derived_seed": derived}
        self._rng = random.Random(derived)
        self.coach = coach or GeometricCoach()
        self.ledger = InstructionLedger(clock_domain="simulator_seconds")
        self.windows: list[dict] = []
        self._current: int | None = None
        self._active: str | None = None
        self._last_now: float | None = None
        self._last_acquired = 0.
        self._episode = 0
        self._ended = False
        self._cutoff = False
        self._action_ids: dict[str, dict] = {}
        self._audits: dict[str, dict] = {}
        self._pending_events: list[dict] = []
        self._pending_closed: list[dict] = []

    def _window(self):
        return self.windows[self._current] if self._current is not None else None

    def _cue(self, instruction_id):
        for window in self.windows:
            for cue in window["instructions"]:
                if cue["instruction_id"] == instruction_id:
                    return cue
        raise KeyError(instruction_id)

    def _event(self, cue, kind, now, text, *, target=False, available_at=None):
        identity = cue["instruction_id"] if kind == "instruction" else f"{cue['instruction_id']}:{kind}"
        event = InstructionEvent(identity, text, now, now,
                                 now if available_at is None else available_at,
                                 kind=kind, target_id=cue["instruction_id"] if target else None)
        self.ledger.record(event)
        self._pending_events.append(asdict(event))

    def _open(self, now, acquired):
        p = guidance_probability(acquired, self.settings["total"], condition=self.settings["condition"],
                                 initialization=self.settings["initialization"])
        draw = self._rng.random()  # Exactly once, including p=0/1; never reward-dependent.
        window = {"window_id": f"coach-e{self._episode}-w{len(self.windows)}", "episode": self._episode,
                  "start_s": now, "scheduled_end_s": now+self.settings["window_s"],
                  "acquired_at_start": acquired, "probability": p, "gate_draw": draw, "enabled": draw < p,
                  "instructions": [], "actions": [], "abstentions": 0, "closed_at_s": None,
                  "close_reason": None, "assessable": False, "eligible_instruction_ids": [],
                  "closing_transition_id": None, "reward_receipt": None}
        self.windows.append(window)
        self._current = len(self.windows)-1

    def issue(self, proposal: Mapping, *, now: float, available_at_s: float | None = None) -> str | None:
        """Issue once, or safety-supersede. Never inject an unenabled proposal."""
        _nonnegative(now, "issue time")
        available = now if available_at_s is None else _nonnegative(available_at_s, "availability")
        if available < now:
            raise ValueError("cannot backdate instruction availability")
        window = self._window()
        if (not window or not window["enabled"] or
                max(now, available) >= window["scheduled_end_s"]):
            return None
        if now < window["start_s"]:
            raise ValueError("cue predates window")
        if not proposal.get("text"):
            window["abstentions"] += 1
            return None
        if window["instructions"]:
            previous = window["instructions"][-1]
            if not proposal.get("safety") or (previous["status"] == "completed" and
                                               previous["text"] == proposal["text"]):
                return None
        if self._active:
            previous = self._cue(self._active)
            if previous["text"] == proposal["text"]:
                return self._active
            self._close_cue(previous, "superseded", now)
        identity = f"{window['window_id']}:cue{len(window['instructions'])}"
        cue = {"instruction_id": identity, "text": proposal["text"], "maneuver": proposal.get("maneuver"),
               "issued_at_s": now, "available_at_s": available, "status": "active",
               "ended_at_s": None, "completed_at_s": None}
        window["instructions"].append(cue)
        self._active = identity
        self._audits[identity] = {"reason": proposal.get("reason"),
                                  "diagnostics": deepcopy(proposal.get("diagnostics", {}))}
        self._event(cue, "instruction", now, cue["text"], available_at=available)
        return identity

    def _close_cue(self, cue, status, now):
        if cue["status"] != "active":
            return
        cue.update(status=status, ended_at_s=now)
        if status == "completed":
            cue["completed_at_s"] = now
        # A never-available cue must not acquire a backdated closing event.
        event_time = max(now, cue["available_at_s"])
        self._event(cue, "completed" if status == "completed" else "interrupted", event_time,
                    f"Guidance {status}: {cue['text']}", target=True)
        self._active = None

    def complete(self, instruction_id: str | None = None, *, now: float):
        _nonnegative(now, "completion time")
        identity = instruction_id or self._active
        if identity is None:
            return
        cue = self._cue(identity)
        if now < cue["available_at_s"]:
            raise ValueError("completion predates cue availability")
        self._close_cue(cue, "completed", now)

    def record_action(self, receipt: CommandReceipt | Mapping, *, generated_at_s: float,
                      decision_time_s: float | None = None, source: str = "policy",
                      exposed_instruction_ids: list[str] | None = None) -> bool:
        """Causal opportunity, not a compliance label. Use ACTUAL receipt controls.

        A fallback, stale pre-cue inference, or insufficient receipt-time reaction
        interval is never credited. decision_time_s is the observation/history
        cutoff actually used by inference, not the later wall-clock start time.
        A missing decision time conservatively uses generation time; callers
        with delayed frames must supply the actual history cutoff. Explicit
        exposure IDs must come from the history actually consumed by inference;
        an empty list means no cue exposure, while None preserves legacy timing.
        """
        data = asdict(receipt) if isinstance(receipt, CommandReceipt) else deepcopy(dict(receipt))
        received = _nonnegative(data.get("received_at_s", data.get("accepted_at_s")), "receipt time")
        generated = _nonnegative(generated_at_s, "generation time")
        decision = generated if decision_time_s is None else _nonnegative(decision_time_s, "decision time")
        if not decision <= generated <= received:
            raise ValueError("history cutoff must precede generation and receipt")
        if exposed_instruction_ids is not None:
            if (not isinstance(exposed_instruction_ids, (list, tuple)) or
                    any(not isinstance(v, str) or not v for v in exposed_instruction_ids)):
                raise ValueError("exposed instruction IDs must be a list of nonempty strings")
            exposed_instruction_ids = sorted(set(exposed_instruction_ids))
        control = data["executed_controls"]
        control = control if isinstance(control, Control) else Control(**control)
        action_id = f"{data.get('session_id', self._episode)}:{data['sequence_id']}"
        action = {"action_id": action_id, "generated_at_s": generated, "accepted_at_s": received,
                  "decision_time_s": decision,
                  "exposed_instruction_ids": exposed_instruction_ids,
                  "executed_controls": asdict(control), "accepted": bool(data["accepted"]),
                  "safety_status": data.get("safety_status", "executing"), "source": source}
        if action_id in self._action_ids:
            previous = self._action_ids[action_id]
            if {**previous, "exposed_instruction_ids": previous.get("exposed_instruction_ids")} != action:
                raise ValueError("conflicting action receipt")
            return False
        # Receipt-time ownership can be resolved before the next acquisition
        # is harvested. Such empty windows never get retrospective proposals.
        current = self._window()
        while current and received >= current["scheduled_end_s"]:
            end = current["scheduled_end_s"]
            self._close(end, "expiry")
            self._open(end, self._last_acquired)
            current = self._window()
        owners = [w for w in self.windows if w["start_s"] <= received and
                  (received < (w["closed_at_s"] if w["closed_at_s"] is not None else w["scheduled_end_s"])
                   or (w["close_reason"] == "terminal" and received == w["closed_at_s"]))]
        if not owners:
            raise ValueError("receipt has no physical coaching window")
        window = owners[-1]
        if window["reward_receipt"] is not None:
            raise ValueError("late evidence cannot change an immutable reward receipt")
        window["actions"].append(action)
        self._action_ids[action_id] = action
        if window["closed_at_s"] is not None:
            eligible = [] if window["close_reason"] == "infrastructure_truncation" else self._eligible(window)
            window.update(eligible_instruction_ids=eligible, assessable=bool(eligible))
            self._pending_closed = [deepcopy(window) if w["window_id"] == window["window_id"] else w
                                    for w in self._pending_closed]
        return True

    def _eligible(self, window):
        eligible = []
        cutoff = window["closed_at_s"]
        for cue in window["instructions"]:
            if (cue["status"] == "cancelled" or
                    (window["close_reason"] == "fade_cutoff" and cue["status"] != "completed")):
                continue
            end = min(cutoff, cue["ended_at_s"] if cue["ended_at_s"] is not None else cutoff)
            for action in window["actions"]:
                if (action["accepted"] and action["source"] in ("policy", "teacher", "student")
                        and action["safety_status"] == "executing"
                        and (action.get("exposed_instruction_ids") is None or
                             cue["instruction_id"] in action["exposed_instruction_ids"])
                        and cue["available_at_s"] <= action["generated_at_s"]
                        and cue["available_at_s"] <= action["decision_time_s"]
                        and action["accepted_at_s"] >= cue["available_at_s"]+self.settings["reaction_s"]
                        and action["accepted_at_s"] < end):
                    eligible.append(cue["instruction_id"])
                    break
        return eligible

    def _maybe_complete(self, proposal, diagnostics, now):
        if not self._active:
            return
        cue = self._cue(self._active)
        if now <= cue["available_at_s"] or proposal["reason"] == "unreliable_geometry":
            return
        maneuver = cue["maneuver"] or ""
        audit = proposal.get("diagnostics", {})
        original = self._audits[cue["instruction_id"]]["diagnostics"]
        speed = diagnostics.get("geometry", {}).get("speed_mps", math.inf)
        aligned = abs(audit.get("aim_heading_error_rad", math.inf)) <= self.coach.config.heading_exit_rad
        completed = False
        if maneuver == "stop":
            completed = speed <= .02
        elif maneuver == "reverse":
            origin = original.get("position_m")
            position = audit.get("position_m")
            if origin and position:
                yaw = original["chassis_yaw_rad"]
                backwards = -(position[0]-origin[0])*math.cos(yaw) - (position[1]-origin[1])*math.sin(yaw)
                completed = backwards >= .08 and speed <= .02
        elif maneuver.startswith("steer_"):
            completed = aligned and proposal.get("maneuver") == "forward"
        elif maneuver.startswith("pass_"):
            completed = (audit.get("full_footprint_passed_barrier", False)
                         or audit.get("route_stage", 0) > original.get("route_stage", 0)
                         or ("_turn_" in maneuver and aligned))
        if completed:
            self.complete(cue["instruction_id"], now=now)

    def _close(self, now, reason, transition_id=None):
        window = self._window()
        if window is None:
            return
        if self._active:
            self._close_cue(self._cue(self._active), "cancelled" if reason in
                            ("fade_cutoff", "infrastructure_truncation") else "expired", now)
        window.update(closed_at_s=now, close_reason=reason, closing_transition_id=transition_id)
        eligible = [] if reason == "infrastructure_truncation" else self._eligible(window)
        window.update(eligible_instruction_ids=eligible, assessable=bool(eligible))
        self._pending_closed.append(deepcopy(window))
        self._current = None

    def _result(self, now, diagnostic=None):
        active = self._cue(self._active) if self._active else None
        window = self._window()
        visible = active if (active and active["available_at_s"] <= now and window
                             and now < window["scheduled_end_s"]) else None
        status = "active" if visible else window["instructions"][-1]["status"] if (
            window and window["instructions"] and not active) else "inactive"
        result = {"text": visible["text"] if visible else None,
                  "status": status, "instruction_id": self._active,
                  "events": deepcopy(self._pending_events), "closed_windows": deepcopy(self._pending_closed),
                  "window_id": self._window()["window_id"] if self._window() else None,
                  "diagnostics": deepcopy(diagnostic)}
        self._pending_events.clear()
        self._pending_closed.clear()
        return result

    def bind_transition(self, window_id: str, transition_id: str) -> dict:
        """Bind an elapsed boundary to its enclosing captured physical transition."""
        if not isinstance(transition_id, str) or not transition_id:
            raise ValueError("closing transition needs a stable identity")
        window = next(w for w in self.windows if w["window_id"] == window_id)
        if window["closed_at_s"] is None:
            raise ValueError("cannot bind an open coaching window")
        previous = window["closing_transition_id"]
        if previous is not None and previous != transition_id:
            raise ValueError("closing physical transition is immutable")
        window["closing_transition_id"] = transition_id
        self._pending_closed = [deepcopy(window) if w["window_id"] == window_id else w for w in self._pending_closed]
        return deepcopy(window)

    def _bind_pending(self, transition_id):
        if transition_id is not None:
            for window in list(self._pending_closed):
                if window["closing_transition_id"] is None:
                    self.bind_transition(window["window_id"], transition_id)

    def update(self, now: float, acquisitions: float, diagnostics: Mapping,
               config: SimulatorConfig, terminal: bool = False, *,
               transition_id: str | None = None, available_at_s: float | None = None,
               publication_clock: Callable[[], float] | None = None) -> dict:
        """Propose from diagnostics, then sample publication time before issuing.

        The owner serializes publication with inference and captures the next
        observation afterwards. A clock callable is never checkpointed. Explicit
        availability can delay publication further, but cannot predate the clock.
        """
        now = _nonnegative(now, "now")
        acquisitions = _nonnegative(acquisitions, "acquisitions")
        if publication_clock is not None and not callable(publication_clock):
            raise TypeError("publication_clock must be callable")
        if self._last_now is not None and (now < self._last_now or acquisitions < self._last_acquired):
            raise ValueError("simulator/acquisition clocks must be monotone")
        if self._ended:
            self._episode += 1
            self._ended = False
            self.coach.reset()
        crossing = not self.settings["initialization"] and acquisitions >= .75*self.settings["total"]
        window = self._window()
        while window and now >= window["scheduled_end_s"]:
            end = window["scheduled_end_s"]
            exact_end = math.isclose(now, end, rel_tol=0, abs_tol=1e-10)
            reason = ("fade_cutoff" if crossing and not self._cutoff and exact_end else
                      "terminal" if terminal and exact_end else "expiry")
            self._close(end, reason, transition_id)
            if reason == "fade_cutoff":
                self._cutoff = True
            if terminal and exact_end:
                break
            start_acquired = acquisitions if exact_end else self._last_acquired
            self._open(end, start_acquired)
            window = self._window()
        if crossing and not self._cutoff:
            # A receipt may already have rolled an exact-time boundary before
            # this acquisition was harvested. Cancel that boundary's unfinished
            # eligibility, and reuse (do not redraw) its new zero-probability gate.
            for old in self.windows:
                if old["closed_at_s"] == now and old["close_reason"] == "expiry":
                    if old["reward_receipt"] is not None:
                        raise ValueError("fade cutoff cannot revise an already claimed reward")
                    old["close_reason"] = "fade_cutoff"
                    eligible = self._eligible(old)
                    old.update(eligible_instruction_ids=eligible, assessable=bool(eligible))
                    self._pending_closed = [deepcopy(old) if w["window_id"] == old["window_id"] else w
                                            for w in self._pending_closed]
            current = self._window()
            if current and current["start_s"] == now and not current["instructions"]:
                current.update(probability=0., enabled=False, acquired_at_start=acquisitions)
            else:
                self._close(now, "fade_cutoff", transition_id)
            self._cutoff = True
        if terminal:
            self._close(now, "terminal", transition_id)
            self._ended = True
            self._last_now, self._last_acquired = now, acquisitions
            self._bind_pending(transition_id)
            return self._result(now)
        if self._window() is None:
            self._open(now, acquisitions)
        proposal = None
        if self._window()["enabled"]:
            proposal = self.coach.propose(diagnostics, config)
            # An issue can be later than capture, never earlier than geometry.
            diagnostic_time = diagnostics.get("simulator_time_s", diagnostics.get("time_s"))
            if diagnostic_time is not None and diagnostic_time > now+1e-9:
                raise ValueError("cannot backdate current geometry to an earlier issue time")
            if publication_clock is not None:
                published = _nonnegative(publication_clock(), "publication time")
                if published < now:
                    raise ValueError("cannot backdate publication before diagnostic time")
                now = published
                if available_at_s is not None:
                    available_at_s = max(now, _nonnegative(available_at_s, "availability"))
            if now < self._window()["scheduled_end_s"]:
                self._maybe_complete(proposal, diagnostics, now)
                self.issue(proposal, now=now, available_at_s=available_at_s)
        self._last_now, self._last_acquired = now, acquisitions
        self._bind_pending(transition_id)
        return self._result(now, proposal)

    def advance(self, now: float, acquired: float, diagnostics: Mapping,
                config: SimulatorConfig, **kwargs) -> dict:
        return self.update(now, acquired, diagnostics, config, **kwargs)

    def finish(self, now: float, terminal: bool = True, *, transition_id: str | None = None) -> dict:
        """True task end may be scored; infrastructure truncation cancels. Pause by saving."""
        now = _nonnegative(now, "finish time")
        if self._last_now is not None and now < self._last_now:
            raise ValueError("finish time moved backwards")
        window = self._window()
        while window and now > window["scheduled_end_s"]:
            end = window["scheduled_end_s"]
            self._close(end, "expiry", transition_id)
            self._open(end, self._last_acquired)
            window = self._window()
        self._close(now, "terminal" if terminal else "infrastructure_truncation", transition_id)
        self._last_now, self._ended = now, True
        self._bind_pending(transition_id)
        return self._result(now)

    def policy_context(self, as_of: float, scheduled_expiry: bool = False) -> dict:
        """Language-only view; optional known deadlines survive delayed advancement.

        Future expiry is a policy-view annotation, never evidence of actual
        completion and never a mutation of the reward/instruction ledger. The
        caller filters status events by availability when rendering each frame.
        Actual endings after as_of are unknown, even in a later checkpoint.
        """
        context = {"instructions": self.ledger.context(as_of),
                   "statuses": self.ledger.model_status_events(as_of)}
        if not scheduled_expiry:
            return context
        cues = {c["instruction_id"]: (w, c) for w in self.windows for c in w["instructions"]}
        events = [e for e in self.ledger.events(as_of) if e.kind in ("instruction", "correction")]
        for entry, event in zip(context["instructions"], events):
            if event.event_id not in cues:
                continue
            window, cue = cues[event.event_id]
            known_ends = [t for t in (window["closed_at_s"], cue["ended_at_s"], entry["completed_at"])
                          if t is not None and t <= as_of]
            # A cancelled, never-available cue can only appear as zero-length
            # history, not with a completion preceding its own availability.
            deadline = max(entry["issued_at"], min([window["scheduled_end_s"], *known_ends]))
            entry["completed_at"] = deadline
            if not known_ends:
                context["statuses"].append({"text": f"Guidance expired: {cue['text']}",
                                            "available_at": deadline, "kind": "robot_status"})
        context["statuses"].sort(key=lambda e: (e["available_at"], e["text"]))
        return context

    def compliance_window(self, window_id: str) -> dict:
        """Causal text/control evidence only; caller supplies bounded camera evidence."""
        window = next(w for w in self.windows if w["window_id"] == window_id)
        if window["closed_at_s"] is None:
            raise ValueError("compliance evidence must have a fixed physical cutoff")
        return {key: deepcopy(window[key]) for key in
                ("window_id", "start_s", "closed_at_s", "close_reason", "instructions", "actions",
                 "assessable", "eligible_instruction_ids", "closing_transition_id")}

    def claim_reward(self, window_id: str, judgment: str, *, transition_id: str) -> dict:
        """Separate exactly-once receipt, not an inference cache. Units are RAW +/-2."""
        if judgment not in ("adherent", "violated", "uncertain"):
            raise ValueError("invalid compliance judgment; inference errors are not neutral labels")
        window = next(w for w in self.windows if w["window_id"] == window_id)
        if window["closed_at_s"] is None:
            raise ValueError("cannot pay an open window")
        if window["closing_transition_id"] is None:
            raise ValueError("bind the closing physical transition before scoring")
        if transition_id != window["closing_transition_id"]:
            raise ValueError("bonus belongs only to the closing physical transition")
        raw = {"adherent": 2., "violated": -2., "uncertain": 0.}[judgment] if window["assessable"] else 0.
        receipt = {"window_id": window_id, "judgment": judgment, "transition_id": transition_id,
                   "raw": raw, "encoded": raw/10.}
        previous = window["reward_receipt"]
        if previous is not None:
            if previous != receipt:
                raise ValueError("conflicting immutable reward receipt")
            return {**deepcopy(previous), "newly_claimed": False, "pay_raw": 0., "pay_encoded": 0.}
        window["reward_receipt"] = receipt
        return {**receipt, "newly_claimed": True, "pay_raw": raw, "pay_encoded": raw/10.}

    def state_dict(self) -> dict:
        return deepcopy({"schema_version": SCHEMA_VERSION, "settings": self.settings,
                         "seed_manifest": self.seed_manifest, "rng_state": self._rng.getstate(),
                         "coach": self.coach.state_dict(), "ledger": self.ledger.snapshot(),
                         "windows": self.windows, "current": self._current, "active": self._active,
                         "last_now": self._last_now, "last_acquired": self._last_acquired,
                         "episode": self._episode, "ended": self._ended, "cutoff": self._cutoff,
                         "action_ids": self._action_ids, "audits": self._audits,
                         "pending_events": self._pending_events, "pending_closed": self._pending_closed})

    def load_state_dict(self, value: Mapping):
        if (value.get("schema_version") != SCHEMA_VERSION or value.get("settings") != self.settings
                or value.get("seed_manifest") != self.seed_manifest):
            raise ValueError("incompatible coaching window checkpoint")
        value = deepcopy(value)
        self.coach.load_state_dict(value["coach"])
        self.ledger = InstructionLedger.from_snapshot(value["ledger"])

        def tuples(x):
            return tuple(tuples(v) for v in x) if isinstance(x, (tuple, list)) else x

        self._rng.setstate(tuples(value["rng_state"]))
        self.windows = value["windows"]
        for key in ("current", "active", "last_now", "last_acquired", "episode", "ended", "cutoff",
                    "action_ids", "audits", "pending_events", "pending_closed"):
            setattr(self, "_"+key, value[key])
