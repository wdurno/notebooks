"""CPU-only Phase 1 scene diagnostics, NEVER primary demonstrations.

The feasibility controller reads privileged pose/gain but moves only through
calibrated commands and Bullet wheel dynamics. Only the explicitly labeled
reward-panel administration teleports a vehicle. No model or hardware is used.
The parent owns RunStore, provenance, sealing, and the completion transaction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from src.robotics.actions import ACTION_NAMES, action_distribution_to_vector
from src.robotics.geometry import TerminalGeometry, in_success_region
from src.robotics.protocol import Control, RobotCommand
from src.robotics.simulator import PyBulletBackend, SimulatorConfig, vertical_fov_deg


SCENE_SCHEMA_VERSION = 1
DEFAULT_SEEDS = (4, 5, 6, 7, 8)
INSTRUCTION = "Navigate around the obstacle, approach the red target, and stop at a safe distance."
DIAGNOSTIC = "PRIVILEGED scripted feasibility diagnostic; not teacher/student demonstrations"


@dataclass(frozen=True)
class SceneAuditConfig:
    simulator: SimulatorConfig = SimulatorConfig(realtime=False)
    controller_period_s: float = .5
    recording_period_s: float = .5
    lookahead_m: float = .25
    passage_offset_m: float = .60
    stop_clearance_m: float = .36
    make_videos: bool = True
    video_fps: float = 5.
    anticipatory_braking: bool = False

    def __post_init__(self):
        cfg = self.simulator
        if cfg.realtime:
            raise ValueError("scene audit requires explicitly accelerated CPU diagnostics")
        if not math.isclose(cfg.deadline_s, 20., rel_tol=0, abs_tol=1e-10):
            raise ValueError("Phase 1 feasibility uses the fixed 20-second task deadline")
        if cfg.width * 3 != cfg.height * 4:
            raise ValueError("Phase 1 camera audit requires 4:3 RGB")
        for value in (self.controller_period_s, self.recording_period_s):
            if (not math.isfinite(value) or value <= 0
                    or not math.isclose(value / cfg.physics_dt_s, round(value / cfg.physics_dt_s),
                                        rel_tol=0, abs_tol=1e-8)):
                raise ValueError("diagnostic periods must be positive integral physics ticks")
        if self.controller_period_s > cfg.control_interval_s:
            raise ValueError("diagnostic controller period exceeds protocol command duration")
        if (self.recording_period_s < self.controller_period_s
                or not math.isclose(self.recording_period_s / self.controller_period_s,
                                    round(self.recording_period_s / self.controller_period_s),
                                    rel_tol=0, abs_tol=1e-8)):
            raise ValueError("recording period must be an integral number of control periods")
        if not .1 <= self.lookahead_m <= .6 or not .45 <= self.passage_offset_m <= .7:
            raise ValueError("invalid diagnostic path parameters")
        if not .28 <= self.stop_clearance_m <= .42:
            raise ValueError("braking target must be inside the success clearance interval")
        if not math.isfinite(self.video_fps) or self.video_fps <= 0:
            raise ValueError("video fps must be finite and positive")
        if type(self.anticipatory_braking) is not bool:
            raise ValueError("anticipatory_braking must be boolean")


def _json(path: Path, value) -> None:
    with path.open("x", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected raw HWC uint8 RGB")
    with path.open("xb") as handle:
        Image.fromarray(rgb).save(handle, format="PNG")


def _yaw(diagnostics: dict) -> float:
    x, y, z, w = diagnostics["orientation_xyzw"]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _drive_distribution(steering: float, drive: float) -> tuple[float, ...]:
    """Reachable legacy mixture, with neutral absolute camera channels.

    Both inputs are normalized. Steering does not create an implicit drive:
    forward/backward mass cancels it when needed. Clip to the convex action
    hull rather than claiming feasibility with unreachable box controls.
    """
    drive = max(-1., min(1., drive))
    limit = (1 + drive) / 2
    steering = max(-limit, min(limit, steering))
    turn = abs(steering)
    forward, backward = max(0., drive - turn), max(0., turn - drive)
    q = [0.] * 8
    q[0 if steering < 0 else 1] = turn
    q[2], q[3] = forward, backward
    q[7] = max(0., 1 - turn - forward - backward)
    return tuple(q)


class PrivilegedFeasibilityController:
    """Adaptive pure pursuit over a privileged bypass, not a learning policy."""

    def __init__(self, diagnostics: dict, config: SceneAuditConfig):
        self.config = config
        layout = diagnostics["layout"]
        self.side = layout["mirror"]
        sx, sy = layout["start"]
        ox, oy = layout["obstacle"]
        tx, ty = layout["target"]
        self.waypoints = np.asarray([
            (sx, sy), (ox - .55, oy + self.side * .40),
            (ox + .12, oy + self.side * config.passage_offset_m), (tx, ty),
        ])
        self.segment = 0
        self.braking = False

    def action(self, diagnostics: dict) -> tuple[Control, tuple[float, ...]]:
        cfg, calibration = self.config, self.config.simulator.calibration
        position = np.asarray(diagnostics["position_m"][:2])
        yaw = _yaw(diagnostics)
        geometry = diagnostics["geometry"]
        if (geometry["clearance_m"] <= cfg.stop_clearance_m
                and abs(geometry["heading_error_rad"]) <= math.radians(15)):
            self.braking = True
        if self.braking:
            q = (0.,) * 7 + (1.,)
            return calibration.map_distribution(q), q
        # Project onto the polyline and walk a fixed metric lookahead forward.
        while self.segment < len(self.waypoints) - 2:
            start, end = self.waypoints[self.segment:self.segment + 2]
            if np.dot(position - end, end - start) < 0:
                break
            self.segment += 1
        start, end = self.waypoints[self.segment:self.segment + 2]
        fraction = float(np.clip(np.dot(position - start, end - start) /
                                 np.dot(end - start, end - start), 0., 1.))
        aim = start + fraction * (end - start)
        remaining = cfg.lookahead_m
        for endpoint in self.waypoints[self.segment + 1:]:
            distance = float(np.linalg.norm(endpoint - aim))
            if distance >= remaining:
                aim = aim + (endpoint - aim) * remaining / distance
                break
            remaining -= distance
            aim = endpoint
        delta = aim - position
        heading_error = _wrap(math.atan2(delta[1], delta[0]) - yaw)
        curvature = 2 * math.sin(heading_error) / max(.05, float(np.linalg.norm(delta)))
        # Positive Bullet curvature is left; the protocol steering is RIGHT-positive.
        steering = -math.atan(.20 * curvature) / diagnostics["steering_gain"]
        speed = calibration.forward_limit_mps
        if geometry["clearance_m"] < .65:
            speed = min(speed, .12)
            if cfg.anticipatory_braking:
                speed = min(speed, max(0., (geometry["clearance_m"] - (cfg.stop_clearance_m - .02)) /
                                      cfg.controller_period_s))
        q = _drive_distribution(steering / calibration.steering_limit_rad,
                                speed / calibration.forward_limit_mps)
        return calibration.map_distribution(q), q


def _capture_pair(sim: PyBulletBackend, directory: Path, index: int, *, observation=None) -> dict:
    ego = (np.frombuffer(observation.rgb, dtype=np.uint8).reshape(
        observation.height, observation.width, 3) if observation is not None else sim.render())
    topdown = sim.render_topdown()
    ego_path, topdown_path = directory / f"ego-{index:04d}.png", directory / f"topdown-{index:04d}.png"
    _save_rgb(ego_path, ego)
    _save_rgb(topdown_path, topdown)
    return {"time_s": sim.episode_time_s, "ego_time_s": sim.episode_time_s,
            "topdown_time_s": sim.episode_time_s, "ego_path": str(ego_path),
            "topdown_path": str(topdown_path), "ego_std": float(ego.std()),
            "topdown_std": float(topdown.std()),
            "instruction": f"Privileged scripted feasibility ONLY. {INSTRUCTION}"}


def _control_caption(controls: Control) -> str:
    return (f"pan={math.degrees(controls.pan_rad):.0f} tilt={math.degrees(controls.tilt_rad):.0f} "
            f"steer={math.degrees(controls.steering_rad):.0f} deg; v={controls.drive_mps:.2f} m/s")


def _episode(sim: PyBulletBackend, directory: Path, seed: int, domain: str,
             config: SceneAuditConfig) -> dict:
    directory.mkdir()
    exposure = 0. if domain == "A" else .5 * sim.config.total_training_intervals
    observation = sim.reset(seed=seed, exposure_intervals=exposure)
    initial = sim.diagnostics()
    controller = PrivilegedFeasibilityController(initial, config)
    steps = round(config.controller_period_s / sim.config.physics_dt_s)
    record_every = round(config.recording_period_s / config.controller_period_s)
    frames, rows = [], []
    path_length = 0.
    previous_position = np.asarray(initial["position_m"][:2])
    started = time.perf_counter()
    sequence = 0
    while True:
        before = sim.diagnostics()
        controls, q = controller.action(before)
        if sequence % record_every == 0 or before["end_reason"]:
            frame = _capture_pair(sim, directory, len(frames), observation=observation)
            frame.update(drift_alpha=before["alpha"], end_reason=before["end_reason"],
                         outcome=f"PRIVILEGED DIAGNOSTIC: {before['end_reason'] or 'running'}",
                         requested_control=_control_caption(controls),
                         executed_control=_control_caption(observation.executed_controls),
                         requested_control_si=asdict(controls),
                         executed_control_si=asdict(observation.executed_controls))
            frames.append(frame)
        if before["end_reason"]:
            break
        command = RobotCommand(sim.run_id, observation.session_id, sequence, observation.observation_id,
                               controls, sim.time_s, config.controller_period_s,
                               sim.time_s + config.controller_period_s)
        receipt = sim.command(command)
        if not receipt.accepted:
            raise RuntimeError(f"diagnostic command rejected: {receipt.safety_status}")
        result = sim.advance(steps, training=False)
        observation = result.observation
        after = sim.diagnostics()
        position = np.asarray(after["position_m"][:2])
        path_length += float(np.linalg.norm(position - previous_position))
        previous_position = position
        rows.append({"sequence_id": sequence, "start_time_s": before["physical_time_s"],
                     "time_s": after["physical_time_s"], "requested_control": asdict(controls),
                     "receipt": asdict(receipt), "executed_control": asdict(observation.executed_controls),
                     "action_probabilities": q, "diagnostics": after,
                     "reward_components": asdict(result.reward_components), "timing": result.timing})
        sequence += 1
    wall_s = time.perf_counter() - started
    final = sim.diagnostics()
    _json(directory / "trajectory.json", {"privileged": True, "initial": initial,
          "controller_waypoints_m": controller.waypoints.tolist(), "steps": rows})
    _json(directory / "frames.json", {"frames": frames, "topdown_reader_only": True})
    video = None
    if config.make_videos:
        from .report import assemble_video

        video = assemble_video(frames, directory / "scene.mp4", seed=seed, fps=config.video_fps)
        video["diagnostic"] = DIAGNOSTIC
    outcome = {"seed": seed, "domain": domain, "layout": initial["layout"],
               "passage_side": controller.side, "success": final["end_reason"] == "success",
               "end_reason": final["end_reason"], "completion_time_s": final["physical_time_s"],
               "geometry": final["geometry"], "held_success_s": final["held_success_s"],
               "alpha": final["alpha"], "steering_gain": final["steering_gain"],
               "exposure_unchanged": final["exposure_intervals"] == exposure,
               "fallback_ticks": final["fallback_ticks"], "path_length_m": path_length,
               "wall_s": wall_s, "simulated_seconds_per_wall_second": final["physical_time_s"] / wall_s,
               "control_steps": len(rows), "trajectory_path": str(directory / "trajectory.json"),
               "frames_path": str(directory / "frames.json"), "video": video,
               "minimum_frame_std": min(min(f["ego_std"], f["topdown_std"]) for f in frames)}
    _json(directory / "outcome.json", outcome)
    return outcome


def _red_centroid(rgb: np.ndarray) -> list[float] | None:
    rgb = rgb.astype(float)
    mask = (rgb[:, :, 0] > 90) & (rgb[:, :, 0] > 1.8 * rgb[:, :, 1]) & (rgb[:, :, 0] > 1.8 * rgb[:, :, 2])
    yy, xx = np.nonzero(mask)
    return [float(xx.mean()), float(yy.mean())] if len(xx) >= 3 else None


def _mapping_audit(sim: PyBulletBackend, directory: Path, seed: int) -> dict:
    directory.mkdir()
    calibration = sim.config.calibration
    primitives = []
    for index, name in enumerate(ACTION_NAMES):
        q = tuple(float(j == index) for j in range(8))
        controls = calibration.map_distribution(q)
        calibration.validate(controls)
        primitives.append({"name": name, "probabilities": q,
                           "normalized_vector": action_distribution_to_vector(q), "controls": asdict(controls)})
    mixtures = {}
    for name, indices in (("opposing_steering", (0, 1)), ("opposing_drive", (2, 3)),
                          ("opposing_pan", (4, 5))):
        q = tuple(.5 if j in indices else 0. for j in range(8))
        mixtures[name] = {"probabilities": q, "controls": asdict(calibration.map_distribution(q))}
    motions = []
    for name, controls, exposure in (
        ("forward", calibration.map_distribution((0, 0, 1, 0, 0, 0, 0, 0)), 0.),
        ("reverse", calibration.map_distribution((0, 0, 0, 1, 0, 0, 0, 0)), 0.),
        ("right", calibration.map_distribution((0, 1, 0, 0, 0, 0, 0, 0)), 0.),
        ("right_B", calibration.map_distribution((0, 1, 0, 0, 0, 0, 0, 0)),
         .5 * sim.config.total_training_intervals),
        ("left", calibration.map_distribution((1, 0, 0, 0, 0, 0, 0, 0)), 0.),
        ("steering_only", Control(tilt_rad=calibration.tilt_neutral_rad,
                                  steering_rad=calibration.steering_limit_rad), 0.),
    ):
        sim.reset(seed=seed, exposure_intervals=exposure)
        before = sim.diagnostics()
        result = sim.step(controls, training=False)
        after = sim.diagnostics()
        delta = np.asarray(after["position_m"][:2]) - before["position_m"][:2]
        forward_m = float(np.dot(delta, (math.cos(_yaw(before)), math.sin(_yaw(before)))))
        sim.stop()
        sim.advance(round(1. / sim.config.physics_dt_s), training=False, capture=False)
        motions.append({"name": name, "controls": asdict(controls), "forward_displacement_m": forward_m,
                        "displacement_m": float(np.linalg.norm(delta)),
                        "yaw_change_rad": _wrap(_yaw(after) - _yaw(before)),
                        "alpha": after["alpha"], "steering_gain": after["steering_gain"],
                        "mean_forward_speed_mps": forward_m / result.reward_components.elapsed_s,
                        "stop_speed_mps_after_1s": sim.diagnostics()["geometry"]["speed_mps"],
                        "elapsed_s": result.reward_components.elapsed_s,
                        "receipt_accepted": result.receipt.accepted})
    cameras = []
    for name, q in (
        ("neutral", (0, 0, 0, 0, 0, 0, 0, 1)),
        ("pan_left", (0, 0, 0, 0, .5, 0, 0, .5)),
        ("pan_right", (0, 0, 0, 0, 0, .5, 0, .5)),
        ("tilt_up", (0, 0, 0, 0, 0, 0, .5, .5)),
    ):
        sim.reset(seed=seed, exposure_intervals=0.)
        controls = calibration.map_distribution(q)
        result = sim.step(controls, training=False)
        rgb = np.frombuffer(result.observation.rgb, dtype=np.uint8).reshape(sim.config.height, sim.config.width, 3)
        path = directory / f"camera-{name}.png"
        _save_rgb(path, rgb)
        cameras.append({"name": name, "controls": asdict(controls), "rgb_path": str(path),
                        "target_centroid_px": _red_centroid(rgb), "rgb_std": float(rgb.std())})
    by_name = {row["name"]: row for row in motions}
    centers = {row["name"]: row["target_centroid_px"] for row in cameras}
    visible = all(center is not None for center in centers.values())
    checks = {
        "forward_positive": by_name["forward"]["forward_displacement_m"] > .03,
        "reverse_negative": by_name["reverse"]["forward_displacement_m"] < -.015,
        "right_positive_steering_turns_right": by_name["right"]["yaw_change_rad"] < -.02,
        "B_reduces_steering_response": (
            0 < abs(by_name["right_B"]["yaw_change_rad"]) < .85 * abs(by_name["right"]["yaw_change_rad"])),
        "left_negative_steering_turns_left": by_name["left"]["yaw_change_rad"] > .02,
        "steering_alone_does_not_drive": by_name["steering_only"]["displacement_m"] < .003,
        "brakes_within_1s": all(row["stop_speed_mps_after_1s"] <= .02 for row in motions),
        "all_commands_accepted": all(row["receipt_accepted"] for row in motions),
        "target_visible_for_camera_probes": visible,
        "pan_left_moves_target_right": visible and centers["pan_left"][0] > centers["neutral"][0] + 5,
        "pan_right_moves_target_left": visible and centers["pan_right"][0] < centers["neutral"][0] - 5,
        "tilt_up_moves_target_down": visible and centers["tilt_up"][1] > centers["neutral"][1] + 5,
        "absolute_camera_neutral": primitives[7]["controls"] == asdict(calibration.neutral()),
        "opposing_drive_cancels": mixtures["opposing_drive"]["controls"] == asdict(calibration.neutral()),
        "opposing_pan_cancels": mixtures["opposing_pan"]["controls"] == asdict(calibration.neutral()),
        "opposing_steering_preserves_forward": (
            mixtures["opposing_steering"]["controls"] ==
            asdict(Control(tilt_rad=calibration.tilt_neutral_rad, drive_mps=calibration.forward_limit_mps))),
        "camera_nonblank": all(row["rgb_std"] > 5 for row in cameras),
    }
    audit = {"calibration": asdict(calibration), "physically_validated": False,
             "units": {"camera": "absolute radians; pan left-positive, tilt up-positive",
                       "steering": "right-positive radians", "drive": "metres/second"},
             "horizontal_fov_deg": sim.config.horizontal_fov_deg,
             "vertical_fov_deg": vertical_fov_deg(sim.config.horizontal_fov_deg, sim.config.width, sim.config.height),
             "rgb_shape": [sim.config.height, sim.config.width, 3], "primitives": primitives,
             "mixtures": mixtures, "motion_probes": motions, "camera_probes": cameras,
             "checks": checks, "passed": all(checks.values()),
             "steering_only_probe": "direct SI box command; outside the learned convex action hull"}
    _json(directory / "audit.json", audit)
    return audit


def reward_panel_model_input(panel: dict) -> dict:
    """Whitelist RGB/timing/language only; never send panel truth to a model."""
    value = panel["model_input"]
    return {"rgb_paths": list(value["rgb_paths"]), "timestamps_s": list(value["timestamps_s"]),
            "instruction": value["instruction"]}


def _place_reward_panel(sim: PyBulletBackend, xy: tuple[float, float], yaw: float) -> None:
    """ADMINISTRATIVE known-pose reward setup, NEVER a feasibility maneuver."""
    p, cid = sim._p, sim.client_id
    height = sim.diagnostics()["position_m"][2]
    p.resetBasePositionAndOrientation(sim.vehicle_id, (*xy, height),
                                      p.getQuaternionFromEuler((0, 0, yaw)), physicsClientId=cid)
    p.resetBaseVelocity(sim.vehicle_id, (0, 0, 0), (0, 0, 0), physicsClientId=cid)


def _reward_panels(sim: PyBulletBackend, directory: Path, seed: int) -> list[dict]:
    directory.mkdir()
    panels = []
    for domain in ("A", "B"):
        for state in ("far_deadline_failure", "near_goal_not_success", "stopped_success", "obstacle_collision"):
            exposure = 0. if domain == "A" else .5 * sim.config.total_training_intervals
            sim.reset(seed=seed, exposure_intervals=exposure)
            layout = sim.diagnostics()["layout"]
            administrative = state != "far_deadline_failure"
            if state == "near_goal_not_success":
                _place_reward_panel(sim, (layout["target"][0] - .81, layout["target"][1]), 0.)
            elif state == "stopped_success":
                _place_reward_panel(sim, (layout["target"][0] - .61, layout["target"][1]), 0.)
            elif state == "obstacle_collision":
                ox, oy = layout["obstacle"]
                # Front chassis overlaps the near obstacle face by 1 cm;
                # centering on the slab hides it below the ego camera.
                _place_reward_panel(sim, (ox - .125 - .16 + .01, oy), 0.)
            panel_id = f"panel-{len(panels):03d}"
            folder = directory / panel_id
            folder.mkdir()
            history = []
            if state == "far_deadline_failure":
                sim.advance(round(19. / sim.config.physics_dt_s), training=False, capture=False)
            for index in range(3):
                if index:
                    ticks = 1 if state == "obstacle_collision" else round(.5 / sim.config.physics_dt_s)
                    sim.advance(ticks, training=False, capture=False)
                path = folder / f"rgb-{index:02d}.png"
                rgb = sim.render()
                _save_rgb(path, rgb)
                history.append({"rgb_path": str(path), "time_s": sim.episode_time_s})
                if sim.diagnostics()["end_reason"]:
                    break
            truth = sim.diagnostics()
            blue_pixels = int(((rgb[:, :, 2] > 80) & (rgb[:, :, 2] > 1.8 * rgb[:, :, 0]) &
                               (rgb[:, :, 2] > 1.3 * rgb[:, :, 1])).sum())
            expected = {"far_deadline_failure": "deadline", "near_goal_not_success": None,
                        "stopped_success": "success", "obstacle_collision": "collision"}[state]
            panel = {"panel_id": panel_id, "seed": seed, "domain": domain, "state": state,
                     "setup": "administrative_known_pose_NOT_feasibility" if administrative else "seeded_reset_no_motion",
                     "truth": truth, "expected_end_reason": expected,
                     "obstacle_visible_rgb": blue_pixels >= 10,
                     "truth_verified": truth["end_reason"] == expected and (
                         not in_success_region(TerminalGeometry(**truth["geometry"])) if state == "near_goal_not_success" else True
                     ) and (blue_pixels >= 10 if state == "obstacle_collision" else True),
                     "model_input": {"rgb_paths": [row["rgb_path"] for row in history],
                                     "timestamps_s": [row["time_s"] for row in history], "instruction": INSTRUCTION},
                     "frozen_evaluator_status": "pending_parent_RGB_only_scoring"}
            panels.append(panel)
    _json(directory / "truth.json", {"panels": panels})
    _json(directory / "model_inputs.json", {"panels": [
        {"panel_id": panel["panel_id"], **reward_panel_model_input(panel)} for panel in panels]})
    return panels


def run_scene_audit(output: Path, seeds: Sequence[int] = DEFAULT_SEEDS,
                    config: SceneAuditConfig | None = None) -> dict:
    """Write NEW diagnostic payloads and return JSON data for the parent's store.

    Each seed is paired at fixed A/B endpoints with no training exposure.
    Default feedback matches the nominal .5s cadence, but CPU physics is
    accelerated; this does not measure operational inference deadlines.
    A partial failure stays incomplete. Never creates a COMPLETED marker.
    """
    config = config or SceneAuditConfig()
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    output = Path(output).expanduser().resolve()
    if any((parent / "COMPLETED").exists() for parent in (output, *output.parents)):
        raise FileExistsError("cannot add scene diagnostics inside a completed unit")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _json(output / "config.json", {"schema_version": SCENE_SCHEMA_VERSION,
          "config": asdict(config), "seeds": seeds, "diagnostic": DIAGNOSTIC})
    with PyBulletBackend(config.simulator, run_id="phase1-scene-diagnostic") as sim:
        mapping = _mapping_audit(sim, output / "mapping", seeds[0])
        scenes = [_episode(sim, output / f"seed-{seed}-{domain}", seed, domain, config)
                  for seed in seeds for domain in ("A", "B")]
        panels = _reward_panels(sim, output / "reward-panels", seeds[0])
    summary = {"scene_schema_version": SCENE_SCHEMA_VERSION, "phase": 1,
               "kind": "phase1_scene_audit", "diagnostic": DIAGNOSTIC,
               "accelerated": True, "primary_demonstrations": False, "hardware_connected": False,
               "controller_privileged": True,
               "camera_mode": "pan/tilt fixed at calibrated neutral in BOTH A/B feasibility scenes",
               "feedback_faster_than_nominal": config.controller_period_s < config.simulator.control_interval_s,
               "config": asdict(config), "seeds": seeds, "scene_outcomes": scenes,
               "successes": sum(scene["success"] for scene in scenes), "scene_count": len(scenes),
               "mirrors_observed": sorted({scene["layout"]["mirror"] for scene in scenes}),
               "all_feasible": all(scene["success"] for scene in scenes), "mapping_audit": mapping,
               "reward_panels": panels, "reward_panel_truth_verified": all(p["truth_verified"] for p in panels),
               "reward_model_inputs_path": str(output / "reward-panels/model_inputs.json"),
               "frozen_evaluator_status": "pending_parent_RGB_only_scoring",
               "videos": [scene["video"] for scene in scenes if scene["video"]],
               "wall_s": time.perf_counter() - started, "output": str(output),
               "limitations": ["Privileged controller is not a teacher or student performance estimate.",
                               "Accelerated CPU feedback does not test wall-clock inference deadlines.",
                               "No physical calibration or hardware validation.",
                               "Frozen Qwen panel scoring belongs to the parent; not run here."]}
    summary["performance"] = {
        "total_simulated_task_s": sum(scene["completion_time_s"] for scene in scenes),
        "trajectory_wall_s": sum(scene["wall_s"] for scene in scenes),
        "stored_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "stored_bytes_excludes_summary": True,
    }
    _json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    result = run_scene_audit(args.output, args.seeds, SceneAuditConfig(make_videos=not args.no_video))
    print(json.dumps({key: result[key] for key in ("output", "successes", "scene_count", "all_feasible", "wall_s")}))


if __name__ == "__main__":
    main()
