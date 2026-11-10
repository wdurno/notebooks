"""Bounded CPU Phase 1 diagnostics; no models, downloads, GPU, or hardware."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from numerical_experiment.picar import phase1_scene as scene
from src.robotics.actions import Calibration, action_distribution_to_vector
from src.robotics.geometry import TerminalGeometry, in_success_region
from src.robotics.simulator import SimulatorConfig


def small_config(**kwargs):
    return scene.SceneAuditConfig(simulator=SimulatorConfig(realtime=False, width=160, height=120),
                                   make_videos=False, **kwargs)


@pytest.mark.parametrize("change", [
    {"controller_period_s": .001}, {"controller_period_s": .6},
    {"recording_period_s": .15}, {"recording_period_s": .05},
    {"lookahead_m": float("nan")}, {"stop_clearance_m": .5},
    {"video_fps": 0}, {"video_fps": float("nan")},
])
def test_config_rejects_invalid_diagnostic_settings(change):
    with pytest.raises(ValueError):
        replace(small_config(), **change)


@pytest.mark.parametrize("change", [
    {"realtime": True}, {"deadline_s": 19.}, {"width": 160, "height": 160},
])
def test_audit_cannot_silently_change_task_deadline_or_camera(change):
    with pytest.raises(ValueError):
        scene.SceneAuditConfig(simulator=replace(small_config().simulator, **change))


@pytest.mark.parametrize("drive", [-1., -.8, -.1, 0., .1, .8, 1.])
@pytest.mark.parametrize("steering", [-2., -.5, 0., .5, 2.])
def test_controller_commands_are_reachable_legacy_mixtures(drive, steering):
    q = scene._drive_distribution(steering, drive)
    assert len(q) == 8 and min(q) >= 0 and sum(q) == pytest.approx(1)
    pan, tilt, turn, actual_drive = action_distribution_to_vector(q)
    assert (pan, tilt) == (0., 0.)
    assert actual_drive == pytest.approx(drive)
    assert turn == pytest.approx(np.clip(steering, -(1 + drive) / 2, (1 + drive) / 2))
    controls = Calibration().map_distribution(q)
    Calibration().validate(controls)


def _diagnostics(mirror=1):
    return {"layout": {"start": (.5, 0.), "obstacle": (1.575, 0.),
                       "target": (2.9, 0.), "mirror": mirror},
            "position_m": (.5, 0., .035), "orientation_xyzw": (0., 0., 0., 1.),
            "steering_gain": .65, "geometry": {"clearance_m": 2.14, "heading_error_rad": 0.}}


def test_privileged_pursuit_uses_right_positive_steering_and_mirrors_path():
    config = small_config()
    a = scene.PrivilegedFeasibilityController(_diagnostics(1), config)
    b = scene.PrivilegedFeasibilityController(_diagnostics(-1), config)
    np.testing.assert_array_equal(a.waypoints[:, 0], b.waypoints[:, 0])
    np.testing.assert_array_equal(a.waypoints[:, 1], -b.waypoints[:, 1])
    left, q_left = a.action(_diagnostics(1))
    right, q_right = b.action(_diagnostics(-1))
    assert left.steering_rad < 0 < right.steering_rad
    assert left.steering_rad == -right.steering_rad
    assert left.drive_mps == right.drive_mps == .2
    assert config.simulator.calibration.map_distribution(q_left) == left
    assert config.simulator.calibration.map_distribution(q_right) == right


def test_braking_is_latched_through_continuous_one_second_stop():
    diagnostics = _diagnostics()
    controller = scene.PrivilegedFeasibilityController(diagnostics, small_config())
    diagnostics["geometry"]["clearance_m"] = .35
    action, q = controller.action(diagnostics)
    assert action == Calibration().neutral()
    assert q == (0.,) * 7 + (1.,)
    diagnostics["geometry"]["clearance_m"] = .37
    assert controller.action(diagnostics)[0] == action


def test_optional_braking_caps_approach_distance_without_changing_legacy():
    diagnostics = _diagnostics()
    diagnostics["geometry"]["clearance_m"] = .4
    base = scene.SceneAuditConfig(simulator=SimulatorConfig(realtime=False, width=160,
        height=120, control_interval_s=1.25), controller_period_s=1.25, recording_period_s=1.25,
        make_videos=False)
    old = scene.PrivilegedFeasibilityController(diagnostics, base).action(diagnostics)[0]
    new = scene.PrivilegedFeasibilityController(diagnostics, replace(base, anticipatory_braking=True)).action(diagnostics)[0]
    assert old.drive_mps == pytest.approx(.12)
    assert new.drive_mps == pytest.approx((.4 - .34) / 1.25)
    assert new.drive_mps * 1.25 <= .4 - .34 + 1e-12


def test_model_input_is_an_explicit_rgb_timing_language_whitelist():
    panel = {"truth": {"position_m": [2.3, 0.]}, "state": "success", "seed": 1,
             "model_input": {"rgb_paths": ["rgb.png"], "timestamps_s": [1.],
                             "instruction": scene.INSTRUCTION, "layout": "must not leak"}}
    value = scene.reward_panel_model_input(panel)
    assert set(value) == {"rgb_paths", "timestamps_s", "instruction"}
    value["rgb_paths"].append("another.png")
    assert panel["model_input"]["rgb_paths"] == ["rgb.png"]


@pytest.mark.parametrize("seeds", [(), (-1,), (True,), (1, 1), (1.5,)])
def test_invalid_seeds_do_not_create_output(tmp_path, seeds):
    output = tmp_path / "audit"
    with pytest.raises(ValueError):
        scene.run_scene_audit(output, seeds, small_config())
    assert not output.exists()


def test_existing_directory_and_completed_ancestor_are_never_modified(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        scene.run_scene_audit(output, (4,), small_config())
    assert list(output.iterdir()) == [sentinel]
    (tmp_path / "COMPLETED").touch()
    with pytest.raises(FileExistsError, match="completed"):
        scene.run_scene_audit(tmp_path / "new", (4,), small_config())
    assert not (tmp_path / "new").exists()


def test_completed_ancestor_guard_follows_symlinks(tmp_path):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "COMPLETED").touch()
    alias = tmp_path / "alias"
    alias.symlink_to(sealed, target_is_directory=True)
    with pytest.raises(FileExistsError, match="completed"):
        scene.run_scene_audit(alias / "new", (4,), small_config())
    assert not (sealed / "new").exists()


def test_bounded_bullet_audit_pairs_endpoints_and_never_teleports_controller(tmp_path, monkeypatch):
    bullet = pytest.importorskip("pybullet")
    original_reset = bullet.resetBasePositionAndOrientation
    original_panel_setup = scene._place_reward_panel
    administrative = False
    teleports = []

    def guarded_reset(*args, **kwargs):
        assert administrative, "feasibility must use actual wheel physics, never reset pose"
        teleports.append(args)
        return original_reset(*args, **kwargs)

    def panel_setup(*args, **kwargs):
        nonlocal administrative
        administrative = True
        try:
            return original_panel_setup(*args, **kwargs)
        finally:
            administrative = False

    monkeypatch.setattr(bullet, "resetBasePositionAndOrientation", guarded_reset)
    monkeypatch.setattr(scene, "_place_reward_panel", panel_setup)
    output = tmp_path / "audit"
    result = scene.run_scene_audit(output, (4, 6), small_config())
    assert len(teleports) == 6
    assert result["all_feasible"] and result["successes"] == result["scene_count"] == 4
    assert result["mapping_audit"]["passed"]
    assert result["mirrors_observed"] == [-1, 1]
    assert result["accelerated"] and not result["primary_demonstrations"]
    assert not result["hardware_connected"]
    assert result["reward_panel_truth_verified"]
    assert result["frozen_evaluator_status"] == "pending_parent_RGB_only_scoring"
    assert not list(output.rglob("COMPLETED"))
    assert not list(output.rglob("manifest.json"))
    assert result["videos"] == []
    assert result["performance"]["stored_bytes"] > 0
    json.dumps(result, allow_nan=False)
    outcomes = result["scene_outcomes"]
    for seed in (4, 6):
        a, b = [row for row in outcomes if row["seed"] == seed]
        assert a["layout"] == b["layout"]
        assert (a["alpha"], b["alpha"]) == (0., 1.)
        assert (a["steering_gain"], b["steering_gain"]) == (1., .65)
    for row in outcomes:
        assert row["end_reason"] == "success" and row["completion_time_s"] < 20.
        assert row["held_success_s"] >= 1. - 1e-10
        assert in_success_region(TerminalGeometry(**row["geometry"]))
        assert row["exposure_unchanged"] and row["fallback_ticks"] == 0
        assert row["path_length_m"] > 1.5
        frames = json.loads(Path(row["frames_path"]).read_text())["frames"]
        assert all(f["ego_time_s"] == f["topdown_time_s"] == f["time_s"] for f in frames)
        assert all(Path(f[key]).is_file() for f in frames for key in ("ego_path", "topdown_path"))
        assert row["minimum_frame_std"] > 5
        assert all("PRIVILEGED DIAGNOSTIC" in f["outcome"] for f in frames)
        trajectory = json.loads(Path(row["trajectory_path"]).read_text())
        for step in trajectory["steps"]:
            controls = small_config().simulator.calibration.map_distribution(step["action_probabilities"])
            assert step["requested_control"] == step["receipt"]["executed_controls"] == {
                "pan_rad": controls.pan_rad, "tilt_rad": controls.tilt_rad,
                "steering_rad": controls.steering_rad, "drive_mps": controls.drive_mps}
            assert step["receipt"]["accepted"]
            assert not step["diagnostics"]["geometry"]["contact"]
    assert len(result["reward_panels"]) == 8
    for panel in result["reward_panels"]:
        inputs = scene.reward_panel_model_input(panel)
        assert all(Path(path).suffix == ".png" and Path(path).is_file() for path in inputs["rgb_paths"])
        assert len(inputs["rgb_paths"]) == len(inputs["timestamps_s"])
        if panel["state"] == "stopped_success":
            assert panel["truth"]["held_success_s"] == pytest.approx(1.)
        if panel["state"] == "near_goal_not_success":
            assert panel["truth"]["geometry"]["clearance_m"] > .45
        if panel["state"] == "obstacle_collision":
            assert panel["truth"]["geometry"]["contact"]
            assert panel["obstacle_visible_rgb"]
