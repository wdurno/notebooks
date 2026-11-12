"""Fast Fix 3 geometry/compatibility checks; no models or training runs."""

from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from src.robotics.actions import Calibration
from src.robotics.curriculum import (
    CLEARANCE_M, LAYOUT_VERSION, barrier_sigma, direct_path_blocked,
    feasible_routes, geometry_audit, path_lateral, sample_layout,
)
from src.robotics.simulator import PyBulletBackend, SimulatorConfig


def config(**kwargs):
    return SimulatorConfig(layout_version=LAYOUT_VERSION, realtime=False, width=32, height=24, **kwargs)


@pytest.mark.parametrize("change", [
    {"layout_version": "unknown"}, {"barrier_sigma_easy": 0},
    {"barrier_sigma_hard": float("nan")}, {"barrier_sigma_hard": 2.},
    {"barrier_bounds": (-1., 1.)}, {"barrier_bounds": (.3, .4)},
    {"barrier_bounds": (-.5,)}, {"curriculum_total_intervals": 280},
    {"curriculum_total_intervals": True}, {"curriculum_exposure_override": -1},
    {"curriculum_exposure_override": float("inf")}, {"domain_override": "C"},
])
def test_invalid_settings(change):
    with pytest.raises(ValueError):
        replace(config(), **change)


def test_historical_config_and_json_roundtrip():
    historical = asdict(SimulatorConfig())
    additive = ("task_version", "obstacle_enabled", "layout_version", "barrier_sigma_easy",
                "barrier_sigma_hard", "barrier_bounds", "curriculum_total_intervals",
                "curriculum_exposure_override", "domain_override")
    for name in additive:
        del historical[name]
    saved = json.loads(json.dumps(historical))
    assert SimulatorConfig.from_dict(historical) == SimulatorConfig()
    assert historical == saved
    cfg = config(domain_override="B", curriculum_exposure_override=0.)
    assert SimulatorConfig.from_dict(json.loads(json.dumps(asdict(cfg)))) == cfg
    with pytest.raises(ValueError):
        SimulatorConfig.from_dict({**historical, "unsupported": 1})
    del historical["deadline_s"]
    with pytest.raises(ValueError):
        SimulatorConfig.from_dict(historical)


def test_log_quintic_holds_endpoints_monotonicity_and_minimum_ramp():
    cfg = config()
    assert .7 * cfg.curriculum_total_intervals >= 200
    assert barrier_sigma(0, cfg) == pytest.approx(cfg.barrier_sigma_easy)
    assert barrier_sigma(32, cfg) == pytest.approx(cfg.barrier_sigma_easy)
    assert barrier_sigma(144, cfg) == pytest.approx(math.sqrt(cfg.barrier_sigma_easy * cfg.barrier_sigma_hard))
    assert barrier_sigma(256, cfg) == pytest.approx(cfg.barrier_sigma_hard)
    assert barrier_sigma(10000, cfg) == pytest.approx(cfg.barrier_sigma_hard)
    values = [barrier_sigma(n, cfg) for n in range(321)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    for endpoint in (32, 256):
        h = .01
        assert abs(barrier_sigma(endpoint + h, cfg) - barrier_sigma(endpoint - h, cfg)) < 1e-9
    for invalid in (-1., float("nan")):
        with pytest.raises(ValueError):
            barrier_sigma(invalid, cfg)


def test_seeded_draws_preserve_base_geometry_and_fixed_support():
    cfg = config()
    mirrors = set()
    for seed in range(64):
        easy = sample_layout(seed, 0, cfg)
        hard = sample_layout(seed, 320, cfg)
        assert sample_layout(seed, 0, cfg) == easy
        for field in ("start", "yaw", "target", "mirror"):
            assert easy[field] == hard[field]
        for layout in (easy, hard):
            y = layout["obstacle"][1]
            assert cfg.barrier_bounds[0] <= y <= cfg.barrier_bounds[1]
            assert layout["obstacle"][0] == 1.575
            assert layout["curriculum"]["path_lateral_m"] == path_lateral(layout)
            assert layout["curriculum"]["bounds_m"] == list(cfg.barrier_bounds)
            assert feasible_routes(layout, cfg)
            mirrors.add(layout["mirror"])
            json.dumps(layout, allow_nan=False)
    assert mirrors == {-1, 1}


def test_finite_footprint_blocks_a_clear_point_ray_and_handles_slopes():
    layout = {"start": (.5, 0), "target": (2.9, 0), "obstacle": (1.575, .34), "yaw": 0}
    assert .34 - .25 > 0  # Point ray misses, vehicle wheels do not.
    assert direct_path_blocked(layout)
    assert not direct_path_blocked({**layout, "obstacle": (1.575, .38)})
    diagonal = {**layout, "target": (2.9, .2)}
    assert direct_path_blocked(diagonal)
    mirrored = {**diagonal, "target": (2.9, -.2), "obstacle": (1.575, -.34)}
    assert direct_path_blocked(mirrored)
    grazing = {**layout, "obstacle": (1.575, .25 + .09 + CLEARANCE_M)}
    assert direct_path_blocked(grazing)
    assert not direct_path_blocked({**layout, "obstacle": (-1., 0)})


def test_one_route_screen_retains_side_offsets_and_checks_access_and_b_gain():
    cfg = config()
    layout = {"start": (.5, 0.), "yaw": 0., "target": (2.9, 0.), "obstacle": (1.575, .9)}
    assert feasible_routes(layout, cfg) == ("right",)
    assert not PyBulletBackend.feasible_layout(layout)
    assert feasible_routes({**layout, "obstacle": (1.575, -.9)}, cfg) == ("left",)
    assert not feasible_routes({**layout, "start": (1., 0.)}, cfg)
    assert not feasible_routes({**layout, "target": (2.9, 1.1)}, cfg)
    assert not feasible_routes({**layout, "obstacle": (1.575, 1.)}, cfg)
    # A alone has radius .20/tan(15deg); B cannot fit the endpoint access.
    weak = replace(cfg, calibration=Calibration(steering_limit_rad=math.radians(15)))
    assert .20 / math.tan(weak.calibration.steering_limit_rad) + .11 < .95
    assert not feasible_routes(layout, weak)


def test_truncated_distribution_and_audit_targets_are_on_accepted_layouts():
    audit = geometry_audit(config(), seeds=range(1024))
    assert .4 <= audit["rows"][0]["clear_fraction"] <= .6
    assert audit["rows"][-1]["blocked_fraction"] >= .9
    assert audit["rows"][0]["sigma"] > audit["rows"][-1]["sigma"]
    assert audit["rows"][0]["exposure"] == 0
    assert audit["settings"]["layout_version"] == LAYOUT_VERSION
    assert not audit["scripted_controller_checked"]
    assert not audit["navigation_success_claim"]
    for row in audit["rows"]:
        assert sum(row["lateral_histogram_counts"]) == row["accepted"] == 1024
        assert row["blocked_fraction"] == row["blocked_count"] / row["accepted"]
    # Broad support reaches both edges, with no clipping atoms at the bounds.
    edges = audit["rows"][0]["lateral_histogram_counts"]
    assert edges[0] > 0 and edges[-1] > 0
    json.dumps(audit, allow_nan=False)


def test_rejections_are_logged_without_changing_the_base_layout(monkeypatch):
    from src.robotics import curriculum

    original = curriculum.feasible_routes
    calls = 0

    def reject_first(layout, cfg):
        nonlocal calls
        calls += 1
        return () if calls <= 3 else original(layout, cfg)

    baseline = sample_layout(19, 0, config())
    monkeypatch.setattr(curriculum, "feasible_routes", reject_first)
    screened = sample_layout(19, 0, config())
    assert screened["curriculum"]["rejected_proposals"] == 3
    assert screened["curriculum"]["proposals"] == 4
    assert screened["curriculum"]["rejection_counts"] == {"no_feasible_route": 3}
    for key in ("start", "target", "yaw", "mirror"):
        assert screened[key] == baseline[key]
    monkeypatch.setattr(curriculum, "MAX_LAYOUT_ATTEMPTS", 2)
    monkeypatch.setattr(curriculum, "feasible_routes", lambda *args: ())
    with pytest.raises(RuntimeError, match="after 2 proposals"):
        sample_layout(19, 0, config())


def test_domain_and_exposure_overrides_are_independent_and_fixed_per_reset():
    pytest.importorskip("pybullet")
    cfg = config(domain_override="B", curriculum_exposure_override=0.)
    with PyBulletBackend(cfg) as sim:
        sim.reset(seed=19, exposure_intervals=144.)
        layout = sim.diagnostics()["layout"]
        assert layout["curriculum"]["sigma_m"] == pytest.approx(cfg.barrier_sigma_easy)
        assert sim.diagnostics()["alpha"] == 1.
        sim.advance(240, capture=False)
        assert sim.exposure_intervals > 144.
        assert sim.diagnostics()["layout"] == layout
        sim.advance(240, training=False, capture=False)
        exposure = sim.exposure_intervals
        sim.reset(seed=19)
        assert sim.exposure_intervals == exposure
        assert sim.diagnostics()["layout"]["obstacle"] == layout["obstacle"]
        with PyBulletBackend.from_snapshot(json.loads(json.dumps(sim.snapshot()))) as restored:
            assert restored.diagnostics() == sim.diagnostics()
            assert restored.snapshot() == sim.snapshot()
    with PyBulletBackend(config(domain_override="A", curriculum_exposure_override=320.)) as sim:
        sim.reset(seed=19, exposure_intervals=500.)
        assert sim.diagnostics()["alpha"] == 0.
        assert sim.diagnostics()["layout"]["curriculum"]["sigma_m"] == pytest.approx(.08)


def test_new_layout_snapshot_restores_after_curriculum_exposure_advances():
    pytest.importorskip("pybullet")
    with PyBulletBackend(config()) as sim:
        sim.reset(seed=19, exposure_intervals=144.)
        initial_layout = sim.diagnostics()["layout"]
        sim.advance(240, capture=False)
        assert sim.diagnostics()["layout"] == initial_layout
        with PyBulletBackend.from_snapshot(sim.snapshot()) as restored:
            assert restored.snapshot() == sim.snapshot()
            assert restored.diagnostics()["layout"] == initial_layout
            restored.advance(1, capture=False)
            sim.advance(1, capture=False)
            assert restored.snapshot() == sim.snapshot()


def test_legacy_layout_rng_and_historical_snapshot_are_unchanged():
    pytest.importorskip("pybullet")
    cfg = SimulatorConfig(realtime=False, width=32, height=24)
    with PyBulletBackend(cfg) as sim:
        seed = 191
        rng = np.random.default_rng(seed)
        mirror = 1 if rng.integers(2) else -1
        expected = {
            "start": (float(rng.uniform(.4, .6)), mirror * float(rng.uniform(-.15, .15))),
            "yaw": mirror * float(rng.uniform(-math.radians(10), math.radians(10))),
            "target": (2.9, mirror * float(rng.uniform(-.2, .2))),
            "obstacle": (1.575, mirror * float(rng.uniform(-.15, .15))), "mirror": mirror,
        }
        sim.reset(seed=seed)
        assert sim.diagnostics()["layout"] == expected
        old = sim.snapshot()
        for key in ("layout_version", "barrier_sigma_easy", "barrier_sigma_hard", "barrier_bounds",
                    "curriculum_total_intervals", "curriculum_exposure_override", "domain_override"):
            del old["config"][key]
        del old["sha256"]
        old["sha256"] = PyBulletBackend._snapshot_hash(old)
        with PyBulletBackend.from_snapshot(old) as restored:
            assert restored.diagnostics() == sim.diagnostics()
