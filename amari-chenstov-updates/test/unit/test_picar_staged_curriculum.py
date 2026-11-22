"""CPU-only Fix 4 schedule, geometry, clock and durable-restore checks."""

from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from src.robotics.curriculum import (
    LAYOUT_VERSION, barrier_sigma, direct_path_blocked, feasible_routes,
    geometry_audit, path_lateral, sample_layout,
)
from src.robotics.drift import drift_alpha, staged_drift_alpha, steering_gain
from src.robotics.protocol import Control
from src.robotics.simulator import PyBulletBackend, SimulatorConfig


def config(**changes):
    base = SimulatorConfig(layout_version=LAYOUT_VERSION, realtime=False,
                           width=32, height=24, staged_curriculum=True,
                           curriculum_total_intervals=2048)
    return replace(base, **changes)


@pytest.mark.parametrize("exposure,sigma,alpha", [
    (0., 1., 0.), (64., 1., 0.), (128., 1., 0.),
    (768., .08, 0.), (900., .08, 0.), (1024., .08, 0.),
    (1664., .08, 1.), (1856., .08, 1.), (2048., .08, 1.),
    (3000., .08, 1.), (1e9, .08, 1.),
])
def test_exact_fixed_schedule_endpoints_holds_and_no_return(exposure, sigma, alpha):
    # Only curriculum_total_intervals, not the historical appearance clock, scales Fix 4.
    for cfg in (config(), config(total_training_intervals=4096)):
        assert barrier_sigma(exposure, cfg) == sigma
        assert staged_drift_alpha(exposure) == alpha
        assert steering_gain(alpha) == (1. if alpha == 0. else .65)


def test_log_sigma_and_alpha_follow_separate_quintics_and_are_monotone():
    cfg = config()
    for r in np.linspace(0., 1., 129):
        q = r ** 3 * (10 - 15 * r + 6 * r ** 2)
        assert barrier_sigma(128 + 640 * r, cfg) == pytest.approx(
            math.exp(q * math.log(.08)), rel=1e-13)
        assert staged_drift_alpha(1024 + 640 * r) == pytest.approx(q, abs=2e-15)
        assert staged_drift_alpha(128 + 640 * r) == 0.
        assert barrier_sigma(1024 + 640 * r, cfg) == .08
    assert barrier_sigma(448, cfg) == pytest.approx(math.sqrt(.08))
    assert staged_drift_alpha(1344) == .5
    sigmas = [barrier_sigma(n, cfg) for n in range(4097)]
    alphas = [staged_drift_alpha(n) for n in range(4097)]
    assert all(a >= b for a, b in zip(sigmas, sigmas[1:]))
    assert all(a <= b for a, b in zip(alphas, alphas[1:]))
    gains = [steering_gain(alpha) for alpha in alphas]
    assert all(a >= b for a, b in zip(gains, gains[1:]))


@pytest.mark.parametrize("boundary", [128., 768., 1024., 1664., 2048.])
def test_zero_first_and_second_derivatives_on_both_sides_of_boundaries(boundary):
    cfg = config()
    functions = (lambda n: barrier_sigma(n, cfg), staged_drift_alpha,
                 lambda n: steering_gain(staged_drift_alpha(n)))
    h = .01
    for function in functions:
        center = function(boundary)
        for sign in (-1, 1):
            near = function(boundary + sign * h)
            farther = function(boundary + sign * 2 * h)
            assert abs((near - center) / h) < 1e-8
            assert abs((farther - 2 * near + center) / h ** 2) < 1e-7


def test_fractional_exposures_remain_monotone_without_endpoint_overshoot():
    cfg = config()
    # Tick clocks can approach endpoints much more closely than integer grids.
    offsets = sorted({0., *np.logspace(-10, -1, 101)}, reverse=True)
    for boundary, function, increasing in (
        (128., lambda n: barrier_sigma(n, cfg), False),
        (768., lambda n: barrier_sigma(n, cfg), False),
        (1024., staged_drift_alpha, True),
        (1664., staged_drift_alpha, True),
    ):
        values = [function(boundary - delta) for delta in offsets]
        if increasing:
            assert all(0. <= value <= 1. for value in values)
            assert all(a <= b for a, b in zip(values, values[1:]))
        else:
            assert all(.08 <= value <= 1. for value in values)
            assert all(a >= b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("total", [128, 2048, 4096])
def test_staged_schedule_scales_with_declared_curriculum_total(total):
    cfg = config(curriculum_total_intervals=total)
    full = config()
    scale = total / 2048
    for n in (0., 64., 128., 448., 768., 900., 1024., 1344., 1664., 2048., 3000.):
        assert barrier_sigma(n * scale, cfg) == barrier_sigma(n, full)
        assert staged_drift_alpha(n * scale, total) == staged_drift_alpha(n)
    assert SimulatorConfig.from_dict(json.loads(json.dumps(asdict(cfg)))) == cfg
    for invalid in (0, -1, True, 128.):
        with pytest.raises(ValueError):
            replace(cfg, curriculum_total_intervals=invalid)
        with pytest.raises(ValueError):
            staged_drift_alpha(0., invalid)
    with pytest.raises(ValueError, match="at least 200"):
        config(staged_curriculum=False, curriculum_total_intervals=128)


def test_compressed_geometry_audit_preserves_full_schedule_layout_distributions():
    full = geometry_audit(config(), seeds=range(32))
    compressed = geometry_audit(config(curriculum_total_intervals=128), seeds=range(32))
    assert [row["exposure"] for row in compressed["rows"]] == [0., 8., 28., 48., 64., 84., 104., 128.]
    for a, b in zip(full["rows"], compressed["rows"]):
        assert a["exposure"] == 16 * b["exposure"]
        assert {k: v for k, v in a.items() if k != "exposure"} == {
            k: v for k, v in b.items() if k != "exposure"}


@pytest.mark.parametrize("invalid", [-1., float("nan"), float("inf")])
def test_invalid_schedule_exposures(invalid):
    with pytest.raises(ValueError, match="exposure"):
        staged_drift_alpha(invalid)
    with pytest.raises(ValueError, match="exposure"):
        barrier_sigma(invalid, config())


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", {}, []])
def test_staged_flag_is_a_declared_boolean(invalid):
    with pytest.raises(ValueError, match="staged_curriculum must be boolean"):
        config(staged_curriculum=invalid)
    saved = asdict(config())
    saved["staged_curriculum"] = invalid
    with pytest.raises(ValueError, match="staged_curriculum must be boolean"):
        SimulatorConfig.from_dict(saved)


def test_configuration_json_roundtrip_and_missing_historical_field():
    for cfg in (SimulatorConfig(), config(),
                config(curriculum_exposure_override=768., domain_override="A")):
        saved = json.loads(json.dumps(asdict(cfg), allow_nan=False))
        before = json.loads(json.dumps(saved))
        assert type(saved["staged_curriculum"]) is bool
        assert SimulatorConfig.from_dict(saved) == cfg
        assert saved == before
    historical = asdict(replace(config(), staged_curriculum=False))
    del historical["staged_curriculum"]
    assert SimulatorConfig.from_dict(historical) == replace(config(), staged_curriculum=False)
    assert not SimulatorConfig.from_dict(historical).staged_curriculum
    with pytest.raises(ValueError, match="unsupported or incomplete"):
        SimulatorConfig.from_dict({**historical, "unknown_curriculum": True})
    del historical["total_training_intervals"]
    with pytest.raises(ValueError, match="unsupported or incomplete"):
        SimulatorConfig.from_dict(historical)


def test_historical_default_formulas_are_exactly_unchanged():
    assert SimulatorConfig().staged_curriculum is False

    def quintic(r):
        r = min(1., max(0., r))
        return r ** 3 * (10 + r * (-15 + 6 * r))

    def historical_alpha(exposure, total):
        fraction = exposure / total
        if fraction <= .1 or fraction >= .9:
            return 0.
        if fraction < .45:
            return quintic((fraction - .1) / .35)
        if fraction <= .55:
            return 1.
        return 1. - quintic((fraction - .55) / .35)

    for cfg in (SimulatorConfig(), config(staged_curriculum=False),
                config(staged_curriculum=False, curriculum_exposure_override=144.)):
        for exposure in np.linspace(0., 3000., 3001):
            actual = (exposure if cfg.curriculum_exposure_override is None
                      else cfg.curriculum_exposure_override)
            q = quintic((actual / cfg.curriculum_total_intervals - .1) / .7)
            expected = math.exp((1 - q) * math.log(cfg.barrier_sigma_easy)
                                + q * math.log(cfg.barrier_sigma_hard))
            assert barrier_sigma(exposure, cfg) == expected
            assert drift_alpha(exposure, cfg.total_training_intervals) == historical_alpha(
                exposure, cfg.total_training_intervals)


def test_frozen_overrides_preserve_paired_geometry_and_independent_domains():
    cfg = config()
    a = replace(cfg, curriculum_exposure_override=0., domain_override="A")
    b_initial = replace(cfg, curriculum_exposure_override=768., domain_override="A")
    b_final = replace(b_initial, domain_override="B")
    for seed in range(64):
        broad = sample_layout(seed, 3000., a)
        initial = sample_layout(seed, 3000., b_initial)
        final = sample_layout(seed, 3000., b_final)
        assert initial == final
        assert broad["obstacle"] == sample_layout(seed, 0., cfg)["obstacle"]
        assert initial["obstacle"] == sample_layout(seed, 768., cfg)["obstacle"]
        for field in ("start", "yaw", "target", "mirror"):
            assert broad[field] == initial[field]
        for layout, sigma, override in ((broad, 1., 0.), (initial, .08, 768.)):
            metadata = layout["curriculum"]
            assert metadata["sigma_m"] == sigma
            assert metadata["reset_exposure_intervals"] == 3000.
            assert metadata["curriculum_exposure_intervals"] == override
            assert metadata["path_lateral_m"] == path_lateral(layout)
            assert metadata["direct_path_blocked"] == direct_path_blocked(layout)
            assert cfg.barrier_bounds[0] < layout["obstacle"][1] < cfg.barrier_bounds[1]
            assert feasible_routes(layout, cfg)


def test_full_staged_layout_blockage_audit_and_identical_b_endpoint_distribution():
    audit = geometry_audit(config(), seeds=range(512))
    rows = {row["exposure"]: row for row in audit["rows"]}
    assert tuple(rows) == (0., 128., 448., 768., 1024., 1344., 1664., 2048.)
    assert .4 <= rows[0.]["clear_fraction"] <= .6
    assert rows[768.]["blocked_fraction"] >= .9
    assert rows[0.]["blocked_fraction"] < rows[448.]["blocked_fraction"] < rows[768.]["blocked_fraction"]
    assert rows[0.]["lateral_histogram_counts"] == rows[128.]["lateral_histogram_counts"]
    for n, row in rows.items():
        assert sum(row["lateral_histogram_counts"]) == row["accepted"] == 512
        assert row["blocked_fraction"] == row["blocked_count"] / row["accepted"]
        assert row["alpha"] == staged_drift_alpha(n)
        assert row["steering_gain"] == steering_gain(row["alpha"])
        if n >= 768:
            assert row["lateral_histogram_counts"] == rows[768.]["lateral_histogram_counts"]
            assert row["blocked_count"] == rows[768.]["blocked_count"]
    assert not audit["navigation_success_claim"]
    assert not audit["scripted_controller_checked"]
    json.dumps(audit, allow_nan=False)
    historical = geometry_audit(config(staged_curriculum=False), seeds=range(8))
    assert [row["exposure"] for row in historical["rows"]] == [
        fraction * 2048 for fraction in (0., .1, .25, .45, .6, .8, 1.)]
    assert all("alpha" not in row and "steering_gain" not in row for row in historical["rows"])


@pytest.mark.parametrize("initial_exposure", [127.75, 767.75, 1023.75, 1663.75, 2047.75])
def test_boundary_spanning_episode_and_json_snapshot_replay(initial_exposure):
    pytest.importorskip("pybullet")
    cfg = config()
    with PyBulletBackend(cfg) as sim:
        sim.reset(seed=19, exposure_intervals=initial_exposure)
        layout = json.loads(json.dumps(sim.diagnostics()["layout"]))
        obstacle = sim._p.getBasePositionAndOrientation(sim.obstacle_id, physicsClientId=sim.client_id)
        sim.step(Control(drive_mps=.04, steering_rad=.1))
        current = sim.exposure_intervals
        assert current == pytest.approx(initial_exposure + 1.)
        assert json.loads(json.dumps(sim.diagnostics()["layout"])) == layout
        assert sim._p.getBasePositionAndOrientation(sim.obstacle_id, physicsClientId=sim.client_id) == obstacle
        assert layout["curriculum"]["reset_exposure_intervals"] == initial_exposure
        assert layout["curriculum"]["sigma_m"] == barrier_sigma(initial_exposure, cfg)
        assert sim.diagnostics()["alpha"] == staged_drift_alpha(current)
        snapshot = json.loads(json.dumps(sim.snapshot(), allow_nan=False))
        assert snapshot["initial_exposure"] == initial_exposure
        with PyBulletBackend.from_snapshot(snapshot) as restored:
            assert restored.snapshot() == snapshot
            assert restored.diagnostics() == sim.diagnostics()
            for backend in (sim, restored):
                result = backend.advance(30)
                assert result.timing["alpha"] == backend.diagnostics()["alpha"]
                assert result.timing["steering_gain"] == backend.diagnostics()["steering_gain"]
                backend.advance(15, training=False, capture=False)
            assert restored.diagnostics() == sim.diagnostics()
            assert restored.snapshot() == sim.snapshot()
        current = sim.exposure_intervals
        sim.reset(seed=19)
        assert sim.exposure_intervals == current
        assert sim.diagnostics()["layout"]["curriculum"]["reset_exposure_intervals"] == current
        assert sim.diagnostics()["layout"]["curriculum"]["sigma_m"] == barrier_sigma(current, cfg)


def test_reset_observation_evaluation_and_administration_do_not_advance_clock():
    pytest.importorskip("pybullet")
    with PyBulletBackend(config()) as sim:
        sim.reset(seed=19, exposure_intervals=1344.)
        before = sim.diagnostics()
        # Current alpha interpolates both checker intensities, not only metadata.
        assert before["alpha"] == .5
        for tile, parity in sim._tiles:
            color = sim._p.getVisualShapeData(tile, physicsClientId=sim.client_id)[0][7]
            a, b = ((.72, .30) if parity else (.58, .62))
            assert color == pytest.approx(((a + b) / 2,) * 3 + (1.,), abs=1e-7)
        sim.observe()
        sim.advance(0)
        sim.stop()
        sim.snapshot()
        sim.advance(30, training=False, capture=False)
        assert sim.exposure_intervals == 1344.
        assert sim.diagnostics()["alpha"] == .5
        sim.reset(seed=19)
        assert sim.exposure_intervals == 1344.
        assert sim.diagnostics()["layout"] == before["layout"]
        sim.advance(120, capture=False)
        assert sim.exposure_intervals == pytest.approx(1345.)
        assert sim.diagnostics()["alpha"] > .5


@pytest.mark.parametrize("override,domain,sigma,alpha", [
    (0., "A", 1., 0.), (768., "A", .08, 0.), (768., "B", .08, 1.),
])
def test_frozen_backend_parameters_remain_fixed_while_exposure_advances(override, domain, sigma, alpha):
    pytest.importorskip("pybullet")
    cfg = config(curriculum_exposure_override=override, domain_override=domain)
    with PyBulletBackend(cfg) as sim:
        sim.reset(seed=19, exposure_intervals=1344.)
        sim.advance(120, capture=False)
        assert sim.exposure_intervals == pytest.approx(1345.)
        assert sim.diagnostics()["alpha"] == alpha
        assert sim.diagnostics()["layout"]["curriculum"]["sigma_m"] == sigma
        with PyBulletBackend.from_snapshot(json.loads(json.dumps(sim.snapshot()))) as restored:
            assert restored.snapshot() == sim.snapshot()
            assert restored.diagnostics() == sim.diagnostics()


def test_historical_backend_layout_clock_timing_and_missing_flag_snapshot():
    pytest.importorskip("pybullet")
    cfg = SimulatorConfig(realtime=False, width=32, height=24)
    rng = np.random.default_rng(191)
    mirror = 1 if rng.integers(2) else -1
    layout = {
        "start": (float(rng.uniform(.4, .6)), mirror * float(rng.uniform(-.15, .15))),
        "yaw": mirror * float(rng.uniform(-math.radians(10), math.radians(10))),
        "target": (2.9, mirror * float(rng.uniform(-.2, .2))),
        "obstacle": (1.575, mirror * float(rng.uniform(-.15, .15))), "mirror": mirror,
    }
    with PyBulletBackend(cfg) as sim:
        sim.reset(seed=191, exposure_intervals=500.)
        assert sim.diagnostics()["layout"] == layout
        assert sim.diagnostics()["alpha"] == 1.
        result = sim.advance(120)
        assert "alpha" not in result.timing and "steering_gain" not in result.timing
        assert sim.diagnostics()["alpha"] == drift_alpha(sim.exposure_intervals, 1000)
        historical = sim.snapshot()
        del historical["config"]["staged_curriculum"]
        del historical["sha256"]
        historical["sha256"] = PyBulletBackend._snapshot_hash(historical)
        with PyBulletBackend.from_snapshot(historical) as restored:
            assert not restored.config.staged_curriculum
            assert restored.diagnostics() == sim.diagnostics()
            assert restored.snapshot() == sim.snapshot()


def test_paused_worker_restores_staged_config_and_reset_exposure_from_json():
    pytest.importorskip("pybullet")
    from src.robotics.worker import SimulatorWorker, WORKER_SCHEMA_VERSION

    cfg = config(realtime=True)
    with PyBulletBackend(cfg, run_id="staged-worker") as sim:
        sim.reset(seed=19, exposure_intervals=1023.75)
        sim.advance(120, capture=False)
        sim.stop()
        snapshot = json.loads(json.dumps({
            "worker_schema_version": WORKER_SCHEMA_VERSION,
            "backend": sim.snapshot(), "active": False, "training": True,
        }, allow_nan=False))
        decoded = SimulatorConfig.from_dict(snapshot["backend"]["config"])
        assert decoded == cfg and decoded.staged_curriculum is True
        with SimulatorWorker(decoded, run_id="staged-worker") as worker:
            worker.restore(snapshot)
            assert worker.snapshot() == snapshot
            actual, expected = worker.diagnostics(), sim.diagnostics()
            for field in expected:
                assert actual[field] == expected[field]
            assert actual["worker_active"] is False
            worker.observe()
            assert worker.snapshot()["backend"]["initial_exposure"] == 1023.75
            assert worker.diagnostics()["exposure_intervals"] == expected["exposure_intervals"]


def test_compressed_episode_uses_same_clock_for_blockage_appearance_and_restore():
    pytest.importorskip("pybullet")
    cfg = config(curriculum_total_intervals=128)
    for n in (7.5, 47.5, 63.5, 103.5, 127.5):
        with PyBulletBackend(cfg) as sim:
            sim.reset(seed=19, exposure_intervals=n)
            reset_layout = sim.diagnostics()["layout"]
            sim.advance(120, capture=False)
            assert sim.exposure_intervals == pytest.approx(n + 1.)
            assert sim.diagnostics()["alpha"] == staged_drift_alpha(sim.exposure_intervals, 128)
            assert sim.diagnostics()["layout"] == reset_layout
            with PyBulletBackend.from_snapshot(json.loads(json.dumps(sim.snapshot()))) as restored:
                assert restored.config.curriculum_total_intervals == 128
                assert restored.diagnostics() == sim.diagnostics()
                assert restored.snapshot() == sim.snapshot()
