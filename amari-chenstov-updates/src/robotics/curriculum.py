"""Reset geometry and optional Fix 4 staging, independent of Bullet and policies.

Coordinates and dimensions match the historical simulator. Screening is a
cheap conservative admission diagnostic, not a proof of dynamic reachability.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Iterable

from .drift import smoothstep5, staged_drift_alpha, steering_gain

if TYPE_CHECKING:
    from .simulator import SimulatorConfig

LAYOUT_VERSION = "curriculum-v1"
BARRIER_X = 1.575
BARRIER_HALF_LENGTH = .125
BARRIER_HALF_WIDTH = .25
VEHICLE_HALF_LENGTH = .16
VEHICLE_HALF_WIDTH = .09  # Includes the wheels, not just the chassis.
CLEARANCE_M = .02
WALL_Y = 1.2
MAX_LAYOUT_ATTEMPTS = 1024


def barrier_sigma(exposure: float, config: SimulatorConfig) -> float:
    """Log-quintic blockage ramp; Fix 4 scales 128--768 by total / 2048."""
    if not math.isfinite(exposure) or exposure < 0:
        raise ValueError("curriculum exposure must be finite and nonnegative")
    actual = (exposure if config.curriculum_exposure_override is None
              else config.curriculum_exposure_override)
    if config.staged_curriculum:
        total = config.curriculum_total_intervals
        r = (actual - .0625 * total) / (.3125 * total)
        q = smoothstep5(r) if r <= .5 else 1. - smoothstep5(1. - r)
        if q == 0.:
            return config.barrier_sigma_easy
        if q == 1.:
            return config.barrier_sigma_hard
    else:
        q = smoothstep5((actual / config.curriculum_total_intervals - .1) / .7)
    return math.exp((1 - q) * math.log(config.barrier_sigma_easy)
                    + q * math.log(config.barrier_sigma_hard))


def path_lateral(layout: dict[str, Any]) -> float:
    sx, sy = layout["start"]
    tx, ty = layout["target"]
    return sy + (ty - sy) * (layout["obstacle"][0] - sx) / (tx - sx)


def direct_path_blocked(layout: dict[str, Any], *, clearance_m: float = CLEARANCE_M) -> bool:
    """SAT intersection of a swept oriented vehicle rectangle and the barrier.

    The vehicle follows the start-target line with its full length and wheel
    width. Clearance expands each vehicle half-extent; a grazing approach is
    blocked. This is not a point-ray visibility test.
    """
    if not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("clearance must be finite and nonnegative")
    start, target, obstacle = layout["start"], layout["target"], layout["obstacle"]
    dx, dy = target[0] - start[0], target[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("start and target must differ")
    forward, lateral = (dx / length, dy / length), (-dy / length, dx / length)
    for ax, ay in ((1., 0.), (0., 1.), forward, lateral):
        vehicle_extent = ((VEHICLE_HALF_LENGTH + clearance_m)
                          * abs(ax * forward[0] + ay * forward[1])
                          + (VEHICLE_HALF_WIDTH + clearance_m)
                          * abs(ax * lateral[0] + ay * lateral[1]))
        a, b = ax * start[0] + ay * start[1], ax * target[0] + ay * target[1]
        center = ax * obstacle[0] + ay * obstacle[1]
        extent = BARRIER_HALF_LENGTH * abs(ax) + BARRIER_HALF_WIDTH * abs(ay)
        if max(a, b) + vehicle_extent < center - extent or min(a, b) - vehicle_extent > center + extent:
            return False
    return True


def feasible_routes(layout: dict[str, Any], config: SimulatorConfig) -> tuple[str, ...]:
    """Admit at least one wide turning passage, never require both sides.

    Check walls, endpoint footprints/FOV, and room before/after the barrier
    for a radius-sized turn at *both* endpoint gains. The screen does not use
    exposure, domain override, policy actions or task outcomes.
    """
    sx, sy = layout["start"]
    tx, ty = layout["target"]
    ox, oy = layout["obstacle"]
    footprint = math.hypot(VEHICLE_HALF_LENGTH, VEHICLE_HALF_WIDTH) + CLEARANCE_M
    if not (footprint < sx < ox - BARRIER_HALF_LENGTH < ox + BARRIER_HALF_LENGTH < tx < 3.6 - footprint
            and max(abs(sy), abs(ty)) < WALL_Y - footprint
            and abs(oy) + BARRIER_HALF_WIDTH <= WALL_Y):
        return ()
    if abs(math.atan2(ty - sy, tx - sx) - layout["yaw"]) >= math.radians(45):
        return ()
    passages = {"left": WALL_Y - (oy + BARRIER_HALF_WIDTH),
                "right": oy - BARRIER_HALF_WIDTH + WALL_Y}
    routes = []
    for side, width in passages.items():
        for alpha in (0., 1.):
            angle = config.calibration.steering_limit_rad * steering_gain(alpha)
            if not 0 < angle < math.pi / 2:
                break
            radius = .20 / math.tan(angle)
            needed = radius + VEHICLE_HALF_WIDTH + CLEARANCE_M
            access = min(ox - BARRIER_HALF_LENGTH - sx, tx - ox - BARRIER_HALF_LENGTH)
            if width <= needed or access <= needed:
                break
        else:
            routes.append(side)
    return tuple(routes)


def sample_layout(seed: int, exposure: float, config: SimulatorConfig) -> dict[str, Any]:
    """Pure deterministic reset draw, with rejection diagnostics on acceptance.

    Start/target/yaw keep the legacy draw sequence. A separately named barrier
    RNG makes rejection counts unable to change those paired base coordinates.
    The returned layout is privileged administrative data, never policy input.
    """
    import numpy as np

    if type(seed) is not int or seed < 0:
        raise ValueError("layout seed must be a nonnegative integer")
    if config.layout_version != LAYOUT_VERSION:
        raise ValueError("sample_layout requires curriculum-v1")
    sigma = barrier_sigma(exposure, config)
    rng = np.random.default_rng(seed)
    mirror = 1 if rng.integers(2) else -1
    layout = {
        "start": (float(rng.uniform(.4, .6)), mirror * float(rng.uniform(-.15, .15))),
        "yaw": mirror * float(rng.uniform(-math.radians(10), math.radians(10))),
        "target": (2.9, mirror * float(rng.uniform(-.2, .2))),
        "obstacle": (BARRIER_X, 0.), "mirror": mirror,
    }
    mean = path_lateral(layout)
    low, high = config.barrier_bounds
    normal = NormalDist(mean, sigma)
    lo_cdf, hi_cdf = normal.cdf(low), normal.cdf(high)
    barrier_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x42415252]))
    for rejected in range(MAX_LAYOUT_ATTEMPTS):
        probability = float(barrier_rng.uniform(lo_cdf, hi_cdf))
        probability = min(math.nextafter(1., 0.), max(math.nextafter(0., 1.), probability))
        y = normal.inv_cdf(probability)
        layout["obstacle"] = (BARRIER_X, y)
        routes = feasible_routes(layout, config)
        if routes:
            layout["curriculum"] = {
                "layout_version": LAYOUT_VERSION, "reset_exposure_intervals": float(exposure),
                "curriculum_exposure_intervals": float(exposure if config.curriculum_exposure_override is None
                                                       else config.curriculum_exposure_override),
                "sigma_m": sigma, "path_lateral_m": mean,
                "bounds_m": list(config.barrier_bounds), "clearance_m": CLEARANCE_M,
                "rejected_proposals": rejected, "proposals": rejected + 1,
                "rejection_counts": {"no_feasible_route": rejected},
                "feasible_routes": list(routes),
                "direct_path_blocked": config.obstacle_enabled and direct_path_blocked(layout),
                "barrier_rng_stream": "SeedSequence([seed, 0x42415252])",
            }
            return layout
    raise RuntimeError(f"no feasible curriculum layout after {MAX_LAYOUT_ATTEMPTS} proposals (seed={seed})")


def geometry_audit(config: SimulatorConfig, *, seeds: Iterable[int] = range(512),
                   exposures: Iterable[float] | None = None) -> dict[str, Any]:
    """JSON-ready accepted-layout blockage/density curves; no Bullet or files.

    Rates describe geometry, not navigation success. Endpoint targets are
    reported as diagnostics, never enforced by retuning or filtering seeds.
    """
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("audit seeds must be nonempty and unique")
    if exposures is None:
        exposures = ((fraction * config.curriculum_total_intervals
                      for fraction in (0., .0625, .21875, .375, .5, .65625, .8125, 1.))
                     if config.staged_curriculum else
                     (fraction * config.curriculum_total_intervals
                      for fraction in (0., .1, .25, .45, .6, .8, 1.)))
    exposures = tuple(exposures)
    if not exposures:
        raise ValueError("audit exposures must be nonempty")
    low, high = config.barrier_bounds
    edges = [low + (high - low) * i / 20 for i in range(21)]
    rows = []
    for exposure in exposures:
        layouts = [sample_layout(seed, exposure, config) for seed in seeds]
        blocked = sum(layout["curriculum"]["direct_path_blocked"] for layout in layouts)
        rejected = sum(layout["curriculum"]["rejected_proposals"] for layout in layouts)
        histogram = [0] * 20
        for layout in layouts:
            index = min(19, int((layout["obstacle"][1] - low) / (high - low) * 20))
            histogram[index] += 1
        bin_width = (high - low) / 20
        rows.append({"exposure": float(exposure), "sigma": barrier_sigma(exposure, config),
                     "accepted": len(layouts), "rejected": rejected,
                     "rejection_rate": rejected / (len(layouts) + rejected),
                     "blocked_count": blocked, "blocked_fraction": blocked / len(layouts),
                     "clear_fraction": 1 - blocked / len(layouts),
                     "lateral_histogram_counts": histogram,
                     "lateral_density_per_m": [count / (len(layouts) * bin_width) for count in histogram]})
        if config.staged_curriculum:
            alpha = ({"A": 0., "B": 1.}[config.domain_override]
                     if config.domain_override is not None else
                     staged_drift_alpha(exposure, config.curriculum_total_intervals))
            rows[-1].update(alpha=alpha, steering_gain=steering_gain(alpha))
    return {"schema_version": 1, "settings": asdict(config), "seeds": list(seeds),
            "clearance_m": CLEARANCE_M, "lateral_histogram_edges_m": edges, "rows": rows,
            "early_clear_target": [.4, .6], "late_blocked_target": .9,
            "scripted_controller_checked": False, "navigation_success_claim": False}
