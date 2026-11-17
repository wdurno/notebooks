"""Predeclared RGB-only measurement panels with privileged renderer-mask truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np

from src.robotics.simulator import PyBulletBackend, SimulatorConfig, vertical_fov_deg


@dataclass(frozen=True)
class Scene:
    name: str
    distance_m: float
    bearing_deg: float = 0.
    domain: str = "A"
    barrier: str = "none"
    yaw_deg: float = 0.
    tilt_deg: float = -8.
    visible: bool = True


SCENES = (
    Scene("No ball: rear view", 1.1, yaw_deg=180., visible=False),
    Scene("No ball: side view", 1.35, domain="B", yaw_deg=90., visible=False),
    Scene("Ball fully hidden", 1., barrier="full", visible=False),
    Scene("Tiny, centered", 2.4),
    Scene("Far, left", 2.1, bearing_deg=22.),
    Scene("Far, right", 2., bearing_deg=-22., domain="B"),
    Scene("Far, centered", 1.65, domain="B"),
    Scene("Medium, left", 1.35, bearing_deg=22.),
    Scene("Medium, centered", 1.),
    Scene("Medium, right", .9, bearing_deg=-25., domain="B"),
    Scene("Medium, blue beside", 1.1, bearing_deg=12., domain="B", barrier="beside"),
    Scene("Medium, partly hidden", 1., barrier="partial_left"),
    Scene("Close, centered", .65, domain="B", tilt_deg=-3.),
    Scene("Close, left", .5, bearing_deg=20.),
    Scene("Close, right", .42, bearing_deg=-20., domain="B"),
    Scene("Close, blue beside", .55, bearing_deg=-10., barrier="beside"),
    Scene("Close, partly hidden", .5, domain="B", barrier="partial_right"),
    Scene("Large, centered", .3, domain="B"),
    Scene("Very close, centered", .175, domain="B"),
    Scene("Large, edge-cropped", .19, bearing_deg=30.),
)


def target_mask(segmentation, target_id):
    values = np.asarray(segmentation, dtype=np.int64)
    # Bullet packs object ID into the low 24 bits; background is -1.
    return (values >= 0) & ((values & ((1 << 24) - 1)) == target_id)


def mask_statistics(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("nonempty two-dimensional visibility mask required")
    y, x = np.nonzero(mask)
    count = len(x)
    return {"visible": bool(count), "visible_pixels": count,
        "width_percent": 100. * (int(x.max() - x.min() + 1) if count else 0) / mask.shape[1],
        "left_pixel": int(x.min()) if count else None,
        "right_pixel": int(x.max()) if count else None,
        "center_x_percent": 100. * float(x.mean()) / mask.shape[1] if count else None,
        "center_y_percent": 100. * float(y.mean()) / mask.shape[0] if count else None}


def render_with_mask(sim):
    """Use the simulator's actual camera vectors and audit RGB equivalence."""
    original = sim._camera
    captured = []

    def camera(eye, target, up):
        p, cfg = sim._p, sim.config
        view = p.computeViewMatrix(eye, target, up)
        projection = p.computeProjectionMatrixFOV(
            vertical_fov_deg(cfg.horizontal_fov_deg, cfg.width, cfg.height), cfg.width / cfg.height, .015, 8.)
        result = p.getCameraImage(cfg.width, cfg.height, viewMatrix=view, projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=sim.client_id)
        captured.append(target_mask(np.asarray(result[4]).reshape(cfg.height, cfg.width), sim.target_id))
        return np.asarray(result[2], dtype=np.uint8).reshape(cfg.height, cfg.width, 4)[..., :3].copy()

    sim._camera = camera
    try:
        rgb = sim.render()
    finally:
        sim._camera = original
    if not np.array_equal(rgb, sim.render()):
        raise ValueError("mask-enabled render differs from the policy RGB image")
    return rgb, captured[0]


def render_scene(spec, index):
    from dataclasses import replace

    seed = 291001 + index
    config = SimulatorConfig(realtime=False, width=320, height=240,
                             task_version="ball-contact-v1", obstacle_enabled=False)
    with PyBulletBackend(config, run_id=f"width-benchmark-{index + 1:02d}") as sim:
        sim.reset(seed=seed, exposure_intervals=0. if spec.domain == "A" else 500.)
        p, cid = sim._p, sim.client_id
        tx, ty = sim.diagnostics()["layout"]["target"]
        z = p.getBasePositionAndOrientation(sim.vehicle_id, physicsClientId=cid)[0][2]
        bearing, yaw = math.radians(spec.bearing_deg), math.radians(spec.yaw_deg)
        position = (tx - spec.distance_m * math.cos(bearing), ty - spec.distance_m * math.sin(bearing), z)
        p.resetBasePositionAndOrientation(sim.vehicle_id, position, p.getQuaternionFromEuler((0., 0., yaw)),
                                         physicsClientId=cid)
        sim._control = replace(sim._control, tilt_rad=math.radians(spec.tilt_deg))
        _, unobstructed = render_with_mask(sim)
        if spec.barrier != "none":
            eye = np.asarray(position) + np.asarray((.06 * math.cos(yaw), .06 * math.sin(yaw), .18))
            middle = eye + .5 * (np.asarray((tx, ty, .1)) - eye)
            offsets = {"full": (0., .22), "beside": (.28, .09),
                       "partial_left": (.1, .1), "partial_right": (-.1, .1)}
            offset, half_width = offsets[spec.barrier]
            sim._box((.035, half_width, .24), (float(middle[0]), float(middle[1]) + offset, .24),
                     (.05, .1, .8, 1.), collision=False)
        rgb, mask = render_with_mask(sim)
        truth = mask_statistics(mask)
        if truth["visible"] != spec.visible:
            raise ValueError(f"predeclared visibility mismatch for {spec.name}")
        unoccluded_pixels = int(unobstructed.sum())
        fraction = float(mask.sum()) / unoccluded_pixels if unoccluded_pixels else None
        if spec.barrier.startswith("partial") and not (fraction is not None and .1 < fraction < .9):
            raise ValueError(f"partial occluder did not partially hide {spec.name}: {fraction}")
        truth.update(unobstructed_width_percent=mask_statistics(unobstructed)["width_percent"],
                     visible_area_fraction=fraction)
    return rgb, mask, {"panel_id": f"C{index + 1:02d}", "state": spec.name,
                       "scene_layout_seed": seed, "scene": asdict(spec), "truth": truth}


def render_contact_sheet(path, panels, frames, rows=None):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(15., 10.8), layout="constrained")
    for panel, axis in zip(panels, figure.subplots(4, 5).flat):
        axis.imshow(frames.get(panel["frame_ref"]))
        axis.axis("off")
        title = f"{panel['panel_id']}: {panel['state']}\nMask width {panel['truth']['width_percent']:.1f}%"
        if rows is not None:
            row = next(item for item in rows if item["panel_id"] == panel["panel_id"])
            title += (f" | Qwen {row['width_percent']:.1f}%\nReward {row['reward']:.2f} / 10"
                      if "reward" in row else "\nInvalid response")
        axis.set_title(title, fontsize=9)
    figure.savefig(Path(path), dpi=160)
