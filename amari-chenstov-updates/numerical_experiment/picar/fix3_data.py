"""Administrative image panels for the frozen Fix 3 reward audit."""

from __future__ import annotations

from dataclasses import replace
import math

from src.robotics.actions import Calibration
from src.robotics.simulator import PyBulletBackend, SimulatorConfig
from .fix2_data import history_batch
from .fix3_roles import CenteringReward


PANEL_POSES = (
    ("far-center", 2.1, 0., 0.),
    ("middle-center", 1., 0., 0.),
    ("middle-left", 1., 20., 0.),
    ("middle-right", 1., -20., 0.),
    ("middle-edge", 1., 36., 0.),
    ("pan-centered", 1., 20., -20.),
    ("detour", 1., 25., 0.),
    ("close", .42, 0., 0.),
    ("crop", .18, 0., 0.),
    ("absent", 1., 180., 0.),
)


def render_panels(store, *, split):
    if split not in {"development", "validation"}:
        raise ValueError("unknown reward split")
    saved = store.load_checkpoint(trusted=True) or {"rows": []}
    rows = saved["rows"]
    seeds = (9211, 9212) if split == "development" else (9311, 9312)
    poses = [(seed, pose) for seed in seeds for pose in PANEL_POSES]
    for seed, (name, distance, yaw_deg, pan_deg) in poses[len(rows):]:
        config = SimulatorConfig(width=320, height=240, realtime=False,
            task_version="ball-contact-v1", obstacle_enabled=False)
        with PyBulletBackend(config, run_id=f"fix3-panel-{split}-{seed}-{name}") as sim:
            sim.reset(seed=seed, exposure_intervals=0. if seed % 2 else 500.)
            p, cid = sim._p, sim.client_id
            tx, ty = sim.diagnostics()["layout"]["target"]
            position, _ = p.getBasePositionAndOrientation(sim.vehicle_id, physicsClientId=cid)
            # Only the panel administrator uses geometry; model inputs are RGB.
            yaw = math.radians(yaw_deg if seed % 2 else -yaw_deg)
            pan = math.radians(pan_deg if seed % 2 else -pan_deg)
            p.resetBasePositionAndOrientation(sim.vehicle_id, (tx - distance, ty, position[2]),
                p.getQuaternionFromEuler((0., 0., yaw)), physicsClientId=cid)
            sim._control = replace(Calibration().neutral(), pan_rad=pan)
            if name == "detour":
                sim._box((.10, .20, .08), (tx - .55, ty + .20, .08), (.1, .25, .75, 1))
            rows.append({"panel_id": f"{split}-{seed}-{name}", "seed": seed,
                "split": split, "state": name, "pan_rad": pan,
                "frame_ref": store.frames.put(sim.render()), "distance_m": distance,
                "privileged_admin": True, "model_input_whitelist": ["RGB"]})
        store.checkpoint({"rows": rows}, progress={"phase": 1, "status": "Fix 3 reward images",
            "counts": {"step": len(rows), "target": len(poses)}})
    return {"rows": rows, "split": split, "count": len(rows)}


def score_panels(model, source, frames, store, stopped):
    scorer = CenteringReward(model.encoder)
    saved = store.load_checkpoint(trusted=True) or {"rows": [], "scorer": None}
    if saved["scorer"]:
        scorer.load_state_dict(saved["scorer"])
    rows = saved["rows"]
    for panel in source["rows"][len(rows):]:
        rgb = frames.get(panel["frame_ref"])
        history = history_batch([{"rgb": rgb, "time": 0., "elapsed": 0.,
            "action": (panel["pan_rad"], math.radians(-8), 0., 0.)}],
            contexts=[], deadline=20.)
        value = scorer.score(history, panel["pan_rad"])
        # Own the image as well as the score so the report needs no inference.
        own_ref = store.frames.put(rgb)
        rows.append({**panel, **value, "frame_ref": own_ref})
        store.checkpoint({"rows": rows, "scorer": scorer.state_dict()}, progress={
            "phase": 1, "status": "Fix 3 frozen reward scoring",
            "counts": {"step": len(rows), "target": len(source["rows"])}})
        stopped()
    comparisons = []
    for seed in sorted({r["seed"] for r in rows}):
        by_state = {r["state"]: r for r in rows if r["seed"] == seed}
        for name in ("middle-left", "middle-right", "middle-edge"):
            a, b = by_state["middle-center"], by_state[name]
            comparisons.append({"seed": seed, "contrast": "center-vs-" + name,
                "centering_difference": a["centering"] - b["centering"],
                "proximity_difference": a["proximity"] - b["proximity"],
                "score_difference": a["score"] - b["score"]})
    centering_fraction = sum(r["centering_difference"] > 0 for r in comparisons) / max(1, len(comparisons))
    approach = []
    for seed in sorted({r["seed"] for r in rows}):
        b = {r["state"]: r for r in rows if r["seed"] == seed}
        for far, near in (("far-center", "middle-center"), ("middle-center", "close"), ("far-center", "close")):
            approach.append({"seed": seed, "far": far, "near": near,
                "correct": b[far]["score"] < b[near]["score"], "tie": b[far]["score"] == b[near]["score"]})
    return {"rows": rows, "split": source["split"], "comparisons": comparisons,
        "centering_order_fraction": centering_fraction, "approach_pairs": approach,
        "approach_order_fraction": sum(r["correct"] for r in approach) / max(1, len(approach)),
        "absent_zero": all(r["score"] == 0 for r in rows if r["state"] == "absent"),
        "provenance": scorer.provenance}
