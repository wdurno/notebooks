"""Causal Fix 2 histories and privileged RGB-panel administration only."""

from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
from PIL import Image
import torch

from src.robotics.simulator import PyBulletBackend, SimulatorConfig


TASK = "Drive toward the red ball until contact. Avoid blue obstacles and walls."
PANEL_STATES = ("absent", "far", "intermediate", "close", "occluded", "too_close")


def history_batch(rows, *, contexts=None, deadline, statuses=(), length=4, width=320, height=240,
                  instructions=None, speech_bridge=None):
    rows = rows[-length:]
    if not rows:
        raise ValueError("a history requires an observed frame")
    if instructions is not None:
        if contexts is not None:
            raise ValueError("choose a timestamped ledger or explicit historical context")
        if speech_bridge is not None:
            speech_bridge.drain(instructions)
        as_of = rows[-1].get("decision_time", rows[-1]["time"])
        contexts = instructions.context(as_of)
        statuses = [*statuses, *instructions.model_status_events(as_of)]
    elif speech_bridge is not None:
        raise ValueError("speech input requires the same persistent instruction ledger")
    if contexts is None:
        raise ValueError("explicit causal instructions are required")
    images = [torch.from_numpy(np.array(Image.fromarray(r["rgb"]).resize((width, height))))
              .permute(2, 0, 1).float() / 255 for r in rows]
    elapsed = [r["elapsed"] for r in rows]
    return {"images": torch.stack(images)[None],
        "executed_actions": torch.tensor([r["action"] for r in rows], dtype=torch.float32)[None],
        "elapsed": torch.tensor(elapsed, dtype=torch.float32)[None, :, None],
        "start_time": torch.tensor([max(0., rows[0]["time"] - elapsed[0])]),
        "contexts": [list(contexts)], "status_events": [list(statuses)[-8:]],
        "episode_deadline": torch.tensor([deadline]),
        "decision_time": torch.tensor([[r.get("decision_time", r["time"]) for r in rows]])}


def observation_row(observation, *, previous_time=None):
    return {"rgb": np.frombuffer(observation.rgb, dtype=np.uint8).reshape(
        observation.height, observation.width, 3).copy(), "time": observation.capture_time_s,
        "elapsed": 0. if previous_time is None else observation.capture_time_s - previous_time,
        "action": tuple(asdict(observation.executed_controls).values())}


def panel_history(panel, frames):
    rgb = frames.get(panel["frame_ref"])
    return history_batch([{"rgb": rgb, "time": 0., "elapsed": 0.,
        "action": (0., math.radians(-8), 0., 0.)}],
        contexts=[{"text": TASK, "issued_at": 0.}], deadline=20.)


def make_panels(store, *, split):
    """Administrative poses/occlusions are labels for auditing, never policy inputs."""
    if split not in {"development", "validation"}:
        raise ValueError("unknown panel split")
    saved = store.load_checkpoint(trusted=True) or {"panels": []}
    panels = saved["panels"]
    seeds = range(211, 215) if split == "development" else range(311, 315)
    pending = [(seed, state) for seed in seeds for state in PANEL_STATES]
    for seed, state in pending[len(panels):]:
        domain = "A" if seed % 2 else "B"
        config = SimulatorConfig(realtime=False, width=320, height=240,
                                 task_version="ball-contact-v1", obstacle_enabled=False)
        with PyBulletBackend(config, run_id=f"fix2-panel-{seed}") as sim:
            sim.reset(seed=seed, exposure_intervals=0. if domain == "A" else 500.)
            p, cid = sim._p, sim.client_id
            target = sim.diagnostics()["layout"]["target"]
            tx, ty = target[:2]
            distance = {"absent": 1.1, "far": 2.1, "intermediate": 1.,
                        "close": .42, "occluded": 1., "too_close": .18}[state]
            yaw = math.pi if state == "absent" else 0.
            old_position, _ = p.getBasePositionAndOrientation(sim.vehicle_id, physicsClientId=cid)
            p.resetBasePositionAndOrientation(sim.vehicle_id, (tx - distance, ty, old_position[2]),
                p.getQuaternionFromEuler((0., 0., yaw)), physicsClientId=cid)
            if state == "occluded":
                shape = p.createVisualShape(p.GEOM_BOX, halfExtents=(.08, .22, .25),
                    rgbaColor=(.05, .1, .8, 1.), physicsClientId=cid)
                p.createMultiBody(baseMass=0., baseVisualShapeIndex=shape,
                    basePosition=(tx - .45, ty, .25), physicsClientId=cid)
            rgb = sim.render()
            frame_ref = store.frames.put(rgb)
            panels.append({"panel_id": f"{split}-{seed}-{state}", "seed": seed, "split": split,
                "state": state, "domain": domain, "frame_ref": frame_ref,
                "distance_to_target_center_m": distance, "privileged_admin": True,
                "model_input_whitelist": ["RGB"]})
        store.checkpoint({"panels": panels}, progress={"phase": 1, "status": f"Fix 2 {split} RGB panels",
            "counts": {"step": len(panels), "target": len(pending)}})
    return {"split": split, "panels": panels, "count": len(panels),
            "learner_or_teacher_demonstrations": False}
