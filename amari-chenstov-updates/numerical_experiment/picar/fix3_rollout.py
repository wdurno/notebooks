"""Fixed diagnostic layouts, teacher rollouts and privileged feasibility only."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
import time

from demo.picar.speech import InstructionLedger
from src.robotics.actions import Calibration
from src.robotics.curriculum import direct_path_blocked
from src.robotics.simulator import PyBulletBackend, SimulatorConfig
from .fix2_data import TASK, history_batch, observation_row
from .fix2_roles import _video
from .fix3_roles import CenteringTeacher


def simulator_config(*, easy=False, period=1.):
    return SimulatorConfig(width=160, height=120, realtime=True, control_interval_s=period,
        task_version="ball-contact-v1", layout_version="curriculum-v1",
        total_training_intervals=320, curriculum_total_intervals=320,
        curriculum_exposure_override=0. if easy else None, domain_override="A" if easy else None)


def evaluation_configs():
    # Explicitly fixed geometry strata, never selected from learned outcomes.
    base = simulator_config()
    return [{"difficulty": name,
             "config": replace(base, curriculum_exposure_override=exposure, domain_override=domain),
             "seeds": (195803, 195807)}
            for name, exposure, domain in (("easy", 0., "A"), ("middle", 144., "A"), ("hard", 320., "B"))]


def teacher_rollout(model, store, stopped, *, variant, seed):
    teacher = CenteringTeacher(model.encoder, variant=variant)
    config = replace(simulator_config(easy=True, period=.5), realtime=False, width=320, height=240)
    saved = store.load_checkpoint(trusted=True)
    with PyBulletBackend(config, run_id=f"fix3-teacher-{variant}-{seed}") as sim:
        if saved:
            sim.restore(saved["simulator"])
            observation, rows, commands, statuses = (saved[k] for k in ("observation", "rows", "commands", "statuses"))
            ledger = InstructionLedger.from_snapshot(saved["instructions"])
            deadline = saved["deadline"]
        else:
            observation = sim.reset(seed=seed, exposure_intervals=0.)
            rows, commands, statuses = [], [], []
            ledger = InstructionLedger(clock_domain="simulator_seconds")
            ledger.typed("initial-task", TASK, at=sim.time_s)
            deadline = sim.time_s + config.deadline_s
        while not sim.diagnostics()["end_reason"]:
            row = observation_row(observation, previous_time=rows[-1]["time"] if rows else None)
            rows.append(row)
            history = history_batch(rows, instructions=ledger, deadline=deadline, statuses=statuses)
            before = time.monotonic()
            distribution = teacher.distribution(history, probability_floor=.001)
            q = distribution.executed_probabilities[0].tolist()
            control = Calibration().map_distribution(q)
            top = store.frames.put(sim.render_topdown())
            frame = store.frames.put(row["rgb"])
            result = sim.step(control, training=False)
            commands.append({"time_s": row["time"], "frame_ref": frame, "topdown_ref": top,
                "controls": asdict(control), "probabilities": q,
                "seconds": time.monotonic() - before, "raw_text": teacher.last_diagnostic["raw_text"][0],
                "end_reason": result.end_reason})
            status = teacher.last_diagnostic["statuses"][0]
            if status:
                statuses.append({"text": status, "kind": "robot_status", "available_at": sim.time_s})
            rows, statuses = rows[-4:], statuses[-4:]
            observation = result.observation
            store.checkpoint({"simulator": sim.snapshot(), "observation": observation,
                "rows": rows, "commands": commands, "statuses": statuses,
                "instructions": ledger.snapshot(), "deadline": deadline}, progress={
                    "phase": 1, "status": f"Fix 3 teacher {variant}", "counts": {"step": len(commands)}})
            stopped()
        reason = sim.diagnostics()["end_reason"]
        videos = []
        if seed == 194601 and commands:
            final = {"time_s": sim.time_s, "frame_ref": store.frames.put(sim.render()),
                "topdown_ref": store.frames.put(sim.render_topdown()), "end_reason": reason,
                "controls": asdict(Calibration().neutral())}
            videos.append(_video(store, [*commands, final], seed))
        action_counts = {}
        for command in commands:
            action = max(range(8), key=command["probabilities"].__getitem__)
            action_counts[str(action)] = action_counts.get(str(action), 0) + 1
        return {"variant": variant, "seed": seed, "success": reason == "success", "end_reason": reason,
            "commands": commands, "action_counts": action_counts, "accelerated": True,
            "difficulty": "easy", "layout": sim.diagnostics()["layout"],
            "physical_seconds": sim.episode_time_s, "teacher_provenance": teacher.provenance, "videos": videos}


def scripted_geometry_checks(config):
    """Contact-directed version of the existing privileged pursuit diagnostic."""
    import numpy as np
    from .phase1_scene import PrivilegedFeasibilityController, SceneAuditConfig
    rows = []
    for exposure in (0., 144., 320.):
        for domain in ("A", "B"):
            cfg = replace(config, realtime=False, control_interval_s=.5, width=80, height=60,
                curriculum_exposure_override=exposure, domain_override=domain)
            for seed in (194601, 194602):
                with PyBulletBackend(cfg, run_id=f"fix3-script-{exposure}-{domain}-{seed}") as sim:
                    sim.reset(seed=seed, exposure_intervals=0.)
                    initial = sim.diagnostics()
                    controller = PrivilegedFeasibilityController(initial, SceneAuditConfig(simulator=cfg, make_videos=False))
                    layout = initial["layout"]
                    sx, sy = layout["start"]
                    tx, ty = layout["target"]
                    ox, oy = layout["obstacle"]
                    if not direct_path_blocked(layout):
                        controller.waypoints = np.asarray([(sx, sy), (tx, ty)])
                    else:
                        side = 1 if oy <= 0 else -1
                        controller.waypoints = np.asarray([(sx, sy), (ox - .55, oy + side * .4),
                            (ox + .12, oy + side * .6), (tx, ty)])
                    while not sim.diagnostics()["end_reason"]:
                        state = sim.diagnostics()
                        # This task seeks contact, so disable safe-stop braking.
                        state = {**state, "geometry": {**state["geometry"], "clearance_m": 1.}}
                        control, _ = controller.action(state)
                        sim.step(control, training=False)
                    final = sim.diagnostics()
                    rows.append({"exposure": exposure, "domain": domain, "seed": seed,
                        "blocked": layout["curriculum"]["direct_path_blocked"],
                        "success": final["end_reason"] == "success", "end_reason": final["end_reason"],
                        "physical_seconds": sim.episode_time_s})
    return {"rows": rows, "privileged": True, "used_for_demonstrations": False,
            "interpretation": "A controller failure does not prove geometric impossibility."}
