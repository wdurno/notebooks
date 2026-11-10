"""Bounded generated-reward and teacher diagnostics for Phase 1 Fix 2."""

from __future__ import annotations

from dataclasses import asdict
import time

import torch
from PIL import Image

from demo.picar.speech import InstructionLedger

from .dense_roles import ConversationalTeacher, GeneratedVisualReward
from .fix2_data import TASK, history_batch, observation_row, panel_history
from src.robotics.actions import Calibration
from src.robotics.simulator import PyBulletBackend, SimulatorConfig


def _video(store, frames, seed):
    from .report import assemble_video
    rows = []
    for frame in frames:
        paths = []
        for key in ("frame_ref", "topdown_ref"):
            path = store.root / (frame[key][7:] + ".png")
            if not path.exists():
                Image.fromarray(store.frames.get(frame[key])).save(path)
            paths.append(str(path))
        rows.append({"time_s": frame["time_s"], "ego_path": paths[0], "topdown_path": paths[1],
            "instruction": TASK, "outcome": frame["end_reason"],
            "requested_control": str(frame["controls"]), "executed_control": "recorded action mixture"})
    return assemble_video(rows, store.root / f"teacher-{time.time_ns()}.mp4", seed=seed, fps=5.)


def reward_panel(model, source, panel_store, rubric, store, stopped):
    scorer = GeneratedVisualReward(model.encoder, rubric=rubric)
    saved = store.load_checkpoint(trusted=True) or {"rows": []}
    rows = saved["rows"]
    for panel in source["panels"][len(rows):]:
        history = panel_history(panel, panel_store.frames)
        before = time.monotonic()
        try:
            value = float(scorer.scores(history)[0])
            error = None
        except ValueError as exc:
            value, error = None, str(exc)
        rows.append({**panel, "score": value, "error": error,
                     "seconds": time.monotonic() - before, "diagnostic": scorer.last_diagnostic})
        store.checkpoint({"rows": rows}, progress={"phase": 1, "status": f"Fix 2 {rubric} scores",
            "counts": {"step": len(rows), "target": len(source["panels"])}})
        stopped()
    pairs = []
    for seed in sorted({r["seed"] for r in rows}):
        by_state = {r["state"]: r["score"] for r in rows if r["seed"] == seed}
        for a, b in (("far", "intermediate"), ("intermediate", "close"), ("far", "close")):
            x, y = by_state[a], by_state[b]
            pairs.append({"seed": seed, "farther": a, "nearer": b, "valid": x is not None and y is not None,
                          "correct": x is not None and y is not None and x < y,
                          "tie": x is not None and y is not None and x == y})
    valid = all(row["error"] is None for row in rows)
    present = [r["score"] for r in rows if r["state"] in ("intermediate", "close") and r["score"] is not None]
    absent = [r["score"] for r in rows if r["state"] == "absent" and r["score"] is not None]
    separation = bool(absent and present) and sum(absent) / len(absent) < sum(present) / len(present)
    fraction = sum(p["correct"] for p in pairs) / len(pairs)
    return {"rubric": rubric, "split": source["split"], "rows": rows, "pairs": pairs,
        "valid": valid, "separation_pass": separation, "order_fraction": fraction,
        "reward_pass": valid and separation and fraction >= .8,
        "score_provenance": scorer.provenance, "validation_layouts": sorted({r["seed"] for r in rows})}


def instruction_contrasts(model, history):
    results = []
    for variant in ("ported", "grounded"):
        teacher = ConversationalTeacher(model.encoder, variant=variant)
        rows = []
        for text in ("Stop driving now.", "Drive forward steering left.", "Drive forward steering right."):
            changed = {**history, "contexts": [[{"text": text, "issued_at": 0.}]]}
            try:
                q = teacher.distribution(changed, probability_floor=.001).executed_probabilities[0]
                rows.append({"text": text, "probabilities": q.tolist(), "error": None,
                             "diagnostic": teacher.last_diagnostic})
            except ValueError as exc:
                rows.append({"text": text, "error": str(exc), "diagnostic": teacher.last_diagnostic})
        passed = (all(r["error"] is None for r in rows) and rows[0]["probabilities"][7] > .8
            and rows[1]["probabilities"][0] > rows[1]["probabilities"][1]
            and rows[2]["probabilities"][1] > rows[2]["probabilities"][0])
        results.append({"variant": variant, "rows": rows, "pass": passed})
    return {"conditions": results}


def rollout(model, *, variant, seed, domain, obstacle, store, stopped, period=.5):
    teacher = ConversationalTeacher(model.encoder, variant=variant)
    config = SimulatorConfig(realtime=False, width=320, height=240, control_interval_s=period,
                             task_version="ball-contact-v1", obstacle_enabled=obstacle)
    saved = store.load_checkpoint(trusted=True)
    with PyBulletBackend(config, run_id=f"fix2-{variant}-{seed}-{domain}-{int(obstacle)}") as sim:
        if saved:
            sim.restore(saved["simulator"])
            observation, rows, commands, statuses = (
                saved[k] for k in ("observation", "rows", "commands", "statuses"))
            instructions = InstructionLedger.from_snapshot(saved["instructions"])
            deadline = saved["deadline"]
        else:
            observation = sim.reset(seed=seed, exposure_intervals=0. if domain == "A" else 500.)
            rows, commands, statuses = [], [], []
            instructions = InstructionLedger(clock_domain="simulator_seconds")
            instructions.typed("initial-task", TASK if obstacle else "Drive toward the red ball until contact.",
                               at=sim.time_s)
            deadline = sim.time_s + config.deadline_s
        while not sim.diagnostics()["end_reason"]:
            row = observation_row(observation, previous_time=rows[-1]["time"] if rows else None)
            topdown = sim.render_topdown()
            rows.append(row)
            history = history_batch(rows, instructions=instructions, deadline=deadline, statuses=statuses)
            t0 = time.monotonic()
            try:
                distribution = teacher.distribution(history, probability_floor=.001)
            except ValueError as exc:
                return {"variant": variant, "seed": seed, "domain": domain, "obstacle": obstacle,
                    "split": "development", "success": False, "end_reason": "invalid_teacher_output",
                    "error": str(exc), "commands": commands, "accelerated": True,
                    "teacher_diagnostic": teacher.last_diagnostic}
            q = distribution.executed_probabilities[0].tolist()
            control = Calibration().map_distribution(q)
            result = sim.step(control, training=False)
            frame = store.frames.put(row["rgb"])
            commands.append({"time_s": row["time"], "frame_ref": frame, "probabilities": q,
                "topdown_ref": store.frames.put(topdown),
                "controls": asdict(control), "seconds": time.monotonic() - t0,
                "raw_text": teacher.last_diagnostic["raw_text"][0], "end_reason": result.end_reason})
            status = teacher.last_diagnostic["statuses"][0]
            if status:
                statuses.append({"text": status, "kind": "robot_status", "available_at": sim.time_s})
            statuses = statuses[-4:]
            rows = rows[-4:]
            observation = result.observation
            store.checkpoint({"simulator": sim.snapshot(), "observation": observation,
                "rows": rows, "commands": commands, "statuses": statuses,
                "instructions": instructions.snapshot(), "deadline": deadline},
                progress={"phase": 1, "status": f"Fix 2 teacher {variant}", "counts": {"step": len(commands)}})
            stopped()
        reason = sim.diagnostics()["end_reason"]
        videos = []
        if seed == 601 and commands:
            final = {"time_s": sim.time_s, "frame_ref": store.frames.put(sim.render()),
                "topdown_ref": store.frames.put(sim.render_topdown()), "end_reason": reason,
                "controls": asdict(Calibration().neutral())}
            videos = [_video(store, [*commands, final], seed)]
        return {"variant": variant, "seed": seed, "domain": domain, "obstacle": obstacle,
            "success": reason == "success", "end_reason": reason, "commands": commands,
            "physical_seconds": sim.episode_time_s, "accelerated": True, "coaching": "unassisted",
            "teacher_provenance": teacher.provenance, "task_version": config.task_version, "videos": videos}
