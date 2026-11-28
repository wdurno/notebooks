"""Bounded teaching/evaluation diagnostics, excluded from learner evidence."""

from dataclasses import asdict, replace
import math
import time

import numpy as np
import torch

from src.robotics.protocol import Control, RobotCommand
from src.robotics.simulator import PyBulletBackend
from numerical_experiment.picar.fix2_data import TASK, history_batch, observation_row
from numerical_experiment.picar_head_awac.config import simulator
from .coach import GeometricCoach, chassis_yaw
from .config import seed
from .roles import CoachingTeacher as CenteringTeacher


def _command(sim, observation, control, sequence, seconds=.5):
    command = RobotCommand(sim.run_id, observation.session_id, sequence, observation.observation_id,
                           control, sim.time_s, seconds, sim.time_s + seconds)
    receipt = sim.command(command)
    if not receipt.accepted:
        raise RuntimeError("readiness command was rejected: " + receipt.safety_status)
    return receipt, sim.advance(round(seconds / sim.config.physics_dt_s), training=False)


def _diagnostic_controls(text, calibration):
    """Privileged audit only; never a demonstrator or learner label."""
    if text is None or text.startswith("Stop"):
        return calibration.neutral()
    if "reverse" in text.lower() or "backward" in text.lower():
        return Control(0., calibration.tilt_neutral_rad, 0., -calibration.reverse_limit_mps)
    turn = "left" if "steer left" in text.lower() or "steering left" in text.lower() else (
        "right" if "steer right" in text.lower() or "steering right" in text.lower() else None)
    return Control(0., calibration.tilt_neutral_rad,
                   -calibration.steering_limit_rad if turn == "left" else calibration.steering_limit_rad if turn == "right" else 0.,
                   calibration.forward_limit_mps)


def geometry_audit(store, check):
    state = store.load_checkpoint(trusted=True) or {"rows": []}
    cases = [(domain, layout) for domain in ("initial_A", "moderate_A") for layout in range(6)]
    for domain, layout_seed in cases[len(state["rows"]):]:
        cfg = replace(simulator(fixed=domain), realtime=False)
        coach = GeometricCoach()
        maneuvers, collisions, minimum_clearance = [], [], float("inf")
        with PyBulletBackend(cfg, run_id=f"fix6-geometry-{domain}-{layout_seed}") as sim:
            observation = sim.reset(seed=layout_seed)
            for sequence in range(12):
                before = sim.diagnostics()
                proposal = coach.propose(before, cfg)
                controls = _diagnostic_controls(proposal["text"], cfg.calibration)
                receipt, result = _command(sim, observation, controls, sequence)
                after = sim.diagnostics()
                maneuvers.append({"text": proposal["text"], "reason": proposal["reason"],
                    "controls": asdict(controls), "yaw_change": math.atan2(
                        math.sin(chassis_yaw(after) - chassis_yaw(before)),
                        math.cos(chassis_yaw(after) - chassis_yaw(before)))})
                collisions.append(bool(after["unsafe_contact"]))
                minimum_clearance = min(minimum_clearance, float(after["geometry"]["clearance_m"]))
                observation = result.observation
                if result.terminated:
                    break
            row = {"environment": domain, "seed": layout_seed, "maneuvers": maneuvers,
                   "collision": any(collisions), "minimum_clearance_m": minimum_clearance,
                   "end_reason": sim.diagnostics()["end_reason"]}
        state["rows"].append(row)
        store.checkpoint(state, progress={"stage": "actual-motion geometry audit", "n": len(state["rows"]), "total": len(cases)})
        check()
    return {"ready": not any(r["collision"] for r in state["rows"]), "rows": state["rows"],
            "diagnostic_only": True, "learning_samples": 0,
            "qualification": "Twelve initial/moderate layouts plus unit-test wall/reverse cases; not a universal safety proof."}


def compliance_audit(model, scorer, store, check):
    state = store.load_checkpoint(trusted=True) or {"rows": []}
    # Each panel has matched correct/opposing turns, stops and reverses, plus ambiguity.
    cases = [
        ("Steer left while driving forward.", "left", "adherent"),
        ("Steer left while driving forward.", "right", "violated"),
        ("Steer right while driving forward.", "right", "adherent"),
        ("Steer right while driving forward.", "left", "violated"),
        ("Stop driving now.", "stop", "adherent"),
        ("Stop driving now.", "forward", "violated"),
        ("Reverse straight backward briefly.", "reverse", "adherent"),
        ("Reverse straight backward briefly.", "forward", "violated"),
        ("Steer left while driving forward.", "ambiguous", "uncertain"),
        ("Steer right while driving forward.", "ambiguous", "uncertain"),
    ]
    for index in range(len(state["rows"]), 60):
        panel = "development" if index < 30 else "audit"
        text, action, expected = cases[index % len(cases)]
        cfg = replace(simulator(fixed="initial_A"), realtime=False)
        with PyBulletBackend(cfg, run_id=f"fix6-compliance-{index}") as sim:
            observation = sim.reset(seed=seed(f"compliance:correction-v3:{index}"))
            before = np.frombuffer(observation.rgb, dtype=np.uint8).reshape(observation.height, observation.width, 3)
            calibration = cfg.calibration
            control = _diagnostic_controls(text, calibration)
            if action in ("left", "right", "forward", "reverse", "stop", "ambiguous"):
                control = Control(0., calibration.tilt_neutral_rad,
                    -calibration.steering_limit_rad if action == "left" else calibration.steering_limit_rad if action == "right" else 0.,
                    -calibration.reverse_limit_mps if action == "reverse" else calibration.forward_limit_mps if action in ("left", "right", "forward") else 0.)
            sim.advance(round(.25 / cfg.physics_dt_s), training=False)
            receipt, result = _command(sim, observation, control, 0, seconds=1.)
            after = np.frombuffer(result.observation.rgb, dtype=np.uint8).reshape(observation.height, observation.width, 3)
            evidence = {"window_id": f"{panel}-{index}", "start_time": 0., "end_time": result.observation.capture_time_s,
                "instructions": [{"instruction_id": f"cue-{index}", "text": text, "issued_at": 0.,
                                  "available_at": 0., "ended_at": result.observation.capture_time_s,
                                  "completed_at": None,
                                  "status": "inactive" if action == "ambiguous" else "active"}],
                "status": "inactive" if action == "ambiguous" else "expired",
                "controls": [] if action == "ambiguous" else [{"start_time": receipt.received_at_s,
                    "end_time": result.observation.capture_time_s, "executed": list(asdict(receipt.executed_controls).values()),
                    "fallback": False, "instruction_ids_seen": [f"cue-{index}"], "source": "accepted_command"}],
                "eligible_instruction_ids": [] if action == "ambiguous" else [f"cue-{index}"],
                "image_times": {"before": observation.capture_time_s, "after": result.observation.capture_time_s}}
            before_ref, after_ref = store.frames.put(before), store.frames.put(after)
            tick = time.monotonic()
            judgment = scorer.score(evidence, before, after)
            seconds = time.monotonic() - tick
            state["rows"].append({"index": index, "panel": panel, "expected": expected,
                "judgment": judgment["judgment"], "correct": judgment["judgment"] == expected,
                "seconds": seconds, "evidence": evidence, "before_ref": before_ref, "after_ref": after_ref,
                "reason": judgment["reason"], "cache_key": judgment["cache_key"]})
        store.checkpoint(state, progress={"stage": "Qwen compliance audit", "n": len(state["rows"]), "total": 60,
                                         "audit_correct": sum(r["correct"] for r in state["rows"] if r["panel"] == "audit")})
        check()
    clear = [r for r in state["rows"] if r["panel"] == "audit" and r["expected"] != "uncertain"]
    correct = sum(r["correct"] for r in clear)
    confusion = {expected: {observed: sum(r["expected"] == expected and r["judgment"] == observed
        for r in state["rows"] if r["panel"] == "audit")
        for observed in ("adherent", "violated", "uncertain")}
        for expected in ("adherent", "violated", "uncertain")}
    return {"ready": correct / len(clear) >= .9, "rows": state["rows"], "confusion": confusion,
            "audit_clear_correct": correct, "audit_clear_total": len(clear), "scorer": scorer.provenance,
            "median_scoring_seconds": sorted(r["seconds"] for r in state["rows"])[30], "learning_samples": 0}


def teacher_audit(model, width_scorer, compliance, store, check):
    state = store.load_checkpoint(trusted=True) or {"rows": []}
    teacher = CenteringTeacher(model.encoder)
    # Geometry-only selection: two left, two right and two straight requests.
    # No teacher outcomes were used to select the added right-passage layout.
    cases = [(condition, layout) for layout in (0, 1, 2, 3, 5, 7) for condition in ("U", "G")]
    for condition, layout_seed in cases[len(state["rows"]):]:
        cfg = replace(simulator(fixed="initial_A"), realtime=False)
        with PyBulletBackend(cfg, run_id=f"fix6-teacher-{condition}-{layout_seed}") as sim:
            observation = sim.reset(seed=layout_seed)
            diagnostic = sim.diagnostics()
            proposal = GeometricCoach().propose(diagnostic, cfg)
            text = proposal["text"] if condition == "G" else None
            contexts = [{"text": TASK, "issued_at": 0.}]
            if text:
                contexts.append({"text": text, "issued_at": 0.})
            history = history_batch([observation_row(observation)], contexts=contexts, deadline=cfg.deadline_s,
                                    width=cfg.width, height=cfg.height)
            tick = time.monotonic()
            target = teacher.distribution(history, probability_floor=.001)
            inference_seconds = time.monotonic() - tick
            controls = cfg.calibration.map_distribution(target.executed_probabilities[0].tolist())
            expected_controls = _diagnostic_controls(proposal["text"], cfg.calibration)
            receipt, result = _command(sim, observation, controls, 0, seconds=1.)
            after = sim.diagnostics()
            yaw = math.atan2(math.sin(chassis_yaw(after) - chassis_yaw(diagnostic)),
                             math.cos(chassis_yaw(after) - chassis_yaw(diagnostic)))
            before_ref = store.frames.put(np.frombuffer(observation.rgb, dtype=np.uint8).reshape(observation.height, observation.width, 3))
            after_ref = store.frames.put(np.frombuffer(result.observation.rgb, dtype=np.uint8).reshape(observation.height, observation.width, 3))
            state["rows"].append({"condition": condition, "seed": layout_seed, "cue": text,
                "proposed_cue": proposal["text"], "teacher_action": int(target.executed_probabilities.argmax()),
                "controls": asdict(controls), "yaw_degrees": math.degrees(yaw),
                "expected_steering_rad": expected_controls.steering_rad,
                "inference_seconds": inference_seconds, "collision": bool(after["unsafe_contact"]),
                "before_ref": before_ref, "after_ref": after_ref, "diagnostic": copy_diagnostic(teacher.last_diagnostic)})
        store.checkpoint(state, progress={"stage": "paired coached teacher turns", "n": len(state["rows"]), "total": 12})
        check()
    g = [r for r in state["rows"] if r["condition"] == "G"]
    left = [r for r in g if r["expected_steering_rad"] < -.05
            and r["controls"]["steering_rad"] < -.05 and r["yaw_degrees"] > 1.]
    right = [r for r in g if r["expected_steering_rad"] > .05
             and r["controls"]["steering_rad"] > .05 and r["yaw_degrees"] < -1.]
    timely = sum(r["inference_seconds"] < .9 * cfg.control_interval_s for r in state["rows"])
    return {"ready": bool(left and right) and timely >= 10, "rows": state["rows"], "correct_left_turns": len(left),
            "correct_right_turns": len(right), "teacher": teacher.provenance,
            "timely_calls": timely, "total_calls": len(state["rows"]), "control_period_s": cfg.control_interval_s,
            "qualification": "Six paired one-action diagnostic trials with coaching enabled; not task mastery or the 50% collection schedule.",
            "learning_samples": 0}


def copy_diagnostic(value):
    import copy
    return copy.deepcopy(value)


def resource_profile(model, geometry, audit, teacher, store, check):
    model_seconds = [r["inference_seconds"] for r in teacher["rows"]]
    median_teacher = sorted(model_seconds)[len(model_seconds) // 2]
    scorer_seconds = audit["median_scoring_seconds"] if audit is not None else None
    # Baseline measured 1.65--1.71h/path; this includes extra panels and coaching.
    forecast = {"remaining_hours": [16., 23.], "initialization_hours": [3., 4.5],
        "development_hours": [3.8, 4.8], "assessment_hours": [8., 10.],
        "qualification": "Readiness-role measurements; reforecast after collection and the development pair."}
    result = {"median_teacher_seconds": median_teacher, "median_compliance_seconds": scorer_seconds,
              "cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "adherence_enabled": audit is not None,
              "forecast": forecast, "ready": geometry["ready"] and (audit is None or audit["ready"]) and teacher["ready"]}
    store.checkpoint(result, progress={"stage": "resource forecast", **result})
    check()
    return result
