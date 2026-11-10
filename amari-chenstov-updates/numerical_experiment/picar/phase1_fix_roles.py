"""Artifact-producing RGB-only diagnostics for Phase 1 Fix 1."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .phase1_roles import _batch
from .role_repair import (GroundedEvaluator, GroundedTeacher, last_features,
                         native_hidden, native_text, prepare_role, vocabulary_scores)
from .teacher import FrozenQwenEvaluator, FrozenQwenTeacher
from src.robotics.actions import ACTION_NAMES, Calibration
from src.robotics.simulator import PyBulletBackend, SimulatorConfig


def panel_batch(panel, recipe, *, high_resolution=True):
    source = panel["model_input"]
    times = source["timestamps_s"]
    rows = [{"rgb": np.array(Image.open(path).convert("RGB")), "time": stamp,
             "elapsed": .5 if i == 0 else max(1 / 240, stamp - times[i - 1]),
             "action": (0., math.radians(-8), 0., 0.)}
            for i, (path, stamp) in enumerate(zip(source["rgb_paths"], times))]
    shape = replace(recipe, image_width=320, image_height=240) if high_resolution else recipe
    return _batch(rows, shape)


def audit_interface(model, recipe, scene, store, stopped):
    encoder = model.encoder
    teacher = FrozenQwenTeacher(encoder, primitive_names=ACTION_NAMES)
    saved = store.load_checkpoint(trusted=True) or {"rows": []}
    rows = saved["rows"]
    panels = [p for p in scene["reward_panels"] if p["domain"] == "A"]
    for panel in panels[len(rows):]:
        batch = panel_batch(panel, recipe)
        inputs = prepare_role(encoder, batch, teacher.prompt, frames=1, max_pixels=76800)
        encoder.clear_caches()
        split = encoder.encode_processed(inputs, role="teacher")
        native = native_hidden(encoder, inputs)
        relative = float((split - native).double().norm() / native.double().norm().clamp_min(1e-12))
        probs, diagnostic = vocabulary_scores(encoder, last_features(split, inputs), tuple(map(str, range(8))))
        prompt = ("Describe where the red ball and blue block are in this camera image. "
                  "Is the path straight ahead clear? Do not give motor commands.")
        visual = prepare_role(encoder, batch, prompt, frames=1, max_pixels=76800)
        t0 = time.monotonic()
        description = native_text(encoder, visual, max_new_tokens=48)[0]
        torch.cuda.synchronize()
        rows.append({"panel_id": panel["panel_id"], "state": panel["state"],
            "relative_hidden_l2": relative, "max_hidden_error": float((split - native).abs().max()),
            "parity_pass": relative < 2e-4, "custom_digit_weights": probs[0].cpu().tolist(),
            **diagnostic, "native_description": description,
            "description_seconds": time.monotonic() - t0,
            "input_tokens": int(inputs["attention_mask"].sum()),
            "vision_grid": inputs["image_grid_thw"].cpu().tolist()})
        store.checkpoint({"rows": rows}, progress={"phase": 1, "status": "native interface audit",
                         "counts": {"step": len(rows), "target": len(panels)}})
        stopped()
    return {"rows": rows, "parity_pass": all(row["parity_pass"] for row in rows),
        "precision": "BF16 frozen prefix; matched FP32 adapted suffix/rotary and readout",
        "diagnostic_only": True}


def make_roles(encoder, name):
    if name in ("grounded-visual-v2", "generated-visual-v2"):
        return GroundedTeacher(encoder, mode="generated" if name.startswith("generated") else "restricted",
            frames=3, max_pixels=76800, prompt_version=2), GroundedEvaluator(encoder)
    if name == "legacy":
        return FrozenQwenTeacher(encoder, primitive_names=ACTION_NAMES), FrozenQwenEvaluator(encoder)
    if name == "grounded-low":
        return GroundedTeacher(encoder), GroundedEvaluator(encoder, frames=1, max_pixels=19200)
    if name == "grounded-visual":
        return GroundedTeacher(encoder, frames=3, max_pixels=76800), GroundedEvaluator(encoder)
    if name == "generated-visual":
        return GroundedTeacher(encoder, mode="generated", frames=3, max_pixels=76800), GroundedEvaluator(encoder)
    raise ValueError("unknown versioned role recipe")


def audit_panels(model, recipe, scene, variant, store, stopped):
    teacher, evaluator = make_roles(model.encoder, variant)
    saved = store.load_checkpoint(trusted=True) or {"rows": [], "contrasts": []}
    rows, contrasts = saved["rows"], saved["contrasts"]
    for panel in scene["reward_panels"][len(rows):]:
        batch = panel_batch(panel, recipe, high_resolution=variant != "legacy")
        record = {"panel_id": panel["panel_id"], "state": panel["state"], "domain": panel["domain"]}
        model.encoder.clear_caches()
        t0 = time.monotonic()
        record["potential"] = float(evaluator.evaluate(batch)[0])
        record["evaluator_diagnostic"] = getattr(evaluator, "last_diagnostic", {})
        record["evaluator_seconds"] = time.monotonic() - t0
        t0 = time.monotonic()
        try:
            distribution = teacher.distribution(batch, probability_floor=.001)
            q = distribution.executed_probabilities[0].cpu().tolist()
            record.update(probabilities=q, controls=asdict(Calibration().map_distribution(q)),
                          target_kind=distribution.target_kind, teacher_error=None)
        except ValueError as exc:
            record["teacher_error"] = str(exc)
        record["teacher_seconds"] = time.monotonic() - t0
        record["teacher_diagnostic"] = getattr(teacher, "last_diagnostic", {})
        rows.append(record)
        store.checkpoint({"rows": rows, "contrasts": contrasts}, progress={"phase": 1,
            "status": f"{variant} RGB panels", "counts": {"step": len(rows), "target": len(scene["reward_panels"]) + 3}})
        stopped()
    for instruction in ("Stop the drive immediately.", "Drive forward turning left.", "Drive forward turning right.")[len(contrasts):]:
        batch = panel_batch(scene["reward_panels"][0], recipe, high_resolution=variant != "legacy")
        batch["contexts"] = [[{"text": instruction, "issued_at": 0.}]]
        record = {"instruction": instruction}
        try:
            q = teacher.distribution(batch, probability_floor=.001).executed_probabilities[0].cpu().tolist()
            record.update(probabilities=q, controls=asdict(Calibration().map_distribution(q)), error=None)
        except ValueError as exc:
            record["error"] = str(exc)
        contrasts.append(record)
        store.checkpoint({"rows": rows, "contrasts": contrasts}, progress={"phase": 1,
            "status": f"{variant} instruction contrasts", "counts": {"step": len(rows) + len(contrasts),
                                                                       "target": len(rows) + 3}})
        stopped()
    success = [r["potential"] for r in rows if r["state"] == "stopped_success"]
    unsafe = [r["potential"] for r in rows if r["state"] in ("far_deadline_failure", "obstacle_collision")]
    valid = all(not row.get("teacher_error") for row in rows) and all(not row.get("error") for row in contrasts)
    language_pass = valid and (contrasts[0]["probabilities"][7] > .8
        and contrasts[1]["probabilities"][0] > contrasts[1]["probabilities"][1]
        and contrasts[2]["probabilities"][1] > contrasts[2]["probabilities"][0])
    return {"variant": variant, "rows": rows, "contrasts": contrasts,
        "teacher_output_valid": valid, "language_contrast_pass": language_pass,
        "reward_order_pass": min(success) > max(unsafe),
        "teacher_provenance": teacher.provenance, "evaluator_provenance": evaluator.provenance}


def rollout(model, recipe, variant, seed, domain, store, stopped, *, period=.5):
    """Accelerated teacher skill diagnostic; no learner fit or task-label creation."""
    teacher, _ = make_roles(model.encoder, variant)
    high = variant != "legacy"
    cfg = SimulatorConfig(width=320 if high else recipe.image_width,
        height=240 if high else recipe.image_height, control_interval_s=period, realtime=False)
    saved = store.load_checkpoint(trusted=True)
    with PyBulletBackend(cfg, run_id=f"fix1-{variant}-{seed}-{domain}") as sim:
        if saved and "simulator" in saved:
            sim.restore(saved["simulator"])
            observation = saved["observation"]
            rows, commands = saved["rows"], saved["commands"]
        else:
            observation = sim.reset(seed=seed, exposure_intervals=0 if domain == "A" else 500)
            rows, commands = [], []
        while not sim.diagnostics().get("end_reason"):
            rows.append({"rgb": np.frombuffer(observation.rgb, dtype=np.uint8).reshape(
                observation.height, observation.width, 3).copy(), "time": observation.capture_time_s,
                "elapsed": period, "action": tuple(asdict(observation.executed_controls).values())})
            batch = _batch(rows, replace(recipe, image_width=cfg.width, image_height=cfg.height))
            t0 = time.monotonic()
            try:
                distribution = teacher.distribution(batch, probability_floor=.001)
            except ValueError as exc:
                return {"variant": variant, "seed": seed, "domain": domain, "success": False,
                    "end_reason": "invalid_teacher_output", "error": str(exc), "commands": commands,
                    "policy_privileged": False, "accelerated": True}
            q = distribution.executed_probabilities[0].cpu().tolist()
            control = Calibration().map_distribution(q)
            result = sim.step(control, training=False)
            commands.append({"time_s": sim.episode_time_s, "probabilities": q,
                "controls": asdict(control), "teacher_seconds": time.monotonic() - t0,
                "target_kind": distribution.target_kind, "end_reason": result.end_reason})
            observation = result.observation
            rows = rows[-recipe.context_length:]
            store.checkpoint({"rows": rows, "commands": commands, "observation": observation,
                "simulator": sim.snapshot()}, progress={"phase": 1, "status": f"{variant} {domain} rollout",
                    "counts": {"step": len(commands)}})
            stopped()
            if result.terminated:
                break
        reason = sim.diagnostics()["end_reason"]
        return {"variant": variant, "seed": seed, "domain": domain, "success": reason == "success",
            "end_reason": reason, "physical_seconds": sim.episode_time_s, "commands": commands,
            "policy_privileged": False, "accelerated": True, "teacher_provenance": teacher.provenance}
