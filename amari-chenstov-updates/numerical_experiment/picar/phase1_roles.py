"""Frozen RGB role sanity and real-time control diagnostics for Plan 19 Phase 1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .phase1 import PROJECT, Recipe, _histories, _identity, resolved
from .storage import RunStore


INSTRUCTION = "Go around the blue obstacle, approach the red ball, and stop."


def _batch(rows, recipe):
    import numpy as np
    from PIL import Image
    import torch
    rows = rows[-recipe.context_length:]
    images = []
    for row in rows:
        value = row["rgb"]
        image = Image.fromarray(value).resize((recipe.image_width, recipe.image_height))
        images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255)
    return {"images": torch.stack(images)[None],
        "executed_actions": torch.tensor([row["action"] for row in rows], dtype=torch.float32)[None],
        "elapsed": torch.tensor([row["elapsed"] for row in rows], dtype=torch.float32)[None, :, None],
        "contexts": [[{"text": INSTRUCTION, "issued_at": 0.}]],
        "start_time": torch.tensor([max(0., rows[0]["time"] - rows[0]["elapsed"])])}


def run_roles(output, recipe, scene, *, model_path=None, offline=False, resume=False):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", recipe.allocator)
    import numpy as np
    from PIL import Image
    import torch
    from src.robotics.actions import ACTION_NAMES, Calibration
    from src.robotics.protocol import Control
    from src.robotics.simulator import PyBulletBackend, SimulatorConfig
    from src.robotics.worker import SimulatorWorker
    from .control import ControlSession
    from .learner import LearningBatch, MemoryLearner
    from .models import load_qwen_model
    from .runtime import ModelAccess, sha256_file
    from .teacher import FrozenQwenTeacher, FrozenQwenEvaluator

    identity = _identity(recipe, model_path, "phase1_roles")
    identity["scene_payload"] = scene["output"]
    identity["role_probe_version"] = 1
    identity["role_source_sha256"] = sha256_file(Path(__file__))
    with RunStore(output, identity, resume=resume) as store:
        if store.completed:
            return json.loads((output / "summary.json").read_text())
        started = time.monotonic()
        torch.set_num_threads(2)
        kwargs = dict(top_layers=recipe.top_layers, hidden_size=recipe.hidden_size,
            ltm_rank=recipe.ltm_rank, stm_rank=recipe.stm_rank,
            context_length=recipe.context_length, max_tokens=recipe.max_tokens,
            min_pixels=56 * 56, max_pixels=recipe.max_pixels,
            local_files_only=offline or model_path is not None, seed=191101)
        if model_path is not None:
            kwargs["model_path"] = model_path
        model = load_qwen_model(**kwargs).eval()
        model.encoder.enable_prefix_cache = recipe.frozen_prefix_cache
        teacher = FrozenQwenTeacher(model.encoder, primitive_names=ACTION_NAMES)
        evaluator = FrozenQwenEvaluator(model.encoder)
        panels = []
        for panel in scene["reward_panels"]:
            # Explicit whitelist: no truth, geometry, panel ID, or desired label
            # enters either role's tensor input or language instruction.
            inputs = panel["model_input"]
            times = inputs["timestamps_s"]
            rows = [{"rgb": np.array(Image.open(path).convert("RGB")),
                "time": t, "elapsed": .5 if i == 0 else max(1 / 240, t - times[i - 1]),
                "action": (0., math.radians(-8), 0., 0.)}
                for i, (path, t) in enumerate(zip(inputs["rgb_paths"], times))]
            batch = _batch(rows, recipe)
            model.encoder.clear_caches()
            t = time.monotonic()
            potential = float(evaluator.evaluate(batch)[0])
            distribution = teacher.distribution(batch, probability_floor=.001)
            control = Calibration().map_distribution(distribution.executed_probabilities[0].tolist())
            panels.append({"panel_id": panel["panel_id"], "state": panel["state"],
                "domain": panel["domain"], "truth_verified": panel["truth_verified"],
                "potential": potential, "teacher_control": asdict(control),
                "teacher_probabilities": distribution.executed_probabilities[0].cpu().tolist(),
                "seconds": time.monotonic() - t})
        store.checkpoint({"panels": panels}, progress={"phase": 1, "status": "RGB role panels complete", "counts": {"step": 1}})
        episodes = []
        for seed, domain in ((4, "A"), (5, "A"), (4, "B"), (5, "B")):
            with PyBulletBackend(SimulatorConfig(width=recipe.image_width, height=recipe.image_height,
                    control_interval_s=.5, realtime=False), run_id=f"role-{seed}-{domain}") as sim:
                observation = sim.reset(seed=seed, exposure_intervals=0 if domain == "A" else 500)
                rows, commands = [], []
                t0 = time.monotonic()
                while True:
                    rows.append({"rgb": np.frombuffer(observation.rgb, dtype=np.uint8).reshape(
                        observation.height, observation.width, 3).copy(), "time": observation.capture_time_s,
                        "elapsed": .5, "action": tuple(asdict(observation.executed_controls).values())})
                    batch = _batch(rows, recipe)
                    with torch.no_grad():
                        distribution = teacher.distribution(batch, probability_floor=.001)
                    q = distribution.executed_probabilities[0].cpu()
                    control = Calibration().map_distribution(q.tolist())
                    result = sim.step(control, training=False)
                    commands.append({"time_s": sim.episode_time_s, "probabilities": q.tolist(),
                        "control": asdict(control), "end_reason": result.end_reason})
                    observation = result.observation
                    if result.terminated:
                        break
                episodes.append({"seed": seed, "domain": domain, "success": result.end_reason == "success",
                    "end_reason": result.end_reason, "physical_seconds": sim.episode_time_s,
                    "wall_seconds": time.monotonic() - t0, "commands": commands,
                    "geometry": sim.diagnostics()["geometry"], "accelerated": True,
                    "policy_privileged": False})
                store.checkpoint({"panels": panels, "episodes": episodes}, progress={"phase": 1,
                    "status": "frozen teacher rollout", "counts": {"step": 1 + len(episodes)}})

        base = resolved(recipe)
        learner = MemoryLearner(model, base.learning, base.seeds(0))
        history = _histories(recipe)
        # Warm only hardware kernels; no live control observation is fabricated.
        learner.choose_action(history, explore=False)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        model.encoder.clear_caches()
        learner.choose_action(history, explore=False)
        torch.cuda.synchronize()
        cold_s = time.monotonic() - t0
        period = max(.5, math.ceil(cold_s * 1.5 * 4) / 4)
        period = math.ceil(period * 240) / 240
        live_rows, lock = [], threading.RLock()
        def history_builder(observation):
            row = {"rgb": np.frombuffer(observation.rgb, dtype=np.uint8).reshape(
                observation.height, observation.width, 3).copy(), "time": observation.capture_time_s,
                "elapsed": period, "action": tuple(asdict(observation.executed_controls).values())}
            with lock:
                if not live_rows or row["time"] > live_rows[-1]["time"]:
                    live_rows.append(row)
                visible = [r for r in live_rows if r["time"] <= observation.capture_time_s]
                live_rows[:] = live_rows[-recipe.context_length * 3:]
            return _batch(visible, recipe)
        live = []
        updates = []
        worker = SimulatorWorker(SimulatorConfig(width=recipe.image_width, height=recipe.image_height,
            control_interval_s=period, realtime=True), run_id="phase1-realtime")
        session = ControlSession(worker, ModelAccess(), learner=learner, history_builder=history_builder,
                                 explore=False, pending_capacity=128, own_worker=True)
        try:
            session.start(seed=191107, training=False)
            while True:
                try:
                    interval = session.next_interval(timeout=30)
                except StopIteration:
                    break
                live.append({"index": interval.index, "fallback": interval.fallback_reason,
                    "timing": interval.timing, "end_reason": interval.result.end_reason,
                    "physical_seconds": interval.result.observation.physical_time_s,
                    "terminated": interval.result.terminated})
                if len(live) in (4, 10) and not interval.result.terminated:
                    replay = LearningBatch([history] * recipe.batch, torch.zeros(recipe.batch),
                        torch.full((recipe.batch,), 10), torch.full((recipe.batch, 7), .1), torch.arange(recipe.batch))
                    new_history = {**history, "elapsed": history["elapsed"] + .125}
                    fresh = LearningBatch([new_history], torch.zeros(1), torch.tensor([10]),
                                          torch.full((1, 7), .1), torch.tensor([128]))
                    t0 = time.monotonic()
                    session.update(new=fresh, replay=replay, acquired_count=256)
                    torch.cuda.synchronize()
                    updates.append(time.monotonic() - t0)
                if interval.result.terminated:
                    break
            clock = worker.diagnostics()
        finally:
            session.close()
        eligible = [row for row in live if row["fallback"] != "startup"]
        misses = sum(row["fallback"] is not None for row in eligible)
        fraction = misses / len(eligible) if eligible else 1.
        successes = [row["potential"] for row in panels if row["state"] == "stopped_success"]
        failures = [row["potential"] for row in panels if row["state"] == "far_deadline_failure"]
        reward_order = min(successes) > max(failures)
        teacher_a = [row for row in episodes if row["domain"] == "A"]
        teacher_rate = sum(row["success"] for row in teacher_a) / len(teacher_a)
        blockers = []
        if teacher_rate < .7:
            blockers.append("Frozen Qwen teacher did not demonstrate useful A skill in this small readiness pilot.")
        if not reward_order:
            blockers.append("Frozen RGB potential did not consistently rank success above far failure.")
        if fraction > .1:
            blockers.append("Real-time inference/update pilot missed more than 10% of non-startup control intervals.")
        summary = {"schema_version": 1, "phase": 1, "smoke_only": True, "kind": "phase1_roles",
            "recipe": asdict(recipe), "readiness_pass": not blockers, "blockers": blockers,
            "panels": panels, "teacher_episodes": episodes, "teacher_A_success_rate": teacher_rate,
            "reward_success_above_far": reward_order,
            "control_intervals": live, "control_update_seconds": updates,
            "nonstartup_fallback_fraction": fraction, "control_clock": clock,
            "recommended_control_period_s": period, "active_seconds": time.monotonic() - started,
            "teacher_provenance": teacher.provenance, "evaluator_provenance": evaluator.provenance,
            "limitations": ["Two A and two B teacher episodes are readiness diagnostics, not a reliable success-rate estimate.",
                "Control updates use diagnostic labels solely for timing; live student skill is not measured.",
                "The independent-clock pilot contains two updates; it does not admit one update for every acquired interval.",
                "No physical robot calibration or hardware execution."]}
        store.checkpoint({"stage": "done"}, progress={"phase": 1, "status": "roles complete", "counts": {"step": 6}})
        store.complete(summary)
        return summary


def launch_roles(root, recipe, scene, *, model_path=None, offline=False, resume=False):
    path = root / "units" / "roles-and-control"
    if (path / "COMPLETED").exists():
        return json.loads((path / "summary.json").read_text())
    scene_path = root / "units" / "scene-audit" / "summary.json"
    command = [sys.executable, "-u", "-m", "numerical_experiment.picar.phase1_roles",
        "--output", str(path), "--recipe", json.dumps(asdict(recipe)), "--scene", str(scene_path)]
    if model_path:
        command += ["--model-path", str(model_path)]
    if offline:
        command.append("--offline")
    if resume and path.exists():
        command.append("--resume")
    log = root / "logs" / f"roles-{time.time_ns()}.log"
    with log.open("xb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900,
            env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    if result.returncode:
        raise RuntimeError(f"role/control probe failed; inspect {log}")
    return json.loads((path / "summary.json").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_roles(args.output, Recipe(**json.loads(args.recipe)), json.loads(args.scene.read_text()),
        model_path=args.model_path, offline=args.offline, resume=args.resume)
    print(json.dumps({key: result[key] for key in ("readiness_pass", "blockers", "active_seconds")}, indent=2))


if __name__ == "__main__":
    main()
