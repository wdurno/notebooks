"""Plan 19 Phase 1 Fix 1: bounded, resumable readiness repairs, never Phase 2."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

from .config import canonical_json
from .phase1 import PROJECT, Recipe, resolved
from .runtime import StopController, check_storage, source_fingerprint
from .storage import RunStore


VERSION = "phase1-fix1-v2"


def refresh():
    from .report import build_notebook
    build_notebook(PROJECT / "cache/numerical_experiment/picar")


def run(root, scene_path, *, model_path=None, offline=False, resume=False, max_hours=3.):
    if not 0 < max_hours <= 3:
        raise ValueError("Fix 1 permits at most three active compute hours")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from .learner import MemoryLearner
    from .models import load_qwen_model
    from .phase1_fix_roles import audit_interface, audit_panels, rollout
    from .phase1_fix_control import control_pilot
    from .report import load_report, read_completed_unit
    from .phase1_scene import SceneAuditConfig, run_scene_audit
    from src.robotics.simulator import SimulatorConfig

    scene_path = scene_path.resolve()
    scene = read_completed_unit(scene_path.parent)["summary"]
    if scene.get("kind") != "phase1_scene_audit":
        raise ValueError("a completed Phase 1 scene audit is required")
    recipe = Recipe(frozen_prefix_cache=True)
    config = resolved(recipe).to_dict()
    config["version"] = VERSION
    dependencies = {"phase1_fix", "phase1_fix_roles", "phase1_fix_control", "role_repair", "phase1_scene",
        "phase1", "phase1_roles", "learner", "models", "teacher", "storage", "runtime", "control",
        "config", "schemas", "picar_math", "simulator", "protocol", "actions", "geometry", "drift", "worker"}
    def fingerprint():
        return {name: value for name, value in source_fingerprint().items() if Path(name).stem in dependencies}
    sources = fingerprint()
    previous_campaigns = [u for u in load_report(PROJECT / "cache/numerical_experiment/picar/phase1")["complete_units"]
        if u["summary"].get("kind") == "phase1_fix1_campaign" and Path(u["path"]).resolve() != (root / "orchestration").resolve()]
    prior_seconds = sum(u["summary"]["total_active_seconds"] for u in previous_campaigns)
    identity = {"config": config, "phase1": {"kind": "phase1_fix1_campaign", "max_hours": max_hours},
                "scene_config_hash": json.loads((scene_path.parent / "manifest.json").read_text())["config_hash"],
                "model_source": str(model_path.resolve()) if model_path else config["model"]["model_id"],
                "prior_fix_seconds": prior_seconds,
                "source_sha256": sources}
    root.mkdir(parents=True, exist_ok=True)
    stop = StopController(root)
    stop.install()
    try:
        with RunStore(root / "orchestration", identity, resume=resume, checkpoint_policy="journal-v1") as coordinator:
            if coordinator.completed:
                return json.loads((coordinator.root / "summary.json").read_text())
            saved = coordinator.load_checkpoint(trusted=True) or {"charged_seconds": 0., "units": 0}
            charged, completed = saved["charged_seconds"], saved["units"]
            started = time.monotonic()
            def progress(status, *, report=False):
                elapsed = charged + time.monotonic() - started
                coordinator.checkpoint({"charged_seconds": elapsed, "units": completed},
                    progress={"phase": 1, "status": f"Fix 1: {status}", "active_seconds": elapsed,
                              "counts": {"step": completed}})
                if report:
                    refresh()
                if stop.check():
                    raise InterruptedError("Fix 1 checkpointed stop request")
                if elapsed + prior_seconds >= max_hours * 3600 - 30:
                    raise TimeoutError("Fix 1 compute allowance reached")
            def stopped():
                progress("readiness diagnostics")
            def unit(name, metadata, callback):
                nonlocal completed
                progress(name)
                unit_identity = {**identity, "operation": name, "settings": metadata}
                suffix = hashlib.sha256(canonical_json(unit_identity).encode()).hexdigest()[:12]
                path = root / "units" / f"{name}-{suffix}"
                with RunStore(path, unit_identity, resume=resume, checkpoint_policy="journal-v1") as store:
                    if store.completed:
                        return json.loads((path / "summary.json").read_text())
                    t0 = time.monotonic()
                    payload = callback(store)
                    result = {**payload, "schema_version": 1, "phase": 1, "smoke_only": True,
                        "kind": f"phase1_fix1_{name}", "active_seconds": time.monotonic() - t0,
                        "settings": metadata}
                    if fingerprint() != sources:
                        raise RuntimeError("Fix 1 dependencies changed; preserve attempt and version the repair")
                    store.checkpoint({"stage": "done"}, progress={"phase": 1, "status": f"Fix 1 {name} complete",
                                                                 "counts": {"step": 1}})
                    store.complete(result)
                completed += 1
                progress(f"{name} complete", report=True)
                return result

            progress("loading actual Qwen", report=True)
            check_storage(root, quota_bytes=100_000_000_000, reserve_bytes=20_000_000_000)
            if not torch.cuda.is_available():
                raise RuntimeError("Fix 1 requires actual CUDA, never a fake-model fallback")
            torch.set_num_threads(2)
            kwargs = dict(seed=191101, top_layers=1, hidden_size=64, ltm_rank=4, stm_rank=1,
                context_length=8, max_tokens=2048, min_pixels=56 * 56, max_pixels=19200,
                local_files_only=offline or model_path is not None)
            if model_path is not None:
                kwargs["model_path"] = model_path
            model = load_qwen_model(**kwargs).eval()
            model.encoder.enable_prefix_cache = True
            audit = unit("interface", {}, lambda s: audit_interface(model, recipe, scene, s, stopped))
            panels, developments = {}, {}
            selected = None
            if audit["parity_pass"]:
                for variant in ("legacy", "grounded-low", "grounded-visual-v2", "generated-visual-v2"):
                    panel = unit("panels", {"variant": variant},
                        lambda s, v=variant: audit_panels(model, recipe, scene, v, s, stopped))
                    panels[variant] = panel
                # Two bounded skill rescue attempts. Panels are diagnostics,
                # never demonstrations; actual closed-loop performance decides.
                for variant in ("grounded-visual-v2", "generated-visual-v2"):
                    episodes = []
                    for seed, domain in ((4, "A"), (5, "A"), (4, "B"), (5, "B")):
                        episode = unit("teacher", {"variant": variant, "seed": seed, "domain": domain,
                                                   "split": "development"},
                            lambda s, v=variant, k=seed, d=domain: rollout(model, recipe, v, k, d, s, stopped))
                        episodes.append(episode)
                    developments[variant] = episodes
                    a = [e for e in episodes if e["domain"] == "A"]
                    if sum(e["success"] for e in a) / len(a) >= .7 and panels[variant]["language_contrast_pass"]:
                        selected = variant
                        break
            validation = []
            if selected:
                for seed in range(101, 121):
                    validation.append(unit("teacher", {"variant": selected, "seed": seed,
                        "domain": "A", "split": "readiness-validation"},
                        lambda s, k=seed: rollout(model, recipe, selected, k, "A", s, stopped)))

            learner = MemoryLearner(model, resolved(recipe).learning, resolved(recipe).seeds(0))
            initial = learner.state_dict()
            controls, scenes = [], []
            for period in (.75, 1., 1.25):
                def check_scene(store, p=period):
                    output = store.root / f"payload-{time.time_ns()}"
                    return run_scene_audit(output, seeds=(4, 5, 6, 7, 8), config=SceneAuditConfig(
                        simulator=SimulatorConfig(width=160, height=120, realtime=False, control_interval_s=p),
                        controller_period_s=p, recording_period_s=p, anticipatory_braking=True))
                scenes.append(unit("scene", {"period_s": period}, check_scene))
                for cooperative in (False, True):
                    candidate = replace(recipe, context_length=4)
                    settings = {"period_s": period, "context_length": 4, "cooperative": cooperative}
                    controls.append(unit("control", settings,
                        lambda s, p=period, c=cooperative, r=candidate: control_pilot(model, r, initial, s, stopped,
                                                                       period=p, cooperative=c)))
            valid_teacher = bool(validation) and sum(e["success"] for e in validation) / len(validation) >= .7
            valid_reward = bool(selected) and panels[selected]["reward_order_pass"]
            valid_control = any(c["deadline_pass"] and any(
                p["all_feasible"] and p["settings"]["period_s"] == c["period_s"] for p in scenes) for c in controls)
            blockers = []
            if not audit["parity_pass"]:
                blockers.append("Native/custom multimodal parity audit failed; repair the model path before learning.")
            if not valid_teacher:
                blockers.append("A competent language-responsive Qwen teacher is not validated on 20 fresh layouts.")
            if not valid_reward:
                blockers.append("Visual shaping potential is not validated; no-shaping remains an unselected versioned fallback.")
            if not valid_control:
                blockers.append("No profiled context/period meets both deadline and scripted-controllability guides.")
            elapsed = charged + time.monotonic() - started
            children = sum(json.loads(p.read_text()).get("active_seconds", 0)
                           for p in (root / "units").glob("*/summary.json"))
            summary = {"schema_version": 1, "phase": 1, "smoke_only": True,
                "kind": "phase1_fix1_campaign", "status": "ready_for_checkin", "parity_pass": audit["parity_pass"],
                "teacher_variant": selected, "teacher_validation_successes": sum(e["success"] for e in validation),
                "teacher_validation_count": len(validation), "reward_pass": valid_reward,
                "control_pass": valid_control, "blockers": blockers,
                "controls": [{k: c[k] for k in ("period_s", "context_length", "cooperative", "fallback_count",
                    "eligible_intervals", "fallback_fraction", "deadline_pass", "model_access")} for c in controls],
                "total_active_seconds": elapsed, "active_seconds": max(0., elapsed - children),
                "cumulative_fix_seconds": elapsed + prior_seconds,
                "scientific_phases_started": False, "model_provenance": model.encoder.provenance,
                "configuration": asdict(recipe)}
            coordinator.checkpoint({"charged_seconds": elapsed, "units": completed},
                progress={"phase": 1, "status": "Fix 1 ready for check-in", "counts": {"step": completed}})
            coordinator.complete(summary)
            refresh()
            return summary
    finally:
        stop.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True, help="completed scene-audit summary.json")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-hours", type=float, default=3.)
    args = parser.parse_args(argv)
    if args.resume:
        (args.output / "STOP").unlink(missing_ok=True)
    result = run(args.output, args.scene, model_path=args.model_path, offline=args.offline,
                 resume=args.resume, max_hours=args.max_hours)
    print(json.dumps({k: v for k, v in result.items() if k != "model_provenance"}, indent=2))


if __name__ == "__main__":
    main()
