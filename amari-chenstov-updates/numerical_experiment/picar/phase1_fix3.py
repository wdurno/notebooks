"""Versioned, resumable Fix 3 readiness campaign; never operates hardware."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from .config import PicarConfig, canonical_json
from .phase1 import PROJECT
from .phase1_fix2 import scientific_sources, _prior_repair_seconds
from .runtime import StopController, check_storage, sha256_file
from .storage import RunStore


VERSION = "phase1-fix3-v1"


def prior_seconds(root):
    repairs = _prior_repair_seconds(root)
    diagnostics = PROJECT / "cache/numerical_experiment/picar/diagnostics"
    for path in diagnostics.glob("*/summary.json"):
        if (path.parent / "COMPLETED").exists():
            repairs += json.loads(path.read_text()).get("active_seconds", 0.)
    return repairs


def _software(store):
    result = subprocess.run([sys.executable, "-m", "pytest", "-q",
        *sorted(glob.glob("test/unit/test_picar*.py"))], cwd=PROJECT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    path = store.root / "pytest.txt"
    path.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Fix 3 software acceptance failed: {path}")
    return {"passed": True, "test_result": result.stdout.strip().splitlines()[-1],
            "test_log_sha256": sha256_file(path)}


def refresh():
    from .report import build_notebook
    build_notebook(PROJECT / "cache/numerical_experiment/picar")


def run(output, *, model_path=None, offline=False, resume=False, stage="all", max_hours=3.):
    if not 0 < max_hours <= 3.:
        raise ValueError("cumulative repairs retain the three-hour cap")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    root = Path(output).resolve()
    base = PicarConfig()
    config = replace(base, version=VERSION, mode="production", task_version="ball-contact-v1",
        image_width=160, image_height=120, transitions=320, bootstrap_transitions=300,
        model=replace(base.model, kind="qwen", hidden_size=64, context_length=4, return_bins=161),
        learning=replace(base.learning, replay_batch=1, fd_strength=0.,
            reward_version="dense-visual-centering-v1", return_min=-6., return_max=12.),
        runtime=replace(base.runtime, device="cuda", threads=2))
    config.validate()
    sources = scientific_sources()
    prior = prior_seconds(root)
    config_path = root / "orchestration/config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text())["prior_repair_seconds"]
    identity = {"version": VERSION, "config": config.to_dict(), "campaign_id": str(root),
        "model_source": str(Path(model_path).resolve()) if model_path else config.model.model_id,
        "source_sha256": sources, "prior_repair_seconds": prior, "max_hours": max_hours,
        "model_initialization_seed": 191101, "component_seeds": config.seeds(0),
        "maximum_new_hours": 2., "bootstrap_maximum": 300, "pilot_maximum_per_seed": 320,
        "pilot_seeds": [193701, 193702], "teacher_layouts": list(range(194601, 194607))}
    root.mkdir(parents=True, exist_ok=True)
    stop = StopController(root)
    stop.install()
    try:
        with RunStore(root / "orchestration", identity, resume=resume, checkpoint_policy="journal-v1") as parent:
            if parent.completed:
                return json.loads((parent.root / "summary.json").read_text())
            saved = parent.load_checkpoint(trusted=True) or {"charged_seconds": 0., "units": 0}
            charged, count = saved["charged_seconds"], saved["units"]
            started, last_report = time.monotonic(), 0.
            active_unit = None

            def progress(status, *, report=False):
                nonlocal last_report
                elapsed = charged + time.monotonic() - started
                work = {}
                if active_unit is not None and (active_unit / "progress.json").exists():
                    work = json.loads((active_unit / "progress.json").read_text())
                    status = work.get("status", status)
                    steps = work.get("counts", {})
                    status += f" ({steps.get('step', '?')}/{steps.get('target', '?')})"
                parent.checkpoint({"charged_seconds": elapsed, "units": count}, progress={
                    "phase": 1, "status": f"Fix 3: {status}", "active_seconds": elapsed,
                    "cumulative_repair_seconds": prior + elapsed,
                    "active_unit": None if active_unit is None else str(active_unit),
                    "active_unit_progress": work,
                    "remaining_estimate_seconds": max(0., 4800. - elapsed),
                    "counts": {"step": count, "target": 23}})
                if report or time.monotonic() - last_report >= 900:
                    refresh()
                    last_report = time.monotonic()
                if stop.check():
                    raise InterruptedError("Fix 3 stopped at a durable checkpoint; use --resume")
                if prior + elapsed >= max_hours * 3600 - 60 or elapsed >= 7200 - 60:
                    raise TimeoutError("Fix 3 active compute envelope reached; preserved for review")

            def stopped():
                progress("diagnostic / learner checkpoint")

            def unit(name, settings, callback):
                nonlocal count, active_unit
                progress(name)
                unit_identity = {**identity, "operation": name, "settings": settings}
                suffix = hashlib.sha256(canonical_json(unit_identity).encode()).hexdigest()[:12]
                path = root / "units" / f"{name}-{suffix}"
                active_unit = path
                with RunStore(path, unit_identity, resume=resume, checkpoint_policy="journal-v1") as store:
                    if store.completed:
                        active_unit = None
                        return json.loads((path / "summary.json").read_text())
                    before = time.monotonic()
                    payload = callback(store)
                    if scientific_sources() != sources:
                        raise RuntimeError("Fix 3 scientific sources changed; preserve and version a new campaign")
                    frames = sorted((path / "frames").glob("*.npy"))
                    if any(sha256_file(p) != p.stem for p in frames):
                        raise RuntimeError("Fix 3 frame hash mismatch before completion")
                    result = {**payload, "schema_version": 1, "phase": 1, "smoke_only": True,
                        "kind": f"phase1_fix3_{name}", "settings": payload.get("settings", settings),
                        "frame_sha256": [p.stem for p in frames],
                        "media_sha256": {p.name: sha256_file(p) for p in path.iterdir()
                                         if p.suffix in {".png", ".mp4"}},
                        "active_seconds": time.monotonic() - before, "unit_path": str(path)}
                    # Keep substantive learner checkpoints as the recovery assets.
                    if store.load_checkpoint(trusted=True) is None:
                        store.checkpoint({"stage": "done"}, progress={"phase": 1,
                            "status": f"Fix 3 {name} complete", "counts": {"step": 1}})
                    store.complete(result)
                count += 1
                active_unit = None
                print(json.dumps({"unit": name, "settings": settings,
                    "seconds": result["active_seconds"], "path": str(path)}), flush=True)
                progress(f"{name} complete", report=True)
                return result

            def archive(s):
                path = s.root / "sources.tar.gz"
                with tarfile.open(path, "w:gz") as output_archive:
                    for name in sources:
                        output_archive.add(PROJECT / name, arcname=name)
                return {"source_sha256": sources, "archive_sha256": sha256_file(path)}

            try:
                progress("software acceptance", report=True)
                unit("provenance", {}, archive)
                software = unit("software", {}, _software)
                from .fix3_roles import centering_resolution_audit
                labels = unit("labels", {}, lambda s: centering_resolution_audit(config.learning.temperature))
                from src.robotics.curriculum import geometry_audit
                from .fix3_rollout import simulator_config, scripted_geometry_checks
                from src.robotics.drift import drift_alpha
                geometry = unit("geometry", {}, lambda s: {**geometry_audit(simulator_config()),
                    "alpha_schedule": [{"exposure": x, "alpha": drift_alpha(x, 320)}
                                       for x in (0., 32., 144., 176., 288., 320.)],
                    "scripted_controller_checked": True,
                    "scripted": scripted_geometry_checks(simulator_config())})
                from .fix3_data import render_panels
                panels = {split: unit("panels", {"split": split},
                    lambda s, k=split: render_panels(s, split=k)) for split in ("development", "validation")}
                if stage == "software":
                    return {"status": "software_complete", "geometry": geometry, "labels": labels}
                import torch
                from .models import load_qwen_model
                from .storage import FrameStore
                from .fix3_data import score_panels
                if not torch.cuda.is_available():
                    raise RuntimeError("actual CUDA is required; no fake production fallback")
                check_storage(root, quota_bytes=100_000_000_000, reserve_bytes=20_000_000_000)
                torch.set_num_threads(2)
                progress("loading frozen Qwen and trainable student", report=True)
                kwargs = {} if model_path is None else {"model_path": model_path}
                model = load_qwen_model(**kwargs, local_files_only=offline or model_path is not None,
                    seed=191101, top_layers=1, ltm_rank=4, stm_rank=1, hidden_size=64,
                    context_length=4, return_bins=labels["selected_bins"], min_pixels=56*56,
                    max_pixels=19200, max_tokens=2048).eval()
                model.encoder.enable_prefix_cache = True
                rewards = {}
                for split in ("development", "validation"):
                    frames = FrameStore(Path(panels[split]["unit_path"]) / "frames", readonly=True)
                    rewards[split] = unit("rewards", {"split": split},
                        lambda s, k=split, f=frames: score_panels(model, panels[k], f, s, stopped))
                if stage == "rewards":
                    return {"status": "rewards_complete", "validation": rewards["validation"]}
                from .fix3_rollout import teacher_rollout, simulator_config, evaluation_configs
                teacher = []
                for variant in ("ported", "centering"):
                    for seed in identity["teacher_layouts"]:
                        teacher.append(unit("teacher", {"variant": variant, "seed": seed},
                            lambda s, v=variant, k=seed: teacher_rollout(model, s, stopped,
                                variant=v, seed=k)))
                if stage == "teacher":
                    return {"status": "teacher_complete", "teacher": teacher}
                from .fix3_student import bootstrap, pilot
                teacher_scores = {variant: sum(r["success"] for r in teacher if r["variant"] == variant)
                                  for variant in ("ported", "centering")}
                selected_teacher = max(teacher_scores, key=lambda v: (teacher_scores[v], v == "centering"))
                boot = unit("bootstrap", {"maximum_transitions": 300, "teacher_variant": selected_teacher},
                    lambda s: bootstrap(model, config, simulator_config(easy=True, period=2.),
                        s, stopped, teacher_variant=selected_teacher, max_transitions=300))
                with RunStore(Path(boot["unit_path"]), json.loads(
                        (Path(boot["unit_path"]) / "config.json").read_text()), resume=True,
                        checkpoint_policy="journal-v1") as boot_store:
                    initial = boot_store.load_checkpoint(trusted=True)
                pilots = []
                for seed in identity["pilot_seeds"]:
                    pilots.append(unit("pilot", {"seed": seed, "maximum_transitions": 320},
                        lambda s, k=seed: pilot(model, config, initial,
                            simulator_config(period=1.), s, stopped, seed=k, max_transitions=320,
                            evaluation_configs=evaluation_configs())))
                elapsed = charged + time.monotonic() - started
                child = sum(json.loads(p.read_text()).get("active_seconds", 0.)
                            for p in (root / "units").glob("*/summary.json") if (p.parent / "COMPLETED").exists())
                result = {"kind": "phase1_fix3_campaign", "schema_version": 1, "phase": 1,
                    "smoke_only": True, "status": "ready_for_checkin", "units": count,
                    "total_active_seconds": elapsed, "active_seconds": max(0., elapsed - child),
                    "cumulative_repair_seconds": prior + elapsed, "return_bins": model.return_bins,
                    "software": software["test_result"], "geometry": geometry,
                    "reward_validation": {k: v for k, v in rewards["validation"].items() if k not in {"rows", "provenance"}},
                    "bootstrap_transitions": boot.get("transitions"), "pilot_paths": [p["unit_path"] for p in pilots],
                    "model_provenance": model.encoder.provenance,
                    "physical_hardware_used": False, "frozen_teacher_trained": False,
                    "limitations": ["Bounded development pilot; no STM effect or sample-efficiency inference.",
                        "Teacher/evaluator remain frozen; fixed diagnostic layouts are not fresh confirmation."]}
                parent.complete(result)
                refresh()
                return result
            finally:
                if not parent.completed:
                    elapsed = charged + time.monotonic() - started
                    parent.checkpoint({"charged_seconds": elapsed, "units": count}, progress={
                        "phase": 1, "status": "Fix 3 checkpointed; explicit --resume required",
                        "active_seconds": elapsed, "cumulative_repair_seconds": prior + elapsed,
                        "counts": {"step": count, "target": 23}})
                    refresh()
    finally:
        stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="cache/numerical_experiment/picar/phase1/fix3-v1")
    parser.add_argument("--model-path")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("software", "rewards", "teacher", "all"), default="all")
    parser.add_argument("--max-hours", type=float, default=3.)
    args = parser.parse_args()
    result = run(args.output, model_path=args.model_path, offline=args.offline,
                 resume=args.resume, stage=args.stage, max_hours=args.max_hours)
    print(json.dumps({k: v for k, v in result.items() if k not in {"geometry", "model_provenance", "teacher"}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
