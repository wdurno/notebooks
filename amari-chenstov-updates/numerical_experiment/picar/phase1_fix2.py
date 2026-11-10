"""Plan 19 Phase 1 Fix 2: immutable readiness diagnostics and software acceptance."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from .runtime import StopController, check_storage, sha256_file, source_fingerprint
from .storage import RunStore


VERSION = "phase1-fix2-v2"


def scientific_sources():
    return {key: value for key, value in source_fingerprint().items()
            if not (Path(key).stem in {"analysis", "report", "preview"}
                    or Path(key).stem.endswith("_report"))}


def refresh():
    from .report import build_notebook
    build_notebook(PROJECT / "cache/numerical_experiment/picar")


def _prior_repair_seconds(root):
    total = 0.
    parent = PROJECT / "cache/numerical_experiment/picar/phase1"
    for path in parent.glob("fix*/orchestration"):
        if path.resolve() == (root / "orchestration").resolve():
            continue
        if (path / "COMPLETED").exists():
            data = json.loads((path / "summary.json").read_text())
            total += data.get("total_active_seconds", 0.)
        elif (path / "progress.json").exists():
            total += json.loads((path / "progress.json").read_text()).get("active_seconds", 0.)
    return total


def _software(store):
    files = sorted(glob.glob("test/unit/test_picar*.py"))
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", *files],
        cwd=PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    output = run.stdout
    (store.root / "pytest.txt").write_text(output)
    if run.returncode:
        raise RuntimeError("Fix 2 unit acceptance failed; see unit pytest.txt")
    return {"passed": True, "test_result": output.strip().splitlines()[-1],
        "test_log_sha256": sha256_file(store.root / "pytest.txt"),
        "physical_hardware_used": False, "retired_sleep_scientific_arm": False}


def run(root, *, model_path=None, offline=False, resume=False, max_hours=3., stage="all"):
    if not 0 < max_hours <= 3:
        raise ValueError("Fix 1 plus Fix 2 retain the three-hour repair envelope")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    root = Path(root).resolve()
    previous = _prior_repair_seconds(root)
    base = PicarConfig()
    config = replace(base, version=VERSION, mode="production", task_version="ball-contact-v1",
        image_width=160, image_height=120, model=replace(base.model, kind="qwen", hidden_size=64,
            context_length=4, return_bins=41), learning=replace(base.learning, replay_capacity=8,
            replay_batch=1, reward_version="dense-visual-v1", return_min=-6., return_max=12.),
        runtime=replace(base.runtime, device="cuda", threads=2))
    config.validate()
    sources = scientific_sources()
    identity = {"config": config.to_dict(), "campaign_id": str(root), "version": VERSION,
        "model_source": str(Path(model_path).resolve()) if model_path else config.model.model_id,
        "source_sha256": sources, "prior_repair_seconds": previous, "max_hours": max_hours,
        "component_seeds": config.seeds(0), "model_initialization_seed": 191101}
    root.mkdir(parents=True, exist_ok=True)
    stop = StopController(root)
    stop.install()
    try:
        with RunStore(root / "orchestration", identity, resume=resume, checkpoint_policy="journal-v1") as parent:
            if parent.completed:
                return json.loads((parent.root / "summary.json").read_text())
            saved = parent.load_checkpoint(trusted=True) or {"charged_seconds": 0., "units": 0}
            charged, count = saved["charged_seconds"], saved["units"]
            started = time.monotonic()
            def progress(status, *, report=False):
                elapsed = charged + time.monotonic() - started
                parent.checkpoint({"charged_seconds": elapsed, "units": count}, progress={
                    "phase": 1, "status": f"Fix 2: {status}", "active_seconds": elapsed,
                    "counts": {"step": count}})
                if report:
                    refresh()
                if stop.check():
                    raise InterruptedError("Fix 2 stopped at a checkpoint boundary")
                if previous + elapsed >= max_hours * 3600 - 30:
                    raise TimeoutError("combined repair compute envelope reached")
            def stopped():
                progress("diagnostic checkpoint")
            def unit(name, settings, callback):
                nonlocal count
                progress(name)
                unit_identity = {**identity, "operation": name, "settings": settings}
                suffix = hashlib.sha256(canonical_json(unit_identity).encode()).hexdigest()[:12]
                path = root / "units" / f"{name}-{suffix}"
                with RunStore(path, unit_identity, resume=resume, checkpoint_policy="journal-v1") as store:
                    if store.completed:
                        return json.loads((path / "summary.json").read_text())
                    before = time.monotonic()
                    payload = callback(store)
                    result = {**payload, "schema_version": 1, "phase": 1, "smoke_only": True,
                        "kind": f"phase1_fix2_{name}", "settings": settings,
                        "active_seconds": time.monotonic() - before, "unit_path": str(path)}
                    if scientific_sources() != sources:
                        raise RuntimeError("Fix 2 numerical/role sources changed; preserve attempt and version the new run")
                    store.checkpoint({"stage": "done"}, progress={"phase": 1,
                        "status": f"Fix 2 {name} complete", "counts": {"step": 1}})
                    store.complete(result)
                count += 1
                progress(f"{name} complete", report=True)
                return result
            try:
                progress("software and causal data checks", report=True)
                software = unit("software", {}, _software)
                from .fix2_data import make_panels, panel_history
                from .fix2_labels import resolution_audit
                labels = unit("labels", {}, lambda store: resolution_audit())
                panels = {split: unit("panels", {"split": split}, lambda s, p=split: make_panels(s, split=p))
                          for split in ("development", "validation")}
                if stage == "software":
                    progress("software stage complete; GPU stages pending", report=True)
                    return {"status": "software_complete", "software": software["test_result"]}
                import torch
                from .models import load_qwen_model
                from .fix2_roles import instruction_contrasts, reward_panel, rollout
                from .storage import FrameStore
                if not torch.cuda.is_available():
                    raise RuntimeError("Fix 2 requires actual CUDA; no fake-model fallback")
                check_storage(root, quota_bytes=100_000_000_000, reserve_bytes=20_000_000_000)
                torch.set_num_threads(2)
                progress("loading actual Qwen", report=True)
                kwargs = dict(seed=191101, top_layers=1, hidden_size=64, ltm_rank=4, stm_rank=1,
                    context_length=4, return_bins=labels["selected_bins"] or 41, max_tokens=2048,
                    min_pixels=56 * 56, max_pixels=19200, local_files_only=offline or model_path is not None)
                if model_path:
                    kwargs["model_path"] = model_path
                model = load_qwen_model(**kwargs).eval()
                model.encoder.enable_prefix_cache = True
                config = replace(config, model=replace(config.model, return_bins=model.return_bins))
                # Panel frames have their own immutable owner, not this score unit.
                rewards = {}
                for rubric in ("legacy", "contact"):
                    panel_frames = FrameStore(Path(panels["development"]["unit_path"]) / "frames", readonly=True)
                    from types import SimpleNamespace
                    rewards[rubric] = unit("rewards", {"rubric": rubric, "split": "development"},
                        lambda s, r=rubric, f=panel_frames: reward_panel(model, panels["development"],
                            SimpleNamespace(frames=f), r, s, stopped))
                selected_rubric = max(rewards, key=lambda key: (rewards[key]["reward_pass"],
                    rewards[key]["order_fraction"], key == "contact"))
                frames = FrameStore(Path(panels["validation"]["unit_path"]) / "frames", readonly=True)
                reward_validation = unit("rewards", {"rubric": selected_rubric, "split": "validation"},
                    lambda s: reward_panel(model, panels["validation"], SimpleNamespace(frames=frames),
                                           selected_rubric, s, stopped))
                history = panel_history(panels["development"]["panels"][1], panel_frames)
                contrasts = unit("instructions", {}, lambda s: instruction_contrasts(model, history))
                if stage == "rewards":
                    progress("reward stage complete; teacher/pipeline pending", report=True)
                    return {"status": "rewards_complete", "reward_pass": reward_validation["reward_pass"]}
                developments, selected_teacher = [], None
                for obstacle in (False, True):
                    for variant in ("ported", "grounded"):
                        for seed, domain in ((601, "A"), (602, "A"), (603, "A"),
                                             (611, "B"), (612, "B"), (613, "B")):
                            developments.append(unit("teacher", {"variant": variant, "seed": seed,
                                "domain": domain, "obstacle": obstacle, "split": "development"},
                                lambda s, v=variant, k=seed, o=obstacle, d=domain: rollout(model, variant=v, seed=k,
                                    domain=d, obstacle=o, store=s, stopped=stopped)))
                        if obstacle:
                            rows = [r for r in developments if r["variant"] == variant and r["obstacle"]
                                    and r["domain"] == "A"]
                            language = next(x["pass"] for x in contrasts["conditions"] if x["variant"] == variant)
                            if sum(r["success"] for r in rows) / len(rows) >= .7 and language:
                                selected_teacher = variant
                                break
                validation = []
                if selected_teacher:
                    for seed in range(801, 821):
                        validation.append(unit("teacher", {"variant": selected_teacher, "seed": seed,
                            "domain": "A", "obstacle": True, "split": "readiness-validation"},
                            lambda s, k=seed: rollout(model, variant=selected_teacher, seed=k,
                                domain="A", obstacle=True, store=s, stopped=stopped)))
                if stage == "teacher":
                    progress("teacher diagnostics complete; pipeline pending", report=True)
                    return {"status": "teacher_complete", "selected_teacher": selected_teacher}
                from .learner import MemoryLearner
                from .fix2_pipeline import pipeline
                learner = MemoryLearner(model, config.learning, config.seeds(0))
                initial = learner.state_dict()
                pipelines = [unit("pipeline", {"condition": name, "period_s": 1., "episodes": 3},
                    lambda s, n=name: pipeline(model, config, initial, condition=n, rubric=selected_rubric,
                                               store=s, stopped=stopped)) for name in config.conditions]
                def checkpoint_probe(store):
                    samples = []
                    for index in range(3):
                        before = time.monotonic()
                        store.checkpoint(initial, progress={"phase": 1, "status": "Fix 2 checkpoint timing",
                                                          "counts": {"step": index + 1}})
                        samples.append(time.monotonic() - before)
                    return {"transaction_seconds": sorted(samples)[1], "samples": samples,
                        "learner_state_bytes": max(p.stat().st_size for p in store.root.rglob("state.pt"))}
                checkpoint = unit("checkpoint", {}, checkpoint_probe)
                from .fix2_costs import resource_forecast
                forecast_result = unit("forecast", {}, lambda s: resource_forecast(pipelines,
                    developments, checkpoint, elapsed_hours=(previous + charged + time.monotonic() - started) / 3600))
                def source_archive(store):
                    output = store.root / "sources.tar.gz"
                    with tarfile.open(output, "w:gz") as archive:
                        for name in sources:
                            archive.add(PROJECT / name, arcname=name)
                    return {"source_sha256": sources, "archive_sha256": sha256_file(output)}
                unit("provenance", {}, source_archive)
                teacher_pass = bool(validation) and sum(x["success"] for x in validation) / len(validation) >= .7
                reward_pass = reward_validation["reward_pass"] and labels["resolution_pass"]
                blockers = []
                if not reward_pass:
                    blockers.append("Generated visual scores or return resolution remain below the readiness guide.")
                if not teacher_pass:
                    blockers.append("Full obstacle-task Qwen competence is not validated on 20 fresh layouts.")
                if any(p["fallback_fraction"] > .1 for p in pipelines):
                    blockers.append("The profiled full pipeline exceeds the non-startup control fallback guide.")
                blockers.append("The full study remains subject to resource review and Phase 2 student readiness.")
                elapsed = charged + time.monotonic() - started
                child_seconds = sum(json.loads(p.read_text()).get("active_seconds", 0.)
                                    for p in (root / "units").glob("*/summary.json")
                                    if (p.parent / "COMPLETED").exists())
                summary = {"schema_version": 1, "phase": 1, "smoke_only": True,
                    "kind": "phase1_fix2_campaign", "status": "ready_for_checkin",
                    "reward_pass": reward_pass, "teacher_pass": teacher_pass, "teacher_variant": selected_teacher,
                    "reward_rubric": selected_rubric, "return_bins": model.return_bins,
                    "teacher_validation_count": len(validation),
                    "teacher_validation_successes": sum(x["success"] for x in validation),
                    "total_active_seconds": elapsed, "active_seconds": max(0., elapsed - child_seconds),
                    "cumulative_repair_seconds": previous + elapsed, "blockers": blockers,
                    "deliverables": {"Numeric rewards and return pipeline": "implemented and audited",
                        "Contact semantics": "CPU Bullet tests passed", "Causal instructions/countdown": "implemented and tested",
                        "Retired sleep option": "default-off; CPU acceptance passed",
                        "Pi and speech interfaces": "injected/recorded-input acceptance passed; physical validation pending",
                        "Real-time service": "measured, including stopped labeling/drain; see limitations"},
                    "scientific_phases_started": False, "physical_hardware_used": False,
                    "model_provenance": model.encoder.provenance,
                    "resource_forecast": forecast_result,
                    "pipelines": [{k: p[k] for k in ("condition", "pipeline_seconds", "fallback_fraction",
                        "updates", "online_updates", "sleep_updates", "timings")} for p in pipelines]}
                parent.checkpoint({"charged_seconds": elapsed, "units": count}, progress={"phase": 1,
                    "status": "Fix 2 ready for check-in", "active_seconds": elapsed, "counts": {"step": count}})
                parent.complete(summary)
                refresh()
                return summary
            finally:
                if not parent.completed:
                    elapsed = charged + time.monotonic() - started
                    parent.checkpoint({"charged_seconds": elapsed, "units": count}, progress={"phase": 1,
                        "status": "Fix 2 checkpointed; explicit resume required", "active_seconds": elapsed,
                        "counts": {"step": count}})
                    refresh()
    finally:
        stop.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-hours", type=float, default=3.)
    parser.add_argument("--stage", choices=("all", "software", "rewards", "teacher"), default="all")
    args = parser.parse_args(argv)
    if args.resume:
        (args.output / "STOP").unlink(missing_ok=True)
    result = run(args.output, model_path=args.model_path, offline=args.offline,
                 resume=args.resume, max_hours=args.max_hours, stage=args.stage)
    print(json.dumps({k: v for k, v in result.items() if k != "model_provenance"}, indent=2))


if __name__ == "__main__":
    main()
