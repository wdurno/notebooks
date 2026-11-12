"""Execute Plan 19 Phase 2 only: initial skill, trainability and cost check-in."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.runtime import StopController, check_storage, sha256_file
from numerical_experiment.picar.storage import RunStore
from .analysis import forecast, observed_trainability, panel_summary
from .config import (BASE, EASY_SEEDS, PILOT_SEEDS, RESCUES, VERSION, evaluation_panels,
                     execution_recipe, learner_config, simulator_config, source_hashes)


PROJECT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def audit_initial(state, config):
    if state.get("stage") != "done" or not state.get("distill_receipt"):
        raise ValueError("initial asset must have completed one-time KL and initialization")
    original = state["identity"]["config"]
    for key in ("context_length", "hidden_size", "return_bins", "top_layers", "ltm_rank", "stm_rank"):
        if original["model"][key] != getattr(config.model, key):
            raise ValueError(f"incompatible initial model: {key}")
    for key in ("reward_version", "return_min", "return_max", "gamma", "nominal_period_s", "temperature", "action_variance"):
        if original["learning"][key] != getattr(config.learning, key):
            raise ValueError(f"incompatible initial likelihood: {key}")
    retired = state["retired"]
    eligible = [x["arrival"] for x in state["eligible"]]
    if len(retired) != len(set(retired)) or len(eligible) != len(set(eligible)):
        raise ValueError("duplicate initial evidence identity")
    if set(retired) & set(eligible) or state["learner"]["retired_count"] != len(retired):
        raise ValueError("initial Fisher/replay evidence is not disjoint")
    if len(eligible) != config.learning.replay_capacity or any(not x["learned"] for x in state["eligible"]):
        raise ValueError("initial replay must be a fully fitted bounded buffer")
    return {"acquired_count": state["acquired_count"], "retired_count": len(retired),
        "eligible_count": len(eligible), "distinct_bootstrap_samples": state["distinct_bootstrap_samples"],
        "distill_receipt": state["distill_receipt"], "initial_evidence_disjoint": True}


def language_unit(runner):
    from numerical_experiment.picar_evaluation_video import assert_unchanged
    from .student import language_readiness
    # Even a halt before publication must retain a complete resumable identity.
    runner.save()
    before = runner.learner.state_dict()
    result = language_readiness(runner.learner, runner.s["eligible"])
    assert_unchanged(before, runner.learner.state_dict())
    runner.save()
    return {**result, "learning_state_unchanged": True}


def load_completed_state(path):
    path = Path(path)
    with RunStore(path, read_json(path / "config.json"), resume=True,
                  checkpoint_policy="journal-v1") as store:
        if not store.completed:
            raise ValueError("bootstrap source is incomplete")
        return store.load_checkpoint(trusted=True)


def run(output, *, bootstrap_source=None, model_path=None, offline=False, resume=False,
        stage="all", max_hours=5.):
    if not 0 < max_hours <= 5:
        raise ValueError("Phase 2 has a maximum five-hour active-compute allocation")
    root = Path(output).resolve()
    source = None if bootstrap_source is None else Path(bootstrap_source).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    base = learner_config()
    identity = {"schema_version": 1, "version": VERSION, "config": base.to_dict(),
        "recipe": execution_recipe(), "source_sha256": sources,
        "bootstrap_source": None if source is None else str(source),
        "bootstrap_completion_sha256": None if source is None else sha256_file(source / "COMPLETED"),
        "model_source": str(Path(model_path).resolve()) if model_path else base.model.model_id,
        "model_initialization_seed": 191101, "max_hours": max_hours,
        "component_seeds": base.seeds(0)}
    stop = StopController(root)
    stop.install()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        with RunStore(root / "orchestration", identity, resume=resume,
                      checkpoint_policy="journal-v1") as parent:
            if parent.completed:
                return read_json(parent.root / "summary.json")
            checkpoint = parent.load_checkpoint(trusted=True) or {"charged_seconds": 300., "units": 0}
            charged = checkpoint["charged_seconds"]
            count = sum(1 for _ in (root / "units").glob("*/COMPLETED"))
            started, last_save, last_report = time.monotonic(), 0., 0.
            active = None
            current_stage = "software"

            def elapsed():
                return charged + time.monotonic() - started

            def refresh():
                from numerical_experiment.picar.report import build_notebook
                build_notebook(PROJECT / "cache/numerical_experiment/picar")

            def progress(label=None, *, force=False):
                nonlocal last_save, last_report, current_stage
                if label is not None:
                    current_stage = label
                now = time.monotonic()
                if force or now-last_save >= 15 or stop.check():
                    child = read_json(active / "progress.json") if active and (active / "progress.json").exists() else {}
                    parent.checkpoint({"charged_seconds": elapsed(), "units": count}, progress={
                        "phase": 2, "status": current_stage, "active_seconds": elapsed(),
                        "remaining_estimate_seconds": max(0., 3*3600-elapsed()),
                        "counts": {"step": count, "target": 13},
                        "active_unit": None if active is None else str(active), "active_unit_progress": child})
                    last_save = now
                if force or now-last_report >= 900:
                    refresh()
                    last_report = now
                if stop.check():
                    raise InterruptedError("Phase 2 stopped at a durable checkpoint; remove STOP and --resume")
                if elapsed() >= max_hours*3600-90:
                    raise TimeoutError("Phase 2 active-time ceiling reached; preserve partial results for review")

            def unit(name, settings, callback):
                nonlocal active, count
                progress(name, force=True)
                unit_identity = {**identity, "operation": name, "settings": settings}
                suffix = hashlib.sha256(canonical_json(unit_identity).encode()).hexdigest()[:12]
                path = root / "units" / f"{name}-{suffix}"
                active = path
                with RunStore(path, unit_identity, resume=resume, checkpoint_policy="journal-v1") as store:
                    if store.completed:
                        active = None
                        return read_json(path / "summary.json")
                    before = time.monotonic()
                    payload = callback(store)
                    if source_hashes() != sources:
                        raise RuntimeError("Phase 2 numerical sources changed; create a separate version")
                    result = {**payload, "schema_version": 1, "phase": 2, "smoke_only": True,
                        "kind": f"phase2_{name}", "settings": settings, "unit_path": str(path),
                        "active_seconds": time.monotonic()-before,
                        "media_sha256": {str(p.relative_to(path)): sha256_file(p) for p in path.rglob("*")
                                         if p.suffix in {".png", ".mp4"}},
                        "frame_sha256": [p.stem for p in sorted((path / "frames").glob("*.npy"))]}
                    if store._current is None:
                        store.checkpoint({"stage": "done"}, progress={"phase": 2, "status": name})
                    store.complete(result)
                count += 1
                active = None
                print(json.dumps({"unit": name, "settings": settings, "seconds": result["active_seconds"],
                                  "path": str(path)}), flush=True)
                progress(f"{name} complete", force=True)
                return result

            def provenance(store):
                path = store.root / "sources.tar.gz"
                with tarfile.open(path, "w:gz") as archive:
                    for name in sources:
                        archive.add(PROJECT / name, arcname=str(Path(name).relative_to(PROJECT)) if Path(name).is_absolute() else name)
                return {"source_sha256": sources, "archive_sha256": sha256_file(path)}

            def software(store):
                tests = sorted(str(p.relative_to(PROJECT)) for p in (PROJECT / "test/unit").glob("test_picar*.py"))
                result = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests],
                    cwd=PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
                path = store.root / "pytest.txt"
                path.write_text(result.stdout)
                if result.returncode:
                    raise RuntimeError(f"Phase 2 software checks failed: {path}")
                return {"passed": True, "test_result": result.stdout.strip().splitlines()[-1],
                        "test_log_sha256": sha256_file(path)}

            try:
                unit("provenance", {}, provenance)
                unit("software", {}, software)
                if stage == "software":
                    return {"status": "software_complete", "units": count}
                check_storage(root, quota_bytes=100_000_000_000, reserve_bytes=20_000_000_000)
                import torch
                from numerical_experiment.picar.models import load_qwen_model
                from .student import ReadinessRun
                torch.set_num_threads(2)
                if not torch.cuda.is_available():
                    raise RuntimeError("Phase 2 requires the actual CUDA model; no fake fallback")

                def new_model():
                    progress("loading common model", force=True)
                    kwargs = {} if model_path is None else {"model_path": model_path}
                    model = load_qwen_model(**kwargs, local_files_only=offline,
                        seed=191101, top_layers=1, ltm_rank=4, stm_rank=1, hidden_size=64,
                        context_length=4, return_bins=161, min_pixels=56*56,
                        max_pixels=19200, max_tokens=2048).eval()
                    model.encoder.enable_prefix_cache = True
                    return model

                model = new_model()

                def acquire_initial(store):
                    if source is not None:
                        initial = load_completed_state(source)
                        audit = audit_initial(initial, base)
                        store.checkpoint(initial, progress={"phase": 2, "status": "compatible bootstrap imported"})
                        return {**audit, "source": str(source), "reused": True,
                            "source_marker_sha256": sha256_file(source / "COMPLETED"),
                            "new_teacher_transitions": 0, "new_distillation_steps": 0}
                    runner = ReadinessRun(model, base, simulator_config(bootstrap=True), store,
                        progress, mode="bootstrap", seed=base.seed,
                        max_transitions=BASE.bootstrap_transitions)
                    return {**runner.run(), "reused": False}

                initial_asset = unit("initial_asset", {}, acquire_initial)
                initial = load_completed_state(initial_asset["unit_path"])
                audit_initial(initial, base)

                def make_runner(store, recipe, seed, initial_state):
                    return ReadinessRun(model, learner_config(recipe), simulator_config(recipe),
                        store, progress, mode="pilot", seed=seed, max_transitions=recipe.transitions,
                        initial=initial_state, evaluations=evaluation_panels(recipe))

                def initial_panel(store, initial_state, recipe):
                    runner = make_runner(store, recipe, PILOT_SEEDS[0], initial_state)
                    runner.evaluate("initial")
                    runner.s["stage"] = "done"
                    runner.save()
                    return {"evaluations": runner.s["evaluations"], "videos": runner.s["videos"],
                        "panel_summary": panel_summary(runner.s["evaluations"]),
                        "timings": runner.s["timings"], "learning": False}

                initial_readiness = unit("initial_readiness", {}, lambda s: initial_panel(s, initial, BASE))

                def language(store):
                    runner = make_runner(store, BASE, PILOT_SEEDS[0], initial)
                    return language_unit(runner)
                unit("language", {}, language)

                def teacher_check(store):
                    from numerical_experiment.picar_evaluation_video import assert_unchanged
                    runner = ReadinessRun(model, base, simulator_config(bootstrap=True), store,
                        progress, mode="bootstrap", seed=base.seed, max_transitions=1200)
                    before = runner.learner.state_dict()
                    rows = runner.s.setdefault("teacher_rows", [])
                    for seed in EASY_SEEDS[len(rows):]:
                        runner.episode(runner.sim_config, seed=seed, exposure=0., evaluation=False)
                        current = runner.s["current"]
                        if current.get("truncated"):
                            raise RuntimeError("teacher readiness infrastructure truncation is not a task failure")
                        rows.append({"seed": seed, "success": current["success"],
                            "end_reason": current["end_reason"], "layout": current["layout"],
                            "physical_seconds": sum(r["elapsed_s"] for r in current["intervals"]),
                            "commands": len(current["intervals"]),
                            "action_counts": dict(Counter(str(max(range(8),
                                key=r['probabilities'].__getitem__)) for r in current['intervals']
                                if r.get('probabilities') is not None))})
                        runner.s["episodes"].append(copy.deepcopy(current))
                        runner.s["current"], runner.s["session"], runner.s["history"], runner.s["pending"] = None, None, None, []
                        runner.save()
                    assert_unchanged(before, runner.learner.state_dict())
                    runner.s["stage"] = "done"
                    runner.save()
                    return {"rows": rows, "panel_summary": panel_summary([
                        {"checkpoint": "frozen_teacher", "difficulty": "easy", "rows": rows}]),
                        "learning": False, "used_for_bootstrap": False, "realtime": True,
                        "acquired_count": runner.s["acquired_count"], "controller": "frozen Qwen teacher"}

                teacher = unit("teacher_readiness", {}, teacher_check)
                if stage == "readiness":
                    return {"status": "readiness_panels_complete", "teacher": teacher,
                            "initial_readiness": initial_readiness}

                pilots = []
                rescue_decisions = []

                def learning_pilot(store, recipe, seed, initial_state, initial_panel_result):
                    runner = make_runner(store, recipe, seed, initial_state)
                    # The same exact initial checkpoint/panels need only be evaluated once.
                    if runner.s["stage"] == "initial_evaluation" and not runner.s["evaluations"]:
                        runner.s["evaluations"] = copy.deepcopy(initial_panel_result["evaluations"])
                        runner.s["initial_evaluation_source"] = initial_panel_result["unit_path"]
                        runner.s["stage"] = "collect"
                        runner.save()
                    return runner.run()

                for seed in PILOT_SEEDS:
                    pilots.append(unit("pilot", {"name": BASE.name, "seed": seed, **asdict(BASE)},
                        lambda s, k=seed: learning_pilot(s, BASE, k, initial, initial_readiness)))

                for rescue in RESCUES:
                    if observed_trainability(pilots):
                        rescue_decisions.append({"name": rescue.name, "decision": "not needed after observed development gain"})
                        break
                    continuing = any(read_json(p).get("settings", {}).get("name") == rescue.name
                                     for p in (root / "units").glob("*/config.json"))
                    if not continuing and max_hours*3600-elapsed() < (4500 if rescue.name == "softer_bootstrap" else 2400):
                        rescue_decisions.append({"name": rescue.name, "decision": "not admitted: insufficient remaining budget"})
                        break
                    rescue_decisions.append({"name": rescue.name, "decision": "resumed" if continuing else "admitted"})
                    current_initial, current_panel = initial, initial_readiness
                    if rescue.name == "softer_bootstrap":
                        # New observed controls/targets, not a post-hoc resmoothing of old demonstrations.
                        del model
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = new_model()
                        rescued = unit("bootstrap", asdict(rescue), lambda s: ReadinessRun(model,
                            learner_config(rescue), simulator_config(rescue, bootstrap=True), s,
                            progress, mode="bootstrap", seed=base.seed,
                            max_transitions=rescue.bootstrap_transitions).run())
                        current_initial = load_completed_state(rescued["unit_path"])
                        audit_initial(current_initial, learner_config(rescue))
                        current_panel = unit("initial_readiness", {"name": rescue.name},
                            lambda s: initial_panel(s, current_initial, rescue))
                    pilots.append(unit("pilot", {"name": rescue.name, "seed": PILOT_SEEDS[0], **asdict(rescue)},
                        lambda s, r=rescue, i=current_initial, p=current_panel:
                            learning_pilot(s, r, PILOT_SEEDS[0], i, p)))

                previous_review = PROJECT / "cache/numerical_experiment/picar/phase1/fix3-v2/review/resource-forecast.json"
                prior = read_json(previous_review)["conservative_total_study_seconds_so_far"] if previous_review.exists() else 0.
                diagnostic = PROJECT / "cache/numerical_experiment/picar/diagnostics/fix3-final-hard-video-v1/summary.json"
                if diagnostic.exists():
                    prior += read_json(diagnostic)["active_seconds"]+120.
                resource = unit("forecast", {}, lambda s: forecast(pilots,
                    phase2_seconds=elapsed(), prior_study_seconds=prior))
                result = {"kind": "phase2_campaign", "schema_version": 1, "phase": 2, "smoke_only": True,
                    "status": "ready_for_checkin", "units": count, "teacher_readiness": teacher,
                    "initial_readiness": initial_readiness, "pilot_paths": [p["unit_path"] for p in pilots],
                    "rescue_decisions": rescue_decisions,
                    "trainability_observed": observed_trainability(pilots), "forecast": resource,
                    "total_active_seconds": elapsed(), "active_seconds": 0.,
                    "pre_execution_validation_reserve_seconds": 300.,
                    "next_phase_authorized": False}
                progress("Phase 2 complete; readiness and forecast check-in", force=True)
                parent.checkpoint({"charged_seconds": elapsed(), "units": count}, progress={
                    "phase": 2, "status": "Phase 2 complete; awaiting user check-in",
                    "active_seconds": elapsed(), "counts": {"step": count, "target": count}})
                parent.complete(result)
                refresh()
                return result
            except BaseException:
                # Child state has already been checkpointed at each scientific boundary.
                parent.checkpoint({"charged_seconds": elapsed(), "units": count}, progress={
                    "phase": 2, "status": "checkpointed interruption; explicit resume required",
                    "active_seconds": elapsed(), "counts": {"step": count, "target": 13}})
                refresh()
                raise
    finally:
        stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cache/numerical_experiment/picar/phase2/readiness-v1"))
    parser.add_argument("--bootstrap-source", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("software", "readiness", "all"), default="all")
    parser.add_argument("--max-hours", type=float, default=5.)
    args = parser.parse_args()
    result = run(args.output, bootstrap_source=args.bootstrap_source, model_path=args.model_path,
        offline=args.offline, resume=args.resume, stage=args.stage, max_hours=args.max_hours)
    print(json.dumps({k: result.get(k) for k in ("status", "units", "trainability_observed", "total_active_seconds")}, indent=2))


if __name__ == "__main__":
    main()
