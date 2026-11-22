"""Execute the approved 28-hour Fix 4 repair, never Phase 4 or physical motion."""

from __future__ import annotations

import argparse
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
from numerical_experiment.picar.storage import RunStore, _atomic_bytes
from numerical_experiment.picar.report import read_completed_unit
from numerical_experiment.picar_phase2.campaign import read_json, load_completed_state
from . import VERSION
from .config import (PROJECT, MAX_HOURS, CLOSEOUT_SECONDS, DEFAULT_OUTPUT, DEFAULT_SOURCE,
    DEFAULT_REFERENCES, MODEL_SEED, NORMALIZATION, MAIN_SEEDS, PILOT_SEED, CALIBRATION_SEED,
    REFERENCE_SEED, ENVIRONMENTS, configuration, simulation, fixed_environment, panels,
    protocol, scientific_sources, PRELAUNCH_ACCEPTANCE_SECONDS)

NOTEBOOK = PROJECT/"numerical_experiment/picar/plan19_results.ipynb"


def source_asset(source):
    parent = read_completed_unit(Path(source)/"orchestration")
    if parent["summary"].get("kind") != "reward_campaign":
        raise ValueError("Fix 4 requires the completed Fix 3 campaign")
    asset = parent["summary"]["assets"]["width6_linear"]
    path = Path(asset["path"])
    if not path.resolve().is_relative_to(Path(source).resolve()):
        raise ValueError("initial asset must be inside its declared source campaign")
    read_completed_unit(path)
    if sha256_file(path/"COMPLETED") != asset["id"]:
        raise ValueError("shared initialization completion seal differs")
    return asset


def run_identity(root, source, references, model_path, max_hours):
    history = root/"history.ipynb"
    return {"schema_version": 1, "version": VERSION, "phase": 3,
        "source_campaign": str(Path(source).resolve()),
        "source_completion": sha256_file(Path(source)/"orchestration/COMPLETED"),
        "reference_campaign": str(Path(references).resolve()),
        "reference_completion": sha256_file(Path(references)/"COMPLETED"),
        "theory_sha256": sha256_file(PROJECT/"mathematical_overview.ipynb"),
        "source_sha256": scientific_sources(), "protocol": protocol(), "max_hours": max_hours,
        "history_sha256": sha256_file(history if history.exists() else NOTEBOOK),
        "model_source": str(Path(model_path).resolve()) if model_path else configuration().model.model_id}


def run(output=DEFAULT_OUTPUT, source=DEFAULT_SOURCE, references=DEFAULT_REFERENCES, *,
        model_path=None, offline=False, resume=False, max_hours=MAX_HOURS, stage="all"):
    if not 0 < max_hours <= MAX_HOURS:
        raise ValueError("Fix 4 has an approved 28-hour additional active-compute ceiling")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved = run_identity(root, source, references, model_path, max_hours)
    stop = StopController(root)
    stop.install()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        with RunStore(root/"orchestration", resolved, resume=resume, checkpoint_policy="journal-v1") as parent:
            if parent.completed:
                return read_json(parent.root/"summary.json")
            history = root/"history.ipynb"
            if not history.exists():
                _atomic_bytes(history, NOTEBOOK.read_bytes())
            if sha256_file(history) != resolved["history_sha256"]:
                raise ValueError("immutable history snapshot changed")
            saved = parent.load_checkpoint(trusted=True) or {
                "charged_seconds": PRELAUNCH_ACCEPTANCE_SECONDS, "forecast": {}}
            base, start = saved["charged_seconds"], time.monotonic()
            active, status, execution = None, "acceptance and source checks", None
            last_save = last_report = last_storage = 0.
            forecast = saved.get("forecast", {})

            def elapsed():
                return base+time.monotonic()-start

            def refresh():
                from .report import build_notebook
                build_notebook(root, output=NOTEBOOK, base_notebook=history)

            def progress(label=None, *, force=False, final=False):
                nonlocal last_save, last_report, last_storage, status
                if label is not None:
                    status = label
                now = time.monotonic()
                if force or now-last_save >= 45 or stop.check():
                    child = read_json(active/"progress.json") if active and (active/"progress.json").exists() else {}
                    parent.checkpoint({"charged_seconds": elapsed()+(0. if final else 60.), "forecast": forecast},
                        progress={"schema_version": 1, "version": VERSION, "phase": 3, "status": status,
                            "charged_seconds": elapsed(), "max_hours": max_hours,
                            "remaining_budget_seconds": max(0., max_hours*3600-elapsed()),
                            "active_unit": str(active) if active else None, "active_unit_progress": child,
                            "forecast": forecast,
                            "completed_trajectories": len(list((root/"units").glob("trajectory-*/COMPLETED"))),
                            "planned_trajectories": 4, "next_phase_authorized": False})
                    last_save = now
                if force or now-last_storage >= 900:
                    check_storage(root, quota_bytes=100_000_000_000, reserve_bytes=20_000_000_000)
                    last_storage = now
                if force or now-last_report >= 900:
                    refresh()
                    last_report = now
                if not final and stop.check():
                    raise InterruptedError("Fix 4 checkpointed; resume intentionally with --resume --clear-stop")
                if not final and elapsed() >= max_hours*3600-CLOSEOUT_SECONDS:
                    raise TimeoutError("Fix 4 ceiling reached; checkpoint and closeout reserve preserved")

            def unit_path(name, settings):
                key = hashlib.sha256(canonical_json({**resolved, "operation": name, "settings": settings}).encode()).hexdigest()[:12]
                return root/"units"/(name+"-"+key)

            def unit(name, settings, callback):
                nonlocal active
                path = unit_path(name, settings)
                active = path
                progress(name, force=True)
                with RunStore(path, {**resolved, "operation": name, "settings": settings},
                              resume=resume, checkpoint_policy="journal-v1") as store:
                    if store.completed:
                        result = read_json(path/"summary.json")
                    else:
                        tick = time.monotonic()
                        result = callback(store)
                        if scientific_sources() != resolved["source_sha256"]:
                            raise RuntimeError("scientific code changed during this version; preserve artifacts before correcting")
                        active_seconds = max(time.monotonic()-tick,
                            result.get("timings", {}).get("active_seconds", 0.))
                        result = {**result, "schema_version": 1, "version": VERSION, "phase": 3,
                            "kind": "curriculum_"+name, "settings": settings, "unit_path": str(path),
                            "active_seconds": active_seconds, "execution": execution,
                            "development_only": True, "confirmatory": False,
                            "media_sha256": {p.name: sha256_file(p) for p in path.glob("*.mp4")}}
                        if store._current is None:
                            store.checkpoint({"stage": "done"})
                        store.complete(result)
                active = None
                print(json.dumps({"unit": name, "settings": settings, "path": str(path),
                    "seconds": result["active_seconds"], "status": "complete"}), flush=True)
                progress(name+" complete", force=True)
                return result

            def provenance(store):
                path = store.root/"sources.tar.gz"
                with tarfile.open(path, "w:gz") as archive:
                    for name in resolved["source_sha256"]:
                        archive.add(PROJECT/name, arcname=name)
                    archive.add(PROJECT/"mathematical_overview.ipynb", arcname="mathematical_overview.ipynb")
                return {"archive_sha256": sha256_file(path), "theory_sha256": resolved["theory_sha256"],
                    "source_sha256": resolved["source_sha256"]}

            def software(store):
                paths = sorted((PROJECT/"test/unit").glob("test_picar_curriculum_*.py"))
                paths += [PROJECT/"test/unit/test_picar_staged_curriculum.py"]
                result = subprocess.run([sys.executable, "-m", "pytest", "-q", *map(str, paths)], cwd=PROJECT,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900,
                    env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
                store._write_root("pytest.txt", result.stdout.encode())
                if result.returncode:
                    raise RuntimeError("Fix 4 acceptance failed: "+result.stdout[-6000:])
                return {"passed": True, "tests": result.stdout.strip().splitlines()[-1]}

            try:
                unit("provenance", {}, provenance)
                unit("software", {}, software)
                from .smoke import run as smoke
                unit("cpu_smoke", {}, lambda store: smoke(store, progress))
                asset = source_asset(source)
                initial = load_completed_state(asset["path"])
                def integrity(store):
                    learning = initial["identity"]["config"]["learning"]
                    target = asdict(configuration().learning)
                    for key in ("reward_version", "temperature", "action_variance", "return_min", "return_max", "gamma", "nominal_period_s"):
                        if learning[key] != target[key]:
                            raise ValueError("initial likelihood differs: "+key)
                    if (initial["stage"] != "done" or initial["eligible"]
                            or len(initial["retired"]) != 515 or len(set(initial["retired"])) != 515
                            or initial["learner"]["retired_count"] != 515 or not initial["distill_receipt"]):
                        raise ValueError("initializer must archive each fitted history once and start empty replay")
                    if initial["identity"]["policy"]["fisher_draws"] != 4 or initial["fisher"]["count"] != 515:
                        raise ValueError("initializer Fisher must include all 515 histories with four equal-weight draws")
                    return {"asset": asset, "train_histories": 515, "validation_histories": 65,
                        "train_episodes": 101, "validation_episodes": 12, "new_observations": 0,
                        "equal_weight_four_draw_initial_fisher": True, "initial_pi_used": False,
                        "training_plateau": initial.get("train_plateau", False)}
                unit("integrity", {}, integrity)
                from src.robotics.curriculum import geometry_audit
                def audit(store):
                    return {"layout": geometry_audit(simulation(), seeds=range(198901, 199001),
                        exposures=(0, 128, 448, 768, 1024, 1344, 1664, 2048)),
                        "schedule": protocol()["boundaries"], "new_observations": 0}
                unit("audit", {}, audit)
                if stage == "software":
                    progress("CPU acceptance complete; CUDA work not yet run", force=True, final=True)
                    return {"status": "software_complete"}

                import torch
                from numerical_experiment.picar.models import load_qwen_model
                from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
                from numerical_experiment.picar_reward_repair.scoring import load_references, WidthScorer
                from .student import CurriculumRun
                torch.set_num_threads(2)
                if not torch.cuda.is_available():
                    raise RuntimeError("actual CUDA Qwen is required; no CPU/fake substitute")
                progress("loading shared frozen Qwen", force=True)
                raw = load_qwen_model(**({"model_path": model_path} if model_path else {}),
                    local_files_only=offline, seed=MODEL_SEED, top_layers=1, ltm_rank=4, stm_rank=1,
                    hidden_size=64, context_length=4, return_bins=161,
                    min_pixels=56*56, max_pixels=19200, max_tokens=2048).eval()
                raw.encoder.enable_prefix_cache = True
                model = NormalizedStudent(raw, mode=NORMALIZATION, eps=1e-6).eval()
                execution = {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(), "dtype": "float32_adapted_suffix",
                    "model_provenance": raw.encoder.provenance, "normalization": model.provenance}
                examples = load_references(references, root/"reference_bundle")
                scorer = WidthScorer(model.encoder, examples, root/"score_cache")
                unit("references", {}, lambda store: {"scorer_identity": scorer.identity,
                    "provenance": scorer.provenance, "reference_order": list("ABCFED")})
                # The demonstrator/evaluator is the unchanged native frozen role.
                from numerical_experiment.picar.storage import FrameStore
                reference_unit = read_completed_unit(references)
                expected = next(r for r in reference_unit["summary"]["rows"]
                    if r["panel_id"] == "T18" and r["method"] == "six_examples")
                rgb = FrameStore(Path(reference_unit["config"]["panels_run"])/"frames", readonly=True).get(expected["frame_ref"])
                def parity(store):
                    value = scorer.score_rgb(rgb)
                    if value["score"] != expected["reward"] or value["width_percent"] != expected["width_percent"]:
                        raise ValueError("frozen evaluator parity failed")
                    return {"passed": True, "score": value, "new_observations": 0}
                unit("readiness", {"kind": "evaluator_parity"}, parity)

                bank, bank_seals = [], []
                for i, environment in enumerate(ENVIRONMENTS):
                    def collect(store, env=environment, seed=REFERENCE_SEED+i):
                        runner = CurriculumRun(model, configuration(), fixed_environment(env), store, progress,
                            seed=seed, max_transitions=2048, initial=initial, initial_asset_id=asset["id"],
                            scorer=scorer, evaluations=(), reference_mode=True)
                        return runner.collect_references(env)
                    result = unit("reference_bank", {"environment": environment, "seed": REFERENCE_SEED+i}, collect)
                    bank.extend(load_completed_state(result["unit_path"])["reference_records"])
                    bank_seals.append(sha256_file(Path(result["unit_path"])/"COMPLETED"))
                bank_id = hashlib.sha256(canonical_json(bank_seals).encode()).hexdigest()
                if stage == "references":
                    progress("fixed held-out banks sealed", force=True, final=True)
                    return {"status": "references_complete", "records": len(bank)}

                def trajectory(store, *, seed, strength, ridge=.1, budget=128, main=False):
                    sim = simulation() if main else replace(simulation(), total_training_intervals=budget,
                        curriculum_total_intervals=budget)
                    evaluation = panels()
                    if not main:
                        evaluation = [{**p, "seeds": p["seeds"][:2]} for p in evaluation]
                    return CurriculumRun(model, configuration(strength=strength, ridge=ridge), sim, store, progress,
                        seed=seed, max_transitions=budget, initial=initial, initial_asset_id=asset["id"], scorer=scorer,
                        evaluations=evaluation, checkpoints=(0, budget) if not main else protocol()["checkpoints"],
                        reference_records=bank, reference_id=bank_id).run()

                timing_settings = {"seed": PILOT_SEED, "method": "replay_ewc", "budget": 128,
                    "diagnostic_compressed_schedule": True, "initial_asset_id": asset["id"], "reference_id": bank_id}
                timing = unit("timing", timing_settings, lambda store: trajectory(store, seed=PILOT_SEED, strength=0.))
                from .calibration import fd_scale, select
                probe = unit("fd_probe", {}, lambda store: fd_scale(model, configuration(), load_completed_state(timing["unit_path"])))
                candidates = []
                for strength in probe["strengths"]:
                    settings = {"seed": CALIBRATION_SEED, "method": "stm_fd", "fd_strength": strength,
                        "ridge": .1, "budget": 128, "diagnostic_compressed_schedule": True,
                        "initial_asset_id": asset["id"], "reference_id": bank_id}
                    candidates.append(unit("calibration_run", settings, lambda store, strength=strength:
                        trajectory(store, seed=CALIBRATION_SEED, strength=strength)))
                # Calibrate the control on the SAME development seed, not the
                # unrelated timing stream. These trajectories never enter outcomes.
                control = unit("calibration_run", {"seed": CALIBRATION_SEED, "method": "replay_ewc",
                    "fd_strength": 0., "ridge": .1, "budget": 128, "diagnostic_compressed_schedule": True,
                    "initial_asset_id": asset["id"], "reference_id": bank_id},
                    lambda store: trajectory(store, seed=CALIBRATION_SEED, strength=0.))
                extra = None
                selection = select(control, candidates)
                if selection["warning"]:
                    strength = selection["selected"]["strength"]
                    extra = unit("calibration_run", {"seed": CALIBRATION_SEED, "method": "stm_fd", "fd_strength": strength,
                        "ridge": 1., "budget": 128, "diagnostic_compressed_schedule": True,
                        "initial_asset_id": asset["id"], "reference_id": bank_id},
                        lambda store: trajectory(store, seed=CALIBRATION_SEED, strength=strength, ridge=1.))
                    selection = select(control, candidates+[extra])
                selection = unit("selection", {}, lambda store: selection)
                chosen = selection["selected"]
                # Small-pilot wall time contains fixed evaluation/reference costs.
                # Extrapolate training-event and scoring components separately.
                def projected(summary):
                    n = max(1, summary["acquired_count"])
                    training = (summary["timings"]["fit_seconds"]+summary["timings"]["fisher_seconds"])/n*2048
                    train_score = summary["timings"]["scoring_seconds"] / max(1,
                        n+sum(len(e["intervals"]) for p in summary["evaluations"] for e in p["rows"])) * 2048
                    physical = 2048*simulation().control_interval_s
                    evaluation_count = 5*24+36
                    eval_seconds_per_episode = (summary["active_seconds"]-summary["timings"]["fit_seconds"]
                        -summary["timings"]["fisher_seconds"]-train_score*n/2048-physical*n/2048)/max(1,
                        sum(len(p["rows"]) for p in summary["evaluations"]))
                    return training+train_score+physical+max(5., eval_seconds_per_episode)*evaluation_count
                fd_candidate = next(c for c in candidates+([extra] if extra is not None else [])
                    if c["unit_path"] == chosen["path"])
                estimate = 2*(projected(control)+projected(fd_candidate))
                forecast.update({"main_hours": [estimate/3600, 1.3*estimate/3600],
                    "fixed_evaluation_episodes_per_trajectory": 156, "charged_before_main_hours": elapsed()/3600,
                    "training_intervals": 8192, "artifact_forecast_gb": [5., 25.],
                    "basis": "Separate measured fit, retirement, scoring, physical and fixed-panel overhead; compressed pilot is not evidence of schedule adequacy."})
                unit("forecast", {}, lambda store: {"forecast": forecast, "selection": selection})
                progress("calibration complete; measured main-run forecast published", force=True)
                if stage == "calibration":
                    progress("calibration complete; main trajectories not yet run", force=True, final=True)
                    return {"status": "calibration_complete", "forecast": forecast, "selection": selection}

                completed = []
                for i, seed in enumerate(MAIN_SEEDS):
                    order = ("replay_ewc", "stm_fd") if i == 0 else ("stm_fd", "replay_ewc")
                    settings = {method: {"seed": seed, "method": method, "budget": 2048,
                        "fd_strength": chosen["strength"] if method == "stm_fd" else 0.,
                        "ridge": chosen["ridge"] if method == "stm_fd" else .1,
                        "initial_asset_id": asset["id"], "reference_id": bank_id} for method in order}
                    missing = [m for m in order if not (unit_path("trajectory", settings[m])/"COMPLETED").exists()]
                    remaining_estimate = sum(projected(control if method == "replay_ewc" else fd_candidate)
                        for method in missing)
                    if missing and elapsed()+1.15*remaining_estimate+CLOSEOUT_SECONDS > max_hours*3600:
                        raise TimeoutError("complete next pair does not fit the measured approved ceiling; keep checkpoints and review scope")
                    for method in order:
                        spec = settings[method]
                        completed.append(unit("trajectory", spec, lambda store, spec=spec: trajectory(store,
                            seed=spec["seed"], strength=spec["fd_strength"], ridge=spec["ridge"], budget=2048, main=True)))
                progress("paired development complete; auditing sealed artifacts", force=True)
                seals = []
                for marker in sorted((root/"units").glob("*/COMPLETED")):
                    read_completed_unit(marker.parent)
                    seals.append({"path": str(marker.parent), "sha256": sha256_file(marker)})
                if sha256_file(PROJECT/"mathematical_overview.ipynb") != resolved["theory_sha256"]:
                    raise ValueError("reserved theory notebook changed")
                summary = {"schema_version": 1, "version": VERSION, "phase": 3, "kind": "curriculum_campaign",
                    "completed_trajectories": len(completed), "trajectories": [r["unit_path"] for r in completed],
                    "charged_seconds": elapsed(), "max_hours": max_hours, "forecast": forecast,
                    "selection": selection, "asset": asset, "reference_id": bank_id,
                    "completion_seals": seals, "next_phase_authorized": False, "physical_motion": False}
                parent.checkpoint({"charged_seconds": elapsed(), "forecast": forecast, "stage": "done"},
                    progress={**summary, "status": "complete"})
                parent.complete(summary)
                refresh()
                return summary
            except BaseException as exc:
                progress("stopped at committed checkpoint: "+type(exc).__name__, force=True, final=True)
                raise
    finally:
        stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="run", choices=("run", "status", "stop", "report"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--max-hours", type=float, default=MAX_HOURS)
    parser.add_argument("--stage", choices=("all", "software", "references", "calibration", "main"), default="all")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(read_json(args.output/"orchestration/progress.json"), indent=2))
    elif args.command == "stop":
        if not (args.output/"orchestration/config.json").exists():
            parser.error("cannot stop a campaign that has not started")
        (args.output/"STOP").touch(exist_ok=True)
        print("Checkpoint-boundary stop requested")
    elif args.command == "report":
        from .report import build_notebook
        print(build_notebook(args.output, output=NOTEBOOK, base_notebook=args.output/"history.ipynb"))
    else:
        if args.clear_stop:
            if not args.resume:
                parser.error("--clear-stop requires intentional --resume")
            (args.output/"STOP").unlink(missing_ok=True)
        print(json.dumps(run(args.output, args.source, args.references, model_path=args.model_path,
            offline=args.offline, resume=args.resume, max_hours=args.max_hours, stage=args.stage), indent=2), flush=True)


if __name__ == "__main__":
    main()
