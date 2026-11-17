"""Twelve-hour, restartable reward repair; no Phase 4 or physical actuation."""

from __future__ import annotations

import argparse
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
from numerical_experiment.picar_phase2.campaign import PROJECT, load_completed_state, read_json
from . import VERSION
from .config import (ACCEPTANCE_SECONDS, BASE_RECIPE, CLOSEOUT_SECONDS, CONDITIONS,
    DEFAULT_OUTPUT, DEFAULT_REFERENCES, DEFAULT_SOURCE, DEVELOPMENT_SEEDS, MAX_HOURS,
    MODEL_SEED, NORMALIZATION, configuration, panels, protocol, simulation, sources)


CANONICAL_NOTEBOOK = PROJECT / "numerical_experiment/picar/plan19_results.ipynb"


def run_identity(root, source, references, model_path, max_hours):
    history = root / "history.ipynb"
    return {"schema_version": 1, "phase": 3, "version": VERSION,
        "source_campaign": str(Path(source).resolve()),
        "source_completion": sha256_file(Path(source) / "orchestration/COMPLETED"),
        "reference_campaign": str(Path(references).resolve()),
        "reference_completion": sha256_file(Path(references) / "COMPLETED"),
        "history_notebook_sha256": sha256_file(history if history.exists() else CANONICAL_NOTEBOOK),
        "theory_sha256": sha256_file(PROJECT / "mathematical_overview.ipynb"),
        "source_sha256": sources(), "protocol": protocol(), "max_hours": max_hours,
        "model_source": str(Path(model_path).resolve()) if model_path else configuration().model.model_id}


def run(output=DEFAULT_OUTPUT, source=DEFAULT_SOURCE, references=DEFAULT_REFERENCES, *,
        model_path=None, offline=False, resume=False, max_hours=MAX_HOURS, stage="all"):
    if not 0 < max_hours <= MAX_HOURS:
        raise ValueError("reward repair has a twelve-hour additional active compute ceiling")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved = run_identity(root, source, references, model_path, max_hours)
    stop = StopController(root)
    stop.install()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        with RunStore(root / "orchestration", resolved, resume=resume, checkpoint_policy="journal-v1") as parent:
            if parent.completed:
                return read_json(parent.root / "summary.json")
            history = root / "history.ipynb"
            if not history.exists():
                _atomic_bytes(history, CANONICAL_NOTEBOOK.read_bytes())
            if sha256_file(history) != resolved["history_notebook_sha256"]:
                raise ValueError("historical notebook snapshot changed")
            saved = parent.load_checkpoint(trusted=True) or {"charged_seconds": ACCEPTANCE_SECONDS}
            base, started = saved["charged_seconds"], time.monotonic()
            active, label, execution = None, "source and acceptance checks", None
            last_save = last_report = 0.
            forecasts = saved.get("forecasts", {})

            def elapsed():
                return base + time.monotonic() - started

            def refresh():
                from .report import build_notebook
                build_notebook(root, output=CANONICAL_NOTEBOOK, base_notebook=history)

            def progress(status=None, *, force=False, final=False):
                nonlocal label, last_save, last_report
                if status is not None:
                    label = status
                now = time.monotonic()
                if force or now - last_save >= 45 or stop.check():
                    child = read_json(active / "progress.json") if active and (active / "progress.json").exists() else {}
                    parent.checkpoint({"charged_seconds": elapsed() + (0. if final else 60.), "forecasts": forecasts},
                        progress={"schema_version": 1, "version": VERSION, "phase": 3, "status": label,
                            "repair_active_seconds": elapsed(), "remaining_budget_seconds": max(0., max_hours*3600-elapsed()),
                            "max_hours": max_hours, "active_unit": str(active) if active else None,
                            "active_unit_progress": child, "forecast": forecasts,
                            "completed_trajectories": len(list((root / "units").glob("pilot-*/COMPLETED"))),
                            "planned_trajectories": 4, "next_phase_authorized": False})
                    last_save = now
                if force or now - last_report >= 900:
                    refresh()
                    last_report = now
                if not final and stop.check():
                    raise InterruptedError("reward repair checkpointed; use intentional --clear-stop --resume")
                if not final and elapsed() >= max_hours*3600-CLOSEOUT_SECONDS:
                    raise TimeoutError("reward repair active limit reached; closeout reserve preserved")

            def admit(seconds):
                return max_hours*3600-elapsed()-CLOSEOUT_SECONDS >= 1.15*seconds

            def unit_path(name, settings):
                value = {**resolved, "operation": name, "settings": settings}
                return root / "units" / (name + "-" + hashlib.sha256(canonical_json(value).encode()).hexdigest()[:12])

            def unit(name, settings, callback):
                nonlocal active
                path = unit_path(name, settings)
                active = path
                progress(name, force=True)
                check_storage(root, quota_bytes=100_000_000_000, reserve_bytes=20_000_000_000)
                cfg = {**resolved, "operation": name, "settings": settings}
                with RunStore(path, cfg, resume=resume, checkpoint_policy="journal-v1") as store:
                    if store.completed:
                        result = read_json(path / "summary.json")
                    else:
                        tick = time.monotonic()
                        result = callback(store)
                        if sources() != resolved["source_sha256"]:
                            raise RuntimeError("scientific sources changed; preserve this version before correcting")
                        result = {**result, "schema_version": 1, "phase": 3, "version": VERSION,
                            "kind": "reward_" + name, "smoke_only": True, "development_only": True,
                            "confirmatory": False, "settings": settings, "unit_path": str(path),
                            "active_seconds": time.monotonic() - tick, "execution": execution,
                            "media_sha256": {str(p.relative_to(path)): sha256_file(p) for p in path.glob("*.mp4")}}
                        if store._current is None:
                            store.checkpoint({"stage": "done"})
                        store.complete(result)
                active = None
                print(json.dumps({"unit": name, "settings": settings, "path": str(path),
                    "seconds": result["active_seconds"], "status": "complete"}), flush=True)
                progress(name + " complete", force=True)
                return result

            def provenance(store):
                path = store.root / "sources.tar.gz"
                with tarfile.open(path, "w:gz") as archive:
                    for name in resolved["source_sha256"]:
                        archive.add(PROJECT / name, arcname=name)
                return {"archive_sha256": sha256_file(path), "source_sha256": resolved["source_sha256"],
                    "theory_sha256": resolved["theory_sha256"], "historical_notebook_sha256": resolved["history_notebook_sha256"]}

            def software(store):
                paths = [str(p.relative_to(PROJECT)) for p in sorted((PROJECT / "test/unit").glob("test_picar_reward_*.py"))]
                if len(paths) < 4:
                    raise RuntimeError("reward repair acceptance tests are missing")
                check = subprocess.run([sys.executable, "-m", "pytest", "-q", *paths], cwd=PROJECT,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                store._write_root("pytest.txt", check.stdout.encode())
                if check.returncode:
                    raise RuntimeError("reward repair acceptance failed: " + check.stdout[-5000:])
                return {"passed": True, "test_result": check.stdout.strip().splitlines()[-1]}

            try:
                unit("provenance", {}, provenance)
                from .data import inspect_source, restore_stationary, manifest, image_jobs, relabel
                from .scoring import load_references, WidthScorer
                from .math import reward_arithmetic_audit
                source_info = unit("integrity", {}, lambda store: inspect_source(source))
                unit("references", {}, lambda store: (
                    load_references(references, root / "reference_bundle"),
                    read_json(root / "reference_bundle/references.json"))[1])
                reference_bank = load_references(references, root / "reference_bundle")
                unit("software", {}, software)
                audit = unit("audit", {}, lambda store: {"arithmetic": reward_arithmetic_audit()})
                if not all(r["timely_success_beats_loitering"] for r in audit["arithmetic"].values()):
                    raise ValueError("reward contract fails the loitering audit")
                if stage == "software":
                    progress("software complete; GPU initialization pending", force=True, final=True)
                    return {"status": "software_complete"}

                import torch
                from numerical_experiment.picar.models import load_qwen_model
                from numerical_experiment.picar.storage import FrameStore
                from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
                from .initialization import RewardInitialization, pristine_from_distilled
                from .student import RewardDevelopmentRun
                torch.set_num_threads(2)
                if not torch.cuda.is_available():
                    raise RuntimeError("actual CUDA Qwen required; no CPU/fake replacement")
                progress("loading common frozen Qwen and stationary source", force=True)
                raw = load_qwen_model(**({"model_path": model_path} if model_path else {}),
                    local_files_only=offline, seed=MODEL_SEED, top_layers=1, ltm_rank=4, stm_rank=1,
                    hidden_size=64, context_length=4, return_bins=161,
                    min_pixels=56*56, max_pixels=19200, max_tokens=2048).eval()
                raw.encoder.enable_prefix_cache = True
                model = NormalizedStudent(raw, mode=NORMALIZATION, eps=1e-6).eval()
                execution = {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(), "dtype": "float32_adapted_suffix",
                    "model_provenance": raw.encoder.provenance, "normalization": model.provenance}
                scorer = WidthScorer(model.encoder, reference_bank, root / "score_cache")
                def native_audit(store):
                    from numerical_experiment.picar.report import read_completed_unit
                    from numerical_experiment.picar_reference_reward_probe import prepare_references
                    reference_unit = read_completed_unit(references)
                    expected = next(r for r in reference_unit["summary"]["rows"]
                        if r["panel_id"] == "T18" and r["method"] == "six_examples")
                    frame_root = Path(reference_unit["config"]["panels_run"])/"frames"
                    rgb = FrameStore(frame_root, readonly=True).get(expected["frame_ref"])
                    inputs = prepare_references(model.encoder, reference_bank, rgb, few_shot=True)
                    patches = int(inputs["image_grid_thw"][-1].prod())
                    digest = hashlib.sha256(inputs["pixel_values"][-patches:].contiguous().numpy().tobytes()).hexdigest()
                    if digest != expected["target_pixels_sha256"]:
                        raise ValueError("actual evaluator target processing differs from the frozen diagnostic")
                    value = scorer.score_rgb(rgb)
                    result = {"processor_target_parity": True, "target_pixels_sha256": digest,
                        "input_tokens": int(inputs["input_ids"].shape[-1]), "score": value,
                        "expected_reward": expected["reward"], "expected_width_percent": expected["width_percent"],
                        "native_response_parity": value["score"] == expected["reward"]
                            and value["width_percent"] == expected["width_percent"]}
                    store.checkpoint(result)
                    if not result["native_response_parity"]:
                        raise ValueError("actual frozen evaluator response differs; inspect recorded parity audit before learning")
                    return result
                unit("audit", {"category": "actual_frozen_evaluator_parity"}, native_audit)
                data = restore_stationary(source_info)
                data_info = unit("dataset", {}, lambda store: manifest(data, source_info))
                forecasts["stationary_scoring_hours"] = data_info["unique_images"]*1.6755/3600
                progress("stationary source audited; frozen re-scoring", force=True)

                def score_stationary(store):
                    jobs = image_jobs(data)
                    state = store.load_checkpoint(trusted=True) or {"scores": {}, "cursor": 0, "seconds": 0.}
                    frames = {}
                    for job in jobs[state["cursor"]:]:
                        tick = time.monotonic()
                        bank = frames.setdefault(job["frames_root"], FrameStore(job["frames_root"], readonly=True))
                        state["scores"][job["frame_ref"]] = scorer.score_rgb(bank.get(job["frame_ref"]))
                        state["cursor"] += 1
                        state["seconds"] += time.monotonic()-tick
                        store.checkpoint(state, progress={"phase": 3, "status": "stationary image scores",
                            "counts": {"scored": state["cursor"], "target": len(jobs)}})
                        progress()
                    return {"scores": state["scores"], "scored_images": state["cursor"],
                        "unique_images": len(jobs), "seconds": state["seconds"], "scorer": scorer.provenance,
                        "scorer_identity": scorer.identity, "new_observations": 0}

                scored = unit("scoring", {}, score_stationary)
                edges = torch.linspace(-6., 12., 162, dtype=torch.float32)
                pristine = pristine_from_distilled(data)
                assets, initial_panels, readiness = {}, {}, []
                for condition in CONDITIONS:
                    cfg = configuration(condition, offline_fit=True)
                    labeled = relabel(data, scored["scores"], condition, edges)
                    fit_settings = {"condition": condition, "baseline_coded": labeled["baseline_coded"],
                        "source_fit_completion": source_info["fit_completion_sha256"]}
                    def initializer(store, fitted_state=None):
                        return RewardInitialization(model, cfg, labeled["train"], labeled["validation"], store, progress,
                            pristine=pristine, acquired_count=data["acquired_count"], fitted_state=fitted_state,
                            distill_receipt=data["distill_receipt"], shared_kl_selection=data["shared_kl_selection"])
                    fit = unit("fit", fit_settings, lambda store: initializer(store).run_fit())
                    fitted = load_completed_state(fit["unit_path"])
                    bootstrap = unit("bootstrap", {"condition": condition, "fit": sha256_file(Path(fit["unit_path"])/"COMPLETED")},
                        lambda store: initializer(store, fitted).run_fisher())
                    assets[condition] = {"path": bootstrap["unit_path"],
                        "id": sha256_file(Path(bootstrap["unit_path"])/"COMPLETED")}
                    if stage == "initialize":
                        continue
                    initial = load_completed_state(assets[condition]["path"])
                    cfg = configuration(condition)
                    def evaluate_initial(store):
                        runner = RewardDevelopmentRun(model, cfg, simulation(BASE_RECIPE), store, progress,
                            mode="pilot", seed=197091, max_transitions=640, initial=initial,
                            condition=condition, scorer=scorer, evaluations=panels(BASE_RECIPE),
                            initial_asset_id=assets[condition]["id"])
                        if runner.s["stage"] == "initial_evaluation":
                            runner.evaluate("initial")
                        runner._finish_evaluation_scores()
                        if runner.s["acquired_count"] or runner.s["ready"] or runner.s["eligible"]:
                            raise ValueError("initial evaluation admitted training evidence")
                        return {"condition": condition, "evaluations": runner.s["evaluations"],
                            "videos": runner.s["videos"], "timings": runner.s["timings"], "new_observations": 0,
                            "initial_asset_id": assets[condition]["id"]}
                    initial_panel = unit("initial_evaluation", {"condition": condition,
                        "initial_asset_id": assets[condition]["id"]}, evaluate_initial)
                    initial_panels[condition] = initial_panel["evaluations"]
                    def live_check(store):
                        return RewardDevelopmentRun(model, cfg, simulation(BASE_RECIPE), store, progress,
                            mode="pilot", seed=197092, max_transitions=48, initial=initial,
                            condition=condition, scorer=scorer, evaluations=(),
                            initial_asset_id=assets[condition]["id"]).run()
                    readiness.append(unit("readiness", {"condition": condition, "seed": 197092,
                        "max_transitions": 48, "initial_asset_id": assets[condition]["id"]}, live_check))
                if stage == "initialize":
                    progress("both reward-compatible initial assets complete", force=True, final=True)
                    return {"status": "initialization_complete", "assets": assets}
                live_seconds = sum(r["timings"]["active_seconds"]*640/max(1,r["acquired_count"]) for r in readiness)*2
                # Historical four-pilot evaluation scoring needed about 2,090 images;
                # two separately fitted initial panels add a conservative allowance.
                eval_seconds = 2600*scored["seconds"]/max(1,scored["scored_images"])
                forecasts["four_pilots_hours"] = [live_seconds/3600, (1.3*live_seconds+eval_seconds)/3600]
                forecasts["heldout_scoring_hours"] = eval_seconds/3600
                progress("live checks complete; measured paired-pilot forecast published", force=True)
                if stage == "readiness":
                    progress("ready; full pilots pending", force=True, final=True)
                    return {"status": "readiness_complete", "forecast": forecasts}
                completed, omitted = [], []
                for index, seed in enumerate(DEVELOPMENT_SEEDS):
                    order = CONDITIONS if index % 2 == 0 else tuple(reversed(CONDITIONS))
                    pair_seconds = (live_seconds+eval_seconds)/2
                    missing = [c for c in order if not (unit_path("pilot", {"condition": c,
                        "seed": seed, "max_transitions": 640, "initial_asset_id": assets[c]["id"]})/"COMPLETED").exists()]
                    if missing and not admit(pair_seconds*len(missing)/2):
                        omitted.append({"seed": seed, "conditions": missing, "reason": "complete-pair compute admission"})
                        break
                    for condition in order:
                        cfg = configuration(condition)
                        initial = load_completed_state(assets[condition]["path"])
                        settings = {"condition": condition, "seed": seed, "max_transitions": 640,
                            "initial_asset_id": assets[condition]["id"]}
                        completed.append(unit("pilot", settings, lambda store: RewardDevelopmentRun(
                            model, cfg, simulation(BASE_RECIPE), store, progress,
                            mode="pilot", seed=seed, max_transitions=640, initial=initial,
                            condition=condition, scorer=scorer, evaluations=panels(BASE_RECIPE),
                            initial_evaluations=initial_panels[condition],
                            initial_asset_id=assets[condition]["id"]).run()))
                progress("reward pilot complete; integrity and closeout", force=True, final=True)
                if sha256_file(PROJECT/"mathematical_overview.ipynb") != resolved["theory_sha256"]:
                    raise ValueError("reserved theory notebook changed")
                from numerical_experiment.picar.report import read_completed_unit
                seals = []
                for marker in sorted((root/"units").glob("*/COMPLETED")):
                    read_completed_unit(marker.parent)
                    seals.append({"path": str(marker.parent), "sha256": sha256_file(marker)})
                summary = {"schema_version": 1, "phase": 3, "version": VERSION, "kind": "reward_campaign",
                    "smoke_only": True, "development_only": True, "conditions": list(CONDITIONS),
                    "pilots": [r["unit_path"] for r in completed], "completed_trajectories": len(completed),
                    "omitted": omitted, "assets": assets, "forecast": forecasts,
                    "charged_seconds": elapsed(), "max_hours": max_hours, "completion_seals": seals,
                    "next_phase_authorized": False, "physical_motion": False}
                parent.checkpoint({"charged_seconds": elapsed(), "forecasts": forecasts, "stage": "done"},
                    progress={**summary, "status": "complete" if not omitted else "bounded_partial_complete"})
                parent.complete(summary)
                refresh()
                return summary
            except (InterruptedError, TimeoutError):
                progress("paused at committed checkpoint", force=True, final=True)
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
    parser.add_argument("--stage", choices=("all", "software", "initialize", "readiness", "pilots"), default="all")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(read_json(args.output/"orchestration/progress.json"), indent=2))
        return
    if args.command == "stop":
        if not (args.output/"orchestration/config.json").exists():
            parser.error("cannot stop a campaign that has not started")
        (args.output/"STOP").touch(exist_ok=True)
        print("Checkpoint-boundary stop requested")
        return
    if args.command == "report":
        from .report import build_notebook
        print(build_notebook(args.output, output=CANONICAL_NOTEBOOK, base_notebook=args.output/"history.ipynb"))
        return
    if args.clear_stop:
        if not args.resume:
            parser.error("--clear-stop requires intentional --resume")
        (args.output/"STOP").unlink(missing_ok=True)
    result = run(args.output, args.source, args.references, model_path=args.model_path,
        offline=args.offline, resume=args.resume, max_hours=args.max_hours, stage=args.stage)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
