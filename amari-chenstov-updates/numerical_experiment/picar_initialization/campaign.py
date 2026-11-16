"""Execute only Plan 19 Phase 3 Fix 1, with a ten-hour independent ceiling."""

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
from numerical_experiment.picar.runtime import StopController, sha256_file, check_storage
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_phase2.campaign import PROJECT, load_completed_state, read_json
from .config import (VERSION, MAX_HOURS, PRIOR_SECONDS, MODEL_SEED, NORMALIZATION,
    DATA_LADDER, CLOSEOUT_SECONDS, COLLECTION_SEED, DEVELOPMENT_SEEDS,
    DEFAULT_RECIPE, configuration, simulation, panels, sources, protocol)


class ComputeLimit(TimeoutError):
    pass


def run_identity(teacher_source, model_path=None, max_hours=MAX_HOURS):
    source = Path(teacher_source).resolve()
    return {"schema_version": 1, "version": VERSION, "phase": 3,
        "protocol": protocol(), "source_sha256": sources(), "max_hours": max_hours,
        "teacher_source": str(source), "teacher_completion_sha256": sha256_file(source / "COMPLETED"),
        "pre_execution_validation_reserve_seconds": 600.,
        "model_source": str(Path(model_path).resolve()) if model_path else configuration().model.model_id}


def verify_history(root):
    """Read-only audit of historical seals, sources and the reserved theory file."""
    from numerical_experiment.picar.report import read_completed_unit
    paths = []
    for name in ("development-v1", "development-retention-v1"):
        campaign = Path(root) / name
        if not campaign.is_dir():
            raise ValueError("missing historical campaign: " + str(campaign))
        for marker in sorted(campaign.rglob("COMPLETED")):
            read_completed_unit(marker.parent)
            paths.append(str(marker.parent))
        resolved = read_json(campaign / "orchestration/config.json")
        for name, expected in resolved["source_sha256"].items():
            if sha256_file(PROJECT / name) != expected:
                raise ValueError("historical scientific source differs: " + name)
    theory = sha256_file(PROJECT / "mathematical_overview.ipynb")
    if theory != "eae6e00ce327334cca56755714d1965f21e0ff1dc4e8d3d14487eaa2b7ec8487":
        raise ValueError("reserved mathematical overview differs from the reviewed version")
    return {"completed_units_verified": len(paths), "paths": paths,
            "theory_sha256": theory, "read_only": True}


def run(output, teacher_source, *, model_path=None, offline=False, resume=False,
        max_hours=MAX_HOURS, stage="all"):
    if not 0 < max_hours <= MAX_HOURS:
        raise ValueError("initialization repair has a maximum ten-hour compute allocation")
    root, teacher_source = Path(output).resolve(), Path(teacher_source).resolve()
    identity = run_identity(teacher_source, model_path, max_hours)
    fingerprint = identity["source_sha256"]
    root.mkdir(parents=True, exist_ok=True)
    stop = StopController(root)
    stop.install()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        with RunStore(root / "orchestration", identity, resume=resume,
                      checkpoint_policy="journal-v1") as parent:
            if parent.completed:
                return read_json(parent.root / "summary.json")
            saved = parent.load_checkpoint(trusted=True) or {"charged_seconds": 600.}
            started = time.monotonic()
            active, label = None, "software acceptance"
            last_save, last_report = 0., 0.
            execution = None

            def elapsed():
                return saved["charged_seconds"] + time.monotonic() - started

            def refresh():
                from .report import build_notebook
                build_notebook(PROJECT / "cache/numerical_experiment/picar")

            def progress(status=None, *, force=False, ignore_limit=False):
                nonlocal last_save, last_report, label
                if status is not None:
                    label = status
                now = time.monotonic()
                if force or now - last_save >= 45 or stop.check():
                    child = read_json(active / "progress.json") if active and (active / "progress.json").exists() else {}
                    # Reserve one heartbeat to cover sudden process loss/replayed work.
                    charge = elapsed() + (0. if ignore_limit else 60.)
                    parent.checkpoint({"charged_seconds": charge}, progress={
                        "schema_version": 1, "phase": 3, "version": VERSION, "status": label,
                        "phase3_active_seconds": elapsed(), "repair_active_seconds": elapsed(),
                        "cumulative_study_seconds": PRIOR_SECONDS + elapsed(),
                        "remaining_budget_seconds": max(0., max_hours * 3600 - elapsed()),
                        "active_unit": str(active) if active else None,
                        "active_unit_progress": child,
                        "completed_trajectories": len(list((root / "units").glob("pilot-*/COMPLETED"))),
                        "next_phase_authorized": False})
                    last_save = now
                if force or now - last_report >= 900:
                    refresh()
                    last_report = now
                if stop.check() and not ignore_limit:
                    raise InterruptedError("repair checkpointed; remove STOP only before intentional resume")
                if not ignore_limit and elapsed() >= max_hours * 3600 - CLOSEOUT_SECONDS:
                    raise ComputeLimit("repair ceiling reached; preserve incomplete units and reserve closeout")

            def admit(seconds):
                return max_hours * 3600 - elapsed() - CLOSEOUT_SECONDS >= 1.15 * seconds

            def unit_path(name, settings):
                resolved = {**identity, "operation": name, "settings": settings}
                suffix = hashlib.sha256(canonical_json(resolved).encode()).hexdigest()[:12]
                return root / "units" / f"{name}-{suffix}"

            def unit(name, settings, callback):
                nonlocal active
                resolved = {**identity, "operation": name, "settings": settings}
                path = unit_path(name, settings)
                active = path
                progress(name, force=True)
                check_storage(root, quota_bytes=100_000_000_000, reserve_bytes=20_000_000_000)
                with RunStore(path, resolved, resume=resume, checkpoint_policy="journal-v1") as store:
                    if store.completed:
                        result = read_json(path / "summary.json")
                    else:
                        tick = time.monotonic()
                        result = callback(store)
                        if sources() != fingerprint:
                            raise RuntimeError("scientific sources changed; version the correction before resuming")
                        result = {**result, "schema_version": 1, "kind": "initialization_" + name,
                            "phase": 3, "smoke_only": True, "development_only": True,
                            "settings": settings, "unit_path": str(path),
                            "active_seconds": time.monotonic() - tick,
                            "execution": execution,
                            "media_sha256": {str(p.relative_to(path)): sha256_file(p)
                                for p in path.rglob("*") if p.suffix in {".png", ".mp4"}}}
                        if store._current is None:
                            store.checkpoint({"stage": "done"})
                        store.complete(result)
                active = None
                print(json.dumps({"unit": name, "path": str(path), "seconds": result["active_seconds"]}), flush=True)
                progress(name + " complete", force=True)
                return result

            def software(store):
                paths = sorted(str(p.relative_to(PROJECT)) for p in
                               (PROJECT / "test/unit").glob("test_picar_initialization*.py"))
                if not paths:
                    raise RuntimeError("repair acceptance tests are missing")
                result = subprocess.run([sys.executable, "-m", "pytest", "-q", *paths],
                    cwd=PROJECT, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
                store._write_root("pytest.txt", result.stdout.encode())
                if result.returncode:
                    raise RuntimeError("repair software acceptance failed: " + result.stdout[-4000:])
                return {"passed": True, "test_result": result.stdout.strip().splitlines()[-1]}

            try:
                def provenance(store):
                    archive = store.root / "sources.tar.gz"
                    with tarfile.open(archive, "w:gz") as target:
                        for name in fingerprint:
                            target.add(PROJECT / name, arcname=name)
                    return {"source_sha256": fingerprint, "archive_sha256": sha256_file(archive)}
                unit("provenance", {}, provenance)
                unit("integrity", {}, lambda store: verify_history(root.parent))
                unit("software", {}, software)
                if stage == "software":
                    progress("software complete; GPU execution pending", force=True)
                    return {"status": "software_complete"}

                import torch
                from numerical_experiment.picar.models import load_qwen_model
                from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
                from numerical_experiment.picar_phase2_rescue.data import restore_demonstrations
                from numerical_experiment.picar_phase3.student import DevelopmentRun
                from numerical_experiment.picar_phase3.calibration import optimizer_probe, fd_probe, explore_controls
                from numerical_experiment.picar.learner import MemoryLearner
                from .data import (CollectionRun, namespace_data, split_records, prefix_episodes,
                                   coverage, evaluate_initial)
                from .numerics import InitializationFit

                torch.set_num_threads(2)
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable; do not silently run this experiment on CPU")
                progress("loading quantized Qwen", force=True)
                raw = load_qwen_model(**({"model_path": model_path} if model_path else {}),
                    local_files_only=offline, seed=MODEL_SEED, top_layers=1, ltm_rank=4, stm_rank=1,
                    hidden_size=64, context_length=4, return_bins=161,
                    min_pixels=56*56, max_pixels=19200, max_tokens=2048).eval()
                raw.encoder.enable_prefix_cache = True
                model = NormalizedStudent(raw, mode=NORMALIZATION, eps=1e-6).eval()
                config = configuration()
                starting_learner = MemoryLearner(model, config.learning, config.seeds(0))
                pristine = starting_learner.state_dict()
                from numerical_experiment.picar_phase3.campaign import learning_metadata
                execution = learning_metadata(starting_learner)
                original = namespace_data(restore_demonstrations(teacher_source), "original")
                records, episodes = original["records"], original["episodes"]
                acquired = original["acquired_count"]
                chunk_index = 0
                fixed_validation_ids = None
                fitted, selected, selected_train, fixed_validation = [], None, None, None
                fit_training_sets = {}

                for target in DATA_LADDER:
                    while len(split_records(records)[0]) < target:
                        next_settings = {"index": chunk_index + 1, "seed": COLLECTION_SEED + chunk_index + 1,
                            "max_intervals": 160, "stationary_A": True}
                        if not unit_path("collection", next_settings).exists() and not admit(600.):
                            break
                        chunk_index += 1
                        def collect(store):
                            collector = CollectionRun(model, config, simulation(DEFAULT_RECIPE, bootstrap=True),
                                store, progress, seed=COLLECTION_SEED + chunk_index, max_transitions=160)
                            return collector.run()
                        chunk = unit("collection", {"index": chunk_index, "seed": COLLECTION_SEED + chunk_index,
                            "max_intervals": 160, "stationary_A": True}, collect)
                        state = load_completed_state(chunk["unit_path"])
                        restored = namespace_data({"records": state["eligible"], "episodes": state["episodes"],
                            "acquired_count": state["acquired_count"]}, f"collection-{chunk_index}", arrival_offset=acquired)
                        records.extend(restored["records"])
                        episodes.extend(restored["episodes"])
                        acquired += restored["acquired_count"]
                    train, validation = split_records(records, fixed_validation_ids=fixed_validation_ids)
                    if fixed_validation_ids is None:
                        fixed_validation_ids = [r["record_id"] for r in validation]
                    train = prefix_episodes(train, target)
                    fixed_validation = validation
                    dataset = unit("dataset", {"target": target}, lambda store: {
                        "target": target, "train_count": len(train), "validation_count": len(validation),
                        "train_ids": [r["record_id"] for r in train],
                        "validation_ids": fixed_validation_ids, "acquired_count": acquired,
                        "coverage": coverage(records, episodes), "original_acquisitions_reused": original["acquired_count"],
                        "new_acquisitions": acquired - original["acquired_count"],
                        "episode_split": "hash_modulo_five; fixed_first_rung_validation",
                        "stationary_A": True, "no_outcome_filtering": True})
                    fit_settings = {"target": target, "dataset": dataset["unit_path"]}
                    if len(train) < min(target, 128) or (
                            not unit_path("fit", fit_settings).exists() and not admit(900.)):
                        break
                    def fit(store):
                        runner = InitializationFit(model, config, train, validation, store, progress,
                            pristine=pristine, acquired_count=acquired)
                        return runner.run_fit()
                    result = unit("fit", fit_settings, fit)
                    fitted.append(result)
                    fit_training_sets[result["unit_path"]] = train
                    selected, selected_train = result, train
                    rung_state = load_completed_state(result["unit_path"])
                    for fit_stage in ("distillation", "accepted_nll"):
                        def capability(store):
                            learner_state = rung_state["distilled_learner"] if fit_stage == "distillation" else rung_state["learner"]
                            evaluation_state = {"stage": "done", "learner": learner_state,
                                "eligible": [], "retired": [], "acquired_count": acquired,
                                "distill_receipt": rung_state["distill_receipt"],
                                "distinct_bootstrap_samples": len(train),
                                "selected_exploration_std": config.learning.exploration_std}
                            panel = copy.deepcopy(panels(DEFAULT_RECIPE)[:1])
                            panel[0]["seeds"] = panel[0]["seeds"][:5]
                            return {"checkpoint": fit_stage, "train_count": len(train),
                                "target": target, "small_panel_not_confirmation": True,
                                **evaluate_initial(model, config, evaluation_state, store, progress,
                                    evaluations=panel, asset_id="fit:" + sha256_file(
                                        Path(result["unit_path"]) / "COMPLETED") + ":" + fit_stage)}
                        unit("evaluation", {"target": target, "checkpoint": fit_stage,
                            "panel": "first_five_easy_development_layouts", "fit": result["unit_path"]}, capability)
                    def growth(store):
                        reason = "continue admitted nested data ladder"
                        continue_ladder = True
                        if elapsed() > 4 * 3600:
                            continue_ladder, reason = False, "preserve development reserve after four hours"
                        elif len(train) < target:
                            continue_ladder, reason = False, "collection admission did not reach target"
                        elif len(fitted) >= 2 and result.get("readiness", {}).get("ready"):
                            old = fitted[-2]["selection"]["nll"]["validation_objective"]
                            new = result["selection"]["nll"]["validation_objective"]
                            if old - new < .01 * max(1., abs(old)):
                                continue_ladder, reason = False, "less than one percent fixed-validation improvement"
                        return {"continue_ladder": continue_ladder, "reason": reason,
                            "charged_seconds_at_decision": elapsed(), "target": target}
                    # A pause must not retroactively change which rungs were admitted.
                    decision = unit("growth", {"target": target, "fit": result["unit_path"]}, growth)
                    if not decision["continue_ladder"]:
                        break

                if selected is None:
                    raise ComputeLimit("no fitted initializer completed inside admission budget")

                selected = min(fitted, key=lambda r: r["selection"]["nll"]["validation_objective"])
                selected_train = fit_training_sets[selected["unit_path"]]
                fit_state = load_completed_state(selected["unit_path"])
                def fisher(store):
                    runner = InitializationFit(model, config, selected_train, fixed_validation, store, progress,
                        pristine=pristine, acquired_count=acquired, fitted_state=fit_state)
                    return runner.run_fisher()
                asset = unit("bootstrap", {"fit": selected["unit_path"], "draws": 4}, fisher)
                initial = load_completed_state(asset["unit_path"])
                asset_id = sha256_file(Path(asset["unit_path"]) / "COMPLETED")

                evaluation_units = []
                for checkpoint in ("distillation", "accepted_nll"):
                    evaluation_initial = initial
                    evaluation_id = asset_id
                    specs = panels(DEFAULT_RECIPE)
                    if checkpoint == "distillation":
                        if "distilled_learner" not in fit_state:
                            raise ValueError("fit lacks a preserved distillation checkpoint")
                        evaluation_initial = {**initial, "learner": fit_state["distilled_learner"]}
                        evaluation_id = sha256_file(Path(selected["unit_path"]) / "COMPLETED")
                        specs = specs[:1]
                    def evaluate(store):
                        return {"checkpoint": checkpoint, **evaluate_initial(model, config,
                            evaluation_initial, store, progress, evaluations=specs, asset_id=evaluation_id)}
                    evaluation_units.append(unit("evaluation", {"checkpoint": checkpoint,
                        "initial_asset": evaluation_id}, evaluate))

                def diagnostic_initial():
                    # Offline-only diagnostic access, never online archive replay.
                    return {**initial, "eligible": selected_train[:8]}

                def probe(store):
                    diagnostic = diagnostic_initial()
                    common = optimizer_probe(model, config, diagnostic)
                    return {**common, **fd_probe(model, config, diagnostic)}
                probes = unit("probe", {"initial_asset": asset_id}, probe)
                common = replace(DEFAULT_RECIPE, learning_rate=probes["selected_learning_rate"],
                    optimizer_steps=probes["selected_optimizer_steps"])
                fd_rows = probes.get("fd_strength_rows", [])
                strength = fd_rows[1]["fd_strength"] if len(fd_rows) >= 2 else None
                fd = replace(common, name="stm_fd", fd_strength=strength,
                    fd_step=probes["selected_fd_step"]) if strength is not None else None
                pilots, omitted = [], []
                initial_panels = evaluation_units[-1]["evaluations"]

                def pilot(recipe, seed, name="pilot"):
                    nonlocal initial_panels
                    def execute(store):
                        runner = DevelopmentRun(model, configuration(recipe), simulation(recipe), store, progress,
                            mode="pilot", seed=seed, max_transitions=recipe.transitions,
                            initial=initial, evaluations=panels(recipe), initial_evaluations=initial_panels,
                            initial_asset_id=asset_id)
                        summary = runner.run()
                        summary["limitations"] = ["Development only; no confirmatory sample-savings claim.",
                            "Shared repaired initialization; all initial training records archived, replay starts empty."]
                        return summary
                    result = unit(name, {**asdict(recipe), "seed": seed, "initial_asset": asset_id}, execute)
                    initial_panels = [p for p in result["evaluations"] if p["checkpoint"] == "initial"]
                    return result

                # Evaluate the repaired initializer even when optimization readiness is unresolved.
                ready = asset.get("readiness", {}).get("ready", False)
                if ready:
                    for seed in DEVELOPMENT_SEEDS:
                        for recipe in (replace(common, name="replay_ewc"), fd):
                            if recipe is None:
                                omitted.append({"seed": seed, "condition": "stm_fd", "reason": "FD resolution failed"})
                                continue
                            estimate = 4200. if recipe.fd_strength else 3120.
                            settings = {**asdict(recipe), "seed": seed, "initial_asset": asset_id}
                            if not unit_path("pilot", settings).exists() and not admit(estimate):
                                omitted.append({"seed": seed, "condition": recipe.name, "reason": "compute admission"})
                                continue
                            pilots.append(pilot(recipe, seed))
                else:
                    omitted.append({"condition": "paired development", "reason": "initialization readiness unresolved"})
                from numerical_experiment.picar_phase3.analysis import selection
                decision = selection(pilots, control_name="replay_ewc") if pilots else {
                    "ready_to_propose_confirmation": False,
                    "reason": "Initialization or development remains unresolved.",
                    "comparisons": [], "sample_savings_available": False}
                final_charge = elapsed() + CLOSEOUT_SECONDS
                result = {"schema_version": 1, "kind": "initialization_campaign", "phase": 3,
                    "status": "ready_for_checkin", "initialization_ready": ready,
                    "selected_fit": selected["unit_path"], "initial_asset": asset["unit_path"],
                    "fits": [r["unit_path"] for r in fitted], "pilots": [r["unit_path"] for r in pilots],
                    "initial_evaluations": [r["unit_path"] for r in evaluation_units],
                    "omitted": omitted, "next_phase_authorized": False,
                    "decision": decision,
                    "repair_active_seconds": final_charge, "phase3_active_seconds": final_charge,
                    "cumulative_study_seconds": PRIOR_SECONDS + final_charge,
                    "closeout_reserve_charged_seconds": CLOSEOUT_SECONDS,
                    "completed_trajectories": len(pilots)}
                parent.checkpoint({"charged_seconds": final_charge}, progress=result)
                parent.complete(result)
                refresh()
                return result
            except BaseException as exc:
                progress("checkpointed: " + type(exc).__name__, force=True, ignore_limit=True)
                raise
    finally:
        stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=Path("cache/numerical_experiment/picar/phase3/initialization-fix1-v1"))
    parser.add_argument("--teacher-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("software", "all"), default="all")
    args = parser.parse_args()
    result = run(args.output, args.teacher_source, model_path=args.model_path,
                 offline=args.offline, resume=args.resume, stage=args.stage)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
