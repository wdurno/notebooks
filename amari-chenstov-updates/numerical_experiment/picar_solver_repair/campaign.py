"""Execute Plan 19 Phase 3 Fix 2 with immutable, resumable units."""

from __future__ import annotations

import argparse
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
from numerical_experiment.picar_phase2.campaign import PROJECT, read_json
from .config import (VERSION, MAX_HOURS, PRIOR_SECONDS, CLOSEOUT_SECONDS, ACCEPTANCE_SECONDS,
    BASE_RECIPE, SOLVERS, FD_STRENGTH, DEVELOPMENT_SEEDS, DEFAULT_OUTPUT, DEFAULT_SOURCE,
    MODEL_SEED, NORMALIZATION, configuration, panels, simulation, solver_steps, sources, protocol)


def identity(source, model_path, max_hours):
    return {"schema_version": 1, "version": VERSION, "phase": 3,
        "source_campaign": str(Path(source).resolve()),
        "source_completion": sha256_file(Path(source) / "orchestration/COMPLETED"),
        "source_sha256": sources(), "protocol": protocol(), "max_hours": max_hours,
        "model_source": str(Path(model_path).resolve()) if model_path else configuration().model.model_id}


def run(output=DEFAULT_OUTPUT, source=DEFAULT_SOURCE, *, model_path=None,
        offline=False, resume=False, max_hours=MAX_HOURS, stage="all"):
    if not 0 < max_hours <= MAX_HOURS:
        raise ValueError("solver repair has a nine-hour additional compute ceiling")
    root = Path(output).resolve()
    resolved = identity(source, model_path, max_hours)
    root.mkdir(parents=True, exist_ok=True)
    stopped = StopController(root)
    stopped.install()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        with RunStore(root / "orchestration", resolved, resume=resume, checkpoint_policy="journal-v1") as parent:
            if parent.completed:
                return read_json(parent.root / "summary.json")
            state = parent.load_checkpoint(trusted=True) or {"charged_seconds": ACCEPTANCE_SECONDS}
            base, started = state["charged_seconds"], time.monotonic()
            active, label = None, "acceptance"
            last_save = last_report = 0.
            execution = None

            def elapsed():
                return base + time.monotonic() - started

            def report():
                from .report import build_notebook
                build_notebook(PROJECT / "cache/numerical_experiment/picar",
                    output=PROJECT / "numerical_experiment/picar/plan19_results.ipynb")

            def progress(status=None, *, force=False, final=False):
                nonlocal label, last_save, last_report
                if status is not None:
                    label = status
                now = time.monotonic()
                if parent.completed:
                    return
                if force or now - last_save >= 45 or stopped.check():
                    child = read_json(active / "progress.json") if active and (active / "progress.json").exists() else {}
                    parent.checkpoint({"charged_seconds": elapsed() + (0. if final else 60.)}, progress={
                        "schema_version": 1, "version": VERSION, "phase": 3, "status": label,
                        "repair_active_seconds": elapsed(), "cumulative_study_seconds": PRIOR_SECONDS + elapsed(),
                        "remaining_budget_seconds": max(0., max_hours*3600-elapsed()),
                        "active_unit": str(active) if active else None, "active_unit_progress": child,
                        "completed_trajectories": len(list((root / "units").glob("pilot-*/COMPLETED"))),
                        "next_phase_authorized": False})
                    last_save = now
                if force or now - last_report >= 900:
                    report()
                    last_report = now
                if not final and stopped.check():
                    raise InterruptedError("solver repair checkpointed; intentionally remove STOP before resume")
                if not final and elapsed() > max_hours*3600-CLOSEOUT_SECONDS:
                    raise TimeoutError("solver repair compute ceiling reached; closeout reserve retained")

            def unit_path(name, settings):
                cfg = {**resolved, "operation": name, "settings": settings}
                digest = hashlib.sha256(canonical_json(cfg).encode()).hexdigest()[:12]
                return root / "units" / (name + "-" + digest)

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
                            raise RuntimeError("scientific source changed; preserve this version before correcting")
                        result = {**result, "schema_version": 1, "kind": "solver_" + name,
                            "phase": 3, "smoke_only": True, "development_only": True,
                            "confirmatory": False, "settings": settings, "unit_path": str(path),
                            "active_seconds": time.monotonic() - tick, "execution": execution,
                            "media_sha256": {str(p.relative_to(path)): sha256_file(p)
                                for p in path.glob("*.mp4")}}
                        if store._current is None:
                            store.checkpoint({"stage": "done"})
                        store.complete(result)
                active = None
                print(json.dumps({"unit": name, "settings": settings, "path": str(path),
                    "seconds": result["active_seconds"], "status": result.get("status", "complete")}), flush=True)
                progress(name + " complete", force=True)
                return result

            def provenance(store):
                archive_path = store.root / "sources.tar.gz"
                with tarfile.open(archive_path, "w:gz") as archive:
                    for name in resolved["source_sha256"]:
                        archive.add(PROJECT / name, arcname=name)
                return {"source_sha256": resolved["source_sha256"], "archive_sha256": sha256_file(archive_path)}

            def software(store):
                paths = sorted(str(p.relative_to(PROJECT)) for p in (PROJECT / "test/unit").glob("test_picar_solver*.py"))
                if not paths:
                    raise RuntimeError("solver tests are missing")
                check = subprocess.run([sys.executable, "-m", "pytest", "-q", *paths],
                    cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, timeout=900)
                store._write_root("pytest.txt", check.stdout.encode())
                if check.returncode:
                    raise RuntimeError("solver acceptance failed: " + check.stdout[-4000:])
                return {"passed": True, "test_result": check.stdout.strip().splitlines()[-1]}

            try:
                unit("provenance", {}, provenance)
                from .data import inspect_source, restore, manifest, initial_panels
                source_info = unit("integrity", {}, lambda store: inspect_source(source))
                unit("software", {}, software)
                if stage == "software":
                    progress("software complete; recorded comparison pending", force=True)
                    return {"status": "software_complete"}

                import torch
                from numerical_experiment.picar.models import load_qwen_model
                from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
                from .recorded import RecordedRun, select_solver
                from .student import SolverDevelopmentRun
                torch.set_num_threads(2)
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable; solver experiment must use the actual model")
                progress("loading common quantized model", force=True)
                raw = load_qwen_model(**({"model_path": model_path} if model_path else {}),
                    local_files_only=offline, seed=MODEL_SEED, top_layers=1, ltm_rank=4, stm_rank=1,
                    hidden_size=64, context_length=4, return_bins=161,
                    min_pixels=56*56, max_pixels=19200, max_tokens=2048).eval()
                raw.encoder.enable_prefix_cache = True
                model = NormalizedStudent(raw, mode=NORMALIZATION, eps=1e-6).eval()
                execution = {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0), "dtype": "float32",
                    "model_provenance": model.provenance}
                initial, segments, validation = restore(source_info)
                unit("dataset", {}, lambda store: manifest(source_info, segments, validation))

                screen_results = []
                def reusable_recorded(settings):
                    """Reference valid v2 measurements; numerical code/data must match."""
                    previous = root.parent / "solver-fix2-v2" / "units"
                    from numerical_experiment.picar.report import read_completed_unit
                    for marker in sorted(previous.glob("recorded-*/COMPLETED")):
                        item = read_completed_unit(marker.parent)
                        old = item["stored_config"]
                        if (old.get("settings") != settings or item["summary"].get("status") != "complete"
                                or old.get("source_completion") != resolved["source_completion"]):
                            continue
                        allowed = {"numerical_experiment/picar_solver_repair/" + name
                                   for name in ("config.py", "campaign.py", "recorded.py")}
                        if any(resolved["source_sha256"].get(name) != expected
                               for name, expected in old["source_sha256"].items() if name not in allowed):
                            raise ValueError("reused recorded numerical source differs")
                        keys = ("metric", "beta2", "optimizer_epsilon", "momentum", "ridge", "base_recipe",
                                "fd_strength", "screen_seed", "screen_events_per_segment", "screen_segments", "screen_initialization")
                        if any(old["protocol"][key] != resolved["protocol"][key] for key in keys):
                            raise ValueError("reused recorded numerical protocol differs")
                        payload = item["summary"]
                        return {**payload, "reused_from": str(marker.parent),
                            "reused_completion_sha256": sha256_file(marker),
                            "measurement_active_seconds": payload["active_seconds"],
                            "reuse_reason": "Completed full-context v2 measurement; only later failure classification/supervision changed.",
                            "not_an_additional_replica": True}
                    return None

                def run_screen(rate):
                    results = []
                    for order, solver in enumerate(SOLVERS):
                        conditions = ("replay_ewc", "stm_fd") if order % 2 == 0 else ("stm_fd", "replay_ewc")
                        for condition in conditions:
                            strength = FD_STRENGTH if condition == "stm_fd" else 0.
                            recipe = replace(BASE_RECIPE, name=condition, fd_strength=strength,
                                learning_rate=rate, optimizer_steps=solver_steps(solver))
                            cfg = configuration(recipe)
                            settings = {"solver_recipe": solver, "condition": condition,
                                "learning_rate": recipe.learning_rate, "events_per_segment": 64}
                            def recorded(store):
                                inherited = reusable_recorded(settings)
                                if inherited is not None:
                                    return inherited
                                return RecordedRun(model, cfg, initial, segments, validation, store, progress,
                                    solver_recipe=solver, condition=condition).run()
                            results.append(unit("recorded", settings, recorded))
                    return results

                # Rotate objective order to reduce systematic warm-cache timing bias.
                screen_results = run_screen(BASE_RECIPE.learning_rate)
                selection_results, selected_rate = screen_results, BASE_RECIPE.learning_rate
                if any(r.get("status") == "numerical_failure" for r in screen_results):
                    selected_rate = .001
                    selection_results = run_screen(selected_rate)
                    screen_results += selection_results
                decision = unit("selection", {}, lambda store: {**select_solver(selection_results),
                    "selected_learning_rate": selected_rate,
                    "common_reduction_reason": "One shared .003-to-.001 correction after a numerical candidate failure."
                        if selected_rate != BASE_RECIPE.learning_rate else None,
                    "recorded_unit_paths": [r["unit_path"] for r in screen_results]})
                if stage == "screen":
                    progress("recorded comparison complete; live development pending", force=True)
                    return decision

                solver = decision["selected_solver"]
                pilots, omitted = [], []
                specs_initial = initial_panels(source_info)
                if solver is not None:
                    for condition in ("replay_ewc", "stm_fd"):
                        recipe = replace(BASE_RECIPE, name=condition,
                            fd_strength=FD_STRENGTH if condition == "stm_fd" else 0.,
                            learning_rate=selected_rate,
                            optimizer_steps=solver_steps(solver))
                        def readiness(store):
                            runner = SolverDevelopmentRun(model, configuration(recipe), simulation(recipe),
                                store, progress, solver_recipe=solver, mode="pilot", seed=196091,
                                max_transitions=48, initial=initial, evaluations=[])
                            result = runner.run()
                            return {**result, "scope": "48_interval_live_latency_only",
                                "not_acquisition_evidence": True}
                        unit("readiness", {"solver_recipe": solver, "condition": condition,
                            "seed": 196091, "max_transitions": 48}, readiness)
                    for seed in DEVELOPMENT_SEEDS:
                        for condition in ("replay_ewc", "stm_fd"):
                            recipe = replace(BASE_RECIPE, name=condition,
                                fd_strength=FD_STRENGTH if condition == "stm_fd" else 0.,
                                learning_rate=selected_rate,
                                optimizer_steps=solver_steps(solver))
                            settings = {**asdict(recipe), "seed": seed, "solver_recipe": solver,
                                "initial_asset": source_info["asset_id"]}
                            estimate = 4800. if solver == "proximal_3" else 3600.
                            if not unit_path("pilot", settings).exists() and elapsed()+1.1*estimate+CLOSEOUT_SECONDS > max_hours*3600:
                                omitted.append({"seed": seed, "condition": condition, "reason": "compute admission"})
                                continue
                            def pilot(store):
                                runner = SolverDevelopmentRun(model, configuration(recipe), simulation(recipe),
                                    store, progress, solver_recipe=solver, mode="pilot", seed=seed,
                                    max_transitions=recipe.transitions, initial=initial, evaluations=panels(recipe),
                                    initial_evaluations=specs_initial, initial_asset_id=source_info["asset_id"])
                                return runner.run()
                            pilots.append(unit("pilot", settings, pilot))
                else:
                    omitted.append({"reason": "no numerically usable solver", "condition": "paired pilots"})
                charge = elapsed()+CLOSEOUT_SECONDS
                result = {"schema_version": 1, "kind": "solver_campaign", "phase": 3,
                    "version": VERSION, "status": "complete_for_review", "selected_solver": solver,
                    "selection": decision, "recorded": [r["unit_path"] for r in screen_results],
                    "pilots": [r["unit_path"] for r in pilots], "omitted": omitted,
                    "completed_trajectories": len(pilots), "repair_active_seconds": charge,
                    "cumulative_study_seconds": PRIOR_SECONDS+charge,
                    "closeout_reserve_charged_seconds": CLOSEOUT_SECONDS, "next_phase_authorized": False}
                parent.checkpoint({"charged_seconds": charge}, progress=result)
                parent.complete(result)
                report()
                return result
            except BaseException as exc:
                if not parent.completed:
                    progress("checkpointed: " + type(exc).__name__, force=True, final=True)
                raise
    finally:
        stopped.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("software", "screen", "all"), default="all")
    args = parser.parse_args()
    result = run(args.output, args.source, model_path=args.model_path, offline=args.offline,
        resume=args.resume, stage=args.stage)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
