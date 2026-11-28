"""Resumable Fix 5A scheduler; source Fix 5 artifacts are read-only."""

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.runtime import StopController
from numerical_experiment.picar.storage import RunStore, _atomic_bytes
from . import VERSION
from .reuse import digest, source_index, load_state, one

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "cache/numerical_experiment/picar/phase3/head-awac-fix5-v3"
OUTPUT = PROJECT / "cache/numerical_experiment/picar/phase3/head-awac-rank-fix5a-v1"
NOTEBOOK = PROJECT / "numerical_experiment/picar/plan19_results.ipynb"


def scalar(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar(v) for v in value]
    return value


def fingerprints():
    from numerical_experiment.picar_head_awac.config import source_hashes
    result = source_hashes()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        if p.name != "report.py":
            result[str(p.relative_to(PROJECT))] = digest(p)
    return result


def endpoint(result, environment):
    panels = [p for p in result["evaluations"] if p["environment"] == environment and p["n"] == 1024]
    if len(panels) != 1 or len(panels[0]["episodes"]) != 8:
        raise ValueError("retention selection needs the full eight-layout endpoint")
    return math.fsum(e["raw_return"] for e in panels[0]["episodes"]) / 8


def choose_correction(results, reference, reference_acquisition=None):
    gaps = [(abs(endpoint(row, "initial_A") - reference), row) for row in results
            if endpoint(row, "initial_A") < reference or reference_acquisition is None
            or endpoint(row, "moderate_A") < reference_acquisition]
    if not gaps:
        return None
    gap, row = max(gaps, key=lambda item: (item[0], item[1]["settings"]["method"] == "td_fd"))
    if gap <= 5:
        return None
    signed = endpoint(row, "initial_A") - reference
    return {"method": row["settings"]["method"], "ridge": .2 if signed < 0 else .05,
            "initial_retention_gap": signed, "reason": "largest absolute development retention gap"}


def run(output=OUTPUT, source=SOURCE, *, model_path=None, resume=False, offline=False,
        max_hours=16., support_seconds=0., stage="all", notebook=NOTEBOOK):
    import torch
    from numerical_experiment.picar_head_awac.learner import Recipe
    from numerical_experiment.picar_head_awac.model import FrozenFeatures
    from numerical_experiment.picar_head_awac.data import apply_parameters
    from .numerics import rebuild_initial, resource_audit, new_learner
    from .trajectory import RankTrajectory

    if (not math.isfinite(max_hours) or not 0 < max_hours <= 16
            or not math.isfinite(support_seconds) or support_seconds < 0):
        raise ValueError("Fix 5A has a sixteen-hour incremental ceiling")
    if stage not in ("profile", "development", "all"):
        raise ValueError("unknown Fix 5A execution stage")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "cache/matplotlib"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_num_threads(2)
    root, source, notebook = Path(output).resolve(), Path(source).resolve(), Path(notebook)
    source_config = json.loads((source / "orchestration/config.json").read_text())
    for name, expected in source_config["sources"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError("frozen numerical source changed: " + name)
    if digest(PROJECT / "mathematical_overview.ipynb") != source_config["theory_sha256"]:
        raise ValueError("theory notebook changed")
    units, receipts = source_index(source)
    identity = {"version": VERSION, "schema_version": 1, "source": str(source),
                "source_receipts": receipts, "sources": fingerprints(),
                "theory_sha256": source_config["theory_sha256"],
                "model_path": str(Path(model_path).resolve()) if model_path else None,
                "protocol": {"ranks": [0, 10, 32, 64], "preferred_rank": 64,
                             "development_seed": 199301, "assessment_seeds": [199101, 199102],
                             "new_intervals": 1024, "physical_motion": False}}
    root.mkdir(parents=True, exist_ok=True)
    stop = StopController(root)
    stop.install()
    try:
        with RunStore(root / "orchestration", identity, resume=resume,
                      checkpoint_policy="journal-v1", reserve_bytes=20_000_000_000) as parent:
            if parent.completed:
                return json.loads((parent.root / "summary.json").read_text())
            history = root / "history.ipynb"
            if not history.exists():
                _atomic_bytes(history, notebook.read_bytes())
            s = parent.load_checkpoint(trusted=True) or {
                "charged_seconds": support_seconds, "support_seconds": support_seconds,
                "completed": [], "unit_charges": {}, "selected": None, "rank": None,
                "correction": None, "status": "running", "forecast": {"remaining_hours": [10., 13.]}}
            s["status"] = "running"
            began, base = time.monotonic(), s["charged_seconds"]
            active, active_started, active_prior = None, None, 0.
            label, last_save, last_report = "source verification", 0., 0.
            planned = 12

            def elapsed():
                return base + time.monotonic() - began

            def refresh():
                from .report import build_notebook
                build_notebook(root, notebook, base_notebook=history)

            def progress(force=False, final=False):
                nonlocal last_save, last_report
                now = time.monotonic()
                if force or now - last_save >= 30 or stop.check():
                    if active is not None and active_started is not None:
                        s["unit_charges"][str(active)] = active_prior + now - active_started
                    s["charged_seconds"] = elapsed()
                    child = json.loads((active / "progress.json").read_text()) if active and (active / "progress.json").exists() else {}
                    row = {"version": VERSION, "schema_version": 1, "status": s["status"], "stage": label,
                           "charged_seconds": elapsed(), "support_seconds": s["support_seconds"],
                           "completed_units": len(s["completed"]), "planned_units": max(planned, len(s["completed"])),
                           "active_unit": str(active) if active else None, "active_progress": child,
                           "forecast": s["forecast"], "selected_rank": s["rank"]}
                    checkpoint = {**s, "charged_seconds": elapsed() + (0 if final else 30)}
                    parent.checkpoint(checkpoint, progress=row)
                    _atomic_bytes(root / "progress.json", (canonical_json(row) + "\n").encode())
                    last_save = now
                if force or now - last_report >= 600:
                    refresh()
                    last_report = now
                if not final:
                    if stop.check():
                        raise InterruptedError("Fix 5A checkpointed; use --resume --clear-stop")
                    if elapsed() >= max_hours * 3600 - 240:
                        raise TimeoutError("Fix 5A allocation reached; closeout reserve preserved")

            def check():
                progress()

            def unit(operation, settings, callback):
                nonlocal active, active_started, active_prior, label
                label = operation
                config = {**identity, "operation": operation, "settings": settings}
                suffix = hashlib.sha256(canonical_json(config).encode()).hexdigest()[:12]
                active = root / "units" / (operation + "-" + suffix)
                with RunStore(active, config, resume=resume, checkpoint_policy="journal-v1",
                              reserve_bytes=20_000_000_000) as store:
                    if store.completed:
                        result = json.loads((active / "summary.json").read_text())
                    else:
                        active_prior = s["unit_charges"].get(str(active), 0.)
                        active_started = time.monotonic()
                        progress(force=True)
                        result = callback(store)
                        if fingerprints() != identity["sources"]:
                            raise ValueError("numerical sources changed during Fix 5A")
                        result = scalar({**result, "version": VERSION, "schema_version": 1,
                                         "operation": operation, "kind": "head_rank_" + operation,
                                         "settings": settings, "unit_path": str(active),
                                         "active_seconds": active_prior + time.monotonic() - active_started,
                                         "runtime": {"torch": torch.__version__, "head_device": "cpu", "encoder_device": "cuda"}})
                        if store._current is None:
                            store.checkpoint({"stage": "done"})
                        store.complete(result)
                if str(active) not in s["completed"]:
                    s["completed"].append(str(active))
                print(canonical_json({"operation": operation, "settings": settings,
                                      "status": "completed", "unit": str(active)}), flush=True)
                active, active_started = None, None
                progress(force=True)
                return result

            try:
                progress(force=True)
                features_unit = one(units, "features")
                fitted_unit = one(units, "initial_fit")
                prepared = load_state(features_unit["unit_path"])
                fitted = load_state(fitted_unit["unit_path"])
                selection = one(units, "selection")["selected"]
                recipe = Recipe(**selection["recipe"])
                recipe = replace(recipe, rank=64)
                matching_geometry = [u for u in units if u["operation"] == "geometry_fisher"
                    and all(u["settings"][k] == getattr(recipe, k)
                            for k in ("temperature", "action_variance", "critic_variance"))]
                if len(matching_geometry) != 1:
                    raise ValueError("source initialization geometry is ambiguous")
                old_geometry = matching_geometry[0]
                reference = load_state(old_geometry["unit_path"])["learner"]["archive"]
                rebuilt = unit("initial_archives", {"ranks": [0, 10, 32, 64], "initial_observations": 515},
                               lambda store: {k: v for k, v in rebuild_initial(
                                   fitted, prepared["train"], recipe, store, check, reference_archive=reference).items()
                                              if k != "archives"})
                archives = load_state(rebuilt["unit_path"])["archives"]
                profile = unit("resource_audit", {"ranks": [0, 10, 32, 64]}, lambda store: {
                    "result": resource_audit(fitted, prepared["train"],
                                             replace(recipe, fd_strength=selection["fd_strength"]), archives, check)})
                health = profile["result"]
                # The preferred rank is a design choice; never select it on held-out rewards.
                if s["rank"] is None:
                    if 0 not in health["healthy_ranks"] or any(
                            not row.get("timing_available") for row in health["timings"] if row["rank"] == 0):
                        s["status"] = "resource_blocked"
                        progress(force=True, final=True)
                        return {"status": "resource_blocked", "reason": "rank-zero audit baseline unavailable",
                                "charged_seconds": elapsed()}
                    selected_rank = None
                    for rank in (64, 32):
                        if rank not in health["healthy_ranks"]:
                            continue
                        extra_seconds = 0.
                        for method in ("td_replay", "td_fd"):
                            row = next(r for r in health["timings"] if r["rank"] == rank and r["method"] == method)
                            base_row = next(r for r in health["timings"] if r["rank"] == 0 and r["method"] == method)
                            extra_seconds += 3 * (3800 * max(0., row["fit_seconds"] - base_row["fit_seconds"])
                                                  + 700 * max(0., row["retirement_seconds"] - base_row["retirement_seconds"]))
                        overhead = extra_seconds / 3600
                        if elapsed()/3600 + 11.5 + overhead <= max_hours - .25:
                            selected_rank = rank
                            break
                    if selected_rank is None:
                        s["status"] = "resource_blocked"
                        progress(force=True, final=True)
                        return {"status": "resource_blocked", "reason": "neither positive rank is healthy and within allocation",
                                "charged_seconds": elapsed()}
                    s["rank"] = selected_rank
                    s["rank_reason"] = "preferred rank numerically healthy and projected within budget" if selected_rank == 64 else "rank-64 resource/numerical assessment required fallback"
                    s["forecast"] = {"remaining_hours": [10., 12.8 + overhead], "rank_fit_overhead_hours": overhead}
                admitted = unit("rank_admission", {}, lambda store: {"rank": s["rank"],
                                  "reason": s["rank_reason"], "forecast": s["forecast"]})
                if stage == "profile":
                    s["status"] = "profile_complete"
                    progress(force=True, final=True)
                    return {"status": s["status"], "rank": s["rank"]}
                recipe = replace(recipe, rank=s["rank"])
                if not torch.cuda.is_available():
                    raise RuntimeError("Qwen evaluation requires CUDA")
                label = "loading frozen Qwen"
                progress(force=True)
                from numerical_experiment.picar.models import load_qwen_model
                from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
                from numerical_experiment.picar_reward_repair.config import MODEL_SEED, NORMALIZATION, DEFAULT_REFERENCES
                from numerical_experiment.picar_reward_repair.scoring import load_references, WidthScorer
                raw = load_qwen_model(**({"model_path": model_path} if model_path else {}),
                    local_files_only=offline, seed=MODEL_SEED, top_layers=1, ltm_rank=4, stm_rank=1,
                    hidden_size=64, context_length=4, return_bins=161, min_pixels=56*56,
                    max_pixels=19200, max_tokens=2048).eval()
                raw.encoder.enable_prefix_cache = True
                model = NormalizedStudent(raw, mode=NORMALIZATION, eps=1e-6).eval()
                apply_parameters(model, {"parameters": prepared["source_parameters"]})
                model.requires_grad_(False)
                features = FrozenFeatures(model, source_config["source_seal"])
                references = load_references(DEFAULT_REFERENCES, root / "reference_bundle")
                import shutil
                if not (root / "score_cache" / "REUSE_COMPLETE").exists():
                    shutil.copytree(source / "score_cache", root / "score_cache", dirs_exist_ok=True)
                    _atomic_bytes(root / "score_cache" / "REUSE_COMPLETE", b"read-only source cache copied\n")
                scorer = WidthScorer(model.encoder, references, root / "score_cache")
                source_scorers = [p.name for p in (source / "score_cache").iterdir()
                                  if p.is_dir() and (p / "identity.json").exists()]
                if scorer.identity not in source_scorers:
                    raise ValueError("frozen evaluator identity differs from reused controls")
                initial_panels = {}
                for phase in ("development", "confirmation"):
                    old = one(units, "initial_evaluation", phase=phase)
                    panels = copy.deepcopy(load_state(old["unit_path"])["evaluations"])
                    for panel in panels:
                        panel["shared_initial_source"] = old["unit_path"]
                        for ep in panel["episodes"]:
                            ep.pop("heldout_nll", None)
                    initial_panels[phase] = panels

                def trajectory(method, cfg, seed, phase):
                    settings = {"method": method, "seed": seed, "phase": phase,
                                "ewc_rank": cfg.rank, "stm_rank": 1, "total": 1024, "recipe": asdict(cfg)}
                    return unit("trajectory", settings, lambda store: RankTrajectory(
                        new_learner(fitted, cfg, seed, method, archives), features, scorer, store, check,
                        seed=seed, total=1024, evaluation_episodes=8,
                        initial_episodes=4 if phase == "confirmation" else 2,
                        development=phase != "confirmation", midpoint_episodes=4 if phase == "confirmation" else 2,
                        initial_evaluations=initial_panels["confirmation" if phase == "confirmation" else "development"]).run())

                expanded = {}
                for method in ("td_replay", "td_fd"):
                    old = one(units, "trajectory", phase="development", method=method, ewc_rank=0)
                    def expand(store, old=old):
                        saved = store.load_checkpoint(trusted=True)
                        state = load_state(old["unit_path"])
                        cfg = Recipe(**old["settings"]["recipe"])
                        learner = new_learner(fitted, cfg, 199301, old["settings"]["method"], archives)
                        if saved is None:
                            # Preserve only endpoint evaluation state; source training state stays sealed.
                            for key in ("replay", "ready", "episodes", "metrics"):
                                state[key] = []
                            state.update(stage="evaluate", target=1024, current=None, session=None, history=None)
                            state["evaluations"] = [p for p in state["evaluations"] if p["n"] == 1024]
                            for p in state["evaluations"]:
                                p["source_unit"] = old["unit_path"]
                            store.checkpoint(state)
                        runner = RankTrajectory(learner, features, scorer, store, check, seed=199301,
                                                total=1024, evaluation_episodes=8, development=True)
                        before = copy.deepcopy(runner.learner.state_dict())
                        runner.evaluate()
                        from numerical_experiment.picar_evaluation_video import assert_unchanged
                        assert_unchanged(before, runner.learner.state_dict())
                        return {"evaluations": runner.s["evaluations"], "source_unit": old["unit_path"],
                                "training_state_unchanged": True, "additional_evaluation_episodes": 12}
                    expanded[method] = unit("evaluation_expansion", {**old["settings"],
                        "phase": "development_expansion", "source_unit": old["unit_path"]}, expand)
                development = []
                for method in ("td_replay", "td_fd"):
                    cfg = replace(recipe, fd_strength=selection["fd_strength"] if method == "td_fd" else 0.)
                    development.append(trajectory(method, cfg, 199301, "development"))
                reference_return = endpoint(expanded["td_replay"], "initial_A")
                correction = choose_correction(development, reference_return,
                                               endpoint(expanded["td_replay"], "moderate_A"))
                if correction is not None:
                    planned += 1
                    s["correction"] = correction
                    unit("correction_decision", {}, lambda store: {"decision": correction})
                    admitted_correction = any(
                        json.loads(p.read_text()).get("settings", {}).get("phase") == "retention_calibration"
                        for p in (root / "units").glob("trajectory-*/config.json"))
                    if admitted_correction or elapsed()/3600 + 1.8 + 7.6 + .25 < max_hours:
                        planned += 1
                        cfg = replace(recipe, ridge=correction["ridge"],
                                      fd_strength=selection["fd_strength"] if correction["method"] == "td_fd" else 0.)
                        development.append(trajectory(correction["method"], cfg, 199301, "retention_calibration"))
                    else:
                        s["correction"]["not_run"] = "insufficient remaining allocation for correction and assessment"
                choices = {}
                for method in ("td_replay", "td_fd"):
                    candidates = [r for r in development if r["settings"]["method"] == method]
                    matched = [r for r in candidates if abs(endpoint(r, "initial_A") - reference_return) <= 5]
                    chosen = max(matched, key=lambda r: endpoint(r, "moderate_A")) if matched else min(
                        candidates, key=lambda r: abs(endpoint(r, "initial_A") - reference_return))
                    choices[method] = {"recipe": chosen["settings"]["recipe"], "development_unit": chosen["unit_path"],
                                       "retention_gap": endpoint(chosen, "initial_A") - reference_return,
                                       "retention_matched": bool(matched)}
                frozen = unit("selection", {}, lambda store: {"selected": choices,
                    "reference_return": reference_return, "rank": s["rank"], "correction": s["correction"]})
                s["selected"] = frozen["selected"]
                if stage == "development":
                    s["status"] = "development_complete"
                    progress(force=True, final=True)
                    return {"status": s["status"], "selected": s["selected"]}
                for seed, methods in ((199101, ("td_replay", "td_fd")), (199102, ("td_fd", "td_replay"))):
                    for method in methods:
                        trajectory(method, Recipe(**s["selected"][method]["recipe"]), seed, "confirmation")
                        remaining = 4 - sum(json.loads((Path(p)/"summary.json").read_text())["settings"].get("phase") == "confirmation"
                                            for p in s["completed"])
                        s["forecast"] = {"remaining_hours": [remaining * 1.5, remaining * 1.9 + .25]}
                _, final_receipts = source_index(source)
                if final_receipts != receipts:
                    raise ValueError("source study changed during the extension")
                s["status"] = "complete"
                s["forecast"] = {"remaining_hours": [0., 0.]}
                progress(force=True, final=True)
                result = {"status": "complete", "version": VERSION, "charged_seconds": elapsed(),
                          "support_seconds": s["support_seconds"], "completed_units": len(s["completed"]),
                          "selected": s["selected"], "source_unchanged": True,
                          "physical_motion": False, "next_phase_authorized": False}
                parent.complete(result)
                refresh()
                return result
            except (InterruptedError, TimeoutError) as exc:
                s["status"] = "paused"
                s["pause_reason"] = str(exc)
                progress(force=True, final=True)
                return {"status": "paused", "reason": str(exc), "charged_seconds": elapsed()}
            except BaseException as exc:
                s["status"] = "failed"
                s["error"] = repr(exc)
                progress(force=True, final=True)
                raise
    finally:
        stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--max-hours", type=float, default=16.)
    parser.add_argument("--support-seconds", type=float, default=0.)
    parser.add_argument("--stage", choices=("profile", "development", "all"), default="all")
    args = parser.parse_args()
    recorded = args.output / "orchestration/config.json"
    if args.resume and recorded.exists():
        previous = json.loads(recorded.read_text())
        if args.model_path is None and previous.get("model_path"):
            args.model_path = Path(previous["model_path"])
            args.offline = True
        if args.source == SOURCE:
            args.source = Path(previous["source"])
    if args.clear_stop:
        (args.output / "STOP").unlink(missing_ok=True)
    options = vars(args)
    options.pop("clear_stop")
    print(json.dumps(run(**options), indent=2), flush=True)


if __name__ == "__main__":
    main()
