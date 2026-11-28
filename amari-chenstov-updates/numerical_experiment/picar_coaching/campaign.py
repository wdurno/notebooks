"""Budgeted, resumable orchestration of Plan 19 Fix 6."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import torch

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.runtime import StopController, check_storage, sha256_file
from numerical_experiment.picar.storage import RunStore, _atomic_bytes
from numerical_experiment.picar_head_awac.campaign import scalar_state
from numerical_experiment.picar_head_rank.reuse import load_state, receipt
from . import VERSION
from .config import (COLLECTION, CONDITIONS, MAX_HOURS, NOTEBOOK, OUTPUT, PROJECT, SOURCE,
                     TOTAL, fingerprints, seed, selected_recipe)


class ReadinessFailure(RuntimeError):
    pass


def run(output=OUTPUT, source=SOURCE, *, model_path=None, offline=False, resume=False,
        stage="all", max_hours=MAX_HOURS, support_seconds=0.):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "cache/matplotlib"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_num_threads(2)
    if not 0 < max_hours <= MAX_HOURS or support_seconds < 0:
        raise ValueError("Fix 6 allocation must be positive and at most 28 hours")
    root, source = Path(output).resolve(), Path(source).resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous_config = root / "orchestration/config.json"
    if resume and model_path is None and previous_config.exists():
        model_path = json.loads(previous_config.read_text()).get("model_path")
    cfg = selected_recipe(source)
    encoder_sources = list((source / "units").glob("features-*/COMPLETED"))
    if len(encoder_sources) != 1:
        raise ValueError("accepted shared frozen encoder source is ambiguous")
    encoder_source = encoder_sources[0].parent
    identity = {"schema_version": 1, "version": VERSION, "sources": fingerprints(),
        "source": str(source), "source_receipt": receipt(source / "orchestration"),
        "common_encoder_receipt": receipt(encoder_source),
        "theory_sha256": sha256_file(PROJECT / "mathematical_overview.ipynb"),
        "model_path": str(Path(model_path).resolve()) if model_path else None,
        "recipe": asdict(cfg), "conditions": list(CONDITIONS),
        "protocol": {"collection_intervals_per_condition": COLLECTION, "online_intervals": TOTAL,
            "development_seed": seed("development"),
            "assessment_seeds": [seed("assessment:0"), seed("assessment:1")],
            "evaluation_checkpoints": [0, 256, 512, 768, 1024], "physical_motion": False,
            "revision": "v4-task-only", "reward_mode": "task_only", "adherence_enabled": False,
            "coaching_seconds": 4., "compliance_raw_magnitude": 0., "reward_coding": 10.,
            "initialization_pairs": 1, "fresh_stream_pairs": 2,
            "pretrained_student_seed": 191101, "initial_critic_seed": 199001}}
    stop = StopController(root)
    stop.install()
    try:
        with RunStore(root / "orchestration", identity, resume=resume, checkpoint_policy="journal-v1",
                      reserve_bytes=20_000_000_000) as parent:
            if parent.completed:
                return json.loads((parent.root / "summary.json").read_text())
            history = root / "history.ipynb"
            if not history.exists():
                _atomic_bytes(history, (PROJECT / NOTEBOOK).read_bytes())
            state = parent.load_checkpoint(trusted=True) or {
                "charged_seconds": support_seconds, "support_seconds": support_seconds,
                "completed": [], "unit_charges": {}, "status": "running",
                "forecast": {"remaining_hours": [18., 25.]}, "decisions": []}
            state["status"] = "running"
            if not state["decisions"]:
                state["decisions"].append({"revision": "v4-task-only",
                    "prior_pilots": ["coaching-fix6-v1", "coaching-fix6-v2", "coaching-fix6-v3"],
                    "reason": "User authorized removing unreliable adherence evaluation; coaching remains the treatment.",
                    "changes": ["identical task reward in U/G; no adherence inference, bonus, queue or gate",
                        "retain V3 teacher prompt, geometric coaching, cue histories and taper",
                        "new immutable root and reward-mode recovery checks; preserve prior readiness results"]})
            began, base = time.monotonic(), state["charged_seconds"]
            active, active_started, active_prior = None, None, 0.
            label, last_save, last_report, last_storage = "software acceptance", 0., 0., 0.
            storage = {}
            planned = 23

            def elapsed():
                return base + time.monotonic() - began

            def refresh():
                from .report import build_notebook
                build_notebook(root, PROJECT / NOTEBOOK, base_notebook=history)

            def progress(force=False, final=False):
                nonlocal last_save, last_report, last_storage, storage
                now = time.monotonic()
                if now - last_storage > 300:
                    storage = check_storage(PROJECT / "cache/numerical_experiment/picar",
                        quota_bytes=500_000_000_000, reserve_bytes=20_000_000_000)
                    last_storage = now
                if force or now - last_save > 30 or stop.check():
                    if active is not None and active_started is not None:
                        state["unit_charges"][str(active)] = active_prior + now - active_started
                    state["charged_seconds"] = elapsed()
                    child = json.loads((active / "progress.json").read_text()) if active and (active / "progress.json").exists() else {}
                    forecast = copy.deepcopy(state["forecast"])
                    if "updated_at_seconds" in forecast:
                        spent = max(0., (elapsed() - forecast["updated_at_seconds"]) / 3600)
                        forecast["remaining_hours"] = [max(0., value - spent) for value in forecast["remaining_hours"]]
                    row = {"schema_version": 1, "version": VERSION, "status": state["status"],
                        "stage": label, "charged_seconds": elapsed(), "support_seconds": state["support_seconds"],
                        "protocol": identity["protocol"],
                        "completed_units": len(state["completed"]), "planned_units": planned,
                        "active_unit": str(active) if active else None, "active_progress": child,
                        "forecast": forecast, "storage": storage, "decisions": state["decisions"]}
                    parent.checkpoint({**state, "charged_seconds": elapsed() + (0 if final else 30)}, progress=row)
                    _atomic_bytes(root / "progress.json", (canonical_json(row) + "\n").encode())
                    last_save = now
                if force or now - last_report > 600:
                    refresh()
                    last_report = now
                if not final:
                    if stop.check():
                        raise InterruptedError("Fix 6 checkpointed; use --resume --clear-stop")
                    if elapsed() > max_hours * 3600 - 2520:
                        raise TimeoutError("Fix 6 allocation reached; 0.7 hours reserved for closeout")

            def check():
                progress()

            def unit(operation, settings, callback):
                nonlocal active, active_started, active_prior, label
                label = operation + (" " + settings.get("condition", ""))
                config = {**identity, "operation": operation, "settings": settings}
                suffix = hashlib.sha256(canonical_json(config).encode()).hexdigest()[:12]
                active = root / "units" / (operation + "-" + suffix)
                with RunStore(active, config, resume=resume, checkpoint_policy="journal-v1",
                              reserve_bytes=20_000_000_000) as store:
                    if store.completed:
                        result = json.loads((active / "summary.json").read_text())
                    else:
                        active_started, active_prior = time.monotonic(), state["unit_charges"].get(str(active), 0.)
                        progress(force=True)
                        result = callback(store)
                        if fingerprints() != identity["sources"]:
                            raise ValueError("Fix 6 numerical sources changed during execution")
                        result = scalar_state({**result, "schema_version": 1, "version": VERSION,
                            "kind": "picar_coaching_" + operation, "operation": operation,
                            "settings": settings, "unit_path": str(active),
                            "active_seconds": active_prior + time.monotonic() - active_started})
                        if store._current is None:
                            store.checkpoint({"stage": "done"})
                        store.complete(result)
                if str(active) not in state["completed"]:
                    state["completed"].append(str(active))
                print(canonical_json({"operation": operation, "settings": settings,
                                      "status": "completed", "unit": str(active)}), flush=True)
                active, active_started = None, None
                progress(force=True)
                return result

            def software(store):
                tests = sorted((PROJECT / "test/unit").glob("test_picar_coaching_*.py"))
                tests += sorted((PROJECT / "test/unit").glob("test_picar_head_awac_*.py"))
                tests += sorted((PROJECT / "test/unit").glob("test_picar_head_rank_numerics.py"))
                result = subprocess.run([sys.executable, "-m", "pytest", "-q", *map(str, tests)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600,
                    env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
                store._write_root("pytest.txt", result.stdout.encode())
                if result.returncode:
                    raise RuntimeError(result.stdout[-7000:])
                with tarfile.open(store.root / "sources.tar.gz", "w:gz") as archive:
                    for name in identity["sources"]:
                        archive.add(PROJECT / name, arcname=name)
                    for name in ("report.py", "video.py"):
                        path = Path(__file__).parent / name
                        archive.add(path, arcname=str(path.relative_to(PROJECT)))
                    for path in sorted((PROJECT / "test/unit").glob("test_picar_coaching_*.py")):
                        archive.add(path, arcname=str(path.relative_to(PROJECT)))
                return {"passed": True, "tests": result.stdout.strip().splitlines()[-1]}

            try:
                progress(force=True)
                unit("software", {}, software)
                if stage == "software":
                    state["status"] = "software_complete"
                    progress(force=True, final=True)
                    return {"status": state["status"]}
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is required for the frozen Qwen roles")
                from numerical_experiment.picar.models import load_qwen_model
                from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
                from numerical_experiment.picar_reward_repair.config import MODEL_SEED, NORMALIZATION, DEFAULT_REFERENCES
                from numerical_experiment.picar_reward_repair.scoring import WidthScorer, load_references
                from .initialization import (representation_parameters, restore_representation, prepare_distillation,
                    distill, prepare_heads, fit_critics, initial_fisher)
                from .readiness import geometry_audit, teacher_audit, resource_profile
                from .trajectory import CoachingTrajectory
                from numerical_experiment.picar_head_awac.model import Heads, FrozenFeatures
                from numerical_experiment.picar_head_awac.learner import Learner
                from numerical_experiment.picar_head_awac.archive import Archive

                label = "loading pinned frozen Qwen"
                progress(force=True)
                raw = load_qwen_model(**({"model_path": model_path} if model_path else {}),
                    local_files_only=offline, seed=MODEL_SEED, top_layers=1, ltm_rank=4, stm_rank=1,
                    hidden_size=64, context_length=4, return_bins=161, min_pixels=56 * 56,
                    max_pixels=19200, max_tokens=2048).eval()
                raw.encoder.enable_prefix_cache = True
                model = NormalizedStudent(raw, mode=NORMALIZATION, eps=1e-6).eval().requires_grad_(False)
                # Reuse the common frozen encoder, never its all-forward fitted
                # projector, recurrence, action head or initial Fisher.
                encoder_parameters = {n: v for n, v in load_state(encoder_source)["source_parameters"].items()
                                      if n.startswith("encoder.")}
                if not encoder_parameters:
                    raise ValueError("accepted frozen encoder has no parameter receipt")
                actual = dict(model.named_parameters())
                with torch.no_grad():
                    for name, value in encoder_parameters.items():
                        actual[name].copy_(value.to(actual[name]))
                model.encoder.clear_caches()
                for module in (model.projection, model.recurrent, model.action_head, model.return_head):
                    module.cpu()
                pristine = representation_parameters(model)
                references = load_references(DEFAULT_REFERENCES, root / "reference_bundle")
                scorer = WidthScorer(model.encoder, references, root / "score_cache")
                geometry = unit("readiness_geometry", {"phase": "readiness"}, lambda store: geometry_audit(store, check))
                teacher = unit("readiness_teacher", {"phase": "readiness"},
                    lambda store: teacher_audit(model, scorer, None, store, check))
                profile = unit("resource_profile", {"phase": "readiness"},
                    lambda store: resource_profile(model, geometry, None, teacher, store, check))
                state["forecast"] = profile["forecast"]
                state["forecast"]["updated_at_seconds"] = elapsed()
                if not (geometry["ready"] and teacher["ready"]):
                    state["status"] = "readiness_failed"
                    progress(force=True, final=True)
                    return {"status": state["status"], "geometry": geometry["ready"],
                            "adherence_enabled": False, "teacher": teacher["ready"]}
                if stage == "readiness":
                    state["status"] = "readiness_complete"
                    progress(force=True, final=True)
                    return {"status": state["status"]}
                assets = {}
                for condition in CONDITIONS:
                    settings = {"condition": condition, "phase": "initialization", "seed": seed("collection")}
                    restore_representation(model, pristine)
                    empty = Learner(Heads(seed=seed("heads")), cfg, seed("collection"), "td_fd")
                    collection = unit("collection", settings, lambda store, condition=condition: CoachingTrajectory(
                        empty, None, scorer, store, check, model=model, adherence_enabled=False,
                        condition=condition, phase="initialization", seed=seed("collection"), total=COLLECTION,
                        collection=True).run())
                    collected = load_state(collection["unit_path"])
                    with RunStore(collection["unit_path"], json.loads((Path(collection["unit_path"]) / "config.json").read_text()),
                                  resume=True, checkpoint_policy="journal-v1") as original:
                        cached = unit("distillation_features", settings, lambda store: {
                            "records": len(prepare_distillation(model, collected, original, store, check)["records"])})
                        semantic = load_state(cached["unit_path"])["records"]
                        fitted = unit("distillation", settings,
                                      lambda store: distill(model, semantic, pristine, store, check))
                        distillation_state = load_state(fitted["unit_path"])
                        restore_representation(model, distillation_state["selected_parameters"])
                        features_unit = unit("features", settings, lambda store: prepare_heads(
                            model, collected["initial_records"], original, store, check, fitted["unit_path"]))
                    prepared = load_state(features_unit["unit_path"])
                    fit_unit = unit("initial_fit", settings, lambda store: {
                        k: v for k, v in fit_critics(prepared, store, check, "cpu").items()
                        if k in {"curves", "epoch", "inventory"}})
                    head_state = load_state(fit_unit["unit_path"])
                    fisher_unit = unit("initial_archive", settings,
                        lambda store: initial_fisher(head_state, prepared["train"], cfg, store, check))
                    assets[condition] = {"parameters": distillation_state["selected_parameters"],
                        "heads": head_state["selected_heads"], "fisher": load_state(fisher_unit["unit_path"])["learner"]["archive"],
                        "initial_ids": [r["id"] for r in prepared["train"]], "identity": fitted["unit_path"]}
                state["forecast"] = {"remaining_hours": [12.5, 15.5], "updated_at_seconds": elapsed(),
                    "basis": "Initialization complete; six 1024-interval paths and closeout, before live-path timing."}
                if stage == "initialization":
                    state["status"] = "initialization_complete"
                    progress(force=True, final=True)
                    return {"status": state["status"]}
                measured_paths = []
                for phase, replica in (("development", 0), ("assessment", 0), ("assessment", 1)):
                    if phase == "assessment" and stage == "development":
                        break
                    for condition in (CONDITIONS if replica == 0 else tuple(reversed(CONDITIONS))):
                        asset = assets[condition]
                        restore_representation(model, asset["parameters"])
                        model.requires_grad_(False)
                        features = FrozenFeatures(model, asset["identity"])
                        stream_seed = seed("development" if phase == "development" else f"assessment:{replica}")
                        heads = Heads(seed=seed("heads"))
                        heads.load_state_dict(asset["heads"])
                        learner = Learner(heads, cfg, stream_seed, "td_fd")
                        learner.archive = Archive.from_state_dict(asset["fisher"], device="cpu")
                        settings = {"condition": condition, "phase": phase, "seed": stream_seed,
                                    "replica": replica, "total": TOTAL, "recipe": asdict(cfg)}
                        completed_path = unit("trajectory", settings, lambda store, condition=condition, phase=phase: CoachingTrajectory(
                            learner, features, scorer, store, check, model=model, adherence_enabled=False,
                            condition=condition, phase=phase, seed=stream_seed, total=TOTAL,
                            initial_training_ids=asset["initial_ids"]).run())
                        measured_paths.append(completed_path["active_seconds"] / 3600)
                    remaining = 6 - len(measured_paths)
                    state["forecast"] = {"remaining_hours": [.7 + remaining * .9 * min(measured_paths),
                        .7 + remaining * 1.35 * max(measured_paths)], "updated_at_seconds": elapsed(),
                        "basis": "Observed complete live-path times, including scored evaluations; planning range, not CI."}
                    if phase == "development" and elapsed() / 3600 + state["forecast"]["remaining_hours"][1] > max_hours:
                        raise TimeoutError("measured fresh-scope forecast exceeds remaining allocation")
                    progress(force=True)
                if stage == "development":
                    state["status"] = "development_complete"
                    progress(force=True, final=True)
                    return {"status": state["status"]}
                def final_audit(store):
                    receipts = []
                    acquisitions = []
                    for path in state["completed"]:
                        path = Path(path)
                        check()
                        receipts.append(receipt(path))
                        summary = json.loads((path / "summary.json").read_text())
                        if summary["operation"] in ("collection", "trajectory"):
                            payload = load_state(path)
                            episodes = payload["episodes"] + [e for p in payload["evaluations"] for e in p["episodes"]]
                            if (payload.get("reward_mode") != "task_only" or any(
                                    ep.get("guidance_return") != 0 or ep.get("scored_guidance_windows") != 0
                                    or ep.get("adherence_enabled") is not False for ep in episodes)):
                                raise ValueError("task-only final reward audit failed")
                        if summary["operation"] != "trajectory":
                            continue
                        payload = load_state(path)
                        retired = payload["retired_ids"]
                        replay = [r["id"] for r in payload["replay"]]
                        initial = assets[summary["settings"]["condition"]]["initial_ids"]
                        if (len(retired) != len(set(retired)) or len(replay) != len(set(replay))
                                or set(retired) & set(replay) or set(initial) & (set(retired) | set(replay))):
                            raise ValueError("final unique evidence/retirement audit failed")
                        if payload["stage"] != "done" or payload["learner"]["retired"] != len(retired):
                            raise ValueError("trajectory final state is incomplete")
                        if payload["learner"]["archive"]["rank"] != 64 or not torch.isfinite(payload["learner"]["vector"]).all():
                            raise ValueError("nonfinite final learner or wrong archive rank")
                        if len(payload["evaluations"]) != 10:
                            raise ValueError("required five-checkpoint paired evaluation panels missing")
                        acquisitions.append({"condition": summary["settings"]["condition"],
                            "phase": summary["settings"]["phase"], "seed": summary["seed"],
                            "captured": payload["n"], "retired": len(retired), "replay": len(replay)})
                    if fingerprints() != identity["sources"]:
                        raise ValueError("scientific source hashes changed")
                    result = {"passed": True, "receipts": receipts, "acquisitions": acquisitions,
                              "theory_unchanged": sha256_file(PROJECT / "mathematical_overview.ipynb") == identity["theory_sha256"]}
                    store.checkpoint(result, progress={"stage": "final integrity audit", "checked_units": len(receipts)})
                    return result
                audit = unit("final_audit", {}, final_audit)
                state["status"] = "completed"
                state["forecast"] = {"remaining_hours": [0., 0.], "updated_at_seconds": elapsed()}
                progress(force=True, final=True)
                result = {"status": "completed", "completed": state["completed"],
                          "charged_seconds": elapsed(), "decisions": state["decisions"], "audit": audit["unit_path"]}
                parent.complete(result)
                refresh()
                return result
            except BaseException as exc:
                state["status"] = "paused" if isinstance(exc, InterruptedError) else "budget_paused" if isinstance(exc, TimeoutError) else "error"
                state["last_error"] = type(exc).__name__ + ": " + str(exc)
                progress(force=True, final=True)
                raise
    finally:
        stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="run", choices=("run", "status", "report", "stop"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--stage", choices=("software", "readiness", "initialization", "development", "all"), default="all")
    parser.add_argument("--max-hours", type=float, default=MAX_HOURS)
    parser.add_argument("--support-seconds", type=float, default=0.)
    args = parser.parse_args()
    if args.command == "status":
        print((args.output / "progress.json").read_text())
    elif args.command == "report":
        from .report import build_notebook
        build_notebook(args.output.resolve(), PROJECT / NOTEBOOK, base_notebook=args.output.resolve() / "history.ipynb")
    elif args.command == "stop":
        _atomic_bytes(args.output / "STOP", b"operator requested checkpoint\n")
    else:
        if args.clear_stop:
            (args.output / "STOP").unlink(missing_ok=True)
        result = run(args.output, args.source, model_path=args.model_path, offline=args.offline,
            resume=args.resume, stage=args.stage, max_hours=args.max_hours, support_seconds=args.support_seconds)
        print(canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
