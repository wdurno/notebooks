"""Finish Phase 2 after a diagnosed saturated projection, within its same cap."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.runtime import StopController, check_storage, sha256_file
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_phase2.analysis import forecast, hard_gain, observed_trainability, panel_summary
from numerical_experiment.picar_phase2.campaign import PROJECT, audit_initial, language_unit, load_completed_state, read_json
from numerical_experiment.picar_phase2.config import BASE, PILOT_SEEDS, evaluation_panels, learner_config, simulator_config, source_hashes


VERSION = "plan19-phase2-rms-v1"
MODES = ("feature_rms", "feature_and_projection_rms")


class StageProgressStore:
    """Expose initialization work rather than a frozen acquisition counter."""
    def __init__(self, store):
        self.store = store

    def __getattr__(self, name):
        return getattr(self.store, name)

    def checkpoint(self, state, *, progress=None, **kwargs):
        progress = dict(progress or {})
        if state["stage"] in {"distill", "initial_nll"}:
            key = "kl_step" if state["stage"] == "distill" else "nll_step"
            progress["counts"] = {"step": state[key], "target": len(state["eligible"])}
        elif state["stage"] in {"initial_fisher", "drain"} and state["identity"]["mode"] == "bootstrap":
            capacity = state["identity"]["config"]["learning"]["replay_capacity"]
            progress["counts"] = {"step": len(state["retired"]),
                "target": len(state["retired"])+len(state["eligible"])-capacity}
        return self.store.checkpoint(state, progress=progress, **kwargs)


def baseline_units(source):
    from numerical_experiment.picar.report import read_completed_unit
    units = [read_completed_unit(p.parent)["summary"]
             for p in sorted((source / "units").glob("pilot-*/COMPLETED"))]
    return [p for p in units if p.get("settings", {}).get("name") == "baseline"]


def seal_baselines(source, *, wait=False):
    """Stop only after both planned baselines; never select a stopping p-value."""
    while len(list((source / "units").glob("pilot-*/COMPLETED"))) < 2:
        if not wait:
            raise RuntimeError("both original baseline pilots must finish first; use --wait")
        if (source / "orchestration/COMPLETED").exists():
            raise RuntimeError("source campaign ended before the two required baselines")
        time.sleep(.5)
    if {p["seed"] for p in baseline_units(source)} != set(PILOT_SEEDS):
        raise RuntimeError("the two distinct planned baseline seeds are required")
    unexpected = [p for p in (source / "units").glob("*/config.json")
                  if read_json(p).get("settings", {}).get("name") in {"no_ridge", "softer_bootstrap"}]
    if unexpected:
        raise RuntimeError("original rescue already started; do not exceed the two-rescue allowance")
    if (source / "orchestration/COMPLETED").exists():
        return read_json(source / "orchestration/summary.json")
    (source / "STOP").touch()
    while True:
        with (source / "orchestration/.writer.lock").open("r") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
                break
            except BlockingIOError:
                time.sleep(.5)
    path = source / "orchestration"
    with RunStore(path, read_json(path / "config.json"), resume=True, checkpoint_policy="journal-v1") as store:
        if store.completed:
            return read_json(path / "summary.json")
        saved = store.load_checkpoint(trusted=True)
        result = {"schema_version": 1, "phase": 2, "smoke_only": True, "kind": "phase2_campaign",
            "status": "superseded_after_baselines", "units": saved["units"],
            "total_active_seconds": saved["charged_seconds"], "active_seconds": 0.,
            "baseline_paths": [p["unit_path"] for p in baseline_units(source)],
            "reason": "Saturated projection found on sealed language probes; replace unexecuted blind rescues with common-model conditioning tests.",
            "original_rescues_executed": 0, "next_phase_authorized": False}
        store.complete(result)
        return result


def scientific_sources():
    return {**source_hashes(),
        **{str(p.relative_to(PROJECT)): sha256_file(p) for p in Path(__file__).parent.rglob("*.py")},
        "numerical_experiment/picar_feature_health.py": sha256_file(PROJECT / "numerical_experiment/picar_feature_health.py")}


def run(output, source_campaign, teacher_source, *, model_path=None, offline=False,
        resume=False, wait=False, max_phase_hours=5.):
    if not 0 < max_phase_hours <= 5:
        raise ValueError("all Phase 2 versions share the original five-hour cap")
    source_campaign, teacher_source, root = map(lambda p: Path(p).resolve(),
                                               (source_campaign, teacher_source, output))
    prior_campaign = seal_baselines(source_campaign, wait=wait)
    prior_seconds = prior_campaign["total_active_seconds"]
    base = replace(learner_config(), version=VERSION)
    sources = scientific_sources()
    identity = {"schema_version": 1, "version": VERSION, "config": base.to_dict(),
        "source_sha256": sources, "source_campaign": str(source_campaign),
        "source_campaign_marker_sha256": sha256_file(source_campaign / "orchestration/COMPLETED"),
        "teacher_source": str(teacher_source), "teacher_marker_sha256": sha256_file(teacher_source / "COMPLETED"),
        "prior_phase2_seconds": prior_seconds, "max_phase_hours": max_phase_hours,
        "model_source": str(Path(model_path).resolve()) if model_path else base.model.model_id,
        "model_initialization_seed": 191101, "normalization_modes": MODES, "normalization_eps": 1e-6,
        "maximum_rescue_attempts": 2, "pilot_seed": PILOT_SEEDS[0], "pilot_transitions": 640,
        "new_teacher_data": 0, "fresh_model_and_fisher": True, "fd_comparison": False}
    root.mkdir(parents=True, exist_ok=True)
    stop = StopController(root)
    stop.install()
    try:
        with RunStore(root / "orchestration", identity, resume=resume, checkpoint_policy="journal-v1") as parent:
            if parent.completed:
                return read_json(parent.root / "summary.json")
            saved = parent.load_checkpoint(trusted=True) or {"charged_seconds": 0.}
            started, last_save, last_report = time.monotonic(), 0., 0.
            count = sum(1 for _ in (root / "units").glob("*/COMPLETED"))
            active = None
            stage = "conditioning rescue acceptance"

            def elapsed():
                return saved["charged_seconds"] + time.monotonic()-started

            def refresh():
                from numerical_experiment.picar.report import build_notebook
                build_notebook(PROJECT / "cache/numerical_experiment/picar")

            def progress(label=None, *, force=False):
                nonlocal last_save, last_report, stage
                stage = label or stage
                now = time.monotonic()
                if force or now-last_save >= 15 or stop.check():
                    work = read_json(active / "progress.json") if active and (active / "progress.json").exists() else {}
                    parent.checkpoint({"charged_seconds": elapsed()}, progress={"phase": 2,
                        "status": stage, "active_seconds": elapsed(), "cumulative_phase2_seconds": prior_seconds+elapsed(),
                        "counts": {"step": count, "target": 12},
                        "active_unit": None if active is None else str(active), "active_unit_progress": work})
                    last_save = now
                if force or now-last_report >= 900:
                    refresh()
                    last_report = now
                if stop.check():
                    raise InterruptedError("normalization rescue checkpointed; remove STOP and --resume")
                if prior_seconds+elapsed() >= max_phase_hours*3600-90:
                    raise TimeoutError("shared Phase 2 compute ceiling reached")

            def unit(name, settings, callback):
                nonlocal active, count
                progress(name, force=True)
                config = {**identity, "operation": name, "settings": settings}
                suffix = hashlib.sha256(canonical_json(config).encode()).hexdigest()[:12]
                path = root / "units" / f"{name}-{suffix}"
                active = path
                with RunStore(path, config, resume=resume, checkpoint_policy="journal-v1") as store:
                    if store.completed:
                        active = None
                        return read_json(path / "summary.json")
                    tick = time.monotonic()
                    result = callback(store)
                    if scientific_sources() != sources:
                        raise RuntimeError("rescue source changed; preserve and version")
                    result = {**result, "kind": f"phase2_{name}", "schema_version": 1,
                        "phase": 2, "smoke_only": True, "settings": settings,
                        "unit_path": str(path), "active_seconds": time.monotonic()-tick,
                        "media_sha256": {str(p.relative_to(path)): sha256_file(p) for p in path.rglob("*")
                                         if p.suffix in {".png", ".mp4"}}}
                    if store._current is None:
                        store.checkpoint({"stage": "done"}, progress={"phase": 2, "status": name})
                    store.complete(result)
                count += 1
                active = None
                print(json.dumps({"unit": name, "settings": settings,
                    "seconds": result["active_seconds"], "path": str(path)}), flush=True)
                progress(name+" complete", force=True)
                return result

            def provenance(store):
                path = store.root / "sources.tar.gz"
                with tarfile.open(path, "w:gz") as archive:
                    for name in sources:
                        archive.add(PROJECT / name, arcname=name)
                return {"source_sha256": sources, "archive_sha256": sha256_file(path)}

            def software(store):
                result = subprocess.run([sys.executable, "-m", "pytest", "-q",
                    "test/unit/test_picar_phase2_rescue.py", "test/unit/test_picar_feature_health.py",
                    "test/unit/test_picar_phase2_rescue_campaign.py"], cwd=PROJECT,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
                path = store.root / "pytest.txt"
                path.write_text(result.stdout)
                if result.returncode:
                    raise RuntimeError(f"rescue acceptance failed: {path}")
                return {"passed": True, "test_result": result.stdout.strip().splitlines()[-1]}

            try:
                unit("provenance", {}, provenance)
                unit("software", {}, software)
                import torch
                from numerical_experiment.picar.models import load_qwen_model
                from numerical_experiment.picar_phase2.student import ReadinessRun, language_readiness
                from .model import NormalizedStudent
                from .data import restore_demonstrations
                torch.set_num_threads(2)
                check_storage(root, quota_bytes=100_000_000_000, reserve_bytes=20_000_000_000)

                class NormalizationRun(ReadinessRun):
                    def __init__(self, model, config, sim_config, store, stopped, *, initial=None, **kwargs):
                        expected = dict(model.provenance)
                        if kwargs.get("mode") == "pilot" and (initial or {}).get("student_normalization") != expected:
                            raise ValueError("initial Fisher/model normalization differs")
                        super().__init__(model, config, sim_config, StageProgressStore(store), stopped,
                                         initial=initial, **kwargs)
                        recorded = self.s.get("student_normalization")
                        if recorded is not None and recorded != expected:
                            raise ValueError("checkpoint model normalization differs")
                        self.s["student_normalization"] = expected

                old_pilots = [read_json(Path(p) / "summary.json") for p in prior_campaign["baseline_paths"]]

                def references(store):
                    rows = []
                    for pilot in old_pilots:
                        panels = {(p["checkpoint"], p["difficulty"]): p["rows"] for p in pilot["evaluations"]}
                        row = {"model": "original baseline", "seed": pilot["seed"],
                            "source": pilot["unit_path"], "new_trajectory": False, "charged_again": False}
                        for checkpoint in ("initial", "final"):
                            for difficulty in ("easy", "hard"):
                                data = panels[checkpoint, difficulty]
                                row[f"{difficulty}_{checkpoint}"] = f"{sum(r['success'] for r in data)}/{len(data)}"
                        rows.append(row)
                    return {"rows": rows, "source_campaign": str(source_campaign),
                        "source_completion_sha256": sha256_file(source_campaign / "orchestration/COMPLETED"),
                        "purpose": "historical baseline reference, not pooled or charged as new runs"}
                unit("reference_baselines", {}, references)
                pilots = []
                previous_mode_seconds = None
                for mode in MODES:
                    if mode != MODES[0] and observed_trainability(pilots):
                        break
                    continuing = any(read_json(p).get("settings", {}).get("name") == mode
                                     for p in (root / "units").glob("*/config.json"))
                    required = 4500 if previous_mode_seconds is None else 1.3*previous_mode_seconds+300
                    if not continuing and max_phase_hours*3600-prior_seconds-elapsed() < required:
                        break
                    mode_start = time.monotonic()
                    kwargs = {} if model_path is None else {"model_path": model_path}
                    raw = load_qwen_model(**kwargs, local_files_only=offline, seed=191101,
                        top_layers=1, ltm_rank=4, stm_rank=1, hidden_size=64, context_length=4,
                        return_bins=161, min_pixels=56*56, max_pixels=19200, max_tokens=2048).eval()
                    raw.encoder.enable_prefix_cache = True
                    model = NormalizedStudent(raw, mode=mode, eps=1e-6).eval()
                    settings = {"name": mode, "normalization": mode, "eps": 1e-6,
                        "ridge": .1, "teacher_floor": .001, "seed": PILOT_SEEDS[0], "transitions": 640}

                    def initialize(store):
                        runner = NormalizationRun(model, base, simulator_config(bootstrap=True), store,
                            progress, mode="bootstrap", seed=base.seed, max_transitions=600)
                        if runner.s["stage"] == "collect":
                            data = restore_demonstrations(teacher_source)
                            runner.s.update(eligible=data["records"], retired=[], ready=[],
                                acquired_count=data["acquired_count"], episodes=data["episodes"],
                                initial_data_provenance=data["provenance"], stage="distill")
                            runner.save()
                        result = runner.run()
                        return {**result, "reused_demonstrations": True, "old_fisher_reused": False,
                            "new_teacher_transitions": 0, "normalization": mode,
                            "data_provenance": runner.s["initial_data_provenance"]}

                    asset = unit("bootstrap", settings, initialize)
                    initial = load_completed_state(asset["unit_path"])
                    audit_initial(initial, base)

                    def runner_for(store):
                        return NormalizationRun(model, base, simulator_config(), store, progress,
                            mode="pilot", seed=PILOT_SEEDS[0], max_transitions=640,
                            initial=initial, evaluations=evaluation_panels())

                    def language(store):
                        return {**language_unit(runner_for(store)), "normalization": mode}
                    language_result = unit("language", settings, language)

                    def health(store):
                        from numerical_experiment.picar_feature_health import analyze_projection
                        return {**analyze_projection(initial, language_result, normalization=mode, eps=1e-6),
                            "source_bootstrap": asset["unit_path"], "source_language": language_result["unit_path"]}
                    unit("feature_health", settings, health)

                    def learn(store):
                        return {**runner_for(store).run(), "normalization": mode}
                    pilots.append(unit("pilot", settings, learn))
                    previous_mode_seconds = max(time.monotonic()-mode_start,
                        asset["timings"]["active_seconds"]+language_result["active_seconds"]+
                        pilots[-1]["timings"]["active_seconds"]+30.)
                    del model, raw, initial
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                prior_review = read_json(PROJECT / "cache/numerical_experiment/picar/phase1/fix3-v2/review/resource-forecast.json")
                prior_study = prior_review["conservative_total_study_seconds_so_far"]+143.
                measured = forecast(pilots, phase2_seconds=prior_seconds+elapsed(), prior_study_seconds=prior_study)
                measured.update(baseline_comparison=[{"seed": p["seed"], "hard_success_gain": hard_gain(p),
                    "source": p["unit_path"]} for p in old_pilots],
                    normalization_rescue_count=len(pilots), old_fisher_reused=False)
                resource = unit("forecast", {}, lambda store: measured)
                result = {"kind": "phase2_campaign", "schema_version": 1, "phase": 2,
                    "smoke_only": True, "status": "ready_for_checkin", "total_active_seconds": elapsed(),
                    "cumulative_phase2_seconds": prior_seconds+elapsed(), "active_seconds": 0.,
                    "trainability_observed": observed_trainability(pilots), "forecast": resource,
                    "pilot_paths": [p["unit_path"] for p in pilots], "units": count,
                    "baseline_paths": prior_campaign["baseline_paths"], "next_phase_authorized": False}
                parent.checkpoint({"charged_seconds": elapsed()}, progress={"phase": 2,
                    "status": "Phase 2 complete; readiness and forecast check-in",
                    "active_seconds": elapsed(), "cumulative_phase2_seconds": prior_seconds+elapsed(),
                    "counts": {"step": count, "target": count}})
                parent.complete(result)
                refresh()
                return result
            except BaseException:
                parent.checkpoint({"charged_seconds": elapsed()}, progress={"phase": 2,
                    "status": "checkpointed interruption; explicit resume required", "active_seconds": elapsed(),
                    "cumulative_phase2_seconds": prior_seconds+elapsed(), "counts": {"step": count, "target": 12}})
                refresh()
                raise
    finally:
        stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cache/numerical_experiment/picar/phase2/readiness-rms-v1"))
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--teacher-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    result = run(args.output, args.source_campaign, args.teacher_source, model_path=args.model_path,
                 offline=args.offline, resume=args.resume, wait=args.wait)
    print(json.dumps({k: result.get(k) for k in ("status", "units", "trainability_observed", "cumulative_phase2_seconds")}, indent=2))


if __name__ == "__main__":
    main()
