"""Resumable Phase 1 campaign, ending with a resource forecast for review."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import canonical_json
from .phase1 import PROJECT, Recipe, _identity, launch_probe, resolved
from .runtime import StopController, check_storage, sha256_file, source_fingerprint
from .storage import RunStore


def _read(path):
    return json.loads(Path(path).read_text())


def _report():
    from .report import build_notebook
    build_notebook(PROJECT / "cache/numerical_experiment/picar")


def _write_unit(root, name, kind, payload, *, recipe, metadata=None, resume=False):
    identity = {"config": resolved(recipe).to_dict(), "phase1": {"kind": kind},
                "metadata": metadata or {}}
    with RunStore(root / "units" / name, identity, resume=resume) as store:
        if store.completed:
            return _read(store.root / "summary.json")
        summary = {"schema_version": 1, "phase": 1, "smoke_only": True, "kind": kind, **payload}
        store.checkpoint({"stage": "done"}, progress={"phase": 1, "status": "done", "counts": {"step": 1}})
        store.complete(summary)
        return summary


def scene_unit(root, *, resume=False):
    from src.robotics.simulator import SimulatorConfig
    from .phase1_scene import SceneAuditConfig, run_scene_audit
    recipe = Recipe()
    identity = {"config": resolved(recipe).to_dict(), "phase1": {"kind": "phase1_scene_audit"},
        "source_sha256": {str(p): sha256_file(p) for p in
            [PROJECT / "numerical_experiment/picar/phase1_scene.py", PROJECT / "src/robotics/simulator.py"]}}
    path = root / "units" / "scene-audit"
    with RunStore(path, identity, resume=resume or (path / "COMPLETED").exists()) as store:
        if store.completed:
            return _read(path / "summary.json")
        # A halted attempt keeps its files; a fresh payload directory retries it.
        started = time.monotonic()
        output = path / f"payload-{time.time_ns()}"
        summary = run_scene_audit(output, config=SceneAuditConfig(
            simulator=SimulatorConfig(width=320, height=240, realtime=False)))
        summary.update(smoke_only=True, active_seconds=time.monotonic() - started,
                       schema_version=1)
        store.checkpoint({"payload": str(output)}, progress={"phase": 1, "status": "done", "counts": {"step": 1}})
        store.complete(summary)
        return summary


def _cached_probe(root, recipe, model_path):
    wanted = _identity(recipe, model_path, "phase1_resource_probe")
    for path in sorted((root / "units").glob("*/summary.json")):
        if (path.parent / "COMPLETED").exists() and (path.parent / "config.json").exists():
            if _read(path.parent / "config.json") == wanted:
                with RunStore(path.parent, wanted, resume=True):
                    return _read(path)
    return None


def campaign(root: Path, *, model_path=None, offline=False, resume=False, max_hours=3., repeats=3):
    root.mkdir(parents=True, exist_ok=True)
    recipe = Recipe(frozen_prefix_cache=True)
    identities = {"config": resolved(recipe).to_dict(), "phase1": {"kind": "phase1_campaign",
        "max_hours": max_hours, "repeats": repeats},
        "model_source": str(model_path.resolve()) if model_path else resolved(recipe).model.model_id}
    stop = StopController(root)
    stop.install()
    orchestration = root / "orchestration"
    try:
        with RunStore(orchestration, identities, resume=resume,
                      metadata={"initial_source_sha256": source_fingerprint()}) as store:
            if store.completed:
                return _read(orchestration / "summary.json")
            previous = store.load_checkpoint(trusted=True) or {"charged_s": 0., "step": 0}
            started = time.monotonic()
            already = previous["charged_s"]
            step = previous["step"]
            def progress(stage):
                nonlocal step
                step += 1
                elapsed = already + time.monotonic() - started
                store.checkpoint({"charged_s": elapsed, "step": step, "source_sha256": source_fingerprint()},
                    transaction_id=f"stage-{step}", progress={"phase": 1, "status": stage,
                        "active_seconds": elapsed, "counts": {"step": step}})
                _report()
                if stop.check():
                    raise InterruptedError("checkpointed Phase 1 stop request")
                if elapsed >= max_hours * 3600:
                    raise TimeoutError("Phase 1 active-time allowance reached")
                check_storage(root, quota_bytes=100_000_000_000, reserve_bytes=20_000_000_000)

            def probe(candidate):
                cached = _cached_probe(root, candidate, model_path)
                if cached is not None:
                    return cached
                elapsed = already + time.monotonic() - started
                if max_hours * 3600 - elapsed < 120:
                    raise TimeoutError("insufficient Phase 1 allowance for another probe")
                try:
                    result = launch_probe(root, candidate, model_path=model_path, offline=offline,
                        resume=resume, repeats=repeats,
                        timeout_s=min(900, max_hours * 3600 - elapsed))
                finally:
                    progress("resource profiling")
                return result

            scene = scene_unit(root, resume=resume)
            progress("scene audit complete")
            baseline = probe(recipe)
            if baseline["status"] != "passed":
                recipe = replace(recipe, image_width=112, image_height=84, max_pixels=112 * 84)
                baseline = probe(recipe)
            if baseline["status"] != "passed":
                recipe = replace(recipe, context_length=4)
                baseline = probe(recipe)
            if baseline["status"] != "passed":
                raise RuntimeError("no admitted top-L=1 baseline; preserve diagnostics and review scope")
            candidates = [baseline]
            two = probe(replace(recipe, top_layers=2))
            candidates.append(two)
            if two["status"] == "passed" and two["memory"]["headroom_bytes"] >= 2_500_000_000:
                candidates.append(probe(replace(recipe, top_layers=4)))
            else:
                _write_unit(root, "skip-top4", "phase1_resource_decision", {
                    "status": "not_run", "reason": "Top-2 profile leaves insufficient margin for top-4",
                    "active_seconds": 0.}, recipe=recipe, resume=resume)
            for alternate in (replace(recipe, hidden_size=128), replace(recipe, ltm_rank=8)):
                candidates.append(probe(alternate))
            batch4 = probe(replace(recipe, batch=4))
            candidates.append(batch4)
            if batch4["status"] == "passed" and batch4["memory"]["headroom_bytes"] >= 2_000_000_000:
                candidates.append(probe(replace(recipe, batch=8)))
            else:
                _write_unit(root, "skip-batch8", "phase1_resource_decision", {
                    "status": "not_run", "reason": "Batch-4 memory does not admit a larger online batch",
                    "active_seconds": 0.}, recipe=recipe, resume=resume)
            candidates.append(probe(replace(recipe, context_length=16)))
            selected = batch4 if batch4["status"] == "passed" else baseline
            chosen = Recipe(**selected["recipe"])
            for alternate in (replace(chosen, fd_probes=2), replace(chosen, fisher_draws=2), replace(chosen, fisher_draws=4)):
                candidates.append(probe(alternate))
            progress("resource shortlist complete")
            # Both actual teacher/evaluator checks and realtime control are
            # separate resumable units, implemented in the next module.
            from .phase1_roles import launch_roles
            try:
                roles = launch_roles(root, chosen, scene, model_path=model_path, offline=offline, resume=resume)
            finally:
                progress("roles and control complete")
            from .phase1_forecast import forecast
            from .report import load_report
            phase0 = load_report(PROJECT / "cache/numerical_experiment/picar/phase0")
            phase0_s = sum(u["summary"].get("active_seconds", 0) for u in phase0["complete_units"])
            phase0_s += sum(d["summary"].get("active_seconds", 0) for d in phase0["diagnostics"])
            # Count retained earlier attempts too. Wall time catches current
            # launches/reporting; child totals catch units completed beforehand.
            phase1_snapshot = load_report(PROJECT / "cache/numerical_experiment/picar/phase1")
            phase1_unit_s = sum(u["summary"].get("active_seconds", 0)
                                for u in phase1_snapshot["complete_units"])
            phase1_s = max(already + time.monotonic() - started, phase1_unit_s)
            spent_h = (phase0_s + phase1_s) / 3600
            period = roles.get("recommended_control_period_s", max(.5,
                math.ceil(selected["timings"]["inference_seconds"] * 1.5 * 4) / 4))
            alternatives = []
            blockers = []
            if not scene.get("all_feasible"):
                blockers.append("Scripted controller did not meet the task criterion on every audited layout.")
            if not scene.get("reward_panel_truth_verified"):
                blockers.append("Some administrative reward-panel truths did not match their intended geometry.")
            if not roles.get("readiness_pass", False):
                blockers += roles.get("blockers", ["Teacher/evaluator or control readiness remains unresolved."])
            for pairs, length in ((6, 2400), (4, 1200)):
                result = forecast(selected, pairs=pairs, transitions=length, control_period_s=period, elapsed_hours=spent_h)
                result["blockers"] = blockers
                result["active_seconds"] = 0.
                result = _write_unit(root, f"forecast-{pairs}pairs-{length}", "phase1_forecast", result,
                    recipe=chosen, metadata={"pairs": pairs, "transitions": length}, resume=resume)
                alternatives.append(result)
            progress("forecast ready for review")
            elapsed = already + time.monotonic() - started
            child_s = sum(_read(path).get("active_seconds", 0) for path in (root / "units").glob("*/summary.json")
                          if (path.parent / "COMPLETED").exists())
            summary = {"schema_version": 1, "phase": 1, "smoke_only": True,
                "kind": "phase1_campaign", "selected_recipe": asdict(chosen), "status": "ready_for_checkin",
                "total_phase1_active_seconds": elapsed, "active_seconds": max(0., elapsed - child_s),
                "forecast_elapsed_hours": spent_h,
                "forecasts": [{"pairs": r["pairs"], "transitions": r["transitions_per_arm"],
                    "hours": r["total_hours_range"]} for r in alternatives], "blockers": blockers,
                "scientific_phases_started": False}
            store.checkpoint({"charged_s": elapsed, "step": step}, progress={"phase": 1,
                "status": "ready for check-in", "counts": {"step": step}})
            store.complete(summary)
            _report()
            return summary
    finally:
        stop.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-hours", type=float, default=3.)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.resume:
        (args.output / "STOP").unlink(missing_ok=True)
    print(json.dumps(campaign(args.output, model_path=args.model_path, offline=args.offline,
        resume=args.resume, max_hours=args.max_hours, repeats=args.repeats), indent=2))


if __name__ == "__main__":
    main()
