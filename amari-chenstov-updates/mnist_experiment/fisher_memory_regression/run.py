"""Standalone resumable Plan 15 Phase 8 calibration through final analysis."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, utc_now
from mnist_experiment.fisher_memory.learner import StopRequested as LearnerStop
from mnist_experiment.fisher_memory.run import budget_lock

from . import PROTOCOL
from .config import Config, Recipe, ROOT, PACKAGE, content_hash
from .data import MNISTSource
from .design import aggregate, choose, rank_candidates, grid, confirmation_manifest
from .runner import Runner, trajectory_id
from .runtime import RuntimeBudget, dataset_manifest, file_bytes, other_compute_seconds, provenance, read_json, setup_torch


class SolverAdequacyError(ValueError):
    pass


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE / "configs/default.json")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets")
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--fit-review", type=Path,
                        help="optional development-only visual review JSON, frozen before confirmation")
    parser.add_argument("--stage", choices=("calibrate", "allocate", "confirm", "analyze", "all"), default="all")
    parser.add_argument("--allocation-hours", type=float, required=True,
                        help="cumulative Phase 8 allowance, never additional time per resume (max 20)")
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--smoke-stop-at", choices=("pretrain", "accepted", "mid_solve", "eviction"))
    return parser.parse_args(argv)


class Study:
    def __init__(self, cfg, args, store, budget):
        self.cfg, self.args, self.store, self.budget = cfg, args, store, budget
        self.started = time.monotonic()
        self.last_status = self.last_report = 0.
        self.completed_this_invocation = 0
        self.records = {}
        self.calibration_namespace = None
        self.status = dict(schema_version=1, protocol=PROTOCOL, smoke=cfg.smoke,
                           status="running", stage=args.stage, current=None,
                           completed=0, planned=0, changes=[], forecast_seconds=None)
        old_status = store.root / "status.json"
        if old_status.exists():
            prior = read_json(old_status)
            for key in ("changes", "family", "stage", "completed", "planned", "forecast_seconds"):
                if key in prior:
                    self.status[key] = prior[key]
        self.runner = Runner(cfg, store, MNISTSource(args.data_root), budget,
                             on_progress=self.progress, stop_at=args.smoke_stop_at)
        for record in store.iter_units(completed_only=True):
            if record["metadata"].get("kind") == "regression_trajectory":
                self.records[record["unit_id"]] = self.runner.record(record["unit_id"])

    def _decision_name(self, name):
        return f"{self.calibration_namespace}-{name}" if self.calibration_namespace and name != "calibration" else name

    def decision(self, name):
        name = self._decision_name(name)
        unit = "decision-" + name
        return read_json(self.store.unit_path(unit) / "result.json") if self.store.is_complete(unit) else None

    def seal(self, name, result):
        existing = self.decision(name)
        if existing is not None:
            if existing != result:
                raise ValueError(f"sealed regression decision changed: {name}")
            return existing
        unit = "decision-" + self._decision_name(name)
        self.store.begin(unit, dict(schema_version=1, kind="regression_decision", name=name))
        self.store.write_json(unit, "result.json", result)
        self.store.complete(unit)
        return result

    def progress(self, *, force=False, final=False, **updates):
        self.status.update(updates)
        if not force and time.monotonic() - self.last_status < 2:
            return
        family = self.status.get("family")
        if family:
            self.status["completed"] = sum(row["family"] == family for row in self.records.values())
            self.status["planned"] = max(self.status["planned"], self.status["completed"])
        manifest = self.decision("confirmation-manifest") if not self.calibration_namespace else None
        if manifest:
            keys = {recipe["recipe_key"] for recipe in manifest["recipes"]}
            self.status["paired_planned"] = len(manifest["replicas"])
            self.status["paired_completed"] = sum(keys <= {
                row["recipe_key"] for row in self.records.values()
                if row["family"] == "confirmation" and row["replica"] == replica}
                for replica in manifest["replicas"])
        self.status.update(active_seconds=self.budget.used,
            allocation_seconds=self.budget.allocation_seconds, heartbeat=utc_now(),
            storage=self.store.storage_status())
        self.store.write_json(None, "status.json", self.status, finalizing=True)
        self.last_status = time.monotonic()
        if final:
            try:
                self.budget.check()
            except (StopRequested, ResourceLimit):
                return
        if force or time.monotonic() - self.last_report >= 900:
            from .report import publish
            snapshot = self.snapshot()
            self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=True)
            target = self.args.notebook or self.store.root / "results.ipynb"
            publish(snapshot, target, store=self.store)
            self.last_report = time.monotonic()

    def snapshot(self):
        calibration = self.decision("calibration")
        allocation = [row for row in self.records.values()
                      if row["family"] == "allocation" and row["target"] == self.cfg.novel_count]
        confirmation = self.decision("confirmation-manifest")
        if confirmation:
            confirmation = {**confirmation, "rows": [row for row in self.records.values()
                                                     if row["family"] == "confirmation"]}
        return dict(schema_version=1, protocol=PROTOCOL, smoke=self.cfg.smoke,
                    status=self.status["status"], stage=self.status["stage"],
                    progress={**self.status, "active_seconds": self.budget.used,
                              "allocation_seconds": self.budget.allocation_seconds},
                    calibration=calibration, allocation=allocation,
                    confirmation=confirmation, analysis=self.decision("analysis"),
                    records=list(self.records.values()), changes=self.status.get("changes", []),
                    revision=self.store.identity["source_hash"][:12],
                    allocation_grid={"capacities": list(self.cfg.capacities),
                                     "replay_counts": list(self.cfg.replay_counts)})

    def job(self, family, replica, recipe, *, target=None, parent_target=None, capture=False, audit=False):
        target = self.cfg.novel_count if target is None else target
        unit = trajectory_id(family, replica, recipe, target)
        if unit in self.records:
            return self.records[unit]
        self.budget.check()
        self.progress(current=unit, stage=family)
        partition = "confirmation" if family == "confirmation" else "development"
        row = self.runner.run(family, replica, partition, recipe, target=target,
                              parent_target=parent_target, capture=capture, audit=audit)
        self.records[unit] = row
        self.completed_this_invocation += 1
        self.status["completed"] += 1
        print(json.dumps(dict(event="regression_trajectory_complete", stage=family,
                             unit=unit, method=recipe.method, novel_count=target,
                             novel_auc=row["novel_auc"], valid=row["valid"])), flush=True)
        self.progress(force=True, current=None)
        if self.args.max_units and self.completed_this_invocation >= self.args.max_units:
            raise StopRequested("requested completed-unit batch reached")
        return row

    def _control_screen(self, policy):
        recipes = [Recipe.from_config(self.cfg, "replay_ewc", ridge=c, policy=policy)
                   for c in self.cfg.ridge_grid]
        rows = []
        for replica in range(self.cfg.calibration_replicas):
            for recipe in recipes:
                rows.append(self.job("calibration", replica, recipe,
                    target=self.cfg.screen_novel_count, capture=True, audit=True))
        return rows

    def _solver_audit(self, rows):
        audits = [item for row in rows for item in read_json(
            self.store.unit_path(row["unit_id"]) / "audits.json")]
        failed = sum(item["enlarged_failed"] for item in audits)
        material = sum(item["relative_objective_gain"] is None or
                       item["relative_objective_gain"] > self.cfg.audit_objective_tolerance for item in audits)
        inadequate = not audits or failed > 0 or material / len(audits) >= .25
        return dict(audit_count=len(audits), material_audits=material, failed_audits=failed,
                    adequate=not inadequate, smoke_gate_bypassed=bool(inadequate and self.cfg.smoke))

    def calibrate(self):
        if self.decision("calibration"):
            return
        try:
            for attempt in range(3):
                self.calibration_namespace = f"calibration-attempt-{attempt}"
                policy = dict(max_iter=self.cfg.max_iter * self.cfg.audit_multiplier ** attempt,
                              max_eval=self.cfg.max_eval * self.cfg.audit_multiplier ** attempt)
                try:
                    self._calibrate_once(policy, attempt)
                    return
                except SolverAdequacyError as exc:
                    change = dict(stage="8B", change=f"Shared solver audit repair {attempt + 1}",
                        reason=str(exc), before=policy,
                        after={key: value * self.cfg.audit_multiplier for key, value in policy.items()},
                        validity="Development only; new recipe identities, same paired streams.")
                    if change not in self.status["changes"]:
                        self.status["changes"].append(change)
                    self.progress(force=True)
            raise ValueError("Shared solver still fails adequacy after two ceiling repairs; inspect sealed audits before a versioned repair")
        finally:
            self.calibration_namespace = None

    def _calibrate_once(self, policy, attempt):
        cfg = self.cfg
        self.status.update(stage="8B", family="calibration", planned=(attempt + 1) * cfg.calibration_replicas *
                           (len(cfg.ridge_grid) + len(cfg.strength_ratios) + 5), completed=0)
        policy_record = self.decision("solver-policy")
        if policy_record is None:
            rows = self._control_screen(policy)
            policy_record = self.seal("solver-policy", dict(policy=policy, attempt=attempt,
                                                             **self._solver_audit(rows)))
        if not policy_record["adequate"] and not cfg.smoke:
            raise SolverAdequacyError("Control-screen objective gaps remain material")
        policy = policy_record["policy"]
        control_rows = self._control_screen(policy)
        ridge_record = self.decision("ridge")
        if ridge_record is None:
            selected = choose(aggregate(control_rows), cfg)
            ridge_record = self.seal("ridge", selected)
        ridge = ridge_record["selected"]["recipe"]["ridge"]
        control = Recipe.from_config(cfg, "replay_ewc", ridge=ridge, policy=policy)
        numerical = self.decision("fd-numerical")
        if numerical is None:
            from .numerical import CalibrationError, calibrate_fd
            snapshots = []
            for replica in range(cfg.calibration_replicas):
                snapshots.extend(self.runner.snapshots("calibration", replica, control, cfg.screen_novel_count))
            model, layout = self.runner._model("calibration", 0)
            try:
                numerical = self.seal("fd-numerical", calibrate_fd(cfg, snapshots, model, layout, self.budget.check))
            except CalibrationError as exc:
                self.seal("fd-numerical-failure", exc.diagnostics)
                self.status["changes"].append(dict(stage="8B", change="FD numerical calibration failed",
                    reason=str(exc), validity="No recipe approved; inspect sealed numerical-failure diagnostics."))
                raise
        lambda0 = numerical["lambda0"]
        fd_step = numerical["selected_step"]
        recipes = [Recipe.from_config(cfg, "fd_replay_ewc", ridge=ridge, strength=lambda0 * ratio,
                                      fd_step=fd_step, policy=policy) for ratio in cfg.strength_ratios]
        screen_rows = []
        for replica in range(cfg.calibration_replicas):
            for recipe in recipes:
                screen_rows.append(self.job("calibration", replica, recipe, target=cfg.screen_novel_count,
                                            capture=True, audit=True))
        shortlist = self.decision("strength-shortlist")
        if shortlist is None:
            ranked, warning = rank_candidates(aggregate(screen_rows), cfg)
            shortlist = self.seal("strength-shortlist", dict(recipes=[row["recipe"] for row in ranked[:2]],
                                                            warning=warning))
        boundary = Recipe.from_config(cfg, "ewc_only", ridge=ridge, policy=policy)
        selected_fd = [Recipe(**recipe) for recipe in shortlist["recipes"]]
        full_rows = []
        for replica in range(cfg.calibration_replicas):
            for recipe in (control, *selected_fd):
                full_rows.append(self.job("calibration", replica, recipe, parent_target=cfg.screen_novel_count,
                                          capture=True, audit=True))
            full_rows.append(self.job("calibration", replica, boundary, audit=True))
        initial_selection = choose(aggregate([row for row in full_rows if row["method"] == "fd_replay_ewc"]), cfg)
        extension = self.decision("strength-extension")
        if extension is None:
            strength = initial_selection["selected"]["recipe"]["strength"]
            grid_values = [r.strength for r in recipes]
            edge = strength == min(grid_values) or strength == max(grid_values)
            estimated = cfg.calibration_replicas * initial_selection["selected"]["mean_active_seconds"] * 1.5
            affordable = (self.budget.remaining - estimated > cfg.confirmation_reserve_seconds + cfg.analysis_reserve_seconds)
            extension = self.seal("strength-extension", dict(run=bool(edge and affordable),
                strength=strength / 10 if strength == min(grid_values) else strength * 10,
                reason="one adjacent edge probe" if edge and affordable else "interior selection or time reservation"))
        if extension["run"]:
            extra = Recipe.from_config(cfg, "fd_replay_ewc", ridge=ridge, strength=extension["strength"],
                                       fd_step=fd_step, policy=policy)
            for replica in range(cfg.calibration_replicas):
                full_rows.append(self.job("calibration", replica, extra, capture=True, audit=True))
        choice = choose(aggregate([row for row in full_rows if row["method"] == "fd_replay_ewc"]), cfg)
        fd_recipe = Recipe(**choice["selected"]["recipe"])
        selected_rows = [row for row in full_rows if row["recipe_key"] in {control.key, fd_recipe.key, boundary.key}]
        final_audits = self.decision("full-solver-audit") or self.seal("full-solver-audit", {
            method: self._solver_audit([row for row in selected_rows if row["method"] == method])
            for method in ("replay_ewc", "fd_replay_ewc", "ewc_only")})
        if not cfg.smoke and any(not audit["adequate"] for audit in final_audits.values()):
            raise SolverAdequacyError("Selected full-budget policies fail early/middle/late objective-gap audits")
        grouped = {row["method"]: row for row in aggregate(selected_rows)}
        envelope = cfg.envelope_factor * max(grouped[m]["seconds_per_arrival"] for m in ("replay_ewc", "fd_replay_ewc"))
        assets = [read_json(self.store.unit_path(f"assets-calibration-{i:04d}") / "health.json")
                  for i in range(cfg.calibration_replicas)]
        initial_seconds = float(np.mean([row["pretrain_seconds"] + row["reference_seconds"] for row in assets]))
        memory_ceiling = int(.8 * torch.cuda.get_device_properties(0).total_memory) if cfg.device == "cuda" else 2**63 - 1
        result = dict(schema_version=1, policy=policy, ridge=ridge, strength=fd_recipe.strength,
            fd_step=fd_step, lambda0=lambda0, numerical=numerical,
            retention_margin_degrees=cfg.retention_margin_degrees,
            base_retention_guard_degrees=cfg.retention_margin_degrees,
            ridge_warning=ridge_record["warning"], strength_warning=choice["warning"],
            envelope_seconds_per_arrival=envelope, learning_seconds_per_observation_limit=envelope,
            memory_ceiling_bytes=memory_ceiling,
            initialization_status=("Better than zero predictor in every pilot replica" if all(
                row["initialization_validation"]["mse"] < row["initialization_validation"]["zero_predictor_mse"]
                for row in assets) else "Weak initialization: inspect pilot validation; no efficacy stop"),
            numerical_status="FD value and gradient checks passed; solver gaps disclosed",
            mean_active_seconds={method: row["mean_active_seconds"] for method, row in grouped.items()},
            mean_initialization_seconds=initial_seconds,
            cost_ratio=grouped["fd_replay_ewc"]["seconds_per_arrival"] / grouped["replay_ewc"]["seconds_per_arrival"],
            reference_estimates=list(grouped.values()), initialization=assets,
            solver_policy=policy_record, full_solver_audits=final_audits, selected_recipe=fd_recipe.mapping(),
            scope="smoke acceptance only" if cfg.smoke else "development only")
        self.seal("calibration", result)
        self.progress(force=True, stage="8B complete")

    def allocate(self):
        if self.decision("confirmation-manifest"):
            if self.args.fit_review:
                recorded = self.decision("confirmation-manifest").get("fit_review_hash")
                if recorded != content_hash(read_json(self.args.fit_review)):
                    raise ValueError("cannot revise a frozen fit registration")
            return
        cfg = self.cfg
        calibration = self.decision("calibration")
        if calibration is None:
            raise ValueError("allocation requires completed calibration")
        design = self.decision("allocation-grid") or self.seal("allocation-grid", grid(
            cfg, calibration, seconds_available=self.budget.remaining))
        recipes = []
        for c, b in design["cells"]:
            for method in ("replay_ewc", "fd_replay_ewc"):
                recipes.append(Recipe.from_config(cfg, method, capacity=c, replay_count=b,
                    ridge=calibration["ridge"], strength=calibration["strength"] if method == "fd_replay_ewc" else 0.,
                    fd_step=calibration["fd_step"], policy=calibration["policy"]))
        boundary = Recipe.from_config(cfg, "ewc_only", ridge=calibration["ridge"], policy=calibration["policy"])
        self.status.update(stage="8C", family="allocation", completed=0, planned=len(design["replicas"]) * (len(recipes) + 1),
                           forecast_seconds=design["predicted_seconds"])
        rows = []
        for replica in design["replicas"]:
            ordered = list(recipes) if replica % 2 == 0 else list(reversed(recipes))
            for recipe in (*ordered, boundary):
                rows.append(self.job("allocation", replica, recipe))
        cells = aggregate([row for row in rows if row["method"] != "ewc_only"])
        kwargs = dict(envelope=calibration["envelope_seconds_per_arrival"],
                      memory_ceiling=calibration["memory_ceiling_bytes"])
        selections = {method: choose([row for row in cells if row["method"] == method], cfg, **kwargs)
                      for method in ("replay_ewc", "fd_replay_ewc")}
        refinement = self.decision("refinement")
        if refinement is None:
            chosen_cells = sorted({(item["selected"]["capacity"], item["selected"]["replay_count"])
                                   for item in selections.values()})
            estimated = 1.5 * cfg.refinement_replicas * len(chosen_cells) * sum(
                calibration["mean_active_seconds"][method] for method in ("replay_ewc", "fd_replay_ewc"))
            run = cfg.refinement_replicas > 0 and self.budget.remaining - estimated > (
                cfg.confirmation_reserve_seconds + cfg.analysis_reserve_seconds)
            refinement = self.seal("refinement", dict(cells=[list(cell) for cell in chosen_cells], run=run,
                replicas=list(range(cfg.allocation_replicas, cfg.allocation_replicas + cfg.refinement_replicas)) if run else []))
        if refinement["run"]:
            shortlisted = [recipe for recipe in recipes if [recipe.capacity, recipe.replay_count] in refinement["cells"]]
            self.status["planned"] += len(refinement["replicas"]) * (len(shortlisted) + 1)
            for replica in refinement["replicas"]:
                for recipe in (*shortlisted, boundary):
                    rows.append(self.job("allocation", replica, recipe))
            cells = aggregate([row for row in rows if row["recipe_key"] in {recipe.key for recipe in shortlisted}])
            selections = {method: choose([row for row in cells if row["method"] == method], cfg, **kwargs)
                          for method in ("replay_ewc", "fd_replay_ewc")}
        self.seal("allocation-selection", selections)
        manifest = confirmation_manifest(cfg, calibration, selections, rows, self.budget.remaining)
        manifest["seed_families"] = {"confirmation": [
            cfg.seed_for("confirmation", replica, "initialization_model") for replica in manifest["replicas"]],
            "analysis": cfg.seed_for("analysis", 0, "paired_bootstrap")}
        manifest["selection_hash"] = content_hash(selections)
        if self.args.fit_review:
            review = read_json(self.args.fit_review)
            if (review.get("selection_hash") != manifest["selection_hash"]
                    or type(review.get("approved")) is not bool or not review.get("reason")
                    or not review.get("reviewer") or review.get("partition") != "development"):
                raise ValueError("fit review must identify this development selection, reviewer, reason and approval")
            manifest.update(fit=review["approved"], fit_registered=review["approved"],
                fit_range=[1, cfg.novel_count], fit_review=review, fit_review_hash=content_hash(review),
                fit_reason=review["reason"])
        self.seal("confirmation-manifest", manifest)
        self.progress(force=True, stage="8C complete", forecast_seconds=manifest["estimated_seconds"])

    def confirm(self):
        if self.decision("confirmation-complete"):
            return
        manifest = self.decision("confirmation-manifest")
        if manifest is None:
            raise ValueError("confirmation requires a sealed development selection and cohort")
        recipes = [Recipe.from_mapping(recipe) for recipe in manifest["recipes"]]
        self.status.update(stage="8D", family="confirmation", completed=0,
                           planned=len(recipes) * len(manifest["replicas"]))
        units = []
        for replica in manifest["replicas"]:
            ordered = recipes if replica % 2 == 0 else list(reversed(recipes))
            for recipe in ordered:
                row = self.job("confirmation", replica, recipe)
                units.append(dict(unit_id=row["unit_id"], seal=read_json(
                    self.store.unit_path(row["unit_id"]) / "COMPLETED")["seal_hash"]))
        self.seal("confirmation-complete", dict(manifest_hash=content_hash(manifest), units=units))
        self.progress(force=True, stage="8D complete")

    def analyze(self):
        if self.decision("analysis"):
            return
        from .analysis import analyze_confirmation
        manifest = self.decision("confirmation-manifest")
        completion = self.decision("confirmation-complete")
        if manifest is None or completion is None or completion["manifest_hash"] != content_hash(manifest):
            raise ValueError("final inference requires the complete sealed confirmation cohort")
        records = [self.runner.record(item["unit_id"]) for item in completion["units"]]
        result = analyze_confirmation(records, manifest,
            seed=self.cfg.seed_for("analysis", 0, "paired_bootstrap"),
            draws=self.cfg.bootstrap_samples, check_stop=self.budget.check)
        result["dependency_seals"] = completion["units"]
        result["manifest_hash"] = content_hash(manifest)
        self.seal("analysis", result)
        self.progress(force=True, stage="8E complete")

    def run(self):
        stages = {"calibrate": self.calibrate, "allocate": self.allocate,
                  "confirm": self.confirm, "analyze": self.analyze}
        for stage in stages if self.args.stage == "all" else [self.args.stage]:
            self.budget.check()
            stages[stage]()
        self.status.update(status="smoke_complete" if self.cfg.smoke else "completed", current=None)


def main(argv=None):
    args = arguments(argv)
    cfg = Config.read(args.config)
    if args.device:
        cfg = dataclasses.replace(cfg, device=args.device)
    cfg.validate()
    if not math.isfinite(args.allocation_hours) or not 0 < args.allocation_hours <= 20:
        raise ValueError("declare a positive cumulative allowance of at most 20 hours")
    if args.max_units is not None and args.max_units <= 0:
        raise ValueError("max-units must be positive")
    if args.smoke_stop_at and not cfg.smoke:
        raise ValueError("artificial interruption points are acceptance-only")
    if not cfg.smoke and cfg.device != "cuda":
        raise ValueError("production requires CUDA; CPU is for bounded acceptance")
    if args.root.resolve().parent != args.budget_root.resolve():
        raise ValueError("study root must be a direct child of the shared Plan 15 budget root")
    setup_torch(cfg)
    start = time.monotonic()
    stopped = False

    def preflight_check():
        if stopped or ((args.root / "STOP").exists() and not args.clear_stop):
            raise StopRequested("stop requested during bounded preflight")
        if args.max_wall_seconds is not None and time.monotonic() - start >= args.max_wall_seconds:
            raise ResourceLimit("invocation allocation exhausted during preflight")

    def stop_preflight(signum, frame):
        nonlocal stopped
        stopped = True

    previous = {sig: signal.signal(sig, stop_preflight) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        data = dataset_manifest(args.data_root, preflight_check)
        details = provenance(cfg, data)
        identity = dict(protocol=PROTOCOL, allocation_family="plan15-phase8",
                        role="smoke" if cfg.smoke else "production", schema_version=1,
                        config_hash=cfg.scientific_hash, source_hash=details["source_hash"],
                        data_hash=details["data_hash"])
        with budget_lock(args.budget_root):
            other = 0. if cfg.smoke else other_compute_seconds(args.budget_root, args.root)
            allocation = args.allocation_hours * 3600
            if other >= allocation:
                raise ResourceLimit("other Phase 8 roots already exhaust the cumulative allowance")
            external = file_bytes(args.budget_root, exclude=args.root, check=preflight_check) + file_bytes(
                args.data_root, check=preflight_check) + 48_000_000
            preflight_seconds = time.monotonic() - start
            remaining_wall = None if args.max_wall_seconds is None else max(0., args.max_wall_seconds - (time.monotonic() - start))
            with Store(args.root, identity, resume=args.resume, clear_stop=args.clear_stop,
                       max_bytes=cfg.max_artifact_bytes, external_bytes=external,
                       free_reserve_bytes=cfg.free_reserve_bytes, finalization_reserve_bytes=32_000_000,
                       max_wall_seconds=remaining_wall, max_active_seconds=allocation - other) as store:
                store.control.restore_active_seconds(store.control.active_seconds + preflight_seconds)
                if stopped:
                    store.control.stop_requested = True
                budget = RuntimeBudget(store, other_seconds=other, allocation_seconds=allocation)
                study = None
                try:
                    if not (store.root / "provenance.json").exists():
                        store.write_json(None, "provenance.json", details)
                        store.write_json(None, "resolved_config.json", cfg.mapping())
                    with store.control.signal_handlers():
                        study = Study(cfg, args, store, budget)
                        study.progress(force=True)
                        study.run()
                except (StopRequested, LearnerStop, ResourceLimit) as exc:
                    if study:
                        study.status.update(status="paused", reason=str(exc))
                    print(json.dumps(dict(event="paused", reason=str(exc))), flush=True)
                    return 2
                except BaseException as exc:
                    if study:
                        study.status.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    try:
                        if study:
                            study.progress(force=True, final=True)
                    finally:
                        budget.close()
        return 0
    except (StopRequested, ResourceLimit) as exc:
        print(json.dumps(dict(event="paused", reason=str(exc))), flush=True)
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
