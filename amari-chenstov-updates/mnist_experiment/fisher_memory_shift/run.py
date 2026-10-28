"""Unattended, resumable execution of Plan 15 Phase 9."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.control import StopRequested, ResourceLimit, utc_now
from mnist_experiment.fisher_memory.learner import StopRequested as LearnerStop
from mnist_experiment.fisher_memory.run import budget_lock
from mnist_experiment.fisher_memory_regression.config import Recipe, ROOT, METHODS, content_hash
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, read_json, setup_torch, dataset_manifest, provenance,
    scientific_sources as engine_sources, file_digest, file_bytes,
)

from . import PROTOCOL, ALLOCATION_FAMILY
from .config import Config
from .data import MNISTSource, Context, default_contexts
from .runner import Runner, trajectory_id, NumericalIntegrityError


def sources():
    result = engine_sources()
    extra = [ROOT / "mnist_experiment/plan15_retention_efficiency.py"]
    extra.extend((ROOT / "mnist_experiment/fisher_memory_retention").glob("*.py"))
    extra.extend(Path(__file__).parent.glob("*.py"))
    for path in extra:
        if path.name not in ("report.py", "acceptance.py"):
            result[str(path.relative_to(ROOT))] = file_digest(path)
    return dict(sorted(result.items()))


def peer_seconds(budget_root, root):
    total = 0.
    for peer in budget_root.iterdir():
        identity = peer / "identity.json"
        if peer.resolve() == root.resolve() or not identity.is_file():
            continue
        info = read_json(identity)["identity"]
        if info.get("allocation_family") != ALLOCATION_FAMILY or info.get("role") == "smoke":
            continue
        runtime = peer / "runtime.json"
        if runtime.exists():
            value = read_json(runtime)["cumulative_active_seconds"]
            lease = peer / "timing_lease.json"
            if lease.exists():
                value = max(value, read_json(lease).get("charged_until", value))
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid Phase 9 compute ledger")
            total += value
    return total


class Study:
    def __init__(self, cfg, args, store, budget):
        self.cfg, self.args, self.store, self.budget = cfg, args, store, budget
        self.last_status = self.last_report = 0.
        self.new_units = 0
        self.status = dict(status="running", stage="9A", current=None,
                           completed=0, planned=0, changes=[])
        if (store.root / "status.json").exists():
            self.status.update(read_json(store.root / "status.json"))
            self.status.update(status="running", reason=None)
        self.runner = Runner(cfg, store, MNISTSource(args.data_root), budget,
                             on_progress=self.progress, stop_at=args.smoke_stop_at)
        self.contexts = {c.key: c for c in cfg.contexts}
        self.runner.contexts = self.contexts
        self.records = {}
        self.job_calibration = {}
        for item in store.iter_units(completed_only=True):
            if item["metadata"].get("kind") == "regression_trajectory":
                self.records[item["unit_id"]] = self.runner.record(item["unit_id"])
        initialized = self.decision("initialization")
        if initialized:
            self.runner.init_epochs = initialized["epochs"]
        contexts = self.decision("contexts")
        if contexts:
            self.contexts = {c["key"]: Context.from_mapping(c) for c in contexts["contexts"]}
            self.runner.contexts = self.contexts
        combined = self.decision("combined-context")
        if combined and combined.get("admitted"):
            c = Context.from_mapping(combined["context"])
            self.contexts[c.key] = c

    def decision(self, name):
        unit = "decision-" + name
        if self.store.is_complete(unit):
            return read_json(self.store.unit_path(unit) / "result.json")
        return None

    def seal(self, name, result):
        old = self.decision(name)
        if old is not None:
            if old != result:
                raise ValueError(f"sealed decision changed: {name}")
            return old
        unit = "decision-" + name
        self.store.begin(unit, dict(kind="shift_decision", schema_version=1, name=name))
        self.store.write_json(unit, "result.json", result)
        self.store.complete(unit)
        return result

    def context_records(self, context, *, partition="development"):
        return [r for r in self.records.values() if r["partition"] == partition
                and r["family"].endswith("--" + context.key)]

    def limits(self, context):
        item = self.decision("calibration-" + context.key)
        if not item:
            return dict(learning_limit=float("inf"), memory_limit=2**63 - 1)
        return dict(learning_limit=item["learning_time_limit"],
                    memory_limit=item["memory_ceiling_bytes"])

    def rows(self, context, *, replicas=None, partial=False):
        from .design import table
        return table(self.context_records(context), self.cfg, replicas=replicas,
                     partial=partial, **self.limits(context))

    def snapshot(self):
        from .design import table
        development = {}
        for context in self.contexts.values():
            records = [r for r in self.context_records(context)
                       if r["replica"] < self.cfg.calibration_replicas]
            development[context.key] = table(records, self.cfg, partial=True,
                                             **self.limits(context)) if records else []
        return dict(schema_version=1, protocol=PROTOCOL, config=self.cfg.mapping(),
            progress=self.status, contexts=[c.mapping() for c in self.contexts.values()],
            initialization=self.decision("initialization"), readiness=self.decision("contexts"),
            development=development, selection=self.decision("selection"),
            manifest=self.decision("confirmation-manifest"), analysis=self.decision("analysis"),
            confirmation=[r for r in self.records.values() if r["partition"] == "confirmation"],
            changes=self.status.get("changes", []))

    def progress(self, force=False, final=False, **updates):
        updates.pop("stage", None)
        self.status.update(updates)
        now = time.monotonic()
        if not force and now - self.last_status < 2:
            return
        self.status.update(schema_version=1, protocol=PROTOCOL, completed=len(self.records),
            active_seconds=self.budget.used, allocation_seconds=self.budget.allocation_seconds,
            heartbeat=utc_now(), storage=self.store.storage_status())
        durations = [r["active_seconds"] for r in self.records.values()]
        if durations:
            self.status["forecast_seconds"] = 1.5 * max(0, self.status["planned"] - len(self.records)) * float(
                np.quantile(durations, .9))
        self.store.write_json(None, "status.json", self.status, finalizing=True)
        self.last_status = now
        if final:
            try:
                self.budget.check()
            except (StopRequested, ResourceLimit):
                return
        if force or now - self.last_report >= 900:
            snapshot = self.snapshot()
            self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=final)
            from .report import publish
            publish(snapshot, self.args.notebook or self.store.root / "results.ipynb", self.store)
            self.last_report = now

    def job(self, context, replica, recipe, stage="development"):
        family = stage + "--" + context.key
        unit = trajectory_id(family, replica, recipe, self.cfg.novel_count)
        capture = stage == "development" and replica == 0
        audit = capture and recipe.ridge == 10 and recipe.replay_count == self.cfg.reference_replay
        if unit in self.records:
            self.check_audit(unit, audit)
            return self.records[unit]
        self.budget.check()
        self.progress(current=unit)
        record = self.runner.run(family, replica, stage, recipe, capture=capture, audit=audit)
        self.check_audit(unit, audit)
        self.records[unit] = record
        self.new_units += 1
        print(json.dumps(dict(event="phase9_trajectory_complete", unit=unit, context=context.key,
            method=recipe.method, replica=replica, novel_auc=record["novel_auc"],
            base_harm=math.sqrt(record["base_auc"]) - math.sqrt(record["frozen_base_mse"]),
            learning_seconds=record["learning_seconds"], valid=record["valid"])), flush=True)
        self.progress(force=False, current=None)
        if self.args.max_units and self.new_units >= self.args.max_units:
            self.progress(force=True)
            raise StopRequested("requested completed-trajectory batch reached")
        return record

    def check_audit(self, unit, requested):
        if not requested or self.cfg.smoke:
            return
        rows = read_json(self.store.unit_path(unit)/"audits.json")
        failed = [r for r in rows if r["enlarged_failed"]]
        material = [r for r in rows if r["relative_objective_gain"] is None
                    or r["relative_objective_gain"] > self.cfg.audit_objective_tolerance]
        if not rows or failed or len(material)/len(rows) >= .25:
            self.seal("audit-failure-" + unit, dict(unit=unit, audits=len(rows),
                failed=len(failed), material=len(material),
                reason="common solver adequacy requires a documented paired repair"))
            raise NumericalIntegrityError("solver audit inadequate; preserve this version and repair paired solves")

    def register(self):
        self.seal("design", dict(protocol=PROTOCOL, schema_version=1, config=self.cfg.mapping(),
            contexts=[c.mapping() for c in self.cfg.contexts],
            source_hash=self.store.identity["source_hash"],
            score="fraction of novel samples saved / matched-quality learning compute multiplier",
            target="observed mean replay novel MSE after 100 unique novel images",
            correction_policy="Plan15 Phase9 autonomous versioned corrections; preserve all evidence"))
        self.progress(force=True)

    def initialize(self):
        if self.decision("initialization"):
            return
        self.status["stage"] = "9B nominal capability"
        candidates = []
        for epochs in self.cfg.init_epoch_stages:
            self.runner.init_epochs = epochs
            checks = []
            for replica in range(self.cfg.calibration_replicas):
                unit, state, _, _, _ = self.runner.assets("development", replica, "development")
                value = state["initialization_validation"]
                checks.append(dict(replica=replica, unit=unit, **value))
                self.progress(force=True)
            ratio = sum(v["mse"] for v in checks) / sum(v["zero_predictor_mse"] for v in checks)
            candidates.append(dict(epochs=epochs, mse_ratio=ratio, checks=checks))
            if ratio <= .5:
                break
        qualified = [row for row in candidates if row["mse_ratio"] <= .5]
        selected = qualified[0] if qualified else min(candidates, key=lambda row: row["mse_ratio"])
        self.runner.init_epochs = selected["epochs"]
        self.seal("initialization", dict(**selected, capability_supported=bool(qualified),
                                         tried_epochs=[r["epochs"] for r in candidates]))
        if not qualified:
            self.status["changes"].append("Nominal angle capability missed the readiness preference; best common epoch recipe retained.")
        self.progress(force=True)

    def readiness(self):
        from .calibration import diagnostics
        if self.decision("contexts"):
            return
        self.status["stage"] = "9B context learnability"
        initial = list(self.contexts.values())
        readiness = {}
        for context in initial:
            rows = [diagnostics(self.runner, context, r) for r in range(self.cfg.calibration_replicas)]
            readiness[context.key] = rows
            self.progress(force=True)
        admitted = list(initial)
        if not self.cfg.smoke:
            for family in ("gamma", "blur"):
                pair = [c for c in initial if c.family == family]
                pair.sort(key=lambda c: c.gamma_max if family == "gamma" else c.blur_max)
                weak, strong = pair
                mean_ratio = lambda c: float(np.mean([r["frozen_degradation_ratio"] for r in readiness[c.key]]))
                recovery = lambda c: all(r["recovered"] for r in readiness[c.key])
                replacement = None
                if max(mean_ratio(weak), mean_ratio(strong)) < 1.1:
                    replacement = Context(key=family + "-extended", family=family,
                        gamma_max=4.5 if family == "gamma" else 1.,
                        blur_max=1.8 if family == "blur" else 0.)
                elif mean_ratio(strong) > 2 and not recovery(strong):
                    replacement = Context(key=family + "-midpoint", family=family,
                        gamma_max=(weak.gamma_max + strong.gamma_max)/2 if family == "gamma" else 1.,
                        blur_max=(weak.blur_max + strong.blur_max)/2 if family == "blur" else 0.)
                if replacement:
                    self.contexts[replacement.key] = replacement
                    self.runner.contexts = self.contexts
                    rows = [diagnostics(self.runner, replacement, r) for r in range(self.cfg.calibration_replicas)]
                    readiness[replacement.key] = rows
                    admitted = [replacement if c.key == strong.key else c for c in admitted]
        sealed = self.seal("contexts", dict(contexts=[c.mapping() for c in admitted],
            readiness=readiness, limitations="Readiness preferences do not cancel weak but valid contexts."))
        self.contexts = {c["key"]: Context.from_mapping(c) for c in sealed["contexts"]}
        self.runner.contexts = self.contexts
        self.progress(force=True)

    def calibrate_context(self, context):
        from .calibration import coefficient
        name = "calibration-" + context.key
        if self.decision(name):
            return self.decision(name)
        self.status["stage"] = "9B numerical and cost calibration"
        reference = Recipe.from_config(self.cfg, "replay_ewc",
                                      replay_count=self.cfg.reference_replay, ridge=10.)
        records = [self.job(context, r, reference) for r in range(self.cfg.calibration_replicas)]
        numerical = coefficient(self.runner, context, records[0])
        limit = 2.5 * sum(r["learning_seconds"] for r in records) / sum(r["observations"] for r in records)
        memory = (int(.8 * torch.cuda.get_device_properties(0).total_memory)
                  if self.cfg.device == "cuda" else 2**63-1)
        audits = [row for r in records for row in read_json(self.store.unit_path(r["unit_id"])/"audits.json")]
        result = self.seal(name, dict(lambda0=numerical["lambda0"], fd_step=numerical["selected_step"],
            learning_time_limit=limit, memory_ceiling_bytes=memory, reference_units=[r["unit_id"] for r in records],
            numerical_unit="numerics-" + context.key, solver_audits=audits,
            numerical_source="fixed reference snapshots at novel events 25 and 75"))
        self.progress(force=True)
        return result

    def calibration(self):
        self.initialize()
        self.readiness()
        initial_contexts = [self.contexts[c["key"]] for c in self.decision("contexts")["contexts"]]
        for context in initial_contexts:
            self.calibrate_context(context)
        if not self.decision("calibration-complete"):
            choices = [self.decision("calibration-" + c.key) for c in initial_contexts]
            common = {self.cfg.fd_step/2, self.cfg.fd_step, self.cfg.fd_step*2}
            for context in initial_contexts:
                record = read_json(self.store.unit_path("numerics-" + context.key)/"result.json")
                audit = record["numerical"]
                passing = {h for h in audit["candidate_steps"] if all(
                    r["passed"] for r in audit["numerical_rows"] if r["fd_step"] == h)}
                common &= passing
            if not common:
                raise ValueError("no common FD step passes every context in both precisions")
            step = min(common, key=lambda h: (abs(math.log2(h/self.cfg.fd_step)), h))
            self.seal("calibration-complete", dict(fd_step=step,
                contexts=list(self.contexts), init_epochs=self.runner.init_epochs,
                calibration_count=len(choices)))
        self.progress(force=True)

    def recipes(self, context):
        from .design import initial_grid
        calibration = self.decision("calibration-" + context.key)
        shared = self.decision("calibration-complete")
        return [replace(r, fd_step=shared["fd_step"])
                for r in initial_grid(self.cfg, calibration["lambda0"])]

    def grid(self, contexts=None):
        contexts = ([self.contexts[c["key"]] for c in self.decision("contexts")["contexts"]]
                    if contexts is None else contexts)
        self.status["stage"] = "9C complete-budget context screen"
        for context in contexts:
            if self.decision("grid-" + context.key):
                continue
            recipes = self.recipes(context)
            self.status["planned"] = len(self.records) + len(recipes) * self.cfg.calibration_replicas
            for replica in range(self.cfg.calibration_replicas):
                for recipe in recipes:
                    self.job(context, replica, recipe)
            self.seal("grid-" + context.key, dict(settings=len(recipes),
                trajectories=len(self.context_records(context)), context=context.mapping()))
            self.progress(force=True)

    def reserve_seconds(self):
        durations = [r["active_seconds"] for r in self.records.values()]
        if self.cfg.smoke:
            return self.cfg.confirmation_reserve_seconds + self.cfg.analysis_reserve_seconds
        asset_seconds = []
        for unit in self.store.iter_units(completed_only=True):
            if unit["metadata"].get("kind") == "shift_assets":
                h = read_json(self.store.unit_path(unit["unit_id"])/"health.json")
                asset_seconds.append(h["pretrain_seconds"] + h["reference_seconds"])
        estimate = self.cfg.confirmation_replicas * (
            3 * max(durations, default=60.) + max(asset_seconds, default=120.) + 10.)
        return max(14400., 1.5 * (estimate + self.cfg.analysis_reserve_seconds))

    def refine(self, context):
        from .design import refinement_proposals
        if self.decision("refinements-" + context.key):
            return
        additions = []
        calibration = self.decision("calibration-" + context.key)
        for index in range(self.cfg.refinement_settings):
            if self.decision("refinement-timing-stop-" + context.key):
                break
            name = f"refinement-{context.key}-{index:02d}"
            proposed = self.decision(name)
            if proposed is None:
                candidates = refinement_proposals(self.rows(context), self.cfg, additions,
                                                  calibration["lambda0"])
                if not candidates:
                    break
                expected = 1.5 * self.cfg.calibration_replicas * max(
                    r["active_seconds"] for r in self.context_records(context))
                if expected + self.reserve_seconds() > self.budget.remaining:
                    self.seal("refinement-timing-stop-" + context.key, dict(index=index,
                        remaining=self.budget.remaining, reserve=self.reserve_seconds()))
                    break
                proposed = self.seal(name, candidates[0])
            additions.append(proposed)
            recipe = Recipe.from_mapping(proposed["recipe"])
            for replica in range(self.cfg.calibration_replicas):
                self.job(context, replica, recipe)
        self.seal("refinements-" + context.key, dict(count=len(additions), additions=additions))
        self.progress(force=True)

    def score(self, context, replicas=None):
        from .design import context_score
        rows = self.context_records(context)
        if replicas is not None:
            rows = [r for r in rows if r["replica"] in replicas]
        return context_score(rows, self.cfg, replicas=replicas, **self.limits(context))

    @staticmethod
    def rank(score):
        # The design helper supplies eligible/score; no significance selection.
        point = score.get("point", {})
        value = point.get("score", point.get("S"))
        eligible = score.get("eligible", False) and value is not None
        return (not eligible, -float(value) if eligible else 0.,
                -score.get("acquisition_gain", -1e100),
                point.get("fd_learning_seconds", float("inf")))

    def combination(self):
        if self.decision("combination-complete") or self.cfg.smoke:
            return
        admitted = self.decision("combined-context")
        if admitted:
            if not admitted["admitted"]:
                return
            context = Context.from_mapping(admitted["context"])
            self._finish_combination(context)
            return
        chosen = []
        for family in ("gamma", "blur"):
            options = [(c, self.score(c)) for c in self.contexts.values() if c.family == family]
            eligible = [(c, s) for c, s in options if s.get("retention_match")
                        and s.get("resource_feasible") and s.get("acquisition_gain", 0) > 0]
            if not eligible:
                self.seal("combined-context", dict(admitted=False,
                    reason="both single families do not supply a feasible positive acquisition lead"))
                return
            chosen.append(min(eligible, key=lambda item: (self.rank(item[1]), item[0].key))[0])
        expected = 1.5 * 70 * max(r["active_seconds"] for r in self.records.values())
        if expected + self.reserve_seconds() > self.budget.remaining:
            self.seal("combined-context", dict(admitted=False, reason="confirmation reserve",
                forecast_seconds=expected, remaining_seconds=self.budget.remaining))
            return
        context = Context(key="gamma-blur-combined", family="composite",
                          gamma_max=chosen[0].gamma_max, blur_max=chosen[1].blur_max)
        self.seal("combined-context", dict(admitted=True, context=context.mapping(),
            based_on=[c.key for c in chosen], forecast_seconds=expected))
        self._finish_combination(context)

    def _finish_combination(self, context):
        self.contexts[context.key] = context
        self.runner.contexts = self.contexts
        from .calibration import diagnostics
        for replica in range(self.cfg.calibration_replicas):
            diagnostics(self.runner, context, replica)
        self.calibrate_context(context)
        self.grid([context])
        self.refine(context)
        self.seal("combination-complete", dict(context=context.mapping(), status="completed"))

    def finalists(self):
        from .design import shortlist, select
        if self.decision("selection"):
            return
        short = self.decision("shortlist")
        if short is None:
            scores = [(c, self.score(c)) for c in self.contexts.values()]
            chosen = sorted(scores, key=lambda item: (self.rank(item[1]), item[0].key))[:2]
            short = self.seal("shortlist", dict(contexts=[
                dict(context=c.mapping(), score=s, recipes=shortlist(self.rows(c), self.cfg))
                for c, s in chosen]))
        expanded = range(self.cfg.calibration_replicas + self.cfg.refinement_replicas)
        reports = []
        self.status["stage"] = "9C finalist development"
        for item in short["contexts"]:
            context = Context.from_mapping(item["context"])
            recipes = [Recipe.from_mapping(r) for values in item["recipes"].values() for r in values]
            for replica in expanded:
                for recipe in recipes:
                    self.job(context, replica, recipe)
            keys = {r.key for r in recipes}
            from .design import table, context_score
            records = [r for r in self.context_records(context) if r["recipe_key"] in keys]
            score = context_score(records, self.cfg, replicas=expanded, **self.limits(context))
            selected = select(table(records, self.cfg, replicas=expanded,
                                    **self.limits(context)), self.cfg)
            reports.append(dict(context=context.mapping(), score=score, policies=selected))
        winner = min(reports, key=lambda item: (self.rank(item["score"]), item["context"]["key"]))
        self.seal("selection", dict(**winner, finalists=reports, development_replicas=list(expanded)))
        self.progress(force=True)

    def confirm(self):
        selected = self.decision("selection")
        if not selected:
            raise ValueError("frozen development selection required")
        context = Context.from_mapping(selected["context"])
        manifest = self.decision("confirmation-manifest")
        if manifest is None:
            recipes = [Recipe.from_mapping(selected["policies"][m]["selected"]["recipe"]) for m in METHODS]
            estimates = [max(r["active_seconds"] for r in self.context_records(context)
                             if r["recipe_key"] == recipe.key) for recipe in recipes]
            init = self.decision("initialization")
            initial_seconds = max(read_json(self.store.unit_path(row["unit"])/"health.json")["pretrain_seconds"]
                                  for row in init["checks"])
            per_pair = 1.5 * (sum(estimates) + initial_seconds + 10.)
            sizes = [self.cfg.confirmation_replicas] if self.cfg.smoke else [12, 10, 8]
            affordable = [n for n in sizes if n * per_pair + self.cfg.analysis_reserve_seconds <= self.budget.remaining]
            if not affordable:
                raise ResourceLimit("eight fresh confirmation pairs cannot fit after development")
            n = affordable[0]
            manifest = dict(schema_version=1, protocol=PROTOCOL, sealed=True, N=self.cfg.novel_count,
                smoke=self.cfg.smoke,
                context=context.mapping(), replicas=list(range(n)),
                recipes=[{**r.mapping(), "recipe_key": r.key} for r in recipes],
                delta=self.cfg.retention_margin_degrees, epsilon=self.cfg.matching_tolerance,
                learning_time_limit=self.limits(context)["learning_limit"],
                memory_ceiling_bytes=self.limits(context)["memory_limit"],
                source_hash=self.store.identity["source_hash"], init_epochs=self.runner.init_epochs,
                bootstrap_seed=self.cfg.seed_for("confirmation", 0, "paired_bootstrap"),
                component_seeds={str(r): {name: self.cfg.seed_for("confirmation", r, name)
                    for name in ("initialization_model", "stream_sources", "stream_domains",
                                 "stream_angles_base", "stream_angles_novel",
                                 "evaluation_angles", "evaluation_strengths",
                                 "factors", "reference")} for r in range(n)},
                timing_admission=dict(per_pair_seconds=per_pair, replicas=n,
                                      remaining_seconds=self.budget.remaining))
            manifest["manifest_hash"] = content_hash(manifest)
            manifest = self.seal("confirmation-manifest", manifest)
        recipes = [Recipe.from_mapping(r) for r in manifest["recipes"]]
        self.status.update(stage="9D fresh confirmation",
                           planned=len(self.records) + len(recipes) * len(manifest["replicas"]))
        for replica in manifest["replicas"]:
            for recipe in recipes:
                self.job(context, replica, recipe, stage="confirmation")
            self.progress(force=True)

    def analyze(self):
        if self.decision("analysis"):
            return
        from .analysis import analyze
        self.status["stage"] = "9E efficiency and inference"
        manifest = self.decision("confirmation-manifest")
        if manifest is None:
            raise ValueError("frozen complete confirmation required")
        context = Context.from_mapping(manifest["context"])
        def load_block(start):
            unit = f"bootstrap-{start:04d}"
            if self.store.is_complete(unit):
                return read_json(self.store.unit_path(unit)/"result.json")
            return None
        def save_block(start, result):
            unit = f"bootstrap-{start:04d}"
            self.store.begin(unit, dict(kind="shift_bootstrap_block", schema_version=1,
                start=start, seed=manifest["bootstrap_seed"], manifest_hash=manifest["manifest_hash"]))
            self.store.write_json(unit, "result.json", result)
            self.store.complete(unit)
        result = analyze(self.context_records(context, partition="confirmation"),
            manifest, seed=manifest["bootstrap_seed"], draws=self.cfg.bootstrap_samples,
            check=self.budget.check, load_bootstrap=load_block, save_bootstrap=save_block)
        result["calibration_cost"] = dict(development_trajectories=sum(
            r["partition"] == "development" for r in self.records.values()),
            novel_training_presentations=sum(r["target"] for r in self.records.values()
                                             if r["partition"] == "development"),
            development_learning_seconds=sum(r["learning_seconds"] for r in self.records.values()
                                              if r["partition"] == "development"))
        self.seal("analysis", result)
        self.progress(force=True)

    def execute(self):
        self.register()
        if self.args.stage in ("calibrate", "all"):
            self.calibration()
        if self.args.stage in ("grid", "all"):
            if not self.decision("calibration-complete"):
                raise ValueError("calibration must finish before grid")
            self.grid()
        if self.args.stage in ("refine", "all"):
            for item in self.decision("contexts")["contexts"]:
                context = self.contexts[item["key"]]
                self.refine(context)
            self.combination()
            self.finalists()
        if self.args.stage in ("confirm", "all"):
            self.confirm()
        if self.args.stage in ("analyze", "all"):
            self.analyze()
        if self.decision("analysis") and not self.decision("closeout"):
            checked = list(self.store.iter_units(completed_only=True))
            self.seal("closeout", dict(status="completed", units=len(checked),
                source_hash=self.store.identity["source_hash"], active_seconds=self.budget.used,
                storage=self.store.storage_status()))
        self.status.update(status="completed" if self.decision("analysis") else "stage_complete",
                           current=None, forecast_seconds=0. if self.decision("analysis") else None)
        self.progress(force=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, default=ROOT/"cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT/"cache/mnist_experiment/datasets")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--acceptance-root", type=Path)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--stage", choices=("calibrate", "grid", "refine", "confirm", "analyze",
                                          "all", "status", "stop", "report"), default="all")
    parser.add_argument("--allocation-hours", type=float, default=20.)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-stop-at", choices=("pretrain", "accepted", "mid_solve", "eviction"))
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.stage in ("status", "stop", "report"):
        if args.stage == "status":
            print(json.dumps(read_json(args.root/"status.json"), indent=2))
        elif args.stage == "stop":
            (args.root/"STOP").touch(exist_ok=True)
            print("Stop requested; the latest accepted checkpoint remains recoverable.")
        else:
            from .report import publish
            publish(read_json(args.root/"report_snapshot.json"),
                    args.notebook or args.root/"results.ipynb")
        return 0
    cfg = (Config.read(args.config) if args.config else
           Config.smoke_config(args.device or "cpu") if args.smoke else Config())
    if args.device:
        cfg = replace(cfg, device=args.device)
    cfg.validate()
    if not 0 < args.allocation_hours <= 20:
        raise ValueError("cumulative allocation must be in (0,20] hours")
    if args.max_units is not None and args.max_units <= 0:
        raise ValueError("positive completed-unit limit required")
    if args.smoke_stop_at and not cfg.smoke:
        raise ValueError("artificial interruptions are smoke-only")
    if not cfg.smoke and cfg.device != "cuda":
        raise ValueError("production requires CUDA")
    if args.root.resolve().parent != args.budget_root.resolve():
        raise ValueError("study root must be directly under the shared Plan 15 budget root")
    setup_torch(cfg)
    started, stopped = time.monotonic(), False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    def check():
        if stopped or ((args.root/"STOP").exists() and not args.clear_stop):
            raise StopRequested("preflight stop requested")
        if args.max_wall_seconds and time.monotonic() - started >= args.max_wall_seconds:
            raise ResourceLimit("invocation time limit")
    prior = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        data = dataset_manifest(args.data_root, check)
        details, code = provenance(cfg, data), sources()
        details.update(protocol=PROTOCOL, sources=code, source_hash=content_hash(code))
        if not cfg.smoke:
            accepted_root = args.acceptance_root or args.budget_root/"continuous_shift_acceptance_v1"
            accepted = read_json(accepted_root/"units/acceptance/result.json")
            if accepted.get("source_hash") != details["source_hash"] or accepted.get("status") != "passed":
                raise ValueError("matching CPU/CUDA recovery acceptance required")
        identity = dict(protocol=PROTOCOL, allocation_family=ALLOCATION_FAMILY,
            role="smoke" if cfg.smoke else "production", schema_version=1,
            source_hash=details["source_hash"], config_hash=cfg.scientific_hash,
            data_hash=details["data_hash"])
        with budget_lock(args.budget_root):
            other = 0. if cfg.smoke else peer_seconds(args.budget_root, args.root)
            allowance = args.allocation_hours * 3600.
            external = (file_bytes(args.budget_root, exclude=args.root, check=check)
                        + file_bytes(args.data_root, check=check) + 64_000_000)
            preflight = time.monotonic() - started
            wall = None if args.max_wall_seconds is None else max(0., args.max_wall_seconds-preflight)
            with Store(args.root, identity, resume=args.resume, clear_stop=args.clear_stop,
                       max_bytes=cfg.max_artifact_bytes, external_bytes=external,
                       free_reserve_bytes=cfg.free_reserve_bytes, finalization_reserve_bytes=32_000_000,
                       max_wall_seconds=wall, max_active_seconds=max(0., allowance-other)) as store:
                store.control.restore_active_seconds(store.control.active_seconds + preflight)
                if stopped:
                    store.control.stop_requested = True
                budget = RuntimeBudget(store, other_seconds=other, allocation_seconds=allowance)
                study = None
                try:
                    if not (store.root/"provenance.json").exists():
                        store.write_json(None, "provenance.json", details)
                        store.write_json(None, "resolved_config.json", cfg.mapping())
                        store.write_json(None, "source_snapshot.json",
                                         {name: (ROOT/name).read_text() for name in code})
                    with store.control.signal_handlers():
                        study = Study(cfg, args, store, budget)
                        study.execute()
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
        for sig, handler in prior.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
