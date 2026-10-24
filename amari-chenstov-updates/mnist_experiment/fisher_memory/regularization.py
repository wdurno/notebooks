"""Resumable Plan 15 Phases 7A--7D, sharing one cumulative compute allowance."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import torch

from .artifacts import Store, iter_units
from .config import ROOT, Config, Treatment, content_hash, provenance
from .control import ResourceLimit, StopRequested
from .data import MNISTSource
from .drip import Progress
from .drip_runner import DripRunner, trajectory_id
from .learner import StopRequested as LearnerStop
from .regularization_audit import FrozenObjectiveAudit
from .regularization_design import (
    PROTOCOL, DEV_FAMILY, CONF_FAMILY, analyze, build_table, confirmation_ledger,
    decision_id, extension, ledger as make_ledger, select, shortlist, treatment,
)
from .run import budget_lock, dataset_manifest, external_bytes


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "mnist_experiment/fisher_memory/configs/default.json")
    parser.add_argument("--stage", choices=("all", "development", "confirmation"), default="all")
    parser.add_argument("--root", type=Path, required=True,
                        help="development root for all/development; confirmation root for confirmation")
    parser.add_argument("--confirmation-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets")
    parser.add_argument("--notebook", type=Path, default=ROOT / "mnist_experiment/fisher_memory_results.ipynb")
    parser.add_argument("--allocation-hours", type=float, required=True,
                        help="aggregate cumulative Phase 7 allowance, including prior invocations/roots")
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int, help="maximum newly completed training trajectories this invocation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--smoke-stop-after-step", type=int)
    parser.add_argument("--smoke-stop-before-audit", action="store_true")
    return parser.parse_args(argv)


def read_json(path):
    return json.loads(Path(path).read_text())


def aggregate_other_seconds(budget, root):
    """All Phase 7 roots in this budget count, including superseded repairs."""
    total = 0.
    for peer in Path(budget).iterdir():
        if peer.resolve() == Path(root).resolve() or not peer.is_dir():
            continue
        identity, runtime = peer / "identity.json", peer / "runtime.json"
        if identity.exists() and read_json(identity).get("identity", {}).get("protocol") == PROTOCOL:
            if runtime.exists():
                value = read_json(runtime)["cumulative_active_seconds"]
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid peer Phase 7 compute ledger")
                total += value
    return total


def _records(store, family, recipes, replicas):
    rows = []
    for replica in range(replicas):
        for stream in ("mixed", "base"):
            for recipe in recipes:
                unit = trajectory_id(family, replica, stream, recipe)
                if store.is_complete(unit):
                    path = store.unit_path(unit)
                    rows.append(dict(**read_json(path / "summary.json"),
                                     rows=read_json(path / "metrics.json"),
                                     metadata=read_json(path / "metadata.json")))
    return rows


class Study:
    def __init__(self, cfg, args, store, settings, other_seconds, invocation):
        from .regularization_report import build_notebook

        self.cfg, self.args, self.store, self.settings = cfg, args, store, settings
        self.other_seconds, self.invocation = other_seconds, invocation
        self.progress = Progress(store, args, settings, builder=build_notebook)
        self.progress.state.update(protocol=PROTOCOL,
            aggregate_allocation_seconds=3600 * args.allocation_hours,
            other_root_seconds=other_seconds, development_root=str(args.source_root) if args.source_root else None,
            forecast_hours=None, expected=None, completed=0)
        def after_commit(unit, state):
            job = self.progress.state.get("current_job")
            if job is not None:
                job.update(accepted_step=state["step"], paired_events=state["paired_events"])
            self.update()
            if args.smoke_stop_after_step and state["step"] >= args.smoke_stop_after_step:
                raise StopRequested("smoke accepted-step interruption")

        self.runner = DripRunner(cfg, store, MNISTSource(args.data_root),
                                 on_progress=self.update, after_commit=after_commit,
                                 phase="7", protocol=PROTOCOL)

    def update(self, *, force=False):
        self.progress.state["root_active_seconds"] = self.store.control.active_seconds
        # Progress publishes the root counter; retain the aggregate explicitly too.
        self.progress.state["aggregate_active_seconds"] = self.other_seconds + self.store.control.active_seconds
        self.progress(force=force)

    def decision(self, name):
        unit = decision_id(name)
        return read_json(self.store.unit_path(unit) / "result.json") if self.store.is_complete(unit) else None

    def seal(self, name, result):
        existing = self.decision(name)
        if existing is not None:
            if existing != result:
                raise ValueError(f"sealed Phase 7 decision changed: {name}")
            return existing
        unit = decision_id(name)
        self.store.begin(unit, dict(schema_version=1, kind="regularization_decision",
                                   protocol=PROTOCOL, decision=name))
        self.store.write_json(unit, "result.json", result)
        self.store.complete(unit)
        return result

    def run_jobs(self, recipes, stage, *, audit=False):
        cfg, store, settings = self.cfg, self.store, self.settings
        self.progress.state.update(stage=stage, status="running", current_job=None)
        replicas = settings["confirmation_replicas"] if stage == "7D" else settings["development_replicas"]
        family = CONF_FAMILY if stage == "7D" else DEV_FAMILY
        self.runner.audit = FrozenObjectiveAudit(cfg, store, settings["audit_events"],
                                                 on_progress=self.update) if audit else None
        if audit and self.args.smoke_stop_before_audit:
            def stop_before_branch(unit, snapshot):
                raise StopRequested("smoke pause with accepted parent and pending audit")
            self.runner.audit._run = stop_before_branch
        jobs = [(replica, stream, recipe) for replica in range(replicas)
                for stream in ("mixed", "base") for recipe in recipes]
        self.progress.state.update(expected=len(jobs), completed=sum(
            store.is_complete(trajectory_id(family, replica, stream, recipe))
            for replica, stream, recipe in jobs))
        self.update(force=True)
        for replica, stream, recipe in jobs:
            unit = trajectory_id(family, replica, stream, recipe)
            if store.is_complete(unit):
                continue
            store.check()
            self.progress.state["current_job"] = dict(replica=replica, stream=stream,
                condition=recipe.condition, strength=recipe.strength,
                ridge_multiplier=recipe.ridge_multiplier, policy_iterations=recipe.inner_steps)
            self.update()
            summary = self.runner.run_trajectory(family, replica, stream, recipe)
            self.invocation["completed"] += 1
            self.progress.state["completed"] += 1
            self.progress.state["current_job"] = None
            print(json.dumps(dict(event="regularization_trajectory_complete", stage=stage,
                unit_id=unit, ridge=recipe.ridge_multiplier, strength=recipe.strength,
                seconds=summary["active_seconds"], rotated_auc=summary["rotated_auc"],
                base_auc=summary["base_auc"])), flush=True)
            self.update(force=True)
            if self.args.max_units and self.invocation["completed"] >= self.args.max_units:
                raise StopRequested("requested trajectory batch complete")

    def recipes(self, policy, ridges=None, strengths=None):
        recipes = []
        for ridge in self.settings["ridge_grid"] if ridges is None else ridges:
            if strengths is None:
                recipes.append(treatment(self.cfg, "replay_ewc", ridge, 0., policy))
                recipes.append(treatment(self.cfg, "finite_difference", ridge, self.settings["lambda0"], policy))
            else:
                recipes.extend(treatment(self.cfg, "finite_difference", ridge, strength, policy)
                               for strength in strengths)
        return recipes

    def audit_summary(self, policy):
        rows = []
        for record in self.store.iter_units(completed_only=True):
            meta = record["metadata"]
            if (meta.get("kind") == "regularization_audit"
                    and meta["policy_iterations"] == policy["inner_steps"]):
                row = read_json(record["path"] / "result.json")
                rows.append({"unit_id": record["unit_id"], **meta, **row})
        expected = self.settings["development_replicas"] * 4 * len(self.settings["audit_events"])
        if len(rows) != expected:
            raise ValueError(f"audit cohort incomplete: {len(rows)}/{expected}")
        material = sum(row["relative_objective_gain"] > .001 for row in rows)
        return dict(schema_version=1, policy=policy, snapshots=rows,
                    completed=len(rows), expected=expected, material_gaps=material,
                    repair_needed=material >= self.settings["audit_repair_threshold_count"])

    def table(self, recipes, policy):
        reference = treatment(self.cfg, "replay_ewc", .1, 0., policy)
        included = list({row.key: row for row in [reference, *recipes]}.values())
        return build_table(_records(self.store, DEV_FAMILY, included,
                                    self.settings["development_replicas"]),
                           recipes, reference, self.settings["development_replicas"])

    def forecast(self, policy):
        records = _records(self.store, DEV_FAMILY, self.recipes(policy, [.1]),
                           self.settings["development_replicas"])
        times = {method: sum(row["active_seconds"] for row in records if row["condition"] == method)
                 / sum(row["condition"] == method for row in records)
                 for method in ("replay_ewc", "finite_difference")}
        # Bounds use mandatory and maximum prospective jobs, before outcome-based choices.
        low = 1.15 * (48 * times["replay_ewc"] + 64 * times["finite_difference"])
        high = 1.20 * (72 * times["replay_ewc"] + 96 * times["finite_difference"])
        self.progress.state["forecast_hours"] = [low / 3600, high / 3600]
        self.seal("timing", dict(schema_version=1, online_seconds=times,
                                 remaining_hours=[low / 3600, high / 3600]))
        self.update(force=True)
        if not self.cfg.smoke and low > (self.store.control.max_active_seconds
                                         - self.store.control.active_seconds):
            raise ResourceLimit("measured minimum Phase 7 forecast exceeds remaining aggregate allowance")

    def development(self):
        settings = self.settings
        chosen_policy = self.decision("policy")
        if chosen_policy is None:
            policy = settings["default_policy"]
            self.run_jobs(self.recipes(policy, [.1]), "7A", audit=True)
            initial = self.decision("audit-initial") or self.seal("audit-initial", self.audit_summary(policy))
            if initial["repair_needed"]:
                policy = settings["repair_policy"]
                self.progress.state["changes"] = [{
                    "reason": "predeclared objective-gap threshold triggered the common numerical repair",
                    "before": settings["default_policy"], "after": policy,
                    "affected": "initial 7A trajectories are diagnostic; grid uses repaired policy"}]
                self.run_jobs(self.recipes(policy, [.1]), "7A", audit=True)
                final = self.decision("audit-repaired") or self.seal("audit-repaired", self.audit_summary(policy))
            else:
                final = initial
            chosen_policy = self.seal("policy", dict(schema_version=1, policy=policy,
                repaired=initial["repair_needed"], optimization_limited=final["repair_needed"],
                material_gaps=final["material_gaps"], audit_count=final["completed"]))
        policy = chosen_policy["policy"]
        if self.decision("timing") is None:
            self.forecast(policy)
        else:
            self.progress.state["forecast_hours"] = self.decision("timing")["remaining_hours"]
        recipes = self.recipes(policy)
        self.run_jobs(recipes, "7B")
        short = self.decision("shortlist")
        if short is None:
            table = self.table(recipes, policy)
            short = self.seal("shortlist", dict(schema_version=1,
                **shortlist(table), table=table, policy=policy))
        candidates = {recipe.key: recipe for recipe in recipes if recipe.condition == "finite_difference"}
        strengths = [settings["lambda0"] / 3, settings["lambda0"], 3 * settings["lambda0"]]
        for ridge in short["ridges"]:
            local = self.recipes(policy, [ridge], strengths)
            self.run_jobs(local, "7C")
            candidates.update({recipe.key: recipe for recipe in local})
            name = "extension-" + str(ridge).replace(".", "_")
            extend = self.decision(name)
            if extend is None:
                value = extension(self.table(local, policy), settings["lambda0"], settings["retention_margin"])
                extend = self.seal(name, dict(schema_version=1, ridge=ridge, strength=value))
            if extend["strength"] is not None:
                extra = self.recipes(policy, [ridge], [extend["strength"]])
                self.run_jobs(extra, "7C")
                candidates.update({recipe.key: recipe for recipe in extra})
        selection = self.decision("selection")
        if selection is None:
            table = self.table(list(candidates.values()), policy)
            row, status = select(table, settings["retention_margin"])
            selection = self.seal("selection", dict(schema_version=1,
                protocol=PROTOCOL, replay=short["replay"], fd=row["treatment"],
                policy=policy, status=status, table=table,
                optimization_limited=chosen_policy["optimization_limited"]))
        self.progress.state.update(stage="7C", status="completed", current_job=None,
                                   selection=selection, completed=self.progress.state["expected"])
        self.update(force=True)
        return selection

    def confirmation(self):
        ledger = self.settings
        recipes = [Treatment(**row) for row in ledger["treatments"]]
        self.run_jobs(recipes, "7D")
        records = _records(self.store, CONF_FAMILY, recipes, ledger["replicas"])
        result = analyze(records, ledger)
        if not result["complete"]:
            raise ValueError("fixed confirmation cohort is incomplete")
        self.seal("results", result)
        self.progress.state.update(stage="7D", status="completed", current_job=None)
        self.update(force=True)
        return result


def _dependency(root, cfg, details):
    root = Path(root)
    identity = read_json(root / "identity.json")["identity"]
    if (identity["protocol"] != PROTOCOL or identity["config_hash"] != cfg.scientific_hash
            or identity["source_hash"] != details["source_hash"]
            or identity["data_hash"] != content_hash(details["dataset_manifest"])):
        raise ValueError("Phase 7 development scientific/data identity differs")
    # The small selection and ledger seals are dependencies; completed trajectories
    # are independently verified when read, rather than replayed on every resume.
    records = {record["unit_id"]: record for record in iter_units(root)}
    selection = records.get(decision_id("selection"))
    if selection is None or selection["status"] != "completed":
        raise ValueError("development has no sealed final selection")
    if any(row["status"] != "completed" for row in records.values()):
        raise ValueError("development still has incomplete units")
    chosen = read_json(selection["path"] / "result.json")
    source = dict(root=str(root.resolve()), identity=identity,
                  selection_seal=selection["completion"]["seal_hash"])
    return chosen, source


def main(argv=None):
    args = arguments(argv)
    cfg = Config.read(args.config)
    if not math.isfinite(args.allocation_hours) or args.allocation_hours <= 0:
        raise ValueError("positive finite aggregate compute allocation is required")
    if args.clear_stop and not args.resume:
        raise ValueError("clearing STOP requires --resume")
    if args.max_units is not None and args.max_units < 1:
        raise ValueError("max-units must be positive")
    if args.max_wall_seconds is not None and (
            not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0):
        raise ValueError("max-wall-seconds must be finite and positive")
    if args.smoke_stop_after_step is not None and args.smoke_stop_after_step < 1:
        raise ValueError("smoke-stop-after-step must be positive")
    if (args.smoke_stop_after_step or args.smoke_stop_before_audit) and not cfg.smoke:
        raise ValueError("smoke interruption hooks cannot change a production run")
    if args.stage == "confirmation" and args.source_root is None:
        raise ValueError("confirmation requires --source-root")
    cfg = dataclasses.replace(cfg, compute_allocation_seconds=3600 * args.allocation_hours)
    cfg.validate()
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing silent CPU substitution")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan15-matplotlib")
    args.root, args.budget_root = args.root.resolve(), args.budget_root.resolve()
    development = args.root if args.stage != "confirmation" else args.source_root.resolve()
    confirmation = (args.root if args.stage == "confirmation" else args.confirmation_root or
                    args.root.with_name(args.root.name.replace("development", "confirmation")))
    confirmation = confirmation.resolve()
    if args.stage == "all" and development == confirmation:
        raise ValueError("--confirmation-root must differ from development root")
    for root in ([development, confirmation] if args.stage == "all" else [args.root]):
        if root == args.budget_root or not root.is_relative_to(args.budget_root):
            raise ValueError("each root must be a distinct child of the shared budget root")
        if root.exists() and not args.resume:
            raise ValueError(f"Phase 7 root exists; use --resume or a fresh root: {root}")
    details = provenance(cfg)
    details.update(protocol=PROTOCOL, dataset_manifest=dataset_manifest(args.data_root))
    invocation = dict(completed=0, started=time.monotonic())
    stages = ["development", "confirmation"] if args.stage == "all" else [args.stage]
    with budget_lock(args.budget_root):
        for stage in stages:
            root = development if stage == "development" else confirmation
            args.root = root
            args.source_root = development if stage == "confirmation" else None
            settings = make_ledger(cfg)
            if stage == "confirmation":
                selected, source = _dependency(development, cfg, details)
                settings.update(confirmation_ledger(cfg, selected, source))
            identity = dict(schema_version=1, protocol=PROTOCOL, stage=stage,
                config_hash=cfg.scientific_hash, source_hash=details["source_hash"],
                data_hash=content_hash(details["dataset_manifest"]), ledger_hash=content_hash(settings))
            other = aggregate_other_seconds(args.budget_root, root)
            remaining = cfg.compute_allocation_seconds - other
            if remaining <= 0:
                print(json.dumps(dict(event="paused", reason="aggregate Phase 7 allocation exhausted")), flush=True)
                return "paused"
            wall = None if args.max_wall_seconds is None else max(
                0., args.max_wall_seconds - (time.monotonic() - invocation["started"]))
            existed = (root / "identity.json").exists()
            with Store(root, identity, resume=args.resume, max_bytes=cfg.max_artifact_bytes,
                external_bytes=external_bytes(args), free_reserve_bytes=cfg.free_reserve_bytes,
                finalization_reserve_bytes=32_000_000, max_wall_seconds=wall,
                max_active_seconds=remaining, clear_stop=args.clear_stop and existed) as store:
                if not (root / "provenance.json").exists():
                    store.write_json(None, "provenance.json", details)
                    store.write_json(None, "resolved_config.json", cfg.mapping())
                study = Study(cfg, args, store, settings, other, invocation)
                study.seal("confirmation-ledger" if stage == "confirmation" else "ledger", settings)
                with store.control.signal_handlers():
                    try:
                        if stage == "development":
                            study.development()
                        else:
                            study.confirmation()
                    except (StopRequested, LearnerStop, ResourceLimit) as error:
                        study.progress.state.update(status="paused", stop_reason=str(error), current_job=None)
                        print(json.dumps(dict(event="paused", reason=str(error))), flush=True)
                        study.update(force=True)
                        return "paused"
                    except BaseException as error:
                        study.progress.state.update(status="failed", failure=f"{type(error).__name__}: {error}")
                        study.update(force=True)
                        raise
                    finally:
                        store.heartbeat(force=True)
    return "completed"


if __name__ == "__main__":
    raise SystemExit(0 if main() == "completed" else 3)
