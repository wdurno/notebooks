"""Resumable historical replay, fresh STM replication and paired blur removal."""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import StopRequested, ResourceLimit, utc_now
from mnist_experiment.fisher_memory.learner import StopRequested as LearnerStop
from mnist_experiment.fisher_memory_grid_registration.data import MNISTSource
from mnist_experiment.fisher_memory_stm_capacity.run import bindings
from mnist_experiment.fisher_memory_stm_capacity.runner import trajectory_id
from mnist_experiment.fisher_memory_stm_capacity.config import content_hash
from mnist_experiment.fisher_memory_regression.runtime import (
    read_json, setup_torch, provenance, dataset_manifest, file_bytes,
)
from . import PROTOCOL
from .config import REPO, FAMILY, HISTORY, ROOT, DATA, POLICIES, SEED, BOOTSTRAP_SEED, INTERIM_SEED, TOLERANCES, configuration, recipe
from .runtime import authorize, atomic_json, sources, historical_sources, peer_seconds, RuntimeBudget, external_bytes
from .runner import Runner, NamespaceStore, record
from .audit import preflight, compare_paths, load_checkpoint


class Study:
    def __init__(self, args, store, budget, historical_config):
        self.args, self.store, self.budget = args, store, budget
        self.historical = configuration(historical_config, smoke=args.smoke, device=args.device)
        self.fresh = configuration(historical_config, fresh=True, smoke=args.smoke, device=args.device)
        self.source = MNISTSource(args.data_root)
        self.runners = {}
        self.last_report = 0.
        self.stage = "8A historical fidelity"
        self.state = "running"
        self.reason = None
        self.current = None
        self.new_paths = 0
        self.records = []
        self.fidelity = self.decision("fidelity") or dict(status="pending")
        self.restore_records()

    def restore_records(self):
        self.records = []
        for entry in self.store.iter_units(completed_only=True):
            if entry["metadata"].get("kind") != "grid_registration_trajectory":
                continue
            value = record(self.store, entry["unit_id"])
            chart, mapping = next((chart, r) for policy, (chart, r) in POLICIES.items()
                                  if chart == value["stm_config"] and r["method"] == value["method"])
            value["policy"] = next(p for p, (c, r) in POLICIES.items()
                                   if c == chart and r["method"] == mapping["method"])
            self.records.append(value)

    def decision(self, name):
        unit = "decision-" + name
        return read_json(self.store.unit_path(unit) / "result.json") if self.store.is_complete(unit) else None

    def seal(self, name, result):
        old = self.decision(name)
        if old is not None:
            if old != result:
                raise artifacts.ImmutableError(f"sealed decision changed: {name}")
            return old
        unit = "decision-" + name
        self.store.begin(unit, dict(kind="phase8_decision", name=name, protocol=PROTOCOL))
        self.store.write_json(unit, "result.json", result)
        self.store.complete(unit)
        return result

    def progress(self, force=False, **updates):
        self.budget.check()
        if "current" in updates:
            self.current = updates["current"]
        fresh = [r for r in self.records if r["namespace"] == "fresh"]
        counts = {c: sum(r["context"] == c for r in fresh) for c in ("blur", "no_blur")}
        replicas = range(self.args.replicas)
        complete = sum(sum(r["replica"] == i for r in fresh) == 5 for i in replicas)
        blurred_blocks = sum(sum(r["replica"] == i and r["context"] == "blur" for r in fresh) == 3
                             for i in replicas)
        counts.update(blurred=counts["blur"], fresh_blocks=blurred_blocks)
        status = dict(status=self.state, stage=self.stage, current=self.current, reason=self.reason,
            pid=os.getpid(), heartbeat=utc_now(), counts=counts, blurred_completed=counts["blur"],
            no_blur_completed=counts["no_blur"], fresh_blocks_completed=complete,
            replicas=self.args.replicas, planned_blurred=3*self.args.replicas,
            planned_no_blur=2*self.args.replicas, historical_completed=sum(r["namespace"] == "historical" for r in self.records),
            charged_hours=self.budget.used/3600, remaining_hours=self.budget.remaining/3600,
            forecast_hours=max(0.,5*self.args.replicas-len(fresh)) *
                max((r["active_seconds"] for r in self.records), default=600.)/3600 + .5,
            accepted_checkpoint=updates.get("novel_count"))
        self.store.write_json(None, "status.json", status, finalizing=True)
        if not force and time.monotonic() - self.last_report < 900:
            return
        analysis = self.decision("analysis")
        if analysis is None and self.args.stage != "fidelity":
            from .analysis import analyze
            if sum(r["context"] == "blur" for r in fresh) >= 6:
                analysis = analyze(fresh, draws=100, seed=INTERIM_SEED, partial=True,
                                   check=self.budget.check)
        snapshot = dict(protocol=PROTOCOL, progress=status, fidelity=self.fidelity,
            settings=[dict(policy=p, chart=c, **v) for p,(c,v) in POLICIES.items()],
            analysis=analysis, changes=self.decision("changes") or [])
        self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=True)
        from .report import publish
        publish(snapshot, self.args.notebook, self.store)
        self.last_report = time.monotonic()

    def runner(self, namespace, context):
        key = namespace, context
        if key not in self.runners:
            cfg = self.historical if namespace == "historical" else self.fresh
            wrapped = NamespaceStore(self.store, namespace, context)
            runner = Runner(cfg, wrapped, self.source, self.budget,
                            on_progress=self.progress, stop_at=self.args.smoke_stop_at)
            runner.blur_max = cfg.blur_max if context == "blur" else 0.
            self.runners[key] = runner
        return self.runners[key]

    def job(self, namespace, context, replica, policy):
        cfg = self.historical if namespace == "historical" else self.fresh
        chart = POLICIES[policy][0]
        logical = trajectory_id(chart, "confirmation--grid", replica, recipe(cfg, policy), cfg.novel_count)
        physical = NamespaceStore(self.store, namespace, context).key(logical)
        if any(r["physical_unit"] == physical for r in self.records):
            return next(r for r in self.records if r["physical_unit"] == physical)
        self.current = physical
        self.progress()
        runner = self.runner(namespace, context)
        with bindings():
            runner.run_chart(chart, "confirmation--grid", replica, "confirmation", recipe(cfg, policy))
        value = record(self.store, physical)
        value["policy"] = policy
        self.records.append(value)
        self.new_paths += 1
        print(json.dumps(dict(event="trajectory_completed", namespace=namespace, context=context,
              replica=replica, policy=policy, risk=value["novel_auc"], seconds=value["active_seconds"])), flush=True)
        if self.args.max_paths and self.new_paths >= self.args.max_paths:
            raise StopRequested("requested bounded invocation completed")
        return value

    def fidelity_checks(self):
        if self.decision("fidelity"):
            self.fidelity = self.decision("fidelity")
            if self.fidelity["status"] != "passed":
                raise ValueError("fidelity did not pass; fresh runs blocked")
            return
        self.stage = "8A historical fidelity"
        self.seal("acceptance-contract", dict(tolerances=TOLERANCES, historical_replica=0,
            fresh_seed=SEED, replicas=self.args.replicas, policies=list(POLICIES),
            smoke=self.args.smoke, historical_seed=self.historical.seed))
        if self.args.smoke:
            for p in POLICIES:
                self.job("historical", "blur", 0, p)
            self.fidelity = self.seal("fidelity", dict(status="passed", smoke_only=True))
            return
        audit = self.decision("historical-inputs")
        if audit is None:
            audit = preflight(self.args.history_root, self.args.data_root, self.historical, self.budget.check)
            self.seal("historical-inputs", audit)
        if not self.decision("historical-statistics"):
            from .analysis import historical_audit
            result = historical_audit(audit["records"], self.args.history_root,
                                      seed=self.historical.seed_for("confirmation--grid", 0, "bootstrap"),
                                      check=self.budget.check)
            self.seal("historical-statistics", result)
        checks = []
        for p in POLICIES:
            value = self.job("historical", "blur", 0, p)
            original = trajectory_id(POLICIES[p][0], "confirmation--grid", 0, recipe(self.historical,p),100)
            result = self.decision("parity-" + p)
            if result is None:
                result = compare_paths(self.args.history_root, original, self.store, value["physical_unit"])
                self.seal("parity-" + p, result)
            checks.append(result)
        self.fidelity = self.seal("fidelity", dict(status="passed", same_seed_paths=checks,
            source_files=audit["source_files"], statistics=self.decision("historical-statistics")))

    def reserve(self):
        old = self.decision("fresh-manifest")
        if old:
            return
        costs = [r["active_seconds"] for r in self.records if r["namespace"] == "historical"]
        initial = [self.store.load_checkpoint(e["unit_id"])["pretrain_seconds"]
                   for e in self.store.iter_units(completed_only=True)
                   if e["metadata"].get("kind") == "grid_registration_assets"]
        trajectory = max(costs, default=600.)
        initialization = max(initial, default=300.) + 120.
        required = 1.25 * self.args.replicas * (5*trajectory + initialization) + 1800.
        if not self.args.smoke and required > self.budget.remaining:
            raise ResourceLimit(f"fixed fresh allocation/reserve requires {required/3600:.2f} h; available {self.budget.remaining/3600:.2f} h")
        self.seal("fresh-manifest", dict(replicas=list(range(self.args.replicas)), seed=SEED,
                  policies={p:recipe(self.fresh,p).mapping() for p in POLICIES},
                  required_seconds=required, remaining_seconds=self.budget.remaining,
                  fixed_paths=5*self.args.replicas))

    def execute(self):
        self.progress(force=True)
        self.fidelity_checks()
        self.progress(force=True)
        if self.args.stage == "fidelity":
            self.state, self.stage = "fidelity_passed", "8A complete"
            return
        self.reserve()
        self.stage = "8B fresh statistical replication"
        for replica in range(self.args.replicas):
            policies = list(POLICIES)
            offset = replica % len(policies)
            for policy in policies[offset:] + policies[:offset]:
                self.job("fresh", "blur", replica, policy)
            self.progress(force=True)
        self.stage = "8C paired removal of blur"
        for replica in range(self.args.replicas):
            for policy in (("R1", "F1") if replica % 2 == 0 else ("F1", "R1")):
                self.job("fresh", "no_blur", replica, policy)
            self.progress(force=True)
        self.stage = "8D inference and final audit"
        if self.decision("analysis") is None:
            from .analysis import analyze
            result = analyze([r for r in self.records if r["namespace"] == "fresh"],
                             draws=100 if self.args.smoke else 2000, seed=BOOTSTRAP_SEED,
                             check=self.budget.check,
                             load_bootstrap=lambda start:self.decision(f"bootstrap-{start:04d}"),
                             save_bootstrap=lambda start,block:self.seal(f"bootstrap-{start:04d}",block))
            self.seal("analysis", result)
        self.integrity()
        self.state, self.stage, self.current = "completed", "Phase 8 complete", None
        self.progress(force=True)
        self.store.write_json(None, "study_completed.json", dict(status="completed", completed_at=utc_now(),
              trajectories=len(self.records), charged_seconds=self.budget.used,
              analysis_unit="decision-analysis", fidelity="passed"))

    def integrity(self):
        if self.decision("integrity"):
            return
        units = list(artifacts.iter_units(self.store.root, full=True, check=self.budget.check))
        if any(u["status"] != "completed" for u in units):
            raise ValueError("final audit found incomplete units")
        fresh = [r for r in self.records if r["namespace"] == "fresh"]
        for replica in range(self.args.replicas):
            group = [r for r in fresh if r["replica"] == replica]
            if len(group) != 5 or len({r["plan_hash"] for r in group}) != 1:
                raise ValueError("fresh pairing/allocation failed")
            if len({tuple(row["observations"] for row in r["metrics"]) for r in group}) != 1:
                raise ValueError("paired context exposure clock changed")
            for r in group:
                state = self.store.load_checkpoint(r["physical_unit"])
                retired = [x["exposure"] for x in state["retirements"]]
                if len(set(retired)) != len(retired):
                    raise ValueError("duplicate retirement")
        self.seal("integrity", dict(status="passed", checked_units=len(units),
            paired_blocks=self.args.replicas, sources=sources(self.args.history_root)))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--history-root", type=Path, default=HISTORY)
    p.add_argument("--data-root", type=Path, default=DATA)
    p.add_argument("--notebook", type=Path, default=REPO / "mnist_experiment/plan18_results.ipynb")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--stage", choices=("all", "fidelity", "status", "stop", "report"), default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--clear-stop", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--max-paths", type=int)
    p.add_argument("--smoke-stop-at", choices=("pretrain", "accepted", "mid_solve", "eviction"))
    return p.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.stage in ("status", "stop", "report"):
        if args.stage == "stop":
            (args.root / "STOP").touch()
        elif args.stage == "status":
            print(json.dumps(read_json(args.root / "status.json"), indent=2))
        else:
            from .report import publish
            publish(read_json(args.root / "report_snapshot.json"), args.notebook)
        return 0
    if args.replicas != 8 and not args.smoke or args.device != "cuda" and not args.smoke:
        raise ValueError("production requires CUDA and eight fresh paired replicas")
    if args.smoke_stop_at and not args.smoke:
        raise ValueError("artificial stops are smoke-only")
    approved = authorize()
    historical = read_json(args.history_root / "resolved_config.json")
    cfg = configuration(historical, smoke=args.smoke, device=args.device)
    setup_torch(cfg)
    code = sources(args.history_root)
    data_manifest = dataset_manifest(args.data_root)
    if data_manifest != historical_sources(args.history_root)["dataset_manifest"]:
        raise ValueError("historical dataset changed before run/resume")
    identity = dict(protocol=PROTOCOL, allocation_family="plan18", role="smoke" if args.smoke else "production",
        config_hash=content_hash(dict(config=cfg.mapping(), fresh_seed=SEED, replicas=args.replicas)),
        source_hash=content_hash(code), sources=code, data_hash=content_hash(data_manifest),
        historical_identity=artifacts._identity(args.history_root)["identity_hash"],
        authorization_sha256=approved["sha256"])
    if not args.smoke:
        accepted = read_json(FAMILY / "phase8_acceptance.json")
        if accepted.get("status") != "passed" or accepted.get("source_hash") != identity["source_hash"]:
            raise ValueError("matching CPU/CUDA/recovery acceptance is required before production")
    other = peer_seconds(args.root)
    limit = approved["payload"]["additional_seconds"]
    before = time.monotonic()
    with artifacts.Store(args.root, identity, resume=args.resume, clear_stop=args.clear_stop,
            external_bytes=external_bytes(args.root,args.data_root), free_reserve_bytes=20_000_000_000,
            finalization_reserve_bytes=64_000_000, max_active_seconds=max(0.,limit-other)) as store:
        store.control.restore_active_seconds(store.control.active_seconds + time.monotonic()-before)
        budget = RuntimeBudget(store, other_seconds=other, allocation_seconds=limit)
        study = None
        try:
            if not (args.root / "provenance.json").exists():
                detail = provenance(cfg, data_manifest)
                detail.update(protocol=PROTOCOL, sources=code, source_hash=identity["source_hash"],
                              authorization=approved, fresh_seed=SEED)
                store.write_json(None,"provenance.json",detail)
                store.write_json(None,"resolved_config.json",cfg.mapping())
            with store.control.signal_handlers():
                study = Study(args,store,budget,historical)
                study.execute()
        except (StopRequested,LearnerStop,ResourceLimit) as exc:
            if study:
                study.state, study.reason = "paused", str(exc)
            print(json.dumps(dict(status="paused",reason=str(exc))),flush=True)
            return 2
        except BaseException as exc:
            if study:
                study.state, study.reason = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if study:
                try:
                    study.progress(force=True)
                except (StopRequested,LearnerStop,ResourceLimit):
                    atomic_json(args.root / "status.json", dict(status=study.state,stage=study.stage,
                                reason=study.reason,pid=os.getpid(),charged_hours=budget.used/3600))
            budget.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
