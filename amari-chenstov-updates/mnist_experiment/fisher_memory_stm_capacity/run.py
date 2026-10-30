"""Autonomous, resumable execution of Plan 15 Phase 12."""

import argparse
from contextlib import contextmanager
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
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, read_json, setup_torch, dataset_manifest, provenance, file_digest, file_bytes,
)
from mnist_experiment.fisher_memory_grid_registration import learner as grid_learner
from mnist_experiment import plan15_grid_subspace as subspace
from . import PROTOCOL, ALLOCATION_FAMILY
from .config import Config, CHARTS, NEW_CHARTS, FD, REPLAY, REFERENCE, INCUMBENT, ROOT, Recipe, content_hash, arm, frozen_recipe
from .runner import Runner, trajectory_id
from . import model, core, design


def sources():
    result = subspace.sources()
    for path in Path(__file__).parent.glob("*.py"):
        if path.name not in ("report.py", "acceptance.py"):
            result[str(path.relative_to(ROOT))] = file_digest(path)
    wrapper = ROOT/"mnist_experiment/plan15_stm_capacity.py"
    result[str(wrapper.relative_to(ROOT))] = file_digest(wrapper)
    return dict(sorted(result.items()))


def peer_seconds(budget_root, root):
    total = 0.
    for peer in budget_root.iterdir():
        if peer.resolve() == root.resolve() or not (peer/"identity.json").is_file():
            continue
        identity = read_json(peer/"identity.json")["identity"]
        if identity.get("allocation_family") != ALLOCATION_FAMILY or identity.get("role") == "smoke":
            continue
        value = read_json(peer/"runtime.json")["cumulative_active_seconds"]
        if (peer/"timing_lease.json").exists():
            value = max(value, read_json(peer/"timing_lease.json").get("charged_until", value))
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid Phase 12 peer compute ledger")
        total += value
    return total


def process_alive(root, pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        parts = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        if not any(m in parts for m in ("mnist_experiment.plan15_stm_capacity",
                    "mnist_experiment.fisher_memory_stm_capacity.run")):
            return False
        requested = Path(parts[parts.index("--root")+1])
        if not requested.is_absolute():
            requested = Path(f"/proc/{pid}/cwd").resolve()/requested
        return requested.resolve() == Path(root).resolve()
    except (OSError, ValueError, IndexError, UnicodeError):
        return False


@contextmanager
def bindings():
    previous = grid_learner.compress_fisher
    try:
        grid_learner.compress_fisher = subspace.compress_fisher
        with core.bindings():
            yield
    finally:
        grid_learner.compress_fisher = previous


class Study:
    def __init__(self, cfg, args, store, budget):
        from mnist_experiment.fisher_memory_grid_registration.data import MNISTSource
        self.cfg, self.args, self.store, self.budget = cfg, args, store, budget
        self.last_status = self.last_report = 0.
        self.new_units = 0
        self.status = dict(status="running", stage="12A acceptance", current=None,
            planned=46+8*cfg.confirmation_replicas, mandatory_development=46,
            optional_development=0, changes=[])
        if (store.root/"status.json").exists():
            self.status.update(read_json(store.root/"status.json"))
            self.status.update(status="running", reason=None)
        self.runner = Runner(cfg, store, MNISTSource(args.data_root), budget,
            on_progress=self.progress, stop_at=args.smoke_stop_at)
        self.records = {}
        for entry in store.iter_units(completed_only=True):
            if entry["metadata"].get("kind") == "grid_registration_trajectory":
                self.records[entry["unit_id"]] = self.runner.record(entry["unit_id"])

    def decision(self, name):
        unit = "decision-"+name
        return read_json(self.store.unit_path(unit)/"result.json") if self.store.is_complete(unit) else None

    def seal(self, name, result):
        previous = self.decision(name)
        if previous is not None:
            if previous != result:
                raise ValueError(f"sealed decision changed: {name}")
            return previous
        unit = "decision-"+name
        self.store.begin(unit, dict(kind="stm_capacity_decision", name=name, schema_version=1))
        self.store.write_json(unit, "result.json", result)
        self.store.complete(unit)
        return result

    def snapshot(self):
        result = self.decision("analysis")
        confirmed = [r for r in self.records.values() if r["partition"] == "confirmation"]
        complete = [i for i in range(self.cfg.confirmation_replicas)
                    if sum(r["replica"] == i for r in confirmed) == 8]
        if result is None and len(complete) >= 2:
            from .analysis import analyze
            result = analyze([r for r in confirmed if r["replica"] in complete],
                N=self.cfg.novel_count, partition="confirmation", replicas=complete,
                draws=min(200,self.cfg.bootstrap_samples),
                seed=self.cfg.seed_for("confirmation--grid",0,"interim_bootstrap"),
                delta=self.cfg.retention_margin,epsilon=self.cfg.matching_tolerance,
                check=self.budget.check)
            result.update(is_final=False, partial_confirmation=True)
        return dict(protocol=PROTOCOL, progress=self.status,
            configurations=[dict(name=c, stm_config=c, stm_parameters=model.COUNTS[c][1],
                                 ltm_parameters=model.COUNTS[c][0]) for c in CHARTS],
            development=design.table(self.records.values(), self.cfg),
            selection=self.decision("selection"), analysis=result,
            calibration={c:self.decision("numerics-"+c) for c in NEW_CHARTS},
            reference=REFERENCE, incumbent=INCUMBENT, changes=self.status["changes"])

    def progress(self, force=False, final=False, **updates):
        updates.pop("stage", None)
        self.status.update(updates)
        now = time.monotonic()
        if not force and now-self.last_status < 3:
            return
        counts = {p:sum(r["partition"] == p for r in self.records.values())
                  for p in ("development", "confirmation")}
        confirm = [r for r in self.records.values() if r["partition"] == "confirmation"]
        blocks = sum(sum(r["replica"] == i for r in confirm) == 8
                     for i in range(self.cfg.confirmation_replicas))
        self.status.update(protocol=PROTOCOL, pid=os.getpid(), process_verified=True,
            completed=len(self.records), completed_by_partition=counts, paired_blocks=blocks,
            completed_blocks=blocks, planned_blocks=self.cfg.confirmation_replicas,
            completed_pairs=sum(sum(r["replica"] == i and r["stm_config"] == c for r in confirm) == 2
                for i in range(self.cfg.confirmation_replicas) for c in CHARTS),
            planned_pairs=4*self.cfg.confirmation_replicas,
            mandatory_completed=min(46,counts["development"])+counts["confirmation"],
            mandatory_planned=46+8*self.cfg.confirmation_replicas,
            optional_completed=max(0,counts["development"]-46),
            optional_planned=self.status.get("optional_development",0),
            development_status="completed" if self.decision("selection") else "running",
            confirmation_status="completed" if self.decision("analysis") else "running" if confirm else "pending",
            confirmation_blocks_target=self.cfg.confirmation_replicas,
            active_seconds=self.budget.used, allocation_seconds=self.budget.allocation_seconds,
            heartbeat=utc_now(), storage=self.store.storage_status())
        if updates.get("novel_count") is not None:
            self.status["last_accepted_checkpoint"] = f"{updates.get('current')}; novel count {updates['novel_count']}"
        durations = [r["active_seconds"] for r in self.records.values()]
        if durations:
            self.status["forecast_seconds"] = 1.25*max(0,self.status["planned"]-len(self.records))*float(np.quantile(durations,.9))
        self.store.write_json(None, "status.json", self.status, finalizing=True)
        self.last_status = now
        if final:
            try:
                self.budget.check()
            except (StopRequested,ResourceLimit):
                return
        if (self.last_report == 0 or final or now-self.last_report >= 900):
            snapshot = self.snapshot()
            self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=True)
            from .report import publish
            publish(snapshot, self.args.notebook or self.store.root/"results.ipynb", self.store)
            self.last_report = now

    def job(self, chart, replica, recipe, partition="development"):
        family = partition+"--grid"
        unit = trajectory_id(chart, family, replica, recipe, self.cfg.novel_count)
        if unit in self.records:
            return self.records[unit]
        self.progress(current=unit)
        capture = partition == "development"
        audit = capture and (recipe.method == REPLAY and math.isclose(recipe.ridge, math.sqrt(10))
                            or chart == "all_r2" and recipe.method == FD)
        result = self.runner.run_chart(chart, family, replica, partition, recipe,
                                      capture=capture, audit=audit)
        self.records[unit] = result
        self.new_units += 1
        print(json.dumps(dict(event="phase12_trajectory_complete", unit=unit, chart=chart,
            method=recipe.method, replica=replica, partition=partition,
            novel_auc=result["novel_auc"], H=math.sqrt(result["base_auc"])-math.sqrt(result["frozen_base_mse"]),
            learning_seconds=result["learning_seconds"], valid=result["valid"])), flush=True)
        self.progress(current=None)
        if self.args.max_units and self.new_units >= self.args.max_units:
            self.progress(force=True)
            raise StopRequested("completed-trajectory invocation limit")
        return result

    def recipe(self, method, **kwargs):
        return Recipe.from_config(self.cfg, method, **kwargs)

    def calibrate(self):
        if self.decision("calibration"):
            return
        self.status["stage"] = "12B shared initialization and replay pilots"
        for r in range(self.cfg.calibration_replicas):
            for method in (REPLAY, FD):
                self.job("all_r2", r, frozen_recipe(self.cfg, method))
            for chart in NEW_CHARTS:
                self.job(chart, r, self.recipe(REPLAY, ridge=math.sqrt(10)))
        from mnist_experiment.fisher_memory_grid_registration.numerical import calibrate_fd
        self.status["stage"] = "12B FD values, gradients and strength calibration"
        for chart in NEW_CHARTS:
            if self.decision("numerics-"+chart):
                continue
            snapshots = []
            for r in range(self.cfg.calibration_replicas):
                recipe = self.recipe(REPLAY, ridge=math.sqrt(10))
                saved = self.runner.captured(chart, "development--grid", r, recipe)
                snapshots += [s for s in saved if (s["n"] in (25,75) or self.cfg.smoke) and len(s["replay_y"])]
            with model.use_chart(chart):
                _, _, _, fitted, layout = self.runner.assets("development--grid", 0, "development")
                audit = calibrate_fd(self.cfg, snapshots, fitted, layout, self.budget.check)
            ratios = [r for r in audit["gradient_ratios"] if r["status"] == "valid"]
            a = float(np.median([r["data_gradient_norm"] for r in ratios]))
            b = float(np.median([r["fd_gradient_norm"] for r in ratios]))
            if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
                raise ValueError("unresolved FD gradient calibration")
            self.seal("numerics-"+chart, dict(stm_config=chart, lambda0=.1*a/b,
                selected_step=audit["selected_step"], numerical=audit,
                data_gradient_norm_median=a, fd_gradient_norm_median=b))
            self.progress(force=True)
        self.seal("calibration", dict(status="completed", replicas=list(range(self.cfg.calibration_replicas))))
        self.progress(force=True)

    def screen(self):
        if self.decision("selection"):
            return
        self.calibrate()
        self.status["stage"] = "12C bounded ridge and strength screen"
        for replica in range(self.cfg.calibration_replicas):
            for chart in NEW_CHARTS:
                numerical = self.decision("numerics-"+chart)
                for ridge in self.cfg.ridge_grid:
                    self.job(chart, replica, self.recipe(REPLAY, ridge=ridge))
                for ratio in self.cfg.strength_ratios:
                    self.job(chart, replica, self.recipe(FD, ridge=1., strength=numerical["lambda0"]*ratio,
                                                       fd_step=numerical["selected_step"]))
        selected = design.select(design.table(self.records.values(), self.cfg), self.cfg)
        second = self.decision("second-ridge")
        if second is None:
            second = {}
            for chart in NEW_CHARTS:
                fd = selected[arm(chart, FD)]["selected"]["recipe"]
                replay = selected[arm(chart, REPLAY)]["selected"]["recipe"]
                second[chart] = {**fd, "ridge": 10. if replay["ridge"] == 10. else math.sqrt(10)}
            self.seal("second-ridge", second)
        for replica in range(self.cfg.calibration_replicas):
            for chart, recipe in second.items():
                self.job(chart, replica, Recipe.from_mapping(recipe))
        self.status["stage"] = "12C bounded optional refinements"
        for index in range(self.cfg.refinement_settings):
            name = f"refinement-{index:02d}"
            proposal = self.decision(name)
            if proposal is None:
                if not self.optional_fits(2):
                    break
                rows = design.table(self.records.values(), self.cfg)
                choices = design.select(rows, self.cfg)
                chart = NEW_CHARTS[index % len(NEW_CHARTS)]
                used = {r["recipe_key"] for r in rows if r["stm_config"] == chart}
                recipe = design.refinement(rows, choices, chart, used)
                if recipe is None:
                    break
                proposal = self.seal(name, dict(stm_config=chart, recipe=recipe,
                    active_seconds=self.budget.used, reason="bounded observed frontier refinement"))
            for replica in range(self.cfg.calibration_replicas):
                self.job(proposal["stm_config"], replica, Recipe.from_mapping(proposal["recipe"]))
            self.status["optional_development"] = 2*(index+1)
            self.status["planned"] = 46+2*(index+1)+8*self.cfg.confirmation_replicas
            self.progress(force=True)
        selected = design.select(design.table(self.records.values(), self.cfg), self.cfg)
        self.status["stage"] = "12C selected-policy numerical audits"
        self.audit_selected(selected)
        self.seal("selection", selected)
        self.progress(force=True)

    def optional_fits(self, trajectories):
        if self.cfg.smoke:
            return False
        rows = list(self.records.values())
        cost = max([r["active_seconds"] for r in rows] or [600.])
        reserve = 1.25*(8*self.cfg.confirmation_replicas*cost+1800.)
        return self.budget.remaining >= reserve + trajectories*cost

    def audit_selected(self, selected):
        from .numerical import audit_policy
        for chart in CHARTS:
            for method in (REPLAY, FD):
                name = "policy-audit-"+arm(chart, method)
                if self.decision(name):
                    continue
                recipe = Recipe.from_mapping(selected[arm(chart, method)]["selected"]["recipe"])
                snapshots = self.runner.captured(chart, "development--grid", 0, recipe)
                snapshots = [s for s in snapshots if (s["n"] in (25,75) or self.cfg.smoke)
                             and len(s["replay_y"])]
                with model.use_chart(chart):
                    _, _, data, fitted, layout = self.runner.assets("development--grid", 0, "development")
                    result = audit_policy(self.cfg, recipe, snapshots, fitted, layout, data, self.budget.check)
                self.seal(name, result)
                if not result["passed"] and not self.cfg.smoke:
                    raise ValueError(f"selected-policy numerical audit failed: {name}")
                self.progress(force=True)

    def confirmation(self):
        self.screen()
        selected = self.decision("selection")
        manifest = self.decision("confirmation-manifest")
        if manifest is None:
            self.seal("confirmation-manifest", dict(replicas=list(range(self.cfg.confirmation_replicas)),
                recipes={a:v["selected"]["recipe"] for a,v in selected.items()},
                reference=REFERENCE, incumbent=INCUMBENT,
                source_hash=self.store.identity["source_hash"] if hasattr(self.store,"identity") else content_hash(sources()),
                config_hash=self.cfg.scientific_hash, N=self.cfg.novel_count))
        self.status["stage"] = "12D fresh paired confirmation"
        arms = [(c,m) for c in CHARTS for m in (REPLAY,FD)]
        for r in range(self.cfg.confirmation_replicas):
            rotated = arms[r % len(arms):] + arms[:r % len(arms)]
            for chart, method in rotated:
                self.job(chart, r, Recipe.from_mapping(selected[arm(chart,method)]["selected"]["recipe"]), "confirmation")

    def analyze(self):
        self.confirmation()
        if self.decision("analysis"):
            return
        from .analysis import analyze
        self.status["stage"] = "12E sample efficiency and uncertainty"
        manifest = self.decision("confirmation-manifest")
        records = [r for r in self.records.values() if r["partition"] == "confirmation"]
        result = analyze(records, N=self.cfg.novel_count, partition="confirmation",
            replicas=manifest["replicas"], draws=self.cfg.bootstrap_samples,
            seed=self.cfg.seed_for("confirmation--grid",0,"bootstrap"),
            delta=self.cfg.retention_margin, epsilon=self.cfg.matching_tolerance, check=self.budget.check,
            load_bootstrap=lambda start:self.decision(f"bootstrap-{start:04d}"),
            save_bootstrap=lambda start,block:self.seal(f"bootstrap-{start:04d}",block))
        self.seal("analysis", result)
        self.progress(force=True)

    def execute(self):
        self.progress(force=True)
        if not self.decision("design"):
            lineage = {}
            if not self.cfg.smoke:
                parent = self.args.budget_root/"grid_registration_subspace_v2"
                for name in ("design","initialization","readiness","archive-rank","selection","analysis-grid","closeout"):
                    record = subspace.sealed(parent,"decision-"+name,self.budget.check)
                    lineage[name] = record["completion"]["seal_hash"]
                selected = read_json(parent/"units/decision-selection/result.json")["tasks"]["grid"]["policies"]
                for method in (FD,REPLAY):
                    if selected[method]["selected"]["recipe"] != frozen_recipe(self.cfg,method).mapping():
                        raise ValueError("historical anchor differs from its sealed Phase 11 policy")
            self.seal("design", dict(config=self.cfg.mapping(), source_hash=content_hash(sources()),
                authorized="2026-10-29; 20 active hours", reference=REFERENCE, incumbent=INCUMBENT,
                parent_root="grid_registration_subspace_v2", parent_seals=lineage))
        if self.args.stage == "calibrate":
            self.calibrate()
        elif self.args.stage == "screen":
            self.screen()
        elif self.args.stage == "confirm":
            self.confirmation()
        else:
            self.analyze()
            self.seal("closeout", dict(status="completed", active_seconds=self.budget.used,
                trajectories=len(self.records), storage=self.store.storage_status()))
            self.status.update(status="completed", stage="12E completed", current=None)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, default=ROOT/"cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT/"cache/mnist_experiment/datasets")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--acceptance-root", type=Path)
    parser.add_argument("--stage", choices=("all","calibrate","screen","confirm","analyze","status","stop","report"), default="all")
    parser.add_argument("--device", choices=("cpu","cuda"))
    parser.add_argument("--dtype", choices=("float32","float64"))
    parser.add_argument("--allocation-hours", type=float, default=20.)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-stop-at", choices=("pretrain","accepted","mid_solve","eviction"))
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.stage in ("status","stop","report"):
        if args.stage == "stop":
            (args.root/"STOP").touch(exist_ok=True)
            print("Cooperative stop requested; resume preserves accepted checkpoints.")
        elif args.stage == "status":
            status = read_json(args.root/"status.json")
            print(json.dumps({**status,"process_alive":process_alive(args.root,status.get("pid"))},indent=2))
        else:
            from .report import publish
            snapshot = read_json(args.root/"report_snapshot.json")
            snapshot["progress"]["process_verified"] = process_alive(args.root,snapshot["progress"].get("pid"))
            publish(snapshot,args.notebook or args.root/"results.ipynb")
        return 0
    cfg = Config.read(args.config) if args.config else Config.smoke_config(args.device or "cpu") if args.smoke else Config()
    if args.device:
        cfg = replace(cfg,device=args.device)
    if args.dtype:
        cfg = replace(cfg,dtype=args.dtype)
    cfg.validate()
    if not 0 < args.allocation_hours <= 20 or args.max_units is not None and args.max_units <= 0:
        raise ValueError("invalid cumulative allowance/unit count")
    if not cfg.smoke and cfg.device != "cuda" or args.smoke_stop_at and not cfg.smoke:
        raise ValueError("production requires CUDA; artificial interruption is smoke-only")
    if args.root.resolve().parent != args.budget_root.resolve():
        raise ValueError("run root must be directly inside shared Plan 15 budget root")
    setup_torch(cfg)
    started, stopped = time.monotonic(), False
    def stop(signum,frame):
        nonlocal stopped
        stopped = True
    def check():
        if stopped or (args.root/"STOP").exists() and not args.clear_stop:
            raise StopRequested("preflight stop requested")
    prior = {sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        data, code = dataset_manifest(args.data_root,check), sources()
        details = provenance(cfg,data)
        details.update(protocol=PROTOCOL, sources=code, source_hash=content_hash(code))
        if not cfg.smoke:
            accepted = read_json((args.acceptance_root or args.budget_root/"stm_capacity_acceptance_v1")/"units/acceptance/result.json")
            if accepted.get("status") != "passed" or accepted.get("source_hash") != details["source_hash"]:
                raise ValueError("matching CPU/CUDA acceptance required before production")
        identity = dict(protocol=PROTOCOL, allocation_family=ALLOCATION_FAMILY,
            role="smoke" if cfg.smoke else "production", schema_version=1,
            source_hash=details["source_hash"], config_hash=cfg.scientific_hash, data_hash=details["data_hash"])
        with budget_lock(args.budget_root):
            other = 0. if cfg.smoke else peer_seconds(args.budget_root,args.root)
            external = file_bytes(args.budget_root,exclude=args.root,check=check)+file_bytes(args.data_root,check=check)+64_000_000
            with Store(args.root,identity,resume=args.resume,clear_stop=args.clear_stop,
                    max_bytes=cfg.max_artifact_bytes,external_bytes=external,
                    free_reserve_bytes=cfg.free_reserve_bytes,finalization_reserve_bytes=32_000_000,
                    max_wall_seconds=args.max_wall_seconds,max_active_seconds=max(0.,args.allocation_hours*3600-other)) as store:
                store.control.restore_active_seconds(store.control.active_seconds+time.monotonic()-started)
                budget = RuntimeBudget(store,other_seconds=other,allocation_seconds=args.allocation_hours*3600)
                study = None
                try:
                    if not (args.root/"provenance.json").exists():
                        store.write_json(None,"provenance.json",details)
                        store.write_json(None,"resolved_config.json",cfg.mapping())
                        store.write_json(None,"source_snapshot.json",{n:(ROOT/n).read_text() for n in code})
                    with store.control.signal_handlers(), bindings():
                        study = Study(cfg,args,store,budget)
                        study.execute()
                except (StopRequested,LearnerStop,ResourceLimit) as exc:
                    if study:
                        study.status.update(status="paused",reason=str(exc))
                    print(json.dumps(dict(event="paused",reason=str(exc))),flush=True)
                    return 2
                except BaseException as exc:
                    if study:
                        study.status.update(status="failed",reason=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    try:
                        if study:
                            study.progress(force=True,final=True)
                    finally:
                        budget.close()
        return 0
    finally:
        for sig,handler in prior.items():
            signal.signal(sig,handler)


if __name__ == "__main__":
    raise SystemExit(main())

