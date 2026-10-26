"""Resumable Phase 8.1 search, retention matching, confirmation and reporting."""

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
from mnist_experiment.fisher_memory_regression.data import MNISTSource
from mnist_experiment.fisher_memory_regression.runner import Runner, trajectory_id
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, read_json, setup_torch, dataset_manifest, provenance,
    scientific_sources as engine_sources, file_digest, file_bytes,
)

from . import PROTOCOL, ALLOCATION_FAMILY
from .config import Config
from .design import initial_grid, table, refinement_proposals, shortlist, select


def sources():
    result=engine_sources()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name!="report.py": result[str(path.relative_to(ROOT))]=file_digest(path)
    return result


def peer_seconds(budget_root,root):
    total=0.
    for peer in budget_root.iterdir():
        identity=peer/"identity.json"
        if peer.resolve()==root.resolve() or not identity.is_file(): continue
        info=read_json(identity)["identity"]
        if info.get("allocation_family")!=ALLOCATION_FAMILY or info.get("role")=="smoke": continue
        runtime=peer/"runtime.json"
        if not runtime.exists(): continue
        value=read_json(runtime)["cumulative_active_seconds"]
        lease=peer/"timing_lease.json"
        if lease.exists(): value=max(value,read_json(lease).get("charged_until",value))
        if not math.isfinite(value) or value<0: raise ValueError("invalid peer compute ledger")
        total+=value
    return total


class Study:
    def __init__(self,cfg,args,store,budget):
        self.cfg,self.args,self.store,self.budget=cfg,args,store,budget
        self.last_status=self.last_report=0.
        self.new_units=0
        self.status=dict(status="running",stage="8.1A",current=None,completed=0,planned=0,
                         forecast_seconds=None,changes=[])
        if (store.root/"status.json").exists():
            self.status.update(read_json(store.root/"status.json"))
            self.status["status"]="running"
        self.runner=Runner(cfg,store,MNISTSource(args.data_root),budget,
                           on_progress=self.progress,stop_at=args.smoke_stop_at)
        self.records={}
        for item in store.iter_units(completed_only=True):
            if item["metadata"].get("kind")=="regression_trajectory":
                self.records[item["unit_id"]]=self.runner.record(item["unit_id"])

    def decision(self,name):
        unit="decision-"+name
        return read_json(self.store.unit_path(unit)/"result.json") if self.store.is_complete(unit) else None

    def seal(self,name,result):
        old=self.decision(name)
        if old is not None:
            if old!=result: raise ValueError(f"sealed decision changed: {name}")
            return old
        unit="decision-"+name
        self.store.begin(unit,dict(kind="retention_decision",schema_version=1,name=name))
        self.store.write_json(unit,"result.json",result)
        self.store.complete(unit)
        return result

    def development(self):
        return [r for r in self.records.values() if r["partition"]=="development"]

    def snapshot(self):
        initial=[r for r in self.development() if r["replica"]<self.cfg.calibration_replicas]
        return dict(schema_version=1,protocol=PROTOCOL,config=self.cfg.mapping(),progress=self.status,
            development=table(initial,self.cfg,partial=True),selection=self.decision("selection"),
            analysis=self.decision("analysis"),changes=self.status.get("changes",[]),
            confirmation=[r for r in self.records.values() if r["partition"]=="confirmation"],
            analysis_href=os.path.relpath(self.store.unit_path("decision-analysis")/"result.json",
                (self.args.notebook or self.store.root/"results.ipynb").parent))

    def progress(self,force=False,final=False,**updates):
        # Runner callbacks carry its family name; retain the phase-specific stage.
        updates.pop("stage",None)
        self.status.update(updates)
        now=time.monotonic()
        if not force and now-self.last_status<2: return
        self.status.update(schema_version=1,protocol=PROTOCOL,completed=len(self.records),
            active_seconds=self.budget.used,heartbeat=utc_now(),storage=self.store.storage_status())
        if len(self.records)>=3 and self.status["status"]!="completed":
            remaining=max(0,self.status.get("planned",0)-len(self.records))
            if self.status["stage"].startswith("8.1B"):
                remaining=max(remaining,len(initial_grid(self.cfg))*self.cfg.calibration_replicas
                    +self.cfg.refinement_settings*self.cfg.calibration_replicas
                    +6*self.cfg.refinement_replicas+3*self.cfg.confirmation_replicas-len(self.records))
            self.status["forecast_seconds"]=1.5*remaining*float(np.quantile(
                [r["active_seconds"] for r in self.records.values()],.9))
            self.status["forecast_basis"]="conservative complete-unit timing; includes all allowed refinement slots during grid"
        self.store.write_json(None,"status.json",self.status,finalizing=True)
        self.last_status=now
        if final:
            try: self.budget.check()
            except (StopRequested,ResourceLimit): return
        if force or now-self.last_report>=900:
            from .report import publish
            snapshot=self.snapshot()
            self.store.write_json(None,"report_snapshot.json",snapshot,finalizing=True)
            publish(snapshot,self.args.notebook or self.store.root/"results.ipynb",self.store)
            self.last_report=now

    def audited(self,replica,recipe):
        return (replica==0 and recipe.ridge in (min(self.cfg.ridge_grid),max(self.cfg.ridge_grid))
                and recipe.replay_count in (0,max(self.cfg.replay_counts))
                and (recipe.method!="fd_replay_ewc" or recipe.strength==self.cfg.historical_strength*max(self.cfg.strength_ratios)))

    def job(self,family,replica,recipe):
        unit=trajectory_id(family,replica,recipe,self.cfg.novel_count)
        if unit in self.records: return self.records[unit]
        self.budget.check();self.progress(current=unit)
        partition="confirmation" if family=="confirmation" else "development"
        audit=family=="development" and self.audited(replica,recipe)
        record=self.runner.run(family,replica,partition,recipe,audit=audit,capture=audit)
        self.records[unit]=record;self.new_units+=1
        print(json.dumps(dict(event="phase81_trajectory_complete",unit=unit,method=recipe.method,
            replica=replica,novel_auc=record["novel_auc"],base_harm=math.sqrt(record["base_auc"])-math.sqrt(record["frozen_base_mse"]),
            seconds=record["learning_seconds"],valid=record["valid"])),flush=True)
        self.progress(force=True,current=None)
        if self.args.max_units and self.new_units>=self.args.max_units:
            raise StopRequested("requested completed-unit batch reached")
        return record

    def register(self):
        self.seal("design",dict(protocol=PROTOCOL,schema_version=1,config=self.cfg.mapping(),
            initial_recipes=[r.mapping() for r in initial_grid(self.cfg)],
            delta=self.cfg.retention_margin_degrees,epsilon=self.cfg.matching_tolerance,
            source_hash=self.store.identity["source_hash"],created_from="Phase 8.1 approved 2026-10-25",
            budget_definition="100 unique online novel images after separate calibration; all background arrivals retained",
            selection="full-budget independent regularization; development retention band; no confirmation retuning"))

    def grid(self):
        if self.decision("grid-complete"): return
        recipes=initial_grid(self.cfg)
        self.status.update(stage="8.1B initial grid",planned=len(recipes)*self.cfg.calibration_replicas)
        for replica in range(self.cfg.calibration_replicas):
            for recipe in recipes: self.job("development",replica,recipe)
        rows=table(self.development(),self.cfg)
        audits=[]
        for row in self.development():
            audits.extend(read_json(self.store.unit_path(row["unit_id"])/"audits.json"))
        self.seal("grid-complete",dict(settings=len(rows),trajectories=len(self.development()),
            mean_unit_seconds=float(np.mean([r["active_seconds"] for r in self.development()])),
            solver_audits=audits,solver_failures=sum(r["solver_failures"] for r in self.development()),
            timing_limit=self.cfg.learning_time_limit))

    def refinements(self):
        if self.decision("refinements-complete"): return
        additions=[]
        for index in range(self.cfg.refinement_settings):
            name=f"refinement-{index:03d}"
            record=self.decision(name)
            if record is None:
                rows=table(self.development(),self.cfg)
                candidates=refinement_proposals(rows,self.cfg,additions)
                if not candidates: break
                durations=[r["active_seconds"] for r in self.development()]
                estimate=self.cfg.calibration_replicas*max(durations)*1.5
                reserve=self.cfg.confirmation_reserve_seconds+self.cfg.analysis_reserve_seconds
                if self.budget.remaining<estimate+reserve:
                    self.seal("refinement-timing-stop",dict(index=index,estimated_seconds=estimate,
                        reserve_seconds=reserve,reason="confirmation/analysis reserved before additional settings"))
                    break
                proposal=candidates[0]
                record=self.seal(name,dict(recipe=proposal["recipe"],kind=proposal["kind"],
                    reason="Prospective local retention refinement on completed four-replica means",
                    based_on_units=sorted(r["unit_id"] for r in self.development())))
            additions.append(record)
            recipe=Recipe.from_mapping(record["recipe"])
            self.status.update(stage="8.1B refinement",planned=len(self.records)+self.cfg.calibration_replicas)
            for replica in range(self.cfg.calibration_replicas): self.job("development",replica,recipe)
        self.seal("refinements-complete",dict(additions=additions,count=len(additions)))

    def finalists(self):
        if self.decision("selection"): return
        short=self.decision("shortlist")
        if short is None:
            short=self.seal("shortlist",shortlist(table(self.development(),self.cfg),self.cfg))
        recipes=[Recipe.from_mapping(r) for values in short.values() for r in values]
        count=self.cfg.calibration_replicas+self.cfg.refinement_replicas
        self.status.update(stage="8.1C finalist calibration",planned=len(self.records)+len(recipes)*self.cfg.refinement_replicas)
        for replica in range(count):
            for recipe in recipes: self.job("development",replica,recipe)
        keys={r.key for r in recipes}
        rows=table([r for r in self.development() if r["recipe_key"] in keys],self.cfg,replicas=range(count))
        self.seal("selection",select(rows,self.cfg))

    def confirm(self):
        selected=self.decision("selection")
        if selected is None: raise ValueError("frozen development choices required")
        manifest=self.decision("confirmation-manifest")
        if manifest is None:
            recipes=[Recipe.from_mapping(selected[m]["selected"]["recipe"]) for m in METHODS]
            dev=self.development()
            estimates=[]
            for recipe in recipes:
                estimates.append(max(r["active_seconds"] for r in dev if r["recipe_key"]==recipe.key))
            health=[read_json(self.store.unit_path(f"assets-development-{r:04d}")/"health.json")
                    for r in range(self.cfg.calibration_replicas)]
            initialization=max(h["pretrain_seconds"]+h["reference_seconds"] for h in health)
            per_pair=1.5*(sum(estimates)+initialization+10)
            choices=[self.cfg.confirmation_replicas] if self.cfg.smoke else [12,10,8]
            affordable=[n for n in choices if n*per_pair+self.cfg.analysis_reserve_seconds<=self.budget.remaining]
            if not affordable: raise ResourceLimit("eight fresh confirmation pairs do not fit the remaining allowance")
            n=affordable[0]
            manifest=dict(schema_version=1,protocol=PROTOCOL,sealed=True,N=self.cfg.novel_count,
                replicas=list(range(n)),recipes=[{**r.mapping(),"recipe_key":r.key} for r in recipes],
                delta=self.cfg.retention_margin_degrees,epsilon=self.cfg.matching_tolerance,
                learning_time_limit=self.cfg.learning_time_limit,memory_ceiling_bytes=self.cfg.memory_ceiling_bytes,
                selection_reasons={m:v["reason"] for m,v in selected.items()},
                timing_admission=dict(seconds_per_pair=per_pair,replicas=n,remaining_seconds=self.budget.remaining),
                bootstrap_seed=self.cfg.seed_for("confirmation",0,"paired_bootstrap"),
                seed_components={str(i):{name:self.cfg.seed_for("confirmation",i,name)
                    for name in ("initialization_model","stream_ids","stream_novel","evaluation_angles_base","evaluation_angles_novel")}
                    for i in range(n)})
            manifest["manifest_hash"]=content_hash(manifest)
            manifest=self.seal("confirmation-manifest",manifest)
        recipes=[Recipe.from_mapping(r) for r in manifest["recipes"]]
        self.status.update(stage="8.1C fresh confirmation",planned=len(self.records)+sum(
            trajectory_id("confirmation",r,p,self.cfg.novel_count) not in self.records
            for r in manifest["replicas"] for p in recipes))
        for replica in manifest["replicas"]:
            for recipe in recipes: self.job("confirmation",replica,recipe)

    def analyze(self):
        if self.decision("analysis"): return
        from .analysis import analyze
        manifest=self.decision("confirmation-manifest")
        if manifest is None: raise ValueError("confirmation manifest required")
        self.status["stage"]="8.1D analysis"
        rows=[r for r in self.records.values() if r["partition"]=="confirmation"]
        result=analyze(rows,manifest,seed=manifest["bootstrap_seed"],draws=self.cfg.bootstrap_samples,check=self.budget.check)
        result["calibration_cost"]=dict(development_trajectories=len(self.development()),
            novel_training_presentations=sum(r["target"] for r in self.development()),
            validation_images_per_replica_per_domain=10*self.cfg.eval_per_class,
            development_replicas=sorted({r["replica"] for r in self.development()}),
            development_learning_seconds=sum(r["learning_seconds"] for r in self.development()),
            development_active_seconds=sum(r["active_seconds"] for r in self.development()))
        plans=[read_json(self.store.unit_path(f"assets-development-{r:04d}")/"plan.json")
               for r in result["calibration_cost"]["development_replicas"]]
        result["calibration_cost"]["validation_unique_source_images_per_domain"]={domain:len({
            i for p in plans for i in p["evaluation_ids"][domain]}) for domain in ("base","novel")}
        result["calibration_cost"]["validation_image_angle_pairs_per_domain"]={domain:len({
            (i,angle) for p in plans for i,angle in zip(p["evaluation_ids"][domain],p["evaluation_angles"][domain],strict=True)})
            for domain in ("base","novel")}
        result["calibration_cost"]["validation_prediction_presentations"]=(
            len(self.development())*(self.cfg.novel_count+1)*2*10*self.cfg.eval_per_class)
        self.seal("analysis",result)

    def closeout(self):
        if not self.decision("analysis") or self.decision("closeout"): return
        units=list(self.store.iter_units())
        for item in units:
            self.budget.check()
            if item["status"]!="completed" or not self.store.is_complete(item["unit_id"],full=True):
                raise ValueError("closeout found an incomplete computational unit")
        self.seal("closeout",dict(units_verified=len(units),source_hash=self.store.identity["source_hash"],
            complete_trajectories=len(self.records),solver_failures=sum(r["solver_failures"] for r in self.records.values()),
            active_seconds=self.budget.used,storage=self.store.storage_status()))

    def run(self):
        self.register(); self.progress(force=True)
        stages=("grid","refine","confirm","analyze")
        wanted=stages if self.args.stage=="all" else (self.args.stage,)
        if "grid" in wanted: self.grid()
        if "refine" in wanted: self.grid();self.refinements();self.finalists()
        if "confirm" in wanted: self.confirm()
        if "analyze" in wanted: self.analyze()
        self.closeout()
        self.status.update(status="completed" if self.decision("analysis") else "stage_complete",current=None,
                           forecast_seconds=0. if self.decision("analysis") else None)
        self.progress(force=True)


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--budget-root",type=Path,default=ROOT/"cache/mnist_experiment/plan15")
    parser.add_argument("--data-root",type=Path,default=ROOT/"cache/mnist_experiment/datasets")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--acceptance-root",type=Path)
    parser.add_argument("--notebook",type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"))
    parser.add_argument("--stage",choices=("grid","refine","confirm","analyze","all"),default="all")
    parser.add_argument("--allocation-hours",type=float,default=20.)
    parser.add_argument("--max-wall-seconds",type=float)
    parser.add_argument("--max-units",type=int)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--clear-stop",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--smoke-stop-at",choices=("pretrain","accepted","mid_solve","eviction"))
    return parser.parse_args(argv)


def main(argv=None):
    args=arguments(argv)
    cfg=Config.read(args.config) if args.config else Config.smoke_config(args.device or "cpu") if args.smoke else Config()
    if args.device: cfg=replace(cfg,device=args.device)
    cfg.validate()
    if not 0<args.allocation_hours<=20: raise ValueError("cumulative allowance must be in (0,20] hours")
    if args.max_units is not None and args.max_units<=0: raise ValueError("positive unit limit required")
    if args.smoke_stop_at and not cfg.smoke: raise ValueError("artificial stop points are smoke-only")
    if not cfg.smoke and cfg.device!="cuda": raise ValueError("production requires CUDA")
    if args.root.resolve().parent!=args.budget_root.resolve(): raise ValueError("root must be directly under shared Plan 15 budget")
    setup_torch(cfg);started=time.monotonic();stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    def check():
        if stopped or ((args.root/"STOP").exists() and not args.clear_stop): raise StopRequested("preflight stop requested")
        if args.max_wall_seconds and time.monotonic()-started>=args.max_wall_seconds: raise ResourceLimit("invocation time limit")
    prior={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        data=dataset_manifest(args.data_root,check)
        details=provenance(cfg,data); code=sources()
        details.update(protocol=PROTOCOL,sources=code,source_hash=content_hash(code))
        if not cfg.smoke:
            acceptance_root=args.acceptance_root or args.budget_root/"retention_acceptance_v1"
            accepted=read_json(acceptance_root/"units/acceptance/result.json")
            if accepted.get("source_hash")!=details["source_hash"] or accepted.get("status")!="passed":
                raise ValueError("current scientific source requires matching CPU/CUDA acceptance")
        identity=dict(protocol=PROTOCOL,allocation_family=ALLOCATION_FAMILY,role="smoke" if cfg.smoke else "production",
            schema_version=1,source_hash=details["source_hash"],config_hash=cfg.scientific_hash,data_hash=details["data_hash"])
        with budget_lock(args.budget_root):
            other=0. if cfg.smoke else peer_seconds(args.budget_root,args.root)
            allowance=args.allocation_hours*3600
            if other>=allowance: raise ResourceLimit("peer roots exhaust Phase 8.1 allowance")
            external=file_bytes(args.budget_root,exclude=args.root,check=check)+file_bytes(args.data_root,check=check)+48_000_000
            preflight=time.monotonic()-started
            wall=None if args.max_wall_seconds is None else max(0.,args.max_wall_seconds-preflight)
            with Store(args.root,identity,resume=args.resume,clear_stop=args.clear_stop,
                max_bytes=cfg.max_artifact_bytes,external_bytes=external,free_reserve_bytes=cfg.free_reserve_bytes,
                finalization_reserve_bytes=32_000_000,max_wall_seconds=wall,max_active_seconds=allowance-other) as store:
                store.control.restore_active_seconds(store.control.active_seconds+preflight)
                if stopped: store.control.stop_requested=True
                budget=RuntimeBudget(store,other_seconds=other,allocation_seconds=allowance)
                study=None
                try:
                    if not (store.root/"provenance.json").exists():
                        store.write_json(None,"provenance.json",details);store.write_json(None,"resolved_config.json",cfg.mapping())
                        store.write_json(None,"source_snapshot.json",{name:(ROOT/name).read_text() for name in code})
                    with store.control.signal_handlers():
                        study=Study(cfg,args,store,budget);study.run()
                except (StopRequested,LearnerStop,ResourceLimit) as exc:
                    if study: study.status.update(status="paused",reason=str(exc))
                    print(json.dumps(dict(event="paused",reason=str(exc))),flush=True);return 2
                except BaseException as exc:
                    if study: study.status.update(status="failed",reason=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    try:
                        if study: study.progress(force=True,final=True)
                    finally: budget.close()
        return 0
    except (StopRequested,ResourceLimit) as exc:
        print(json.dumps(dict(event="paused",reason=str(exc))),flush=True);return 2
    finally:
        for sig,handler in prior.items(): signal.signal(sig,handler)


if __name__=="__main__":
    raise SystemExit(main())
