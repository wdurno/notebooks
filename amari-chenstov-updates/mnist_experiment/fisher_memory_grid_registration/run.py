"""Unattended Plan 15 Phase 11 execution with immutable paired units."""

from dataclasses import replace
import argparse
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
    RuntimeBudget, read_json, setup_torch, dataset_manifest, provenance,
    scientific_sources as engine_sources, file_digest, file_bytes,
)
from . import PROTOCOL, ALLOCATION_FAMILY
from .config import Config, TASKS, METHODS, Recipe, ROOT, content_hash
from .runner import Runner, trajectory_id, NumericalIntegrityError


def sources():
    result = engine_sources()
    for path in (ROOT / "mnist_experiment/fisher_memory_registration").glob("*.py"):
        if path.name not in ("report.py", "acceptance.py"):
            result[str(path.relative_to(ROOT))] = file_digest(path)
    for path in Path(__file__).parent.glob("*.py"):
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
        if (peer / "runtime.json").exists():
            value = read_json(peer / "runtime.json")["cumulative_active_seconds"]
            if (peer / "timing_lease.json").exists():
                value = max(value, read_json(peer / "timing_lease.json").get("charged_until", value))
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid Phase 11 time ledger")
            total += value
    return total


def process_alive(root, pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        parts=Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        if "mnist_experiment.fisher_memory_grid_registration.run" not in parts or "--root" not in parts:
            return False
        requested=Path(parts[parts.index("--root")+1])
        if not requested.is_absolute():
            requested=Path(f"/proc/{pid}/cwd").resolve()/requested
        return requested.resolve()==Path(root).resolve()
    except (OSError,ValueError,IndexError,UnicodeError):
        return False


class Study:
    def __init__(self, cfg, args, store, budget):
        from .data import MNISTSource
        self.cfg, self.args, self.store, self.budget = cfg, args, store, budget
        self.last_status = self.last_report = 0.
        self.new_units = 0
        self.status = dict(status="running", stage="11A", current=None,
                           completed=0, planned=104, changes=[])
        if (store.root / "status.json").exists():
            self.status.update(read_json(store.root / "status.json"))
            self.status.update(status="running", reason=None)
        self.runner = Runner(cfg, store, MNISTSource(args.data_root), budget,
                             on_progress=self.progress, stop_at=args.smoke_stop_at)
        self.records = {}
        for item in store.iter_units(completed_only=True):
            if item["metadata"].get("kind") == "grid_registration_trajectory":
                self.records[item["unit_id"]] = self.runner.record(item["unit_id"])
        init = self.decision("initialization")
        if init:
            self.runner.init_epochs = init["epochs"]
            self.runner.channels = tuple(init["channels"])
        readiness = self.decision("readiness")
        if readiness:
            self.runner.blur_max = readiness["blur_max"]
        rank = self.decision("archive-rank")
        if rank:
            self.runner.archive_rank = rank["rank"]
        profile = self.decision("resource-assessment-wide") if self.runner.channels == (4,8) else None
        profile = profile or self.decision("resource-assessment")
        if profile:
            self.runner.compression_backend = profile["selected_backend"]

    def decision(self, name):
        unit = "decision-" + name
        return read_json(self.store.unit_path(unit)/"result.json") if self.store.is_complete(unit) else None

    def seal(self, name, result):
        old = self.decision(name)
        if old is not None:
            if old != result:
                raise ValueError(f"sealed decision changed: {name}")
            return old
        unit = "decision-" + name
        self.store.begin(unit, dict(kind="grid_registration_decision", schema_version=1, name=name))
        self.store.write_json(unit, "result.json", result)
        self.store.complete(unit)
        return result

    def task_records(self, task, partition="development"):
        return [r for r in self.records.values() if r["partition"] == partition
                and r["family"] == partition + "--" + task.key]

    def rows(self, task, replicas=None, partial=False):
        from .design import table
        records = self.task_records(task)
        if replicas is None:
            replicas = list(range(self.cfg.calibration_replicas))
        return table([r for r in records if r["replica"] in replicas], self.cfg,
                     replicas=replicas, partial=partial)

    def snapshot(self):
        from .data import mixture_probability, drift_strength
        initialized=self.decision("initialization") or {}
        initialization={t.key:{**initialized,"checks":[r for r in initialized.get("checks",[]) if r["task"]==t.key]} for t in TASKS}
        readiness=self.decision("readiness")
        selection=self.decision("selection")
        profile = self.decision("resource-assessment-wide") if self.runner.channels == (4,8) else None
        profile = profile or self.decision("resource-assessment")
        if profile:
            profile = {**profile, "backend": profile["selected_backend"],
                "whole_update_profile": self.decision("whole-learning-profile") or profile["whole_update_profile"], "rows": [
                dict(name=case, status="measured", **{backend.split("_")[1] + "_seconds":
                    next((sum(r["transaction_seconds"] for r in attempt.get("rows", []) if not r["warmup"]) /
                          max(1, sum(not r["warmup"] for r in attempt.get("rows", [])))
                         for attempt in profile["attempts"] if attempt["backend"] == backend and attempt["case"] == case), None)
                    for backend in profile["backends"]}) for case in sorted({a["case"] for a in profile["attempts"]})]}
        return dict(schema_version=1, protocol=PROTOCOL, config=self.cfg.mapping(),
            progress=self.status, tasks=[t.mapping() for t in TASKS],
            initialization=initialization, readiness=readiness,
            preview=(readiness or {}).get("preview"),
            schedule=dict(group=list(range(1,301)),
                p=[mixture_probability(j,self.cfg.novel_probability,self.cfg.ramp_groups) for j in range(1,301)],
                u=[drift_strength(j) for j in range(1,301)]),
            geometry=self.decision("archive-rank"),
            resource_assessment=self.decision("resource-assessment"),
            learning_profile=self.decision("whole-learning-profile"),
            decisions={"resource-assessment": profile},
            development={t.key:self.rows(t, partial=True) for t in TASKS},
            selection=(selection or {}).get("tasks"), manifest=self.decision("confirmation-manifest"),
            analysis={t.key:self.decision("analysis-"+t.key) for t in TASKS},
            common_coordinate_contrast=self.decision("common-coordinate-contrast"),
            confirmation=[r for r in self.records.values() if r["partition"] == "confirmation"],
            changes=self.status.get("changes", []))

    def progress(self, force=False, final=False, **updates):
        updates.pop("stage", None)
        self.status.update(updates)
        now = time.monotonic()
        if not force and now-self.last_status < 3:
            return
        self.status.update(schema_version=1, protocol=PROTOCOL, pid=os.getpid(),
            process_verified=True,
            completed=len(self.records), active_seconds=self.budget.used,
            allocation_seconds=self.budget.allocation_seconds, heartbeat=utc_now(),
            storage=self.store.storage_status(),
            completed_by_task={t.key:sum(r.get("context_key")==t.key for r in self.records.values()) for t in TASKS})
        if updates.get("novel_count") is not None:
            self.status["last_accepted_checkpoint"] = f"{updates.get('current')}; novel count {updates['novel_count']}"
        durations = [r["active_seconds"] for r in self.records.values()]
        if durations:
            self.status["forecast_seconds"] = 1.5*max(0,self.status["planned"]-len(self.records))*float(np.quantile(durations,.9))
        self.store.write_json(None, "status.json", self.status, finalizing=True)
        self.last_status = now
        if final:
            try:
                self.budget.check()
            except (StopRequested, ResourceLimit):
                return
        if self.last_report == 0 or final or self.status.get("status")=="completed" or now-self.last_report >= 900:
            snapshot = self.snapshot()
            self.store.write_json(None, "report_snapshot.json", snapshot, finalizing=final)
            from .report import publish
            publish(snapshot, self.args.notebook or self.store.root/"results.ipynb", self.store)
            self.last_report = now

    def job(self, task, replica, recipe, stage="development", capture=False, audit=False):
        family = stage+"--"+task.key
        unit = trajectory_id(family, replica, recipe, self.cfg.novel_count)
        if unit in self.records:
            return self.records[unit]
        self.progress(current=unit)
        partition = "confirmation" if stage == "confirmation" else "development"
        record = self.runner.run(family, replica, partition, recipe, capture=capture, audit=audit)
        self.records[unit] = record
        self.new_units += 1
        print(json.dumps(dict(event="phase11_trajectory_complete", unit=unit,
            task=task.key, replica=replica, method=recipe.method, novel_auc=record["novel_auc"],
            H=math.sqrt(record["base_auc"])-math.sqrt(record["frozen_base_mse"]),
            learning_seconds=record["learning_seconds"], valid=record["valid"])), flush=True)
        self.progress(current=None)
        if self.args.max_units and self.new_units >= self.args.max_units:
            self.progress(force=True)
            raise StopRequested("completed-trajectory invocation limit")
        return record

    def initialize(self):
        if self.decision("initialization"):
            return
        self.status["stage"] = "11B registration pretraining"
        trials = []
        for channels in ((self.cfg.channels,) if self.cfg.smoke else (self.cfg.channels,(4,8))):
            self.runner.channels = channels
            if channels != self.cfg.channels:
                self.resource_assessment(channels=channels)
            for epochs in self.cfg.init_epoch_stages:
                self.runner.init_epochs = epochs
                checks = []
                for replica in range(self.cfg.calibration_replicas):
                    for task in TASKS:
                        unit,state,*_ = self.runner.assets("development--"+task.key,replica,"development")
                        checks.append(dict(task=task.key,replica=replica,unit=unit,
                                           **state["initialization_validation"]))
                        self.progress(force=True)
                ratios = {t.key:sum(r["mse"] for r in checks if r["task"]==t.key)
                    / sum(r["zero_predictor_mse"] for r in checks if r["task"]==t.key) for t in TASKS}
                cell_ratios = np.mean([np.asarray(r["per_cell_mse"]) for r in checks], axis=0) / np.mean(
                    [np.asarray(r["per_cell_zero_mse"]) for r in checks], axis=0)
                trial = dict(epochs=epochs,channels=list(channels),ratios=ratios,checks=checks,
                             cell_ratios=cell_ratios.tolist(),
                             capability_supported=all(v<=.5 for v in ratios.values()) and bool((cell_ratios<=.7).all()))
                trials.append(trial)
                print(json.dumps(dict(event="phase11_capability",**{k:v for k,v in trial.items() if k!="checks"})),flush=True)
                if trial["capability_supported"] or self.cfg.smoke:
                    break
            if trials[-1]["capability_supported"] or self.cfg.smoke:
                break
        qualified = [r for r in trials if r["capability_supported"]]
        chosen = qualified[0] if qualified else min(trials,key=lambda r:max(r["ratios"].values()))
        self.runner.channels,self.runner.init_epochs=tuple(chosen["channels"]),chosen["epochs"]
        self.seal("initialization",dict(**chosen,trials=[{k:v for k,v in r.items() if k!="checks"} for r in trials]))
        if not qualified:
            self.status["changes"].append("Pretraining missed the 50% readiness preference; best valid common recipe retained.")
        self.progress(force=True)

    def readiness(self):
        if self.decision("readiness"):
            return
        from .calibration import diagnostics
        self.status["stage"]="11B blur learnability"
        evidence={}
        def assess(blur):
            self.runner.blur_max=blur
            rows=[diagnostics(self.runner,TASKS[0],r) for r in range(self.cfg.calibration_replicas)]
            evidence[str(blur)]=rows
            return float(np.mean([r["frozen_degradation_ratio"] for r in rows])),all(r["recovered"] for r in rows)
        initial=self.cfg.blur_max
        ratio,recovery=assess(initial)
        alternative=None if self.cfg.smoke else (2.4 if ratio<1.1 else 1.2 if ratio>2 and not recovery else None)
        chosen=initial
        if alternative:
            r2,rec2=assess(alternative)
            if (ratio<1.1 and r2>ratio and not (r2>2 and not rec2)) or (ratio>2 and r2<ratio and r2>=1.1):
                chosen=alternative
        self.runner.blur_max=chosen
        self.seal("readiness",dict(blur_max=chosen,grid=evidence,
            preview=evidence[str(chosen)][0].get("preview"),
            scope="extra-label development validation; diagnostic weights never seed online learning"))
        self.progress(force=True)

    def rank_calibration(self):
        if self.decision("archive-rank"):
            return
        from .learner import compress_fisher
        from .curvature import reference_fisher
        from .model import memory_indices
        from mnist_experiment.fisher_memory.learner import _fisher_matrix
        self.status["stage"]="11B Fisher geometry"
        rows=[]
        for task in TASKS:
            recipe=Recipe.from_config(self.cfg,"replay_ewc",ridge=1.)
            record=self.job(task,0,recipe,stage="pilot",capture=True)
            _,asset,data,model,layout=self.runner.assets("pilot--"+task.key,0,"development")
            ltm,stm=memory_indices(layout)
            matrices=[("initial",asset["reference_fisher"])]
            state=self.store.load_checkpoint(record["unit_id"])
            for snapshot in state["snapshots"]:
                if snapshot["n"] in (25,75) or self.cfg.smoke:
                    from .data import blur_pairs
                    diagnostic_x = self.runner._nominal["diagnostic_reserve_x"]
                    generator = torch.Generator().manual_seed(self.cfg.seed_for("pilot",0,"rank-panel-strengths"))
                    strengths = .5+.5*torch.rand(len(diagnostic_x),generator=generator,dtype=torch.float64)
                    diagnostic_x = blur_pairs(diagnostic_x,strengths,self.runner.blur_max,self.runner.check)
                    matrices.append((f"n{snapshot['n']}",reference_fisher(model,layout,
                        snapshot["theta"].to(self.cfg.device),diagnostic_x.to(device=self.cfg.device,dtype=snapshot["theta"].dtype),
                        indices=ltm,chunk_size=self.cfg.fisher_chunk,check_stop=self.runner.check).cpu()))
            for name,matrix in matrices:
                self.budget.check()
                generator=torch.Generator().manual_seed(self.cfg.seed_for("development",0,"rank_quadratics:"+task.key+name))
                probes=torch.randn(16,len(ltm),generator=generator,dtype=torch.float64)
                probes=probes/probes.norm(dim=-1,keepdim=True)
                exact=torch.einsum("ki,ij,kj->k",probes,matrix,probes)
                for rank in (8,16,32,64,128):
                    packed,_=compress_fisher(matrix,rank=rank,check_stop=self.runner.check,
                                            backend=self.runner.compression_backend)
                    approx=_fisher_matrix(packed)
                    q=torch.einsum("ki,ij,kj->k",probes,approx,probes)
                    rows.append(dict(task=task.key,snapshot=name,rank=rank,
                        relative_frobenius=float((matrix-approx).norm()/matrix.norm().clamp_min(1e-30)),
                        quadratic_relative_median=float(((q-exact).abs()/exact.abs().clamp_min(1e-30)).median())))
            theta=asset["theta"].to(self.cfg.device)
            sample=reference_fisher(model,layout,theta,data["reference_x"][:8].to(theta),indices=stm,
                chunk_size=self.cfg.fisher_chunk,check_stop=self.runner.check).double()
            eig=torch.linalg.eigvalsh(sample).clamp_min(0)
            self.seal("geometry-"+task.key,dict(trace=float(eig.sum()),
                effective_rank=float(eig.sum().square()/eig.square().sum().clamp_min(1e-30)),
                eigenvalues=eig.cpu().tolist(),inputs=8,meaning="STM principal block of the same model Fisher"))
        acceptable=[rank for rank in (8,16,32,64,128) if max(r["relative_frobenius"] for r in rows if r["rank"]==rank)<=.1]
        rank=acceptable[0] if acceptable else 128
        self.seal("archive-rank",dict(rank=rank,adequate=bool(acceptable),rows=rows,
            pilot_partition="independent pilot seeds; excluded from policy selection"))
        self.runner.archive_rank=rank
        self.progress(force=True)

    def learning_profile(self):
        if self.decision("whole-learning-profile"):
            return
        records = self.task_records(TASKS[0])
        rows = []
        for method in ("replay_ewc", "fd_replay_ewc"):
            available = [r for r in records if r["method"] == method]
            if not available:
                continue
            rows.append(dict(method=method, trajectories=len(available),
                mean_active_seconds=float(np.mean([r["active_seconds"] for r in available])),
                mean_learning_seconds=float(np.mean([r["learning_seconds"] for r in available])),
                input_seconds=float(np.mean([r["input_seconds"] for r in available])),
                solve_seconds=float(np.mean([r["fit_seconds"] for r in available])),
                jacobian_fisher_seconds=float(np.mean([r["jacobian_fisher_seconds"] for r in available])),
                compression_transaction_seconds=float(np.mean([r["compression_seconds"] for r in available])),
                evaluation_seconds=float(np.mean([r["evaluation_seconds"] for r in available])),
                peak_vram_bytes=max(r["peak_vram_bytes"] for r in available),
                references=[r["unit_id"] for r in available]))
        # Whole-trajectory timing includes all intervening nominal arrivals.
        durations = [r["active_seconds"] for r in records]
        remaining = max(0, 100-len(records))
        asset_costs = [read_json(self.store.unit_path(u["unit_id"])/"health.json")
                       for u in self.store.iter_units(completed_only=True)
                       if u["metadata"].get("kind")=="grid_registration_assets"]
        per_asset = max((a["pretrain_seconds"]+a["reference_seconds"] for a in asset_costs),default=0.)
        forecast = 1.5 * (max(durations, default=0.) * remaining
                          + (self.cfg.confirmation_replicas+self.cfg.finalist_extra_replicas)*per_asset
                          + self.cfg.analysis_reserve_seconds)
        result = dict(status="measured_full_trajectories", rows=rows,
            compression_backend=self.runner.compression_backend, archive_rank=self.runner.archive_rank,
            upper_workload_forecast_seconds=forecast, remaining_allocation_seconds=self.budget.remaining,
            fits_upper_workload=forecast<=self.budget.remaining,
            forecast_scope="maximum efficacy trajectories, fresh initialization and analysis with 50% contingency",
            initial_assets=[dict(unit_id=u["unit_id"], health=read_json(self.store.unit_path(u["unit_id"])/"health.json"))
                for u in self.store.iter_units(completed_only=True) if u["metadata"].get("kind")=="grid_registration_assets"])
        self.seal("whole-learning-profile", result)
        print(json.dumps(dict(event="phase11_learning_profile",**{k:v for k,v in result.items() if k!="initial_assets"})),flush=True)
        self.progress(force=True)

    def calibration(self):
        self.resource_assessment()
        self.initialize()
        self.readiness()
        self.rank_calibration()
        from .calibration import coefficient
        self.status["stage"]="11B FD and solver calibration"
        for task in TASKS:
            if self.decision("calibration-"+task.key):
                continue
            recipe=Recipe.from_config(self.cfg,"replay_ewc",ridge=1.)
            records=[self.job(task,r,recipe,capture=(r==0),audit=(r==0 and not self.cfg.smoke))
                     for r in range(self.cfg.calibration_replicas)]
            calibration=coefficient(self.runner,task,records[0])
            if not self.cfg.smoke:
                audits=read_json(self.store.unit_path(records[0]["unit_id"])/"audits.json")
                failures=[a for a in audits if a["enlarged_failed"] or a["relative_objective_gain"] is None
                          or a["relative_objective_gain"]>self.cfg.audit_objective_tolerance]
                if not audits or len(failures)/len(audits)>=.25:
                    self.seal("numerical-warning-"+task.key,dict(audits=audits,failures=len(failures)))
                    raise NumericalIntegrityError("reference solve audit requires a versioned common repair")
            self.seal("calibration-"+task.key,dict(lambda0=calibration["lambda0"],
                selected_step=calibration["selected_step"],reference_units=[r["unit_id"] for r in records],
                numerical=calibration["numerical"]))
            self.progress(force=True)
        if not self.decision("calibration-complete"):
            passing={self.cfg.fd_step/2,self.cfg.fd_step,self.cfg.fd_step*2}
            for task in TASKS:
                audit=self.decision("calibration-"+task.key)["numerical"]
                passing&={h for h in audit["candidate_steps"] if all(r["passed"] for r in audit["numerical_rows"] if r["fd_step"]==h and r["status"]!="skipped")}
            if not passing:
                raise NumericalIntegrityError("no common FD step passes both tasks")
            step=min(passing,key=lambda h:(abs(math.log2(h/self.cfg.fd_step)),h))
            if not self.cfg.smoke:
                from .numerical import calibrate_fd
                for task in TASKS:
                    strength = self.decision("calibration-"+task.key)["lambda0"]
                    recipe = replace(Recipe.from_config(self.cfg,"fd_replay_ewc",ridge=1.,strength=strength), fd_step=step)
                    fd_records = [self.job(task,r,recipe,capture=(r==0),audit=(r==0))
                                  for r in range(self.cfg.calibration_replicas)]
                    state = self.store.load_checkpoint(fd_records[0]["unit_id"])
                    _,_,_,model,layout = self.runner.assets("development--"+task.key,0,"development")
                    audit = calibrate_fd(self.cfg,[s for s in state["snapshots"] if s["n"] in (25,75)],
                                         model,layout,self.runner.check)
                    solver = state["audits"]
                    failures = [a for a in solver if a["enlarged_failed"] or a["relative_objective_gain"] is None
                                or a["relative_objective_gain"] > self.cfg.audit_objective_tolerance]
                    if not solver or len(failures)/len(solver)>=.25 or any(
                            not row["passed"] for row in audit["numerical_rows"] if row["fd_step"]==step):
                        self.seal("fd-numerical-warning-"+task.key,dict(solver=solver,numerical=audit))
                        raise NumericalIntegrityError("FD numerical audit requires a versioned common repair")
                    self.seal("fd-calibration-"+task.key,dict(reference_units=[r["unit_id"] for r in fd_records],
                                                            solver=solver,numerical=audit))
            self.seal("calibration-complete",dict(fd_step=step,rank=self.runner.archive_rank,
                channels=list(self.runner.channels),init_epochs=self.runner.init_epochs,blur_max=self.runner.blur_max,
                compression_backend=self.runner.compression_backend))
        self.learning_profile()

    def resource_assessment(self, channels=None):
        channels = channels or self.runner.channels
        name = "resource-assessment" if channels == self.cfg.channels else "resource-assessment-wide"
        existing = self.decision(name)
        if existing:
            self.runner.compression_backend = existing["selected_backend"]
            return
        from .compression import assess_resources
        self.status["stage"] = "11B bounded resource assessment"
        self.progress(force=True)
        dimension = 19 * channels[0] + (9 * channels[0] + 1) * channels[1] + (9 * channels[1] + 1) * 54
        assessment = assess_resources(dimension=dimension, rank=self.cfg.archive_rank,
            device=self.cfg.device, check_stop=self.runner.check)
        if assessment["selected_backend"] is None:
            self.seal("resource-assessment-failure", assessment)
            raise NumericalIntegrityError("no validated exact compression backend")
        self.seal(name, assessment)
        self.runner.compression_backend = assessment["selected_backend"]
        self.progress(force=True)

    def recipes(self,task):
        from .design import initial_grid
        step=self.decision("calibration-complete")["fd_step"]
        return [replace(r,fd_step=step if r.method=="fd_replay_ewc" else self.cfg.fd_step)
                for r in initial_grid(self.cfg,self.decision("calibration-"+task.key)["lambda0"])]

    def grid(self):
        self.status["stage"]="11C policy screen"
        for replica in range(self.cfg.calibration_replicas):
            for task in TASKS:
                for recipe in self.recipes(task):
                    # Exact reference-grid collisions retain their capture/audit provenance.
                    self.job(task,replica,recipe)
                self.progress(force=True)
        if not self.decision("grid"):
            self.seal("grid",dict(tasks=[t.key for t in TASKS],replicas=self.cfg.calibration_replicas))

    def reserve_seconds(self,count=None):
        count=count or self.cfg.confirmation_replicas
        total=0.
        for task in TASKS:
            records=self.task_records(task)
            times=[r["active_seconds"] for r in records]
            health=[]
            for unit in self.store.iter_units(completed_only=True):
                if unit["metadata"].get("kind")=="grid_registration_assets" and unit["metadata"].get("output_dim")==task.output_dim:
                    h=read_json(self.store.unit_path(unit["unit_id"])/"health.json")
                    health.append(h["pretrain_seconds"]+h["reference_seconds"])
            total+=count*(3*max(times,default=90)+max(health,default=120)+10)
        estimate=1.5*(total+self.cfg.analysis_reserve_seconds)
        return estimate if self.cfg.smoke else max(18_000.,estimate)

    def refine(self):
        from .design import refinement_proposals,shortlist,select,table
        self.status["stage"]="11C bounded refinement"
        for task in TASKS:
            additions=[]
            for index in range(self.cfg.refinement_settings):
                name=f"refinement-{task.key}-{index:02d}"
                proposal=self.decision(name)
                if proposal is None:
                    candidates=refinement_proposals(self.rows(task),self.cfg,additions,self.decision("calibration-"+task.key)["lambda0"])
                    if not candidates or self.budget.remaining < self.reserve_seconds()+600:
                        break
                    proposal=self.seal(name,candidates[0])
                additions.append(proposal)
                for replica in range(self.cfg.calibration_replicas):
                    self.job(task,replica,Recipe.from_mapping(proposal["recipe"]))
            self.progress(force=True)
        frozen=self.decision("shortlist")
        if frozen is None:
            chosen={t.key:shortlist(self.rows(t),self.cfg) for t in TASKS}
            durations=[r["active_seconds"] for r in self.records.values() if r["partition"]=="development"]
            cost=max(durations,default=90.)
            jobs=sum(len(group) for groups in chosen.values() for group in groups.values())*self.cfg.finalist_extra_replicas
            optional_jobs=sum(max(0,len(group)-1) for groups in chosen.values() for group in groups.values())*self.cfg.finalist_extra_replicas
            reserved=self.reserve_seconds()
            omit_alternatives=not self.cfg.smoke and self.budget.remaining < reserved+1.5*cost*jobs
            if omit_alternatives:
                chosen={key:{method:values[:1] for method,values in groups.items()} for key,groups in chosen.items()}
                self.status["changes"].append("Unstarted alternative finalists omitted to protect fresh confirmation compute.")
            self.seal("finalist-admission",dict(remaining_seconds=self.budget.remaining,
                confirmation_reserve_seconds=reserved,trajectory_seconds=cost,original_jobs=jobs,
                omitted_optional_jobs=optional_jobs if omit_alternatives else 0,
                reason="timing-only reserve protection"))
            frozen=self.seal("shortlist",chosen)
        expanded=list(range(self.cfg.calibration_replicas+self.cfg.finalist_extra_replicas))
        self.status["stage"]="11C finalist development"
        reports={}
        for task in TASKS:
            values=frozen[task.key]
            recipes=[Recipe.from_mapping(v) for group in values.values() for v in group]
            for replica in expanded:
                for recipe in recipes:
                    unit=trajectory_id("development--"+task.key,replica,recipe,self.cfg.novel_count)
                    if not self.cfg.smoke and replica>=self.cfg.calibration_replicas and unit not in self.records:
                        durations=[r["active_seconds"] for r in self.task_records(task)]
                        if self.budget.remaining < self.reserve_seconds(self.cfg.preferred_min_confirmation)+1.5*max(durations,default=90.):
                            raise ResourceLimit("finalist work would consume the fresh confirmation reserve")
                    self.job(task,replica,recipe)
            keys={r.key for r in recipes}
            records=[r for r in self.task_records(task) if r["recipe_key"] in keys and r["replica"] in expanded]
            reports[task.key]=dict(context=task.mapping(),policies=select(table(records,self.cfg,replicas=expanded),self.cfg))
        if not self.decision("selection"):
            self.seal("selection",dict(tasks=reports,development_replicas=expanded))
        self.progress(force=True)

    def confirm(self):
        selected=self.decision("selection")
        if selected is None:
            raise ValueError("development selection required")
        manifest=self.decision("confirmation-manifest")
        if manifest is None:
            sizes=[self.cfg.confirmation_replicas] if self.cfg.smoke else [12,10,8]
            affordable=[n for n in sizes if self.reserve_seconds(n)<=self.budget.remaining]
            if not affordable:
                raise ResourceLimit("fresh confirmation cannot fit remaining allocation")
            n=affordable[0]
            tasks={}
            for task in TASKS:
                policies=selected["tasks"][task.key]["policies"]
                recipes=[Recipe.from_mapping(policies[m]["selected"]["recipe"]) for m in METHODS]
                replay=policies["replay_ewc"]["selected"]
                item=dict(schema_version=1,protocol=PROTOCOL,sealed=True,N=self.cfg.novel_count,
                    smoke=self.cfg.smoke,context=task.mapping(),replicas=list(range(n)),
                    recipes=[{**r.mapping(),"recipe_key":r.key} for r in recipes],
                    delta=self.cfg.retention_margin,epsilon=self.cfg.matching_tolerance,
                    retention_margin=self.cfg.retention_margin,matching_tolerance=self.cfg.matching_tolerance,
                    envelope_factor=self.cfg.envelope_factor,
                    learning_time_limit=self.cfg.envelope_factor*replay["seconds_per_arrival"],
                    memory_ceiling_bytes=10_000_000_000,source_hash=self.store.identity["source_hash"],
                    init_epochs=self.runner.init_epochs,channels=list(self.runner.channels),
                    archive_rank=self.runner.archive_rank,blur_max=self.runner.blur_max,
                    bootstrap_seed=self.cfg.seed_for("confirmation",0,"paired_bootstrap"))
                item["manifest_hash"]=content_hash(item)
                tasks[task.key]=item
            manifest=self.seal("confirmation-manifest",dict(tasks=tasks,replicas=list(range(n)),
                timing_only_admission=dict(remaining_seconds=self.budget.remaining,forecast_seconds=self.reserve_seconds(n))))
        self.status["stage"]="11D fresh confirmation"
        for replica in manifest["replicas"]:
            for task in TASKS:
                ordered = manifest["tasks"][task.key]["recipes"]
                offset = replica % len(ordered)
                for value in ordered[offset:] + ordered[:offset]:
                    self.job(task,replica,Recipe.from_mapping(value),stage="confirmation")
                self.progress(force=True)

    def analyze(self):
        from .analysis import analyze
        self.status["stage"]="11E efficiency and uncertainty"
        manifest=self.decision("confirmation-manifest")
        if manifest is None:
            raise ValueError("frozen confirmation manifest required")
        for task in TASKS:
            if self.decision("analysis-"+task.key):
                continue
            item=manifest["tasks"][task.key]
            def load(start):
                unit=f"bootstrap-{task.key}-{start:04d}"
                return read_json(self.store.unit_path(unit)/"result.json") if self.store.is_complete(unit) else None
            def save(start,result):
                unit=f"bootstrap-{task.key}-{start:04d}"
                self.store.begin(unit,dict(kind="grid_registration_bootstrap",start=start,
                    manifest_hash=item["manifest_hash"],seed=item["bootstrap_seed"]))
                self.store.write_json(unit,"result.json",result)
                self.store.complete(unit)
            result=analyze(self.task_records(task,"confirmation"),item,seed=item["bootstrap_seed"],
                draws=self.cfg.bootstrap_samples,check=self.budget.check,load_bootstrap=load,save_bootstrap=save)
            self.seal("analysis-"+task.key,result)
            self.progress(force=True)

    def execute(self):
        if not self.decision("design"):
            self.seal("design",dict(config=self.cfg.mapping(),source_hash=self.store.identity["source_hash"],
                tasks=[t.mapping() for t in TASKS],authorized="2026-10-28; 20 active hours"))
        self.progress(force=True)
        if self.args.stage in ("calibrate","all"):
            self.calibration()
        if self.args.stage in ("grid","all"):
            if not self.decision("calibration-complete"):
                raise ValueError("calibration required")
            self.grid()
        if self.args.stage in ("refine","all"):
            self.refine()
        if self.args.stage in ("confirm","all"):
            self.confirm()
        if self.args.stage in ("analyze","all"):
            self.analyze()
        completed=all(self.decision("analysis-"+t.key) for t in TASKS)
        if completed and not self.decision("closeout"):
            self.seal("closeout",dict(status="completed",units=len(list(self.store.iter_units(completed_only=True))),
                source_hash=self.store.identity["source_hash"],active_seconds=self.budget.used,
                storage=self.store.storage_status()))
        self.status.update(status="completed" if completed else "stage_complete",current=None)
        self.progress(force=True)


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--budget-root",type=Path,default=ROOT/"cache/mnist_experiment/plan15")
    p.add_argument("--data-root",type=Path,default=ROOT/"cache/mnist_experiment/datasets")
    p.add_argument("--config",type=Path)
    p.add_argument("--acceptance-root",type=Path)
    p.add_argument("--notebook",type=Path)
    p.add_argument("--device",choices=("cpu","cuda"))
    p.add_argument("--stage",choices=("calibrate","grid","refine","confirm","analyze","all","status","stop","report"),default="all")
    p.add_argument("--allocation-hours",type=float,default=20.)
    p.add_argument("--max-wall-seconds",type=float)
    p.add_argument("--max-units",type=int)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--clear-stop",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--smoke-stop-at",choices=("pretrain","accepted","mid_solve","eviction"))
    return p.parse_args(argv)


def main(argv=None):
    args=arguments(argv)
    if args.stage in ("status","stop","report"):
        if args.stage=="stop":
            (args.root/"STOP").touch(exist_ok=True)
            print("Stop requested; resume retains accepted checkpoints.")
        elif args.stage=="status":
            status=read_json(args.root/"status.json")
            status["process_alive"]=process_alive(args.root,status.get("pid"))
            print(json.dumps(status,indent=2))
        else:
            from .report import publish
            snapshot=read_json(args.root/"report_snapshot.json")
            snapshot["progress"]["process_verified"]=process_alive(args.root,snapshot["progress"].get("pid"))
            publish(snapshot,args.notebook or args.root/"results.ipynb")
        return 0
    cfg=Config.read(args.config) if args.config else Config.smoke_config(args.device or "cpu") if args.smoke else Config()
    if args.device:
        cfg=replace(cfg,device=args.device)
    cfg.validate()
    if not 0<args.allocation_hours<=20 or (args.max_units is not None and args.max_units<=0):
        raise ValueError("invalid compute or completed-unit limit")
    if args.smoke_stop_at and not cfg.smoke:
        raise ValueError("artificial interruption is smoke-only")
    if not cfg.smoke and cfg.device!="cuda":
        raise ValueError("production requires CUDA")
    if args.root.resolve().parent!=args.budget_root.resolve():
        raise ValueError("root must be directly under the shared budget root")
    setup_torch(cfg)
    started,stopped=time.monotonic(),False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    def check():
        if stopped or ((args.root/"STOP").exists() and not args.clear_stop):
            raise StopRequested("preflight stop requested")
        if args.max_wall_seconds and time.monotonic()-started>=args.max_wall_seconds:
            raise ResourceLimit("invocation time limit")
    prior={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        data=dataset_manifest(args.data_root,check)
        details,code=provenance(cfg,data),sources()
        details.update(protocol=PROTOCOL,sources=code,source_hash=content_hash(code))
        if not cfg.smoke:
            accepted=read_json((args.acceptance_root or args.budget_root/"grid_registration_acceptance_v1")/"units/acceptance/result.json")
            if accepted.get("source_hash")!=details["source_hash"] or accepted.get("status")!="passed":
                raise ValueError("matching CPU/CUDA correctness and recovery acceptance required")
        identity=dict(protocol=PROTOCOL,allocation_family=ALLOCATION_FAMILY,
            role="smoke" if cfg.smoke else "production",schema_version=1,
            source_hash=details["source_hash"],config_hash=cfg.scientific_hash,data_hash=details["data_hash"])
        with budget_lock(args.budget_root):
            other=0. if cfg.smoke else peer_seconds(args.budget_root,args.root)
            allowance=args.allocation_hours*3600
            external=file_bytes(args.budget_root,exclude=args.root,check=check)+file_bytes(args.data_root,check=check)+64_000_000
            preflight=time.monotonic()-started
            wall=None if args.max_wall_seconds is None else max(0.,args.max_wall_seconds-preflight)
            with Store(args.root,identity,resume=args.resume,clear_stop=args.clear_stop,
                max_bytes=cfg.max_artifact_bytes,external_bytes=external,
                free_reserve_bytes=cfg.free_reserve_bytes,finalization_reserve_bytes=32_000_000,
                max_wall_seconds=wall,max_active_seconds=max(0.,allowance-other)) as store:
                store.control.restore_active_seconds(store.control.active_seconds+preflight)
                budget=RuntimeBudget(store,other_seconds=other,allocation_seconds=allowance)
                study=None
                try:
                    if not (store.root/"provenance.json").exists():
                        store.write_json(None,"provenance.json",details)
                        store.write_json(None,"resolved_config.json",cfg.mapping())
                        store.write_json(None,"source_snapshot.json",{n:(ROOT/n).read_text() for n in code})
                    with store.control.signal_handlers():
                        study=Study(cfg,args,store,budget)
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


if __name__=="__main__":
    raise SystemExit(main())
