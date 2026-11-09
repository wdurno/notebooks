"""Resumable Plan 18 implementation, calibration, and paired fresh execution."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import time

import torch

from mnist_experiment.diagonal_lfu.data import load_data
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested, request_stop, utc_now
from .analysis import summarize_path
from .config import ARMS, LABELS, Config, Recipe, content_hash
from .curvature import fd_values, fisher_diagonal, observed_scores, paired_change
from .data import make_assets, sample_transforms, render_panel
from .learner import fit_step, prepare_objective
from .model import build_model, memory_indices, flat_prediction, gaussian_nll, merge_parameters
from .runtime import REPO, FAMILY, DATA_ROOT, read_json, open_store
from .trajectory import Reader, cpu_tree, initial_fit, run_trajectory, evaluate_panel


class Scheduler:
    def __init__(self, cfg, store, budget, raw, *, max_units=None):
        self.cfg, self.store, self.budget, self.raw = cfg, store, budget, raw
        self.max_units, self.new_paths = max_units, 0
        self.phase, self.current = "0: implementation acceptance" if cfg.smoke else "1: readiness and timing", {}
        self.records, self.assets_cache, self.initial_cache, self.readers = {}, {}, {}, {}
        self.last_status, self.last_report, self.last_console = 0., 0., 0.
        self.selection, self.numeric = None, None
        for unit in store.iter_units(completed_only=True):
            meta, uid = unit["metadata"], unit["unit_id"]
            if meta.get("kind") == "trajectory":
                self.records[uid] = self.load_record(uid, meta)
        if (store.root / "selection.json").exists():
            self.selection = read_json(store.root / "selection.json")
        if (store.root / "numerics.json").exists():
            self.numeric = read_json(store.root / "numerics.json")

    def load_record(self, uid, metadata):
        path = self.store.unit_path(uid)
        return dict(unit_id=uid, metadata=metadata, status="completed",
            summary=read_json(path / "summary.json"), metrics=read_json(path / "metrics.json"))

    def status(self, state="running"):
        paths = list(self.records.values())
        counts = {cohort: sum(r["metadata"]["cohort"] == cohort for r in paths)
                  for cohort in ("development", "fresh", "stationary")}
        planned = (self.selection["main_replicas"] + self.selection["stationary_replicas"]) * 5 if self.selection else None
        fresh_done = counts["fresh"] + counts["stationary"]
        eta = None
        if self.selection:
            eta = max(0, planned-fresh_done)*self.selection["seconds_per_path"]*1.25
        return dict(protocol=self.cfg.protocol, status=state, phase=self.phase,
            heartbeat=utc_now(), current=self.current, completed=counts, total_paths=planned,
            complete_trajectories=len(paths), active_seconds=self.budget.used,
            max_active_seconds=self.budget.allocation_seconds, remaining_seconds=self.budget.remaining,
            forecast_remaining_seconds=eta, selection=self.selection,
            numerical_policy=self.numeric, resources=self.store.storage_status())

    def pulse(self, *, force=False, report=False, **current):
        self.budget.check()
        if current:
            self.current = {key: value for key, value in current.items() if key != "metrics"}
        now = time.monotonic()
        if force or now-self.last_status >= 10:
            self.store.write_json(None, "status.json", self.status())
            self.last_status = now
        if force or now-self.last_console >= 60:
            print(json.dumps({key: self.status()[key] for key in
                ("phase", "current", "completed", "active_seconds", "remaining_seconds")}), flush=True)
            self.last_console = now
        if report or now-self.last_report >= 900:
            self.publish()

    def publish(self, status="running", *, figures=True):
        self.store.write_json(None, "status.json", self.status(status), finalizing=True)
        if (not figures or self.store.control.stop_requested or (self.store.root/"STOP").exists()
                or self.budget.remaining < 30):
            # A user reclaiming the node should not wait on figure generation.
            return
        try:
            from .report import build_report
            build_report(self.store.root, cfg=self.cfg, check=self.budget.check,
                         notebook_path=self.store.root/"smoke_results.ipynb" if self.cfg.smoke else None)
        except ImportError:
            if not self.cfg.smoke:
                raise
        self.last_report = time.monotonic()

    def decision(self, name, make):
        uid = f"decision-{name}"
        if self.store.begin(uid, dict(kind="decision", name=name)) is None:
            return read_json(self.store.unit_path(uid)/"decision.json")
        result = make()
        self.store.write_json(uid, "decision.json", result)
        self.store.complete(uid)
        return result

    def assets(self, cohort, replica):
        key = (cohort, replica)
        if key in self.assets_cache:
            return self.assets_cache[key]
        uid = f"assets-{cohort}-{replica:03}"
        metadata = dict(kind="assets", cohort=cohort, replica=replica)
        if self.store.begin(uid, metadata) is None:
            result = self.store.load_checkpoint(uid)
        else:
            result = make_assets(self.cfg, replica, cohort, self.raw)
            self.store.save_checkpoint(uid, cpu_tree(result))
            self.store.write_json(uid, "summary.json", result["metadata"])
            self.store.complete(uid)
        self.assets_cache[key] = result
        return result

    def reader(self, cohort, replica):
        key = (cohort, replica)
        if key not in self.readers:
            # Only a single initialization cache and bounded holdouts are retained.
            self.readers.clear()
            self.readers[key] = Reader(self.raw, self.assets(cohort, replica), self.cfg, self.budget.check)
        return self.readers[key]

    def initial(self, cohort, replica):
        key = (cohort, replica)
        if key in self.initial_cache:
            return self.initial_cache[key]
        assets = self.assets(cohort, replica)
        uid = f"initial-{cohort}-{replica:03}"
        metadata = dict(kind="initialization", cohort=cohort, replica=replica,
            pairing_asset_hash=assets["metadata"]["asset_hash"],
            model_seed=self.cfg.seed_for(cohort, replica, "initialization_model"),
            factor_seed=self.cfg.seed_for(cohort, replica, "factors"))
        result = initial_fit(self.store, self.budget, self.cfg, uid, metadata,
                             self.reader(cohort, replica), pulse=self.pulse)
        self.initial_cache[key] = result
        # The 2,300-grid render is useful during initialization only.
        self.reader(cohort, replica).cache.pop("init", None)
        return result

    def effective_cfg(self):
        if not self.numeric:
            return self.cfg
        return replace(self.cfg, fd_dtype=self.numeric["fd_dtype"],
            max_iter=self.numeric["max_iter"], max_eval=self.numeric["max_eval"])

    def trajectory(self, cohort, replica, recipe, *, readiness=False):
        cfg = self.effective_cfg()
        numeric = dict(fd_dtype=cfg.fd_dtype if recipe.strength else "unused",
                       max_iter=cfg.max_iter, max_eval=cfg.max_eval)
        uid = f"path-{cohort}-{replica:03}-{recipe.key}-{content_hash(numeric)[:8]}"
        if uid in self.records:
            return self.records[uid]
        if self.max_units is not None and self.new_paths >= self.max_units:
            raise StopRequested("completed the requested invocation trajectory count")
        assets = self.assets(cohort, replica)
        initial = self.initial(cohort, replica)
        metadata = dict(kind="trajectory", cohort=cohort, replica=replica,
            recipe=recipe.mapping(), pairing_asset_hash=assets["metadata"]["asset_hash"],
            model_seed=self.cfg.seed_for(cohort, replica, "initialization_model"),
            factor_seed=self.cfg.seed_for(cohort, replica, "factors"),
            numerical_policy=numeric, global_sign=assets["metadata"]["assignment"]["global_sign"],
            seed_derivation="config.seed_for(cohort,replica,component); per-batch replay/probe seeds in diagnostics")
        reader = self.reader(cohort, replica)
        reader.cfg = cfg
        run_trajectory(self.store, self.budget, cfg, uid, metadata, initial, reader,
            pulse=self.pulse, audit=readiness and replica == 0, shadows=readiness)
        result = self.load_record(uid, metadata)
        self.records[uid] = result
        self.new_paths += 1
        self.current = {}
        self.pulse(force=True, report=True)
        return result

    def records_for(self, recipe, replicas):
        values = [r for r in self.records.values() if r["metadata"]["cohort"] == "development"
                  and r["metadata"]["recipe"] == recipe.mapping()
                  and r["metadata"]["replica"] in replicas]
        # Old numerical revisions remain visible but never enter a current recipe.
        wanted = self.effective_cfg()
        values = [r for r in values if r["metadata"]["numerical_policy"]["max_iter"] == wanted.max_iter
                  and r["metadata"]["numerical_policy"]["max_eval"] == wanted.max_eval
                  and (not recipe.strength or r["metadata"]["numerical_policy"]["fd_dtype"] == wanted.fd_dtype)]
        by_replica = {r["metadata"]["replica"]: r for r in values}
        if any(i not in by_replica for i in replicas):
            raise ValueError("candidate lacks the paired development replicas")
        return [by_replica[i] for i in replicas]

    def aggregate(self, recipe, replicas):
        rows = self.records_for(recipe, replicas)
        summaries = [r["summary"] for r in rows]
        fields = ("novel_auc", "Havg", "H100", "learning_seconds")
        if any(s.get(f) is None or not math.isfinite(s[f]) for s in summaries for f in fields):
            raise ValueError("development candidate has unsupported first-100-novel metrics")
        return dict(recipe=recipe.mapping(), replicas=list(replicas),
            **{key: statistics.mean(s[key] for s in summaries) for key in fields})

    def rank(self, recipe, replicas, reference=None):
        row = self.aggregate(recipe, replicas)
        if reference is None:
            violation = max(0., row["Havg"]-self.cfg.retention_ceiling, row["H100"]-self.cfg.retention_ceiling)
        else:
            reference_row = self.aggregate(reference, replicas)
            violation = max(0., *(abs(row[f]-reference_row[f])-self.cfg.retention_tolerance for f in ("Havg", "H100")))
        return (round(violation, 8), round(row["novel_auc"], 8), row["learning_seconds"],
                recipe.ridge, recipe.strength, recipe.process_scale, recipe.key)

    def numerical_audit(self, readiness):
        cfg = self.cfg
        if cfg.smoke:
            return dict(fd_dtype=cfg.fd_dtype, max_iter=cfg.max_iter, max_eval=cfg.max_eval,
                        lambda0=.1, audit=[], smoke=True)
        first = readiness[0]
        payload = self.store.load_checkpoint(first["unit_id"])
        assets = self.assets("development", 0)
        reader = self.reader("development", 0)
        model, layout = build_model(cfg.seed_for("development", 0, "initialization_model"),
            device=cfg.device, dtype=getattr(torch, cfg.dtype), factor_seed=cfg.seed_for("development",0,"factors"))
        ltm, stm = (x.to(cfg.device) for x in memory_indices(layout))
        checks, data_norms, penalty_norms = [], [], []
        initial=self.initial("development",0)
        merged0,layout0,w0=merge_parameters(model,layout,initial["theta"].to(cfg.device))
        initial_mapping=initial["diagonal"].to(cfg.device)
        initial_mapping=initial_mapping.clamp_min(max(1e-12,1e-6*float(initial_mapping.mean())))
        need_double, need_more = False, False
        for snapshot in payload["audit_states"]:
            self.budget.check()
            theta = snapshot["theta"].to(cfg.device)
            new_x, new_y = reader.stream(snapshot["positions"])
            replay_x, replay_y = reader.stream(snapshot["replay_ids"])
            ages = snapshot["ages"].to(cfg.device)
            probes = snapshot["probes"].to(cfg.device)
            t64 = theta.double().requires_grad_(True)
            values = fd_values(model, layout, t64, replay_x.double(), probes.double(), cfg.fd_step)
            auxiliary = (ages * values.mean(0)/len(stm)).mean()
            g64 = torch.autograd.grad(auxiliary, t64)[0]
            data_loss = .5*gaussian_nll(flat_prediction(model,layout,t64,new_x.double()),new_y.double()).mean()
            data_loss += .5*gaussian_nll(flat_prediction(model,layout,t64,replay_x.double()),replay_y.double()).mean()
            g_data = torch.autograd.grad(data_loss, t64)[0]
            norm_data, norm_fd = float(g_data.norm()), float(g64.norm())
            if norm_fd > 0 and math.isfinite(norm_fd) and math.isfinite(norm_data):
                data_norms.append(norm_data)
                penalty_norms.append(norm_fd)
            comparisons=[]
            for dtype in (torch.float32, torch.float64):
                for h in (.005,.01,.02):
                    parameter=theta.to(dtype).requires_grad_(True)
                    trial=(ages*fd_values(model,layout,parameter,replay_x.to(dtype),probes.to(dtype),h).double().mean(0)/len(stm)).mean()
                    gradient=torch.autograd.grad(trial,parameter)[0].double()
                    value_error=abs(float(trial)-float(auxiliary))/max(abs(float(auxiliary)),1e-20)
                    gradient_error=float((gradient-g64).norm()/g64.norm().clamp_min(1e-20))
                    comparisons.append(dict(dtype=str(dtype),h=h,value_error=value_error,gradient_error=gradient_error))
                    if dtype==torch.float32 and max(value_error,gradient_error)>.01:
                        need_double=True
                    if dtype==torch.float64 and max(value_error,gradient_error)>.01:
                        raise FloatingPointError("bilinear-head FD identity failed in float64")
            accepted=snapshot["accepted"].to(cfg.device)
            delta,z,paired=paired_change(model,layout,theta,accepted,replay_x,indices=ltm,
                chunk_size=cfg.fisher_chunk,output_chunk_size=cfg.output_chunk,check=self.budget.check)
            long_cfg=replace(cfg,max_iter=cfg.max_iter*2,max_eval=cfg.max_eval*2)
            reference,solve=fit_step(model,layout,theta,snapshot["diagonal"].to(cfg.device),
                snapshot["anchor"].to(cfg.device),new_x,new_y,replay_x,replay_y,ages,probes,
                pi=snapshot["pi"],recipe=Recipe("A"),cfg=long_cfg,
                initial_fisher_scale=float(self.initial("development",0)["diagonal"].mean()),check=self.budget.check)
            base_prediction=flat_prediction(model,layout,accepted,replay_x).detach()
            first_prediction=flat_prediction(model,layout,theta,replay_x).detach()
            long_prediction=flat_prediction(model,layout,reference,replay_x).detach()
            extra_motion=float((long_prediction-base_prediction).norm()/((base_prediction-first_prediction).norm().clamp_min(1e-8)))
            objective_gap=max(0.,snapshot["solve"]["final_objective"]-solve["final_objective"])/max(1.,abs(snapshot["solve"]["final_objective"]-27*math.log(8*math.pi)))
            need_more |= objective_gap>1e-3 or extra_motion>.05
            response=[]
            for pi in (.01,.025,.1):
                point,fit=fit_step(model,layout,theta,snapshot["diagonal"].to(cfg.device),
                    snapshot["anchor"].to(cfg.device),new_x,new_y,replay_x,replay_y,ages,probes,
                    pi=pi,recipe=Recipe("A"),cfg=cfg,
                    initial_fisher_scale=float(self.initial("development",0)["diagonal"].mean()),check=self.budget.check)
                prediction=flat_prediction(model,layout,point,replay_x).detach()
                old_x,old_y=reader.panel("old")
                novel_x,novel_y=reader.panel("novel")
                old_error=evaluate_panel(model,layout,point,old_x,old_y,cfg,self.budget.check)
                new_error=evaluate_panel(model,layout,point,novel_x,novel_y,cfg,self.budget.check)
                response.append(dict(pi=pi, prediction_motion=float((prediction-first_prediction).norm()),
                    gradient_inf_norm=fit["gradient_inf_norm"],objective=fit["final_objective"],
                    old_mse=old_error["mse"],novel_mse=new_error["mse"]))
            kl_check=[]
            for fraction in (.25,.5,1.):
                difference=(accepted-theta)*fraction
                predicted=flat_prediction(model,layout,theta+difference,replay_x)
                kl_check.append(dict(fraction=fraction,
                    predictive_kl=float((predicted.double()-first_prediction.double()).square().sum(-1).mean()/8),
                    archive_quadratic=float((snapshot["diagonal"].to(cfg.device)*difference[ltm].double().square()).sum()/2)))
            end=(snapshot["batch"]-1)*cfg.samples_per_step
            available=list(range(max(0,end-32),end))
            audit_x,audit_y=reader.stream(available)
            merged1,layout1,w1=merge_parameters(model,layout,theta)
            score0=observed_scores(merged0,layout0,w0,audit_x,audit_y,
                                   chunk_size=cfg.fisher_chunk,check=self.budget.check)
            score1=observed_scores(merged1,layout1,w1,audit_x,audit_y,
                                   chunk_size=cfg.fisher_chunk,check=self.budget.check)
            mapping1=fisher_diagonal(merged1,layout1,w1,audit_x,
                chunk_size=cfg.fisher_chunk,output_chunk_size=cfg.output_chunk,check=self.budget.check)
            mapping1=mapping1.clamp_min(max(1e-12,1e-6*float(mapping1.mean())))
            residual=score0/initial_mapping-score1/mapping1-(w1-w0).double()
            metric=snapshot["diagonal"].to(cfg.device)
            mean_error=float((metric*residual.mean(0).square()).sum())
            noise=float((metric*residual.var(0,unbiased=True)/len(residual)).sum())
            checks.append(dict(batch=snapshot["batch"],data_gradient_norm=norm_data,
                fd_gradient_norm=norm_fd,fd_comparisons=comparisons,paired=paired,
                doubled_objective_gap=objective_gap,doubled_extra_motion=extra_motion,response=response,
                archive_kl=kl_check,anchor_mapping=dict(inputs=len(available),
                    residual_energy=mean_error,residual_noise=noise,
                    energy_to_noise=mean_error/max(noise,1e-30))))
        if len(data_norms)<2:
            raise ValueError("FD strength calibration needs two resolved aged states")
        return dict(fd_dtype="float64" if need_double else cfg.fd_dtype,
            max_iter=cfg.max_iter*(2 if need_more else 1),max_eval=cfg.max_eval*(2 if need_more else 1),
            lambda0=.1*statistics.median(data_norms)/statistics.median(penalty_norms),audit=checks)

    def readiness(self):
        self.phase="1: initialization, readiness and numerical audit"
        rows=[self.trajectory("development",i,Recipe("A"),readiness=True) for i in range(2)]
        extras=self.decision("readiness-extras",self.readiness_extras)
        self.numeric=self.decision("numerical-policy",lambda:self.numerical_audit(rows))
        self.store.write_json(None,"numerics.json",self.numeric)
        self.store.write_json(None,"readiness.json",{**extras,"numerical_policy":self.numeric,
            "initializations":[read_json(self.store.unit_path(f"initial-development-{i:03}")/"summary.json") for i in range(2)]})
        self.pulse(force=True,report=True)

    def readiness_extras(self):
        reader=self.reader("development",0)
        x,y=reader.panel("novel",slice(0,1),device=False)
        gallery=torch.cat((x[0,0],x[0,1]),dim=1)
        result=dict(rendered_grid=gallery.tolist(),rendered_grid_caption="Independent novel holdout: reference / transformed",
                    expected_novel_grids=float(reader.assets["stream"]["p"].sum()*self.cfg.samples_per_step),
                    capability=None)
        if self.cfg.smoke:
            return result
        uid="diagnostic-capability"
        metadata=dict(kind="diagnostic",name="extra-label-capability",additional_training_grids=96,validation_grids=32)
        if self.store.begin(uid,metadata) is None:
            result["capability"]=read_json(self.store.unit_path(uid)/"summary.json")
            return result
        initial=self.initial("development",0)
        model,layout=build_model(self.cfg.seed_for("development",0,"initialization_model"),
            device=self.cfg.device,dtype=getattr(torch,self.cfg.dtype),factor_seed=self.cfg.seed_for("development",0,"factors"))
        layout.copy_vector_to_module(model,initial["theta"].to(self.cfg.device))
        panel=dict(reader.assets["diagnostic"])
        rng=torch.Generator().manual_seed(self.cfg.seed_for("development",0,"capability-transforms"))
        severity=.5+.5*torch.rand(self.cfg.diagnostic_size,generator=rng,dtype=torch.float64)
        means=severity[:,None]*self.cfg.angle_max*reader.assets["signs"]
        parameters,_=sample_transforms(means.flatten(),self.cfg.seed_for("development",0,"capability-affines"),
            angle_std=self.cfg.angle_std,angle_limit=self.cfg.angle_limit)
        panel["transforms"]=parameters.reshape(-1,9,6)
        inputs,targets=render_panel(self.raw,panel,check=self.budget.check)
        inputs=inputs.to(self.cfg.device,getattr(torch,self.cfg.dtype))
        targets=targets.to(self.cfg.device,getattr(torch,self.cfg.dtype))
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        state=self.store.load_checkpoint(uid)
        if state:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
        else:
            state=dict(epoch=0,offset=0,history=[],initial_validation=None)
        if state["initial_validation"] is None:
            theta=layout.flatten_module(model,detach=True)
            state["initial_validation"]=evaluate_panel(model,layout,theta,inputs[96:],targets[96:],self.cfg,self.budget.check)
        for epoch in range(state["epoch"],10):
            order=torch.randperm(96,generator=torch.Generator().manual_seed(self.cfg.seed_for("development",0,f"capability-order:{epoch}")))
            offset0=state["offset"] if epoch==state["epoch"] else 0
            for offset in range(offset0,96,32):
                self.budget.check()
                ids=order[offset:offset+32].to(self.cfg.device)
                optimizer.zero_grad(set_to_none=True)
                loss=gaussian_nll(model(inputs[ids]),targets[ids]).mean()
                loss.backward()
                optimizer.step()
                state.update(epoch=epoch,offset=offset+len(ids),
                    model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()))
                self.store.save_checkpoint(uid,state)
            theta=layout.flatten_module(model,detach=True)
            validation=evaluate_panel(model,layout,theta,inputs[96:],targets[96:],self.cfg,self.budget.check)
            state["history"].append(dict(epoch=epoch+1,**validation))
            state.update(epoch=epoch+1,offset=0)
            self.store.save_checkpoint(uid,state)
        summary=dict(initial_validation=state["initial_validation"],history=state["history"],
            diagnostic_only=True,initialized_scientific_path=False)
        self.store.write_json(uid,"summary.json",summary)
        self.store.complete(uid)
        result["capability"]=summary
        return result

    def observer_candidates(self):
        ranking=[]
        for eta in self.cfg.process_scales:
            scores=[]
            for replica in range(2):
                candidates=[r for r in self.records.values() if r["metadata"]["cohort"]=="development"
                    and r["metadata"]["replica"]==replica and r["metadata"]["recipe"]==Recipe("A").mapping()]
                payload=next((self.store.load_checkpoint(r["unit_id"]) for r in candidates
                              if self.store.load_checkpoint(r["unit_id"]).get("shadow_logs",{}).get(str(eta))),None)
                if payload is None:
                    raise ValueError("readiness shadow observer evidence missing")
                logs=payload["shadow_logs"][str(eta)]
                segments=sorted({r["segment"] for r in logs})
                scores.append(statistics.mean(statistics.mean(r["innovation_log_score"] for r in logs
                    if r["segment"]==s) for s in segments))
            ranking.append(dict(eta=eta,score=statistics.mean(scores)))
        finite=[r for r in ranking if math.isfinite(r["score"])]
        if len(finite)<2:
            raise ValueError("two finite shadow process scales required")
        best=sorted(finite,key=lambda r:(-r["score"],r["eta"]))[:2]
        return dict(etas=[r["eta"] for r in best],ranking=ranking,
            implementation="three predefined shadow filters assimilated causally online; no score-tape duplication")

    def calibration(self):
        self.phase="2: retention calibration"
        ratios=(1.,) if self.cfg.smoke else self.cfg.strength_ratios
        ridges=(1.,) if self.cfg.smoke else self.cfg.ridge_candidates
        b_first=[Recipe("B",ridge=1.,strength=self.numeric["lambda0"]*ratio) for ratio in ratios]
        for recipe in b_first:
            self.trajectory("development",0,recipe)
        winner=min(b_first,key=lambda r:self.rank(r,[0]))
        b_first += [Recipe("B",ridge=c,strength=winner.strength) for c in ridges if c!=1.]
        for recipe in b_first:
            self.trajectory("development",0,recipe)
        finalists=sorted(b_first,key=lambda r:self.rank(r,[0]))[:2]
        for recipe in finalists:
            self.trajectory("development",1,recipe)
        b=min(finalists,key=lambda r:self.rank(r,[0,1]))
        selected={"B":b}
        for arm in ("A","C"):
            candidates=[Recipe(arm,ridge=c,strength=b.strength if arm!="A" else 0.) for c in ridges]
            for recipe in candidates:
                self.trajectory("development",0,recipe)
            top=sorted(candidates,key=lambda r:self.rank(r,[0],b))[:2]
            for recipe in top:
                self.trajectory("development",1,recipe)
            selected[arm]=min(top,key=lambda r:self.rank(r,[0,1],b))
        shortlist=self.decision("observer-candidates",self.observer_candidates)
        adaptive={}
        etas=shortlist["etas"][:1] if self.cfg.smoke else shortlist["etas"]
        for eta in etas:
            adaptive[eta]={}
            for arm in ("D","E"):
                candidates=[Recipe(arm,ridge=c,strength=b.strength,process_scale=eta) for c in ridges]
                for recipe in candidates:
                    self.trajectory("development",0,recipe)
                recipe=min(candidates,key=lambda r:self.rank(r,[0],b))
                self.trajectory("development",1,recipe)
                adaptive[eta][arm]=recipe
        def eta_rank(eta):
            ranks=[self.rank(recipe,[0,1],b) for recipe in adaptive[eta].values()]
            return max(r[0] for r in ranks),statistics.mean(r[1] for r in ranks),sum(r[2] for r in ranks),eta
        eta=min(etas,key=eta_rank)
        selected.update(adaptive[eta])
        selected,refinements=self.refine(selected)
        # A changed B also changes the retention target for the fixed-weight arms.
        for arm in ("A","C"):
            paired=[]
            for record in self.records.values():
                if record["metadata"]["cohort"]!="development":
                    continue
                candidate=Recipe(**record["metadata"]["recipe"])
                if candidate.arm!=arm or (arm=="C" and candidate.strength!=selected["B"].strength):
                    continue
                try:
                    self.aggregate(candidate,[0,1])
                    paired.append(candidate)
                except ValueError:
                    continue
            selected[arm]=min(set(paired),key=lambda r:self.rank(r,[0,1],selected["B"]))
        # A refined B changes the matching target; preserve a joint D/E scale selection.
        for eta in etas:
            for arm in ("D","E"):
                options={Recipe(**r["metadata"]["recipe"]) for r in self.records.values()
                    if r["metadata"]["cohort"]=="development"
                    and r["metadata"]["recipe"]["arm"]==arm
                    and r["metadata"]["recipe"]["process_scale"]==eta}
                paired=[]
                for candidate in options:
                    try:
                        self.aggregate(candidate,[0,1])
                        paired.append(candidate)
                    except ValueError:
                        continue
                adaptive[eta][arm]=min(paired,key=lambda r:self.rank(r,[0,1],selected["B"]))
        def final_eta_rank(eta):
            ranks=[self.rank(r,[0,1],selected["B"]) for r in adaptive[eta].values()]
            return max(r[0] for r in ranks),statistics.mean(r[1] for r in ranks),sum(r[2] for r in ranks),eta
        selected.update(adaptive[min(etas,key=final_eta_rank)])
        self.freeze(selected,{**shortlist,"refinements":refinements})

    def refine(self, selected):
        if self.cfg.smoke:
            return selected,dict(attempted=[],reason="smoke")
        attempted=[]
        for iteration in range(2):
            sealed=f"decision-ridge-proposal-{iteration}"
            if self.store.is_complete(sealed):
                proposal=Recipe(**read_json(self.store.unit_path(sealed)/"decision.json")["recipe"])
                for replica in range(2):
                    self.trajectory("development",replica,proposal)
                old=selected[proposal.arm]
                reference=selected["B"] if proposal.arm!="B" else None
                if self.rank(proposal,[0,1],reference)<self.rank(old,[0,1],reference):
                    selected={**selected,proposal.arm:proposal}
                attempted.append(dict(proposal=proposal.mapping(),selected=selected[proposal.arm].mapping(),
                    violation=self.rank(selected[proposal.arm],[0,1],reference)[0]))
                continue
            reference=selected["B"]
            violations={a:self.rank(r,[0,1],None if a=="B" else reference)[0]
                        for a,r in selected.items()}
            worst=sorted((a for a,v in violations.items() if v>0),key=lambda a:(-violations[a],a))
            if not worst:
                return selected,dict(attempted=attempted,reason="all_retention_matches_supported")
            path_seconds=statistics.median(r["summary"]["online_wall_seconds"] for r in self.records.values())
            initialization_seconds=statistics.median(x["wall_seconds"] for x in self.initial_cache.values())
            reserve=(30*path_seconds+6*initialization_seconds)*1.25+3600
            if reserve+2*path_seconds*1.25>self.budget.remaining:
                return selected,dict(attempted=attempted,reason="reserve_smaller_fresh_cohort")
            proposal=None
            for arm in worst:
                base=selected[arm]
                siblings=[]
                for record in self.records.values():
                    meta=record["metadata"]
                    recipe=Recipe(**meta["recipe"])
                    if meta["cohort"]=="development" and meta["replica"]==0 and recipe.arm==arm:
                        if recipe.strength==base.strength and recipe.process_scale==base.process_scale:
                            siblings.append(recipe)
                seen=sorted({r.ridge for r in siblings})
                row=self.aggregate(base,[0,1])
                target=(self.aggregate(selected["B"],[0,1]) if arm!="B" else
                        {"Havg":self.cfg.retention_ceiling,"H100":self.cfg.retention_ceiling})
                gaps=[row[f]-target[f] for f in ("Havg","H100")]
                overprotected=max(gaps)<-self.cfg.retention_tolerance
                underprotected=max(gaps)>self.cfg.retention_tolerance
                if base.ridge==seen[0] and base.ridge>0 and overprotected:
                    # A lower boundary gets a true zero-ridge probe, not an ever-smaller numerical proxy.
                    ridge=0.
                elif base.ridge==seen[-1] and underprotected:
                    ridge=base.ridge*10
                else:
                    neighbors=[c for c in seen if c!=base.ridge]
                    if not neighbors or base.ridge==0:
                        continue
                    choices=sorted(neighbors,key=lambda c:(abs(math.log(c/base.ridge)) if c else math.inf,c))
                    ridge=math.sqrt(base.ridge*choices[0])
                candidate=replace(base,ridge=ridge)
                if candidate.ridge in seen:
                    continue
                proposal=candidate
                break
            if proposal is None:
                return selected,dict(attempted=attempted,reason="no_untried_bounded_ridge")
            recorded=self.decision(f"ridge-proposal-{iteration}",lambda:dict(recipe=proposal.mapping(),
                reason="largest_retention_mismatch",violations=violations))
            proposal=Recipe(**recorded["recipe"])
            for replica in range(2):
                self.trajectory("development",replica,proposal)
            old=selected[proposal.arm]
            reference=selected["B"] if proposal.arm!="B" else None
            if self.rank(proposal,[0,1],reference)<self.rank(old,[0,1],reference):
                selected={**selected,proposal.arm:proposal}
            attempted.append(dict(proposal=proposal.mapping(),selected=selected[proposal.arm].mapping(),
                violation=self.rank(selected[proposal.arm],[0,1],reference)[0]))
        return selected,dict(attempted=attempted,reason="two_recipe_limit")

    def freeze(self, selected, shortlist):
        def create():
            paths=[r["summary"]["online_wall_seconds"] for r in self.records.values()]
            per_path=statistics.median(paths)
            init_time=statistics.median(v["wall_seconds"] for v in self.initial_cache.values())
            main,stationary=self.cfg.main_replicas,self.cfg.stationary_replicas
            def estimate(n):
                return (n+stationary)*(5*per_path+init_time)*1.25
            if not self.cfg.smoke and estimate(main)+3600>self.budget.remaining:
                main=4
            if not self.cfg.smoke and estimate(main)+3600>self.budget.remaining:
                raise ResourceLimit("remaining Plan 18 budget cannot admit the smaller balanced fresh cohort")
            return dict(frozen=True,frozen_at=utc_now(),recipes={a:selected[a].mapping() for a in ARMS},
                main_replicas=main,stationary_replicas=stationary,seconds_per_path=per_path,
                forecast_fresh_seconds=estimate(main),observer_shortlist=shortlist,
                numerical_policy=self.numeric,calibration={
                    a:self.aggregate(selected[a],[0,1]) for a in ARMS},
                matching={a:self.rank(selected[a],[0,1],selected["B"])[0] for a in ARMS},
                bootstrap_seed=self.cfg.seed_for("fresh",0,"bootstrap"),
                candidate_paths=len([r for r in self.records.values() if r["metadata"]["cohort"]=="development"]))
        self.selection=self.decision("selection",create)
        self.store.write_json(None,"selection.json",self.selection)
        self.pulse(force=True,report=True)

    def run(self):
        self.pulse(force=True,report=True)
        if not self.selection:
            if not self.numeric:
                self.readiness()
            self.calibration()
        self.phase="3: fresh paired exploration"
        n=self.selection["main_replicas"]
        blocks=[("fresh",i) for i in range(n)]
        blocks.insert(min(2,len(blocks)),("stationary",0))
        blocks.insert(len(blocks)-1,("stationary",1))
        for cohort,replica in blocks:
            offset=self.cfg.seed_for(cohort,replica,"condition_order")%len(ARMS)
            order=ARMS[offset:]+ARMS[:offset]
            for arm in order:
                self.trajectory(cohort,replica,Recipe(**self.selection["recipes"][arm]))
        self.phase="4: inference and closeout"
        self.pulse(force=True,report=True)
        audit=self.decision("final-integrity",self.integrity_audit)
        self.store.write_json(None,"study_completed.json",dict(completed_at=utc_now(),
            active_seconds=self.budget.used,selection=self.selection,paths=len(self.records),
            integrity=audit))
        self.publish("completed")
        print(json.dumps(dict(status="completed",active_hours=self.budget.used/3600)),flush=True)

    def integrity_audit(self):
        units=list(self.store.iter_units(completed_only=True))
        verified=[]
        for unit in units:
            self.budget.check()
            if not self.store.is_complete(unit["unit_id"],full=True):
                raise ValueError("required completed unit failed final validation")
            verified.append(unit["unit_id"])
        blocks=[]
        for cohort,count in (("fresh",self.selection["main_replicas"]),
                             ("stationary",self.selection["stationary_replicas"])):
            for replica in range(count):
                rows=[r for r in self.records.values() if r["metadata"]["cohort"]==cohort
                      and r["metadata"]["replica"]==replica]
                if (len(rows)!=len(ARMS) or {r["metadata"]["recipe"]["arm"] for r in rows}!=set(ARMS)
                        or len({r["metadata"]["pairing_asset_hash"] for r in rows})!=1):
                    raise ValueError("fresh five-arm block is incomplete or unpaired")
                for record in rows:
                    self.budget.check()
                    state=self.store.load_checkpoint(record["unit_id"])
                    if state["step"]!=self.cfg.num_p_steps:
                        raise ValueError("fresh trajectory does not cover the complete schedule")
                    exposed=list(range(self.cfg.stream_size))
                    if state["retired_ids"]+state["replay"]!=exposed:
                        raise ValueError("eviction ledger contains missing or duplicate source grids")
                    if not torch.isfinite(state["diagonal"]).all() or (state["diagonal"]<0).any():
                        raise ValueError("final diagonal archive is not finite nonnegative")
                blocks.append(dict(cohort=cohort,replica=replica,
                    pairing_asset_hash=rows[0]["metadata"]["pairing_asset_hash"]))
        return dict(checked_at=utc_now(),full_checksums=True,verified_units=verified,
            complete_paired_blocks=blocks,config_hash=self.cfg.scientific_hash,
            source_identity=self.store.identity_hash)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--root",type=Path,default=FAMILY/"production_v1")
    parser.add_argument("--data-root",type=Path,default=DATA_ROOT)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--clear-stop",action="store_true")
    parser.add_argument("--request-stop",action="store_true")
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--max-units",type=int)
    parser.add_argument("--max-active-hours",type=float,default=20.)
    parser.add_argument("--max-wall-seconds",type=float)
    args=parser.parse_args(argv)
    cfg=Config.read(args.config)
    if args.request_stop:
        request_stop(args.root)
        print("Stop requested; the next safe transaction boundary will exit.",flush=True)
        return 0
    if args.report_only:
        from .report import build_report
        build_report(args.root,cfg=cfg,
                     notebook_path=args.root/"smoke_results.ipynb" if cfg.smoke else None)
        return 0
    try:
        store,budget=open_store(cfg,args.root,args.data_root,resume=args.resume,
            clear_stop=args.clear_stop,allocation_hours=args.max_active_hours,max_wall_seconds=args.max_wall_seconds)
    except (StopRequested,ResourceLimit) as error:
        print(json.dumps(dict(status="paused",reason=str(error),phase="store startup")),flush=True)
        return 0
    scheduler=None
    try:
        with store.control.signal_handlers():
            if (store.root/"study_completed.json").exists():
                print(json.dumps(dict(status="already_completed",active_hours=budget.used/3600)),flush=True)
                return 0
            scheduler=Scheduler(cfg,store,budget,load_data(args.data_root),max_units=args.max_units)
            scheduler.run()
    except (StopRequested,ResourceLimit) as error:
        if scheduler:
            scheduler.publish("paused" if isinstance(error,StopRequested) else "resource_limited",figures=False)
        print(json.dumps(dict(status="paused",reason=str(error),active_hours=budget.used/3600)),flush=True)
        return 0
    finally:
        budget.close()
        store.close()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
