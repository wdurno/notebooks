"""Original grid assets and solver schedule with resumable diagonal additions."""

import copy
from dataclasses import replace
import resource
import time

import torch

from mnist_experiment.fisher_memory.initialize import cpu_tree
from mnist_experiment.fisher_memory.control import StopRequested
from mnist_experiment.fisher_memory_grid_registration import learner as original
from mnist_experiment.fisher_memory_grid_registration.runner import settings, predict
from mnist_experiment.fisher_memory_stm_capacity import model as capacity
from mnist_experiment.fisher_memory_stm_capacity.runner import Runner as OriginalRunner, trajectory_id
from mnist_experiment.fisher_memory_stm_capacity.run import bindings
from mnist_experiment.fisher_memory_regression.runtime import read_json,synchronize
from mnist_experiment.fisher_memory_regression.config import content_hash
from mnist_experiment.combined_memory.reproduction.runner import NamespaceStore
from mnist_experiment.combined_memory.broad_drift.observer import Controller
from mnist_experiment.kalman_pi.policies import PolicySpec, concentration
from . import PROTOCOL
from .config import observer_config
from . import curvature,learner


class PolicyStore(NamespaceStore):
    def __init__(self,base,namespace,policy):
        super().__init__(base,namespace,policy.name)
        self.policy=policy

    def begin(self,unit,metadata):
        if "trajectory-" in unit:
            metadata={**metadata,**self.policy.mapping()}
        return super().begin(unit,metadata)


def record(store,unit):
    path=store.unit_path(unit)
    meta=read_json(path/"manifest.json")["metadata"]
    value={**read_json(path/"summary.json"),"metrics":read_json(path/"metrics.json"),
           "retired_audit":read_json(path/"retirement.json")["audit"],"physical_unit":unit,
           "namespace":meta["namespace"],"representation":meta["representation"],"policy":meta["policy"]}
    return value


class Runner:
    def __init__(self,cfg,store,source,budget,namespace,*,progress=lambda **kw:None,stop_at=None):
        self.cfg,self.store,self.source,self.budget=cfg,store,source,budget
        self.namespace,self.progress,self.stop_at=namespace,progress,stop_at
        self.runners={}

    def engine(self,policy):
        if policy.name not in self.runners:
            self.runners[policy.name]=OriginalRunner(self.cfg,PolicyStore(self.store,self.namespace,policy),
                self.source,self.budget,on_progress=self.progress,stop_at=self.stop_at)
        runner=self.runners[policy.name]
        runner.store.base.policy=policy
        return runner

    def assets(self,replica,partition,policy):
        engine=self.engine(policy)
        family=f"{partition}--grid"
        with bindings(),capacity.use_chart("head_r1"):
            result=engine.assets(family,replica,partition)
        return engine,result

    def observer_asset(self,engine,asset,shared,data,model,layout):
        store=engine.store.base
        unit="observer-reference-"+asset
        merged,merged_layout,anchor,mapping=curvature.merge(model,layout,shared["theta"].to(self.cfg.device))
        if store.is_complete(unit):
            return merged,merged_layout,anchor,store.load_checkpoint(unit)
        store.begin(unit,dict(kind="phase9_observer_reference",asset=asset,plan_hash=shared["plan_hash"],mapping=mapping))
        state=store.load_checkpoint(unit) or dict(offset=0,rows=[],seconds=0.)
        reference=self.cfg.reference_size
        for start in range(state["offset"],reference,self.cfg.fisher_chunk):
            self.budget.check()
            tick=time.perf_counter()
            end=min(reference,start+self.cfg.fisher_chunk)
            scores=curvature.observed_scores(merged,merged_layout,anchor,
                data["reference_x"][start:end].to(anchor),data["reference_y"][start:end].to(anchor),check=self.budget.check)
            synchronize(self.cfg.device)
            state["seconds"]+=time.perf_counter()-tick
            state["rows"].append(scores.cpu())
            state["offset"]=end
            if end%16==0 or end==reference:
                store.save_checkpoint(unit,state)
                self.progress(current=unit)
        values=torch.cat(state["rows"])
        archive=shared["learner"]["fisher_L"]
        state.update(variance=values.var(0,unbiased=True),
            diagonal=archive["factor"].double().square().sum(1)+archive["residual_diagonal"].double(),
            mapping=mapping,anchor=anchor.detach().cpu())
        store.save_checkpoint(unit,state)
        store.write_json(unit,"result.json",dict(count=reference,seconds=state["seconds"],mapping=mapping))
        store.complete(unit)
        return merged,merged_layout,anchor,state

    def shadows(self,replica,partition,scales):
        from .config import Policy
        policy=Policy("B")
        engine,(asset,shared,data,model,layout)=self.assets(replica,partition,policy)
        merged,ml,anchor,initial=self.observer_asset(engine,asset,shared,data,model,layout)
        unit=f"observer-shadows-{partition}-{replica:04d}"
        store=engine.store.base
        if store.is_complete(unit):
            return read_json(store.unit_path(unit)/"result.json")
        store.begin(unit,dict(kind="phase9_shadow_scale",scales=list(scales),asset=asset))
        state=store.load_checkpoint(unit) or dict(group=0,controllers={},logs={str(s):[] for s in scales})
        controllers={}
        for scale in scales:
            controller=Controller(PolicySpec("kalman",scale),observer_config(self.cfg),.025,
                                  state["controllers"].get(str(scale)))
            if "anchor" not in controller.state:
                controller.initialize(initial["anchor"],initial["diagonal"],initial["variance"])
            controllers[str(scale)]=controller
        groups=data["plan"]["stream_groups"]
        order=sorted(set(groups))
        for offset in range(state["group"],len(order)):
            ids=[i for i,g in enumerate(groups) if g==order[offset]]
            scores=curvature.observed_scores(merged,ml,anchor,data["stream_x"][ids].to(anchor),
                                            data["stream_y"][ids].to(anchor),check=self.budget.check).cpu()
            for key,controller in controllers.items():
                diagnostics=controller.assimilate(scores,offset)
                state["logs"][key].append(dict(group=offset,**diagnostics))
                state["controllers"][key]=cpu_tree(controller.state)
            state["group"]=offset+1
            store.save_checkpoint(unit,state)
        cold=observer_config(self.cfg).cold_start
        result=dict(replica=replica,scores={key:sum(r["innovation_log_score"] for r in rows[cold:]) /
            max(1,len(rows)-min(cold,len(rows))) for key,rows in state["logs"].items()},logs=state["logs"])
        # Smoke trajectories can end before the scientific 16-group cold start.
        if all(len(v)<=cold for v in state["logs"].values()):
            result["scores"]={key:sum(r["innovation_log_score"] for r in rows)/len(rows) for key,rows in state["logs"].items()}
        store.write_json(unit,"result.json",result)
        store.complete(unit)
        return result

    def run(self,policy,replica,partition):
        cfg,base=self.cfg,self.store
        engine=self.engine(policy)
        recipe=policy.recipe(cfg)
        family=f"{partition}--grid"
        unit=trajectory_id("head_r1",family,replica,recipe,cfg.novel_count)
        physical=engine.store.key(unit)
        if base.is_complete(physical):
            return record(base,physical)
        if policy.name=="R":
            with bindings():
                engine.run_chart("head_r1",family,replica,partition,recipe)
            return record(base,physical)
        engine,(asset,shared,data,model,layout)=self.assets(replica,partition,policy)
        store=engine.store.base
        ltm,stm=capacity.memory_indices(layout)
        meta=dict(kind="grid_registration_trajectory",family=family,partition=partition,replica=replica,
            recipe=recipe.mapping(),recipe_key=recipe.key,asset_id=asset,plan_hash=shared["plan_hash"],
            target=cfg.novel_count,**engine.store.chart_identity,**policy.mapping())
        store.begin(unit,meta)
        state=store.load_checkpoint(unit)
        if state is None:
            state=dict(learner=learner.initial_learner(shared["learner"]),next_segment=0,n=0,rows=[],
                fit_records=[],retirements=[],retired_audit=[],audits=[],layout=layout.metadata(),
                active_seconds=0.,learning_seconds=0.,input_seconds=0.,fit_seconds=0.,eviction_seconds=0.,
                jacobian_fisher_seconds=0.,compression_seconds=0.,evaluation_seconds=0.,audit_seconds=0.,
                observer_seconds=0.,q_ref=1/cfg.init_size,q_archive=1/cfg.init_size,
                group=None,group_pi=None,pending_scores=[],controller=None,group_diagnostics={},peak_vram_bytes=0)
            state["rows"].append(engine._row(model,layout,state["learner"]["theta"].to(cfg.device),data,state,0,0))
            store.save_checkpoint(unit,state)
        controller=None
        if policy.adaptive:
            merged,ml,anchor,observer_initial=self.observer_asset(engine,asset,shared,data,model,layout)
            controller=Controller(PolicySpec("kalman",policy.process_scale),observer_config(cfg),.025,
                                  state["controller"])
            if state["controller"] is None:
                controller.initialize(observer_initial["anchor"],observer_initial["diagonal"],observer_initial["variance"])
        x,y=data["stream_x"],data["stream_y"]
        segments,groups=data["plan"]["segments"],data["plan"]["stream_groups"]
        if cfg.device=="cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(state["next_segment"],len(segments)):
            if state["n"]==cfg.novel_count:
                break
            self.budget.check()
            tick=time.perf_counter()
            batch=segments[step]
            ids=list(range(batch["start"],batch["end"]))
            candidate=copy.deepcopy(state)
            old=state["learner"]
            group=batch["group"]-1
            if state["group"]!=group:
                if state["pending_scores"]:
                    raise ValueError("new group would discard pending scores")
                action=(controller.choose(state["q_ref"],old["fisher_L"]["values"],4,group)
                        if controller else dict(applied_pi=.025,raw_pi=.025,cold_start=False))
                candidate.update(group=group,group_pi=action["applied_pi"],group_diagnostics=action)
            pi=candidate["group_pi"]
            fit_cfg=replace(settings(cfg,recipe),pi=pi)
            staged=original.stage_arrival(old,ids,ids)
            selected=original.replay_indices(old,recipe.replay_count,cfg.seed_for(family,replica,f"replay:{step}")).tolist()
            replay=[old["replay"][i]["input_ref"] for i in selected]
            theta=old["theta"].to(cfg.device)
            new_x,new_y=x[ids].to(theta),y[ids].to(theta)
            replay_x,replay_y=x[replay].to(theta),y[replay].to(theta)
            ages=staged["ages"][selected].to(theta.device)
            seed=cfg.seed_for(family,replica,f"probes:{step}")
            synchronize(cfg.device)
            input_seconds=time.perf_counter()-tick+len(ids)*shared["stream_preparation_seconds"]/len(x)
            fit_tick=time.perf_counter()
            checks=0
            def fit_check():
                nonlocal checks
                self.budget.check()
                checks+=1
                if self.stop_at=="mid_solve" and checks==9:
                    raise StopRequested("Phase 9 interruption inside fit")
            accepted,diagnostics=learner.fit_step(model,layout,theta,new_x,new_y,replay_x,replay_y,
                ages,old["fisher_L"]["values"],old["anchor_L"],recipe.method,fit_cfg,seed,check=fit_check)
            if diagnostics["solver_failed"]:
                raise FloatingPointError(f"Phase 9 solve failed: {diagnostics}")
            synchronize(cfg.device)
            fit_seconds=time.perf_counter()-fit_tick
            eviction_tick=time.perf_counter()
            evicted=[r["input_ref"] for r in staged["evicted"]]
            measurement=evicted if evicted else ids
            if policy.transport:
                delta,fresh=curvature.paired_change(model,layout,theta,accepted,x[measurement].to(theta),ltm.to(theta.device),check=self.budget.check)
            else:
                delta=torch.zeros(len(ltm),dtype=torch.float64,device=theta.device)
                fresh=(curvature.diagonal(model,layout,accepted,x[evicted].to(theta),ltm.to(theta.device),check=self.budget.check)
                       if evicted else None)
            synchronize(cfg.device)
            jacobian_seconds=time.perf_counter()-eviction_tick
            if self.stop_at=="eviction" and evicted:
                raise StopRequested("Phase 9 interruption before eviction transaction")
            accepted_state,q_archive,transaction=learner.commit(old,accepted,ids,fresh,delta,pi=pi,q_archive=state["q_archive"])
            synchronize(cfg.device)
            eviction_seconds=time.perf_counter()-eviction_tick
            observer_tick=time.perf_counter()
            group_end=(batch["end"]==len(groups) or groups[batch["end"]]!=batch["group"])
            observed={}
            if controller:
                scores=curvature.observed_scores(merged,ml,anchor,new_x.to(anchor),new_y.to(anchor),check=self.budget.check).cpu()
                candidate["pending_scores"].append(scores)
                if group_end:
                    all_scores=torch.cat(candidate["pending_scores"])
                    observed=controller.assimilate(all_scores,group)
                    candidate["q_ref"]=concentration(state["q_ref"],pi,len(all_scores))
                    candidate["pending_scores"]=[]
                candidate["controller"]=cpu_tree(controller.state)
            synchronize(cfg.device)
            observer_seconds=time.perf_counter()-observer_tick if controller else 0.
            candidate.update(learner=accepted_state,q_archive=q_archive,next_segment=step+1,n=batch["novel_count"])
            for name,value in dict(input_seconds=input_seconds,fit_seconds=fit_seconds,
                eviction_seconds=eviction_seconds,jacobian_fisher_seconds=jacobian_seconds,
                observer_seconds=observer_seconds,learning_seconds=input_seconds+fit_seconds+eviction_seconds+observer_seconds).items():
                candidate[name]+=value
            for exposure in evicted:
                candidate["retirements"].append(dict(exposure=exposure,observations=batch["end"],
                    n=candidate["n"],novel=bool(data["plan"]["stream_novel"][exposure])))
            evaluation_tick=time.perf_counter()
            if candidate["n"]>state["n"]:
                row=engine._row(model,layout,accepted,data,candidate,candidate["n"],batch["end"])
                row.update(candidate["group_diagnostics"])
                candidate["rows"].append(row)
            synchronize(cfg.device)
            candidate["evaluation_seconds"]+=time.perf_counter()-evaluation_tick
            candidate["fit_records"].append(dict(step=step,n=candidate["n"],observations=batch["end"],
                selected_replay_ids=replay,selected_count=len(replay),group=group,group_end=group_end,
                input_seconds=input_seconds,fit_seconds=fit_seconds,eviction_seconds=eviction_seconds,
                observer_seconds=observer_seconds,**cpu_tree(diagnostics),**transaction,
                **candidate["group_diagnostics"],**observed))
            candidate["active_seconds"]+=time.perf_counter()-tick
            if cfg.device=="cuda":
                candidate["peak_vram_bytes"]=max(candidate["peak_vram_bytes"],torch.cuda.max_memory_allocated())
            store.save_checkpoint(unit,candidate)
            state=candidate
            self.progress(current=physical,novel_count=state["n"])
            if self.stop_at=="split_group" and controller and state["pending_scores"]:
                raise StopRequested("Phase 9 interruption with pending split-group scores")
            if self.stop_at=="accepted":
                raise StopRequested("Phase 9 interruption after accepted segment")
        if state["n"]!=cfg.novel_count or len(state["rows"])!=cfg.novel_count+1 or state["pending_scores"]:
            raise ValueError("incomplete trajectory or unassimilated group")
        rows=state["rows"]
        summary=dict(schema_version=1,protocol=PROTOCOL,unit_id=unit,family=family,partition=partition,
            context_key="grid",task="grid",output_dim=54,archive_rank=0,channels=[4,6],blur_max=cfg.blur_max,
            replica=replica,method=recipe.method,recipe=recipe.mapping(),recipe_key=recipe.key,
            capacity=recipe.capacity,replay_count=recipe.replay_count,target=cfg.novel_count,
            novel_auc=sum(r["novel_mse"] for r in rows[1:])/cfg.novel_count,
            base_auc=sum(r["base_mse"] for r in rows[1:])/cfg.novel_count,
            frozen_novel_mse=rows[0]["novel_mse"],frozen_base_mse=rows[0]["base_mse"],
            observations=rows[-1]["observations"],asset_id=asset,plan_hash=shared["plan_hash"],pairing_id=shared["plan_hash"],
            arm=f"head_r1--{recipe.method}",valid=True,status="completed",**engine.store.chart_identity,**policy.mapping(),
            stream_preparation_seconds=shared["stream_preparation_seconds"],parent=None,parent_seal=None,
            peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            solver_failures=0,unconverged_updates=sum(not r["converged"] for r in state["fit_records"]))
        for key in ("active_seconds","learning_seconds","input_seconds","fit_seconds","eviction_seconds",
                    "jacobian_fisher_seconds","compression_seconds","evaluation_seconds","observer_seconds","audit_seconds","peak_vram_bytes"):
            summary[key]=state[key]
        store.write_json(unit,"summary.json",summary)
        store.write_json(unit,"metrics.json",rows)
        store.write_csv(unit,"metrics.csv",[{k:v for k,v in r.items() if k not in ("per_digit_mse","per_cell_mse","coordinate_mse")} for r in rows])
        store.write_json(unit,"fits.json",state["fit_records"])
        store.write_json(unit,"retirement.json",dict(retirements=state["retirements"],audit=state["retired_audit"]))
        store.write_json(unit,"audits.json",state["audits"])
        store.complete(unit)
        return record(base,physical)
