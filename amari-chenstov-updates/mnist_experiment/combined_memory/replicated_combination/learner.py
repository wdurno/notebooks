"""Vector EWC with the historical objective, probes and bounded optimizer."""

import copy
import math
import time
import torch

from mnist_experiment.fisher_memory.learner import _bounded_solve
from mnist_experiment.fisher_memory_grid_registration.learner import (
    PROBE_SEED_COMPONENT, stage_arrival, validate_state,
)
from mnist_experiment.fisher_memory_stm_capacity import core
from mnist_experiment.fisher_memory_stm_capacity.model import flat_prediction, memory_indices
from mnist_experiment.fisher_memory_grid_registration.curvature import gaussian_nll
from mnist_experiment.kalman_pi.policies import concentration
from src.seeding import derive_component_seed


def initial_learner(shared):
    result=copy.deepcopy(shared)
    value=result["fisher_L"]
    result["fisher_L"] = dict(kind="diagonal",values=(value["factor"].double().square().sum(1)
                                                     +value["residual_diagonal"].double()))
    result["archive_diagnostics"]={}
    validate_state(result)
    return result


def prepare_objective(model,layout,start,new_x,new_y,replay_x,replay_y,ages,diagonal,anchor,
                      condition,settings,seed,check=lambda:None):
    """Match the original summed-coordinate NLL and STM trace normalization."""
    ltm,stm=(v.to(start.device) for v in memory_indices(layout))
    d=diagonal.detach().double().to(start.device).clone()
    fixed_anchor=anchor.detach().double().to(start.device).clone()
    fixed_stm=start[stm].detach().double().clone()
    omega=(1-settings.pi)/settings.pi
    kappa=settings.ridge_ratio*float(d.mean())
    probe_seed=derive_component_seed(seed,PROBE_SEED_COMPONENT)
    generator=torch.Generator(device="cpu").manual_seed(probe_seed)
    probes=(2*torch.randint(2,(settings.num_probes,len(stm)),generator=generator)-1).to(start)
    strength=settings.lambda_f if condition=="fd_replay_ewc" else 0.
    ages=ages.detach().double().to(start.device).clone()
    def objective(candidate):
        check()
        zero=candidate.double().sum()*0.
        def mean_nll(x,y):
            value=zero
            for offset in range(0,len(y),settings.chunk_size):
                check()
                pred=flat_prediction(model,layout,candidate,x[offset:offset+settings.chunk_size])
                value=value+gaussian_nll(pred,y[offset:offset+settings.chunk_size]).sum()/len(y)
            return value
        new_loss=mean_nll(new_x,new_y)
        replay_loss=mean_nll(replay_x,replay_y) if len(replay_y) else zero
        data_loss=((1-settings.replay_weight)*new_loss+settings.replay_weight*replay_loss
                   if len(replay_y) else new_loss)
        dl=candidate[ltm].double()-fixed_anchor
        ds=candidate[stm].double()-fixed_stm
        ewc=omega/2*(d*dl.square()).sum()
        ridge=omega*kappa/2*(dl.square().sum()+ds.square().sum())
        auxiliary=zero
        if strength:
            for offset in range(0,len(replay_y),settings.chunk_size):
                check()
                values=core.fd_values(model,layout,candidate,replay_x[offset:offset+settings.chunk_size],
                                      probes,settings.fd_step).mean(0)/len(stm)
                auxiliary=auxiliary+(ages[offset:offset+settings.chunk_size]*values).sum()/len(replay_y)
        return data_loss+ewc+ridge+strength*auxiliary,dict(new_nll=new_loss,replay_nll=replay_loss,
            data_loss=data_loss,ewc=ewc,ridge=ridge,auxiliary=auxiliary,weighted_auxiliary=strength*auxiliary)
    return objective,dict(probe_seed=probe_seed,kappa=kappa,num_probes=settings.num_probes,
        fd_step=settings.fd_step,stm_dimension=len(stm),new_count=len(new_y),replay_count=len(replay_y),
        fd_active=bool(strength and len(replay_y)),lambda_f=strength,archive_scale=float(d.mean()))


def fit_step(*args,check=lambda:None):
    started=time.perf_counter()
    model,layout,start,*remaining=args
    objective,meta=prepare_objective(model,layout,start,*remaining,check=check)
    settings=args[-2]
    accepted,diagnostics=_bounded_solve(start,objective,settings,check)
    diagnostics.update(meta,elapsed_seconds=time.perf_counter()-started)
    return accepted.detach(),diagnostics


def commit(old,accepted,ids,fresh,delta,*,pi,q_archive):
    staged=stage_arrival(old,ids,ids)
    e=len(staged["evicted"])
    eta=-math.expm1(e/4*math.log1p(-pi)) if e else 0.
    d=old["fisher_L"]["values"].double()
    raw=(1-eta)*(d+delta.detach().cpu().double())
    if e:
        raw=raw+eta*fresh.detach().cpu().double()
    if not torch.isfinite(raw).all():
        raise FloatingPointError("nonfinite unprojected archive")
    result=copy.deepcopy(old)
    result.update(theta=accepted.detach().cpu(),fisher_L=dict(kind="diagonal",values=raw.clamp_min(0)),
        replay=staged["replay"],consolidated_ids=old["consolidated_ids"]+[r["exposure_id"] for r in staged["evicted"]],
        cumulative_count=staged["cumulative_count"],accepted_arrivals=old["accepted_arrivals"]+1,
        archive_updates=old["archive_updates"]+bool(e))
    if e:
        result["anchor_L"]=result["theta"][result["ltm_indices"]].clone()
        q_archive=concentration(q_archive,eta,e)
    diagnostics=dict(evicted_count=e,eta=eta,negative_fraction=float((raw<0).double().mean()),
        negative_mass=float((-raw).clamp_min(0).sum()),projection_norm=float((raw.clamp_min(0)-raw).norm()),
        correction_norm=float(delta.double().norm()),archive_trace_before=float(d.sum()),archive_trace_after=float(raw.clamp_min(0).sum()))
    diagnostics['lfu_relative_correction']=float(delta.double().norm()/d.norm().clamp_min(1e-30))
    result["archive_diagnostics"]=diagnostics
    validate_state(result)
    return result,q_archive,diagnostics
