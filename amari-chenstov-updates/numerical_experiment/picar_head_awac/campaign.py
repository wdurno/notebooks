"""Versioned, interruption-safe Fix 5 implementation and experiment entry."""

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
import math
from pathlib import Path
import time
import tarfile

import torch

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.storage import RunStore, _atomic_bytes
from numerical_experiment.picar.runtime import StopController, sha256_file
from numerical_experiment.picar_phase2.campaign import read_json, load_completed_state
from . import VERSION
from .config import DEFAULT_OUTPUT, DEFAULT_SOURCE, NOTEBOOK, MAIN_SEEDS, RANKS, source_hashes
from .learner import Learner, Recipe
from .model import Heads, FrozenFeatures
from .archive import Archive

PROJECT=Path(__file__).resolve().parents[2]


def scalar_state(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value,Path):
        return str(value)
    if isinstance(value,dict):
        return {k:scalar_state(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [scalar_state(v) for v in value]
    return value


def run(output=DEFAULT_OUTPUT,source=DEFAULT_SOURCE,*,model_path=None,resume=False,
        offline=False,max_hours=13.,stage="all",reuse_profile=None,continue_from=None):
    os.environ.setdefault("MPLBACKEND","Agg")
    os.environ.setdefault("MPLCONFIGDIR",str(PROJECT/"cache/matplotlib"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    torch.set_num_threads(2)
    if not math.isfinite(max_hours) or not 0 < max_hours <= 20.:
        raise ValueError("active compute allocation must be finite and at most twenty hours")
    root=Path(output).resolve()
    source=Path(source).resolve()
    root.mkdir(parents=True,exist_ok=True)
    stop=StopController(root)
    stop.install()
    identity={"version":VERSION,"numerical_revision":"head-awac-v3","schema_version":1,"source":str(source),
        "source_seal":sha256_file(source/"orchestration/COMPLETED"),"sources":source_hashes(),
        "theory_sha256":sha256_file(PROJECT/"mathematical_overview.ipynb"),
        "model_path":str(model_path) if model_path else None,
        "reuse_profile":str(Path(reuse_profile).resolve()) if reuse_profile else None,
        "protocol":{"fresh_seeds":list(MAIN_SEEDS),"methods":["mc_replay","td_replay","td_fd"],
                    "ranks":list(RANKS),"new_intervals":1024,"reward":"width6_signed_square_div10",
                    "working_likelihood":"conditional_gaussian_actor_reward_frozen_reference_v1",
                    "weight_bounds":[-5.,5.],"physical_motion":False}}
    continuation=None
    try:
        if continue_from:
            from .continuation import Continuation
            previous_root=Path(continue_from).resolve()
            if previous_root==root:
                raise ValueError("continuation requires a separate output directory")
            identity["continuation"]={"source":str(previous_root),
                "parent_config_sha256":sha256_file(previous_root/"orchestration/config.json")}
            continuation=Continuation(previous_root,identity)
        with RunStore(root/"orchestration",identity,resume=resume,checkpoint_policy="journal-v1",
                      reserve_bytes=20_000_000_000) as parent:
            if parent.completed:
                return read_json(parent.root/"summary.json")
            history=root/"history.ipynb"
            if not history.exists():
                _atomic_bytes(history,((Path(continue_from)/"history.ipynb") if continuation
                                      else PROJECT/NOTEBOOK).read_bytes())
            previous_root=Path(reuse_profile).resolve() if reuse_profile else None
            previous_units={}
            inherited_charge=0.
            if previous_root:
                previous_progress=read_json(previous_root/"progress.json")
                if previous_progress["status"]!="profile_complete":
                    raise ValueError("reused resource profile must have finished its bounded scope")
                inherited_charge=previous_progress["charged_seconds"]
                for marker in sorted((previous_root/"units").glob("*/COMPLETED")):
                    old=read_json(marker.parent/"summary.json")
                    previous_units[old["operation"]]=old
            s=parent.load_checkpoint(trusted=True)
            if s is None:
                s=continuation.initial_state() if continuation else {
                    "charged_seconds":inherited_charge,"completed":[],"selected":None,
                    "forecast":{},"unit_charges":{},"budget_limits":[]}
            if not s["budget_limits"] or s["budget_limits"][-1]["max_hours"]!=max_hours:
                s["budget_limits"].append({"max_hours":max_hours,"charged_seconds":s["charged_seconds"]})
            started=time.monotonic()
            base=s["charged_seconds"]
            label="implementation acceptance"
            active=None
            unit_started=None
            unit_prior=0.
            last_save=last_report=0.
            planned=28
            def elapsed():
                return base+time.monotonic()-started
            def progress(force=False,final=False):
                nonlocal last_save,last_report
                now=time.monotonic()
                if force or now-last_save>30 or stop.check():
                    if active and unit_started is not None:
                        s.setdefault("unit_charges",{})[str(active)]=unit_prior+now-unit_started
                    s["charged_seconds"]=elapsed()
                    state={**s,"charged_seconds":elapsed()+(0 if final else 30)}
                    child=read_json(active/"progress.json") if active and (active/"progress.json").exists() else {}
                    row={"schema_version":1,"version":VERSION,"status":"running" if not final else s.get("status","paused"),
                        "stage":label,"charged_seconds":elapsed(),"completed_units":len(s["completed"]),
                        "planned_units":max(planned,len(s["completed"])),"active_unit":str(active) if active else None,
                        "forecast":s["forecast"],"acquisitions":child.get("n"),"accepted_fits":child.get("fits"),
                        "active_progress":child}
                    parent.checkpoint(state,progress=row)
                    _atomic_bytes(root/"progress.json",(canonical_json(row)+"\n").encode())
                    last_save=now
                if force or now-last_report>600:
                    from .report import build_notebook
                    build_notebook(root,PROJECT/NOTEBOOK,base_notebook=history)
                    last_report=now
                if not final:
                    if stop.check():
                        raise InterruptedError("Fix 5 saved; resume with --resume --clear-stop")
                    if elapsed()>=max_hours*3600-180:
                        raise TimeoutError("Fix 5 active limit reached with closeout reserve")
            def check():
                progress()
            def unit(operation,settings,callback):
                nonlocal active,label,unit_started,unit_prior
                label=operation
                unit_identity={**identity,"operation":operation,"settings":scalar_state(settings)}
                suffix=hashlib.sha256(canonical_json(unit_identity).encode()).hexdigest()[:12]
                active=root/"units"/f"{operation}-{suffix}"
                with RunStore(active,unit_identity,resume=resume,checkpoint_policy="journal-v1",
                              reserve_bytes=20_000_000_000) as store:
                    if store.completed:
                        result=read_json(active/"summary.json")
                    else:
                        imported,prior=continuation.restore(store,operation,scalar_state(settings)) if continuation else (None,0.)
                        if imported is not None:
                            result={**imported,"version":VERSION,"schema_version":1,
                                    "operation":operation,"kind":"head_awac_"+operation,
                                    "settings":scalar_state(settings),"unit_path":str(active)}
                            store.complete(result)
                            path=str(active)
                            if path not in s["completed"]:
                                s["completed"].append(path)
                            active=None
                            progress(force=True)
                            return result
                        unit_prior=s.setdefault("unit_charges",{}).get(str(active),prior)
                        unit_started=time.monotonic()
                        progress(force=True)
                        tick=time.monotonic()
                        result=callback(store)
                        if source_hashes()!=identity["sources"]:
                            raise RuntimeError("numerical sources changed; version correction before further units")
                        result=scalar_state({**result,"version":VERSION,"schema_version":1,
                            "operation":operation,"kind":"head_awac_"+operation,"settings":settings,
                            "active_seconds":unit_prior+time.monotonic()-tick,"unit_path":str(active),
                            "runtime":{"head_device":"cpu","encoder_device":"cuda",
                                       "torch":torch.__version__,"cuda":torch.version.cuda}})
                        if store._current is None:
                            store.checkpoint({"stage":"done"})
                        store.complete(result)
                path=str(active)
                if path not in s["completed"]:
                    s["completed"].append(path)
                active=None
                unit_started=None
                print(canonical_json({"unit":operation,"settings":settings,"path":path,"status":"complete"}),flush=True)
                progress(force=True)
                return result

            def reuse(operation,store):
                old=previous_units[operation]
                path=Path(old["unit_path"])
                saved=load_completed_state(path)
                store.checkpoint(saved,progress={"stage":"verified compatible initialization reuse"})
                return {k:v for k,v in old.items() if k not in ("version","operation","kind","settings",
                    "unit_path","active_seconds","schema_version")} | {
                    "reused_from":str(path),"source_completion_sha256":sha256_file(path/"COMPLETED"),
                    "reused_numerical_contract":"unchanged frozen features/heads/base conditional Fisher"}

            try:
                def software(store):
                    import subprocess,sys
                    paths=sorted((PROJECT/"test/unit").glob("test_picar_head_awac_*.py"))
                    check_result=subprocess.run([sys.executable,"-m","pytest","-q",*map(str,paths)],
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600,
                        env={**os.environ,"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2"})
                    store._write_root("pytest.txt",check_result.stdout.encode())
                    if check_result.returncode:
                        raise RuntimeError(check_result.stdout[-7000:])
                    with tarfile.open(store.root/"sources.tar.gz","w:gz") as archive:
                        for name in identity["sources"]:
                            archive.add(PROJECT/name,arcname=name)
                    return {"passed":True,"tests":check_result.stdout.strip().splitlines()[-1],
                            "model_contract":identity["protocol"]["working_likelihood"]}
                unit("software",{},software)
                if stage=="software":
                    s["status"]="software_complete"
                    progress(force=True,final=True)
                    return {"status":"software_complete"}
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA Qwen is required for actual experiments")
                from numerical_experiment.picar.models import load_qwen_model
                from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
                from numerical_experiment.picar_reward_repair.config import MODEL_SEED,NORMALIZATION,DEFAULT_REFERENCES
                from numerical_experiment.picar_reward_repair.scoring import load_references,WidthScorer
                from .initialization import prepare,fit,initial_archive
                label="loading frozen Qwen"
                progress(force=True)
                raw=load_qwen_model(**({"model_path":model_path} if model_path else {}),
                    local_files_only=offline,seed=MODEL_SEED,top_layers=1,ltm_rank=4,stm_rank=1,
                    hidden_size=64,context_length=4,return_bins=161,min_pixels=56*56,
                    max_pixels=19200,max_tokens=2048).eval()
                raw.encoder.enable_prefix_cache=True
                model=NormalizedStudent(raw,mode=NORMALIZATION,eps=1e-6).eval()
                features=FrozenFeatures(model,identity["source_seal"])
                references=load_references(DEFAULT_REFERENCES,root/"reference_bundle")
                if previous_root and not (root/"score_cache").exists():
                    import shutil
                    shutil.copytree(previous_root/"score_cache",root/"score_cache")
                scorer=WidthScorer(model.encoder,references,root/"score_cache")
                prepared_unit=unit("features",{},lambda store: reuse("features",store) if previous_root else (
                    prepare(model,source,features,store,check),
                    {"train":515,"validation":65,"new_observations":0,"frozen":True,
                     "runtime":raw.encoder.provenance})[1])
                prepared=load_completed_state(prepared_unit["unit_path"])
                def feature_health(store):
                    matrix=torch.stack([r["feature"] for r in prepared["train"]]).double()
                    centered=matrix-matrix.mean(0)
                    singular=torch.linalg.svdvals(centered)
                    energy=singular.square()
                    return {"feature_dimension":matrix.shape[1],
                        "centered_variance_trace":float(energy.sum()/len(matrix)),
                        "leading_variance_share":float(energy[0]/energy.sum()),
                        "median_channel_std":float(matrix.std(0).median()),
                        "mean_feature_norm":float(matrix.norm(dim=1).mean()),
                        "interpretation":"Large first component does not prove that smaller components lack task information."}
                unit("feature_health",{},feature_health)
                # Restoring a completed feature unit still needs identical encoder weights.
                from .data import apply_parameters
                apply_parameters(model,{"parameters":prepared["source_parameters"]})
                model.requires_grad_(False)
                features.clear()
                fit_unit=unit("initial_fit",{},lambda store: reuse("initial_fit",store) if previous_root else {
                    "result":{k:v for k,v in fit(prepared,store,check,"cuda").items()
                              if k in ("curves","inventory","epoch")},
                    "train_count":515,"validation_count":65})
                fitted=load_completed_state(fit_unit["unit_path"])
                base_recipe=Recipe(rank=10)
                fisher_unit=unit("initial_fisher",asdict(base_recipe),lambda store: reuse("initial_fisher",store) if previous_root else {
                    k:v for k,v in initial_archive(fitted,prepared["train"],base_recipe,store,check,"cuda").items()
                    if k!="learner"})
                initial=load_completed_state(fisher_unit["unit_path"])["learner"]
                geometries={(3.,16.,4.):initial}
                def geometry(recipe):
                    key=(recipe.temperature,recipe.action_variance,recipe.critic_variance)
                    if key not in geometries:
                        cfg=replace(Recipe(rank=10),temperature=recipe.temperature,
                            action_variance=recipe.action_variance,critic_variance=recipe.critic_variance)
                        result=unit("geometry_fisher",asdict(cfg),lambda store: {
                            k:v for k,v in initial_archive(fitted,prepared["train"],cfg,store,check,"cpu").items()
                            if k!="learner"})
                        geometries[key]=load_completed_state(result["unit_path"])["learner"]
                    return geometries[key]
                def new_learner(recipe,method,seed):
                    heads=Heads().cpu()
                    heads.load_state_dict(fitted["selected_heads"])
                    learner=Learner(heads,recipe,seed,method)
                    old=geometries[(recipe.temperature,recipe.action_variance,recipe.critic_variance)]["archive"]
                    factor=old["factor"][:,:recipe.rank].cpu()
                    learner.archive=Archive(old["diagonal"].cpu(),factor=factor,rank=recipe.rank)
                    return learner
                def gradient_scale(recipe):
                    from .likelihood import joint_kl, data_loss
                    learner=new_learner(recipe,"td_fd",199802)
                    records=prepared["train"][:16]
                    data,design,_=learner.design(records)
                    vector=learner.vector.clone().requires_grad_()
                    losses=data_loss(learner.coordinates.predict(vector,data["feature"],data["action"]),
                                     data["action"],design)
                    g=torch.autograd.grad((losses["nll"]+losses["disagreement"]).mean(),vector)[0]
                    generator=torch.Generator(device=learner.device).manual_seed(199803)
                    draws=torch.randn((len(records),4,7),device=learner.device,generator=generator)
                    probe=torch.zeros_like(vector)
                    probe[learner.stm]=(2*torch.randint(2,(len(learner.stm),),device=learner.device,generator=generator)-1).to(vector.dtype)
                    epsilon=.01
                    fd=.5*(joint_kl(learner.coordinates,vector+epsilon*.5*probe,vector-epsilon*.5*probe,
                                   data["feature"],design,draws)
                           +joint_kl(learner.coordinates,vector-epsilon*.5*probe,vector+epsilon*.5*probe,
                                   data["feature"],design,draws))
                    penalty=fd.mean()/(epsilon**2*len(learner.stm))
                    fd_gradient=torch.autograd.grad(penalty,vector)[0]
                    scale=float(.3*g.norm()/fd_gradient.norm().clamp_min(1e-12))
                    if not math.isfinite(scale):
                        raise FloatingPointError("FD gradient calibration is nonfinite")
                    # Calibrate relative force; do not reuse the old transformer's coefficient.
                    return {"fd_scale":scale,"data_gradient_norm":float(g.norm()),
                            "fd_unit_gradient_norm":float(fd_gradient.norm()),
                            "fd_force_ratio":.3}
                def cached_screen(store):
                    rows=[]
                    for cap,size,rounds in ((32,8,1),(128,32,2),(256,64,4)):
                        for rank in RANKS:
                            for fd in (0.,1.):
                                cfg=replace(Recipe(),rank=rank,capacity=cap,batch_size=size,
                                    rounds=rounds,fd_strength=fd)
                                learner=new_learner(cfg,"td_fd" if fd else "td_replay",199801)
                                records=copy.deepcopy(prepared["train"][:size+1])
                                for i,r in enumerate(records):
                                    r["arrival"]=i
                                durations=[]
                                for repeat in range(3):
                                    tick=time.monotonic()
                                    metrics=learner.fit(records,cap,check)
                                    durations.append(time.monotonic()-tick)
                                tick=time.monotonic()
                                learner.retire(records[0])
                                retirement_seconds=time.monotonic()-tick
                                rows.append({"recipe":asdict(cfg),"fit_seconds":sorted(durations)[1],
                                    "retirement_seconds":retirement_seconds,
                                    "gradient_norm":metrics["gradient_norm"],"data_nll":metrics["data_nll"],
                                    "ewc_suppression":metrics["ewc_proposal_difference"],
                                    "archive":learner.archive.diagnostics()})
                                store.checkpoint({"rows":rows},progress={"stage":"cached resource survey",
                                    "n":len(rows),"total":24})
                                check()
                    return {"candidates":rows,"selected":asdict(Recipe(rank=10)),
                            **gradient_scale(Recipe(rank=10))}
                screen=unit("cached_screen",{},cached_screen)
                initial_panels={}
                def shared_initial(recipe,phase):
                    # All recipes start with identical heads and deterministic
                    # inference; likelihood scales do not alter this behavior.
                    key=(phase=="confirmation",)
                    if key in initial_panels:
                        return initial_panels[key]
                    from .trajectory import Trajectory
                    cfg=replace(recipe,fd_strength=0.)
                    def callback(store):
                        runner=Trajectory(new_learner(cfg,"td_replay",199801),features,scorer,store,check,
                            seed=199801,total=1024,evaluation_episodes=2,initial_episodes=4 if key[-1] else 2,
                            development=not key[-1])
                        if runner.s["stage"]=="evaluate":
                            runner.evaluate()
                        return {"evaluations":runner.s["evaluations"],"initial_policy_shared":True}
                    result=unit("initial_evaluation",{"phase":"confirmation" if key[-1] else "development",
                        "shared_policy":"common_initialized_deterministic_heads"},callback)
                    panels=load_completed_state(result["unit_path"])["evaluations"]
                    # Observed behavior is common; NLL responses differ in MC.
                    for panel in panels:
                        panel["shared_initial_source"]=result["unit_path"]
                        for ep in panel["episodes"]:
                            ep.pop("heldout_nll",None)
                    initial_panels[key]=panels
                    return panels
                def trajectory(method,recipe,seed,total,phase,eval_count=2,initial_count=2):
                    from .trajectory import Trajectory
                    geometry(recipe)
                    shared=shared_initial(recipe,phase)
                    settings={"method":method,"seed":seed,"phase":phase,"ewc_rank":recipe.rank,
                              "stm_rank":1,"total":total,"recipe":asdict(recipe)}
                    return unit("trajectory",settings,lambda store: Trajectory(
                        new_learner(recipe,method,seed),features,scorer,store,check,
                        seed=seed,total=total,evaluation_episodes=eval_count,
                        initial_episodes=initial_count,development=phase!="confirmation",
                        initial_evaluations=shared,midpoint_episodes=min(4,eval_count)).run())
                pilot=previous_units.get("trajectory") if previous_root else trajectory(
                    "td_replay",Recipe(rank=3),199201,128,"resource_pilot",eval_count=2,initial_count=2)
                scored_intervals=sum(len(ep["scores"]) for ep in pilot["episodes"])+sum(
                    len(ep["scores"]) for panel in pilot["evaluations"] for ep in panel["episodes"])
                seconds_per_interval=pilot["active_seconds"]/scored_intervals
                expected_evaluation_intervals=2400
                forecast_hours=seconds_per_interval*(11*1024+4*128+expected_evaluation_intervals)/3600
                s["forecast"]={"pilot_seconds_per_interval":seconds_per_interval,
                    "remaining_seconds":[forecast_hours*3600*.65,forecast_hours*3600*1.35],
                    "scope":"five full development/calibration paths, six fresh paths, four short resource paths, shared initial evaluations",
                    "evaluation_intervals_allowance":expected_evaluation_intervals,
                    "fresh_panel_episodes":{"initial":4,"midpoint":4,"final":8}}
                unit("forecast",{},lambda store: {"forecast":s["forecast"],"pilot":pilot["unit_path"],
                    "charged_profile_seconds":inherited_charge,"total_scored_pilot_intervals":scored_intervals,
                    "pilot_training_intervals":pilot["n"],"pilot_evaluation_intervals":scored_intervals-pilot["n"]})
                if stage=="profile":
                    s["status"]="profile_complete"
                    progress(force=True,final=True)
                    return {"status":"profile_complete","forecast":s["forecast"]}
                if continuation:
                    from .continuation import remaining_training_intervals
                    old_summaries=[read_json(p/"summary.json") for p in continuation.units.values()
                                   if (p/"COMPLETED").exists()]
                    active_path=read_json(Path(continue_from)/"progress.json").get("active_unit")
                    active_progress=read_json(Path(active_path)/"progress.json") if active_path else None
                    remaining=remaining_training_intervals(old_summaries,active_progress)
                    s["forecast"]["remaining_training_intervals"]=remaining
                    s["forecast"]["admission_note"]=("Verified continuation may finish partially within the hard cumulative cap; "
                        "completed paths are not charged again. Pilot range is historical, not a remaining-work forecast.")
                if not resume and forecast_hours*1.05+elapsed()/3600>max_hours:
                    raise TimeoutError("resource pilot requires a revised scope/forecast within compute allocation")
                # Check stronger reuse and a small likelihood/optimizer bracket on
                # cached data before choosing a shared online resource recipe.
                candidates=[]
                for cap,b,r,tau,var,rate,std in ((32,8,1,3.,4.,.001,4.),
                    (128,32,2,3.,4.,.001,4.),(256,64,4,1.,16.,.001,2.),
                    (128,32,2,.3,16.,.003,4.)):
                    recipe=replace(Recipe(),capacity=cap,batch_size=b,rounds=r,rank=10,
                                   temperature=tau,critic_variance=var,learning_rate=rate,exploration_std=std)
                    result=trajectory("td_replay",recipe,199202,128,"resource_development",eval_count=2,initial_count=2)
                    panels=[p for p in result["evaluations"] if p["n"]==128]
                    score=sum(ep["raw_return"] for p in panels for ep in p["episodes"])/sum(len(p["episodes"]) for p in panels)
                    candidates.append((score,recipe))
                recipe=max(candidates,key=lambda item:item[0])[1]
                development=[]
                fd_calibration=unit("fd_calibration",asdict(recipe),lambda store: gradient_scale(recipe))
                fd_strength=fd_calibration["fd_scale"]
                for rank in (0,recipe.rank):
                    for strength in (0.,fd_strength):
                        cfg=replace(recipe,rank=rank,fd_strength=strength)
                        result=trajectory("td_fd" if strength else "td_replay",cfg,199301,1024,
                                          "development",eval_count=2,initial_count=2)
                        development.append((cfg,result))
                def quality(result,environment):
                    panel=next(p for p in result["evaluations"] if p["n"]==1024 and p["environment"]==environment)
                    return sum(ep["raw_return"] for ep in panel["episodes"])/len(panel["episodes"])
                # Keep a shared rank for the FD contrast. Select its paired mean
                # observed quality; preserve every candidate and retention gap.
                rank_scores={rank:sum(quality(res,"initial_A")+quality(res,"moderate_A")
                    for cfg,res in development if cfg.rank==rank) for rank in (0,recipe.rank)}
                selected_rank=max(rank_scores,key=rank_scores.get)
                recipe=replace(recipe,rank=selected_rank)
                control=next(res for cfg,res in development if cfg.rank==selected_rank and cfg.fd_strength==0)
                fd=next(res for cfg,res in development if cfg.rank==selected_rank and cfg.fd_strength>0)
                # Refine FD/ridge using the same complete development curriculum.
                # The ledger is finite and no confirmation outcome enters selection.
                fd_candidates=[(replace(recipe,fd_strength=fd_strength),fd)]
                for multiplier,ridge in ((.25,1.),):
                    cfg=replace(recipe,fd_strength=fd_strength*multiplier,ridge=ridge)
                    res=trajectory("td_fd",cfg,199301,1024,"retention_calibration",eval_count=2,initial_count=2)
                    fd_candidates.append((cfg,res))
                retention_reference=quality(control,"initial_A")
                matched=[(cfg,res) for cfg,res in fd_candidates
                         if abs(quality(res,"initial_A")-retention_reference)<=5.]
                chosen_fd,fd=max(matched or fd_candidates,key=lambda item:quality(item[1],"moderate_A"))
                selection={"recipe":asdict(recipe),"fd_strength":chosen_fd.fd_strength,
                           "fd_ridge":chosen_fd.ridge,"rank_scores":rank_scores,
                           "initial_retention_gap":quality(fd,"initial_A")-quality(control,"initial_A"),
                           "retention_matched":abs(quality(fd,"initial_A")-quality(control,"initial_A"))<=5.,
                           "criterion":"observed initial/moderate raw return frontier, not TD/NLL minimum",
                           "development_paths":[res["unit_path"] for _,res in development]}
                s["selected"]=selection
                unit("selection",{},lambda store: {"selected":selection})
                if stage=="development":
                    s["status"]="development_complete"
                    progress(force=True,final=True)
                    return {"status":"development_complete","selected":selection}
                results=[]
                for seed in MAIN_SEEDS:
                    for method in ("mc_replay","td_replay","td_fd"):
                        cfg=replace(recipe,fd_strength=selection["fd_strength"] if method=="td_fd" else 0.,
                                    ridge=selection["fd_ridge"] if method=="td_fd" else recipe.ridge)
                        results.append(trajectory(method,cfg,seed,1024,"confirmation",eval_count=8,initial_count=4))
                planned=len(s["completed"])
                s["status"]="complete"
                progress(force=True,final=True)
                result={"status":"complete","selected":selection,"charged_seconds":elapsed(),
                        "trajectories":[r["unit_path"] for r in results],"completed_units":planned,
                        "physical_motion":False,"next_phase_authorized":False}
                parent.complete(result)
                return result
            except BaseException:
                s["status"]="paused"
                progress(force=True,final=True)
                raise
    finally:
        stop.close()
        if continuation is not None:
            continuation.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--model-path",type=Path)
    parser.add_argument("--reuse-profile",type=Path)
    parser.add_argument("--continue-from",type=Path)
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--clear-stop",action="store_true")
    parser.add_argument("--max-hours",type=float,default=13.)
    parser.add_argument("--stage",choices=("software","profile","development","all"),default="all")
    args=parser.parse_args()
    if args.clear_stop:
        (args.output/"STOP").unlink(missing_ok=True)
    kwargs=vars(args)
    kwargs.pop("clear_stop")
    print(json.dumps(run(**kwargs),indent=2),flush=True)


if __name__=="__main__":
    main()
