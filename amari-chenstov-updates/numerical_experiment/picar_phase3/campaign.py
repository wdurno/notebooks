"""Run bounded Plan 19 development only; never automatically enter Phase 4."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar.runtime import StopController, check_storage, sha256_file
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_phase2.campaign import PROJECT, audit_initial, load_completed_state, read_json
from .analysis import candidate, select_control, selection, forecast
from .config import (VERSION, MAX_HOURS, MODEL_SEED, BOOTSTRAP_SEED, DEVELOPMENT_SEEDS,
    NORMALIZATION, Recipe, configuration, simulation, panels, sources, recipe_dict, likelihood_key)


def learning_metadata(learner):
    import torch
    return {"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
        "device":str(learner.device),"dtype":str(learner.dtype),
        "device_name":torch.cuda.get_device_name(learner.device) if learner.device.type=="cuda" else "CPU",
        "parameter_counts":{bank:sum(learner.params[name].numel() for name in names)
                            for bank,names in learner.banks.items()},
        "model_provenance":learner.model.encoder.provenance}


def run(output, teacher_source, prior_review, *, model_path=None, offline=False,
        resume=False, max_hours=MAX_HOURS, stage_limit="all"):
    if stage_limit not in {"software","all"}:
        raise ValueError("stage must be software or all")
    if not 0 < max_hours <= MAX_HOURS:
        raise ValueError("Phase 3 has a six-hour maximum, not a renewed whole-study allocation")
    root, teacher_source, prior_review = [Path(p).resolve() for p in (output,teacher_source,prior_review)]
    from numerical_experiment.picar.report import read_completed_unit
    previous = read_completed_unit(prior_review)["summary"]
    prior_seconds = previous["study_charged_seconds"]
    scientific_sources = sources()
    identity = {"schema_version":1,"version":VERSION,"phase":3,"max_hours":max_hours,
        "source_sha256":scientific_sources,"teacher_source":str(teacher_source),
        "teacher_completion_sha256":sha256_file(teacher_source/"COMPLETED"),
        "prior_review":str(prior_review),"prior_completion_sha256":sha256_file(prior_review/"COMPLETED"),
        "prior_study_seconds":prior_seconds,"model_seed":MODEL_SEED,
        "model_source":str(Path(model_path).resolve()) if model_path else configuration(Recipe("identity")).model.model_id,
        "normalization":NORMALIZATION,"development_seeds":DEVELOPMENT_SEEDS,
        "maximum_repair_rounds":2,"next_phase_authorized":False,
        "pre_execution_validation_reserve_seconds":420.,"closeout_reserve_seconds":300.}
    root.mkdir(parents=True,exist_ok=True)
    stop = StopController(root)
    stop.install()
    os.environ.setdefault("MPLBACKEND","Agg")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    try:
        with RunStore(root/"orchestration",identity,resume=resume,checkpoint_policy="journal-v1") as parent:
            if parent.completed:
                return read_json(parent.root/"summary.json")
            saved = parent.load_checkpoint(trusted=True) or {"charged_seconds":420.}
            started,last_save,last_report = time.monotonic(),0.,0.
            count = sum(1 for _ in (root/"units").glob("*/COMPLETED"))
            active,stage = None,"development acceptance"

            def elapsed():
                return saved["charged_seconds"]+time.monotonic()-started

            def refresh():
                from numerical_experiment.picar.report import build_notebook
                build_notebook(PROJECT/"cache/numerical_experiment/picar")

            def progress(label=None,*,force=False):
                nonlocal last_save,last_report,stage
                stage = label or stage
                now = time.monotonic()
                if force or now-last_save >= 15 or stop.check():
                    child = read_json(active/"progress.json") if active and (active/"progress.json").exists() else {}
                    parent.checkpoint({"charged_seconds":elapsed()},progress={"phase":3,"status":stage,
                        "phase3_active_seconds":elapsed(),"cumulative_study_seconds":prior_seconds+elapsed(),
                        "remaining_budget_seconds":max(0.,max_hours*3600-elapsed()),
                        "completed_trajectories":sum(1 for _ in (root/"units").glob("pilot-*/COMPLETED")),
                        "counts":{"step":count,"target":19},"active_unit":None if active is None else str(active),
                        "active_unit_progress":child})
                    last_save = now
                if force or now-last_report >= 900:
                    refresh()
                    last_report = now
                if stop.check():
                    raise InterruptedError("Phase 3 checkpointed; remove STOP before intentional --resume")
                if elapsed() >= max_hours*3600-390 or prior_seconds+elapsed() >= 72000-390:
                    raise TimeoutError("Phase 3/study compute ceiling reached; preserve partial artifacts")

            def unit(name,settings,callback):
                nonlocal active,count
                config = {**identity,"operation":name,"settings":settings}
                suffix = hashlib.sha256(canonical_json(config).encode()).hexdigest()[:12]
                path = root/"units"/f"{name}-{suffix}"
                active = path
                progress(name,force=True)
                with RunStore(path,config,resume=resume,checkpoint_policy="journal-v1") as store:
                    if store.completed:
                        active = None
                        return read_json(path/"summary.json")
                    tick = time.monotonic()
                    result = callback(store)
                    if sources() != scientific_sources:
                        raise RuntimeError("scientific source changed; preserve the run and version the correction")
                    result = {**result,"schema_version":1,"phase":3,"kind":f"phase3_{name}",
                        "smoke_only":True,"development_only":True,"settings":settings,
                        "unit_path":str(path),"active_seconds":time.monotonic()-tick,
                        "media_sha256":{str(p.relative_to(path)):sha256_file(p) for p in path.rglob("*")
                                        if p.suffix in {".png",".mp4"}}}
                    if store._current is None:
                        store.checkpoint({"stage":"done"},progress={"phase":3,"status":name})
                    store.complete(result)
                count += 1
                active = None
                print(json.dumps({"unit":name,"settings":settings,"seconds":result["active_seconds"],
                    "path":str(path)}),flush=True)
                progress(name+" complete",force=True)
                return result

            def provenance(store):
                path = store.root/"sources.tar.gz"
                with tarfile.open(path,"w:gz") as archive:
                    for name in scientific_sources:
                        archive.add(PROJECT/name,arcname=name)
                return {"source_sha256":scientific_sources,"archive_sha256":sha256_file(path)}

            def software(store):
                env = {**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
                result = subprocess.run([sys.executable,"-m","pytest","-q",
                    "test/unit/test_picar_phase3_calibration.py","test/unit/test_picar_phase3_analysis.py",
                    "test/unit/test_picar_phase3_student.py","test/unit/test_picar_phase3_probes.py",
                    "test/unit/test_picar_phase3_report.py","test/unit/test_picar_report.py"],
                    cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
                (store.root/"pytest.txt").write_text(result.stdout)
                if result.returncode:
                    raise RuntimeError(f"Phase 3 software acceptance failed: {store.root/'pytest.txt'}")
                return {"passed":True,"test_result":result.stdout.strip().splitlines()[-1]}

            try:
                unit("provenance",{},provenance)
                unit("software",{},software)
                if stage_limit == "software":
                    progress("software accepted; GPU work pending",force=True)
                    return {"status":"software_complete","units":count,"phase3_active_seconds":elapsed(),
                            "cumulative_study_seconds":prior_seconds+elapsed()}
                import torch
                from numerical_experiment.picar.models import load_qwen_model
                from numerical_experiment.picar.learner import MemoryLearner
                from numerical_experiment.picar.fix3_student import _batch
                from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
                from numerical_experiment.picar_phase2_rescue.data import restore_demonstrations
                from .calibration import bootstrap_statistics,explore_controls,optimizer_probe,fd_probe
                from .student import DevelopmentRun
                torch.set_num_threads(2)
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable; do not silently substitute a CPU experiment")
                check_storage(root,quota_bytes=100_000_000_000,reserve_bytes=20_000_000_000)
                data = restore_demonstrations(teacher_source)
                calibration = unit("calibration",{"name":"recorded_return_scales"},
                    lambda store:{**bootstrap_statistics(data["records"]),"data_provenance":data["provenance"]})
                base = Recipe("common",temperature=calibration["selected_temperature"],
                              action_variance=calibration["action_variance"])
                model,current_key = None,None
                initials,assets = {},{}

                def ensure_model(recipe):
                    nonlocal model,current_key
                    key = likelihood_key(recipe)
                    if model is not None and current_key != key:
                        del model
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = None
                    if model is None:
                        kwargs = {} if model_path is None else {"model_path":model_path}
                        raw = load_qwen_model(**kwargs,local_files_only=offline,seed=MODEL_SEED,
                            top_layers=1,ltm_rank=4,stm_rank=1,hidden_size=64,context_length=4,
                            return_bins=161,min_pixels=56*56,max_pixels=19200,max_tokens=2048).eval()
                        raw.encoder.enable_prefix_cache = True
                        model = NormalizedStudent(raw,mode=NORMALIZATION,eps=1e-6).eval()
                        current_key = key
                    return model

                def initial_for(recipe):
                    key = likelihood_key(recipe)
                    if key not in initials:
                        def initialize(store):
                            runner = DevelopmentRun(ensure_model(recipe),configuration(recipe,bootstrap=True),
                                simulation(recipe,bootstrap=True),store,progress,mode="bootstrap",
                                seed=BOOTSTRAP_SEED,max_transitions=600)
                            if runner.s["stage"] == "collect":
                                runner.s.update(eligible=copy.deepcopy(data["records"]),retired=[],ready=[],
                                    acquired_count=data["acquired_count"],episodes=copy.deepcopy(data["episodes"]),
                                    initial_data_provenance=data["provenance"],stage="distill")
                                runner.save()
                            return {**runner.run(),"data_provenance":data["provenance"],
                                "normalization":NORMALIZATION,"old_fisher_reused":False,
                                "new_teacher_transitions":0,"execution":learning_metadata(runner.learner)}
                        assets[key] = unit("bootstrap",{"name":"common_initialization","likelihood_key":key},initialize)
                        initials[key] = load_completed_state(assets[key]["unit_path"])
                        audit_initial(initials[key],configuration(recipe,bootstrap=True))
                    return initials[key]

                initial = initial_for(base)

                def shared_probe(store):
                    current = ensure_model(base)
                    result = optimizer_probe(current,configuration(base),initial)
                    learner = MemoryLearner(current,configuration(base).learning,configuration(base).seeds(0))
                    learner.load_state_dict(initial["learner"])
                    means = [learner.choose_action(_batch(r,initial["eligible"]).histories[0],explore=False)["logits"].tolist()
                             for r in initial["eligible"]]
                    return {**result,**explore_controls(means,simulation(base).calibration)}
                shared = unit("probe",{"name":"common_optimizer_and_motion"},shared_probe)
                common = replace(base,learning_rate=shared["selected_learning_rate"],
                    optimizer_steps=shared["selected_optimizer_steps"],exploration_std=shared["selected_exploration_std"])
                pilots,recipes,initial_evaluations,skipped = [],{},{},[]

                def pilot(recipe,seed,*,require_admission=True):
                    settings = {**recipe_dict(recipe),"seed":seed}
                    running = any(read_json(p).get("settings")==settings for p in (root/"units").glob("pilot-*/config.json"))
                    if require_admission and not running:
                        prior = [p for p in pilots if bool(p["settings"]["fd_strength"])==bool(recipe.fd_strength)]
                        estimate = max((p["timings"]["active_seconds"] for p in prior),default=3400. if recipe.fd_strength else 2800.)
                        if max_hours*3600-elapsed() < 1.2*estimate+600:
                            skipped.append({"settings":settings,"reason":"insufficient remaining admission allowance",
                                "remaining_seconds":max_hours*3600-elapsed(),"estimated_seconds":estimate})
                            return None
                    state = initial_for(recipe)
                    key = likelihood_key(recipe)
                    def learn(store):
                        runner = DevelopmentRun(ensure_model(recipe),configuration(recipe),simulation(recipe),
                            store,progress,mode="pilot",seed=seed,max_transitions=recipe.transitions,
                            initial=state,evaluations=panels(recipe),
                            initial_evaluations=initial_evaluations.get(key),
                            initial_asset_id=sha256_file(Path(assets[key]["unit_path"])/"COMPLETED"))
                        return {**runner.run(),"execution":learning_metadata(runner.learner)}
                    result = unit("pilot",settings,learn)
                    pilots.append(result)
                    recipes[recipe.name] = recipe
                    initial_evaluations.setdefault(key,[p for p in result["evaluations"] if p["checkpoint"]=="initial"])
                    return result

                for ridge in (.1,.03,.3):
                    recipe = replace(common,name=f"control_ridge_{ridge:g}",ridge=ridge)
                    if pilot(recipe,DEVELOPMENT_SEEDS[0]) is None:
                        break
                if not pilots:
                    raise RuntimeError("budget could not admit a development trajectory")
                chosen,_ = select_control(pilots)
                control = recipes[chosen["name"]]
                repairs = []
                if chosen["new_gain_endpoint"] < .1-1e-12:
                    repair = replace(control,name="common_plasticity_repair",pi=.05,
                        optimizer_steps=min(4,max(2,2*control.optimizer_steps)))
                    repaired = pilot(repair,DEVELOPMENT_SEEDS[0])
                    if repaired is not None:
                        repairs.append({"round":1,"purpose":"Shared plasticity/solve check after absent acquisition",
                                        "recipe":recipe_dict(repair),"source":repaired["unit_path"]})
                        chosen,_ = select_control(pilots)
                        control = recipes[chosen["name"]]
                pilot(control,DEVELOPMENT_SEEDS[1])
                state = initial_for(control)
                numerical = unit("probe",{"name":"fd_resolution_and_cost","recipe":recipe_dict(control)},
                    lambda store:fd_probe(ensure_model(control),configuration(control),state))
                if numerical["fd_convergence_pass"]:
                    strength = numerical["fd_strength_rows"][1]["fd_strength"]
                    fd = replace(control,name="fd_ratio_01",fd_strength=strength,
                                 fd_step=numerical["selected_fd_step"])
                    first = pilot(fd,DEVELOPMENT_SEEDS[0])
                    if first is not None:
                        pilot(fd,DEVELOPMENT_SEEDS[1])
                        current = selection(pilots,control_name=control.name)
                        relevant = [r for r in current["comparisons"] if r["name"]==fd.name]
                        if relevant and not all(r["retention_matched"] for r in relevant) and len(repairs)<2:
                            mean_gap = sum(r["retention_endpoint_difference"] for r in relevant)/len(relevant)
                            index = 2 if mean_gap > 0 else 0
                            alternate = replace(control,name=f"fd_ratio_{numerical['fd_strength_rows'][index]['target_gradient_ratio']:g}",
                                fd_strength=numerical["fd_strength_rows"][index]["fd_strength"],fd_step=fd.fd_step)
                            result = pilot(alternate,DEVELOPMENT_SEEDS[0])
                            if result is not None:
                                repairs.append({"round":len(repairs)+1,"purpose":"Retention-qualified FD strength adjustment",
                                    "recipe":recipe_dict(alternate),"source":result["unit_path"]})
                decision = selection(pilots,control_name=control.name)
                decision.update(repair_rounds=repairs,fd_convergence_pass=numerical["fd_convergence_pass"],
                    budget_skips=skipped,
                    selected_control_recipe=recipe_dict(control),
                    selected_fd_recipe=None if decision["selected_fd"] is None else recipe_dict(recipes[decision["selected_fd"]]),
                    confirmation_proposal={"pairs":4,"transitions_per_condition":1200,"authorized":False},
                    normalization=NORMALIZATION,return_design={"bins":161,"range":[-6.,12.],"temperature":control.temperature,
                    "action_variance":control.action_variance},capability_target=.7,
                    capability_target_status="Original proposed target, not demonstrated reachable; sample crossing remains unsupported unless fresh data attain it.")
                selected = unit("selection",{},lambda store:decision)
                measured = forecast(pilots,decision,phase3_seconds=elapsed()+300.,prior_study_seconds=prior_seconds)
                measured["closeout_reserve_seconds"] = 300.
                resource = unit("forecast",{},lambda store:measured)
                result = {"schema_version":1,"phase":3,"kind":"phase3_campaign","smoke_only":True,
                    "status":"ready_for_checkin","total_active_seconds":elapsed()+300.,
                    "phase3_active_seconds":elapsed()+300.,"cumulative_study_seconds":prior_seconds+elapsed()+300.,
                    "active_seconds":0.,"units":count,"pilot_paths":[p["unit_path"] for p in pilots],
                    "selection":selected,"forecast":resource,"next_phase_authorized":False}
                parent.checkpoint({"charged_seconds":elapsed()+300.},progress={"phase":3,
                    "status":"Phase 3 complete; development check-in, no automatic Phase 4",
                    "phase3_active_seconds":elapsed()+300.,"cumulative_study_seconds":prior_seconds+elapsed()+300.,
                    "counts":{"step":count,"target":count}})
                parent.complete(result)
                refresh()
                return result
            except BaseException:
                parent.checkpoint({"charged_seconds":elapsed()},progress={"phase":3,
                    "status":"checkpointed interruption; explicit resume required","phase3_active_seconds":elapsed(),
                    "cumulative_study_seconds":prior_seconds+elapsed(),"counts":{"step":count,"target":19}})
                refresh()
                raise
    finally:
        stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("cache/numerical_experiment/picar/phase3/development-v1"))
    parser.add_argument("--teacher-source",type=Path,required=True)
    parser.add_argument("--prior-review",type=Path,required=True)
    parser.add_argument("--model-path",type=Path)
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stage",choices=("software","all"),default="all")
    args = parser.parse_args()
    result = run(args.output,args.teacher_source,args.prior_review,model_path=args.model_path,
                 offline=args.offline,resume=args.resume,stage_limit=args.stage)
    print(json.dumps({k:result.get(k) for k in ("status","units","phase3_active_seconds","cumulative_study_seconds")},indent=2))


if __name__ == "__main__":
    main()
