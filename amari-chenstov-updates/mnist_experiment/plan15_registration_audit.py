"""Run planned FD optimizer audits, then resume the frozen Phase 10 scheduler.

This operator adds measurement on two existing grid recipes. The registered
learner and its accepted parameter updates remain the accepted source version.
The operator source and audit evidence receive their own immutable provenance.
"""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.control import StopRequested, ResourceLimit
from mnist_experiment.fisher_memory.run import budget_lock
from mnist_experiment.fisher_memory_regression.runtime import RuntimeBudget, read_json, file_digest, file_bytes, setup_torch
from mnist_experiment.fisher_memory_registration.config import Config, TASKS, Recipe, ROOT, content_hash
from mnist_experiment.fisher_memory_registration.data import MNISTSource
from mnist_experiment.fisher_memory_registration.run import peer_seconds, sources
from mnist_experiment.fisher_memory_registration.runner import Runner


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,default=ROOT/"cache/mnist_experiment/datasets")
    parser.add_argument("--notebook",type=Path,default=ROOT/"mnist_experiment/fisher_memory_results.ipynb")
    parser.add_argument("--resume-scheduler",action="store_true")
    args=parser.parse_args(argv)
    cfg=Config.read(args.root/"resolved_config.json")
    identity=read_json(args.root/"identity.json")["identity"]
    if identity["source_hash"]!=content_hash(sources()):
        raise ValueError("frozen scientific source differs; do not bypass recovery identity")
    setup_torch(cfg)
    budget_root=args.root.parent
    with budget_lock(budget_root):
        other=peer_seconds(budget_root,args.root)
        with Store(args.root,identity,resume=True,clear_stop=True,
            external_bytes=file_bytes(budget_root,exclude=args.root)+file_bytes(args.data_root),
            max_bytes=cfg.max_artifact_bytes,free_reserve_bytes=cfg.free_reserve_bytes,
            max_active_seconds=max(0.,72000-other),finalization_reserve_bytes=32_000_000) as store:
            budget=RuntimeBudget(store,other_seconds=other)
            try:
                with store.control.signal_handlers():
                    def decision(name):
                        return read_json(store.unit_path("decision-"+name)/"result.json")
                    init,calibration,readiness=map(decision,("initialization","calibration-complete","readiness"))
                    status=read_json(args.root/"status.json")
                    status.update(status="numerical_audit",stage="10B FD optimizer audit",pid=os.getpid(),process_verified=True)
                    def progress(**updates):
                        status.update(updates,active_seconds=budget.used)
                        store.write_json(None,"status.json",status,finalizing=True)
                    runner=Runner(cfg,store,MNISTSource(args.data_root),budget,on_progress=progress)
                    runner.init_epochs=init["epochs"]
                    runner.channels=tuple(init["channels"])
                    runner.archive_rank=calibration["rank"]
                    runner.blur_max=readiness["blur_max"]
                    for task in TASKS:
                        unit="decision-fd-solver-audit-"+task.key
                        if store.is_complete(unit):
                            continue
                        scale=decision("calibration-"+task.key)["lambda0"]
                        recipe=Recipe.from_config(cfg,"fd_replay_ewc",ridge=math.sqrt(10),strength=scale,
                            fd_step=calibration["fd_step"])
                        record=runner.run("development--"+task.key,0,"development",recipe,capture=True,audit=True)
                        audits=read_json(store.unit_path(record["unit_id"])/"audits.json")
                        failures=[a for a in audits if a["enlarged_failed"] or a["relative_objective_gain"] is None
                                  or a["relative_objective_gain"]>cfg.audit_objective_tolerance]
                        passed=bool(audits) and (cfg.smoke or len(failures)/len(audits)<.25)
                        result=dict(task=task.mapping(),trajectory=record["unit_id"],audits=audits,
                            failures=len(failures),passed=passed,source_hash=identity["source_hash"],
                            operator_source=file_digest(Path(__file__)),operator_path=str(Path(__file__).relative_to(ROOT)),
                            trajectory_seal=read_json(store.unit_path(record["unit_id"])/"COMPLETED")["seal_hash"],
                            reason="Planned early/late doubled-budget optimizer audit; grid recipe reused without changing learning settings")
                        store.begin(unit,dict(kind="registration_fd_optimizer_audit",task=task.key))
                        store.write_json(unit,"result.json",result)
                        store.complete(unit)
                        print(json.dumps(dict(event="phase10_fd_solver_audit",task=task.key,passed=passed,
                            audits=len(audits),failures=len(failures))),flush=True)
                        if not passed:
                            raise ValueError("FD solve audit requires a documented common numerical repair")
                    status.update(status="audit_complete",stage="10B FD optimizer audit passed",current=None)
                    status.setdefault("changes",[]).append("Planned FD optimizer audits completed before policy selection; accepted grid recipes reused.")
                    progress()
            except (StopRequested,ResourceLimit):
                raise
            finally:
                budget.close()
    if args.resume_scheduler:
        os.execv(sys.executable,[sys.executable,"-u","-m","mnist_experiment.fisher_memory_registration.run",
            "--root",str(args.root),"--notebook",str(args.notebook),"--resume","--clear-stop"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
