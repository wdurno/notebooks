"""Seal CPU/CUDA correctness and recovery evidence and charge its compute once."""

import argparse
import json
from pathlib import Path

from mnist_experiment.fisher_memory.artifacts import Store, iter_units
from mnist_experiment.fisher_memory.run import budget_lock
from mnist_experiment.fisher_memory_grid_registration.acceptance import tests
from mnist_experiment.fisher_memory_regression.runtime import RuntimeBudget, read_json, file_bytes
from . import PROTOCOL, ALLOCATION_FAMILY
from .config import ROOT, content_hash
from .run import sources, peer_seconds


def smoke(path, source_hash, check):
    identity = read_json(path/"identity.json")["identity"]
    if identity.get("role") != "smoke" or identity.get("source_hash") != source_hash:
        raise ValueError("smoke does not validate current sources")
    cfg = read_json(path/"resolved_config.json")
    if not cfg["smoke"] or cfg["novel_count"] > 5:
        raise ValueError("tiny smoke required")
    units = list(iter_units(path,full=True,check=check))
    if any(row["status"] != "completed" for row in units):
        raise ValueError("unfinished smoke units")
    required = {"decision-selection","decision-confirmation-manifest","decision-analysis","decision-closeout"}
    if not required <= {u["unit_id"] for u in units}:
        raise ValueError("full scheduler smoke required")
    return dict(path=str(path),device=cfg["device"],units=len(units),
        runtime_seconds=read_json(path/"runtime.json")["cumulative_active_seconds"],
        seals={r["unit_id"]:r["completion"]["seal_hash"] for r in units})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--budget-root",type=Path,default=ROOT/"cache/mnist_experiment/plan15")
    parser.add_argument("--smoke-root",type=Path,action="append",required=True)
    parser.add_argument("--test-results",type=Path,action="append",required=True)
    parser.add_argument("--superseded-root",type=Path,action="append",default=[])
    parser.add_argument("--preliminary-test-seconds",type=float,default=0.)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args(argv)
    code = sources()
    evidence = [tests(p) for p in args.test_results]
    inputs = dict(tests=evidence,smokes=[str(p) for p in args.smoke_root],
        superseded=[str(p) for p in args.superseded_root],prior=args.preliminary_test_seconds)
    identity = dict(protocol=PROTOCOL,allocation_family=ALLOCATION_FAMILY,role="acceptance",
                    source_hash=content_hash(code),inputs_hash=content_hash(inputs))
    if not 0 <= args.preliminary_test_seconds < 72000 or args.root.resolve().parent != args.budget_root.resolve():
        raise ValueError("invalid acceptance budget/root")
    with budget_lock(args.budget_root):
        other = peer_seconds(args.budget_root,args.root)
        with Store(args.root,identity,resume=args.resume,external_bytes=file_bytes(args.budget_root,exclude=args.root),
                   finalization_reserve_bytes=32_000_000,max_active_seconds=max(0.,72000-other)) as store:
            budget = RuntimeBudget(store,other_seconds=other)
            try:
                with store.control.signal_handlers():
                    if not store.is_complete("acceptance"):
                        smokes = [smoke(p,content_hash(code),budget.check) for p in args.smoke_root]
                        if {s["device"] for s in smokes} != {"cpu","cuda"}:
                            raise ValueError("CPU and CUDA smoke required")
                        superseded = [dict(path=str(p),runtime_seconds=read_json(p/"runtime.json")["cumulative_active_seconds"])
                                      for p in args.superseded_root]
                        charged = args.preliminary_test_seconds+sum(r["process_seconds"] for r in evidence)
                        charged += sum(s["runtime_seconds"] for s in smokes+superseded)
                        store.control.restore_active_seconds(charged)
                        budget.check()
                        result = dict(status="passed",protocol=PROTOCOL,source_hash=content_hash(code),
                            tests=evidence,smokes=smokes,superseded=superseded,charged_evidence_seconds=charged,
                            total_tests=sum(r["tests"] for r in evidence),sources=code,production_started=False)
                        store.begin("acceptance",dict(kind="stm_capacity_acceptance",schema_version=1))
                        store.write_json("acceptance","result.json",result)
                        for i,path in enumerate(args.test_results):
                            store.write_json("acceptance",f"junit-{i}.json",dict(xml=path.read_text()))
                        store.complete("acceptance")
                    print(json.dumps(dict(status="passed",charged_seconds=budget.used)),flush=True)
            finally:
                budget.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
