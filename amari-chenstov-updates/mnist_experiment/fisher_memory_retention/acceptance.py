"""Seal CPU/CUDA acceptance and charge test compute to the Phase 8.1 allowance."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.run import budget_lock
from mnist_experiment.fisher_memory_regression.config import ROOT,content_hash
from mnist_experiment.fisher_memory_regression.runtime import RuntimeBudget,file_bytes,file_digest

from . import PROTOCOL,ALLOCATION_FAMILY
from .run import sources


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--junit",type=Path,action="append",required=True)
    parser.add_argument("--extra-seconds",type=float,default=0.)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    checks=[]
    for path in args.junit:
        xml=ET.parse(path).getroot()
        suites=[xml] if xml.tag=="testsuite" else list(xml.findall("testsuite"))
        item=dict(name=path.name,sha256=file_digest(path),xml=path.read_text(),
            tests=sum(int(s.get("tests",0)) for s in suites),
            failures=sum(int(s.get("failures",0))+int(s.get("errors",0)) for s in suites),
            skipped=sum(int(s.get("skipped",0)) for s in suites),
            seconds=sum(float(s.get("time",0)) for s in suites))
        if item["failures"] or item["tests"]==0: raise ValueError("failed or empty acceptance suite")
        checks.append(item)
    if args.extra_seconds<0: raise ValueError("invalid additional test charge")
    code=sources();digest=content_hash(code)
    identity=dict(protocol=PROTOCOL,allocation_family=ALLOCATION_FAMILY,role="acceptance",source_hash=digest,
                  input_hash=content_hash([{k:v for k,v in c.items() if k!="xml"} for c in checks]))
    with budget_lock(args.root.parent):
        external=file_bytes(args.root.parent,exclude=args.root)+file_bytes(ROOT/"cache/mnist_experiment/datasets")
        with Store(args.root,identity,resume=args.resume,external_bytes=external) as store:
            if not store.is_complete("acceptance"):
                store.begin("acceptance",dict(kind="retention_acceptance",source_hash=digest))
                store.write_json("acceptance","result.json",dict(status="passed",source_hash=digest,
                    sources=code,checks=checks,total_tests=sum(c["tests"] for c in checks),
                    charged_seconds=sum(c["seconds"] for c in checks)+args.extra_seconds))
                store.complete("acceptance")
                store.control.restore_active_seconds(store.control.active_seconds+
                    sum(c["seconds"] for c in checks)+args.extra_seconds)
            budget=RuntimeBudget(store)
            budget.close()
    return 0


if __name__=="__main__": raise SystemExit(main())
