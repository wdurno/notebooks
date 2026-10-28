"""Seal Phase 9 CPU/CUDA evidence and charge it once to the shared allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from mnist_experiment.fisher_memory.artifacts import Store, iter_units
from mnist_experiment.fisher_memory.run import budget_lock
from mnist_experiment.fisher_memory_regression.config import ROOT, content_hash
from mnist_experiment.fisher_memory_regression.runtime import (
    RuntimeBudget, read_json, file_digest, file_bytes,
)

from . import PROTOCOL, ALLOCATION_FAMILY
from .run import peer_seconds, sources


def tests(path):
    document = ET.parse(path).getroot()
    suites = [document] if document.tag == "testsuite" else list(document.iter("testsuite"))
    counts = {name: sum(int(s.get(name, 0)) for s in suites)
              for name in ("tests", "errors", "failures", "skipped")}
    if not counts["tests"] or any(counts[k] for k in ("errors", "failures", "skipped")):
        raise ValueError("acceptance requires passed, unskipped selected checks")
    return dict(path=str(path), sha256=file_digest(path), **counts,
                process_seconds=sum(float(s.get("time", 0)) for s in suites) + 2.)


def smoke(path, source_hash, check):
    identity = read_json(path/"identity.json")["identity"]
    if identity.get("role") != "smoke" or identity.get("source_hash") != source_hash:
        raise ValueError("smoke does not validate the current scientific source")
    cfg = read_json(path/"resolved_config.json")
    if not cfg["smoke"] or cfg["novel_count"] > 5:
        raise ValueError("acceptance requires tiny smokes")
    units = list(iter_units(path, full=True, check=check))
    if any(row["status"] != "completed" for row in units):
        raise ValueError("final smoke contains unfinished units")
    names = {row["unit_id"] for row in units}
    required = {"decision-calibration-complete", "decision-selection", "decision-confirmation-manifest",
                "decision-analysis", "decision-closeout"}
    if not required <= names:
        raise ValueError("full scheduler smoke required")
    return dict(path=str(path), device=cfg["device"], dtype=cfg["dtype"], units=len(units),
        runtime_seconds=read_json(path/"runtime.json")["cumulative_active_seconds"],
        seals={row["unit_id"]:row["completion"]["seal_hash"] for row in units})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, default=ROOT/"cache/mnist_experiment/plan15")
    parser.add_argument("--smoke-root", type=Path, action="append", required=True)
    parser.add_argument("--superseded-root", type=Path, action="append", default=[])
    parser.add_argument("--test-results", type=Path, action="append", required=True)
    parser.add_argument("--preliminary-test-seconds", type=float, default=600.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.preliminary_test_seconds < 72000:
        raise ValueError("invalid preliminary-test accounting")
    code = sources()
    source_hash = content_hash(code)
    evidence = [tests(path) for path in args.test_results]
    inputs = dict(tests=evidence, roots=[str(p) for p in args.smoke_root],
                  superseded=[str(p) for p in args.superseded_root],
                  prior_seconds=args.preliminary_test_seconds)
    identity = dict(protocol=PROTOCOL, allocation_family=ALLOCATION_FAMILY, role="acceptance",
                    source_hash=source_hash, inputs_hash=content_hash(inputs))
    if args.root.resolve().parent != args.budget_root.resolve():
        raise ValueError("acceptance must be directly under shared Plan 15 storage")
    with budget_lock(args.budget_root):
        other = peer_seconds(args.budget_root, args.root)
        with Store(args.root, identity, resume=args.resume, external_bytes=file_bytes(
                args.budget_root, exclude=args.root), finalization_reserve_bytes=32_000_000,
                max_active_seconds=max(0.,72000-other)) as store:
            budget = RuntimeBudget(store, other_seconds=other)
            try:
                with store.control.signal_handlers():
                    if not store.is_complete("acceptance"):
                        smokes = [smoke(p, source_hash, budget.check) for p in args.smoke_root]
                        if {s["device"] for s in smokes} != {"cpu", "cuda"}:
                            raise ValueError("both CPU and CUDA smokes required")
                        superseded = [dict(path=str(p), runtime_seconds=read_json(
                            p/"runtime.json")["cumulative_active_seconds"]) for p in args.superseded_root]
                        charged = args.preliminary_test_seconds + sum(v["process_seconds"] for v in evidence)
                        charged += sum(s["runtime_seconds"] for s in smokes + superseded)
                        store.control.restore_active_seconds(charged)
                        budget.check()
                        result = dict(status="passed", protocol=PROTOCOL, source_hash=source_hash,
                            tests=evidence, smokes=smokes, superseded=superseded,
                            total_tests=sum(v["tests"] for v in evidence),
                            charged_evidence_seconds=charged, preliminary_test_seconds=args.preliminary_test_seconds,
                            sources=code, production_started=False)
                        store.begin("acceptance", dict(kind="shift_acceptance", schema_version=1))
                        store.write_json("acceptance", "result.json", result)
                        for index, path in enumerate(args.test_results):
                            store.write_json("acceptance", f"junit-{index}.json",
                                             dict(xml=path.read_text(), sha256=file_digest(path)))
                        store.complete("acceptance")
                    result = read_json(store.unit_path("acceptance")/"result.json")
                    print(json.dumps(dict(status=result["status"], tests=result["total_tests"],
                        charged_seconds=budget.used, source_hash=result["source_hash"])))
            finally:
                budget.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
