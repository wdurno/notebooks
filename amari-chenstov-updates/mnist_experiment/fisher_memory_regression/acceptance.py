"""Seal measured Phase 8A acceptance and publish the pending research scaffold.

This reads test/smoke evidence; it never trains or schedules production work.
Smoke runtime is charged here exactly once to the shared Phase 8 allocation.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from mnist_experiment.fisher_memory.artifacts import Store, iter_units
from mnist_experiment.fisher_memory.run import budget_lock

from . import PROTOCOL
from .config import ROOT, content_hash
from .report import publish
from .runtime import RuntimeBudget, file_bytes, file_digest, read_json, scientific_sources


def test_evidence(path):
    document = ET.parse(path).getroot()
    suites = [document] if document.tag == "testsuite" else list(document.iter("testsuite"))
    if not suites:
        raise ValueError("acceptance requires JUnit test suites")
    totals = {name: sum(int(s.get(name, 0)) for s in suites)
              for name in ("tests", "failures", "errors", "skipped")}
    if not totals["tests"] or any(totals[name] for name in ("failures", "errors", "skipped")):
        raise ValueError("acceptance tests must pass without skipped coverage")
    duration = sum(float(s.get("time", 0)) for s in suites)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("invalid test duration")
    return dict(path=str(path.relative_to(ROOT)), sha256=file_digest(path), **totals,
                process_seconds=duration + 2.,
                accounting="JUnit suite wall time plus conservative 2-second process overhead")


def smoke_evidence(path, expected_source_hash, check):
    identity = read_json(path / "identity.json")["identity"]
    if identity.get("role") != "smoke" or identity.get("source_hash") != expected_source_hash:
        raise ValueError("acceptance needs a completed smoke of the current scientific source")
    cfg = read_json(path / "resolved_config.json")
    if not cfg["smoke"] or cfg["novel_count"] > 5:
        raise ValueError("Phase 8A accepts tiny smokes only")
    units = list(iter_units(path, full=True, check=check))
    if any(row["status"] != "completed" for row in units):
        raise ValueError("smoke contains unfinished units")
    required = {"decision-calibration", "decision-allocation-selection", "decision-confirmation-manifest",
                "decision-confirmation-complete", "decision-analysis"}
    if not required <= {row["unit_id"] for row in units}:
        raise ValueError("acceptance requires every scheduler stage")
    analysis = path / "units/decision-analysis/result.json"
    if read_json(analysis)["status"] != "completed":
        raise ValueError("smoke must exercise final analysis")
    return dict(path=str(path.relative_to(ROOT)), device=cfg["device"], dtype=cfg["dtype"],
                scientific_source_hash=expected_source_hash, units=len(units),
                novel_count=cfg["novel_count"], runtime_seconds=read_json(
                    path / "runtime.json")["cumulative_active_seconds"], bytes=file_bytes(path, check=check),
                seals={row["unit_id"]: row["completion"]["seal_hash"] for row in units})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, action="append", required=True)
    parser.add_argument("--superseded-smoke-root", type=Path, action="append", default=[])
    parser.add_argument("--test-results", type=Path, action="append", required=True)
    parser.add_argument("--preliminary-test-seconds", type=float, required=True)
    parser.add_argument("--notebook", type=Path, default=ROOT / "mnist_experiment/fisher_memory_results.ipynb")
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.preliminary_test_seconds) or args.preliminary_test_seconds < 0:
        raise ValueError("invalid preliminary-test accounting")
    args.root, args.budget_root = args.root.resolve(), args.budget_root.resolve()
    if args.root.parent != args.budget_root:
        raise ValueError("acceptance ledger must be a direct child of the shared budget root")
    sources = scientific_sources()
    source_hash = content_hash(sources)
    inputs = dict(smokes=[str(path.resolve()) for path in args.smoke_root],
        superseded=[str(path.resolve()) for path in args.superseded_smoke_root],
        tests=[str(path.resolve()) for path in args.test_results], prior=args.preliminary_test_seconds)
    identity = dict(protocol=PROTOCOL, allocation_family="plan15-phase8", role="acceptance",
        source_hash=source_hash, inputs_hash=content_hash(inputs), recorder_hash=file_digest(Path(__file__)))
    with budget_lock(args.budget_root):
        with Store(args.root, identity, resume=args.resume, external_bytes=file_bytes(
                args.budget_root, exclude=args.root), finalization_reserve_bytes=32_000_000) as store:
            budget = RuntimeBudget(store)
            try:
                with store.control.signal_handlers():
                    if store.is_complete("acceptance"):
                        result = read_json(store.unit_path("acceptance") / "result.json")
                    else:
                        tests = [test_evidence(path.resolve()) for path in args.test_results]
                        smokes = [smoke_evidence(path.resolve(), source_hash, budget.check)
                                  for path in args.smoke_root]
                        if {row["device"] for row in smokes} != {"cpu", "cuda"}:
                            raise ValueError("both CPU and CUDA acceptance are required")
                        prior = [dict(path=str(path.resolve().relative_to(ROOT)), runtime_seconds=read_json(
                            path / "runtime.json")["cumulative_active_seconds"]) for path in args.superseded_smoke_root]
                        charged = (args.preliminary_test_seconds + sum(row["process_seconds"] for row in tests)
                                   + sum(row["runtime_seconds"] for row in smokes + prior))
                        # Durable restore is idempotent if acceptance was interrupted before its seal.
                        store.control.restore_active_seconds(charged)
                        budget.pulse(force=True)
                        result = dict(schema_version=1, protocol=PROTOCOL, status="8A complete",
                            sources=sources, source_hash=source_hash, tests=tests, smokes=smokes,
                            superseded_smokes=prior, preliminary_test_seconds=args.preliminary_test_seconds,
                            charged_evidence_seconds=charged, total_tests=sum(row["tests"] for row in tests),
                            production_started=False,
                            limitation="Tiny smokes establish software recovery, not scientific effectiveness or production throughput.")
                        store.begin("acceptance", dict(kind="phase8_acceptance", schema_version=1))
                        store.write_json("acceptance", "result.json", result)
                        store.complete("acceptance")
                    snapshot = dict(schema_version=1, protocol=PROTOCOL, smoke=True,
                        status="8A complete; 8B--8E pending", stage="8A complete", revision=source_hash[:12],
                        acceptance=result, progress=dict(completed=1, planned=1, active_seconds=budget.used,
                            allocation_seconds=72000, storage=store.storage_status()),
                        calibration=None, allocation=[], confirmation=None, analysis=None, records=[], changes=[])
                    store.write_json(None, "report_snapshot.json", snapshot)
                    publish(snapshot, args.notebook, store=store)
                    print(json.dumps(dict(status=result["status"], tests=result["total_tests"],
                                          charged_seconds=budget.used, source_hash=source_hash)))
            finally:
                budget.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
