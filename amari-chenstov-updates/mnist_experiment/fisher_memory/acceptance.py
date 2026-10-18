"""Seal Phase 0 readiness from completed smoke artifacts and actual pytest XML."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .artifacts import Store, iter_units
from .config import CONDITIONS, ENVIRONMENTS, ROOT, content_hash, scientific_sources
from .run import budget_lock, file_bytes
from .report import MAX_NOTEBOOK_BYTES


def test_result(path: Path, *, minimum: int, allow_skips=False):
    tree = ET.parse(path).getroot()
    cases = list(tree.iter("testcase"))
    failed = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    if failed or len(cases) - skipped < minimum or (skipped and not allow_skips):
        raise ValueError(f"acceptance tests incomplete or failed: {path}")
    return dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                passed=len(cases) - skipped, skipped=skipped, failed=failed,
                seconds=sum(float(case.get("time", 0)) for case in cases))


def inspect_run(root: Path, source_hash: str, kind: str):
    identity = json.loads((root / "identity.json").read_text())["identity"]
    if identity["source_hash"] != source_hash:
        raise ValueError(f"source changed after acceptance run: {root}")
    provenance = json.loads((root / "provenance.json").read_text())
    if not provenance["config"]["smoke"]:
        raise ValueError("production evidence must not masquerade as smoke acceptance")
    records = list(iter_units(root, full=True))
    if any(record["status"] != "completed" for record in records):
        raise ValueError(f"incomplete acceptance artifact: {root}")
    trajectories = [r for r in records if r["manifest"]["metadata"].get("kind") == "trajectory"]
    summaries = []
    for record in trajectories:
        summary = json.loads((Path(record["path"]) / "summary.json").read_text())
        if summary["nonfinite"] or summary["solver_failures"]:
            raise ValueError("smoke trajectory reported a numerical solver failure")
        summaries.append(summary)
    if kind in ("cpu", "cuda"):
        pairs = {(r["manifest"]["metadata"]["environment"], r["manifest"]["metadata"]["condition"])
                 for r in trajectories}
        if pairs != {(env, condition) for env in ENVIRONMENTS for condition in CONDITIONS}:
            raise ValueError("acceptance must cover all six conditions in all three environments")
        if provenance["device"] != kind:
            raise ValueError("acceptance device mismatch")
    else:
        status = json.loads((root / "status.json").read_text())
        if status.get("confirmation_complete") is not True:
            raise ValueError("tiny staged scheduler did not finish its declared cohort")
    runtime = json.loads((root / "runtime.json").read_text())
    return dict(root=str(root), identity=identity, device=provenance["device"], dtype=provenance["dtype"],
                completed_units=len(records), trajectories=len(trajectories), artifact_bytes=file_bytes(root),
                active_seconds=runtime["cumulative_active_seconds"],
                online_seconds=sum(s["active_seconds"] for s in summaries))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cpu", "cuda", "pipeline", "unit_tests", "cpu_recovery", "cuda_recovery", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to(args.budget_root.resolve()) or args.output.resolve() == args.budget_root.resolve():
        raise ValueError("acceptance output must be a child of the shared budget root")
    source_hash = content_hash(scientific_sources())
    with budget_lock(args.budget_root):
        runs = {kind: inspect_run(getattr(args, kind), source_hash, kind) for kind in ("cpu", "cuda", "pipeline")}
        tests = dict(unit=test_result(args.unit_tests, minimum=100, allow_skips=True),
                     cpu_recovery=test_result(args.cpu_recovery, minimum=24),
                     cuda_recovery=test_result(args.cuda_recovery, minimum=16))
        result = dict(schema_version=1, phase="0", status="ready_for_handoff", source_hash=source_hash,
                      tests=tests, runs=runs, scientific_production_started=False,
                      note="Acceptance and timing only; production needs an explicit compute allocation.")
        with Store(args.output, dict(source_hash=source_hash, acceptance_hash=content_hash(result)),
                   external_bytes=file_bytes(args.budget_root, exclude=args.output)
                   + file_bytes(ROOT / "cache/mnist_experiment/datasets") + MAX_NOTEBOOK_BYTES,
                   free_reserve_bytes=100_000_000) as store:
            store.begin("phase0-acceptance", dict(kind="acceptance", schema_version=1))
            store.write_json("phase0-acceptance", "acceptance.json", result)
            store.complete("phase0-acceptance")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
