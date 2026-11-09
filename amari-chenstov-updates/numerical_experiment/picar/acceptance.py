"""Reproducible CPU acceptance plus an optional explicitly requested CUDA check."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

from .config import load_config
from .runtime import sha256_file, source_fingerprint


def run(output: Path, *, resume=False, qwen=False, model_path=None, offline=False):
    from .engine import run_smoke
    from .storage import RunStore
    config = load_config("numerical_experiment/picar/configs/smoke.json")
    sources = source_fingerprint()
    identity = {"config": config.to_dict(), "purpose": "phase0_acceptance", "source_sha256": sources,
                "qwen": qwen}
    with RunStore(output, identity, resume=resume) as store:
        if store.completed:
            return json.loads((output / "summary.json").read_text())
        started = time.monotonic()
        attempt = uuid.uuid4().hex[:12]
        report = output / f"tests-{attempt}.xml"
        log = output / f"tests-{attempt}.log"
        tests = sorted(str(p) for folder in ("test/unit", "test/integration")
                       for p in Path(folder).glob("test_picar*.py"))
        environment = {**os.environ, "MPLCONFIGDIR": str(output.absolute() / "matplotlib"),
                       "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"}
        with log.open("xb") as stream:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests,
                f"--junitxml={report}"], stdout=stream, stderr=subprocess.STDOUT, env=environment)
        if result.returncode:
            raise RuntimeError(f"CPU acceptance failed; inspect {log}")
        suites = ET.parse(report).getroot()
        counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites.iter("testsuite"))
                  for key in ("tests", "failures", "errors", "skipped")}
        paired = run_smoke(config, output / "paired", resume=resume)
        if paired["status"] != "completed":
            return {"status": "paused", "paired": paired}
        gpu = None
        if qwen:
            from .qwen_acceptance import run_acceptance
            gpu = run_acceptance(output / f"qwen-{attempt}", model_path=model_path, offline=offline)
        if source_fingerprint() != sources:
            raise RuntimeError("implementation changed during acceptance; retain this attempt and use a new directory")
        total_seconds = time.monotonic() - started
        child_seconds = sum(json.loads(path.read_text()).get("active_seconds", 0.)
                            for path in (output / "paired").rglob("summary.json"))
        summary = {"phase": 0, "smoke_only": True, "kind": "implementation_acceptance",
            "tests": counts, "test_log": str(log), "test_log_sha256": sha256_file(log),
            "junit_sha256": sha256_file(report), "total_invocation_seconds": total_seconds,
            "active_seconds": max(0., total_seconds - child_seconds - (gpu or {}).get("active_seconds", 0.)),
            "paired_status": paired["status"], "qwen_requested": qwen,
            "qwen_fd_convergence": None if gpu is None else gpu["fd_convergence_pass"]}
        store.checkpoint({"acceptance": summary}, progress={"phase": 0, "status": "done"})
        store.complete(summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qwen", action="store_true", help="also use CUDA; downloads pinned weights by default")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output, resume=args.resume, qwen=args.qwen,
        model_path=args.model_path, offline=args.offline), indent=2))


if __name__ == "__main__":
    main()
