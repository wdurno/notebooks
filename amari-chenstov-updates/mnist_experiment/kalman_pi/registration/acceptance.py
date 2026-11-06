"""Phase 6 acceptance, completed-artifact audit and bounded offline diagnostics."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan17-registration-mpl")

from mnist_experiment.diagonal_lfu.report import _atomic, _json_bytes
from mnist_experiment.fisher_memory.control import utc_now

from .analysis import summarize
from .config import Config
from .report import build_report
from .runtime import ACCEPTANCE_LEDGER, FAMILY, REPO, file_digest, historical, phase_seconds, read_json


TESTS = (
    "test/unit/test_kalman_pi_registration_model_data.py",
    "test/unit/test_kalman_pi_registration_learner.py",
    "test/unit/test_kalman_pi_registration_runtime.py",
    "test/unit/test_kalman_pi_registration_analysis_report.py",
    "test/unit/test_kalman_pi_registration_report_guard.py",
    "test/unit/test_kalman_pi_registration_acceptance.py",
    "test/integration/test_kalman_pi_registration_resume.py",
)


def _payload(root, unit_id):
    import torch
    from mnist_experiment.fisher_memory.artifacts import SCHEMA, _hash_json
    path = Path(root) / "units" / unit_id
    current = read_json(path / "latest.json")["current"]
    checkpoint = path / current["file"]
    if file_digest(checkpoint) != current["sha256"]:
        raise ValueError("corrupt diagnostic checkpoint")
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (envelope["schema"] != SCHEMA or envelope["unit_id"] != unit_id
            or envelope["identity_hash"] != read_json(Path(root) / "identity.json")["identity_hash"]
            or envelope["manifest_hash"] != _hash_json(read_json(path / "manifest.json"))):
        raise ValueError("diagnostic checkpoint identity mismatch")
    return envelope["payload"]


def noise_comparison(scores, full_variance, diagonal, cfg):
    import torch
    scores, full_variance, diagonal = scores.double(), full_variance.double(), diagonal.double()
    if scores.ndim != 2 or len(scores) < 2 or scores.shape[1:] != full_variance.shape or diagonal.shape != full_variance.shape:
        raise ValueError("nested audit requires matching per-example scores and diagonals")
    nested = scores.var(0, unbiased=True)
    floor = max(cfg.observer_floor_absolute, cfg.observer_floor_relative * float(diagonal.mean()))
    weights = diagonal / diagonal.clamp_min(floor).square()
    if (not torch.isfinite(scores).all() or not torch.isfinite(full_variance).all()
            or not torch.isfinite(diagonal).all() or (full_variance < 0).any() or (diagonal < 0).any()):
        raise ValueError("invalid noise audit moments")
    def ratio(a, b):
        return float(a / b) if float(b) > 0 else None
    return {"nested_count": len(scores), "raw_trace_ratio": ratio(nested.sum(), full_variance.sum()),
            "fisher_weighted_noise_ratio": ratio((weights * nested).sum(), (weights * full_variance).sum()),
            "diagonal_relative_l2": ratio((nested - full_variance).norm(), full_variance.norm()),
            "reference_is_truth": False, "additional_training_labels": 0}


def closeout(root, data_root):
    from mnist_experiment.fisher_memory.artifacts import iter_units
    from .runtime import scientific_sources
    from .report import _validated
    root = Path(root).resolve()
    identity = read_json(root / "identity.json")
    cfg = Config.from_mapping(identity["identity"]["config"])
    if not (root / "study_completed.json").is_file():
        raise ValueError("closeout requires a completed study")
    with (FAMILY / ".allocation.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        own, peers = historical._charged_seconds(root), phase_seconds(root)
        if own + peers + 300 > 72000:
            raise RuntimeError("Phase 6 allocation cannot admit closeout reserve")
        ledger = read_json(ACCEPTANCE_LEDGER)
        entry = {"id": "phase6E-closeout-" + uuid.uuid4().hex[:12], "seconds": 300.,
                 "status": "running", "started_at": utc_now(),
                 "description": "Full artifact audit, nested already-seen-label noise check and final publication"}
        ledger["entries"].append(entry)
        _atomic(ACCEPTANCE_LEDGER, _json_bytes(ledger))
        started, success = time.monotonic(), False
        def check():
            if time.monotonic() - started > 300:
                raise RuntimeError("bounded closeout reserve exhausted")
        try:
            snapshot, result, _ = _validated(root)
            units = list(iter_units(root, full=True, check=check))
            if snapshot["status"] != "completed" or any(u["status"] != "completed" for u in units):
                raise ValueError("closeout cannot conceal incomplete work")
            if identity["identity"]["sources"] != scientific_sources(check):
                raise ValueError("scientific sources changed")
            selected = read_json(root / "units/decision-selection/decision.json")
            if snapshot["selection"] != selected or read_json(root / "selection.json") != selected:
                raise ValueError("frozen selection changed")
            paths = {t["unit_id"]: t for t in snapshot["trajectories"]}
            for unit in units:
                if unit["metadata"].get("kind") != "trajectory":
                    continue
                uid = unit["unit_id"]
                expected = {"unit_id": uid, **unit["metadata"], "completed": True,
                    "metrics": read_json(root / "units" / uid / "metrics.json"),
                    "summary": read_json(root / "units" / uid / "summary.json")}
                if paths.pop(uid, None) != expected:
                    raise ValueError("snapshot differs from sealed trajectory")
            if paths or result["counts"]["incomplete"]:
                raise ValueError("snapshot contains unsealed trajectories")
            diagnostics = read_json(root / "units/diagnostics-score-and-solver/summary.json")
            diagnostics.update(shadow_ranking=selected["shadow_ranking"], retention_candidates=selected["retention_candidates"])
            producer = file_digest(Path(__file__))
            audit_path = root / ("noise_calibration_audit-" + producer[:12] + ".json")
            if audit_path.exists():
                noise = read_json(audit_path)
                if noise["identity_hash"] != identity["identity_hash"] or noise["producer_sha256"] != producer:
                    raise ValueError("noise audit identity changed")
            else:
                import torch
                from .adapters import BatchReader
                from .data import load_cached
                from .model import build_model, observed_scores
                historical.setup(cfg)
                raw = load_cached(data_root)
                model, layout = build_model(0, device=cfg.device, dtype=getattr(torch, cfg.dtype))
                rows = []
                for replica in range(cfg.dev_replicas):
                    check()
                    uid = f"assets-development-r{replica:03}-a{selected['angle_max']:g}-h{selected['hold_factor']:g}"
                    assets = _payload(root, uid)
                    initial = _payload(root, f"initial-development-r{replica:03}")
                    calibration = _payload(root, f"calibration-development-r{replica:03}")
                    ids = assets["panels"]["calibration"]["ids"]
                    if not torch.isin(ids, assets["panels"]["init"]["ids"]).all():
                        raise ValueError("noise audit would introduce unseen labels")
                    count = max(2, cfg.calibration_size // 2)
                    values = BatchReader(cfg, assets, raw, check)("calibration", torch.arange(count))
                    scores = observed_scores(model, layout, initial["theta"].to(cfg.device),
                        values["inputs"], values["targets"], chunk_size=cfg.chunk_size, check=check).cpu()
                    rows.append({"replica": replica, "full_count": calibration["count"],
                        **noise_comparison(scores, calibration["variance"], initial["diagonal"], cfg)})
                noise = {"identity_hash": identity["identity_hash"], "producer_sha256": producer,
                    "kind": "post-run-nested-noise-audit", "rows": rows,
                    "interpretation": "Nested subset versus correlated full calibration, not a truth estimate or policy retuning."}
                _atomic(audit_path, _json_bytes(noise))
            diagnostics["nested_noise_audit"] = noise
            audit = {"status": "passed", "identity_hash": identity["identity_hash"],
                "producer_sha256": producer, "sealed_units": len(units),
                "scientific_source_files": len(identity["identity"]["sources"]),
                "trajectory_counts": result["counts"], "checked_at": utc_now()}
            diagnostics["final_artifact_audit"] = audit
            snapshot["diagnostics"].update(diagnostics)
            progress = snapshot["progress"]
            hours = (own + peers + time.monotonic() - started) / 3600
            progress.update(heartbeat=utc_now(), active_hours=hours, phase6_active_hours=hours,
                remaining_hours=20 - hours, total_family_active_hours=hours + progress["historical_active_hours"])
            _atomic(root / "progress.json", _json_bytes(progress))
            _atomic(root / "report_snapshot.json", _json_bytes(snapshot))
            manifest = build_report(root, Path(read_json(root / "report/latest.json")["notebook"]))
            _atomic(root / (entry["id"] + ".json"), _json_bytes({**audit, "report_dir": manifest["report_dir"]}))
            success = True
            print(json.dumps({**audit, "noise_audit": noise["rows"], "report_dir": manifest["report_dir"]}), flush=True)
        finally:
            entry.update(seconds=time.monotonic() - started, status="completed" if success else "failed", completed_at=utc_now())
            _atomic(ACCEPTANCE_LEDGER, _json_bytes(ledger))
    return 0


def publish_pending(root, cfg, *, historical_root, status="awaiting_gpu", test_result=None):
    root = Path(root).resolve()
    if (root / "identity.json").exists():
        raise ValueError("pending publication cannot replace an initialized scientific study")
    latest = read_json(Path(historical_root) / "report/latest.json")
    directory = Path(latest["report_dir"]).resolve()
    if not (directory / "COMPLETED").is_file():
        raise ValueError("historical report must be sealed")
    phase_time = phase_seconds(root)
    historical_time = historical.peer_seconds(root)
    progress = {"status": status, "phase": "6A: CPU acceptance complete; awaiting CUDA acceptance",
        "heartbeat": utc_now(), "revision": f"{cfg.protocol}/{root.name}",
        "completed_trajectories": 0, "completed_fresh": 0, "planned_fresh": None,
        "completed_blocks": 0, "planned_blocks": None, "active_hours": phase_time / 3600,
        "phase6_active_hours": phase_time / 3600, "historical_active_hours": historical_time / 3600,
        "total_family_active_hours": (phase_time + historical_time) / 3600,
        "allocation_hours": 20., "remaining_hours": max(0., 20 - phase_time / 3600),
        "forecast_seconds": None, "acceptance": test_result or {},
        "reason": "User requested a hold at Phase 6A until the NVIDIA driver is restored."}
    config = {**cfg.mapping(), "bootstrap_seed": cfg.seed}
    snapshot = {"schema": 1, "kind": "plan17-registration-snapshot", "study_id": root.name,
        "status": status, "phase": progress["phase"], "config": config, "progress": progress,
        "selection": {}, "trajectories": [], "readiness": {},
        "diagnostics": {"cpu_acceptance": test_result or {}, "cuda_acceptance": "pending",
                        "production_started": False},
        "analysis": summarize([], {}, bootstrap_draws=cfg.bootstrap_draws, seed=cfg.seed, config=config),
        "historical_report_dir": str(directory), "historical_notebook_sha256": file_digest(directory / "report.ipynb")}
    root.mkdir(parents=True, exist_ok=True)
    _atomic(root / "progress.json", _json_bytes(progress))
    _atomic(root / "report_snapshot.json", _json_bytes(snapshot))
    return build_report(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=FAMILY / "registration_v1")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/production.json")
    parser.add_argument("--historical-root", type=Path, default=FAMILY / "retention_supplement_v1")
    parser.add_argument("--publish-pending-only", action="store_true")
    parser.add_argument("--closeout", action="store_true")
    parser.add_argument("--data-root", type=Path, default=REPO / "cache/mnist_experiment/datasets")
    args = parser.parse_args(argv)
    if args.closeout:
        if args.publish_pending_only:
            parser.error("closeout and pending publication are mutually exclusive")
        return closeout(args.root, args.data_root)
    FAMILY.mkdir(parents=True, exist_ok=True)
    with (FAMILY / ".allocation.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = read_json(ACCEPTANCE_LEDGER) if ACCEPTANCE_LEDGER.exists() else {"schema": 1, "entries": []}
        if phase_seconds(args.root) + 600 > 72000:
            raise RuntimeError("Phase 6 allocation cannot admit acceptance reserve")
        entry = {"id": "phase6A-cpu-" + uuid.uuid4().hex[:12], "seconds": 600.,
                 "status": "running", "started_at": utc_now(),
                 "description": "CPU tests and pending report; separately rooted CLI smoke compute excluded"}
        ledger["entries"].append(entry)
        _atomic(ACCEPTANCE_LEDGER, _json_bytes(ledger))
        started, code = time.monotonic(), 0
        previous_snapshot = args.root / "report_snapshot.json"
        previous = read_json(previous_snapshot) if args.publish_pending_only and previous_snapshot.exists() else {}
        result = previous.get("diagnostics", {}).get("cpu_acceptance", {"cpu_tests": "not rerun", "cuda_tests": "pending"})
        try:
            if not args.publish_pending_only:
                command = [sys.executable, "-m", "pytest", "-q", *TESTS]
                process = subprocess.run(command, cwd=REPO, timeout=540, check=False,
                                         text=True, capture_output=True)
                print(process.stdout, end="", flush=True)
                print(process.stderr, end="", file=sys.stderr, flush=True)
                code = process.returncode
                result.update(cpu_tests="passed" if not code else "failed", output=process.stdout[-1200:])
            if not code:
                # Keep a crash reserve until atomic report publication has finished.
                entry.update(seconds=time.monotonic() - started + 60., status="running")
                _atomic(ACCEPTANCE_LEDGER, _json_bytes(ledger))
                manifest = publish_pending(args.root, Config.from_json(args.config),
                    historical_root=args.historical_root, test_result=result)
                print(json.dumps({"notebook": manifest["notebook"], "report_dir": manifest["report_dir"],
                                  "status": "Phase 6A CPU accepted; CUDA acceptance pending"}), flush=True)
        finally:
            entry.update(seconds=time.monotonic() - started, status="completed" if code == 0 else "failed",
                         completed_at=utc_now())
            _atomic(ACCEPTANCE_LEDGER, _json_bytes(ledger))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
