"""Fresh, resumable Plan 15 Phase 5C confirmation of rare-domain acquisition."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

import torch

from .artifacts import Store, iter_units
from .config import ROOT, Config, Treatment, content_hash, provenance
from .control import ResourceLimit, StopRequested
from .data import MNISTSource
from .drip import LEDGER as PILOT_LEDGER, SELECTION, Progress, _policy
from .drip_runner import DripRunner, trajectory_id
from .learner import StopRequested as LearnerStop
from .run import budget_lock, dataset_manifest, external_bytes


PROTOCOL = "plan15-drip-confirmation-v1"
FAMILY = "drip_confirmation"
LEDGER = "drip-confirmation-ledger-v1"
RESULTS = "drip-confirmation-results-v1"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "mnist_experiment/fisher_memory/configs/default.json")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets")
    parser.add_argument("--notebook", type=Path, default=ROOT / "mnist_experiment/fisher_memory_results.ipynb")
    parser.add_argument("--allocation-hours", type=float, required=True,
                        help="cumulative active-time ceiling")
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--smoke-stop-after-step", type=int)
    return parser.parse_args(argv)


def reference_ledger(source_root, cfg, data_hash):
    """Freeze the sealed development decision without altering its source root."""
    identity = json.loads((source_root / "identity.json").read_text())
    old = json.loads((source_root / "provenance.json").read_text())
    if (identity["identity"]["config_hash"] != cfg.scientific_hash
            or old["config_hash"] != cfg.scientific_hash):
        raise ValueError("confirmation configuration differs from development")
    if identity["identity"]["data_hash"] != data_hash:
        raise ValueError("cached dataset differs from development")
    if old["source_hash"] != identity["identity"]["source_hash"]:
        raise ValueError("development source provenance differs from its identity")
    records = {record["unit_id"]: record for record in iter_units(source_root, full=True)}
    if not records or any(r["status"] != "completed" for r in records.values()):
        raise ValueError("development root is incomplete")
    if PILOT_LEDGER not in records or SELECTION not in records:
        raise ValueError("development ledger or sealed selection missing")
    pilot = json.loads((records[PILOT_LEDGER]["path"] / "result.json").read_text())
    selection = json.loads((records[SELECTION]["path"] / "result.json").read_text())
    if (pilot["protocol"] != "plan15-drip-development-v1"
            or selection["protocol"] != pilot["protocol"]):
        raise ValueError("incompatible development protocol")
    selected = selection["selected_strength"]
    if not math.isfinite(selected) or selected <= 0:
        raise ValueError("invalid sealed FD strength")
    baseline = Treatment.default(cfg, "replay_ewc", **_policy(cfg))
    treatment = Treatment.default(cfg, "finite_difference", selected, **_policy(cfg))
    if pilot["baseline"] != baseline.mapping():
        raise ValueError("confirmation baseline differs from development")
    for replica in range(pilot["screen_start"], pilot["screen_start"] + pilot["screen_replicas"]):
        for stream in ("mixed", "base"):
            for condition in (baseline, treatment):
                unit = trajectory_id(pilot["family"], replica, stream, condition)
                if unit not in records:
                    raise ValueError("selected development comparison incomplete")
    # Numerical learning code must be identical; only orchestration/metadata changed.
    allowed = {"mnist_experiment/fisher_memory/config.py",
               "mnist_experiment/fisher_memory/drip.py",
               "mnist_experiment/fisher_memory/drip_runner.py"}
    current = provenance(cfg)
    changed = [name for name, digest in old["sources"].items()
               if current["sources"].get(name) != digest]
    if set(changed) - allowed:
        raise ValueError(f"development numerical sources changed: {sorted(set(changed) - allowed)}")
    return dict(schema_version=1, protocol=PROTOCOL, family=FAMILY,
                replicas=2 if cfg.smoke else 12, event_target=2 if cfg.smoke else 100,
                baseline=baseline.mapping(), treatment=treatment.mapping(),
                selected_strength=selected, angle=30., retention_margin=.01,
                ramp_batches=2 if cfg.smoke else 40, plateau=.5 if cfg.smoke else .1,
                optimizer_policy=_policy(cfg), bootstrap_samples=cfg.bootstrap_samples,
                inference_seed=cfg.seed_for(FAMILY, 0, "replica_bootstrap"),
                alpha=.05, alternative="greater", statistic="mean paired acquisition gain",
                source=dict(identity_hash=identity["identity_hash"],
                            config_hash=old["config_hash"], source_hash=old["source_hash"],
                            selection_seal=records[SELECTION]["completion"]["seal_hash"],
                            selection=selection, metadata_source_changes=changed),
                note="Fresh confirmation only; development replicas excluded. No early significance checks.")


def main(argv=None):
    from .drip_confirmation_report import analyze, build_notebook, collect

    args = arguments(argv)
    cfg = Config.read(args.config)
    if not math.isfinite(args.allocation_hours) or args.allocation_hours <= 0:
        raise ValueError("positive finite allocation is required")
    if args.max_units is not None and args.max_units < 1:
        raise ValueError("max-units must be positive")
    if args.clear_stop and not args.resume:
        raise ValueError("clearing STOP requires --resume")
    if args.smoke_stop_after_step is not None and (not cfg.smoke or args.smoke_stop_after_step < 1):
        raise ValueError("accepted-step interruption is smoke-only")
    root, budget = args.root.resolve(), args.budget_root.resolve()
    if root == budget or not root.is_relative_to(budget):
        raise ValueError("run root must be a distinct child of the shared budget root")
    if root == args.source_root.resolve():
        raise ValueError("confirmation requires a fresh root")
    if root.exists() and not args.resume:
        raise ValueError("confirmation root exists; use --resume or a new root")
    cfg = dataclasses.replace(cfg, compute_allocation_seconds=3600 * args.allocation_hours)
    cfg.validate()
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing silent CPU substitution")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan15-matplotlib")
    details = provenance(cfg)
    details["dataset_manifest"] = dataset_manifest(args.data_root)
    with budget_lock(budget):
        ledger = reference_ledger(args.source_root.resolve(), cfg,
                                  content_hash(details["dataset_manifest"]))
        identity = dict(schema_version=1, protocol=PROTOCOL, config_hash=cfg.scientific_hash,
                        source_hash=details["source_hash"],
                        data_hash=content_hash(details["dataset_manifest"]),
                        ledger_hash=content_hash(ledger))
        with Store(root, identity, resume=args.resume,
                max_bytes=cfg.max_artifact_bytes, external_bytes=external_bytes(args),
                free_reserve_bytes=cfg.free_reserve_bytes,
                finalization_reserve_bytes=32_000_000,
                max_wall_seconds=args.max_wall_seconds,
                max_active_seconds=cfg.compute_allocation_seconds,
                clear_stop=args.clear_stop) as store:
            if not (root / "provenance.json").exists():
                details["source_lineage"] = ledger["source"]
                store.write_json(None, "provenance.json", details)
                store.write_json(None, "resolved_config.json", cfg.mapping())
            if not store.is_complete(LEDGER):
                store.begin(LEDGER, dict(kind="drip_confirmation_ledger", protocol=PROTOCOL,
                                         ledger_hash=content_hash(ledger)))
                store.write_json(LEDGER, "result.json", ledger)
                store.complete(LEDGER)
            progress = Progress(store, args, ledger, builder=build_notebook)
            progress.state.update(stage="confirmation", status="running")

            def after_commit(unit, state):
                if progress.state.get("current_job") is not None:
                    progress.state["current_job"].update(
                        accepted_step=state["step"], paired_events=state["paired_events"])
                    progress()
                if args.smoke_stop_after_step and state["step"] >= args.smoke_stop_after_step:
                    raise StopRequested("smoke accepted-step interruption")

            runner = DripRunner(cfg, store, MNISTSource(args.data_root), on_progress=progress,
                                after_commit=after_commit, phase="5C", protocol=PROTOCOL)
            treatments = (Treatment(**ledger["baseline"]), Treatment(**ledger["treatment"]))
            completed_here = 0
            with store.control.signal_handlers():
                try:
                    progress(force=True)
                    for replica in range(ledger["replicas"]):
                        for stream in ("mixed", "base"):
                            for treatment in treatments:
                                unit = trajectory_id(FAMILY, replica, stream, treatment)
                                if store.is_complete(unit):
                                    continue
                                progress.state.update(current_job=dict(
                                    replica=replica, stream=stream, condition=treatment.condition,
                                    strength=treatment.strength))
                                progress()
                                summary = runner.run_trajectory(FAMILY, replica, stream, treatment)
                                completed_here += 1
                                progress.state["current_job"] = None
                                print(json.dumps(dict(event="drip_confirmation_complete", unit_id=unit,
                                                      seconds=summary["active_seconds"],
                                                      rotated_auc=summary["rotated_auc"],
                                                      base_auc=summary["base_auc"])), flush=True)
                                progress(force=True)
                                if args.max_units and completed_here >= args.max_units:
                                    raise StopRequested("requested unit batch complete")
                    data = collect(root)
                    result = analyze(data["units"], ledger)
                    if not result["complete"]:
                        raise ValueError("confirmation analysis requires the complete frozen cohort")
                    if not store.is_complete(RESULTS):
                        store.begin(RESULTS, dict(kind="drip_confirmation_results", protocol=PROTOCOL,
                                                 ledger_hash=content_hash(ledger)))
                        store.write_json(RESULTS, "result.json", result)
                        store.complete(RESULTS)
                    else:
                        sealed = json.loads((store.unit_path(RESULTS) / "result.json").read_text())
                        if result != sealed:
                            raise ValueError("sealed confirmation analysis changed")
                    progress.state.update(status="completed", stage="complete", current_job=None)
                except (StopRequested, LearnerStop, ResourceLimit) as exc:
                    progress.state.update(status="paused", current_job=None, stop_reason=str(exc))
                    print(json.dumps(dict(event="paused", reason=str(exc))), flush=True)
                except BaseException as exc:
                    progress.state.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    progress(force=True)
                    store.heartbeat(force=True)
            return progress.state["status"]


if __name__ == "__main__":
    if main() != "completed":
        raise SystemExit(3)
