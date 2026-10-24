"""Resumable Plan 15 Phase 5B rare rotated-domain development pilot."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import tempfile
import time
from pathlib import Path

import nbformat
import torch

from .artifacts import Store
from .config import ROOT, Config, Treatment, content_hash, provenance
from .control import ResourceLimit, StopRequested, utc_now
from .data import MNISTSource
from .drip_report import build_notebook
from .drip_runner import DripRunner, trajectory_id
from .learner import StopRequested as LearnerStop
from .report import MAX_NOTEBOOK_BYTES, reserved_publisher
from .run import budget_lock, dataset_manifest, external_bytes


PROTOCOL = "plan15-drip-development-v1"
LEDGER = "drip-development-ledger-v1"
SELECTION = "drip-development-selection-v1"
FAMILY = "drip_development"
GRID = (.00178, .0219, .269, 2.689, 26.89, 53.63)
POLICY = dict(inner_steps=100, max_evaluations=150)


def _policy(cfg):
    return dict(inner_steps=cfg.inner_steps, max_evaluations=cfg.max_evaluations) if cfg.smoke else POLICY


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "mnist_experiment/fisher_memory/configs/default.json")
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
    parser.add_argument("--smoke-stop-after-step", type=int,
                        help="smoke-only accepted-step interruption check")
    return parser.parse_args(argv)


def _ledger(cfg):
    baseline = Treatment.default(cfg, "replay_ewc", **_policy(cfg))
    strengths = (.269,) if cfg.smoke else GRID
    return dict(schema_version=1, protocol=PROTOCOL, family=FAMILY,
                diagnostic_replicas=1 if cfg.smoke else 3,
                screen_replicas=1 if cfg.smoke else 4,
                screen_start=1 if cfg.smoke else 3,
                event_target=2 if cfg.smoke else 100,
                grid=list(strengths), baseline=baseline.mapping(),
                optimizer_policy=_policy(cfg),
                retention_margin=.01, ramp_batches=2 if cfg.smoke else 40,
                plateau=.5 if cfg.smoke else .1,
                note="Development selection only; no pilot replica enters confirmation inference.")


def _treatment(cfg, strength):
    return Treatment.default(cfg, "finite_difference", strength, **_policy(cfg))


def _summary(store, replica, stream, treatment):
    unit = trajectory_id(FAMILY, replica, stream, treatment)
    if not store.is_complete(unit):
        raise ValueError(f"required drip trajectory incomplete: {unit}")
    return json.loads((store.unit_path(unit) / "summary.json").read_text())


def _table(store, cfg, ledger, strengths):
    baseline = Treatment(**ledger["baseline"])
    table = []
    for strength in strengths:
        treatment = _treatment(cfg, strength)
        pairs = []
        for replica in range(ledger["screen_start"], ledger["screen_start"] + ledger["screen_replicas"]):
            control = _summary(store, replica, "mixed", baseline)
            treated = _summary(store, replica, "mixed", treatment)
            valid = (not treated["nonfinite"] and treated["solver_failures"] == 0
                     and not control["nonfinite"] and control["solver_failures"] == 0)
            values = (control["rotated_auc"], treated["rotated_auc"],
                      control["base_auc"], treated["base_auc"])
            valid = valid and all(math.isfinite(value) for value in values)
            pairs.append(dict(replica=replica, valid=valid,
                              acquisition_gain=values[0] - values[1],
                              base_harm=values[3] - values[2],
                              online_seconds=treated["active_seconds"],
                              under_solved=treated["under_solved"]))
        table.append(dict(strength=strength, pairs=pairs,
                          valid=all(pair["valid"] for pair in pairs),
                          mean_gain=sum(pair["acquisition_gain"] for pair in pairs) / len(pairs),
                          mean_harm=sum(pair["base_harm"] for pair in pairs) / len(pairs),
                          online_seconds=sum(pair["online_seconds"] for pair in pairs) / len(pairs)))
    return table


def select(table, margin):
    valid = [row for row in table if row["valid"]]
    if not valid:
        raise ValueError("entire development strength grid is numerically invalid; version a repaired pilot")
    feasible = [row for row in valid if row["mean_harm"] <= margin and row["mean_gain"] > 0]
    pool = feasible or valid
    chosen = min(pool, key=lambda row: (-row["mean_gain"], row["online_seconds"], row["strength"]))
    return chosen, "retention_feasible" if feasible else "pilot_warning_acquisition_fallback"


class Progress:
    def __init__(self, store, args, ledger, *, builder=build_notebook):
        self.store, self.args, self.ledger = store, args, ledger
        self.builder = builder
        self.last_report = 0.
        previous = store.root / "status.json"
        self.state = json.loads(previous.read_text()) if previous.exists() else dict(
            schema_version=1, protocol=ledger["protocol"], status="running", stage="diagnostic",
            current_job=None, pilot=None, changes=[])

    def __call__(self, *, force=False):
        store = self.store
        self.state.update(store.storage_status(), active_seconds=store.control.active_seconds,
                          heartbeat=utc_now())
        store.write_json(None, "status.json", self.state, finalizing=True)
        if force or time.monotonic() - self.last_report >= 900:
            content = nbformat.writes(self.builder(self.args.notebook, store.root)).encode()
            if len(content) > MAX_NOTEBOOK_BYTES:
                raise ValueError("drip notebook exceeds publication limit")
            target = store.root / "reports" / "fisher_memory_results.ipynb"
            reserved_publisher(store, finalizing=force)(target, content)
            with store.reserve(len(content), finalizing=True):
                self.args.notebook.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix=".plan15-drip-report-", dir=self.args.notebook.parent)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(name, self.args.notebook)
                finally:
                    Path(name).unlink(missing_ok=True)
            self.last_report = time.monotonic()


def main(argv=None):
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
    ledger = _ledger(cfg)
    identity = dict(schema_version=1, protocol=PROTOCOL, config_hash=cfg.scientific_hash,
                    source_hash=details["source_hash"],
                    data_hash=content_hash(details["dataset_manifest"]),
                    ledger_hash=content_hash(ledger))
    if root.exists() and not args.resume:
        raise ValueError("development root exists; use --resume or a new root")
    with budget_lock(budget), Store(root, identity, resume=args.resume,
            max_bytes=cfg.max_artifact_bytes, external_bytes=external_bytes(args),
            free_reserve_bytes=cfg.free_reserve_bytes,
            finalization_reserve_bytes=32_000_000,
            max_wall_seconds=args.max_wall_seconds,
            max_active_seconds=cfg.compute_allocation_seconds,
            clear_stop=args.clear_stop) as store:
        if not (store.root / "provenance.json").exists():
            store.write_json(None, "provenance.json", details)
            store.write_json(None, "resolved_config.json", cfg.mapping())
        if not store.is_complete(LEDGER):
            store.begin(LEDGER, dict(kind="drip_ledger", protocol=PROTOCOL,
                                     ledger_hash=content_hash(ledger)))
            store.write_json(LEDGER, "result.json", ledger)
            store.complete(LEDGER)
        progress = Progress(store, args, ledger)
        stop_after = args.smoke_stop_after_step
        def after_commit(unit, state):
            if progress.state.get("current_job") is not None:
                progress.state["current_job"].update(
                    accepted_step=state["step"], paired_events=state["paired_events"])
                progress()
            if stop_after is not None and state["step"] >= stop_after:
                raise StopRequested("smoke accepted-step interruption")
        runner = DripRunner(cfg, store, MNISTSource(args.data_root), on_progress=progress,
                            after_commit=after_commit)
        baseline = Treatment(**ledger["baseline"])
        completed_here = 0

        def run(replica, stream, treatment, stage):
            nonlocal completed_here
            unit = trajectory_id(FAMILY, replica, stream, treatment)
            if store.is_complete(unit):
                return _summary(store, replica, stream, treatment)
            progress.state.update(stage=stage, status="running",
                                  current_job=dict(replica=replica, stream=stream,
                                                   condition=treatment.condition,
                                                   strength=treatment.strength))
            progress()
            result = runner.run_trajectory(FAMILY, replica, stream, treatment)
            completed_here += 1
            progress.state["current_job"] = None
            print(json.dumps(dict(event="drip_trajectory_complete", unit_id=unit,
                                  seconds=result["active_seconds"],
                                  rotated_auc=result["rotated_auc"],
                                  base_auc=result["base_auc"])), flush=True)
            progress(force=True)
            if args.max_units and completed_here >= args.max_units:
                raise StopRequested("requested unit batch complete")
            return result

        progress(force=True)
        with store.control.signal_handlers():
            try:
                for replica in range(ledger["diagnostic_replicas"]):
                    run(replica, "mixed", baseline, "diagnostic")
                    run(replica, "base", baseline, "diagnostic")
                strengths = list(ledger["grid"])
                for replica in range(ledger["screen_start"],
                                     ledger["screen_start"] + ledger["screen_replicas"]):
                    run(replica, "mixed", baseline, "strength_screen")
                    for strength in strengths:
                        run(replica, "mixed", _treatment(cfg, strength), "strength_screen")
                table = _table(store, cfg, ledger, strengths)
                chosen, classification = select(table, ledger["retention_margin"])
                if chosen["strength"] in (strengths[0], strengths[-1]) and not cfg.smoke:
                    boundary = strengths[0] / 10 if chosen["strength"] == strengths[0] else strengths[-1] * 3
                    for replica in range(ledger["screen_start"],
                                         ledger["screen_start"] + ledger["screen_replicas"]):
                        run(replica, "mixed", _treatment(cfg, boundary), "boundary_probe")
                    strengths.append(boundary)
                    table = _table(store, cfg, ledger, strengths)
                    chosen, classification = select(table, ledger["retention_margin"])
                pilot = dict(schema_version=1, protocol=PROTOCOL,
                             selected_strength=chosen["strength"], classification=classification,
                             table=table, grid_with_boundary=strengths,
                             selection_rule="largest paired mean acquisition gain with base harm <= .01; "
                                            "if no positive feasible gain, best valid acquisition warning")
                if not store.is_complete(SELECTION):
                    store.begin(SELECTION, dict(kind="drip_selection", protocol=PROTOCOL,
                                                ledger_hash=content_hash(ledger)))
                    store.write_json(SELECTION, "result.json", pilot)
                    store.complete(SELECTION)
                else:
                    sealed = json.loads((store.unit_path(SELECTION) / "result.json").read_text())
                    if sealed != pilot:
                        raise ValueError("sealed development selection changed")
                progress.state["pilot"] = pilot
                progress(force=True)
                for replica in range(ledger["screen_start"],
                                     ledger["screen_start"] + ledger["screen_replicas"]):
                    run(replica, "base", baseline, "base_controls")
                    run(replica, "base", _treatment(cfg, chosen["strength"]), "base_controls")
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
