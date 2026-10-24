"""Fresh, resumable sigmoid-JVP strength reassessment for Plan 15."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import tempfile
import time
from pathlib import Path

import torch

from .artifacts import Store, iter_units
from .config import ROOT, Config, Treatment, content_hash, provenance
from .control import ResourceLimit, StopRequested, utc_now
from .data import MNISTSource
from .learner import StopRequested as LearnerStop
from .metrics import paired_summary
from .report import reserved_publisher
from .run import budget_lock, dataset_manifest, external_bytes
from .runner import Runner


PROTOCOL = "plan15-jvp-repair-v1"
LEDGER_UNIT = "jvp-repair-ledger-v1"
SELECTION_UNIT = "jvp-repair-selection-v1"
ENVIRONMENT = "rotation_sigmoid"
PILOT_FAMILY = "jvp_repair_pilot"
CONFIRM_FAMILY = "jvp_repair_confirmation"
GRID_FACTORS = (1 / 30, 1 / 10, 1 / 3, 1., 3.)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "mnist_experiment/fisher_memory/configs/default.json")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets")
    parser.add_argument("--notebook", type=Path, default=ROOT / "mnist_experiment/fisher_memory_jvp_repair_results.ipynb")
    parser.add_argument("--pilot-replicas", type=int, default=3)
    parser.add_argument("--confirm-replicas", type=int, default=12)
    parser.add_argument("--allocation-hours", type=float, required=True,
                        help="cumulative active-time ceiling, not additional time")
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    return parser.parse_args(argv)


def source_policy(root: Path, cfg: Config, data_hash: str) -> dict:
    """Read the sealed FD selection solely to fix the shared policy and matched strength."""
    identity = json.loads((root / "identity.json").read_text())["identity"]
    provenance_record = json.loads((root / "provenance.json").read_text())
    if (identity["config_hash"] != cfg.scientific_hash
            or provenance_record["config_hash"] != cfg.scientific_hash
            or identity["data_hash"] != data_hash
            or provenance_record["source_hash"] != identity["source_hash"]):
        raise ValueError("source configuration, dataset, or provenance mismatch")
    records = list(iter_units(root, completed_only=True, full=True))
    selections = [record for record in records if record["metadata"].get("name") == "selection"
                  and record["metadata"].get("environment") == ENVIRONMENT]
    if len(selections) != 1:
        raise ValueError("expected exactly one sealed sigmoid pilot selection")
    selected = selections[0]
    result = json.loads((selected["path"] / "result.json").read_text())["result"]
    finite_difference = Treatment(**result["shortlist"]["finite_difference"])
    if finite_difference.condition != "finite_difference" or finite_difference.strength <= 0:
        raise ValueError("source did not select a positive finite-difference strength")
    controls = [record for record in records if record["metadata"].get("family") == "confirmation"
                and record["metadata"].get("environment") == ENVIRONMENT
                and record["metadata"].get("replica") == 0
                and record["metadata"].get("condition") == "replay_ewc"]
    if len(controls) != 1:
        raise ValueError("expected one sealed sigmoid confirmation control")
    for key in ("inner_steps", "max_evaluations", "replay_weight", "ridge_multiplier", "probes"):
        if getattr(finite_difference, key) != controls[0]["metadata"]["config"][key]:
            raise ValueError(f"source policy mismatch: {key}")
    return dict(treatment=finite_difference.mapping(), source_hash=identity["source_hash"],
                selection_seal=selected["completion"]["seal_hash"],
                control_seal=controls[0]["completion"]["seal_hash"])


def build_ledger(source_root: Path, cfg: Config, data_hash: str,
                 pilot_replicas: int, confirm_replicas: int) -> dict:
    if pilot_replicas < 2 or confirm_replicas < 2:
        raise ValueError("repair requires at least two pilot and confirmation replicas")
    policy = source_policy(source_root, cfg, data_hash)
    reference = Treatment(**policy["treatment"])
    baseline = dataclasses.replace(reference, condition="replay_ewc", strength=0., label="")
    grid = [dataclasses.replace(reference, condition="jvp", strength=reference.strength * factor,
                                label="").mapping() for factor in GRID_FACTORS]
    if len({Treatment(**item).key for item in grid}) != len(grid):
        raise ValueError("strength grid must have unique treatment identities")
    return dict(schema_version=1, protocol=PROTOCOL, environment=ENVIRONMENT,
                pilot_family=PILOT_FAMILY, confirm_family=CONFIRM_FAMILY,
                pilot_replicas=pilot_replicas, confirm_replicas=confirm_replicas,
                baseline=baseline.mapping(), grid_factors=list(GRID_FACTORS), grid=grid,
                matched_strength=reference.strength, matched_treatment=grid[GRID_FACTORS.index(1.)],
                retention_margin=cfg.retention_margin, source_root=str(source_root.resolve()),
                source_config_hash=cfg.scientific_hash, source_data_hash=data_hash,
                source_policy=policy,
                interpretation="Post-hoc JVP repair; selection uses fresh pilot seeds and confirmation uses a disjoint seed family.")


def unit_id(family: str, replica: int, treatment: Treatment) -> str:
    return f"trajectory-{family}-{ENVIRONMENT}-{replica:04d}-{treatment.key}"


def _summary(store: Store, family: str, replica: int, treatment: Treatment) -> dict:
    unit = unit_id(family, replica, treatment)
    if not store.is_complete(unit):
        raise ValueError(f"required trajectory is incomplete: {unit}")
    result = json.loads((store.unit_path(unit) / "summary.json").read_text())
    if result["unit_id"] != unit or result["nonfinite"] or result["solver_failures"]:
        raise ValueError(f"invalid completed trajectory: {unit}")
    return result


def select_pilot(store: Store, ledger: dict) -> dict:
    """Prioritize mean acquisition subject to the declared retention margin."""
    family = ledger["pilot_family"]
    baseline = Treatment(**ledger["baseline"])
    controls = {r: _summary(store, family, r, baseline) for r in range(ledger["pilot_replicas"])}
    table = []
    for item in ledger["grid"]:
        treatment = Treatment(**item)
        rows = {r: _summary(store, family, r, treatment) for r in controls}
        current = paired_summary({r: s["current_auc"] for r, s in controls.items()},
                                 {r: s["current_auc"] for r, s in rows.items()})
        historical = paired_summary({r: s["historical_auc"] for r, s in controls.items()},
                                    {r: s["historical_auc"] for r, s in rows.items()})
        if (current["n_pairs"] != len(controls) or historical["n_pairs"] != len(controls)):
            raise ValueError("pilot selection requires complete paired current and post-eviction AUC")
        table.append(dict(treatment=item, key=treatment.key, strength=treatment.strength,
                          current_gain=current["mean_gain"], historical_gain=historical["mean_gain"],
                          favorable_current=current["favorable_replicas"],
                          favorable_historical=historical["favorable_replicas"],
                          online_seconds=sum(row["active_seconds"] for row in rows.values()) / len(rows),
                          retention_feasible=historical["mean_gain"] >= -ledger["retention_margin"]))
    feasible = [row for row in table if row["retention_feasible"]]
    if feasible:
        winner = max(feasible, key=lambda row: (row["current_gain"], row["historical_gain"],
                                                 -row["strength"]))
        classification = "retention_feasible"
    else:
        winner = max(table, key=lambda row: (min(row["current_gain"], row["historical_gain"]),
                                               row["current_gain"], -row["strength"]))
        classification = "diagnostic_retention_fallback"
    confirm = [winner["treatment"]]
    if winner["key"] != Treatment(**ledger["matched_treatment"]).key:
        confirm.append(ledger["matched_treatment"])
    return dict(schema_version=1, protocol=PROTOCOL, classification=classification,
                selected=winner["treatment"], confirm=confirm, pilot_table=table,
                primary_rule="maximum paired mean current NLL-AUC gain among strengths with historical harm <= margin; no cost tie",
                matched_rule="always confirm the finite-difference-selected coefficient as a fixed secondary diagnostic")


class Progress:
    def __init__(self, store: Store, ledger: dict, notebook: Path):
        self.store, self.ledger, self.notebook = store, ledger, notebook
        self.last_report = 0.
        self.completed = set()
        self.selection = None
        self.state = dict(schema_version=1, protocol=PROTOCOL, status="running", stage="pilot",
                          current_job=None, heartbeat=utc_now())

    def __call__(self, *, force: bool = False):
        from .jvp_repair_report import refresh
        store = self.store
        planned_pilot = self.ledger["pilot_replicas"] * (1 + len(self.ledger["grid"]))
        planned_confirm = self.ledger["confirm_replicas"] * (
            1 + (len(self.selection["confirm"]) if self.selection else 2))
        self.state.update(store.storage_status(), active_seconds=store.control.active_seconds,
                          completed_units=len(self.completed), planned_pilot=planned_pilot,
                          planned_confirmation=planned_confirm, heartbeat=utc_now())
        store.write_json(None, "status.json", self.state, finalizing=True)
        if force or time.monotonic() - self.last_report >= 900:
            target = refresh(store.root, publish=reserved_publisher(store, finalizing=force))
            content = target.read_bytes()
            with store.reserve(len(content), finalizing=True):
                self.notebook.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix=".plan15-jvp-repair-report-", dir=self.notebook.parent)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(name, self.notebook)
                finally:
                    Path(name).unlink(missing_ok=True)
            self.last_report = time.monotonic()


def main(argv=None) -> str:
    args = arguments(argv)
    cfg = Config.read(args.config)
    if cfg.smoke:
        raise ValueError("production repair requires a non-smoke Plan 15 configuration")
    if (args.pilot_replicas < 2 or args.confirm_replicas < 2
            or args.max_units is not None and args.max_units < 1):
        raise ValueError("invalid replica or unit count")
    if not math.isfinite(args.allocation_hours) or args.allocation_hours <= 0:
        raise ValueError("declare a finite positive cumulative compute allocation")
    if args.clear_stop and not args.resume:
        raise ValueError("clearing STOP requires --resume")
    root, source_root, budget = args.root.resolve(), args.source_root.resolve(), args.budget_root.resolve()
    if (root == source_root or root == budget or not root.is_relative_to(budget)
            or not source_root.is_relative_to(budget)):
        raise ValueError("repair and source roots must be distinct children of the shared budget root")
    cfg = dataclasses.replace(cfg, compute_allocation_seconds=3600 * args.allocation_hours)
    cfg.validate()
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; do not silently substitute CPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan15-matplotlib")
    details = provenance(cfg)
    details["dataset_manifest"] = dataset_manifest(args.data_root)
    data_hash = content_hash(details["dataset_manifest"])
    with budget_lock(args.budget_root):
        ledger = build_ledger(source_root, cfg, data_hash, args.pilot_replicas, args.confirm_replicas)
        identity = dict(schema_version=1, config_hash=cfg.scientific_hash,
                        source_hash=details["source_hash"], data_hash=data_hash,
                        repair_ledger_hash=content_hash(ledger))
        if args.root.exists() and not args.resume:
            raise ValueError("repair root exists; pass --resume or use a new root")
        with Store(args.root, identity, resume=args.resume, max_bytes=cfg.max_artifact_bytes,
                   external_bytes=external_bytes(args), free_reserve_bytes=cfg.free_reserve_bytes,
                   finalization_reserve_bytes=32_000_000, max_wall_seconds=args.max_wall_seconds,
                   max_active_seconds=cfg.compute_allocation_seconds, clear_stop=args.clear_stop) as store:
            if not (store.root / "provenance.json").exists():
                store.write_json(None, "provenance.json", details)
                store.write_json(None, "resolved_config.json", cfg.mapping())
            if not store.is_complete(LEDGER_UNIT):
                store.begin(LEDGER_UNIT, dict(kind="repair_ledger", protocol=PROTOCOL,
                                             hash=content_hash(ledger)))
                store.write_json(LEDGER_UNIT, "result.json", ledger)
                store.complete(LEDGER_UNIT)
            progress = Progress(store, ledger, args.notebook)
            runner = Runner(cfg, store, MNISTSource(args.data_root), on_progress=progress)
            pilot = [Treatment(**ledger["baseline"]), *(Treatment(**item) for item in ledger["grid"])]
            progress.completed = {unit_id(PILOT_FAMILY, r, treatment)
                                  for r in range(ledger["pilot_replicas"]) for treatment in pilot
                                  if store.is_complete(unit_id(PILOT_FAMILY, r, treatment))}
            if store.is_complete(SELECTION_UNIT):
                progress.selection = json.loads((store.unit_path(SELECTION_UNIT) / "result.json").read_text())
                confirm = [Treatment(**ledger["baseline"]),
                           *(Treatment(**item) for item in progress.selection["confirm"])]
                progress.completed.update(unit_id(CONFIRM_FAMILY, r, treatment)
                                          for r in range(ledger["confirm_replicas"]) for treatment in confirm
                                          if store.is_complete(unit_id(CONFIRM_FAMILY, r, treatment)))
            progress(force=True)
            with store.control.signal_handlers():
                try:
                    for r in range(ledger["pilot_replicas"]):
                        for treatment in pilot:
                            unit = unit_id(PILOT_FAMILY, r, treatment)
                            if unit in progress.completed:
                                continue
                            progress.state.update(stage="pilot", current_job=dict(replica=r, strength=treatment.strength))
                            progress()
                            summary = runner.run_trajectory(PILOT_FAMILY, "jvp-repair-pilot", ENVIRONMENT, r, treatment)
                            progress.completed.add(unit)
                            progress.state["current_job"] = None
                            print(json.dumps(dict(event="repair_trajectory_complete", unit_id=unit,
                                                  seconds=summary["active_seconds"])), flush=True)
                            progress()
                            if args.max_units and runner.completed_this_invocation >= args.max_units:
                                raise StopRequested("requested unit batch complete")
                    if progress.selection is None:
                        selection = select_pilot(store, ledger)
                        store.begin(SELECTION_UNIT, dict(kind="repair_selection", protocol=PROTOCOL,
                                                         ledger_hash=content_hash(ledger)))
                        store.write_json(SELECTION_UNIT, "result.json", selection)
                        store.complete(SELECTION_UNIT)
                        progress.selection = selection
                    progress.state.update(stage="confirmation", selection=progress.selection["selected"])
                    progress(force=True)
                    confirm = [Treatment(**ledger["baseline"]),
                               *(Treatment(**item) for item in progress.selection["confirm"])]
                    for r in range(ledger["confirm_replicas"]):
                        for treatment in confirm:
                            unit = unit_id(CONFIRM_FAMILY, r, treatment)
                            if unit in progress.completed:
                                continue
                            progress.state["current_job"] = dict(replica=r, strength=treatment.strength)
                            progress()
                            summary = runner.run_trajectory(CONFIRM_FAMILY, "jvp-repair-confirmation",
                                                            ENVIRONMENT, r, treatment)
                            progress.completed.add(unit)
                            progress.state["current_job"] = None
                            print(json.dumps(dict(event="repair_trajectory_complete", unit_id=unit,
                                                  seconds=summary["active_seconds"])), flush=True)
                            progress()
                            if (args.max_units and runner.completed_this_invocation >= args.max_units
                                    and len(progress.completed) < progress.state["planned_pilot"]
                                    + progress.state["planned_confirmation"]):
                                raise StopRequested("requested unit batch complete")
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
