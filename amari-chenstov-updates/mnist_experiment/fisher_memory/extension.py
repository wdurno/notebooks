"""Resumable, separately sealed Plan 15 curvature-method extension."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from .artifacts import Store, iter_units
from .config import ENVIRONMENTS, ROOT, Config, Treatment, content_hash, provenance
from .control import ResourceLimit, StopRequested, utc_now
from .data import MNISTSource
from .learner import StopRequested as LearnerStop
from .report import reserved_publisher
from .run import budget_lock, dataset_manifest, external_bytes
from .runner import Runner


METHODS = ("jvp", "directional")
LEDGER_UNIT = "extension-ledger-v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "mnist_experiment/fisher_memory/configs/default.json")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="completed Plan 15 root containing sealed controls and pilot selections")
    parser.add_argument("--root", type=Path, required=True, help="new root for extension artifacts")
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets")
    parser.add_argument("--notebook", type=Path, default=ROOT / "mnist_experiment/fisher_memory_extension_results.ipynb")
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--environments", nargs="+", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--replicas", type=int, default=12)
    parser.add_argument("--allocation-hours", type=float, required=True,
                        help="cumulative active-time ceiling, not additional time")
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    return parser.parse_args(argv)


def _read_result(record):
    return json.loads((record["path"] / "result.json").read_text())["result"]


def _one(records, description):
    if len(records) != 1:
        raise ValueError(f"expected one completed {description}, found {len(records)}")
    return records[0]


def initial_state_hash(shared: dict) -> str:
    """Hash the exact learner state cloned into each paired condition."""
    def freeze(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                        sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
        if isinstance(value, dict):
            return {key: freeze(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [freeze(item) for item in value]
        return value

    learner = shared["learner"]
    keys = ("theta", "ltm_indices", "anchor_L", "fisher_L", "replay",
            "consolidated_ids", "cumulative_count", "accepted_arrivals")
    return content_hash({key: freeze(learner[key]) for key in keys})


def _asset_checkpoint(record):
    pointer = json.loads((record["path"] / "latest.json").read_text())
    path = record["path"] / pointer["current"]["file"]
    envelope = torch.load(path, map_location="cpu", weights_only=False)
    return envelope["payload"]


def reference_ledger(source_root: Path, cfg: Config, environments: tuple[str, ...],
                     methods: tuple[str, ...], replicas: int, data_hash: str) -> dict:
    """Verify the old root once and freeze only the needed pilot/control evidence."""
    provenance_path = source_root / "provenance.json"
    old = json.loads(provenance_path.read_text())
    identity = json.loads((source_root / "identity.json").read_text())["identity"]
    if old["config_hash"] != cfg.scientific_hash or identity["config_hash"] != cfg.scientific_hash:
        raise ValueError("source confirmation configuration differs from the extension")
    if old["source_hash"] != identity["source_hash"]:
        raise ValueError("source confirmation provenance differs from its sealed identity")
    if identity["data_hash"] != data_hash:
        raise ValueError("cached dataset differs from the source confirmation")
    records = list(iter_units(source_root, completed_only=True, full=True))
    old_ledger_record = _one([r for r in records if r["metadata"].get("name") == "confirmation-ledger"],
                             "confirmation ledger")
    old_ledger = _read_result(old_ledger_record)
    if replicas < 1 or replicas > old_ledger["replicas"]:
        raise ValueError("replicas must be between one and the source confirmation count")
    selections = {e: _one([r for r in records if r["metadata"].get("name") == "selection"
                           and r["metadata"].get("environment") == e], f"{e} selection")
                  for e in environments}
    jobs = []
    asset_hashes = {}
    for replica in range(replicas):
        for environment in environments:
            baseline = _one([r for r in records if r["metadata"].get("family") == "confirmation"
                             and r["metadata"].get("environment") == environment
                             and r["metadata"].get("replica") == replica
                             and r["metadata"].get("condition") == "replay_ewc"],
                            f"{environment}/{replica} replay/EWC control")
            kind = "mixture" if environment == "digit9_mixture" else "rotation"
            asset = _one([r for r in records if r["unit_id"] == f"assets-confirmation-{kind}-{replica:04d}"],
                         f"{environment}/{replica} paired assets")
            if asset["unit_id"] not in asset_hashes:
                asset_hashes[asset["unit_id"]] = initial_state_hash(_asset_checkpoint(asset))
            selected = _read_result(selections[environment])["shortlist"]
            control = json.loads((baseline["path"] / "summary.json").read_text())
            if baseline["metadata"]["config_hash"] != cfg.scientific_hash or control["unit_id"] != baseline["unit_id"]:
                raise ValueError("source confirmation control identity mismatch")
            if control["nonfinite"] or control["solver_failures"]:
                raise ValueError("source confirmation control is numerically unhealthy")
            for method in methods:
                treatment = Treatment(**selected[method])
                if treatment.condition != method:
                    raise ValueError("pilot treatment method mismatch")
                for key in ("inner_steps", "max_evaluations", "replay_weight", "ridge_multiplier", "probes"):
                    if getattr(treatment, key) != baseline["metadata"]["config"][key]:
                        raise ValueError(f"pilot {method} does not share control setting {key}")
                jobs.append(dict(environment=environment, replica=replica, method=method,
                                 treatment=treatment.mapping(), control=control,
                                 control_unit=baseline["unit_id"],
                                 control_seal=baseline["completion"]["seal_hash"],
                                 asset_plan_hash=asset["metadata"]["plan_hash"],
                                 initial_state_hash=asset_hashes[asset["unit_id"]]))
    return dict(schema_version=1, protocol="plan15-extension-v1", scope=dict(
        environments=list(environments), methods=list(methods), replicas=replicas),
        source_root=str(source_root.resolve()), source_config_hash=cfg.scientific_hash,
        source_hash=identity["source_hash"], source_data_hash=data_hash,
        source_confirmation_ledger_seal=old_ledger_record["completion"]["seal_hash"],
        source_confirmation_protocol_hash=old_ledger["evaluation_protocol_hash"], jobs=jobs,
        interpretation="Post-selection paired extension; do not pool with the original confirmation claim.")


def _job_id(job):
    treatment = Treatment(**job["treatment"])
    return f"trajectory-confirmation-{job['environment']}-{job['replica']:04d}-{treatment.key}"


class ExtensionProgress:
    def __init__(self, store: Store, ledger: dict, notebook: Path):
        self.store, self.ledger, self.notebook = store, ledger, notebook
        self.last_report = 0.
        self.completed = {_job_id(job) for job in ledger["jobs"] if store.is_complete(_job_id(job))}
        self.state = dict(schema_version=1, protocol="plan15-extension-v1", status="running",
                          planned_units=len(ledger["jobs"]), completed_units=len(self.completed),
                          scope=ledger["scope"], current_job=None, heartbeat=utc_now())

    def __call__(self, *, force=False):
        from .extension_report import refresh
        store = self.store
        self.state.update(store.storage_status(), active_seconds=store.control.active_seconds,
                          heartbeat=utc_now())
        self.state["completed_units"] = len(self.completed)
        store.write_json(None, "status.json", self.state, finalizing=True)
        if force or time.monotonic() - self.last_report >= 900:
            path = refresh(store.root, publish=reserved_publisher(store, finalizing=force))
            content = path.read_bytes()
            with store.reserve(len(content), finalizing=True):
                self.notebook.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix=".plan15-extension-report-", dir=self.notebook.parent)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(name, self.notebook)
                finally:
                    Path(name).unlink(missing_ok=True)
            self.last_report = time.monotonic()


def main(argv=None):
    args = parse_args(argv)
    cfg = Config.read(args.config)
    if cfg.smoke:
        raise ValueError("extension requires the production Plan 15 configuration")
    if not args.methods or len(set(args.methods)) != len(args.methods):
        raise ValueError("methods must be nonempty and unique")
    if not args.environments or len(set(args.environments)) != len(args.environments):
        raise ValueError("environments must be nonempty and unique")
    if args.replicas < 1 or args.max_units is not None and args.max_units < 1:
        raise ValueError("replicas and max-units must be positive")
    if args.allocation_hours <= 0:
        raise ValueError("declare a positive cumulative compute allocation")
    if args.clear_stop and not args.resume:
        raise ValueError("clearing STOP requires --resume")
    budget = args.budget_root.resolve()
    root, source_root = args.root.resolve(), args.source_root.resolve()
    if root == source_root or root == budget or not root.is_relative_to(budget) or not source_root.is_relative_to(budget):
        raise ValueError("source and extension roots must be distinct children of the shared budget root")
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
        ledger = reference_ledger(source_root, cfg, tuple(args.environments), tuple(args.methods),
                                  args.replicas, data_hash)
        ledger_hash = content_hash(ledger)
        identity = dict(schema_version=1, config_hash=cfg.scientific_hash,
                        source_hash=details["source_hash"], data_hash=data_hash,
                        extension_ledger_hash=ledger_hash)
        if args.root.exists() and not args.resume:
            raise ValueError("extension root exists; pass --resume or use a new root")
        with Store(args.root, identity, resume=args.resume, max_bytes=cfg.max_artifact_bytes,
                   external_bytes=external_bytes(args), free_reserve_bytes=cfg.free_reserve_bytes,
                   finalization_reserve_bytes=32_000_000, max_wall_seconds=args.max_wall_seconds,
                   max_active_seconds=cfg.compute_allocation_seconds, clear_stop=args.clear_stop) as store:
            if not (store.root / "provenance.json").exists():
                store.write_json(None, "provenance.json", details)
                store.write_json(None, "resolved_config.json", cfg.mapping())
            if not store.is_complete(LEDGER_UNIT):
                store.begin(LEDGER_UNIT, dict(kind="extension_ledger", hash=ledger_hash, schema_version=1))
                store.write_json(LEDGER_UNIT, "result.json", ledger)
                store.complete(LEDGER_UNIT)
            progress = ExtensionProgress(store, ledger, args.notebook)
            runner = Runner(cfg, store, MNISTSource(args.data_root), on_progress=progress)
            progress(force=True)
            with store.control.signal_handlers():
                try:
                    for job in ledger["jobs"]:
                        unit_id = _job_id(job)
                        if store.is_complete(unit_id):
                            continue
                        progress.state["current_job"] = dict(environment=job["environment"],
                                                              replica=job["replica"], method=job["method"])
                        progress()
                        _, shared = runner._assets("confirmation", job["environment"], job["replica"])
                        if shared["assets"]["plan_hash"] != job["asset_plan_hash"]:
                            raise ValueError("newly materialized paired asset plan differs from source confirmation")
                        if initial_state_hash(shared) != job["initial_state_hash"]:
                            raise ValueError("initial model/Fisher/replay state differs from source confirmation")
                        summary = runner.run_trajectory("confirmation", "phase5-extension", job["environment"],
                                                        job["replica"], Treatment(**job["treatment"]))
                        progress.completed.add(unit_id)
                        progress.state["current_job"] = None
                        print(json.dumps(dict(event="extension_trajectory_complete", unit_id=unit_id,
                                              seconds=summary["active_seconds"])), flush=True)
                        progress()
                        if (args.max_units and runner.completed_this_invocation >= args.max_units
                                and len(progress.completed) < len(ledger["jobs"])):
                            raise StopRequested("requested unit batch complete")
                    progress.state.update(status="completed", current_job=None)
                except (StopRequested, LearnerStop, ResourceLimit) as exc:
                    progress.state.update(status="paused", stop_reason=str(exc))
                    print(json.dumps(dict(event="paused", reason=str(exc))), flush=True)
                except BaseException as exc:
                    progress.state.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    progress(force=True)
                    store.heartbeat(force=True)


if __name__ == "__main__":
    main()
