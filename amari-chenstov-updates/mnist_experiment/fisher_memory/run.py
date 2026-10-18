"""Standalone Plan 15 scheduler and Phase 0 acceptance entry point."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from .artifacts import Store
from .config import CONDITIONS, ROOT, Config, Treatment, provenance
from .control import ResourceLimit, StopRequested, utc_now
from .data import MNISTSource
from .learner import StopRequested as LearnerStop
from .report import MAX_NOTEBOOK_BYTES, refresh, reserved_publisher
from .runner import Runner


def file_bytes(root: Path, *, exclude: Path | None = None) -> int:
    total = 0
    if not root.exists():
        return total
    for parent, directories, files in os.walk(root):
        if exclude is not None:
            directories[:] = [d for d in directories if (Path(parent) / d).resolve() != exclude.resolve()]
        for name in files:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError(f"budget dependency must not be a symlink: {path}")
            total += path.stat().st_size
    return total


@contextlib.contextmanager
def budget_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".budget.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Plan 15 invocation owns the shared storage budget") from exc
        yield


def external_bytes(args):
    orphan_exports = sum(path.stat().st_size for path in args.notebook.parent.glob(".plan15-report-*"))
    return (file_bytes(args.data_root) + file_bytes(args.budget_root, exclude=args.root)
            + MAX_NOTEBOOK_BYTES + orphan_exports)


def dataset_manifest(root: Path):
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("dataset dependencies must be regular files")
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    if not entries:
        raise ValueError("cached dataset is missing; acceptance/production never silently download")
    return entries


class Progress:
    def __init__(self, store, cfg, notebook=None):
        self.store, self.cfg, self.notebook = store, cfg, notebook
        self.last_report = 0.
        self.state = dict(schema_version=1, protocol_revision=cfg.protocol, phase="0" if cfg.smoke else "1",
                          status="running", changes=[], report_seconds=0.,
                          note="Smoke acceptance only; no scientific inference." if cfg.smoke else "Development begins.")

    def __call__(self, *, force=False):
        store = self.store
        self.state.update(store.storage_status(), active_seconds=store.control.active_seconds, heartbeat=utc_now())
        store.write_json(None, "status.json", self.state, finalizing=True)
        if force or time.monotonic() - self.last_report >= 900:
            tick = time.perf_counter()
            path = refresh(store.root, publish=reserved_publisher(store, finalizing=force))
            if self.notebook is not None:
                # The external final copy is charged at its maximum size by the
                # run admission ledger; this reservation covers its temporary copy.
                content = path.read_bytes()
                with store.reserve(len(content), finalizing=True):
                    self.notebook.parent.mkdir(parents=True, exist_ok=True)
                    fd, name = tempfile.mkstemp(prefix=".plan15-report-", dir=self.notebook.parent)
                    try:
                        with os.fdopen(fd, "wb") as stream:
                            stream.write(content)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(name, self.notebook)
                    finally:
                        Path(name).unlink(missing_ok=True)
            self.last_report = time.monotonic()
            self.state["report_seconds"] += time.perf_counter() - tick
            store.write_json(None, "status.json", self.state, finalizing=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets")
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--notebook", type=Path, default=ROOT / "mnist_experiment/fisher_memory_results.ipynb")
    parser.add_argument("--mode", choices=("study", "acceptance", "pipeline-smoke"), default="study")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true", help="explicitly clear a prior STOP file after acquiring the writer lock")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--allocation-hours", type=float, help="cumulative active-time ceiling, not added time")
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, help="acceptance only")
    parser.add_argument("--benchmark", action="store_true", help="acceptance only: time production-shaped solves, without a scientific pilot")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    cfg = Config.read(args.config)
    if args.device:
        cfg = dataclasses.replace(cfg, device=args.device)
    if args.dtype:
        cfg = dataclasses.replace(cfg, dtype=args.dtype)
    if args.allocation_hours is not None:
        cfg = dataclasses.replace(cfg, compute_allocation_seconds=3600 * args.allocation_hours)
    cfg.validate()
    if (args.mode != "study") != cfg.smoke:
        raise ValueError("acceptance/pipeline-smoke require smoke=true; study requires smoke=false")
    if cfg.compute_allocation_seconds <= 0:
        raise ValueError("declare a positive compute allocation before execution")
    if args.conditions and args.mode != "acceptance":
        raise ValueError("production conditions come from the frozen selection ledger")
    if args.benchmark and args.mode != "acceptance":
        raise ValueError("timing benchmark belongs to acceptance, not the scientific scheduler")
    if args.max_units is not None and args.max_units <= 0:
        raise ValueError("max-units must be positive")
    if not args.root.resolve().is_relative_to(args.budget_root.resolve()) or args.root.resolve() == args.budget_root.resolve():
        raise ValueError("run root must be a child of the shared Plan 15 budget root")
    torch.set_num_threads(cfg.threads)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/plan15-matplotlib")
    torch.use_deterministic_algorithms(True)
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; do not silently substitute CPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    details = provenance(cfg)
    identity = {key: details[key] for key in ("schema_version", "config_hash", "source_hash")}
    from .config import content_hash
    details["dataset_manifest"] = dataset_manifest(args.data_root)
    identity["data_hash"] = content_hash(details["dataset_manifest"])
    # Two previous generations plus temporary checkpoint and report publication
    # are admitted independently; reserve room for the final status/report.
    with budget_lock(args.budget_root), Store(args.root, identity, resume=args.resume, max_bytes=cfg.max_artifact_bytes,
               external_bytes=external_bytes(args), free_reserve_bytes=cfg.free_reserve_bytes,
               finalization_reserve_bytes=32_000_000, max_wall_seconds=args.max_wall_seconds,
               max_active_seconds=cfg.compute_allocation_seconds, clear_stop=args.clear_stop) as store:
        if not (store.root / "provenance.json").exists():
            store.write_json(None, "provenance.json", details)
            store.write_json(None, "resolved_config.json", cfg.mapping())
        progress = Progress(store, cfg, args.notebook)
        runner = Runner(cfg, store, MNISTSource(args.data_root), on_progress=progress)
        progress(force=True)
        with store.control.signal_handlers():
            try:
                if args.mode == "acceptance":
                    for env in cfg.environments:
                        for condition in args.conditions or CONDITIONS:
                            strength = .1 if condition in ("jvp", "finite_difference", "directional", "distillation") else 0.
                            treatment = Treatment.default(cfg, condition, strength)
                            summary = runner.run_trajectory("acceptance", "0", env, 0, treatment)
                            print(json.dumps(dict(event="trajectory_complete", unit_id=summary["unit_id"],
                                                  seconds=summary["active_seconds"], nonfinite=summary["nonfinite"])), flush=True)
                            if args.max_units and runner.completed_this_invocation >= args.max_units:
                                raise StopRequested("requested unit batch complete")
                    progress.state.update(status="acceptance_complete", phase="0")
                    if args.benchmark:
                        from .benchmark import benchmark
                        measured = benchmark(runner)
                        progress.state["benchmark_seconds"] = {key: value["seconds"] for key, value in measured["methods"].items()}
                        print(json.dumps(dict(event="timing_benchmark_complete", seconds=progress.state["benchmark_seconds"])), flush=True)
                else:
                    from .scheduler import run_study
                    run_study(runner, progress=progress, max_units=args.max_units)
                    progress.state.update(status="completed", phase="4")
            except (StopRequested, LearnerStop, ResourceLimit) as exc:
                progress.state.update(status="paused", stop_reason=str(exc))
                print(json.dumps(dict(event="paused", reason=str(exc))), flush=True)
            except BaseException as exc:
                progress.state.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                progress(force=True)
                store.heartbeat(force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
