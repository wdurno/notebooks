"""Phase 7E: sealed-trajectory, CPU-only sample-equivalence analysis.

Fits describe cohort-mean predictive NLL, never a training objective. Whole
paired replicas, not correlated post-batch checkpoints, are bootstrap units.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np

from src.seeding import derive_component_seed
from .fisher_memory.artifacts import ArtifactError, Store, iter_units
from .fisher_memory.config import ROOT, content_hash, scientific_sources
from .fisher_memory.control import ResourceLimit, RuntimeControl, StopRequested, utc_now
from .fisher_memory.regularization_report import CONFIRMATION, _final_results, _read, collect
from .fisher_memory.run import budget_lock, file_bytes
from .fisher_memory.report import MAX_NOTEBOOK_BYTES

PROTOCOL = "plan15-efficiency-v1"
SCHEMA = 1
PLAN = "phase7e-plan-v1"
ANALYSIS = "phase7e-analysis-v1"
REVIEW = "phase7e-visual-review-v1"
METHODS = ("replay_ewc", "finite_difference")
SEED_NAME = "plan15.phase7e.whole_paired_replica_bootstrap"
CONFIG = dict(schema_version=SCHEMA, protocol=PROTOCOL, replicas=12, event_target=100,
              draws=2000, seed=derive_component_seed(20261024, SEED_NAME), seed_name=SEED_NAME,
              fit_start=1, fit_end=100, slope_zero_tolerance=1e-12,
              percentile_method="linear", target="fitted_replay_at_100",
              dtype="float64", max_extra_bytes=100_000_000)


def fit_curve(curve):
    """OLS with intercept on events 1..100; index zero is reference only."""
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (101,) or not np.isfinite(values).all() or (values < 0).any():
        raise ArtifactError("a finite nonnegative NLL curve at events 0..100 is required")
    n = np.arange(1, 101, dtype=np.float64)
    alpha, beta = np.linalg.lstsq(np.column_stack((np.ones(100), n)), values[1:], rcond=None)[0]
    fitted = alpha + beta * n
    return dict(intercept=float(alpha), slope=float(beta), fitted=fitted.tolist(),
                rmse=float(np.sqrt(np.mean((values[1:] - fitted) ** 2))))


def match_curves(replay, fd):
    """Retain numerical inversions even when their interpretation fails a guard."""
    fits = dict(replay_ewc=fit_curve(replay), finite_difference=fit_curve(fd))
    if not np.isclose(replay[0], fd[0], rtol=0, atol=1e-12):
        raise ArtifactError("paired methods do not share the initial reference")
    q = fits["replay_ewc"]["fitted"][-1]
    initial = float(replay[0])
    issues = []
    for method, fit in fits.items():
        if fit["slope"] >= -CONFIG["slope_zero_tolerance"]:
            issues.append(f"{method}_nonnegative_or_degenerate_slope")
        if not np.isfinite(fit["fitted"]).all() or min(fit["fitted"]) < 0:
            issues.append(f"{method}_invalid_fitted_nll")
    beta = fits["finite_difference"]["slope"]
    crossing = ((q - fits["finite_difference"]["intercept"]) / beta
                if abs(beta) > CONFIG["slope_zero_tolerance"] else None)
    if crossing is not None and not np.isfinite(crossing):
        crossing = None
    if crossing is None:
        issues.append("undefined_crossing")
    if initial <= q:
        issues.append("target_already_met_without_adaptation")
    in_range = crossing is not None and 1 <= crossing <= 100
    return dict(fits=fits, target=q, initial=initial, crossing=crossing,
                saved=None if crossing is None else 100 - crossing,
                saved_percent=None if crossing is None else 100 * (1 - crossing / 100),
                issues=issues, inversion_valid=not issues, in_range=in_range,
                status="inversion_invalid" if issues else "interpolated" if in_range else "range_limited")


def resample_indices(seed=CONFIG["seed"]):
    return np.random.default_rng(seed).integers(0, 12, size=(2000, 12)).tolist()


def analyze(curves, indices, check=lambda: None, progress=lambda count: None):
    """Recompute both fits, the target, and initialization in every paired draw."""
    arrays = {method: np.asarray(curves[method], dtype=np.float64) for method in METHODS}
    if any(array.shape != (12, 101) for array in arrays.values()):
        raise ArtifactError("Phase 7E requires twelve complete paired 100-event trajectories")
    if not np.allclose(arrays[METHODS[0]][:, 0], arrays[METHODS[1]][:, 0], rtol=0, atol=1e-12):
        raise ArtifactError("unpaired initial references")
    samples = np.asarray(indices)
    if (samples.shape != (2000, 12) or samples.dtype.kind not in "iu"
            or (samples < 0).any() or (samples >= 12).any()):
        raise ArtifactError("invalid whole-paired-replica bootstrap indices")
    means = {method: array.mean(axis=0).tolist() for method, array in arrays.items()}
    point = match_curves(means[METHODS[0]], means[METHODS[1]])
    draws = []
    for i, sample in enumerate(samples):
        check()
        value = match_curves(*(arrays[method][sample].mean(axis=0) for method in METHODS))
        # Per-draw coefficients suffice; mean/fitted trajectories are retained for the point fit.
        value["fits"] = {method: {k: v for k, v in fit.items() if k != "fitted"}
                         for method, fit in value["fits"].items()}
        draws.append(dict(draw=i, **value))
        if (i + 1) % 100 == 0:
            progress(i + 1)
    invalid = sum(not draw["inversion_valid"] for draw in draws)
    outside = sum(draw["crossing"] is not None and not draw["in_range"] for draw in draws)
    ci = None
    if not invalid:
        ci = {field: np.quantile([draw[field] for draw in draws], [.025, .975],
                                 method=CONFIG["percentile_method"]).tolist()
              for field in ("crossing", "saved", "saved_percent")}
    interval_in_range = ci is not None and ci["crossing"][0] >= 1 and ci["crossing"][1] <= 100
    return (dict(schema_version=SCHEMA, protocol=PROTOCOL, means=means, point=point,
                 bootstrap=dict(draws=len(draws), invalid=invalid, outside_range=outside,
                                issues=dict(Counter(issue for draw in draws for issue in draw["issues"])),
                                ci=ci, interval_in_range=interval_in_range,
                                interval_status="unavailable_invalid_draws" if invalid else
                                "interpolated" if interval_in_range else "range_limited")), draws)


def dependency_snapshot(roots, *, full=False, check=lambda: None):
    """Read-only seals plus full checksums when executing, never modify dependencies."""
    result = {}
    for root in roots:
        check()
        root = Path(root).resolve()
        units = list(iter_units(root, full=full, check=check))
        if any(unit["status"] != "completed" for unit in units):
            raise ArtifactError(f"unfinished dependency: {root}")
        result[str(root)] = dict(identity=_read(root / "identity.json"),
                                 seals={unit["unit_id"]: unit["completion"]["seal_hash"] for unit in units})
    return result


def load_source(root):
    """Require the sealed native 7D ledger and exactly its fresh cohort."""
    data = collect(root)
    result = _final_results(data)
    ledger = data["decisions"].get("confirmation-ledger", {}).get("result", {})
    treatments = ledger.get("treatments", [])
    if (result is None or ledger.get("replicas") != 12 or ledger.get("event_target") != 100
            or ledger.get("angle") != 30 or ledger.get("smoke") is True
            or len(treatments) != 2 or {t["condition"] for t in treatments} != set(METHODS)
            or any(t["ridge_multiplier"] != 1 or t["strength"] !=
                   (160.89 if t["condition"] == "finite_difference" else 0) for t in treatments)
            or result.get("n_pairs") != 12 or data["development"] is None):
        raise ArtifactError("Phase 7E needs the complete twelve-replica selected 7D experiment")
    curves = {}
    for method in METHODS:
        units = sorted((unit for unit in data["trajectories"]
                        if unit["metadata"]["family"] == CONFIRMATION
                        and unit["metadata"]["stream"] == "mixed"
                        and unit["metadata"]["condition"] == method), key=lambda u: u["metadata"]["replica"])
        if [unit["metadata"]["replica"] for unit in units] != list(range(12)):
            raise ArtifactError("missing or duplicated mixed-training paired replicas")
        curves[method] = [[row["rotated_nll"] for row in unit["rows"]] for unit in units]
        arm = next(arm for arm in result["arm_summaries"]
                   if arm["condition"] == method and arm["stream"] == "mixed")
        if not np.isclose(np.mean(np.asarray(curves[method])[:, 1:]), arm["rotated_auc"], rtol=0, atol=1e-12):
            raise ArtifactError("source trajectories disagree with original sealed acquisition NLL")
    gain = np.mean(np.asarray(curves[METHODS[0]])[:, 1:] - np.asarray(curves[METHODS[1]])[:, 1:])
    if not np.isclose(gain, result["primary"]["mean_gain"], rtol=0, atol=1e-12):
        raise ArtifactError("source trajectories disagree with original sealed primary contrast")
    return dict(curves=curves, established=result,
                roots=[str(Path(root).resolve()), str(data["development"]["root"].resolve())])


def analysis_sources():
    names = ["mnist_experiment/plan15_efficiency.py", "src/seeding.py"]
    names += [f"mnist_experiment/fisher_memory/{name}.py" for name in
              ("artifacts", "config", "control", "run", "report", "regularization_report")]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def provenance():
    import torch
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                cuda=torch.version.cuda, device="cpu", dtype="float64",
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                sources=analysis_sources(),
                report_source_hash=hashlib.sha256((ROOT / "mnist_experiment/plan15_efficiency_report.py").read_bytes()).hexdigest())


def other_study_seconds(budget_root, root):
    total = 0.
    for peer in Path(budget_root).iterdir():
        if peer.resolve() == Path(root).resolve() or not (peer / "identity.json").is_file():
            continue
        protocol = _read(peer / "identity.json")["identity"].get("protocol")
        if protocol in (PROTOCOL, "plan15-regularization-v1") or peer.name == "phase7_acceptance_v1":
            seconds = _read(peer / "runtime.json")["cumulative_active_seconds"]
            if not isinstance(seconds, (int, float)) or seconds < 0:
                raise ArtifactError("invalid peer compute ledger")
            total += seconds
    return total


def execute(store, inputs, dependencies, *, fit_review=None, fit_note=None):
    """Small transactional units; a halt may repeat only the unfinished bootstrap."""
    meta = dict(schema_version=SCHEMA, protocol=PROTOCOL)
    if store.begin(PLAN, dict(meta, kind="efficiency_plan")) is not None:
        store.write_json(PLAN, "plan.json", dict(config=CONFIG, indices=resample_indices(),
                                               dependencies=dependencies, inputs=inputs, provenance=provenance()))
        store.complete(PLAN)
    plan = _read(store.unit_path(PLAN) / "plan.json")
    if (plan["config"] != CONFIG or plan["dependencies"] != dependencies or plan["inputs"] != inputs
            or plan["indices"] != resample_indices()):
        raise ArtifactError("sealed Phase 7E plan/inputs differ on resume")

    def progress(count):
        store.write_json(None, "status.json", dict(schema_version=SCHEMA, stage="7E", status="running",
                                                   completed=count, expected=2000, heartbeat=utc_now(),
                                                   active_seconds=store.control.active_seconds))
        store.heartbeat(force=True)

    if store.begin(ANALYSIS, dict(meta, kind="efficiency_analysis")) is not None:
        started = utc_now()
        progress(0)
        result, draws = analyze(inputs["curves"], plan["indices"], check=store.check, progress=progress)
        store.write_json(ANALYSIS, "result.json", dict(result, config_hash=content_hash(CONFIG),
                                                       dependencies_hash=content_hash(dependencies),
                                                       established=inputs["established"],
                                                       started_at=started, completed_at=utc_now(), status="completed"))
        store.write_json(ANALYSIS, "draws.json", draws)
        store.complete(ANALYSIS)
    if fit_review is not None:
        if not fit_note or not fit_note.strip():
            raise ValueError("a visual fit review requires --fit-note")
        value = dict(schema_version=SCHEMA, assessment=fit_review, note=fit_note.strip())
        if store.begin(REVIEW, dict(meta, kind="efficiency_visual_review")) is not None:
            store.write_json(REVIEW, "review.json", dict(value, reviewed_at=utc_now()))
            store.complete(REVIEW)
        elif any(_read(store.unit_path(REVIEW) / "review.json")[key] != val for key, val in value.items()):
            raise ArtifactError("visual review already sealed; a changed review needs a new analysis root")
    return _read(store.unit_path(ANALYSIS) / "result.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets",
                        help="read-only dataset size accounting; no data are loaded")
    parser.add_argument("--notebook", type=Path, default=ROOT / "mnist_experiment/fisher_memory_results.ipynb")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=900)
    parser.add_argument("--fit-review", choices=("approximate", "poor"))
    parser.add_argument("--fit-note")
    args = parser.parse_args(argv)
    args.root, args.source_root, args.budget_root = (p.resolve() for p in (args.root, args.source_root, args.budget_root))
    if (not args.root.is_relative_to(args.budget_root) or args.root == args.budget_root
            or args.root == args.source_root or args.source_root.is_relative_to(args.root)
            or args.root.is_relative_to(args.source_root)):
        parser.error("analysis must be a separate child of the shared Plan 15 budget root")
    if args.clear_stop and not args.resume:
        parser.error("--clear-stop requires --resume")
    if args.fit_review and not args.fit_note:
        parser.error("--fit-review requires --fit-note")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plan15-matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    started = time.monotonic()
    # Store clears an authorized STOP only after checking the resume identity.
    preflight = RuntimeControl(args.root / ".preflight" if args.clear_stop else args.root,
                               max_wall_seconds=args.max_wall_seconds)
    with budget_lock(args.budget_root), preflight.signal_handlers():
        preflight.check()
        inputs = load_source(args.source_root)
        dependencies = dependency_snapshot(inputs["roots"], full=True, check=preflight.check)
        source_identity = dependencies[str(args.source_root)]["identity"]["identity"]
        frozen_source = content_hash(scientific_sources())
        if source_identity["source_hash"] != frozen_source:
            raise ArtifactError("frozen 7A--7D scientific source identity changed")
        identity = dict(protocol=PROTOCOL, schema_version=SCHEMA, config_hash=content_hash(CONFIG),
                        source_hash=content_hash(analysis_sources()), dependencies_hash=content_hash(dependencies))
        other_bytes = file_bytes(args.budget_root, exclude=args.root)
        if not args.data_root.resolve().is_relative_to(args.budget_root):
            other_bytes += file_bytes(args.data_root)
        other_seconds = other_study_seconds(args.budget_root, args.root)
        remaining_seconds = min(900, 26 * 3600 - other_seconds)
        if remaining_seconds <= 0:
            raise ResourceLimit("shared Phase 7 active-compute allocation exhausted")
        # The notebook's current copy and replacement temp coexist; bound both.
        exports = 2 * MAX_NOTEBOOK_BYTES + sum(p.stat().st_size for p in args.notebook.parent.glob(".plan15-report-*"))
        maximum = min(500_000_000_000, other_bytes + exports + CONFIG["max_extra_bytes"])
        preflight.check()
        preflight_seconds = time.monotonic() - started
        with Store(args.root, identity, resume=args.resume, clear_stop=args.clear_stop,
                   max_bytes=maximum, external_bytes=other_bytes + exports,
                   free_reserve_bytes=2_000_000_000, finalization_reserve_bytes=32_000_000,
                   max_wall_seconds=args.max_wall_seconds - (time.monotonic() - started),
                   max_active_seconds=remaining_seconds) as store, store.control.signal_handlers():
            store.control.base_seconds += preflight_seconds
            try:
                result = execute(store, inputs, dependencies, fit_review=args.fit_review, fit_note=args.fit_note)
                if dependency_snapshot(inputs["roots"], full=True, check=store.check) != dependencies:
                    raise ArtifactError("scientific dependencies changed during Phase 7E")
                from .plan15_efficiency_report import publish
                publish(store, args.notebook)
                store.write_json(None, "status.json", dict(schema_version=SCHEMA, stage="7E",
                    status="completed" if store.is_complete(REVIEW) else "awaiting_visual_review",
                    completed=2000, expected=2000, heartbeat=utc_now(), active_seconds=store.control.active_seconds,
                    aggregate_phase7_seconds=other_seconds + store.control.active_seconds,
                    unchanged_dependency_units=sum(len(dep["seals"]) for dep in dependencies.values()),
                    storage=store.storage_status()))
                print(json.dumps(dict(point={k: v for k, v in result["point"].items() if k != "fits"},
                                      bootstrap=result["bootstrap"], root=str(args.root)), indent=2))
            except (StopRequested, ResourceLimit) as exc:
                store.write_json(None, "status.json", dict(schema_version=SCHEMA, stage="7E", status="paused",
                                                           reason=str(exc), heartbeat=utc_now(),
                                                           active_seconds=store.control.active_seconds), finalizing=True)
                print(str(exc))
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
