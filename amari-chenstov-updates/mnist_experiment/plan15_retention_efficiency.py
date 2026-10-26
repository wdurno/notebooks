"""Phase 8.1E: CPU-only, immutable-artifact local sample equivalence.

Interpolation concerns predictive cohort-mean MSE, never the training loss.
All bootstrap samples contain whole paired replicas, not independent checkpoints.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from src.seeding import derive_component_seed
from .fisher_memory.artifacts import ArtifactError, Store, iter_units
from .fisher_memory.control import RuntimeControl, StopRequested, ResourceLimit, utc_now
from .fisher_memory.run import budget_lock
from .fisher_memory_regression.config import ROOT, Recipe, content_hash
from .fisher_memory_regression.runtime import RuntimeBudget, read_json, file_bytes
from .fisher_memory_retention import PROTOCOL as PARENT_PROTOCOL, ALLOCATION_FAMILY
from .fisher_memory_retention.run import peer_seconds, sources as parent_sources

PROTOCOL = "plan15-retention-sample-equivalence-v1"
METHODS = ("fd_replay_ewc", "replay_ewc")
FIELDS = ("novel_mse", "base_mse", "learning_seconds", "observations")
SEED_NAME = "plan15.phase81e.whole_paired_replica_bootstrap"
CONFIG = dict(protocol=PROTOCOL, schema_version=1, replicas=12, N=100,
              draws=2000, block_size=100, seed_name=SEED_NAME,
              acceptance_reserve_seconds=60.,
              seed=derive_component_seed(20261025, SEED_NAME),
              target="observed_mean_replay_MSE_at_100",
              interpolation="adjacent_observed_checkpoints_only",
              crossing="last_entry_persisting_through_observed_endpoint",
              interval="percentile_95_only_if_all_draws_valid", dtype="float64")
INTERVAL_FIELDS = ("matching_budget", "saved_novel", "saved_percent", "learning_time_ratio",
                   "fd_learning_seconds", "replay_learning_seconds", "fd_observations",
                   "replay_observations", "fd_old_rmse", "replay_old_rmse", "old_rmse_difference")


def crossing(curve, target):
    """Last interpolated entry below target, with every earlier entry retained."""
    y = np.asarray(curve, dtype=np.float64)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all() or (y < 0).any():
        raise ArtifactError("finite nonnegative predictive MSE curve required")
    if not np.isfinite(target) or target < 0:
        raise ArtifactError("finite nonnegative target required")
    entries = [float(k + (y[k] - target) / (y[k] - y[k + 1]))
               for k in range(len(y) - 1) if y[k] > target and y[k + 1] <= target]
    status = "valid"
    budget = None
    if y[0] <= target:
        status = "target_already_met_at_initialization"
    elif y[-1] > target:
        status = "target_not_reached_at_observed_endpoint"
    else:
        budget = entries[-1]
        if budget < 1:
            status = "crossing_below_first_post_observation_checkpoint"
    return dict(status=status, matching_budget=budget, downward_crossings=entries,
                multiple_crossings=len(entries) > 1, target_mse=float(target), initial_mse=float(y[0]))


def validate_arrays(inputs):
    arrays = {m: {f: np.asarray(inputs[m][f], dtype=np.float64) for f in FIELDS} for m in METHODS}
    shape = arrays[METHODS[0]][FIELDS[0]].shape
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 3:
        raise ArtifactError("complete paired replica-by-checkpoint arrays required")
    if any(v.shape != shape or not np.isfinite(v).all() or (v < 0).any()
           for fields in arrays.values() for v in fields.values()):
        raise ArtifactError("nonfinite, negative or inconsistent trajectory arrays")
    for method in METHODS:
        for field in ("learning_seconds", "observations"):
            values = arrays[method][field]
            if (values[:, 0] != 0).any() or (np.diff(values, axis=1) < 0).any():
                raise ArtifactError("cumulative costs/exposures must start at zero and not decrease")
        if (arrays[method]["learning_seconds"][:, -1] <= 0).any():
            raise ArtifactError("positive measured learning time required")
    for field in ("novel_mse", "base_mse"):
        if not np.allclose(arrays[METHODS[0]][field][:, 0], arrays[METHODS[1]][field][:, 0],
                           rtol=1e-12, atol=1e-12):
            raise ArtifactError("unpaired initial prediction panels")
    if not np.array_equal(arrays[METHODS[0]]["observations"], arrays[METHODS[1]]["observations"]):
        raise ArtifactError("unpaired stream exposures")
    return arrays


def match_means(means):
    fd, replay = (means[m] for m in METHODS)
    result = crossing(fd["novel_mse"], float(replay["novel_mse"][-1]))
    if result["status"] != "valid":
        return result
    budget = result["matching_budget"]
    grid = np.arange(len(fd["novel_mse"]))
    N = len(grid) - 1
    fd_time = float(np.interp(budget, grid, fd["learning_seconds"]))
    replay_time = float(replay["learning_seconds"][-1])
    fd_old = float(np.sqrt(np.interp(budget, grid, fd["base_mse"])))
    replay_old = float(np.sqrt(replay["base_mse"][-1]))
    result.update(saved_novel=N - budget, saved_percent=100 * (1 - budget / N),
        fd_learning_seconds=fd_time, replay_learning_seconds=replay_time,
        learning_time_ratio=fd_time / replay_time,
        same_budget_time_ratio=float(fd["learning_seconds"][-1] / replay_time),
        fd_observations=float(np.interp(budget, grid, fd["observations"])),
        replay_observations=float(replay["observations"][-1]),
        fd_old_rmse=fd_old, replay_old_rmse=replay_old, old_rmse_difference=fd_old - replay_old)
    return result


def sampled_result(arrays, indices=None):
    means = {m: {f: a.mean(axis=0) if indices is None else a[indices].mean(axis=0)
                 for f, a in fields.items()} for m, fields in arrays.items()}
    return match_means(means)


def summarize(arrays, draws):
    point = sampled_result(arrays)
    invalid = sum(d["status"] != "valid" for d in draws)
    ci = None if invalid else {field: np.quantile([d[field] for d in draws], [.025, .975]).tolist()
                              for field in INTERVAL_FIELDS}
    return dict(schema_version=1, protocol=PROTOCOL, point=point,
        means={m: {f: a.mean(axis=0).tolist() for f, a in fields.items()} for m, fields in arrays.items()},
        bootstrap=dict(draws=len(draws), invalid=invalid, statuses=dict(Counter(d["status"] for d in draws)),
            multiple_crossing_draws=sum(d["multiple_crossings"] for d in draws), ci=ci,
            interval_status="unavailable_unresolved_draws" if invalid else "nominal_percentile_95"),
        interpretation="Post hoc equivalence of interpolated cohort-mean risk; not individual first-passage times, "
            "guaranteed future samples saved, a pure novel-label intervention, or equal-compute superiority.",
        assumptions="Independent paired replicas, adequate squared-error moments and a stable local crossing. "
            "Intervals omit post hoc design uncertainty and unobserved between-checkpoint behavior. "
            "Unequal-budget old-domain endpoint errors do not extend the original H equivalence certificate.")


def dependency_snapshot(root, check=lambda: None):
    units = list(iter_units(root, full=True, check=check))
    if any(u["status"] != "completed" for u in units):
        raise ArtifactError("all source computational units must be complete")
    return dict(identity=read_json(root / "identity.json"),
                seals={u["unit_id"]: u["completion"]["seal_hash"] for u in units})


def load_source(root, dependencies):
    identity = dependencies["identity"]["identity"]
    if identity.get("protocol") != PARENT_PROTOCOL or identity.get("role") != "production":
        raise ArtifactError("completed production Phase 8.1 source required")
    seals = dependencies["seals"]
    if not all("decision-" + name in seals for name in ("analysis", "closeout", "confirmation-manifest")):
        raise ArtifactError("source confirmation, analysis and closeout must be sealed")
    read = lambda name: read_json(root / "units" / ("decision-" + name) / "result.json")
    manifest, analysis = read("confirmation-manifest"), read("analysis")
    unhashed = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if (manifest.get("protocol") != PARENT_PROTOCOL or manifest.get("sealed") is not True
            or manifest.get("replicas") != list(range(12)) or manifest.get("N") != 100
            or content_hash(unhashed) != manifest.get("manifest_hash")
            or analysis.get("manifest_hash") != manifest["manifest_hash"]
            or analysis.get("status") != "completed"):
        raise ArtifactError("incompatible or incomplete frozen confirmation cohort")
    recipes = [Recipe.from_mapping(r) for r in manifest["recipes"]]
    if len(recipes) != 3 or {r.method for r in recipes} != {"ewc_only", *METHODS}:
        raise ArtifactError("all three frozen confirmation policies are required")
    all_records = []
    for unit in seals:
        path = root / "units" / unit
        meta = read_json(path / "manifest.json")["metadata"]
        if meta.get("kind") == "regression_trajectory" and meta.get("partition") == "confirmation":
            all_records.append({**read_json(path / "summary.json"), "metrics": read_json(path / "metrics.json")})
    # Reuse the native complete-cohort validator and compare, without overwriting its analysis.
    from .fisher_memory_retention.analysis import analyze
    checked = analyze(all_records, manifest, seed=manifest["bootstrap_seed"], draws=2)
    for method, item in checked["methods"].items():
        if not np.allclose(item["curves"]["novel"]["mean_mse"],
                           analysis["methods"][method]["curves"]["novel"]["mean_mse"], rtol=1e-12, atol=1e-12):
            raise ArtifactError("source analysis disagrees with sealed predictions")
    inputs = {}
    for method in METHODS:
        records = sorted((r for r in all_records if r["method"] == method), key=lambda r: r["replica"])
        inputs[method] = {f: [[row[f] for row in r["metrics"]] for r in records] for f in FIELDS}
    arrays = validate_arrays(inputs)
    return arrays, manifest, analysis


def analysis_sources():
    names = parent_sources()
    path = Path(__file__)
    return {**names, str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()}


def execute(store, arrays, dependencies, manifest, established, *, max_units=None, check=None):
    check = check or store.check
    config_hash = content_hash(CONFIG)
    inputs_hash = content_hash({m: {f: a.tolist() for f, a in fs.items()} for m, fs in arrays.items()})
    indices = np.random.default_rng(CONFIG["seed"]).integers(0, 12, size=(CONFIG["draws"], 12))
    added = 0

    def seal(unit, kind, value):
        nonlocal added
        metadata = dict(kind=kind, schema_version=1, config_hash=config_hash, inputs_hash=inputs_hash)
        if store.begin(unit, metadata) is None:
            return read_json(store.unit_path(unit) / "result.json")
        store.write_json(unit, "result.json", value)
        store.complete(unit)
        added += 1
        if max_units and added >= max_units:
            raise StopRequested("requested completed-unit batch reached")
        return value

    plan = dict(config=CONFIG, inputs_hash=inputs_hash, dependencies=dependencies,
                bootstrap_indices=indices.tolist(), parent_manifest_hash=manifest["manifest_hash"])
    if seal("design", "sample_equivalence_design", plan) != plan:
        raise ArtifactError("sealed local-equivalence design changed")
    draws = []
    for start in range(0, CONFIG["draws"], CONFIG["block_size"]):
        check()
        unit = f"bootstrap-{start:04d}"
        if store.is_complete(unit):
            block = read_json(store.unit_path(unit) / "result.json")
        else:
            block = []
            for i in range(start, min(start + CONFIG["block_size"], CONFIG["draws"])):
                check()
                block.append(dict(draw=i, **sampled_result(arrays, indices[i])))
            block = seal(unit, "paired_bootstrap_block", block)
        if [r["draw"] for r in block] != list(range(start, start + CONFIG["block_size"])):
            raise ArtifactError("bootstrap block cursor mismatch")
        draws.extend(block)
        store.write_json(None, "status.json", dict(stage="8.1E bootstrap", status="running",
            completed=len(draws), planned=CONFIG["draws"], heartbeat=utc_now()))
    result = summarize(arrays, draws)
    result.update(parent_manifest_hash=manifest["manifest_hash"],
        original_primary=established["comparisons"]["fd_replay_ewc_vs_replay_ewc"],
        bootstrap_seed=CONFIG["seed"], inputs_hash=inputs_hash)
    if seal("analysis", "local_sample_equivalence", result) != result:
        raise ArtifactError("sealed sample-equivalence result changed")
    return result


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, default=ROOT / "cache/mnist_experiment/plan15")
    parser.add_argument("--data-root", type=Path, default=ROOT / "cache/mnist_experiment/datasets")
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-wall-seconds", type=float, default=600.)
    args = parser.parse_args(argv)
    if args.max_units is not None and args.max_units <= 0:
        raise ValueError("positive completed-unit batch size required")
    if args.max_wall_seconds <= 0 or args.max_wall_seconds > 600:
        raise ValueError("follow-up invocation must fit ten minutes")
    if args.root.resolve().parent != args.budget_root.resolve() or args.root.resolve() == args.source_root.resolve():
        raise ValueError("new analysis root must be directly inside shared budget and differ from source")
    started = time.monotonic()
    preflight = RuntimeControl(args.root, max_wall_seconds=args.max_wall_seconds)
    def preflight_check():
        if not args.clear_stop:
            preflight.check()
        elif preflight.stop_requested:
            raise StopRequested("stop requested during preflight")
        elif time.monotonic() - started >= args.max_wall_seconds:
            raise ResourceLimit("invocation wall-clock allocation exhausted")
    with budget_lock(args.budget_root), preflight.signal_handlers():
        preflight_check()
        dependencies = dependency_snapshot(args.source_root, preflight_check)
        arrays, manifest, established = load_source(args.source_root, dependencies)
        sources = analysis_sources()
        identity = dict(protocol=PROTOCOL, schema_version=1, allocation_family=ALLOCATION_FAMILY, role="analysis",
            config_hash=content_hash(CONFIG), source_hash=content_hash(sources), dependencies_hash=content_hash(dependencies))
        other = peer_seconds(args.budget_root, args.root)
        remaining = min(600., 20 * 3600 - other)
        if remaining <= 0:
            raise ResourceLimit("Phase 8.1 aggregate active-compute allowance exhausted")
        external = file_bytes(args.budget_root, exclude=args.root, check=preflight_check)
        if not args.source_root.resolve().is_relative_to(args.budget_root.resolve()):
            external += file_bytes(args.source_root, check=preflight_check)
        if not args.data_root.resolve().is_relative_to(args.budget_root.resolve()):
            external += file_bytes(args.data_root, check=preflight_check)
        external += 48_000_000
        with Store(args.root, identity, resume=args.resume, clear_stop=args.clear_stop,
            max_bytes=500_000_000_000, external_bytes=external, free_reserve_bytes=2_000_000_000,
            finalization_reserve_bytes=32_000_000, max_active_seconds=remaining,
            max_wall_seconds=max(.001, args.max_wall_seconds - (time.monotonic() - started))) as store:
            store.control.restore_active_seconds(store.control.active_seconds + time.monotonic() - started)
            if not (store.root / "provenance.json").exists():
                store.control.restore_active_seconds(store.control.active_seconds + CONFIG["acceptance_reserve_seconds"])
            budget = RuntimeBudget(store, other_seconds=other, allocation_seconds=20 * 3600)
            try:
                with store.control.signal_handlers():
                    if not (store.root / "provenance.json").exists():
                        store.write_json(None, "provenance.json", dict(config=CONFIG, sources=sources,
                            python=platform.python_version(), numpy=np.__version__, device="cpu", dtype="float64",
                            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                            git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))))
                        report = "mnist_experiment/plan15_retention_efficiency_report.py"
                        store.write_json(None, "source_snapshot.json", {name: (ROOT / name).read_text()
                            for name in (*sources, report)})
                    result = execute(store, arrays, dependencies, manifest, established,
                                     max_units=args.max_units, check=budget.check)
                    if dependency_snapshot(args.source_root, budget.check) != dependencies:
                        raise ArtifactError("source computational units changed during follow-up")
                    if store.begin("closeout", dict(kind="sample_equivalence_closeout")) is not None:
                        for unit in store.iter_units():
                            budget.check()
                            if unit["unit_id"] != "closeout" and not store.is_complete(unit["unit_id"], full=True):
                                raise ArtifactError("incomplete analysis unit at closeout")
                        store.write_json("closeout", "result.json", dict(parent_seals_unchanged=True,
                            parent_units=len(dependencies["seals"]), bootstrap_draws=CONFIG["draws"]))
                        store.complete("closeout")
                    from .plan15_retention_efficiency_report import publish
                    publish(result, store, args.notebook)
                    store.write_json(None, "status.json", dict(stage="8.1E", status="completed",
                        completed=CONFIG["draws"], planned=CONFIG["draws"], heartbeat=utc_now(),
                        active_seconds=budget.used, storage=store.storage_status()))
                    print(json.dumps(dict(point=result["point"], bootstrap=result["bootstrap"]), indent=2))
            except (StopRequested, ResourceLimit) as exc:
                store.write_json(None, "status.json", dict(stage="8.1E", status="paused", reason=str(exc),
                    heartbeat=utc_now(), active_seconds=budget.used), finalizing=True)
                return 2
            finally:
                budget.close()
    return 0


def main(argv=None):
    try:
        return _main(argv)
    except (StopRequested, ResourceLimit) as exc:
        print(json.dumps(dict(status="paused", reason=str(exc))))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
