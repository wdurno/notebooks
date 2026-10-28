"""Phase 9 paired, observed-curve inference; no training or artifact mutation.

MSE defines acquisition and matching. H and matched nominal RMSE take square
roots *within* replicas. Source completion seals are checked by the Store caller.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import math

import numpy as np

from mnist_experiment.fisher_memory_regression.analysis import (
    flatten_record, replica_draw_indices, summarize_trajectory,
)
from mnist_experiment.fisher_memory_regression.config import METHODS, Recipe, content_hash

from . import PROTOCOL

FD, REPLAY, EWC = METHODS
SUMMARY_FIELDS = ("novel_auc", "base_auc", "frozen_base_mse", "frozen_novel_mse",
                  "observations", "learning_seconds")
MATCH_FIELDS = ("matching_budget", "saved_novel", "saved_fraction", "saved_percent",
                "learning_time_ratio", "S", "fd_learning_seconds", "replay_learning_seconds",
                "fd_observations", "replay_observations", "fd_old_rmse", "replay_old_rmse",
                "old_rmse_difference", "fd_old_deterioration", "replay_old_deterioration")


def _number(value, name, *, minimum=0., infinity=False):
    if isinstance(value, (str, bool)):
        raise ValueError(f"{name} must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if math.isnan(value) or value < minimum or (not infinity and not math.isfinite(value)):
        raise ValueError(f"invalid {name}")
    return value


def _replicas(values):
    values = list(values)
    if not values or any(type(r) is not int or r < 0 for r in values) or len(set(values)) != len(values):
        raise ValueError("distinct nonnegative integer replicas required")
    return sorted(values)


def record_context(row):
    family = row.get("family", "")
    if not isinstance(family, str) or "--" not in family:
        raise ValueError("family must encode stage--context_key")
    stage, key = family.split("--", 1)
    if not stage or not key or row.get("context_key", key) != key:
        raise ValueError("context identity disagrees with family")
    return key


def validate_records(records, N, partition, *, context_key=None, check=lambda: None):
    """Validate all supplied evidence, including invalid-policy diagnostics.

    Never relabel a family or silently remove a failed replica. The returned
    flat dictionaries are new objects; original records are not modified.
    """
    rows, seen, contexts, paired, plans = [], set(), set(), {}, {}
    try:
        for raw in records:
            check()
            row = flatten_record(raw)
            if row.get("protocol") != PROTOCOL or row.get("schema_version") != 1:
                raise ValueError("incompatible trajectory protocol/schema")
            if row.get("partition") != partition:
                raise ValueError(f"{partition}-only evidence required; confirmation cannot influence development")
            key = record_context(row)
            contexts.add(key)
            if context_key is not None and key != context_key:
                raise ValueError("trajectory differs from frozen context")
            if partition == "confirmation" and row["family"] != f"confirmation--{key}":
                raise ValueError("confirmation family must be confirmation--context_key")
            if row.get("status", "completed") not in ("completed", "complete"):
                raise ValueError("incomplete trajectory")
            replica = row["replica"]
            _replicas([replica])
            recipe = Recipe.from_mapping(row["recipe"])
            if (row["recipe_key"] != recipe.key or row["method"] != recipe.method
                    or row["capacity"] != recipe.capacity or row["replay_count"] != recipe.replay_count):
                raise ValueError("trajectory recipe identity mismatch")
            identity = recipe.key, replica
            if identity in seen:
                raise ValueError("duplicate recipe/replica trajectory")
            seen.add(identity)
            if type(row["target"]) is not int or row["target"] != N or row.get("N", N) != N:
                raise ValueError("complete full-budget trajectory required")
            if type(row["valid"]) is not bool:
                raise ValueError("explicit numerical validity required")
            metrics = row["metrics"]
            if not metrics or metrics[0]["n"] != 0:
                raise ValueError("initial prediction checkpoint required")
            summary = summarize_trajectory(metrics, N=N,
                frozen_base_mse=row["frozen_base_mse"], frozen_novel_mse=row["frozen_novel_mse"])
            for name in SUMMARY_FIELDS:
                if not math.isclose(_number(row[name], name), summary[name], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"summary disagrees with complete metrics: {name}")
            for name in ("active_seconds", "peak_vram_bytes"):
                _number(row[name], name)
            plan = row["plan_hash"]
            if not isinstance(plan, str) or not plan or row.get("pairing_id") != plan:
                raise ValueError("missing or inconsistent pairing provenance")
            if plans.setdefault(plan, replica) != replica:
                raise ValueError("distinct replicas cannot reuse a data plan")
            reference = paired.setdefault(replica, row)
            if reference["plan_hash"] != plan or any(not math.isclose(row[k], reference[k],
                    rel_tol=1e-10, abs_tol=1e-10) for k in ("frozen_base_mse", "frozen_novel_mse")):
                raise ValueError("unpaired initialization, panels or data plans")
            if [m["observations"] for m in metrics] != [m["observations"] for m in reference["metrics"]]:
                raise ValueError("unpaired exposure counts")
            for field in ("gamma", "blur", "gamma_max", "blur_max", "strength", "u", "drift_strength"):
                if any(field in m for m in (*metrics, *reference["metrics"])):
                    if not all(field in m for m in (*metrics, *reference["metrics"])) or any(
                        not math.isclose(_number(a[field], field), _number(b[field], field), rel_tol=1e-10, abs_tol=1e-10)
                        for a, b in zip(metrics, reference["metrics"])):
                        raise ValueError("unpaired drift trajectory")
            row = {**row, "recipe": recipe.mapping(), "context_key": key}
            rows.append(row)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"malformed or incomplete trajectory: {error}") from error
    if len(contexts) > 1:
        raise ValueError("filter records to one context before analysis/selection")
    return rows


def crossing(curve, target):
    """Last downward entry persisting to N; retain all observed entries.

    This is the pure 8.1E crossing rule, repeated here to avoid importing its
    CLI, runtime and frozen-source dependencies into Phase 9 analysis.
    """
    y = np.asarray(curve, dtype=float)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("finite nonnegative MSE curve required")
    target = _number(target, "target MSE")
    entries = [float(k + (y[k] - target) / (y[k] - y[k + 1]))
               for k in range(len(y) - 1) if y[k] > target and y[k + 1] <= target]
    budget, status = None, "valid"
    if y[0] <= target:
        status = "target_already_met_at_initialization"
    elif y[-1] > target:
        status = "target_not_reached_at_observed_endpoint"
    else:
        budget = entries[-1]
        if budget < 1:
            status = "crossing_below_first_post_observation_checkpoint"
    return dict(status=status, matching_budget=budget, downward_crossings=entries,
                multiple_crossings=len(entries) > 1, target_mse=target, initial_mse=float(y[0]))


def _arrays(groups):
    return {method: {field: np.asarray([[m[field] for m in row["metrics"]] for row in rows], dtype=float)
            for field in ("novel_mse", "base_mse", "learning_seconds", "observations")}
            for method, rows in groups.items()}


def _point(arrays, indices=None):
    fd, replay = (arrays[m] for m in (FD, REPLAY))
    if indices is None:
        indices = np.arange(len(fd["novel_mse"]))
    mean = lambda arm, field: arm[field][indices].mean(axis=0)
    target = float(mean(replay, "novel_mse")[-1])
    initial = float(mean(replay, "novel_mse")[0])
    result = crossing(mean(fd, "novel_mse"), target)
    result.update(S=None, useful_target=bool(initial > 0 and target <= .95 * initial),
                  replay_target_rmse=math.sqrt(target),
                  target_mse_gain_fraction=None if initial == 0 else 1 - target / initial)
    if result["status"] != "valid":
        return result
    budget = result["matching_budget"]
    N = fd["novel_mse"].shape[1] - 1
    grid = np.arange(N + 1)
    interp = lambda arm, field: float(np.interp(budget, grid, mean(arm, field)))
    fd_time, replay_time = interp(fd, "learning_seconds"), float(mean(replay, "learning_seconds")[-1])
    if fd_time <= 0 or replay_time <= 0:
        result["status"] = "invalid_learning_time_denominator"
        return result
    # Interpolate each replica's MSE before taking its root, not sqrt(mean MSE).
    old_fd = np.sqrt([np.interp(budget, grid, curve) for curve in fd["base_mse"][indices]])
    old_replay = np.sqrt(replay["base_mse"][indices, -1])
    initial_old = np.sqrt(replay["base_mse"][indices, 0])
    ratio = fd_time / replay_time
    saved = (N - budget) / N
    if not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(saved / ratio):
        result["status"] = "invalid_learning_time_denominator"
        return result
    result.update(n_F=budget, n_R=N, saved_novel=N - budget, saved_fraction=saved,
        saved_percent=100 * saved, learning_time_ratio=ratio, S=saved / ratio,
        fd_learning_seconds=fd_time, replay_learning_seconds=replay_time,
        fd_observations=interp(fd, "observations"), replay_observations=float(mean(replay, "observations")[-1]),
        fd_old_rmse=float(old_fd.mean()), replay_old_rmse=float(old_replay.mean()),
        old_rmse_difference=float((old_fd - old_replay).mean()),
        fd_old_deterioration=float((old_fd - initial_old).mean()),
        replay_old_deterioration=float((old_replay - initial_old).mean()))
    for field in ("gamma", "blur", "gamma_max", "blur_max", "strength", "u", "drift_strength"):
        if field in fd:
            result[f"fd_final_{field}"] = interp(fd, field)
            result[f"replay_final_{field}"] = float(mean(replay, field)[-1])
    return result


def point_equivalence(records_by_method):
    """Bootstrap-free point score for an exact paired policy cohort.

    Keys must include FD and replay; EWC is optional here, but mandatory for
    context selection and final inference. No classifier or family is relabeled.
    """
    if not {FD, REPLAY} <= set(records_by_method) or set(records_by_method) - set(METHODS):
        raise ValueError("FD and replay policy records required")
    flat = [flatten_record(row) for group in records_by_method.values() for row in group]
    if not flat:
        raise ValueError("empty point cohort")
    N, partition = flat[0]["target"], flat[0]["partition"]
    rows = validate_records(flat, N, partition)
    groups = {method: sorted((r for r in rows if r["method"] == method), key=lambda r: r["replica"])
              for method in records_by_method}
    for method, raw in records_by_method.items():
        if any(flatten_record(r)["method"] != method for r in raw):
            raise ValueError("method group disagrees with trajectory")
    reference = groups[FD]
    if not reference or any([r["replica"] for r in group] != [r["replica"] for r in reference]
                            or len({r["recipe_key"] for r in group}) != 1 for group in groups.values()):
        raise ValueError("point score requires complete paired single-policy replicas")
    if not all(r["valid"] for r in rows):
        return dict(status="invalid_policy", S=None, useful_target=False)
    arrays = _arrays(groups)
    _add_drift(arrays, groups)
    return _point(arrays)


def _add_drift(arrays, groups):
    for field in ("gamma", "blur", "gamma_max", "blur_max", "strength", "u", "drift_strength"):
        if all(field in m for group in groups.values() for r in group for m in r["metrics"]):
            for method, rows in groups.items():
                arrays[method][field] = np.asarray([[_number(m[field], field) for m in r["metrics"]]
                                                   for r in rows])


def _cohort(records, manifest, check):
    from .config import Context

    if (manifest.get("protocol") != PROTOCOL or manifest.get("schema_version", 1) != 1
            or manifest.get("sealed", True) is not True):
        raise ValueError("incompatible or unsealed confirmation manifest")
    if not manifest.get("source_hash") or not manifest.get("manifest_hash"):
        raise ValueError("frozen source and manifest hashes required")
    if manifest["manifest_hash"] != content_hash({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("manifest hash disagrees with frozen design")
    replicas = _replicas(manifest["replicas"])
    N, smoke = manifest["N"], manifest.get("smoke", False)
    if type(smoke) is not bool or type(N) is not int:
        raise ValueError("explicit boolean smoke and integer N required")
    if not smoke and (len(replicas) not in (8, 10, 12) or N != 100):
        raise ValueError("final inference requires a frozen 8/10/12-replica, 100-event cohort")
    if smoke and (len(replicas) < 2 or not 2 <= N < 100):
        raise ValueError("smoke analysis requires at least two replicas and 2 <= N < 100")
    context = Context.from_mapping(manifest["context"])
    recipes = [Recipe.from_mapping(r) for r in manifest["recipes"]]
    if len(recipes) != 3 or {r.method for r in recipes} != set(METHODS):
        raise ValueError("all three unique frozen adaptive methods required")
    if any(raw.get("recipe_key") != r.key for raw, r in zip(manifest["recipes"], recipes)):
        raise ValueError("frozen recipe keys required")
    expected = {(r.key, replica) for r in recipes for replica in replicas}
    rows = validate_records(records, N, "confirmation", context_key=context.key, check=check)
    if any(not r["valid"] for r in rows):
        raise ValueError("invalid confirmation trajectory cannot be excluded")
    if {(r["recipe_key"], r["replica"]) for r in rows} != expected:
        raise ValueError("incomplete or unplanned frozen confirmation cohort")
    frozen = {r.key: r for r in recipes}
    if any(Recipe.from_mapping(r["recipe"]) != frozen[r["recipe_key"]] for r in rows):
        raise ValueError("confirmation recipe differs from frozen policy")
    return {m: sorted((r for r in rows if r["method"] == m), key=lambda r: r["replica"]) for m in METHODS}


def analyze(records, manifest, seed, draws=2000, check=lambda: None, *,
            load_bootstrap=None, save_bootstrap=None):
    """JSON-safe final analysis of the entire frozen three-arm cohort.

    Work is stop-aware in blocks of at most 100 whole-replica draws. An invalid
    inversion draw withholds *all* ordinary inversion/endpoint intervals.
    Full-window intervals remain available independently of inversion validity.
    """
    check()
    groups = _cohort(records, manifest, check)
    N, smoke = manifest["N"], manifest.get("smoke", False)
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed and at least two draws required")
    delta, epsilon = (_number(manifest[k], k) for k in ("delta", "epsilon"))
    limits = manifest.get("limits", manifest)
    learning_limit = _number(limits.get("learning_limit", limits.get("learning_time_limit")), "learning limit")
    memory_limit = _number(limits.get("memory_limit", limits.get("memory_ceiling_bytes")), "memory limit")
    replicas = [r["replica"] for r in groups[FD]]
    arrays = _arrays(groups)
    _add_drift(arrays, groups)
    samples = np.empty((draws, len(replicas)), dtype=int)
    inversions = []
    for start in range(0, draws, 100):
        check()
        end = min(start + 100, draws)
        block = load_bootstrap(start) if load_bootstrap else None
        expected = np.asarray([replica_draw_indices(len(replicas), seed, d)
                               for d in range(start, end)])
        if block is not None:
            cached = np.asarray(block["indices"])
            if (block["start"] != start or block["end"] != end or
                    not np.array_equal(cached, expected) or
                    [r["draw"] for r in block["inversions"]] != list(range(start, end))):
                raise ValueError("bootstrap checkpoint does not match its frozen draws")
            samples[start:end] = cached
            inversions.extend(block["inversions"])
        else:
            samples[start:end] = expected
            values = [dict(draw=d, **_point(arrays, samples[d])) for d in range(start, end)]
            inversions.extend(values)
            if save_bootstrap:
                save_bootstrap(start, dict(start=start, end=end,
                    indices=expected.tolist(), inversions=values))

    def interval(values):
        values = np.asarray(values, dtype=float)
        means = values[samples].mean(axis=1)
        return dict(mean=float(values.mean()), ci=np.quantile(means, [.025, .975]).tolist(),
            upper_one_sided_95=float(np.quantile(means, .95)), n_pairs=len(replicas),
            standard_error=float(values.std(ddof=1) / math.sqrt(len(replicas))))

    result = dict(schema_version=1, protocol=PROTOCOL, status="smoke_completed" if smoke else "completed",
        is_final=not smoke,
        manifest_hash=manifest["manifest_hash"], source_hash=manifest["source_hash"],
        context=manifest["context"], replicas=replicas, N=N, delta=delta, epsilon=epsilon,
        methods={}, comparisons={}, bootstrap=dict(draws=draws, seed=seed,
            resampling_unit="paired_whole_replica", confidence_level=.95,
            curve_bands="pointwise_not_simultaneous"),
        assumptions="Nominal paired inference conditional on frozen policies and reused benchmark panels; "
                    "independent replicas and adequate moments. No smoothing, global fit or extrapolation.")
    vectors = {}
    for method in METHODS:
        arms, fields = groups[method], arrays[method]
        acquisition = fields["novel_mse"][:, 1:].mean(axis=1)
        H = np.sqrt(fields["base_mse"][:, 1:].mean(axis=1)) - np.sqrt(fields["base_mse"][:, 0])
        vectors[method] = acquisition, H
        retention = interval(H)
        retention.update(point_ceiling_feasible=retention["mean"] <= delta,
                         ceiling_supported=retention["upper_one_sided_95"] <= delta)
        curves = dict(n=list(range(N + 1)), per_replica={"replicas": replicas})
        for domain in ("base", "novel"):
            matrix = fields[f"{domain}_mse"]
            boot = np.empty((draws, N + 1))
            for start in range(0, draws, 100):
                check()
                boot[start:start + 100] = matrix[samples[start:start + 100]].mean(axis=1)
            ci = np.quantile(boot, [.025, .975], axis=0)
            curves[domain] = dict(mean_mse=matrix.mean(axis=0).tolist(), ci_mse=ci.tolist(),
                mean_rmse=np.sqrt(matrix.mean(axis=0)).tolist(), ci_rmse=np.sqrt(ci).tolist())
            curves["per_replica"][f"{domain}_mse"] = matrix.tolist()
            for metric in ("mae", "accuracy", "accuracy_10", "accuracy10", "accuracy_within_10_degrees"):
                field = f"{domain}_{metric}"
                present = [field in m for a in arms for m in a["metrics"]]
                if any(present):
                    if not all(present):
                        raise ValueError(f"incomplete optional curve: {field}")
                    values = np.asarray([[_number(m[field], field) for m in a["metrics"]] for a in arms])
                    if metric.startswith("accuracy") and (values > 1).any():
                        raise ValueError("accuracy must be a fraction")
                    curves[domain][f"mean_{metric}"] = values.mean(axis=0).tolist()
                    curves["per_replica"][field] = values.tolist()
        for field in ("observations", "learning_seconds", "gamma", "blur", "gamma_max", "blur_max", "strength", "u", "drift_strength"):
            if field in fields:
                curves[field] = fields[field].mean(axis=0).tolist()
        seconds = float(fields["learning_seconds"][:, -1].sum() / fields["observations"][:, -1].sum())
        vram = max(a["peak_vram_bytes"] for a in arms)
        costs = dict(mean_learning_seconds=float(fields["learning_seconds"][:, -1].mean()),
            seconds_per_arrival=seconds, peak_vram_bytes=vram,
            resource_feasible=seconds <= learning_limit and vram <= memory_limit)
        for name in ("evaluation_seconds", "fit_seconds", "eviction_seconds", "input_seconds", "active_seconds", "audit_seconds"):
            if all(name in a for a in arms):
                costs[f"mean_{name}"] = float(np.mean([_number(a[name], name) for a in arms]))
        retired = [item for a in arms for item in a.get("retired_audit", [])]
        result["methods"][method] = dict(recipe=arms[0]["recipe"], recipe_key=arms[0]["recipe_key"],
            acquisition=interval(acquisition), retention=retention,
            frozen_gain=interval(fields["novel_mse"][:, 0] - acquisition), costs=costs, curves=curves,
            retirement_audit=dict(count=len(retired), mean_squared_error=float(np.mean([
                _number(i["squared_error"], "retired error") for i in retired])) if retired else None,
                interpretation="Descriptive retired training-cohort errors, not held-out retention."))
    for a, b in combinations(METHODS, 2):
        gain = interval(vectors[b][0] - vectors[a][0])
        difference = interval(vectors[a][1] - vectors[b][1])
        equivalent = difference["ci"][0] >= -epsilon and difference["ci"][1] <= epsilon
        result["comparisons"][f"{a}_vs_{b}"] = dict(first=a, second=b, acquisition_gain=gain,
            retention_difference=difference, retention_equivalence_supported=equivalent,
            retention_noninferiority_supported=difference["upper_one_sided_95"] <= epsilon,
            primary=(a == FD and b == REPLAY))
    primary = result["comparisons"][f"{FD}_vs_{REPLAY}"]
    point = _point(arrays)
    invalid = sum(d["status"] != "valid" for d in inversions)
    all_valid = point["status"] == "valid" and invalid == 0
    ci = {f: np.quantile([d[f] for d in inversions], [.025, .975]).tolist() for f in MATCH_FIELDS} if all_valid else None
    uppers = {f: float(np.quantile([d[f] for d in inversions], .95))
              for f in ("old_rmse_difference", "fd_old_deterioration", "replay_old_deterioration")} if all_valid else None
    full_ok = bool(primary["retention_noninferiority_supported"] and all(
        result["methods"][m]["retention"]["ceiling_supported"] for m in (FD, REPLAY)))
    matched_ok = bool(uppers is not None and uppers["old_rmse_difference"] <= epsilon
        and uppers["fd_old_deterioration"] <= delta and uppers["replay_old_deterioration"] <= delta)
    resource_ok = all(result["methods"][m]["costs"]["resource_feasible"] for m in (FD, REPLAY))
    result["efficiency"] = dict(point=point, retention_qualified=full_ok and matched_ok,
        full_budget_retention_supported=full_ok, matched_budget_retention_supported=matched_ok,
        full_budget_retention_label="equivalent" if primary["retention_equivalence_supported"] else (
            "FD_better_retention" if primary["retention_difference"]["ci"][1] < -epsilon else "not_equivalent"),
        matched_upper_one_sided_95=uppers, resource_feasible=resource_ok,
        bootstrap=dict(draws=draws, invalid=invalid, statuses=dict(Counter(d["status"] for d in inversions)),
            multiple_crossing_draws=sum(d["multiple_crossings"] for d in inversions), ci=ci,
            interval_status="nominal_percentile_95" if all_valid else "unavailable_unresolved_draws",
            draws_detail=inversions),
        evidence_positive_acquisition=primary["acquisition_gain"]["ci"][0] > 0,
        evidence_sample_savings=bool(not smoke and all_valid and ci["saved_fraction"][0] > 0 and full_ok and matched_ok
                                     and resource_ok and point["useful_target"]),
        point_reaches_025=bool(point["S"] is not None and point["S"] >= .25),
        interpretation="Complete-policy equivalence: matched budgets include different background exposure "
                       "and drift history. A point score alone does not confirm savings or S > 0.25.")
    result["G"] = primary["acquisition_gain"]
    return result
