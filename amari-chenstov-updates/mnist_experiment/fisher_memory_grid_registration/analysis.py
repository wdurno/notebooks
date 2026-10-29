"""Artifact-only Phase 11 inference, using squared-pixel error per landmark.

Each grid has 54 coordinates representing 27 two-dimensional landmarks.
Replica trajectories, rather than individual cells, are the resampling unit.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
from types import SimpleNamespace

import numpy as np

from mnist_experiment.fisher_memory_regression.analysis import (
    flatten_record, replica_draw_indices, summarize_trajectory as _summarize,
)
from mnist_experiment.fisher_memory_regression.config import METHODS, Recipe, content_hash
from src.seeding import derive_component_seed

from . import PROTOCOL

FD, REPLAY, EWC = METHODS
TASK_DIMS = {"grid": 54}
SUMMARY_FIELDS = ("novel_auc", "base_auc", "frozen_base_mse", "frozen_novel_mse",
                  "observations", "learning_seconds")
DRIFT_FIELDS = ("u", "p", "blur", "sigma", "drift_strength", "group_index")
MATCH_FIELDS = ("matching_budget", "saved_novel", "saved_fraction", "saved_percent",
                "learning_time_ratio", "S", "fd_learning_seconds", "replay_learning_seconds",
                "fd_observations", "replay_observations", "fd_old_rmse", "replay_old_rmse",
                "old_rmse_difference", "fd_old_deterioration", "replay_old_deterioration")


def summarize_trajectory(*args, **kwargs):
    """Reuse exact-count scalar arithmetic, but never emit the old protocol."""
    return {**_summarize(*args, **kwargs), "protocol": PROTOCOL}


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
        raise ValueError("family must encode stage--taskkey")
    stage, key = family.split("--", 1)
    if not stage or key not in TASK_DIMS or row.get("context_key", key) != key:
        raise ValueError("task identity disagrees with family/context_key")
    if type(row.get("output_dim")) is not int or row["output_dim"] != TASK_DIMS[key]:
        raise ValueError("output_dim disagrees with the 54-coordinate grid task")
    return key


def validate_records(records, N, partition, *, context_key=None, check=lambda: None):
    """Validate all evidence, never deleting invalid or incomplete replicas.

    Recipe/data-plan identity follows the generic trajectory schema. Completion
    seals and frozen source manifests are verified by the caller's Store.
    """
    if type(N) is not int or N < 2 or partition not in ("development", "confirmation"):
        raise ValueError("integer N >= 2 and development/confirmation partition required")
    rows, seen, contexts, paired, plans = [], set(), set(), {}, {}
    try:
        for raw in records:
            check()
            row = flatten_record(raw)
            if row.get("protocol") != PROTOCOL or row.get("schema_version") != 1:
                raise ValueError("incompatible registration protocol/schema")
            if row.get("partition") != partition:
                raise ValueError(f"{partition}-only evidence required")
            key = record_context(row)
            contexts.add(key)
            if context_key is not None and key != context_key:
                raise ValueError("trajectory differs from requested task")
            if partition == "confirmation" and row["family"] != f"confirmation--{key}":
                raise ValueError("confirmation family must be confirmation--taskkey")
            if row.get("status", "completed") not in ("completed", "complete"):
                raise ValueError("incomplete trajectory")
            _replicas([row["replica"]])
            recipe = Recipe.from_mapping(row["recipe"])
            if (row["recipe_key"] != recipe.key or row["method"] != recipe.method
                    or row["capacity"] != recipe.capacity or row["replay_count"] != recipe.replay_count):
                raise ValueError("trajectory recipe identity mismatch")
            identity = recipe.key, row["replica"]
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
            summary = summarize_trajectory(metrics, N=N, frozen_base_mse=row["frozen_base_mse"],
                                           frozen_novel_mse=row["frozen_novel_mse"])
            for name in SUMMARY_FIELDS:
                if not math.isclose(_number(row[name], name), summary[name], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"summary disagrees with complete metrics: {name}")
            for name in ("active_seconds", "peak_vram_bytes"):
                _number(row[name], name)
            for domain in ("base", "novel"):
                for metric in metrics:
                    coordinate = metric.get("coordinate_mse", {}).get(domain)
                    if coordinate is not None:
                        coords = np.asarray(coordinate, dtype=float)
                        if (coords.shape != (row["output_dim"],) or not np.isfinite(coords).all()
                                or (coords < 0).any()):
                            raise ValueError("invalid per-coordinate squared errors")
                        risk = float(coords.sum() / 27)
                        if not math.isclose(risk, metric[f"{domain}_mse"], rel_tol=1e-8, abs_tol=1e-9):
                            raise ValueError("risk must be squared-coordinate sum / 27 landmarks")
                    common = f"{domain}_common_mse"
                    if common in metric:
                        value = _number(metric[common], common)
                        expected = (metric[f"{domain}_mse"] if row["output_dim"] == 1 else
                                    coordinate[0] if coordinate is not None else value)
                        if not math.isclose(value, expected, rel_tol=1e-8, abs_tol=1e-9):
                            raise ValueError("common-coordinate risk disagrees with coordinate one")
            plan = row["plan_hash"]
            if not isinstance(plan, str) or not plan or row.get("pairing_id") != plan:
                raise ValueError("missing or inconsistent pairing provenance")
            if plans.setdefault(plan, row["replica"]) != row["replica"]:
                raise ValueError("distinct replicas cannot reuse a data plan")
            reference = paired.setdefault(row["replica"], row)
            if reference["plan_hash"] != plan or any(not math.isclose(row[k], reference[k],
                    rel_tol=1e-10, abs_tol=1e-10) for k in ("frozen_base_mse", "frozen_novel_mse")):
                raise ValueError("unpaired initialization, panels or data plans")
            if [m["observations"] for m in metrics] != [m["observations"] for m in reference["metrics"]]:
                raise ValueError("unpaired exposure counts")
            for field in DRIFT_FIELDS:
                if any(field in m for m in (*metrics, *reference["metrics"])):
                    if not all(field in m for m in (*metrics, *reference["metrics"])) or any(
                        not math.isclose(_number(a[field], field), _number(b[field], field),
                                         rel_tol=1e-10, abs_tol=1e-10)
                        for a, b in zip(metrics, reference["metrics"])):
                        raise ValueError("unpaired drift trajectory")
            rows.append({**row, "recipe": recipe.mapping(), "context_key": key})
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"malformed or incomplete trajectory: {error}") from error
    if len(contexts) > 1:
        raise ValueError("filter records to one task before analysis/selection")
    return rows


def crossing(curve, target):
    """Last interpolated downward entry that persists to the observed endpoint."""
    y = np.asarray(curve, dtype=float)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("finite nonnegative risk curve required")
    target = _number(target, "target risk")
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
    fields = ("novel_mse", "base_mse", "learning_seconds", "observations")
    arrays = {m: {f: np.asarray([[p[f] for p in r["metrics"]] for r in rows], dtype=float)
                   for f in fields} for m, rows in groups.items()}
    for field in DRIFT_FIELDS:
        if all(field in p for rows in groups.values() for r in rows for p in r["metrics"]):
            for method, rows in groups.items():
                arrays[method][field] = np.asarray([[p[field] for p in r["metrics"]] for r in rows])
    return arrays


def _point(arrays, indices=None, output_dim=54):
    fd, replay = (arrays[m] for m in (FD, REPLAY))
    indices = np.arange(len(fd["novel_mse"])) if indices is None else indices
    mean = lambda arm, field: arm[field][indices].mean(axis=0)
    target, initial = float(mean(replay, "novel_mse")[-1]), float(mean(replay, "novel_mse")[0])
    result = crossing(mean(fd, "novel_mse"), target)
    result.update(S=None, useful_target=bool(initial > 0 and target <= .95 * initial),
                  replay_target_rmse=math.sqrt(target),
                  target_mse_gain_fraction=None if initial == 0 else 1 - target / initial)
    if result["status"] != "valid":
        return result
    budget, N = result["matching_budget"], fd["novel_mse"].shape[1] - 1
    grid = np.arange(N + 1)
    interp = lambda arm, field: float(np.interp(budget, grid, mean(arm, field)))
    fd_time, replay_time = interp(fd, "learning_seconds"), float(mean(replay, "learning_seconds")[-1])
    if fd_time <= 0 or replay_time <= 0:
        result["status"] = "invalid_learning_time_denominator"
        return result
    # Interpolate replica risks BEFORE rooting; sqrt(mean replica risk) is different.
    old_fd = np.sqrt([np.interp(budget, grid, c) for c in fd["base_mse"][indices]])
    old_replay, old_initial = np.sqrt(replay["base_mse"][indices, -1]), np.sqrt(replay["base_mse"][indices, 0])
    ratio, saved = fd_time / replay_time, (N - budget) / N
    if not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(saved / ratio):
        result["status"] = "invalid_learning_time_denominator"
        return result
    result.update(n_F=budget, n_R=N, saved_novel=N - budget, saved_fraction=saved,
        saved_percent=100 * saved, learning_time_ratio=ratio, S=saved / ratio,
        fd_learning_seconds=fd_time, replay_learning_seconds=replay_time,
        fd_observations=interp(fd, "observations"), replay_observations=float(mean(replay, "observations")[-1]),
        fd_old_rmse=float(old_fd.mean()), replay_old_rmse=float(old_replay.mean()),
        old_rmse_difference=float((old_fd - old_replay).mean()),
        fd_old_deterioration=float((old_fd - old_initial).mean()),
        replay_old_deterioration=float((old_replay - old_initial).mean()))
    for arm in ("fd", "replay"):
        result[f"{arm}_digit_pairs"] = 9 * result[f"{arm}_observations"]
        result[f"{arm}_novel_digit_pairs"] = 9 * result["n_F" if arm == "fd" else "n_R"]
        result[f"{arm}_image_views"] = 18 * result[f"{arm}_observations"]
        result[f"{arm}_scalar_labels"] = output_dim * result[f"{arm}_observations"]
    for field in DRIFT_FIELDS:
        if field in fd:
            result[f"fd_final_{field}"] = interp(fd, field)
            result[f"replay_final_{field}"] = float(mean(replay, field)[-1])
    return result


def _groups(rows, methods, replicas=None):
    groups = {m: sorted((r for r in rows if r["method"] == m), key=lambda r: r["replica"]) for m in methods}
    expected = _replicas(replicas if replicas is not None else sorted({r["replica"] for r in rows}))
    if any([r["replica"] for r in group] != expected or len({r["recipe_key"] for r in group}) != 1
           for group in groups.values()):
        raise ValueError("complete paired single-policy replicas required for each method")
    return groups, expected


def point_equivalence(records_by_method):
    """Bootstrap-free point score. Requires FD/replay; EWC is optional."""
    if not {FD, REPLAY} <= set(records_by_method) or set(records_by_method) - set(METHODS):
        raise ValueError("FD and replay policy records required")
    raw = {m: [flatten_record(r) for r in group] for m, group in records_by_method.items()}
    if any(not group or any(r["method"] != m for r in group) for m, group in raw.items()):
        raise ValueError("empty or inconsistent method group")
    flat = [r for group in raw.values() for r in group]
    rows = validate_records(flat, flat[0]["target"], flat[0]["partition"])
    groups, _ = _groups(rows, raw)
    if not all(r["valid"] for r in rows):
        return dict(status="invalid_policy", S=None, useful_target=False)
    return _point(_arrays(groups), output_dim=rows[0]["output_dim"])


def paired_analysis(records, cfg, N=None, partition="confirmation", context_key=None,
                    replicas=None, check=lambda: None, *, seed=None, draws=None,
                    load_bootstrap=None, save_bootstrap=None):
    """Analyze one full three-arm task cohort, never a complete-case subset.

    Config is duck-typed. Required: novel_count, smoke, confirmation_replicas,
    calibration_replicas, retention_margin, matching_tolerance, envelope_factor,
    bootstrap_samples and seed (or seed_for). Store/runner verifies sealed policy
    membership before calling. Explicit ``replicas`` binds a frozen cohort.
    Cache hooks use (start)->block and (start, block); blocks bind all input data.
    Use the SAME seed/replicas across tasks for a secondary paired y1 contrast.
    """
    check()
    N = cfg.novel_count if N is None else N
    replicas = _replicas(range(cfg.confirmation_replicas if partition == "confirmation"
                               else cfg.calibration_replicas) if replicas is None else replicas)
    smoke = cfg.smoke
    if type(smoke) is not bool:
        raise ValueError("explicit boolean smoke required")
    if len(replicas) < 2 or (not smoke and partition == "confirmation"
                             and (N != 100 or len(replicas) not in (8, 10, 12))):
        raise ValueError("confirmation requires 8/10/12 replicas through 100; smoke needs at least two")
    rows = validate_records(records, N, partition, context_key=context_key, check=check)
    groups, replicas = _groups(rows, METHODS, replicas)
    if any(not r["valid"] for r in rows):
        raise ValueError("invalid trajectory cannot be excluded from paired inference")
    if seed is None:
        seed = (cfg.seed_for("analysis", 0, "paired_bootstrap") if hasattr(cfg, "seed_for") else
                derive_component_seed(cfg.seed, f"{PROTOCOL}:paired_bootstrap"))
    draws = cfg.bootstrap_samples if draws is None else draws
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed and at least two draws required")
    delta = _number(cfg.retention_margin, "retention margin")
    epsilon = _number(cfg.matching_tolerance, "matching tolerance")
    factor = _number(cfg.envelope_factor, "envelope factor", minimum=1.)
    arrays, dim = _arrays(groups), rows[0]["output_dim"]
    identity = content_hash(dict(protocol=PROTOCOL, records=sorted(rows, key=lambda r: (r["method"], r["replica"])),
                                 replicas=replicas, seed=seed, draws=draws))
    samples, inversions = np.empty((draws, len(replicas)), dtype=int), []
    for start in range(0, draws, 100):
        check()
        end = min(start + 100, draws)
        expected = np.asarray([replica_draw_indices(len(replicas), seed, d) for d in range(start, end)])
        block = load_bootstrap(start) if load_bootstrap else None
        if block is not None:
            if (block.get("input_hash") != identity or block.get("start") != start or block.get("end") != end
                    or not np.array_equal(block.get("indices"), expected)
                    or [r["draw"] for r in block.get("inversions", [])] != list(range(start, end))):
                raise ValueError("bootstrap checkpoint differs from input cohort/draws")
            values = block["inversions"]
        else:
            values = []
            for d, indices in zip(range(start, end), expected):
                check()
                values.append(dict(draw=d, **_point(arrays, indices, dim)))
            if save_bootstrap:
                save_bootstrap(start, dict(input_hash=identity, start=start, end=end,
                                           indices=expected.tolist(), inversions=values))
        samples[start:end] = expected
        inversions.extend(values)

    def interval(values):
        values = np.asarray(values, dtype=float)
        means = values[samples].mean(axis=1)
        return dict(mean=float(values.mean()), ci=np.quantile(means, [.025, .975]).tolist(),
            upper_one_sided_95=float(np.quantile(means, .95)), n_pairs=len(replicas),
            standard_error=float(values.std(ddof=1) / math.sqrt(len(replicas))))

    result = dict(protocol=PROTOCOL, schema_version=1, status="smoke_completed" if smoke else "completed",
        is_final=partition == "confirmation" and not smoke, partition=partition,
        context_key=rows[0]["context_key"], output_dim=dim, replicas=replicas, N=N, delta=delta, epsilon=epsilon,
        units=dict(risk="square_pixels_per_landmark", error="RMS_landmark_pixels",
                   observation="nine_digit_grid", digit_pairs_per_observation=9),
        methods={}, comparisons={}, bootstrap=dict(seed=seed, draws=draws, confidence_level=.95,
            resampling_unit="paired_whole_replica", curve_bands="pointwise_not_simultaneous"),
        assumptions="Nominal paired intervals conditional on frozen policies and fixed disjoint held-out panels "
                    "on a reused benchmark; independent replicas and adequate moments. Not familywise inference.")
    vectors = {}
    for method in METHODS:
        check()
        arms, fields = groups[method], arrays[method]
        A = fields["novel_mse"][:, 1:].mean(axis=1)
        H = np.sqrt(fields["base_mse"][:, 1:].mean(axis=1)) - np.sqrt(fields["base_mse"][:, 0])
        vectors[method] = A, H
        retention = interval(H)
        retention.update(point_ceiling_feasible=retention["mean"] <= delta,
                         ceiling_supported=retention["upper_one_sided_95"] <= delta)
        curves = dict(n=list(range(N + 1)), n_pairs=len(replicas), per_replica={"replicas": replicas})
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
            for metric in ("common_mse", "mae", "accuracy1"):
                name = f"{domain}_{metric}"
                present = [name in p for a in arms for p in a["metrics"]]
                if any(present):
                    if not all(present):
                        raise ValueError(f"incomplete optional curve: {name}")
                    values = np.asarray([[_number(p[name], name) for p in a["metrics"]] for a in arms])
                    if metric == "accuracy1" and (values > 1).any():
                        raise ValueError("accuracy1 must be a fraction")
                    curves[domain][f"mean_{metric}"] = values.mean(axis=0).tolist()
                    curves["per_replica"][name] = values.tolist()
        for name, values in fields.items():
            if name not in ("base_mse", "novel_mse"):
                curves[name] = values.mean(axis=0).tolist()
        seconds = float(fields["learning_seconds"][:, -1].sum() / fields["observations"][:, -1].sum())
        costs = dict(mean_learning_seconds=float(fields["learning_seconds"][:, -1].mean()),
                     seconds_per_arrival=seconds, peak_vram_bytes=max(a["peak_vram_bytes"] for a in arms))
        for name in ("evaluation_seconds", "fit_seconds", "eviction_seconds", "input_seconds", "active_seconds", "audit_seconds"):
            if all(name in a for a in arms):
                costs[f"mean_{name}"] = float(np.mean([_number(a[name], name) for a in arms]))
        item = dict(recipe=arms[0]["recipe"], recipe_key=arms[0]["recipe_key"], acquisition=interval(A),
                    retention=retention, frozen_gain=interval(fields["novel_mse"][:, 0] - A), costs=costs, curves=curves)
        if "novel_common_mse" in curves["per_replica"] or dim == 1:
            common = np.asarray(curves["per_replica"].get("novel_common_mse", fields["novel_mse"]))[:, 1:].mean(axis=1)
            item["common_coordinate_acquisition"] = interval(common)
            item["per_replica_common_coordinate_acquisition"] = common.tolist()
        result["methods"][method] = item
    for a, b in combinations(METHODS, 2):
        difference = interval(vectors[a][1] - vectors[b][1])
        result["comparisons"][f"{a}_vs_{b}"] = dict(first=a, second=b,
            acquisition_gain=interval(vectors[b][0] - vectors[a][0]), retention_difference=difference,
            retention_equivalence_supported=difference["ci"][0] >= -epsilon and difference["ci"][1] <= epsilon,
            retention_noninferiority_supported=difference["upper_one_sided_95"] <= epsilon, primary=a == FD and b == REPLAY)
    primary = result["comparisons"][f"{FD}_vs_{REPLAY}"]
    point = _point(arrays, output_dim=dim)
    invalid = sum(d["status"] != "valid" for d in inversions)
    all_valid = point["status"] == "valid" and invalid == 0
    ci = {f: np.quantile([d[f] for d in inversions], [.025, .975]).tolist() for f in MATCH_FIELDS} if all_valid else None
    uppers = {f: float(np.quantile([d[f] for d in inversions], .95)) for f in
              ("old_rmse_difference", "fd_old_deterioration", "replay_old_deterioration")} if all_valid else None
    full_ok = bool(primary["retention_noninferiority_supported"] and all(
        result["methods"][m]["retention"]["ceiling_supported"] for m in (FD, REPLAY)))
    matched_ok = bool(uppers and uppers["old_rmse_difference"] <= epsilon
        and uppers["fd_old_deterioration"] <= delta and uppers["replay_old_deterioration"] <= delta)
    fd_cost, replay_cost = (result["methods"][m]["costs"]["seconds_per_arrival"] for m in (FD, REPLAY))
    relative_cost_ok = fd_cost <= factor * replay_cost
    result["efficiency"] = dict(point=point, retention_qualified=full_ok and matched_ok,
        full_budget_retention_supported=full_ok, matched_budget_retention_supported=matched_ok,
        matched_upper_one_sided_95=uppers, relative_cost_feasible=relative_cost_ok,
        equal_budget_learning_time_ratio=None if replay_cost <= 0 else fd_cost / replay_cost,
        bootstrap=dict(draws=draws, invalid=invalid, statuses=dict(Counter(d["status"] for d in inversions)),
            multiple_crossing_draws=sum(d["multiple_crossings"] for d in inversions), ci=ci,
            interval_status="nominal_percentile_95" if all_valid else "unavailable_unresolved_draws", draws_detail=inversions),
        evidence_positive_acquisition=primary["acquisition_gain"]["ci"][0] > 0,
        evidence_sample_savings=bool(result["is_final"] and all_valid and ci["saved_fraction"][0] > 0
            and full_ok and matched_ok and relative_cost_ok and point["useful_target"]),
        interpretation="Matched-quality compute includes all background arrivals, both views, replay, FD and retirement. "
                       "Online pairs saved are not total annotations saved. S=0.25 is not a success threshold.")
    result["G"] = primary["acquisition_gain"]
    return result


def analyze(records, manifest, *, seed, draws=2000, check=lambda: None,
            load_bootstrap=None, save_bootstrap=None):
    """Phase9-style sealed-manifest adapter to :func:`paired_analysis`.

    Manifest fields: protocol, context={key, output_dim}, replicas, N, recipes
    (or selected_recipes), source_hash, manifest_hash, retention_margin and
    matching_tolerance; optional smoke and envelope_factor. Recipe lists or
    method-keyed mappings must bind exactly one hashed recipe for all three arms.
    Store verifies the source seal; this adapter verifies its manifest binding.
    """
    if (manifest.get("protocol") != PROTOCOL or manifest.get("schema_version", 1) != 1
            or manifest.get("sealed", True) is not True):
        raise ValueError("incompatible or unsealed registration manifest")
    if any(not isinstance(manifest.get(k), str) or not manifest[k] for k in ("source_hash", "manifest_hash")):
        raise ValueError("frozen source and manifest hashes required")
    if manifest["manifest_hash"] != content_hash(
            {k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("manifest hash disagrees with frozen design")
    context = manifest["context"]
    if (context.get("key") not in TASK_DIMS or type(context.get("output_dim")) is not int
            or context["output_dim"] != TASK_DIMS[context["key"]]):
        raise ValueError("manifest task/output_dim mismatch")
    raw = manifest.get("recipes", manifest.get("selected_recipes"))
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list) or len(raw) != len(METHODS):
        raise ValueError("all three frozen recipes required")
    recipes = [Recipe.from_mapping(r) for r in raw]
    if any(r.get("recipe_key") != recipe.key for r, recipe in zip(raw, recipes)):
        raise ValueError("frozen recipe keys required")
    if {r.method for r in recipes} != set(METHODS):
        raise ValueError("all three unique frozen adaptive methods required")
    rows = [flatten_record(r) for r in records]
    if any("source_hash" in r and r["source_hash"] != manifest["source_hash"] for r in rows):
        raise ValueError("trajectory source differs from frozen manifest")
    expected = {r.key: r for r in recipes}
    if any(r["recipe_key"] not in expected or Recipe.from_mapping(r["recipe"]) != expected[r["recipe_key"]]
           for r in rows):
        raise ValueError("trajectory differs from frozen recipe")
    cfg = SimpleNamespace(novel_count=manifest.get("N", 100), smoke=manifest.get("smoke", False),
        confirmation_replicas=len(manifest["replicas"]), calibration_replicas=2,
        retention_margin=manifest.get("retention_margin", manifest.get("delta", .5)),
        matching_tolerance=manifest.get("matching_tolerance", manifest.get("epsilon", .1)),
        envelope_factor=manifest.get("envelope_factor", 2.5), bootstrap_samples=draws)
    if type(cfg.smoke) is not bool:
        raise ValueError("explicit boolean smoke required")
    for canonical, alias in (("retention_margin", "delta"), ("matching_tolerance", "epsilon")):
        if canonical in manifest and alias in manifest and manifest[canonical] != manifest[alias]:
            raise ValueError("manifest retention aliases disagree")

    def load(start):
        block = load_bootstrap(start)
        if block is not None and any(block.get(k) != manifest[k] for k in ("manifest_hash", "source_hash")):
            raise ValueError("bootstrap checkpoint differs from frozen source/manifest")
        return block

    def save(start, block):
        save_bootstrap(start, {**block, "manifest_hash": manifest["manifest_hash"], "source_hash": manifest["source_hash"]})

    result = paired_analysis(rows, cfg, context_key=context["key"], replicas=manifest["replicas"],
        seed=seed, draws=draws, check=check, load_bootstrap=load if load_bootstrap else None,
        save_bootstrap=save if save_bootstrap else None)
    for name in ("source_hash", "manifest_hash", "context"):
        if name in manifest:
            result[name] = manifest[name]
    return result


def common_coordinate_contrast(vector, scalar, check=lambda: None):
    """Secondary vector-minus-scalar FD gains on y1, using shared replica draws."""
    for key in ("replicas", "N", "partition"):
        if vector[key] != scalar[key]:
            raise ValueError("cross-task contrast requires shared paired cohorts")
    if vector["output_dim"] != 6 or scalar["output_dim"] != 1 or vector["bootstrap"] != scalar["bootstrap"]:
        raise ValueError("vector/scalar analyses must use identical named bootstrap draws")
    gains = []
    for result in (vector, scalar):
        try:
            gains.append(np.asarray(result["methods"][REPLAY]["per_replica_common_coordinate_acquisition"])
                         - np.asarray(result["methods"][FD]["per_replica_common_coordinate_acquisition"]))
        except KeyError as error:
            raise ValueError("runner-computed coordinate-one errors required") from error
    values = gains[0] - gains[1]
    bootstrap = vector["bootstrap"]
    means = []
    for draw in range(bootstrap["draws"]):
        check()
        means.append(float(values[replica_draw_indices(len(values), bootstrap["seed"], draw)].mean()))
    return dict(mean=float(values.mean()), ci=np.quantile(means, [.025, .975]).tolist(),
        n_pairs=len(values), units="square_pixels", secondary=True,
        interpretation="Vector minus scalar replay-minus-FD y1 acquisition gains. Head size, pretraining and "
                       "extra supervision also differ; this does not isolate output dimension.")
