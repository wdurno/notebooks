"""Small artifact-only Plan 18 summaries; no learner or scheduler imports.

Risks are squared pixels per two-dimensional landmark, not per coordinate.
Roots for retention deteriorations are taken within replicas. Only whole
five-arm blocks enter inference, and all contrasts share sign-stratified draws.
Unsupported measurements are represented by None, never by extrapolation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import math

import numpy as np

from . import PROTOCOL

ARMS = tuple("ABCDE")
LABELS = dict(zip(ARMS, ("Replay/EWC", "STM incumbent", "STM + transport",
                        "STM + adaptation", "Combined")))
KINDS = {"initial", "regular", "novel", "zero_novel"}


def setting(cfg, name, default=None):
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def _budget(cfg):
    value = setting(cfg, "analysis_novel_count", 100)
    if type(value) is not int or value < 1:
        raise ValueError("analysis_novel_count must be a positive integer")
    return value


def _stream_size(cfg):
    return setting(cfg, "stream_size", 2 * (setting(cfg, "hold_steps", 160)
        + setting(cfg, "ramp_steps", 304)) * setting(cfg, "samples_per_step", 4))


def _number(value, name, *, minimum=0):
    if isinstance(value, (str, bool)) or value is None:
        raise ValueError(f"{name} must be finite numeric data")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def _rows(rows):
    rows = list(rows)
    if not rows or rows[0].get("kind") != "initial":
        raise ValueError("initial metric row required")
    previous = [-1.] * 4
    for index, row in enumerate(rows):
        if row.get("kind") not in KINDS or (index and row["kind"] == "initial"):
            raise ValueError("invalid metric row kind")
        for field in ("accepted_step", "incoming_grids", "novel_count"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f"nonnegative integer {field} required")
        values = [row[f] for f in ("accepted_step", "incoming_grids", "novel_count")]
        values.append(_number(row.get("learning_seconds"), "learning_seconds"))
        if any(a < b for a, b in zip(values, previous)):
            raise ValueError("metric clocks and cumulative learning time must be nondecreasing")
        previous = values
        if row["novel_count"] > row["incoming_grids"]:
            raise ValueError("novel count exceeds incoming grids")
        if not index and any(values[:3]):
            raise ValueError("initial clocks must be zero")
        if row["kind"] == "zero_novel" and row["novel_count"] != 0:
            raise ValueError("zero_novel checkpoint must precede novel exposure")
        for field in ("old_mse", "novel_mse"):
            _number(row.get(field), field)
        for field in ("current_mse", "old_accuracy", "novel_accuracy", "current_accuracy"):
            value = row.get(field)
            if value is not None:
                if _number(value, field) > 1 and "accuracy" in field:
                    raise ValueError("one-pixel accuracy must be a fraction")
        if row["kind"] in ("initial", "regular") and row.get("current_mse") is None:
            raise ValueError("regular current-law risk required")
    return rows


def _interpolate(x, y, grid):
    """Interpolate only within measured support, including optional fields."""
    if not len(x):
        return [None] * len(grid)
    values = np.asarray([np.nan if v is None else v for v in y], dtype=float)
    output = np.interp(grid, x, values)
    return [float(v) if x[0] <= n <= x[-1] and np.isfinite(v) else None
            for n, v in zip(grid, output)]


def novel_curve(rows, cfg):
    """First post-batch state per actual count, with an explicit pre-novel zero.

    Regular evaluations at a repeated novel count must not replace the earlier
    novel-containing batch. The frozen initialization is a separate baseline.
    """
    rows = _rows(rows)
    N = _budget(cfg)
    first = next((r for r in rows if r["novel_count"] > 0), None)
    zero = None
    if first is not None:
        candidates = [r for r in rows if r["novel_count"] == 0
                      and r["accepted_step"] < first["accepted_step"]]
        if candidates and candidates[-1]["accepted_step"] == first["accepted_step"] - 1:
            zero = candidates[-1]
    points = [zero] if zero is not None else []
    seen = {0}
    for row in rows:
        n = row["novel_count"]
        if n not in seen and row["kind"] in ("regular", "novel"):
            points.append(row)
            seen.add(n)
            if n >= N:
                break
    x = [r["novel_count"] for r in points]
    grid = list(range(N + 1))
    result = dict(n=grid, observed_counts=x,
                  observed_incoming_grids=[r["incoming_grids"] for r in points],
                  observed_steps=[r["accepted_step"] for r in points],
                  supported=bool(zero is not None and x[-1] >= N),
                  frozen_old_mse=rows[0]["old_mse"], frozen_novel_mse=rows[0]["novel_mse"])
    for field in ("old_mse", "novel_mse", "learning_seconds", "incoming_grids",
                  "old_accuracy", "novel_accuracy", "old_nll", "novel_nll"):
        result[field] = _interpolate(x, [r.get(field) for r in points], grid)
    return result


def _trap_mean(x, y, lo, hi):
    if hi <= lo or x[0] > lo or x[-1] < hi:
        return None
    inner = x[(x > lo) & (x < hi)]
    clock = np.concatenate(([lo], inner, [hi]))
    values = np.interp(clock, x, y)
    return float(np.sum(np.diff(clock) * (values[1:] + values[:-1]) / 2) / (hi - lo))


def summarize_path(rows, cfg):
    """Return scalar trajectory metrics; incomplete support remains explicit.

    Current-law trapezoids use only regular evaluations on the incoming-grid
    clock. Novel-budget risk is the arithmetic mean at integers 1,...,N.
    """
    rows = _rows(rows)
    curve = novel_curve(rows, cfg)
    N = _budget(cfg)
    regular = {r["incoming_grids"]: r for r in rows if r["kind"] in ("initial", "regular")}
    x = np.asarray(sorted(regular), dtype=float)
    y = np.asarray([regular[n]["current_mse"] for n in x], dtype=float)
    total = _stream_size(cfg)
    m = setting(cfg, "samples_per_step", 4)
    hold, ramp = setting(cfg, "hold_steps", 160) * m, setting(cfg, "ramp_steps", 304) * m
    result = dict(analysis_novel_count=N, novel_supported=curve["supported"],
        incoming_grids=rows[-1]["incoming_grids"], novel_count=rows[-1]["novel_count"],
        learning_seconds=rows[-1]["learning_seconds"],
        frozen_old_mse=rows[0]["old_mse"], frozen_novel_mse=rows[0]["novel_mse"],
        zero_novel_mse=curve["novel_mse"][0], zero_old_mse=curve["old_mse"][0],
        novel_auc=None, old_auc=None, Havg=None, H100=None,
        novel_mse_100=curve["novel_mse"][N], old_mse_100=curve["old_mse"][N],
        learning_seconds_100=curve["learning_seconds"][N],
        final_old_mse=rows[-1]["old_mse"], final_novel_mse=rows[-1]["novel_mse"])
    for domain in ("old", "novel", "current"):
        last = regular[x[-1]] if domain == "current" else rows[-1]
        result[f"final_{domain}_accuracy"] = last.get(f"{domain}_accuracy")
        result[f"final_{domain}_nll"] = last.get(f"{domain}_nll")
    for domain in ("old", "novel"):
        values = curve[f"{domain}_nll"][1:]
        result[f"{domain}_nll_auc"] = (float(np.mean(values)) if curve["supported"]
                                       and all(v is not None for v in values) else None)
    if curve["supported"]:
        result["novel_auc"] = float(np.mean(curve["novel_mse"][1:]))
        result["old_auc"] = float(np.mean(curve["old_mse"][1:]))
        baseline = math.sqrt(rows[0]["old_mse"])
        result["Havg"] = math.sqrt(result["old_auc"]) - baseline
        result["H100"] = math.sqrt(curve["old_mse"][N]) - baseline
    segments = {"whole": (0, total), "initial_quiet": (0, hold),
                "outbound": (hold, hold + ramp), "return": (hold + ramp, hold + 2 * ramp),
                "final_quiet": (hold + 2 * ramp, total)}
    for label, (lo, hi) in segments.items():
        risk = _trap_mean(x, y, lo, hi)
        result[f"current_{label}_mse"] = risk
        result[f"current_{label}_rms"] = None if risk is None else math.sqrt(risk)
    for label, parts in (("quiet", ("initial_quiet", "final_quiet")),
                         ("ramp", ("outbound", "return"))):
        values = [result[f"current_{p}_mse"] for p in parts]
        risk = float(np.mean(values)) if all(v is not None for v in values) else None
        result[f"current_{label}_mse"] = risk
        result[f"current_{label}_rms"] = None if risk is None else math.sqrt(risk)
    result["retention_avg"] = result["Havg"]
    result["retention_100"] = result["H100"]
    return result


def crossing(curve, target, *, frozen_initial=None):
    """Last downward entry persisting to N; never monotone-smooth the risk."""
    y = np.asarray(curve, dtype=float)
    q = _number(target, "target")
    result = dict(status="valid", n=None, downward_crossings=[], multiple_crossings=False)
    if y.ndim != 1 or len(y) < 2:
        raise ValueError("risk curve must contain zero and post-novel measurements")
    if not np.isfinite(y).all():
        result["status"] = "unsupported_observed_support"
    elif (y < 0).any():
        raise ValueError("risk curve must be nonnegative")
    elif frozen_initial is not None and _number(frozen_initial, "frozen initialization") <= q:
        result["status"] = "target_already_met_at_frozen_initialization"
    elif y[0] <= q:
        result["status"] = "target_already_met_at_zero_novel"
    elif y[-1] > q:
        result["status"] = "target_not_reached_at_endpoint"
    else:
        entries = [float(i + (y[i] - q) / (y[i] - y[i + 1]))
                   for i in range(len(y) - 1) if y[i] > q and y[i + 1] <= q]
        result.update(n=entries[-1], downward_crossings=entries, multiple_crossings=len(entries) > 1)
    return result


def paired_draws(replicas, draws, seed, signs=None):
    """Resample entire paired replicas within their fixed global-sign stratum."""
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0:
        raise ValueError("at least two draws and a nonnegative integer seed required")
    if not replicas or len(set(replicas)) != len(replicas):
        raise ValueError("distinct replicas required")
    signs = [(-1 if r % 2 else 1) for r in replicas] if signs is None else signs
    if len(signs) != len(replicas) or any(s not in (-1, 1) for s in signs):
        raise ValueError("global signs must be -1 or +1 per replica")
    rng = np.random.default_rng(seed)
    indices = np.empty((draws, len(replicas)), dtype=int)
    for sign in (-1, 1):
        group = np.flatnonzero(np.asarray(signs) == sign)
        if len(group):
            indices[:, group] = rng.choice(group, size=(draws, len(group)), replace=True)
    return indices


def selected_recipes(selection):
    """Accept arm-keyed recipes, a recipe list, or the scheduler's wrapper."""
    if not selection:
        return {}
    raw = (selection.get("recipes", selection.get("selected_recipes", selection))
           if isinstance(selection, Mapping) else selection)
    if isinstance(raw, Mapping):
        raw = [raw[a] for a in ARMS if a in raw]
    result = {}
    for item in raw:
        item = item.get("recipe", item)
        if item.get("arm") not in ARMS or item["arm"] in result:
            raise ValueError("selection must contain unique arm recipes")
        result[item["arm"]] = _recipe(item)
    if result and set(result) != set(ARMS):
        raise ValueError("selection must bind all five arms")
    return result


def _recipe(raw):
    if not isinstance(raw, Mapping) or raw.get("arm") not in ARMS:
        raise ValueError("five-arm recipe required")
    return dict(arm=raw["arm"], **{k: _number(raw.get(k), k)
                                for k in ("ridge", "strength", "process_scale")})


def _record(raw, cfg):
    metadata = raw.get("metadata", {})
    row = {**raw.get("summary", {}), **raw, **metadata}
    if row.get("kind", "trajectory") != "trajectory":
        return None
    if row.get("cohort") not in ("development", "fresh", "stationary"):
        raise ValueError("trajectory cohort required")
    if type(row.get("replica")) is not int or row["replica"] < 0:
        raise ValueError("nonnegative integer replica required")
    row["recipe"] = _recipe(row.get("recipe"))
    row["arm"] = row["recipe"]["arm"]
    row["status"] = row.get("status", "completed")
    row["pairing_asset_hash"] = next((row[k] for k in (
        "pairing_asset_hash", "pairing_hash", "pairing_id", "assets_hash") if row.get(k)), None)
    if row["status"] in ("completed", "complete"):
        if not row["pairing_asset_hash"]:
            raise ValueError("completed trajectory requires pairing asset hash")
        row["metrics"] = _rows(row["metrics"])
        computed = summarize_path(row["metrics"], cfg)
        for key, expected in computed.items():
            if key in raw.get("summary", {}):
                supplied = raw["summary"][key]
                if (expected is None) != (supplied is None) or (expected is not None and
                        not math.isclose(float(expected), float(supplied), rel_tol=1e-8, abs_tol=1e-9)):
                    raise ValueError(f"summary disagrees with accepted metrics: {key}")
        row.update(computed)
    return row


def _interval(values, samples):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return dict(mean=None, ci=None, upper_one_sided_95=None, status="unsupported", n_pairs=len(values))
    result = dict(mean=float(values.mean()), ci=None, upper_one_sided_95=None,
                  status="descriptive", n_pairs=len(values), per_replica=values.tolist())
    if samples is not None:
        means = values[samples].mean(axis=1)
        result.update(ci=np.quantile(means, [.025, .975]).tolist(),
                      upper_one_sided_95=float(np.quantile(means, .95)), status="nominal_percentile_95")
    return result


def _equivalence(arrays, arm, reference, indices):
    method, control = arrays[arm], arrays[reference]
    mean = lambda data, field: data[field][indices].mean(axis=0)
    N = method["novel_mse"].shape[1] - 1
    target = float(mean(control, "novel_mse")[-1])
    result = dict(reference=reference, arm=arm, target_mse=target if math.isfinite(target) else None,
                  status="unsupported_reference", n=None, saved_fraction=None, saved_percent=None,
                  matched_time_ratio=None, equal100_time_ratio=None, S=None,
                  matched_old_gap=None, full_old_gap=None, weak_target=None)
    if not math.isfinite(target):
        return result
    zero = float(mean(control, "novel_mse")[0])
    result.update(weak_target=bool(not math.isfinite(zero) or zero <= 0 or 1 - target / zero < .05),
                  reference_gain_fraction=None if not math.isfinite(zero) or zero <= 0 else 1 - target / zero)
    reference_crossing = crossing(mean(control, "novel_mse"), target,
        frozen_initial=float(control["frozen_novel_mse"][indices].mean()))
    match = crossing(mean(method, "novel_mse"), target,
        frozen_initial=float(method["frozen_novel_mse"][indices].mean()))
    result.update(match)
    if reference_crossing["status"] != "valid":
        result.update(status="reference_" + reference_crossing["status"], n=None)
    ref_time = float(mean(control, "learning_seconds")[-1])
    end_time = float(mean(method, "learning_seconds")[-1])
    if ref_time > 0 and math.isfinite(end_time):
        result["equal100_time_ratio"] = end_time / ref_time
    if result["status"] != "valid":
        return result
    n = float(N) if arm == reference else result["n"]
    grid = np.arange(N + 1)
    cost = float(np.interp(n, grid, mean(method, "learning_seconds")))
    old = np.asarray([np.interp(n, grid, c) for c in method["old_mse"][indices]])
    ref_old = control["old_mse"][indices, -1]
    result.update(n=n, matched_old_gap=float(np.mean(np.sqrt(old) - np.sqrt(ref_old))),
        full_old_gap=float(np.mean(np.sqrt(method["old_mse"][indices, -1]) - np.sqrt(ref_old))))
    if not (math.isfinite(cost) and cost > 0 and math.isfinite(ref_time) and ref_time > 0):
        result["status"] = "invalid_learning_time_denominator"
        return result
    ratio = cost / ref_time
    result.update(saved_fraction=1 - n / N, saved_percent=100 * (1 - n / N),
                  matched_learning_seconds=cost, reference_learning_seconds=ref_time,
                  matched_time_ratio=ratio, S=(1 - n / N) / ratio,
                  matched_incoming_grids=float(np.interp(n, grid, mean(method, "incoming_grids"))),
                  reference_incoming_grids=float(mean(control, "incoming_grids")[-1]))
    result["matched_novel_digit_pairs"] = 9 * n
    return result


def _cohort(blocks, cfg, cohort, check, bootstrap_seed=None):
    replicas = sorted(blocks)
    result = dict(replicas=replicas, n_pairs=len(replicas), methods={}, comparisons={}, efficiency={})
    if not replicas:
        return {**result, "status": "pending"}
    draws = setting(cfg, "bootstrap_draws", 2000)
    seed = (cfg.seed_for("fresh", 0, "bootstrap") if hasattr(cfg, "seed_for") else
            int.from_bytes(hashlib.sha256(f"{setting(cfg, 'seed', 181106)}:plan18:paired_bootstrap".encode())
                           .digest()[:8], "big"))
    seed = seed if bootstrap_seed is None else bootstrap_seed
    signs = [blocks[r]["A"].get("global_sign", -1 if r % 2 else 1) for r in replicas]
    samples = paired_draws(replicas, draws, seed, signs) if cohort == "fresh" and len(replicas) >= 2 else None
    result["bootstrap"] = dict(seed=seed, draws=draws if samples is not None else 0,
        stratification="replica_global_sign_or_parity", resampling_unit="whole_paired_replica",
        confidence_level=.95, intervals="nominal_not_familywise" if samples is not None else "descriptive_only")
    arrays, summaries = {}, {}
    N = _budget(cfg)
    for arm in ARMS:
        check()
        rows = [blocks[r][arm] for r in replicas]
        curves = [novel_curve(row["metrics"], cfg) for row in rows]
        arrays[arm] = {k: np.asarray([c[k] for c in curves], dtype=float) for k in
                       ("old_mse", "novel_mse", "learning_seconds", "incoming_grids")}
        arrays[arm]["frozen_novel_mse"] = np.asarray([r["frozen_novel_mse"] for r in rows])
        summaries[arm] = {k: np.asarray([r[k] for r in rows], dtype=float) for k in
            ("novel_auc", "Havg", "H100", "current_whole_rms", "current_quiet_rms",
             "current_ramp_rms", "final_old_mse", "final_novel_mse", "learning_seconds")}
        methods = {k: _interval(v, samples) for k, v in summaries[arm].items()}
        methods["recipe"] = rows[0]["recipe"]
        methods["curves"] = {"n": list(range(N + 1))}
        for field in ("old_mse", "novel_mse", "old_accuracy", "novel_accuracy"):
            matrix = np.asarray([c[field] for c in curves], dtype=float)
            mean = matrix.mean(axis=0)
            ci = np.quantile(matrix[samples].mean(axis=1), [.025, .975], axis=0) if samples is not None else None
            methods["curves"][field] = dict(
                mean=[float(v) if math.isfinite(v) else None for v in mean],
                ci=None if ci is None else [[float(v) if math.isfinite(v) else None for v in a] for a in ci],
                per_replica=[[float(v) if math.isfinite(v) else None for v in a] for a in matrix])
        result["methods"][arm] = methods
    for name, weights in (("E-B", {"E": 1, "B": -1}), ("E-A", {"E": 1, "A": -1}),
                          ("C-B", {"C": 1, "B": -1}), ("D-B", {"D": 1, "B": -1}),
                          ("interaction", {"E": 1, "C": -1, "D": -1, "B": 1})):
        result["comparisons"][name] = {k: _interval(sum(w * summaries[a][k] for a, w in weights.items()), samples)
            for k in ("novel_auc", "Havg", "H100", "current_whole_rms")}
    if cohort == "stationary":
        return {**result, "status": "diagnostic", "qualification":
                "Stationary novel panel measures untrained generalization; no novel-sample S or bootstrap intervals."}
    indices = np.arange(len(replicas))
    for arm, reference in [(a, "A") for a in ARMS] + [("E", "B")]:
        check()
        point = _equivalence(arrays, arm, reference, indices)
        boot = []
        if samples is not None:
            for i, sample in enumerate(samples):
                if i % 25 == 0:
                    check()
                boot.append(_equivalence(arrays, arm, reference, sample))
        failures = Counter(p["status"] for p in boot if p["status"] != "valid")
        inverse_ok = bool(boot and not failures and point["status"] == "valid")
        fields = ("n", "saved_fraction", "saved_percent", "matched_time_ratio", "S", "matched_old_gap")
        ci = {k: np.quantile([p[k] for p in boot], [.025, .975]).tolist() for k in fields} if inverse_ok else None
        full_gap = np.sqrt(arrays[arm]["old_mse"][:, -1]) - np.sqrt(arrays[reference]["old_mse"][:, -1])
        full = _interval(full_gap, samples)
        upper = float(np.quantile([p["matched_old_gap"] for p in boot], .95)) if inverse_ok else None
        tolerance = setting(cfg, "retention_tolerance", .10)
        supported = bool(upper is not None and full["upper_one_sided_95"] is not None
                         and upper < tolerance and full["upper_one_sided_95"] < tolerance)
        result["efficiency"][f"{arm}-{reference}"] = dict(point=point, ci=ci,
            full_budget_old_gap=full, matched_old_gap_upper_one_sided_95=upper,
            retention_tolerance=tolerance, retention_supported=supported,
            bootstrap_crossing_failures=sum(failures.values()),
            bootstrap_failure_frequency=sum(failures.values()) / len(boot) if boot else None,
            bootstrap_failure_statuses=dict(failures),
            inverse_interval_status="nominal_percentile_95" if inverse_ok else "withheld_unsupported_or_descriptive",
            sample_saving_supported=bool(supported and not point["weak_target"] and ci and ci["saved_fraction"][0] > 0))
    return {**result, "status": "paired_complete_blocks"}


def summarize(records, cfg, selection=None, *, check=lambda: None):
    """Summarize validated records, retaining partial blocks as progress only.

    Records may be flat or contain Store metadata plus summary/metrics. Store
    sealing is validated by the report's reader. This function also checks
    recipe, initialization, exposure-clock, and pairing-asset agreement.
    """
    recipes = selected_recipes(selection)
    frozen = selection if isinstance(selection, Mapping) else {}
    rows = []
    for raw in records:
        check()
        row = _record(raw, cfg)
        if row is not None:
            rows.append(row)
    grouped = {k: {} for k in ("fresh", "stationary")}
    development, incomplete, seen, assets = [], [], set(), {}
    for row in rows:
        check()
        complete = row["status"] in ("complete", "completed")
        if row["cohort"] == "development":
            development.append({k: row.get(k) for k in ("unit_id", "replica", "recipe", "status",
                "novel_auc", "Havg", "H100", "learning_seconds", "valid")})
            continue
        identity = row["cohort"], row["replica"], row["arm"]
        if identity in seen:
            raise ValueError("duplicate cohort/replica/arm trajectory")
        seen.add(identity)
        if recipes and row["recipe"] != recipes[row["arm"]]:
            raise ValueError("fresh recipe differs from frozen selection")
        block = grouped[row["cohort"]].setdefault(row["replica"], {})
        if complete:
            block[row["arm"]] = row
        else:
            incomplete.append(dict(cohort=row["cohort"], replica=row["replica"], arm=row["arm"]))
    cohorts = {}
    for cohort, candidates in grouped.items():
        blocks, partial = {}, []
        for replica, block in sorted(candidates.items()):
            if set(block) != set(ARMS):
                partial.append(dict(replica=replica, complete_arms=sorted(block)))
                continue
            reference = block["A"]
            asset = reference["pairing_asset_hash"]
            if (cohort, asset) in assets and assets[cohort, asset] != replica:
                raise ValueError("distinct replicas reuse a pairing asset")
            assets[cohort, asset] = replica
            for row in block.values():
                if row.get("valid", True) is not True:
                    raise ValueError("numerically invalid complete block cannot enter inference")
                if row["pairing_asset_hash"] != asset:
                    raise ValueError("unpaired asset hashes in complete block")
                for field in ("frozen_old_mse", "frozen_novel_mse"):
                    if not math.isclose(row[field], reference[field], rel_tol=1e-9, abs_tol=1e-10):
                        raise ValueError("unpaired frozen initialization")
                for field in ("model_seed", "factor_seed", "source_hash", "protocol"):
                    if row.get(field) != reference.get(field):
                        raise ValueError(f"unpaired provenance: {field}")
                if row.get("global_sign", (-1 if replica % 2 else 1)) != reference.get(
                        "global_sign", (-1 if replica % 2 else 1)):
                    raise ValueError("unpaired global sign")
                clock = lambda r: [(m["accepted_step"], m["incoming_grids"], m["novel_count"])
                    for m in r["metrics"] if m["kind"] in ("initial", "regular")]
                if clock(row) != clock(reference):
                    raise ValueError("unpaired incoming/novel clocks")
                if row["incoming_grids"] != _stream_size(cfg):
                    raise ValueError("completed trajectory does not cover the full incoming stream")
            blocks[replica] = block
        for arm in ARMS:
            policies = {tuple(b[arm]["recipe"].items()) for b in blocks.values()}
            if len(policies) > 1:
                raise ValueError("one frozen recipe per arm required across replicas")
        result = _cohort(blocks, cfg, cohort, check, frozen.get("bootstrap_seed"))
        result["partial_blocks"] = partial
        allocation = "main_replicas" if cohort == "fresh" else "stationary_replicas"
        expected = frozen.get(allocation, setting(cfg, allocation, 6 if cohort == "fresh" else 2))
        result["planned_pairs"] = expected
        result["allocation_complete"] = len(blocks) == expected and not partial
        cohorts[cohort] = result
    return dict(protocol=PROTOCOL, metric_schema=1,
        status="completed" if all(c["allocation_complete"] for c in cohorts.values()) else "pending_or_partial",
        cohorts=cohorts, development=development, incomplete=incomplete,
        selection=recipes, methods=cohorts["fresh"]["methods"],
        comparisons=cohorts["fresh"]["comparisons"], efficiency=cohorts["fresh"]["efficiency"],
        units=dict(risk="squared_pixels_per_landmark", error="RMS_landmark_pixels",
                   novel="unique_novel_grids", incoming="all_incoming_grids", digit_pairs_per_grid=9),
        qualification="Nominal intervals across whole paired replicas, conditional on frozen recipes and "
            "reused independent test panels. Partial blocks are progress only. No familywise or global-optimality claim.")
