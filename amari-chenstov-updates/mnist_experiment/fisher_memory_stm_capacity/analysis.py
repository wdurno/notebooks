"""Artifact-only Phase 12 inference, paired across all eight STM policies.

Risk is squared-coordinate error / 27 landmarks. Sample matching inverts mean
risk, while retention takes roots within replicas. No models or matrices load
here. Completion/source seals are the caller's responsibility.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from mnist_experiment.fisher_memory_grid_registration import analysis as grid
from mnist_experiment.fisher_memory_regression.analysis import flatten_record, replica_draw_indices

PROTOCOL = "plan15-stm-capacity-v1"
FD, REPLAY = "fd_replay_ewc", "replay_ewc"
STM_CONFIGS = {
    "head_r1": dict(ranks=[0, 0, 1], stm_parameters=108, ltm_parameters=3268, total_parameters=3376),
    "all_r1": dict(ranks=[1, 1, 1], stm_parameters=172, ltm_parameters=3268, total_parameters=3440),
    "all_r2": dict(ranks=[2, 2, 2], stm_parameters=344, ltm_parameters=3268, total_parameters=3612),
    "all_r4": dict(ranks=[4, 4, 4], stm_parameters=688, ltm_parameters=3268, total_parameters=3956),
}
ARMS = tuple(f"{chart}--{method}" for chart in STM_CONFIGS for method in (FD, REPLAY))
REFERENCE, INCUMBENT = f"all_r2--{REPLAY}", f"all_r2--{FD}"
DEFAULT_SEED = 2026102912
MATCH_FIELDS = ("matching_budget", "saved_novel", "saved_fraction", "saved_percent",
                "learning_time_ratio", "S", "arm_learning_seconds", "reference_learning_seconds",
                "arm_observations", "reference_observations", "arm_old_rmse", "reference_old_rmse",
                "old_rmse_difference", "arm_old_deterioration", "reference_old_deterioration")
RETENTION_FIELDS = ("old_rmse_difference", "arm_old_deterioration", "reference_old_deterioration")


def summarize_trajectory(*args, **kwargs):
    return {**grid.summarize_trajectory(*args, **kwargs), "protocol": PROTOCOL}


def validate_records(records, N, partition, *, check=lambda: None):
    """Adapt the frozen grid validator per chart, without rewriting recipe keys.

    Recipe hashes may coincide across charts; (chart, recipe, replica) is the
    identity. Optional source/config hashes bind the whole cohort, and optional
    chart hashes/orderings must be identical within each chart.
    """
    grouped = defaultdict(list)
    try:
        for raw in records:
            check()
            row = flatten_record(raw)
            chart = row.get("stm_config")
            if row.get("protocol") != PROTOCOL or row.get("schema_version") != 1:
                raise ValueError("incompatible STM-capacity protocol/schema")
            if chart not in STM_CONFIGS or row.get("method") not in (FD, REPLAY):
                raise ValueError("unknown STM chart or adaptive method")
            if row.get("arm") != f"{chart}--{row['method']}":
                raise ValueError("arm disagrees with STM chart/method")
            if row.get("family") != f"{partition}--grid":
                raise ValueError("family must agree with partition--grid")
            for field, expected in STM_CONFIGS[chart].items():
                if field in row and row[field] != expected:
                    raise ValueError(f"incompatible chart {field}")
            grouped[chart].append({**row, "protocol": grid.PROTOCOL})
        rows = []
        for group in grouped.values():
            rows.extend({**r, "protocol": PROTOCOL} for r in grid.validate_records(
                group, N, partition, context_key="grid", check=check))
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f"malformed STM-capacity trajectory: {error}") from error
    if not rows:
        raise ValueError("no STM-capacity trajectories")
    for field in ("source_hash", "config_hash", "scientific_hash"):
        if any(field in r for r in rows):
            value = rows[0].get(field)
            if not isinstance(value, str) or not value or any(r.get(field) != value for r in rows):
                raise ValueError(f"incompatible cohort {field}")
    for chart in grouped:
        group = [r for r in rows if r["stm_config"] == chart]
        for field in ("chart", "chart_hash", "parameter_order_hash", "layout_hash"):
            if any(field in r for r in group):
                if field not in group[0] or group[0][field] is None or any(
                        field not in r or r[field] != group[0][field] for r in group):
                    raise ValueError(f"incompatible {chart} {field}")
    paired, plans = {}, {}
    for row in rows:
        check()
        reference = paired.setdefault(row["replica"], row)
        plan = row["plan_hash"]
        if plans.setdefault(plan, row["replica"]) != row["replica"]:
            raise ValueError("distinct replicas cannot reuse a data plan")
        if plan != reference["plan_hash"] or any(not math.isclose(row[k], reference[k],
                rel_tol=1e-10, abs_tol=1e-10) for k in ("frozen_base_mse", "frozen_novel_mse")):
            raise ValueError("unpaired cross-chart initialization, panels or data plans")
        if [p["observations"] for p in row["metrics"]] != [p["observations"] for p in reference["metrics"]]:
            raise ValueError("unpaired cross-chart exposure counts")
        for field in grid.DRIFT_FIELDS:
            first, second = row["metrics"], reference["metrics"]
            if any(field in p for p in (*first, *second)) and (
                    not all(field in p for p in (*first, *second)) or any(
                        not math.isclose(grid._number(a[field], field), grid._number(b[field], field),
                                         rel_tol=1e-10, abs_tol=1e-10)
                        for a, b in zip(first, second))):
                raise ValueError("unpaired cross-chart drift trajectory")
    return rows


def _cohort(records, N, partition, replicas, check):
    rows = validate_records(records, N, partition, check=check)
    expected = grid._replicas(replicas if replicas is not None else sorted({r["replica"] for r in rows}))
    if len(expected) < 2:
        raise ValueError("at least two complete paired replicas required")
    groups = {a: sorted((r for r in rows if r["arm"] == a), key=lambda r: r["replica"]) for a in ARMS}
    if any([r["replica"] for r in g] != expected or len({r["recipe_key"] for r in g}) != 1
           for g in groups.values()):
        raise ValueError("complete eight-arm paired single-policy cohort required; no replica exclusions")
    if any(not r["valid"] for r in rows):
        raise ValueError("invalid trajectory cannot be excluded from paired inference")
    return groups, expected


def _match(arrays, arm, indices):
    if arm != REFERENCE:
        point = grid._point({grid.FD: arrays[arm], grid.REPLAY: arrays[REFERENCE]}, indices, 54)
        # Legacy mathematical helpers use FD/replay names, even for replay-size controls.
        return {k.replace("fd_", "arm_").replace("replay_", "reference_")
                if k not in ("n_F", "n_R") else {"n_F": "n_arm", "n_R": "n_reference"}[k]: v
                for k, v in point.items()}
    fields = arrays[REFERENCE]
    risk = fields["novel_mse"][indices].mean(axis=0)
    N = len(risk) - 1
    seconds = float(fields["learning_seconds"][indices, -1].mean())
    observations = float(fields["observations"][indices, -1].mean())
    old = np.sqrt(fields["base_mse"][indices, -1])
    deterioration = float((old - np.sqrt(fields["base_mse"][indices, 0])).mean())
    result = dict(status="valid" if seconds > 0 else "invalid_learning_time_denominator",
        reference_by_definition=True, target_mse=float(risk[-1]), initial_mse=float(risk[0]),
        useful_target=bool(risk[0] > 0 and risk[-1] <= .95*risk[0]),
        target_mse_gain_fraction=None if risk[0] == 0 else float(1-risk[-1]/risk[0]),
        reference_target_rmse=math.sqrt(risk[-1]), downward_crossings=[], multiple_crossings=False,
        matching_budget=N, n_arm=N, n_reference=N, saved_novel=0., saved_fraction=0.,
        saved_percent=0., learning_time_ratio=1., S=0., arm_learning_seconds=seconds,
        reference_learning_seconds=seconds, arm_observations=observations, reference_observations=observations,
        arm_old_rmse=float(old.mean()), reference_old_rmse=float(old.mean()), old_rmse_difference=0.,
        arm_old_deterioration=deterioration, reference_old_deterioration=deterioration)
    for prefix in ("arm", "reference"):
        result.update({f"{prefix}_digit_pairs": 9*observations, f"{prefix}_novel_digit_pairs": 9*N,
                       f"{prefix}_image_views": 18*observations, f"{prefix}_scalar_labels": 54*observations})
    return result


def bootstrap_chunk(arrays, *, start, stop, seed=DEFAULT_SEED, check=lambda: None):
    """Return checkpointable scalar inverses, never per-draw curves.

    ``arrays`` is the output of the stable grid ``_arrays`` helper, keyed by
    all eight arm names. Draw identity depends only on (seed, replica count,
    draw index), so scheduler chunk boundaries cannot change the bootstrap.
    The scheduler must bind saved chunks to its frozen input/configuration hash.
    """
    if set(arrays) != set(ARMS) or type(start) is not int or type(stop) is not int or not 0 <= start < stop:
        raise ValueError("all eight arms and a nonempty nonnegative draw range required")
    n = len(arrays[REFERENCE]["novel_mse"])
    result = []
    for draw in range(start, stop):
        check()
        indices = replica_draw_indices(n, seed, draw)
        result.append(dict(draw=draw, indices=indices.tolist(),
                           arms={a: _match(arrays, a, indices) for a in ARMS}))
    return result


def _interval(values, samples):
    values = np.asarray(values, dtype=float)
    means = values[samples].mean(axis=1)
    return dict(mean=float(values.mean()), ci=np.quantile(means, [.025, .975]).tolist(),
        upper_one_sided_95=float(np.quantile(means, .95)), n_pairs=len(values),
        standard_error=float(values.std(ddof=1)/math.sqrt(len(values))))


def _curve(values, samples, check):
    # Only this temporary draws-by-checkpoint array is needed for pointwise bands.
    means = np.empty((len(samples), values.shape[1]))
    for start in range(0, len(samples), 100):
        check()
        means[start:start+100] = values[samples[start:start+100]].mean(axis=1)
    return dict(mean=values.mean(axis=0).tolist(), ci=np.quantile(means, [.025, .975], axis=0).tolist())


def _retired(rows, N):
    values = defaultdict(list)
    for row in rows:
        by_n, seen = defaultdict(list), set()
        for item in row.get("retired_audit", []):
            n, exposure = item.get("n"), item.get("exposure")
            if (type(n) is not int or not 1 <= n <= N or type(exposure) is not int
                    or exposure < 0 or (n, exposure) in seen):
                raise ValueError("invalid/duplicate retired-training-cohort diagnostic")
            seen.add((n, exposure))
            by_n[n].append(grid._number(item["squared_error"], "retired squared error"))
        for n, risks in by_n.items():
            values[n].append(float(np.mean(risks)))
    counts = sorted(values)
    return dict(n=counts, mean_mse=[float(np.mean(values[n])) for n in counts],
                n_replicas=[len(values[n]) for n in counts], held_out=False) if counts else None


def analyze(records, *, N=100, partition="confirmation", replicas=None, draws=2000,
            seed=DEFAULT_SEED, delta=.5, epsilon=.1, check=lambda: None,
            load_bootstrap=None, save_bootstrap=None):
    """Analyze one complete eight-arm cohort; return finite JSON-compatible data.

    Every headline arm uses all_r2 Replay/EWC at N as target/time denominator.
    Within-size contrasts do not invert or redefine that reference. Invalid
    inverse draws are retained and withhold only that arm's ordinary inverse CI
    and paired S differences involving it, never full-budget A/H intervals.
    Tiny synthetic cohorts are supported but cannot be final confirmation.
    Optional cache callbacks are load(start)->block and save(start, block).
    Each 100-draw block binds all records, cohort, analysis settings and indices.
    """
    check()
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0:
        raise ValueError("at least two draws and nonnegative integer seed required")
    delta, epsilon = grid._number(delta, "delta"), grid._number(epsilon, "epsilon")
    groups, replicas = _cohort(records, N, partition, replicas, check)
    arrays = grid._arrays(groups)
    identity = grid.content_hash(dict(protocol=PROTOCOL, schema_version=1, N=N, partition=partition,
        replicas=replicas, draws=draws, seed=seed, delta=delta, epsilon=epsilon,
        records=[r for arm in ARMS for r in groups[arm]]))
    samples, inverses = np.empty((draws, len(replicas)), dtype=int), {a: [] for a in ARMS}
    for start in range(0, draws, 100):
        check()
        end = min(start+100, draws)
        block = load_bootstrap(start) if load_bootstrap else None
        if block is not None:
            try:
                valid = (block["input_hash"] == identity and block["start"] == start and block["end"] == end
                    and block["payload_hash"] == grid.content_hash({k: v for k, v in block.items() if k != "payload_hash"})
                    and [d["draw"] for d in block["values"]] == list(range(start, end))
                    and all(d["indices"] == replica_draw_indices(len(replicas), seed, d["draw"]).tolist()
                            and set(d["arms"]) == set(ARMS) for d in block["values"]))
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError("bootstrap checkpoint differs from frozen input/cohort/draws or is corrupt")
        else:
            block = dict(input_hash=identity, start=start, end=end,
                values=bootstrap_chunk(arrays, start=start, stop=end, seed=seed, check=check))
            block["payload_hash"] = grid.content_hash(block)
            if save_bootstrap:
                save_bootstrap(start, block)
        for item in block["values"]:
            samples[item["draw"]] = item["indices"]
            for arm in ARMS:
                inverses[arm].append(item["arms"][arm])
    interval = lambda values: _interval(values, samples)
    result = dict(protocol=PROTOCOL, schema_version=1, status="completed", partition=partition,
        is_final=partition == "confirmation" and N == 100 and len(replicas) in (8, 10),
        context_key="grid", output_dim=54, N=N, replicas=replicas, delta=delta, epsilon=epsilon,
        reference_arm=REFERENCE, incumbent_arm=INCUMBENT, arms={}, common_reference_comparisons={},
        within_size={}, incumbent_differences={},
        input_hash=identity,
        bootstrap=dict(seed=seed, seed_name=f"{PROTOCOL}:paired_bootstrap", draws=draws,
            confidence_level=.95, resampling_unit="paired_whole_replica", shared_across=list(ARMS),
            curve_bands="pointwise_not_simultaneous"),
        units=dict(risk="square_pixels_per_landmark", error="RMS_landmark_pixels",
                   observation="nine_digit_grid", digit_pairs_per_observation=9),
        assumptions="Nominal paired intervals conditional on frozen policies and fixed held-out panels; "
                    "independent replica trajectories, not familywise inference or significance stopping.")
    for field in ("source_hash", "config_hash", "scientific_hash"):
        if field in groups[REFERENCE][0]:
            result[field] = groups[REFERENCE][0][field]
    vectors = {}
    for arm, rows in groups.items():
        check()
        fields = arrays[arm]
        A = fields["novel_mse"][:, 1:].mean(axis=1)
        H = np.sqrt(fields["base_mse"][:, 1:].mean(axis=1)) - np.sqrt(fields["base_mse"][:, 0])
        vectors[arm] = A, H
        retention = interval(H)
        retention.update(point_ceiling_feasible=retention["mean"] <= delta,
                         ceiling_supported=retention["upper_one_sided_95"] <= delta)
        curves = dict(n=list(range(N+1)), n_pairs=len(replicas))
        for domain in ("novel", "base"):
            summary = _curve(fields[f"{domain}_mse"], samples, check)
            curves[domain] = dict(mean_mse=summary["mean"], ci_mse=summary["ci"],
                mean_rmse=np.sqrt(summary["mean"]).tolist(), ci_rmse=np.sqrt(summary["ci"]).tolist())
            for metric in ("mae", "accuracy1"):
                name = f"{domain}_{metric}"
                present = [name in p for row in rows for p in row["metrics"]]
                if not any(present):
                    continue
                if not all(present):
                    raise ValueError(f"incomplete optional curve: {name}")
                values = np.asarray([[grid._number(p[name], name) for p in r["metrics"]] for r in rows])
                if metric == "accuracy1" and (values > 1).any():
                    raise ValueError("accuracy1 must be a fraction")
                optional = _curve(values, samples, check)
                curves[domain].update({f"mean_{metric}": optional["mean"], f"ci_{metric}": optional["ci"]})
        for name, values in fields.items():
            if name not in ("novel_mse", "base_mse"):
                curves[name] = values.mean(axis=0).tolist()
        costs = dict(mean_learning_seconds=float(fields["learning_seconds"][:, -1].mean()),
            seconds_per_arrival=float(fields["learning_seconds"][:, -1].sum()/fields["observations"][:, -1].sum()),
            peak_vram_bytes=max(r["peak_vram_bytes"] for r in rows))
        for field in ("active_seconds", "input_seconds", "fit_seconds", "eviction_seconds", "evaluation_seconds",
                      "audit_seconds", "jacobian_fisher_seconds", "compression_seconds", "stream_preparation_seconds"):
            if all(field in r for r in rows):
                costs[f"mean_{field}"] = float(np.mean([grid._number(r[field], field) for r in rows]))
        item = dict(stm_config=rows[0]["stm_config"], method=rows[0]["method"],
            configuration=dict(STM_CONFIGS[rows[0]["stm_config"]]), recipe=rows[0]["recipe"],
            recipe_key=rows[0]["recipe_key"], acquisition=interval(A), retention=retention,
            per_replica=dict(replicas=replicas, acquisition=A.tolist(), retention=H.tolist(),
                learning_seconds=fields["learning_seconds"][:, -1].tolist(),
                observations=fields["observations"][:, -1].tolist(),
                seconds_per_arrival=(fields["learning_seconds"][:, -1]/fields["observations"][:, -1]).tolist()),
            frozen_gain=interval(fields["novel_mse"][:, 0]-A), costs=costs, curves=curves,
            retired_training_cohort=_retired(rows, N), applied={})
        for field in ("chart", "chart_hash", "parameter_order_hash", "layout_hash"):
            if field in rows[0]:
                item[field] = rows[0][field]
        for domain in ("novel", "base"):
            item["applied"][domain] = dict(endpoint_rms=interval(np.sqrt(fields[f"{domain}_mse"][:, -1])))
            for metric in ("mae", "accuracy1"):
                if f"mean_{metric}" in curves[domain]:
                    item["applied"][domain][f"endpoint_{metric}"] = interval(
                        [r["metrics"][-1][f"{domain}_{metric}"] for r in rows])
        result["arms"][arm] = item

    def comparison(first, second):
        gain = interval(vectors[second][0]-vectors[first][0])
        gain["favorable_replicas"] = int(np.sum(vectors[second][0] > vectors[first][0]))
        difference = interval(vectors[first][1]-vectors[second][1])
        return dict(first=first, second=second, acquisition_gain=gain, retention_difference=difference,
                    retention_noninferiority_supported=difference["upper_one_sided_95"] <= epsilon)

    for arm in ARMS:
        result["common_reference_comparisons"][arm] = comparison(arm, REFERENCE)
    for chart in STM_CONFIGS:
        result["within_size"][chart] = dict(comparison(f"{chart}--{FD}", f"{chart}--{REPLAY}"),
            headline_reference=False, interpretation="Paired full-budget contrasts only; not a differently referenced S.")
    for arm, item in result["arms"].items():
        check()
        points = inverses[arm]
        point = _match(arrays, arm, np.arange(len(replicas)))
        invalid = [dict(draw=d, **{k: p[k] for k in ("status", "matching_budget", "downward_crossings", "target_mse", "initial_mse")})
                   for d, p in enumerate(points) if p["status"] != "valid"]
        all_valid = point["status"] == "valid" and not invalid
        ci = {f: np.quantile([p[f] for p in points], [.025, .975]).tolist() for f in MATCH_FIELDS} if all_valid else None
        upper = {f: float(np.quantile([p[f] for p in points], .95)) for f in RETENTION_FIELDS} if all_valid else None
        full_ok = bool(result["common_reference_comparisons"][arm]["retention_noninferiority_supported"]
            and item["retention"]["ceiling_supported"] and result["arms"][REFERENCE]["retention"]["ceiling_supported"])
        matched_ok = bool(upper and upper["old_rmse_difference"] <= epsilon
            and upper["arm_old_deterioration"] <= delta and upper["reference_old_deterioration"] <= delta)
        reference_seconds = result["arms"][REFERENCE]["costs"]["mean_learning_seconds"]
        equal_ratio = item["costs"]["mean_learning_seconds"]/reference_seconds if reference_seconds > 0 else None
        ratio_draws = arrays[arm]["learning_seconds"][samples, -1].mean(axis=1)
        denominator = arrays[REFERENCE]["learning_seconds"][samples, -1].mean(axis=1)
        equal_ci = np.quantile(ratio_draws/denominator, [.025, .975]).tolist() if (denominator > 0).all() else None
        item["efficiency"] = dict(point=point, full_budget_retention_supported=full_ok,
            matched_budget_retention_supported=matched_ok, retention_qualified=full_ok and matched_ok,
            matched_upper_one_sided_95=upper, equal_budget_learning_time_ratio=equal_ratio,
            equal_budget_learning_time_ratio_ci=equal_ci,
            bootstrap=dict(draws=draws, invalid=len(invalid), invalid_draws=invalid,
                statuses=dict(Counter(p["status"] for p in points)), ci=ci,
                multiple_crossing_draws=sum(p["multiple_crossings"] for p in points),
                multiple_crossings=[dict(draw=d, downward_crossings=p["downward_crossings"])
                                    for d, p in enumerate(points) if p["multiple_crossings"]],
                weak_target_draws=sum(not p["useful_target"] for p in points),
                interval_status="nominal_percentile_95" if all_valid else "unavailable_unresolved_draws"),
            evidence_positive_acquisition=result["common_reference_comparisons"][arm]["acquisition_gain"]["ci"][0] > 0,
            evidence_sample_savings=bool(result["is_final"] and all_valid and ci["saved_fraction"][0] > 0
                                         and full_ok and matched_ok and point["useful_target"]))
    for arm in ARMS:
        if arm == INCUMBENT:
            continue
        first, second = result["arms"][arm]["efficiency"], result["arms"][INCUMBENT]["efficiency"]
        valid = first["bootstrap"]["ci"] is not None and second["bootstrap"]["ci"] is not None
        difference = None
        if first["point"]["status"] == second["point"]["status"] == "valid":
            difference = first["point"]["S"]-second["point"]["S"]
        ci = np.quantile([a["S"]-b["S"] for a, b in zip(inverses[arm], inverses[INCUMBENT])],
                         [.025, .975]).tolist() if valid else None
        result["incumbent_differences"][arm] = dict(comparison(arm, INCUMBENT), S_difference=dict(
            mean=difference, ci=ci, interval_status="nominal_percentile_95" if valid else "unavailable_unresolved_draws",
            shared_draws=True, alternative_fd=arm.endswith("--"+FD)))
    return result
