"""Plan 18 Phase 8 scalar inference, without learners, checkpoints or file IO.

Inputs are sealed legacy Runner.record dictionaries enriched with policy,
context and (optionally) namespace. The caller verifies source/completion seals
and the frozen allocation; this module validates scalar identity and pairing.
Risk is coordinate squared-error sum / 27 landmarks, not / 54 coordinates.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from mnist_experiment.fisher_memory_regression.analysis import flatten_record, replica_draw_indices
from mnist_experiment.fisher_memory_regression.config import content_hash
from mnist_experiment.fisher_memory_stm_capacity import analysis as legacy
from src.seeding import derive_component_seed

from . import PROTOCOL

POLICIES = {"R2": "all_r2--replay_ewc", "R1": "head_r1--replay_ewc",
            "F1": "head_r1--fd_replay_ewc"}
DEFAULT_SEED = derive_component_seed(20261109, f"{PROTOCOL}:fresh:paired_bootstrap")
# Config.seed_for strips the task suffix. This is the final scheduler seed,
# NOT DEFAULT_SEED in the old analyzer, selection_bootstrap or interim_bootstrap.
HISTORICAL_SEED = derive_component_seed(
    legacy.DEFAULT_SEED, f"{legacy.PROTOCOL}:confirmation:0:bootstrap")
CEILING, TOLERANCE = .5, .1
FIELDS = ("novel_mse", "base_mse", "observations", "learning_seconds")
RETENTION_FIELDS = ("old_rmse_difference", "arm_old_deterioration", "reference_old_deterioration")


def _metrics(row, N):
    """Validate even incomplete prefixes; independently check stored summaries."""
    points = row["metrics"]
    if not isinstance(points, list) or not points or len(points) > N + 1:
        raise ValueError("nonempty in-budget metrics required")
    previous_observations, previous_seconds = -1, -1.
    for n, point in enumerate(points):
        if type(point["n"]) is not int or point["n"] != n:
            raise ValueError("metrics must contain ordered contiguous n=0,...,N")
        exposure = point["observations"]
        if type(exposure) is not int or exposure < n or exposure <= previous_observations:
            raise ValueError("strictly increasing integer observations required")
        for field in FIELDS:
            value = point[field]
            if isinstance(value, (str, bool)) or not isinstance(value, (int, float)):
                raise ValueError(f"numeric {field} required")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"finite nonnegative {field} required")
        seconds = point["learning_seconds"]
        if seconds < previous_seconds or (n == 0 and (exposure != 0 or seconds != 0)):
            raise ValueError("invalid cumulative learning_seconds/initial exposure")
        previous_observations, previous_seconds = exposure, seconds
    complete = len(points) == N + 1 and row.get("status", "completed") in ("complete", "completed")
    if complete:
        expected = {f"{d}_auc": sum(p[f"{d}_mse"] for p in points[1:]) / N
                    for d in ("base", "novel")}
        expected.update({f"frozen_{d}_mse": points[0][f"{d}_mse"] for d in ("base", "novel")})
        expected.update({f: points[-1][f] for f in ("observations", "learning_seconds")})
        for field, value in expected.items():
            recorded = legacy.grid._number(row[field], field)
            if not math.isclose(recorded, value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"summary disagrees with metrics: {field}")
    return complete


def _cohort(records, namespace, partial):
    rows, indexed, complete, N = [], {}, {}, None
    try:
        for raw in records:
            row = flatten_record(raw)
            policy, context, replica = row["policy"], row["context"], row["replica"]
            if row.get("namespace", namespace) != namespace:
                raise ValueError(f"{namespace}-only evidence required; namespaces cannot be pooled")
            if (policy not in POLICIES or context not in ("blur", "no_blur")
                    or (context == "no_blur" and (policy == "R2" or namespace == "historical"))):
                raise ValueError("unsupported policy/context")
            if type(replica) is not int or replica < 0:
                raise ValueError("nonnegative integer replica required")
            if row["arm"] != POLICIES[policy]:
                raise ValueError("policy disagrees with historical arm")
            if type(row["target"]) is not int or row["target"] < 2:
                raise ValueError("target budget >= 2 required")
            N = row["target"] if N is None else N
            if row["target"] != N or row.get("N", N) != N:
                raise ValueError("mixed target budgets")
            if row.get("valid") is not True:
                raise ValueError("invalid trajectory cannot be excluded from inference")
            key = context, policy, replica
            if key in indexed:
                raise ValueError("duplicate context/policy/replica trajectory")
            indexed[key], complete[key] = row, _metrics(row, N)
            rows.append(row)
        if not rows:
            raise ValueError("no trajectories")
        # The frozen validator accepts actual subsets, not invented eight-arm records.
        for context in ("blur", "no_blur"):
            group = [r for k, r in indexed.items() if k[0] == context and complete[k]]
            if group:
                legacy.validate_records(group, N, "confirmation")
        for policy in POLICIES:
            if len({r["recipe_key"] for r in rows if r["policy"] == policy}) > 1:
                raise ValueError("one frozen single-policy recipe required across replicas/contexts")
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f"malformed legacy trajectory: {error}") from error

    replicas, excluded = [], []
    for replica in sorted({r["replica"] for r in rows}):
        missing = [p for p in POLICIES if ("blur", p, replica) not in indexed]
        unfinished = [p for p in POLICIES if ("blur", p, replica) in indexed
                      and not complete["blur", p, replica]]
        if missing or unfinished:
            excluded.append(dict(replica=replica, missing=missing, incomplete=unfinished))
        else:
            replicas.append(replica)
    if excluded and not partial:
        raise ValueError(f"incomplete paired blur blocks: {excluded}; use partial=True explicitly")
    if len(replicas) < 2:
        raise ValueError("at least two complete paired replicas required")
    no_blur = any(r["context"] == "no_blur" for r in rows)
    pending = [dict(replica=r, policies=[p for p in ("R1", "F1")
               if not complete.get(("no_blur", p, r), False)]) for r in replicas]
    pending = [p for p in pending if p["policies"]] if no_blur else []
    if pending and not partial:
        raise ValueError(f"incomplete paired no_blur blocks: {pending}; use partial=True explicitly")
    interaction = no_blur and not pending
    groups = {(c, p): [indexed[c, p, r] for r in replicas]
              for c in (("blur", "no_blur") if interaction else ("blur",))
              for p in (POLICIES if c == "blur" else ("R1", "F1"))}
    if interaction:
        for r in replicas:
            first = indexed["blur", "R1", r]
            for p in ("R1", "F1"):
                second = indexed["no_blur", p, r]
                if (first["plan_hash"] != second["plan_hash"]
                        or first["frozen_base_mse"] != second["frozen_base_mse"]
                        or [m["observations"] for m in first["metrics"]]
                        != [m["observations"] for m in second["metrics"]]):
                    raise ValueError("unpaired cross-context data plan, base panel or exposure counts")
                for field in legacy.grid.DRIFT_FIELDS:
                    if field == "blur":
                        continue
                    a, b = first["metrics"], second["metrics"]
                    if any(field in m for m in (*a, *b)) and (
                            not all(field in m for m in (*a, *b)) or any(
                                x[field] != y[field] for x, y in zip(a, b))):
                        raise ValueError("unpaired cross-context non-blur schedule")
    return groups, replicas, N, excluded, pending


def _interval(values, samples):
    values = np.asarray(values, dtype=float)
    means = values[samples].mean(axis=1)
    return dict(mean=float(values.mean()), ci=np.quantile(means, [.025, .975]).tolist(),
                upper_one_sided_95=float(np.quantile(means, .95)), n_pairs=len(values),
                standard_error=float(values.std(ddof=1) / math.sqrt(len(values))))


def crossing(curve, target):
    """Independent unsmoothed last sustained downward crossing on the n clock."""
    y = np.asarray(curve, dtype=float)
    if (y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all() or (y < 0).any()
            or not math.isfinite(target) or target < 0):
        raise ValueError("finite nonnegative curve and target required")
    entries = []
    for right in range(1, len(y)):
        if y[right - 1] > target >= y[right]:
            weight = (y[right - 1] - target) / (y[right - 1] - y[right])
            entries.append(float(right - 1 + weight))
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
                first_crossing=entries[0] if entries else None, multiple_crossings=len(entries) > 1,
                target_mse=float(target), initial_mse=float(y[0]))


def _match(arm, reference, indices, *, reference_by_definition=False):
    """Invert mean MSE; interpolate time/risks, then root retention per replica."""
    mean = lambda fields, name: fields[name][indices].mean(axis=0)
    risk, baseline = mean(arm, "novel_mse"), mean(reference, "novel_mse")
    N, target = len(risk) - 1, float(baseline[-1])
    point = crossing(risk, target) if not reference_by_definition else dict(
        status="valid", matching_budget=N, downward_crossings=[], first_crossing=None,
        multiple_crossings=False, target_mse=target, initial_mse=float(risk[0]),
        reference_by_definition=True)
    point.update(S=None, useful_target=bool(baseline[0] > 0 and target <= .95 * baseline[0]),
        reference_target_rmse=math.sqrt(target),
        target_mse_gain_fraction=float(1 - target / baseline[0]) if baseline[0] > 0 else None)
    if point["status"] != "valid":
        return point
    n, clock = point["matching_budget"], np.arange(N + 1)
    seconds = float(np.interp(n, clock, mean(arm, "learning_seconds")))
    denominator = float(mean(reference, "learning_seconds")[-1])
    if seconds <= 0 or denominator <= 0:
        point["status"] = "invalid_learning_time_denominator"
        return point
    ratio, saved = seconds / denominator, 1 - n / N
    if not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(saved / ratio):
        point["status"] = "invalid_learning_time_denominator"
        return point
    old = np.sqrt([np.interp(n, clock, c) for c in arm["base_mse"][indices]])
    old_reference = np.sqrt(reference["base_mse"][indices, -1])
    initial = np.sqrt(reference["base_mse"][indices, 0])
    point.update(n_arm=n, n_reference=N, saved_novel=N - n, saved_fraction=saved,
        saved_percent=100 * saved, learning_time_ratio=ratio, S=saved / ratio,
        arm_learning_seconds=seconds, reference_learning_seconds=denominator,
        arm_observations=float(np.interp(n, clock, mean(arm, "observations"))),
        reference_observations=float(mean(reference, "observations")[-1]),
        arm_old_rmse=float(old.mean()), reference_old_rmse=float(old_reference.mean()),
        old_rmse_difference=float((old - old_reference).mean()),
        arm_old_deterioration=float((old - initial).mean()),
        reference_old_deterioration=float((old_reference - initial).mean()))
    for prefix in ("arm", "reference"):
        observations = point[f"{prefix}_observations"]
        point.update({f"{prefix}_digit_pairs": 9 * observations,
                      f"{prefix}_novel_digit_pairs": 9 * point[f"n_{prefix}"],
                      f"{prefix}_image_views": 18 * observations,
                      f"{prefix}_scalar_labels": 54 * observations})
    return point


def _efficiency(arm, reference, samples, points, summary, reference_summary, contrast, is_reference):
    point = _match(arm, reference, np.arange(len(samples[0])), reference_by_definition=is_reference)
    invalid = [dict(draw=i, **p) for i, p in enumerate(points) if p["status"] != "valid"]
    resolved = point["status"] == "valid" and not invalid
    ci = {f: np.quantile([p[f] for p in points], [.025, .975]).tolist()
          for f in legacy.MATCH_FIELDS} if resolved else None
    upper = {f: float(np.quantile([p[f] for p in points], .95))
             for f in RETENTION_FIELDS} if resolved else None
    full = bool(contrast["retention_noninferiority_supported"] and summary["retention"]["ceiling_supported"]
                and reference_summary["retention"]["ceiling_supported"])
    matched = bool(upper and upper["old_rmse_difference"] <= TOLERANCE
        and upper["arm_old_deterioration"] <= CEILING and upper["reference_old_deterioration"] <= CEILING)
    numerator, denominator = arm["learning_seconds"][:, -1], reference["learning_seconds"][:, -1]
    boot_denominator = denominator[samples].mean(axis=1)
    return dict(point=point, full_budget_retention_supported=full,
        matched_budget_retention_supported=matched, retention_qualified=full and matched,
        matched_upper_one_sided_95=upper,
        equal_budget_learning_time_ratio=float(numerator.mean() / denominator.mean()) if denominator.mean() > 0 else None,
        equal_budget_learning_time_ratio_ci=np.quantile(numerator[samples].mean(axis=1) / boot_denominator,
            [.025, .975]).tolist() if (boot_denominator > 0).all() else None,
        bootstrap=dict(draws=len(samples), invalid=len(invalid), invalid_draws=invalid,
            statuses=dict(Counter(p["status"] for p in points)), ci=ci,
            interval_status="nominal_percentile_95" if resolved else "unavailable_unresolved_draws",
            weak_target_draws=sum(not p["useful_target"] for p in points),
            multiple_crossing_draws=sum(p["multiple_crossings"] for p in points),
            multiple_crossings=[dict(draw=i, downward_crossings=p["downward_crossings"])
                               for i, p in enumerate(points) if p["multiple_crossings"]]))


def bootstrap_chunk(arrays, *, start, stop, seed=DEFAULT_SEED, check=lambda: None):
    """Checkpointable scalar inverses, using one draw for all paired policies."""
    if set(arrays) != set(POLICIES) or type(start) is not int or type(stop) is not int or not 0 <= start < stop:
        raise ValueError("three blur policies and a nonempty nonnegative draw range required")
    values = []
    for draw in range(start, stop):
        check()
        indices = replica_draw_indices(len(arrays["R2"]["novel_mse"]), seed, draw)
        values.append(dict(draw=draw, indices=indices.tolist(), arms={p: _match(
            arrays[p], arrays["R2"], indices, reference_by_definition=p == "R2") for p in POLICIES}))
    return values


def _analyze(records, draws, seed, partial, namespace, check, load_bootstrap, save_bootstrap):
    check()
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0 or type(partial) is not bool:
        raise ValueError("integer draws >= 2, nonnegative integer seed and boolean partial required")
    groups, replicas, N, excluded, pending = _cohort(records, namespace, partial)
    arrays = {k: {f: np.asarray([[p[f] for p in r["metrics"]] for r in rows], dtype=float)
                  for f in FIELDS} for k, rows in groups.items()}
    samples = np.asarray([replica_draw_indices(len(replicas), seed, i) for i in range(draws)])
    identity = content_hash(dict(protocol=PROTOCOL, schema_version=1, namespace=namespace,
        records=[r for k in sorted(groups) for r in groups[k]], N=N, replicas=replicas,
        draws=draws, seed=seed, delta=CEILING, epsilon=TOLERANCE))
    inverses = {p: [] for p in POLICIES}
    for start in range(0, draws, 100):
        check()
        end = min(draws, start + 100)
        block = load_bootstrap(start) if load_bootstrap else None
        if block is None:
            block = dict(input_hash=identity, start=start, end=end, values=bootstrap_chunk(
                {p: arrays["blur", p] for p in POLICIES}, start=start, stop=end, seed=seed, check=check))
            block["payload_hash"] = content_hash(block)
            if save_bootstrap:
                save_bootstrap(start, block)
        else:
            try:
                valid = (block["input_hash"] == identity and block["start"] == start and block["end"] == end
                    and block["payload_hash"] == content_hash({k: v for k, v in block.items() if k != "payload_hash"})
                    and [v["draw"] for v in block["values"]] == list(range(start, end))
                    and all(v["indices"] == samples[v["draw"]].tolist() and set(v["arms"]) == set(POLICIES)
                            for v in block["values"]))
            except (KeyError, TypeError, ValueError, IndexError):
                valid = False
            if not valid:
                raise ValueError("bootstrap checkpoint differs from frozen input/settings or is corrupt")
        for value in block["values"]:
            for p in POLICIES:
                inverses[p].append(value["arms"][p])
    interval = lambda values: _interval(values, samples)
    has_interaction = ("no_blur", "F1") in groups
    result = dict(protocol=PROTOCOL, schema_version=1, namespace=namespace, N=N, replicas=replicas,
        is_final=namespace == "fresh" and not partial and not excluded and not pending
                 and N == 100 and len(replicas) == 8 and has_interaction,
        provisional=partial or N != 100 or len(replicas) != 8 or not has_interaction,
        cohort=dict(expected_fresh_replicas=8, partial=partial, excluded_blur_blocks=excluded,
                    pending_no_blur_blocks=pending),
        reference_policy="R2", headline="F1-R2", secondary="F1-R1",
        delta=CEILING, epsilon=TOLERANCE, curves={}, summaries={}, contrasts={}, efficiency={},
        input_hash=identity,
        bootstrap=dict(seed=seed, seed_name=(f"{legacy.PROTOCOL}:confirmation:0:bootstrap" if namespace == "historical"
                       else f"{PROTOCOL}:fresh:paired_bootstrap"), draws=draws, confidence_level=.95,
                       resampling_unit="paired_whole_replica", shared_across=[f"{c}/{p}" for c, p in groups],
                       curve_bands="pointwise_not_simultaneous"),
        units=dict(risk="square_pixels_per_landmark", error="RMS_landmark_pixels", observation="whole_grid"),
        assumptions="Nominal paired intervals conditional on frozen policies, source data and held-out panels; "
                    "not familywise inference or significance stopping. The caller verifies freshness and seals.")
    vectors, all_arrival = {}, {}
    for (context, policy), fields in arrays.items():
        check()
        exposure, query, base = (fields[f] for f in ("observations", "novel_mse", "base_mse"))
        # Normalize within each replica before averaging; horizons can differ.
        Q = ((query[:, 1:] + query[:, :-1]) * .5 * np.diff(exposure, axis=1)).sum(axis=1) / exposure[:, -1]
        all_arrival[context, policy] = Q
        A = query[:, 1:].mean(axis=1)
        H = np.sqrt(base[:, 1:].mean(axis=1)) - np.sqrt(base[:, 0])
        vectors[context, policy] = A, H
        retention = interval(H)
        retention.update(point_ceiling_feasible=retention["mean"] <= CEILING,
                         ceiling_supported=retention["upper_one_sided_95"] <= CEILING)
        curves = dict(axis="designated_novel_grids" if context == "blur" else "total_integrated_grids",
                      observations=exposure.mean(axis=0).tolist(),
                      per_replica_observations=exposure.tolist())
        if context == "blur":
            curves["n"] = list(range(N + 1))
        for domain, values in (("query", query), ("base", base)):
            means = values[samples].mean(axis=1)
            risk = values.mean(axis=0)
            ci = np.quantile(means, [.025, .975], axis=0)
            curves[domain] = dict(mean_mse=risk.tolist(), ci_mse=ci.tolist(),
                mean_rmse=np.sqrt(risk).tolist(), ci_rmse=np.sqrt(ci).tolist())
        rows = groups[context, policy]
        summary = dict(retention=retention, endpoint_mse=interval(query[:, -1]),
            endpoint_old_rmse=interval(np.sqrt(base[:, -1])), frozen_mse=interval(query[:, 0]),
            all_arrival_risk=interval(Q), learning_seconds=interval(fields["learning_seconds"][:, -1]),
            retired_training_cohort=legacy._retired(rows, N),
            per_replica=dict(replicas=replicas, retention=H.tolist(), all_arrival_risk=Q.tolist(),
                             endpoint_mse=query[:, -1].tolist(), observations=exposure[:, -1].tolist()))
        if context == "blur":
            summary.update(acquisition=interval(A), frozen_gain=interval(query[:, 0] - A))
            summary["per_replica"]["acquisition"] = A.tolist()
        result["curves"].setdefault(context, {})[policy] = curves
        result["summaries"].setdefault(context, {})[policy] = summary

    def contrast(first, second):
        gain = interval(vectors["blur", second][0] - vectors["blur", first][0])
        gain["favorable_replicas"] = int(np.sum(vectors["blur", second][0] > vectors["blur", first][0]))
        difference = interval(vectors["blur", first][1] - vectors["blur", second][1])
        return dict(first=first, second=second, acquisition_gain=gain,
            endpoint_risk_difference=interval(arrays["blur", first]["novel_mse"][:, -1]
                                             - arrays["blur", second]["novel_mse"][:, -1]),
            retention_difference=difference,
            retention_noninferiority_supported=difference["upper_one_sided_95"] <= TOLERANCE)

    result["contrasts"] = {"F1-R2": dict(contrast("F1", "R2"), headline_reference=True),
                           "F1-R1": dict(contrast("F1", "R1"), headline_reference=False)}
    for policy in POLICIES:
        result["efficiency"][policy] = _efficiency(arrays["blur", policy], arrays["blur", "R2"], samples, inverses[policy],
            result["summaries"]["blur"][policy], result["summaries"]["blur"]["R2"],
            contrast(policy, "R2"), policy == "R2")
    interaction = dict(status="incomplete_paired_blocks" if pending else "not_supplied")
    if has_interaction:
        differences, endpoints, retention = {}, {}, {}
        for context in ("blur", "no_blur"):
            differences[context] = all_arrival[context, "F1"] - all_arrival[context, "R1"]
            endpoints[context] = arrays[context, "F1"]["novel_mse"][:, -1] - arrays[context, "R1"]["novel_mse"][:, -1]
            retention[context] = interval(vectors[context, "F1"][1] - vectors[context, "R1"][1])
        interaction = dict(status="available", delta_blur=interval(differences["blur"]),
            delta_no_blur=interval(differences["no_blur"]),
            interaction=interval(differences["blur"] - differences["no_blur"]),
            endpoint_differences={c: interval(v) for c, v in endpoints.items()},
            endpoint_interaction=interval(endpoints["blur"] - endpoints["no_blur"]),
            retention_differences=retention,
            per_replica={f"{c}/{p}": v.tolist() for (c, p), v in all_arrival.items() if p != "R2"},
            interpretation="F1 minus R1, blur minus no_blur; negative favors F1 in blur. Both training and "
                "context-native evaluation appearance change. Frozen policies have different ridge values; "
                "not an isolated FD effect or a common-test-distribution training-blur effect.")
    result["blur_interaction"] = interaction
    return result


def analyze(records, *, draws=2000, seed=DEFAULT_SEED, partial=False, check=lambda: None,
            load_bootstrap=None, save_bootstrap=None):
    """Analyze fresh R2/R1/F1 blur blocks, optionally paired R1/F1 no_blur paths.

    Legacy fields/protocol remain intact; namespace defaults to fresh if absent.
    N is the legacy target; two or more replicas support smoke/interim analyses.
    partial=True explicitly excludes incomplete three-path blur blocks, listing
    all exclusions. No-blur inference waits for the entire retained cohort, not
    a selected four-path subset. Missing no_blur altogether is a valid interim.
    Corrupt/invalid evidence is never silently dropped. No control savings or S
    is computed. Pairing IDs/plans preserve the shared schedule across contexts;
    initial novel-panel risks may differ, but base-panel risks must match.
    """
    return _analyze(records, draws, seed, partial, "fresh", check, load_bootstrap, save_bootstrap)


def analyze_historical(records, *, draws=2000, seed=HISTORICAL_SEED, check=lambda: None):
    """Independently audit the labelled historical 24-path three-policy subset.

    Never pools historical paths with fresh inference or calls legacy.analyze,
    _point, _match or crossing. The default seed reproduces the final Plan 15
    scheduler's cfg.seed_for('confirmation--grid', 0, 'bootstrap') for its frozen
    config seed, 2026102912. Supply seed explicitly for a different sealed config.
    Existing completed sources and their scalar artifacts are read-only inputs.
    """
    return _analyze(records, draws, seed, False, "historical", check, None, None)


def historical_audit(records, history_root, *, seed=HISTORICAL_SEED, check=lambda: None):
    """Fail closed unless independently recomputed headline agrees with the seal.

    The caller verifies the historical Store's completion/source seals before
    supplying its R2/R1/F1 records. This reads only report_snapshot.json and
    checks its frozen settings, point, inverse CI, retention and risk contrasts.
    Float tolerances are fixed here before examining new efficacy trajectories.
    """
    check()
    path = Path(history_root) / "report_snapshot.json"
    try:
        snapshot = json.loads(path.read_text())
        sealed = snapshot["analysis"]
        if (sealed["protocol"] != legacy.PROTOCOL or sealed["schema_version"] != 1
                or sealed["partition"] != "confirmation" or not sealed["is_final"]
                or sealed["reference_arm"] != POLICIES["R2"]
                or sealed["bootstrap"]["seed"] != seed
                or sealed["bootstrap"]["resampling_unit"] != "paired_whole_replica"
                or sealed["delta"] != CEILING or sealed["epsilon"] != TOLERANCE):
            raise ValueError("incompatible historical report settings/seed")
        computed = analyze_historical(records, draws=sealed["bootstrap"]["draws"], seed=seed, check=check)
        if computed["N"] != sealed["N"] or computed["replicas"] != sealed["replicas"]:
            raise ValueError("historical report budget/replicas disagree")
        checks = []

        def compare(actual, expected, name):
            if isinstance(expected, dict):
                if not isinstance(actual, dict):
                    raise ValueError(f"historical statistics mismatch: {name}")
                for key, value in expected.items():
                    compare(actual[key], value, f"{name}.{key}")
            elif isinstance(expected, list):
                if not isinstance(actual, list) or len(actual) != len(expected):
                    raise ValueError(f"historical statistics mismatch: {name}")
                for i, (a, b) in enumerate(zip(actual, expected)):
                    compare(a, b, f"{name}[{i}]")
            elif type(expected) in (int, float):
                if type(actual) not in (int, float) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"historical statistics mismatch: {name}")
            elif actual != expected:
                raise ValueError(f"historical statistics mismatch: {name}")
            checks.append(name)

        for policy in ("F1", "R2"):
            old = sealed["arms"][POLICIES[policy]]
            new = computed["summaries"]["blur"][policy]
            for field in ("acquisition", "retention"):
                compare(new[field], old[field], f"{policy}.{field}")
            old_eff, new_eff = old["efficiency"], computed["efficiency"][policy]
            fields = (*legacy.MATCH_FIELDS, "status", "target_mse", "initial_mse", "useful_target",
                      "target_mse_gain_fraction", "reference_target_rmse", "downward_crossings", "multiple_crossings")
            compare(new_eff["point"], {k: old_eff["point"][k] for k in fields}, f"{policy}.point")
            for field in ("ci", "invalid", "statuses", "weak_target_draws", "multiple_crossing_draws"):
                compare(new_eff["bootstrap"][field], old_eff["bootstrap"][field], f"{policy}.bootstrap.{field}")
            for field in ("matched_upper_one_sided_95", "retention_qualified", "full_budget_retention_supported",
                          "matched_budget_retention_supported", "equal_budget_learning_time_ratio",
                          "equal_budget_learning_time_ratio_ci"):
                compare(new_eff[field], old_eff[field], f"{policy}.{field}")
        compare(computed["contrasts"]["F1-R2"]["acquisition_gain"],
                sealed["common_reference_comparisons"][POLICIES["F1"]]["acquisition_gain"], "headline.acquisition_gain")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"missing or malformed historical report: {error}") from error
    return dict(protocol=PROTOCOL, schema_version=1, namespace="historical", status="passed",
                report=str(path), report_hash=content_hash(snapshot), input_hash=computed["input_hash"],
                tolerances=dict(absolute=1e-10, relative=1e-10), checks=len(checks), analysis=computed)
