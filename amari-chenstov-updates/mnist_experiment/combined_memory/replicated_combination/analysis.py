"""Pure scalar inference for Plan 18 Phase 9's six actual policies.

Records retain the legacy head_r1 Runner.record schema and add ``policy``,
``namespace`` and ``representation``. Source/completion seals and the frozen
fresh seed allocation are checked by the caller's artifact store. This module
checks scalar completion, identity, single-policy recipes, and pairing. It
never loads a checkpoint, trains, reads files, or invents missing policies.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import math

import numpy as np

from mnist_experiment.fisher_memory_regression.analysis import flatten_record, replica_draw_indices
from mnist_experiment.fisher_memory_regression.config import Recipe, content_hash
from mnist_experiment.fisher_memory_stm_capacity import analysis as legacy
from mnist_experiment.combined_memory.reproduction import analysis as reproduction
from src.seeding import derive_component_seed

PROTOCOL = "plan18-replicated-combination-v1"
POLICIES = ("R", "A", "B", "C", "D", "E")
DIAGONAL_POLICIES = POLICIES[1:]
DEFAULT_SEED = derive_component_seed(20261110, f"{PROTOCOL}:fresh:paired_bootstrap")
CEILING, TOLERANCE, STRENGTH = .5, .01, 3.34056320604949
FIELDS = reproduction.FIELDS
MATCH_FIELDS = legacy.MATCH_FIELDS
RETENTION_FIELDS = reproduction.RETENTION_FIELDS
crossing = reproduction.crossing


def _number(value, name, *, signed=False):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float)):
        raise ValueError(f"numeric {name} required")
    if not math.isfinite(value) or (not signed and value < 0):
        raise ValueError(f"finite {'signed' if signed else 'nonnegative'} {name} required")
    return float(value)


def validate_record(raw, *, check=lambda: None):
    """Return (flattened record, complete); validate even unfinished prefixes.

    An absent status has the historical Runner.record meaning: completion is
    inferred from all N+1 rows. Explicit non-complete statuses are never promoted.
    Artifact seals still belong to the caller. N>=2 supports separate tiny smoke
    configurations; only eight fresh 100-grid blocks can be labelled final.
    """
    check()
    try:
        row = flatten_record(raw)
        policy, namespace, N = row["policy"], row["namespace"], row["target"]
        if policy not in POLICIES or namespace not in ("development", "fresh", "acceptance"):
            raise ValueError("six-policy identity and explicit supported namespace required")
        if type(row["replica"]) is not int or row["replica"] < 0:
            raise ValueError("nonnegative integer replica required")
        if type(N) is not int or N < 2 or row.get("N", N) != N:
            raise ValueError("integer target >= 2 and consistent N required")
        if (row.get("partition") not in ("development", "confirmation")
                or row.get("family") != f"{row.get('partition')}--grid"
                or row.get("output_dim") != 54 or row.get("context_key", "grid") != "grid"):
            raise ValueError("legacy grid family/partition and 54-output task required")
        if row.get("protocol") not in (legacy.PROTOCOL, PROTOCOL) or row.get("schema_version") != 1:
            raise ValueError("legacy or Phase 9 protocol with schema 1 required")
        representation = "low_rank_diagonal" if policy == "R" else "diagonal"
        if row["representation"] != representation:
            raise ValueError("R alone uses low_rank_diagonal; A--E must be diagonal")
        method = legacy.REPLAY if policy == "A" else legacy.FD
        if (row.get("stm_config") != "head_r1" or row.get("method") != method
                or row.get("arm") != f"head_r1--{method}"):
            raise ValueError("all six arms require the same head_r1 chart and assigned method")
        recipe = Recipe.from_mapping(row["recipe"])
        smoke = row.get("smoke", N != 100)
        if type(smoke) is not bool:
            raise ValueError("explicit boolean smoke flag required")
        if (recipe.key != row["recipe_key"] or recipe.method != method
                or ((N == 100 or not smoke) and (recipe.capacity != 32 or recipe.replay_count != 2))
                or row["capacity"] != recipe.capacity or row["replay_count"] != recipe.replay_count
                or recipe.ridge > 1000
                or recipe.strength != (0. if policy == "A" else STRENGTH)
                or (policy in ("B", "R") and recipe.ridge != 1.)):
            raise ValueError("recipe identity or fixed Phase 9 policy settings disagree")
        if N == 100 and (recipe.fd_step != .01 or recipe.probes != 2
                         or recipe.max_iter != 100 or recipe.max_eval != 150):
            raise ValueError("production solve budget and FD settings must match the fixed historical recipe")
        if row.get("valid") is not True:
            raise ValueError("invalid trajectory cannot be excluded from inference")
        if row.get("frozen", True) is not True:
            raise ValueError("policy must be frozen before inference")
        points = row["metrics"]
        if not isinstance(points, list) or not 1 <= len(points) <= N + 1:
            raise ValueError("nonempty in-budget metric prefix required")
        previous_exposure, previous_seconds = -1, -1.
        for n, point in enumerate(points):
            check()
            if type(point["n"]) is not int or point["n"] != n:
                raise ValueError("metrics must contain contiguous ordered n=0,...,N")
            exposure = point["observations"]
            if type(exposure) is not int or exposure < n or exposure <= previous_exposure:
                raise ValueError("strictly increasing integer observations required")
            for field in FIELDS:
                _number(point[field], field)
            seconds = point["learning_seconds"]
            if seconds < previous_seconds or (n == 0 and (exposure != 0 or seconds != 0)):
                raise ValueError("invalid cumulative learning time or initial exposure")
            previous_exposure, previous_seconds = exposure, seconds
        complete = len(points) == N + 1 and row.get("status", "completed") in ("complete", "completed")
        # Validate one genuine record at a time: the old schema's recipe identity
        # does not distinguish six treatment policies sharing a head chart.
        if complete:
            legacy.validate_records([{**row, "protocol": legacy.PROTOCOL}], N, row["partition"], check=check)
        return row, complete
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f"malformed legacy trajectory: {error}") from error


def _same_optional(rows, fields, message):
    for field in fields:
        if any(field in row for row in rows):
            first = rows[0].get(field)
            if first is None or first == "" or any(row.get(field) != first for row in rows):
                raise ValueError(f"{message}: {field}")


def _cohort(records, partial, check):
    indexed, complete = {}, {}
    for raw in records:
        row, done = validate_record(raw, check=check)
        key = row["policy"], row["replica"]
        if key in indexed:
            raise ValueError("duplicate policy/replica trajectory")
        indexed[key], complete[key] = row, done
    rows = list(indexed.values())
    if not rows:
        raise ValueError("no trajectories")
    N, namespace = rows[0]["target"], rows[0]["namespace"]
    if any(row["target"] != N or row["namespace"] != namespace for row in rows):
        raise ValueError("mixed target budgets or namespaces cannot be pooled")
    _same_optional(rows, ("source_hash", "config_hash", "scientific_hash", "chart_hash",
                         "parameter_order_hash", "layout_hash", "factor_mapping", "chart",
                         "smoke", "capacity", "replay_count", "fd_step", "probes", "max_iter", "max_eval"),
                   "incompatible cohort identity")
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        if selected:
            _same_optional(selected, ("recipe_key", "recipe", "combination_recipe", "policy_recipe",
                                     "policy_hash", "process_scale", "lfu_enabled", "pi_mode"),
                           "one frozen recipe required per policy")
    plan_members = {}
    for row in rows:
        plan = row.get("plan_hash")
        if not isinstance(plan, str) or not plan or row.get("pairing_id") != plan:
            raise ValueError("missing or inconsistent pairing provenance")
        if plan_members.setdefault(plan, row["replica"]) != row["replica"]:
            raise ValueError("distinct replicas cannot reuse a data plan")
        reference = next(r for r in rows if r["replica"] == row["replica"])
        _same_optional([reference, row], ("factor_seeds", "initialization_hash", "holdout_hash", "stream_hash"),
                       "unpaired replica identity")
        count = min(len(row["metrics"]), len(reference["metrics"]))
        if plan != reference["plan_hash"] or any(not math.isclose(
                row["metrics"][0][f"{domain}_mse"], reference["metrics"][0][f"{domain}_mse"],
                rel_tol=1e-10, abs_tol=1e-10) for domain in ("base", "novel")):
            raise ValueError("unpaired data plans or frozen initial predictions")
        first, second = row["metrics"][:count], reference["metrics"][:count]
        if [m["observations"] for m in first] != [m["observations"] for m in second]:
            raise ValueError("unpaired exposure clocks")
        for field in legacy.grid.DRIFT_FIELDS:
            if any(field in m for m in (*first, *second)) and (
                    not all(field in m for m in (*first, *second)) or any(
                        not math.isclose(_number(a[field], field), _number(b[field], field),
                                         rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(first, second))):
                raise ValueError("unpaired drift trajectory")
    replicas, excluded = [], []
    for replica in sorted({row["replica"] for row in rows}):
        missing = [p for p in POLICIES if (p, replica) not in indexed]
        unfinished = [p for p in POLICIES if (p, replica) in indexed and not complete[p, replica]]
        if missing or unfinished:
            excluded.append(dict(replica=replica, missing=missing, incomplete=unfinished))
        else:
            replicas.append(replica)
    if excluded and not partial:
        raise ValueError(f"incomplete paired six-policy blocks: {excluded}; use partial=True explicitly")
    if len(replicas) < 2:
        raise ValueError("at least two complete paired six-policy replicas required")
    return {p: [indexed[p, r] for r in replicas] for p in POLICIES}, replicas, N, namespace, excluded


def bootstrap_chunk(arrays, *, start, stop, seed=DEFAULT_SEED, check=lambda: None):
    """Checkpointable whole-six-arm draws, shared across A and B targets."""
    if (set(arrays) != set(POLICIES) or type(start) is not int or type(stop) is not int
            or not 0 <= start < stop):
        raise ValueError("six actual policies and a nonempty nonnegative draw range required")
    values = []
    for draw in range(start, stop):
        check()
        indices = replica_draw_indices(len(arrays["A"]["novel_mse"]), seed, draw)
        values.append(dict(draw=draw, indices=indices.tolist(), references={reference: {
            policy: reproduction._match(arrays[policy], arrays[reference], indices,
                                       reference_by_definition=policy == reference)
            for policy in (POLICIES if reference == "A" else ("C", "D", "E"))}
            for reference in ("A", "B")}))
    return values


def _efficiency(arm, reference, samples, points, summary, reference_summary, contrast, is_reference):
    point = reproduction._match(arm, reference, np.arange(len(samples[0])), reference_by_definition=is_reference)
    invalid = [dict(draw=i, **p) for i, p in enumerate(points) if p["status"] != "valid"]
    resolved = point["status"] == "valid" and not invalid
    ci = {f: np.quantile([p[f] for p in points], [.025, .975]).tolist()
          for f in MATCH_FIELDS} if resolved else None
    upper = {f: float(np.quantile([p[f] for p in points], .95))
             for f in RETENTION_FIELDS} if resolved else None
    full_checks = {name: bool(contrast[f"{name}_noninferiority_supported"]
                   and summary[name]["ceiling_supported"] and reference_summary[name]["ceiling_supported"])
                   for name in ("retention", "endpoint_retention")}
    full = all(full_checks.values())
    matched = (None if upper is None else bool(upper["old_rmse_difference"] <= TOLERANCE
        and upper["arm_old_deterioration"] <= CEILING and upper["reference_old_deterioration"] <= CEILING))
    numerator, denominator = arm["learning_seconds"][:, -1], reference["learning_seconds"][:, -1]
    boot_denominator = denominator[samples].mean(axis=1)
    return dict(point=point, full_budget_retention_supported=full,
        full_budget_retention_checks=full_checks, matched_budget_retention_supported=matched,
        retention_qualified=bool(full and matched), matched_upper_one_sided_95=upper,
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


def analyze(records, *, draws=2000, seed=DEFAULT_SEED, partial=False, check=lambda: None,
            load_bootstrap=None, save_bootstrap=None):
    """Analyze one frozen six-policy cohort; return finite JSON-friendly data.

    ``partial=True`` excludes only whole incomplete six-arm blocks and lists
    them explicitly; malformed/invalid records still raise. At least two
    complete blocks are required. ``efficiency[policy]`` uses A; the additional
    ``efficiency_vs_B`` mapping contains C/D/E. Contrasts are first minus second
    on risk/deterioration scales; acquisition_gain has the opposite sign.

    Bootstrap hooks are load_bootstrap(start)->chunk|None and
    save_bootstrap(start, chunk). Chunks bind the full retained input, seed,
    draw count and guards, carry a checksum, and resume without redrawing.
    """
    check()
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0 or type(partial) is not bool:
        raise ValueError("integer draws >= 2, nonnegative integer seed and boolean partial required")
    groups, replicas, N, namespace, excluded = _cohort(records, partial, check)
    arrays = {p: {f: np.asarray([[m[f] for m in row["metrics"]] for row in rows], dtype=float)
                  for f in FIELDS} for p, rows in groups.items()}
    samples = np.asarray([replica_draw_indices(len(replicas), seed, i) for i in range(draws)])
    identity = content_hash(dict(protocol=PROTOCOL, schema_version=1, namespace=namespace,
        records=[row for p in POLICIES for row in groups[p]], replicas=replicas, N=N,
        draws=draws, seed=seed, delta=CEILING, epsilon=TOLERANCE))
    inverses = {ref: {p: [] for p in (POLICIES if ref == "A" else ("C", "D", "E"))}
                for ref in ("A", "B")}
    for start in range(0, draws, 100):
        check()
        end = min(draws, start + 100)
        block = load_bootstrap(start) if load_bootstrap else None
        if block is None:
            block = dict(input_hash=identity, start=start, end=end, values=bootstrap_chunk(
                arrays, start=start, stop=end, seed=seed, check=check))
            block["payload_hash"] = content_hash(block)
            if save_bootstrap:
                save_bootstrap(start, block)
        else:
            try:
                valid = (block["input_hash"] == identity and block["start"] == start and block["end"] == end
                    and block["payload_hash"] == content_hash({k: v for k, v in block.items() if k != "payload_hash"})
                    and [v["draw"] for v in block["values"]] == list(range(start, end))
                    and all(v["indices"] == samples[v["draw"]].tolist()
                            and set(v["references"]) == {"A", "B"}
                            and all(set(v["references"][ref]) == set(inverses[ref]) for ref in inverses)
                            for v in block["values"]))
            except (KeyError, TypeError, ValueError, IndexError):
                valid = False
            if not valid:
                raise ValueError("bootstrap checkpoint differs from frozen input/settings or is corrupt")
        for value in block["values"]:
            for ref, policies in inverses.items():
                for p in policies:
                    policies[p].append(value["references"][ref][p])
    interval = lambda values: reproduction._interval(values, samples)
    final = (namespace == "fresh" and not partial and not excluded and N == 100
             and replicas == list(range(8)) and not any(row.get("smoke", False) for rows in groups.values() for row in rows))
    result = dict(protocol=PROTOCOL, schema_version=1, namespace=namespace, N=N, replicas=replicas,
        is_final=final, provisional=not final, input_hash=identity, delta=CEILING, epsilon=TOLERANCE,
        cohort=dict(expected_fresh_replicas=8, partial=partial, excluded_blocks=excluded),
        reference_policy="A", secondary_reference_policy="B", curves={}, summaries={}, contrasts={},
        efficiency={}, efficiency_vs_B={}, retention_comparisons={},
        bootstrap=dict(seed=seed, seed_name=f"{PROTOCOL}:{namespace}:paired_bootstrap", draws=draws,
            confidence_level=.95, resampling_unit="paired_whole_six_policy_replica", shared_across=list(POLICIES),
            curve_bands="pointwise_not_simultaneous"),
        units=dict(risk="square_pixels_per_landmark", error="RMS_landmark_pixels", observation="whole_grid"),
        assumptions="Nominal paired intervals conditional on frozen calibrated policies, source data and fixed "
            "holdouts; not familywise inference or significance stopping. The caller verifies freshness/seals. "
            "Interaction compares calibrated policies. R is outside development retention matching.")
    vectors = {}
    for policy, fields in arrays.items():
        check()
        exposure, query, base = (fields[f] for f in ("observations", "novel_mse", "base_mse"))
        A = query[:, 1:].mean(axis=1)
        Havg = np.sqrt(base[:, 1:].mean(axis=1)) - np.sqrt(base[:, 0])
        H100 = np.sqrt(base[:, -1]) - np.sqrt(base[:, 0])
        Q = ((query[:, 1:] + query[:, :-1]) * .5 * np.diff(exposure, axis=1)).sum(axis=1) / exposure[:, -1]
        vectors[policy] = dict(acquisition=A, retention=Havg, endpoint_retention=H100,
            endpoint_mse=query[:, -1], all_arrival_risk=Q)
        curves = dict(axis="designated_novel_grids", n=list(range(N + 1)),
            observations=exposure.mean(axis=0).tolist(), per_replica_observations=exposure.tolist())
        for domain, values in (("query", query), ("base", base)):
            means = values[samples].mean(axis=1)
            risk, ci = values.mean(axis=0), np.quantile(means, [.025, .975], axis=0)
            curves[domain] = dict(mean_mse=risk.tolist(), ci_mse=ci.tolist(),
                mean_rmse=np.sqrt(risk).tolist(), ci_rmse=np.sqrt(ci).tolist(),
                frozen_mean_mse=float(values[:, 0].mean()))
        summary = {name: interval(value) for name, value in vectors[policy].items()}
        for name in ("retention", "endpoint_retention"):
            summary[name].update(point_ceiling_feasible=summary[name]["mean"] <= CEILING,
                ceiling_supported=summary[name]["upper_one_sided_95"] <= CEILING)
        summary.update(frozen_mse=interval(query[:, 0]), frozen_old_rmse=interval(np.sqrt(base[:, 0])),
            endpoint_old_rmse=interval(np.sqrt(base[:, -1])), frozen_gain=interval(query[:, 0] - A),
            learning_seconds=interval(fields["learning_seconds"][:, -1]),
            retired_training_cohort=legacy._retired(groups[policy], N),
            per_replica=dict(replicas=replicas, observations=exposure[:, -1].tolist(),
                **{name: v.tolist() for name, v in vectors[policy].items()}))
        result["curves"][policy], result["summaries"][policy] = curves, summary

    def contrast(first, second):
        v, ref = vectors[first], vectors[second]
        risk = interval(v["acquisition"] - ref["acquisition"])
        gain = interval(ref["acquisition"] - v["acquisition"])
        gain["favorable_replicas"] = int(np.sum(v["acquisition"] < ref["acquisition"]))
        out = dict(first=first, second=second, acquisition_risk_difference=risk,
                   acquisition_gain=gain, favorable_replicas=gain["favorable_replicas"],
                   endpoint_risk_difference=interval(v["endpoint_mse"] - ref["endpoint_mse"]),
                   all_arrival_risk_difference=interval(v["all_arrival_risk"] - ref["all_arrival_risk"]))
        for field, name in (("retention", "retention_difference"),
                            ("endpoint_retention", "endpoint_retention_difference")):
            difference = interval(v[field] - ref[field])
            out[name] = difference
            out[f"{field}_noninferiority_supported"] = difference["upper_one_sided_95"] <= TOLERANCE
            out[f"{field}_equivalent"] = difference["ci"][0] >= -TOLERANCE and difference["ci"][1] <= TOLERANCE
        out["retention_equivalent"] = out["retention_equivalent"] and out["endpoint_retention_equivalent"]
        return out

    for key in ("B-R", "C-B", "D-B", "E-B", "B-A", "E-A", "E-R"):
        result["contrasts"][key] = contrast(*key.split("-"))
    for second, first in combinations(DIAGONAL_POLICIES, 2):
        result["retention_comparisons"][f"{first}-{second}"] = contrast(first, second)
    interaction = {field: vectors["E"][field] - vectors["C"][field] - vectors["D"][field] + vectors["B"][field]
                   for field in vectors["B"]}
    result["contrasts"]["interaction"] = dict(
        coefficients=dict(E=1, C=-1, D=-1, B=1),
        acquisition_risk_difference=interval(interaction["acquisition"]),
        endpoint_risk_difference=interval(interaction["endpoint_mse"]),
        all_arrival_risk_difference=interval(interaction["all_arrival_risk"]),
        retention_difference=interval(interaction["retention"]),
        endpoint_retention_difference=interval(interaction["endpoint_retention"]),
        favorable_replicas=int(np.sum(interaction["acquisition"] < 0)),
        interpretation="E-C-D+B on the risk scale; negative favors the joint increment between calibrated policies.")
    for ref, destination in (("A", "efficiency"), ("B", "efficiency_vs_B")):
        for policy, points in inverses[ref].items():
            result[destination][policy] = _efficiency(arrays[policy], arrays[ref], samples, points,
                result["summaries"][policy], result["summaries"][ref], contrast(policy, ref), policy == ref)
    return result
