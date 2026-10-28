"""Bounded, deterministic Phase 9 development decisions, never confirmation."""

from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from mnist_experiment.fisher_memory_regression.config import METHODS, Recipe

from .analysis import FD, REPLAY, _number, _replicas, flatten_record, point_equivalence, validate_records

HISTORICAL_STRENGTH = 2.6356432540763373


def initial_grid(cfg, lambda0):
    """27 settings at the default grids; capacity is fixed, not crossed."""
    lambda0 = _number(lambda0, "lambda0")
    if lambda0 <= 0:
        raise ValueError("positive calibrated FD scale required")
    recipes = []
    for ridge in cfg.ridge_grid:
        recipes.append(Recipe.from_config(cfg, "ewc_only", ridge=ridge))
        for b in cfg.replay_counts:
            recipes.append(Recipe.from_config(cfg, REPLAY, replay_count=b, ridge=ridge))
            recipes.extend(Recipe.from_config(cfg, FD, replay_count=b, ridge=ridge,
                strength=lambda0 * ratio) for ratio in cfg.strength_ratios)
    return sorted(recipes, key=lambda r: (abs(math.log(r.ridge / 10)), r.replay_count,
                                          METHODS.index(r.method), r.strength, r.key))


def table(records, cfg, replicas=None, *, partial=False, learning_limit=math.inf, memory_limit=math.inf):
    """Per-policy full-window summaries on the same planned paired replicas.

    Caller filters one context. Partial rows are descriptive only: ``complete``
    is false and selection/refinement cannot use them. Failed replicas invalidate
    the entire policy; no complete-case deletion changes its replica cohort.
    """
    expected = _replicas(range(cfg.calibration_replicas) if replicas is None else replicas)
    learning_limit = _number(learning_limit, "learning limit", infinity=True)
    memory_limit = _number(memory_limit, "memory limit", infinity=True)
    rows = validate_records(records, cfg.novel_count, "development")
    groups = defaultdict(list)
    for row in rows:
        if row["replica"] not in expected:
            raise ValueError("unplanned development replica")
        groups[row["recipe_key"]].append(row)
    result = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: r["replica"])
        complete = [r["replica"] for r in group] == expected
        if not complete and not partial:
            continue
        recipe = Recipe.from_mapping(group[0]["recipe"])
        if any(Recipe.from_mapping(r["recipe"]) != recipe for r in group):
            raise ValueError("policy settings changed between replicas")
        H = [math.sqrt(r["base_auc"]) - math.sqrt(r["frozen_base_mse"]) for r in group]
        cost = sum(r["learning_seconds"] for r in group) / sum(r["observations"] for r in group)
        valid = all(r["valid"] for r in group)
        memory = max(r["peak_vram_bytes"] for r in group)
        feasible = bool(complete and valid and cost <= learning_limit and memory <= memory_limit)
        harm = float(np.mean(H))
        result.append(dict(recipe_key=key, recipe=recipe.mapping(), method=recipe.method,
            context_key=group[0]["context_key"], capacity=recipe.capacity, replay_count=recipe.replay_count,
            replicas=[r["replica"] for r in group], count=len(group), complete=complete, valid=valid,
            novel_auc=float(np.mean([r["novel_auc"] for r in group])), base_harm=harm, H=harm,
            per_replica_H=H, frozen_novel_mse=float(np.mean([r["frozen_novel_mse"] for r in group])),
            seconds_per_arrival=cost, mean_learning_seconds=float(np.mean([r["learning_seconds"] for r in group])),
            mean_active_seconds=float(np.mean([r["active_seconds"] for r in group])), max_vram_bytes=memory,
            resource_feasible=feasible, ceiling_feasible=bool(feasible and harm <= cfg.retention_margin_degrees),
            in_band=bool(feasible and cfg.retention_margin_degrees - cfg.matching_tolerance <= harm
                         <= cfg.retention_margin_degrees)))
    return result


def acquisition_order(row):
    recipe = row["recipe"]
    return (row["novel_auc"], row["seconds_per_arrival"], recipe["replay_count"],
            recipe["ridge"], recipe["strength"], row["recipe_key"])


def proximity_order(row, cfg):
    middle = cfg.retention_margin_degrees - cfg.matching_tolerance / 2
    return (abs(row["base_harm"] - middle), *acquisition_order(row))


def _selection_rows(rows):
    rows = list(rows)
    if len({r.get("context_key") for r in rows}) > 1:
        raise ValueError("selection requires one context")
    cohorts = {tuple(r["replicas"]) for r in rows if r.get("complete", True)}
    if len(cohorts) > 1:
        raise ValueError("selection requires the same complete paired replica cohort")
    if len({r["recipe_key"] for r in rows}) != len(rows):
        raise ValueError("duplicate policy summary")
    return [r for r in rows if r["valid"] and r.get("complete", True)]


def select(rows, cfg):
    rows = _selection_rows(rows)
    result = {}
    for method in METHODS:
        valid = [r for r in rows if r["method"] == method]
        if not valid:
            raise ValueError(f"no complete valid development candidate: {method}")
        pool = [r for r in valid if r["resource_feasible"]]
        if not pool:
            chosen = min(valid, key=lambda r: (r["seconds_per_arrival"], *acquisition_order(r)))
            reason = "resource-infeasible diagnostic"
        else:
            band = [r for r in pool if r["in_band"]]
            if band:
                chosen = min(band, key=acquisition_order)
                joint = [r for r in pool if r["base_harm"] < chosen["base_harm"]
                         and r["novel_auc"] < chosen["novel_auc"]]
                if joint:
                    chosen, reason = min(joint, key=acquisition_order), "below-band joint-improvement candidate"
                else:
                    reason = "development retention-matched"
            else:
                chosen, reason = min(pool, key=lambda r: proximity_order(r, cfg)), "retention-unmatched diagnostic"
                joint = [r for r in pool if r["base_harm"] < chosen["base_harm"]
                         and r["novel_auc"] < chosen["novel_auc"]]
                if joint:
                    chosen, reason = min(joint, key=acquisition_order), "joint-improvement candidate outside matching band"
        result[method] = dict(selected=chosen, reason=reason)
    return result


def shortlist(rows, cfg):
    """Selected recipe first, at most one distinct retention alternative."""
    choices, rows = select(rows, cfg), _selection_rows(rows)
    result = {}
    for method in METHODS:
        first = choices[method]["selected"]
        others = [r for r in rows if r["method"] == method and r["recipe_key"] != first["recipe_key"]]
        others = [r for r in others if r["resource_feasible"]] or others
        feasible = [r for r in others if r["ceiling_feasible"]]
        second = min(feasible, key=acquisition_order) if feasible else (
            min(others, key=lambda r: proximity_order(r, cfg)) if others else None)
        result[method] = [first["recipe"], *([second["recipe"]] if second else [])]
    return result


def refinement_proposals(rows, cfg, additions, lambda0):
    """Next deterministic bounded candidates, ready to seal before execution.

    ``additions`` includes every previously sealed proposal, even if pending or
    failed. Its length consumes the lifetime cap. Primary unresolved methods
    retain two slots each; observed reversals never imply a monotone extension.
    """
    lambda0 = _number(lambda0, "lambda0")
    if lambda0 <= 0:
        raise ValueError("positive lambda0 required")
    additions = list(additions)
    cap = min(8, cfg.refinement_settings)
    remaining = cap - len(additions)
    if remaining <= 0:
        return []
    original = list(rows)
    rows = _selection_rows(original)
    used = {r["recipe_key"] for r in original} | {Recipe.from_mapping(a["recipe"]).key for a in additions}
    counts = Counter(a["recipe"]["method"] for a in additions)
    unresolved = {m for m in (FD, REPLAY) if not any(r["method"] == m and r["in_band"] for r in rows)}
    groups = defaultdict(list)
    for row in rows:
        if row["resource_feasible"]:
            r = row["recipe"]
            groups[(r["method"], r["replay_count"], r["strength"])].append(row)
    proposals = {}

    def propose(source, kind, priority, **changes):
        recipe = Recipe.from_mapping({**source["recipe"], **changes})
        if recipe.key not in used:
            item = dict(recipe=recipe.mapping(), recipe_key=recipe.key, kind=kind,
                        priority=[priority, *proximity_order(source, cfg)])
            prior = proposals.get(recipe.key)
            if prior is None or item["priority"] < prior["priority"]:
                proposals[recipe.key] = item

    lo, hi = cfg.retention_margin_degrees - cfg.matching_tolerance, cfg.retention_margin_degrees
    middle = (lo + hi) / 2
    for identity, points in sorted(groups.items()):
        points.sort(key=lambda r: r["recipe"]["ridge"])
        near = min(points, key=lambda r: proximity_order(r, cfg))
        for a, b in zip(points, points[1:]):
            harms = sorted((a["base_harm"], b["base_harm"]))
            if harms[0] < hi and harms[1] > lo and harms[0] != harms[1]:
                propose(min((a, b), key=lambda r: proximity_order(r, cfg)), "midpoint", 0,
                        ridge=math.sqrt(a["recipe"]["ridge"] * b["recipe"]["ridge"]))
        history = [a for a in additions if (a["recipe"]["method"], a["recipe"]["replay_count"],
                                          a["recipe"]["strength"]) == identity]
        # Extend once only, following the observed edge slope toward the band.
        if len(points) >= 2 and not any(p["in_band"] for p in points) and not any(a["kind"] == "edge" for a in history):
            for edge, neighbor, factor in ((points[0], points[1], .1), (points[-1], points[-2], 10.)):
                trend = edge["base_harm"] - neighbor["base_harm"]
                differences = np.diff([p["base_harm"] for p in points])
                reversal = bool((differences > 0).any() and (differences < 0).any())
                if not reversal and trend * (middle - edge["base_harm"]) > 0:
                    propose(edge, "edge", 1, ridge=edge["recipe"]["ridge"] * factor)
        if identity[0] == FD:
            siblings = [r for r in rows if r["method"] == FD and r["resource_feasible"]
                        and r["replay_count"] == identity[1] and r["recipe"]["ridge"] == near["recipe"]["ridge"]]
            best = min(siblings, key=acquisition_order)
            strengths = sorted(r["recipe"]["strength"] for r in siblings)
            strength_extended = any(a["kind"] == "strength-edge" and a["recipe"]["replay_count"] == identity[1]
                                    for a in additions)
            if near is best and len(strengths) >= 2 and not strength_extended:
                if identity[2] == strengths[0]:
                    propose(near, "strength-edge", 2, strength=identity[2] / 10)
                if identity[2] == strengths[-1]:
                    propose(near, "strength-edge", 2, strength=identity[2] * 10)
            if not any(math.isclose(s, HISTORICAL_STRENGTH, rel_tol=1e-12) for s in strengths):
                propose(near, "historical-strength", 3, strength=HISTORICAL_STRENGTH)

    # Reserve primary slots before allowing one method to consume the remainder.
    pending = list(proposals.values())
    ordered = []
    while pending and len(ordered) < remaining:
        need = {m: max(0, 2 - counts[m]) for m in unresolved}
        must_reserve = sum(need.values()) >= remaining - len(ordered)
        available = [p for p in pending if not must_reserve or need.get(p["recipe"]["method"], 0)]
        if not available:
            break
        chosen = min(available, key=lambda p: (p["priority"][0],
            not bool(need.get(p["recipe"]["method"], 0)), counts[p["recipe"]["method"]],
            p["priority"][1:], p["recipe_key"]))
        ordered.append(chosen)
        counts[chosen["recipe"]["method"]] += 1
        pending.remove(chosen)
    return ordered


def context_score(records, cfg, limits=None, *, replicas=None, learning_limit=None, memory_limit=None):
    """Two/four-replica exploratory policy selection and point S, no bootstrap.

    Exposes selected, point, acquisition_gain, retention_match, eligible. A
    matching-qualified point allows FD better retention, but never a ceiling
    violation. Context-specific replay endpoint q is recomputed by _point.
    """
    limits = {} if limits is None else limits
    records = [flatten_record(r) for r in records]
    replicas = _replicas(sorted({r["replica"] for r in records}) if replicas is None else replicas)
    if len(replicas) not in (cfg.calibration_replicas, cfg.calibration_replicas + cfg.refinement_replicas):
        raise ValueError("context scoring requires the complete development cohort")
    rows = table(records, cfg, replicas=replicas,
        learning_limit=limits.get("learning_limit", limits.get("learning_time_limit", math.inf))
                       if learning_limit is None else learning_limit,
        memory_limit=limits.get("memory_limit", limits.get("memory_ceiling_bytes", math.inf))
                     if memory_limit is None else memory_limit)
    selected = select(rows, cfg)
    groups = {m: sorted((r for r in records if r["recipe_key"] == selected[m]["selected"]["recipe_key"]),
                        key=lambda r: r["replica"]) for m in METHODS}
    point = point_equivalence(groups)
    f, r = (selected[m]["selected"] for m in (FD, REPLAY))
    gain = r["novel_auc"] - f["novel_auc"]
    resource = f["resource_feasible"] and r["resource_feasible"]
    difference = f["base_harm"] - r["base_harm"]
    retention_match = bool(resource and f["ceiling_feasible"] and r["ceiling_feasible"]
                           and difference <= cfg.matching_tolerance)
    eligible = bool(point["status"] == "valid" and point["useful_target"]
                    and resource and retention_match and gain > 0)
    warnings = []
    if not resource:
        warnings.append("resource-infeasible")
    if not retention_match:
        warnings.append("retention-unmatched")
    if not point["useful_target"]:
        warnings.append("replay_target_improves_frozen_MSE_by_less_than_5_percent")
    if point["status"] != "valid":
        warnings.append("sample equivalence unresolved: " + point["status"])
    if gain <= 0:
        warnings.append("nonpositive acquisition gain")
    return dict(context_key=f["context_key"], selected=selected, point=point,
        acquisition_gain=gain, G=gain, retention_match=retention_match, eligible=eligible,
        resource_feasible=bool(resource),
        retention_label="equivalent_at_point" if retention_match and abs(difference) <= cfg.matching_tolerance
                        else "FD_better_retention" if retention_match else "unmatched",
        fd_cost=f["seconds_per_arrival"], warnings=warnings, replicas=replicas)


def context_rank_key(score):
    """Ascending key: eligible S, then G, then lower actual FD cost/identity."""
    eligible = score["eligible"]
    return (not eligible, -score["point"]["S"] if eligible else 0.,
            -score["acquisition_gain"], score["fd_cost"], score["context_key"])
