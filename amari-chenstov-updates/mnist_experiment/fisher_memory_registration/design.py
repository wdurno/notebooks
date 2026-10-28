"""Bounded Phase 10 development decisions, independently within each task."""

from collections import Counter, defaultdict
import math

import numpy as np

from mnist_experiment.fisher_memory_regression.config import METHODS, Recipe
from .analysis import FD, REPLAY, EWC, _number, _replicas, validate_records


def initial_grid(cfg, lambda0):
    """25 recipes at production defaults: 5 replay, 5 EWC and 15 FD."""
    lambda0 = _number(lambda0, "lambda0")
    if lambda0 <= 0:
        raise ValueError("positive calibrated FD scale required")
    recipes = []
    for ridge in cfg.ridge_grid:
        recipes.append(Recipe.from_config(cfg, EWC, ridge=ridge))
        recipes.append(Recipe.from_config(cfg, REPLAY, ridge=ridge))
        recipes.extend(Recipe.from_config(cfg, FD, ridge=ridge, strength=lambda0 * ratio)
                       for ratio in cfg.strength_ratios)
    if len({r.key for r in recipes}) != len(recipes):
        raise ValueError("duplicate grid recipes")
    return sorted(recipes, key=lambda r: (r.ridge, METHODS.index(r.method), r.strength, r.key))


def table(records, cfg, replicas=None, *, partial=False, learning_limit=math.inf, memory_limit=math.inf):
    """Per-policy summaries on one exact paired development cohort.

    learning_limit is an optional operational runtime admission limit, NOT the
    FD policy allowance. The latter is relative to selected replay in select().
    """
    expected = _replicas(range(cfg.calibration_replicas) if replicas is None else replicas)
    learning_limit = _number(learning_limit, "learning limit", infinity=True)
    memory_limit = _number(memory_limit, "memory limit", infinity=True)
    groups = defaultdict(list)
    for row in validate_records(records, cfg.novel_count, "development"):
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
            raise ValueError("policy changed between replicas")
        H = [math.sqrt(r["base_auc"]) - math.sqrt(r["frozen_base_mse"]) for r in group]
        harm = float(np.mean(H))
        cost = sum(r["learning_seconds"] for r in group) / sum(r["observations"] for r in group)
        valid, memory = all(r["valid"] for r in group), max(r["peak_vram_bytes"] for r in group)
        feasible = bool(complete and valid and cost <= learning_limit and memory <= memory_limit)
        result.append(dict(recipe_key=key, recipe=recipe.mapping(), method=recipe.method,
            context_key=group[0]["context_key"], output_dim=group[0]["output_dim"],
            capacity=recipe.capacity, replay_count=recipe.replay_count, replicas=[r["replica"] for r in group],
            count=len(group), complete=complete, valid=valid, resource_feasible=feasible,
            novel_auc=float(np.mean([r["novel_auc"] for r in group])), base_harm=harm, H=harm,
            per_replica_H=H, frozen_novel_mse=float(np.mean([r["frozen_novel_mse"] for r in group])),
            seconds_per_arrival=cost, mean_learning_seconds=float(np.mean([r["learning_seconds"] for r in group])),
            mean_active_seconds=float(np.mean([r["active_seconds"] for r in group])), max_vram_bytes=memory,
            ceiling_feasible=bool(feasible and harm <= cfg.retention_margin)))
    return result


def acquisition_order(row):
    r = row["recipe"]
    return (row["novel_auc"], row["seconds_per_arrival"], r["ridge"], r["strength"],
            r["capacity"], r["replay_count"], r["fd_step"], r["probes"], r["max_iter"], r["max_eval"], row["recipe_key"])


def _selection_rows(rows):
    rows = list(rows)
    if len({r["context_key"] for r in rows}) > 1:
        raise ValueError("selection requires one task")
    if len({tuple(r["replicas"]) for r in rows if r["complete"]}) > 1:
        raise ValueError("same complete paired replica cohort required")
    if len({r["recipe_key"] for r in rows}) != len(rows):
        raise ValueError("duplicate policy summary")
    return [r for r in rows if r["valid"] and r["complete"]]


def _failures(row, cfg, replay=None):
    failed = []
    if not row["resource_feasible"]:
        failed.append("runtime_or_memory_admission")
    if _exceeds(row["base_harm"], cfg.retention_margin):
        failed.append("retention_ceiling")
    if row["method"] == FD:
        if _exceeds(row["base_harm"] - replay["base_harm"], cfg.matching_tolerance):
            failed.append("retention_noninferiority")
        if _exceeds(row["seconds_per_arrival"], cfg.envelope_factor * replay["seconds_per_arrival"]):
            failed.append("selected_replay_cost")
    return failed


def _exceeds(value, ceiling):
    # Roots and ratio aggregation can round an exact boundary upward by one ULP.
    return value > ceiling and not math.isclose(value, ceiling, rel_tol=1e-12, abs_tol=1e-12)


def _diagnostic_order(row, cfg, replay=None):
    violation = max(0., row["base_harm"] - cfg.retention_margin)
    if replay is not None and row["method"] == FD:
        violation = max(violation, row["base_harm"] - replay["base_harm"] - cfg.matching_tolerance)
    return (violation, *acquisition_order(row))


def select(rows, cfg):
    """Replay first; never weaken its acquisition to make FD eligible."""
    rows = _selection_rows(rows)
    choices = {}
    for method in (REPLAY, FD, EWC):
        candidates = [r for r in rows if r["method"] == method]
        if not candidates:
            raise ValueError(f"no complete valid development candidate: {method}")
        replay = choices[REPLAY]["selected"] if REPLAY in choices else None
        feasible = [r for r in candidates if not _failures(r, cfg, replay)]
        chosen = min(feasible, key=acquisition_order) if feasible else min(
            candidates, key=lambda r: _diagnostic_order(r, cfg, replay))
        failures = _failures(chosen, cfg, replay)
        choices[method] = dict(selected=chosen, eligible=not failures, failed_conditions=failures,
            reason="development best acquisition under one-sided ceilings" if not failures else
                   "least-retention-violating diagnostic: " + ", ".join(failures))
    return choices


def shortlist(rows, cfg):
    """Selected then best feasible distinct alternative, at most two per arm."""
    rows = _selection_rows(rows)
    selected = select(rows, cfg)
    replay = selected[REPLAY]["selected"]
    result = {}
    for method in METHODS:
        first = selected[method]["selected"]
        others = [r for r in rows if r["method"] == method and r["recipe_key"] != first["recipe_key"]]
        feasible = [r for r in others if not _failures(r, cfg, replay)]
        second = min(feasible, key=acquisition_order) if feasible else (min(
            others, key=lambda r: _diagnostic_order(r, cfg, replay)) if others else None)
        result[method] = [first["recipe"], *([second["recipe"]] if second else [])]
    return result


def refinement_proposals(rows, cfg, additions, lambda0):
    """Deterministic sealed proposals; all prior seals consume the lifetime cap.

    Cost-excluded but valid FD candidates remain in the frontier. Reserve two
    FD and two replay proposals before EWC or borrowing unused primary slots.
    No assumed monotonicity across observed reversals; extend a boundary once.
    """
    lambda0 = _number(lambda0, "lambda0")
    if lambda0 <= 0:
        raise ValueError("positive lambda0 required")
    additions, original = list(additions), list(rows)
    cap = min(8, cfg.refinement_settings)
    remaining = cap - len(additions)
    if remaining <= 0:
        return []
    rows = _selection_rows(original)
    if not rows:
        return []
    replay = select(rows, cfg)[REPLAY]["selected"]
    cost_limit = cfg.envelope_factor * replay["seconds_per_arrival"]
    used = {r["recipe_key"] for r in original} | {Recipe.from_mapping(a["recipe"]).key for a in additions}
    counts = Counter(a["recipe"]["method"] for a in additions)
    groups, proposals = defaultdict(list), {}
    for row in rows:
        r = row["recipe"]
        groups[(r["method"], r["replay_count"], r["strength"])].append(row)

    def proximity(row):
        retention = abs(row["base_harm"] - cfg.retention_margin)
        cost = abs(row["seconds_per_arrival"] / cost_limit - 1) if cost_limit > 0 else math.inf
        return min(retention, cost if row["method"] == FD else math.inf)

    def propose(source, kind, priority, **changes):
        recipe = Recipe.from_mapping({**source["recipe"], **changes})
        if recipe.key in used:
            return
        item = dict(recipe=recipe.mapping(), recipe_key=recipe.key, kind=kind,
                    priority=[priority, proximity(source), *acquisition_order(source)])
        if recipe.key not in proposals or item["priority"] < proposals[recipe.key]["priority"]:
            proposals[recipe.key] = item

    for identity, points in sorted(groups.items()):
        points.sort(key=lambda r: r["recipe"]["ridge"])
        near = min(points, key=lambda r: (proximity(r), *acquisition_order(r)))
        near_pair = min(zip(points, points[1:]), default=None,
                        key=lambda pair: min(proximity(r) for r in pair))
        for a, b in zip(points, points[1:]):
            retention_boundary = ((a["base_harm"] - cfg.retention_margin)
                                  * (b["base_harm"] - cfg.retention_margin) <= 0)
            cost_boundary = identity[0] == FD and ((a["seconds_per_arrival"] - cost_limit)
                                                  * (b["seconds_per_arrival"] - cost_limit) <= 0)
            priority = 0 if retention_boundary or cost_boundary else 1 if (a is near_pair[0] and b is near_pair[1]) else 4
            propose(min((a, b), key=acquisition_order), "midpoint", priority,
                    ridge=math.sqrt(a["recipe"]["ridge"] * b["recipe"]["ridge"]))
        history = [a for a in additions if (a["recipe"]["method"], a["recipe"]["replay_count"],
                                           a["recipe"]["strength"]) == identity]
        if len(points) >= 2:
            diffs = np.diff([p["base_harm"] for p in points])
            reversal = bool((diffs > 0).any() and (diffs < 0).any())
            for side, edge, neighbor, factor in (("low", points[0], points[1], .1),
                                                  ("high", points[-1], points[-2], 10.)):
                trend = edge["base_harm"] - neighbor["base_harm"]
                prior = any(a["kind"] == "ridge-edge-" + side for a in history)
                if not reversal and not prior and trend * (cfg.retention_margin - edge["base_harm"]) > 0:
                    propose(edge, "ridge-edge-" + side, 2, ridge=edge["recipe"]["ridge"] * factor)
        if identity[0] == FD:
            siblings = [r for r in rows if r["method"] == FD and r["recipe"]["ridge"] == near["recipe"]["ridge"]]
            strengths = sorted({r["recipe"]["strength"] for r in siblings})
            best = min(siblings, key=acquisition_order)
            if len(strengths) >= 2 and near is best:
                for side, boundary, factor in (("low", strengths[0], .1), ("high", strengths[-1], 10.)):
                    prior = any(a["kind"] == "strength-edge-" + side and
                                a["recipe"]["ridge"] == near["recipe"]["ridge"] for a in additions)
                    if identity[2] == boundary and not prior:
                        propose(near, "strength-edge-" + side, 3, strength=boundary * factor)

    pending, ordered = list(proposals.values()), []
    # Reserve slots even when a primary method already has feasible settings.
    for _ in range(2):
        for method in (FD, REPLAY):
            if counts[method] >= 2:
                continue
            candidates = [p for p in pending if p["recipe"]["method"] == method]
            if not candidates or len(ordered) >= remaining:
                break
            chosen = min(candidates, key=lambda p: (p["priority"], p["recipe_key"]))
            ordered.append(chosen)
            counts[method] += 1
            pending.remove(chosen)
    while pending and len(ordered) < remaining:
        chosen = min(pending, key=lambda p: (p["recipe"]["method"] == EWC, p["priority"], p["recipe_key"]))
        ordered.append(chosen)
        pending.remove(chosen)
    return ordered
