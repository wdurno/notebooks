"""Deterministic full-budget search and retention matching, development only."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from mnist_experiment.fisher_memory_regression.config import Recipe, METHODS
from mnist_experiment.fisher_memory_regression.analysis import summarize_trajectory

from . import PROTOCOL


def initial_grid(cfg):
    recipes = []
    for c in cfg.ridge_grid:
        recipes.append(Recipe.from_config(cfg, "ewc_only", ridge=c))
        for b in cfg.replay_counts:
            recipes.append(Recipe.from_config(cfg, "replay_ewc", replay_count=b, ridge=c))
            recipes.extend(Recipe.from_config(cfg, "fd_replay_ewc", replay_count=b,
                ridge=c, strength=cfg.historical_strength * ratio) for ratio in cfg.strength_ratios)
    # Start with reference ridge, then interleave methods at every replica.
    return sorted(recipes, key=lambda r: (abs(math.log(r.ridge / 10)), r.replay_count,
                                          METHODS.index(r.method), r.strength))


def table(records, cfg, replicas=None, *, partial=False):
    records = list(records)
    if any(row.get("partition") != "development" for row in records):
        raise ValueError("confirmation must not influence development selection")
    expected = set(range(cfg.calibration_replicas)) if replicas is None else set(replicas)
    groups=defaultdict(list)
    paired={}
    plans={}
    for record in records:
        if record.get("protocol")!=PROTOCOL or record.get("schema_version")!=1 or record.get("family")!="development":
            raise ValueError("incompatible development protocol/family")
        recipe=Recipe.from_mapping(record["recipe"])
        if recipe.key!=record["recipe_key"] or recipe.method!=record["method"]:
            raise ValueError("development recipe identity mismatch")
        replica=record["replica"]
        if type(replica) is not int or replica<0: raise ValueError("invalid replica")
        if record["target"]!=cfg.novel_count: raise ValueError("full-budget development evidence required")
        summary=summarize_trajectory(record["metrics"],N=cfg.novel_count)
        for name in ("novel_auc","base_auc","frozen_base_mse","frozen_novel_mse","observations","learning_seconds"):
            if not math.isclose(summary[name],record[name],rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError("development summary differs from predictions")
        reference=paired.setdefault(replica,record)
        if not record.get("pairing_id") or record.get("pairing_id")!=record.get("plan_hash"):
            raise ValueError("missing development pairing identity")
        if plans.setdefault(record["plan_hash"],replica)!=replica:
            raise ValueError("distinct replicas reuse a data plan")
        if record["pairing_id"]!=reference["pairing_id"] or any(
            not math.isclose(record[k],reference[k],rel_tol=1e-10,abs_tol=1e-10)
            for k in ("frozen_base_mse","frozen_novel_mse")):
            raise ValueError("unpaired development initialization/panels")
        if [m["observations"] for m in record["metrics"]]!=[m["observations"] for m in reference["metrics"]]:
            raise ValueError("unpaired development exposure counts")
        for name in ("active_seconds","peak_vram_bytes"):
            if not math.isfinite(record[name]) or record[name]<0: raise ValueError("invalid development resource metric")
        if type(record.get("valid")) is not bool: raise ValueError("missing trajectory validity")
        groups[recipe.key].append(record)
    rows=[]
    for key,group in groups.items():
        if len({r["replica"] for r in group})!=len(group): raise ValueError("duplicate development replica")
        recipe=group[0]["recipe"]
        rows.append(dict(recipe_key=key,recipe=recipe,method=recipe["method"],capacity=recipe["capacity"],
            replay_count=recipe["replay_count"],replicas=sorted(r["replica"] for r in group),count=len(group),
            novel_auc=float(np.mean([r["novel_auc"] for r in group])),
            base_harm=float(np.mean([math.sqrt(r["base_auc"])-math.sqrt(r["frozen_base_mse"]) for r in group])),
            frozen_novel_mse=float(np.mean([r["frozen_novel_mse"] for r in group])),
            seconds_per_arrival=sum(r["learning_seconds"] for r in group)/sum(r["observations"] for r in group),
            mean_active_seconds=float(np.mean([r["active_seconds"] for r in group])),
            valid=all(r["valid"] for r in group),max_vram_bytes=max(r["peak_vram_bytes"] for r in group)))
    result = []
    for row in rows:
        if not partial and set(row["replicas"]) != expected:
            continue
        row["resource_feasible"] = bool(row["valid"] and
            row["seconds_per_arrival"] <= cfg.learning_time_limit and
            row["max_vram_bytes"] <= cfg.memory_ceiling_bytes)
        row["ceiling_feasible"] = bool(row["resource_feasible"] and
                                      row["base_harm"] <= cfg.retention_margin_degrees)
        row["in_band"] = bool(row["ceiling_feasible"] and row["base_harm"] >=
                              cfg.retention_margin_degrees - cfg.matching_tolerance)
        result.append(row)
    return sorted(result,key=lambda r:r["recipe_key"])


def acquisition_order(row):
    r = row["recipe"]
    return (row["novel_auc"], row["seconds_per_arrival"], r["replay_count"], r["ridge"], r["strength"])


def proximity_order(row, cfg):
    middle = cfg.retention_margin_degrees - cfg.matching_tolerance / 2
    return (abs(row["base_harm"] - middle), *acquisition_order(row))


def shortlist(rows, cfg):
    result = {}
    for method in METHODS:
        valid = [row for row in rows if row["method"] == method and row["valid"]]
        if not valid:
            raise ValueError(f"no valid development candidate: {method}")
        eligible = [row for row in valid if row["resource_feasible"]]
        pool = eligible or valid
        band = [row for row in pool if row["in_band"]]
        first = min(band, key=acquisition_order) if band else min(pool, key=lambda r: proximity_order(r, cfg))
        others = [row for row in pool if row["recipe_key"] != first["recipe_key"]]
        feasible = [row for row in others if row["ceiling_feasible"]]
        second = min(feasible, key=acquisition_order) if feasible else (
            min(others, key=lambda r: proximity_order(r, cfg)) if others else None)
        result[method] = [first["recipe"], *([second["recipe"]] if second else [])]
    return result


def select(rows, cfg):
    result = {}
    for method in METHODS:
        valid = [r for r in rows if r["method"] == method and r["valid"]]
        if not valid:
            raise ValueError(f"missing final candidates: {method}")
        pool = [r for r in valid if r["resource_feasible"]]
        if not pool:
            chosen = min(valid, key=lambda r: r["seconds_per_arrival"])
            reason = "resource-infeasible diagnostic"
        else:
            band = [r for r in pool if r["in_band"]]
            if band:
                chosen = min(band, key=acquisition_order)
                dominant = [r for r in pool if r["base_harm"] < chosen["base_harm"]
                            and r["novel_auc"] < chosen["novel_auc"]]
                if dominant:
                    chosen = min(dominant, key=acquisition_order)
                    reason = "below-band joint-improvement candidate"
                else:
                    reason = "development retention-matched"
            else:
                chosen = min(pool, key=lambda r: proximity_order(r, cfg))
                reason = "retention-unmatched diagnostic"
        result[method] = dict(selected=chosen, reason=reason)
    return result


def refinement_proposals(rows, cfg, additions):
    """One next local step per slice; records, not an assumed monotone curve."""
    used = {row["recipe_key"] for row in rows}
    middle = cfg.retention_margin_degrees - cfg.matching_tolerance / 2
    groups = defaultdict(list)
    for row in rows:
        if row["valid"]:
            r = row["recipe"]
            groups[(r["method"], r["replay_count"], r["strength"])].append(row)
    slices = defaultdict(list)
    for group, points in groups.items():
        slices[group[:2]].append((group, min(points, key=lambda r: proximity_order(r, cfg))))
    chosen_slices = set()
    for alternatives in slices.values():
        for group, _ in sorted(alternatives, key=lambda item: proximity_order(item[1], cfg))[:2]:
            chosen_slices.add(group)
    proposals = []
    for group in sorted(chosen_slices):
        points = sorted(groups[group], key=lambda r: r["recipe"]["ridge"])
        near = min(points, key=lambda r: proximity_order(r, cfg))
        history = [a for a in additions if a["recipe"]["method"] == group[0]
                   and a["recipe"]["replay_count"] == group[1]
                   and a["recipe"]["strength"] == group[2]]
        candidate, reason = None, None
        brackets = [(a,b) for a,b in zip(points,points[1:])
                    if (a["base_harm"]-middle)*(b["base_harm"]-middle) < 0]
        if brackets and sum(a["kind"] == "midpoint" for a in history) < 2:
            a,b = min(brackets, key=lambda p: (abs(math.log(p[1]["recipe"]["ridge"] / p[0]["recipe"]["ridge"])),
                                             min(p[0]["novel_auc"],p[1]["novel_auc"])))
            candidate = {**near["recipe"], "ridge": math.sqrt(a["recipe"]["ridge"]*b["recipe"]["ridge"])}
            reason = "midpoint"
        elif not any(p["in_band"] for p in points):
            # At most two outward ridge extensions per method/allocation.
            allocation_history = [a for a in additions if a["recipe"]["method"]==group[0]
                and a["recipe"]["replay_count"]==group[1] and a["kind"]=="edge"]
            if len(allocation_history)<2:
                edge = points[-1] if near["recipe"]["ridge"] == points[-1]["recipe"]["ridge"] else points[0]
                direction = 3. if edge is points[-1] else 1/3
                candidate = {**edge["recipe"], "ridge": edge["recipe"]["ridge"]*direction}
                reason = "edge"
        if candidate:
            recipe = Recipe.from_mapping(candidate)
            if recipe.key not in used:
                proposals.append(dict(recipe=recipe.mapping(), recipe_key=recipe.key,
                                      kind=reason, priority=proximity_order(near,cfg)))
        if group[0] == "fd_replay_ewc" and near["in_band"]:
            strengths = sorted({g[2] for g in groups if g[:2]==group[:2]})
            initial = sorted(cfg.historical_strength*r for r in cfg.strength_ratios)
            if (group[2] in (min(initial),max(initial)) and len(strengths)==len(initial)):
                strength=group[2]/10 if group[2]==min(initial) else group[2]*10
                recipe=Recipe.from_mapping({**near["recipe"],"strength":strength})
                if recipe.key not in used:
                    proposals.append(dict(recipe=recipe.mapping(),recipe_key=recipe.key,
                        kind="strength-edge",priority=(cfg.matching_tolerance,*acquisition_order(near))))
    counts = {method:sum(a["recipe"]["method"]==method for a in additions) for method in METHODS}
    proposals.sort(key=lambda p:(counts[p["recipe"]["method"]]>=cfg.refinement_settings//4,
                                 counts[p["recipe"]["method"]],p["priority"],p["recipe_key"]))
    return proposals
