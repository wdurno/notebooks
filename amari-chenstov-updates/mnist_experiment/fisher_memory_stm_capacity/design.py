"""Bounded development policies with one shared historical reference."""

from collections import defaultdict
import math

import numpy as np

from mnist_experiment.fisher_memory_regression.analysis import replica_draw_indices
from mnist_experiment.fisher_memory_regression.config import Recipe
from .config import FD, REPLAY, REFERENCE, arm


def upper(values, cfg):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite paired development replicas required")
    seed = cfg.seed_for("development--grid", 0, "selection_bootstrap")
    draws = [float(values[replica_draw_indices(len(values), seed, i)].mean())
             for i in range(cfg.bootstrap_samples)]
    return float(np.quantile(draws, .95))


def table(records, cfg, replicas=None):
    replicas = list(range(cfg.calibration_replicas)) if replicas is None else list(replicas)
    groups = defaultdict(list)
    for row in records:
        if row["partition"] == "development" and row["replica"] in replicas:
            groups[(row["stm_config"], row["recipe_key"])].append(row)
    rows = []
    for (chart, key), group in sorted(groups.items()):
        group.sort(key=lambda r: r["replica"])
        if [r["replica"] for r in group] != replicas:
            continue
        harm = [math.sqrt(r["base_auc"]) - math.sqrt(r["frozen_base_mse"]) for r in group]
        rows.append(dict(stm_config=chart, arm=arm(chart, group[0]["method"]),
            method=group[0]["method"], recipe_key=key, recipe=group[0]["recipe"],
            replicas=replicas, count=len(replicas), valid=all(r["valid"] for r in group),
            novel_auc=float(np.mean([r["novel_auc"] for r in group])), H=float(np.mean(harm)),
            per_replica_H=harm, retention_upper=upper(harm, cfg),
            seconds_per_arrival=sum(r["learning_seconds"] for r in group)/sum(r["observations"] for r in group),
            mean_learning_seconds=float(np.mean([r["learning_seconds"] for r in group]))))
    return rows


def order(row):
    recipe = row["recipe"]
    return (row["novel_auc"], row["seconds_per_arrival"], recipe["ridge"],
            recipe["strength"], row["recipe_key"])


def select(rows, cfg):
    reference = next(r for r in rows if r["arm"] == REFERENCE)
    result = {REFERENCE: {"selected": reference, "eligible": True, "failures": []}}

    def choose(chart, method, replay=None):
        candidates = [r for r in rows if r["stm_config"] == chart and r["method"] == method and r["valid"]]
        if not candidates:
            raise ValueError(f"no numerically valid candidates: {chart}/{method}")
        def failures(row):
            failed = []
            if row["retention_upper"] > cfg.retention_margin:
                failed.append("retention_ceiling")
            if method == FD:
                for label, control in (("common_reference", reference), ("same_chart", replay)):
                    paired = np.asarray(row["per_replica_H"]) - control["per_replica_H"]
                    if upper(paired, cfg) > cfg.matching_tolerance:
                        failed.append("retention_" + label)
                if row["seconds_per_arrival"] > cfg.envelope_factor*reference["seconds_per_arrival"]:
                    failed.append("relative_learning_cost")
            return failed
        feasible = [r for r in candidates if not failures(r)]
        if feasible:
            chosen = min(feasible, key=order)
        else:
            def violation(row):
                excess = max(0.,row["retention_upper"]-cfg.retention_margin)
                if method == FD:
                    for control in (reference,replay):
                        difference = np.asarray(row["per_replica_H"])-control["per_replica_H"]
                        excess = max(excess,upper(difference,cfg)-cfg.matching_tolerance)
                return excess
            chosen = min(candidates,key=lambda r:(violation(r),*order(r)))
        failed = failures(chosen)
        return dict(selected=chosen, eligible=not failed, failures=failed)

    for chart in cfg.charts:
        if chart != "all_r2":
            result[arm(chart, REPLAY)] = choose(chart, REPLAY)
        if chart == "all_r2":
            incumbent = next(r for r in rows if r["arm"] == arm(chart, FD))
            result[arm(chart, FD)] = {"selected": incumbent, "eligible": incumbent["valid"],
                                    "failures": [], "frozen_historical_policy": True}
        else:
            result[arm(chart, FD)] = choose(chart, FD, result[arm(chart, REPLAY)]["selected"])
    return result


def refinement(rows, selected, chart, used):
    """One bounded adjacent strength/ridge proposal from the observed frontier."""
    best = selected[arm(chart, FD)]["selected"]
    r = best["recipe"]
    siblings = [x for x in rows if x["stm_config"] == chart and x["method"] == FD
                and x["recipe"]["ridge"] == r["ridge"]]
    strengths = sorted(x["recipe"]["strength"] for x in siblings)
    choices = []
    if r["strength"] == strengths[0]:
        choices.append({**r, "strength": r["strength"]*.1})
    if r["strength"] == strengths[-1]:
        choices.append({**r, "strength": r["strength"]*10})
    choices += [{**r, "ridge": math.sqrt(r["ridge"]*math.sqrt(10))},
                {**r, "ridge": .1 if r["ridge"] == 1 else math.sqrt(r["ridge"])}]
    return next((p for p in choices if Recipe.from_mapping(p).key not in used), None)

