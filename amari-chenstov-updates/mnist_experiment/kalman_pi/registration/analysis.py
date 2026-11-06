"""Bounded scalar-only Phase 6 summaries, also used for development selection.

Risk is mean squared *landmark distance* (sum of six squared coordinates / 3),
not coordinate MSE or a regularized objective. Arrivals count reference/query
pairs. Roots and paired differences are taken within replicas before averaging.
No execution, model, configuration, data, or torch imports belong here.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

SCHEMA = 1
KIND = "plan17-registration-analysis"
DEFAULT_SEED = 170617
MAX_BOOTSTRAP_DRAWS = 20_000
MAX_TRAJECTORIES = 512
MAX_METRICS = 4096
ARMS = ("fixed", "covariance", "ema", "kalman")
MATCHED_ARMS = ("kalman_matched", "fixed_matched")
RIDGES = (0., .01, .03, .1, .3, 1.)
SEGMENTS = ("initial_hold", "rise", "high_hold", "return", "final_hold")


def _number(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(f"{name} outside [{lower}, {upper}]")
    return float(value)


def _integer(value, name, lower=0, upper=1_000_000):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def landmark_metrics(predictions, targets, weights=None):
    """Score stored six-coordinate predictions; optional observation weights.

    The one-pixel threshold is inclusive, applied to each of three Euclidean
    landmark distances. Gaussian NLL uses the fixed working covariance 4 I_6.
    """
    predicted, target = np.asarray(predictions, dtype=float), np.asarray(targets, dtype=float)
    if predicted.ndim != 2 or predicted.shape[1:] != (6,) or target.shape != predicted.shape or not len(target):
        raise ValueError("predictions and targets must have matching nonempty (N,6) shapes")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("landmark coordinates must be finite")
    squared = np.square(predicted - target).reshape(-1, 3, 2).sum(axis=2)
    weight = np.ones(len(target)) if weights is None else np.asarray(weights, dtype=float)
    if weight.shape != (len(target),) or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("weights must be finite nonnegative observation weights with positive sum")
    risk = float(np.average(squared.mean(axis=1), weights=weight))
    return {"risk": risk, "rms": math.sqrt(risk),
            "accuracy": float(np.average((squared <= 1).mean(axis=1), weights=weight)),
            "nll": float(np.average(squared.sum(axis=1) / 8, weights=weight) + 3 * math.log(8 * math.pi))}


def _boundaries(hold_steps, ramp_steps):
    h = _integer(hold_steps, "hold_steps", 1)
    r = _integer(ramp_steps, "ramp_steps", 1)
    return (0, h, h + r, 2 * h + r, 2 * h + 2 * r, 3 * h + 2 * r)


def _rows(metrics):
    if not isinstance(metrics, list) or len(metrics) > MAX_METRICS:
        raise ValueError(f"metrics must be a list of at most {MAX_METRICS} checkpoints")
    previous_step = previous_arrivals = -1
    for row in metrics:
        if not isinstance(row, dict):
            raise ValueError("metric checkpoint must be an object")
        step = _integer(row.get("step"), "step")
        arrivals = _number(row.get("arrivals"), "arrivals", 0)
        if step <= previous_step or arrivals <= previous_arrivals:
            raise ValueError("checkpoint steps and arrivals must be strictly increasing")
        if previous_step == -1 and (step != 0 or arrivals != 0):
            raise ValueError("metrics must retain the shared initial checkpoint at zero")
        previous_step, previous_arrivals = step, arrivals
        if not isinstance(row.get("segment"), str) or not row["segment"]:
            raise ValueError("checkpoint requires a segment")
        for name in ("old_risk", "shifted_risk"):
            _number(row.get(name), name, 0)
        for name in ("old_accuracy", "shifted_accuracy", "raw_pi"):
            if row.get(name) is not None:
                _number(row[name], name, 0, 1)
        if row.get("applied_pi") is not None:
            _number(row["applied_pi"], "applied_pi", .0001, .2)
        for name in ("motion_energy", "uncertainty_energy", "noise_scale", "innovation_ratio"):
            if row.get(name) is not None:
                _number(row[name], name, 0)
        for name in ("old_nll", "shifted_nll"):
            if row.get(name) is not None:
                _number(row[name], name)


def _area(rows, metric, start=None, end=None):
    selected = [r for r in rows if (start is None or r["step"] >= start) and (end is None or r["step"] <= end)]
    if len(selected) < 2 or any(r.get(metric) is None for r in selected):
        return None
    x = np.asarray([r["arrivals"] for r in selected], dtype=float)
    y = np.asarray([r[metric] for r in selected], dtype=float)
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2) / (x[-1] - x[0]))


def summarize_path(metrics, hold_steps, ramp_steps):
    """Summarize one complete path on its actual-arrival clock.

    Horizons are resolved integer batch counts, not multiplied by a length
    factor here. Every exact segment boundary must have been evaluated. The
    stationary acquisition window uses the same preassigned timestamps.
    ``normalized_score`` falls back to .5*(Aold+Ashifted) if either frozen
    denominator is zero; ``score_basis`` records that absolute-risk ranking.
    """
    _rows(metrics)
    boundaries = _boundaries(hold_steps, ramp_steps)
    if not metrics or metrics[-1]["step"] != boundaries[-1]:
        raise ValueError("complete path must reach its resolved final step")
    if not set(boundaries) <= {r["step"] for r in metrics}:
        raise ValueError("complete path requires exact saved regime boundaries")
    old = _area(metrics, "old_risk")
    shifted = _area(metrics, "shifted_risk", boundaries[1], boundaries[3])
    old0, shifted0 = (float(metrics[0][name]) for name in ("old_risk", "shifted_risk"))
    normalized = old0 > 0 and shifted0 > 0
    regimes = {name: {domain: _area(metrics, domain + "_risk", start, end) for domain in ("old", "shifted")}
               for name, start, end in zip(SEGMENTS, boundaries, boundaries[1:])}
    last = metrics[-1]
    return {"Aold": old, "Ashifted": shifted, "Rold0": old0, "Rshifted0": shifted0,
            "old_rms": math.sqrt(old), "shifted_rms": math.sqrt(shifted),
            "oldgap_frominitial": math.sqrt(old) - math.sqrt(old0),
            "normalized_score": .5 * (old / old0 + shifted / shifted0) if normalized else .5 * (old + shifted),
            "score_basis": "normalized_risk" if normalized else "absolute_risk_zero_initial",
            "Ashifted_whole": _area(metrics, "shifted_risk"), "regimes": regimes,
            "old_accuracy": _area(metrics, "old_accuracy"),
            "shifted_accuracy": _area(metrics, "shifted_accuracy", boundaries[1], boundaries[3]),
            "old_nll": _area(metrics, "old_nll"),
            "shifted_nll": _area(metrics, "shifted_nll", boundaries[1], boundaries[3]),
            "endpoint_old_risk": last["old_risk"], "endpoint_shifted_risk": last["shifted_risk"],
            "endpoint_oldgap_frominitial": math.sqrt(last["old_risk"]) - math.sqrt(old0),
            "post_return_shifted_risk": regimes["final_hold"]["shifted"],
            "final_step": boundaries[-1], "arrivals": last["arrivals"],
            "acquisition_steps": [boundaries[1], boundaries[3]]}


def _selection(selection):
    if selection is None or selection == {}:
        return None
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object or null")
    common = _number(selection.get("common_ridge"), "common_ridge", 0)
    calibrated = selection.get("calibrated_ridge")
    if calibrated is not None:
        _number(calibrated, "calibrated_ridge", 0)
    for name in ("ema_condition", "kalman_condition"):
        if not isinstance(selection.get(name), str) or not selection[name]:
            raise ValueError(f"selection requires {name}")
    for name, cap in (("main_replicas", 8), ("stationary_replicas", 4)):
        _integer(selection.get(name), name, 0, cap)
    arms = selection.get("arms")
    if not isinstance(arms, list) or not all(isinstance(a, dict) for a in arms):
        raise ValueError("selection.arms must be a list of arm/condition/ridge objects")
    by_arm = {a.get("arm"): a for a in arms}
    expected = set(ARMS) | (set(MATCHED_ARMS) if calibrated is not None and calibrated != common else set())
    if set(by_arm) != expected or len(by_arm) != len(arms):
        raise ValueError("selection must retain four common-ridge arms and both required ridge companions")
    for arm, spec in by_arm.items():
        family = arm.removesuffix("_matched")
        condition = selection.get(family + "_condition", family)
        ridge = calibrated if arm in MATCHED_ARMS else common
        if spec.get("condition") != condition or _number(spec.get("ridge"), "arm ridge", 0) != ridge:
            raise ValueError(f"selected physical condition/ridge mismatch for {arm}")
    return by_arm


def _horizon(t, selection, config):
    resolved = []
    for name in ("hold_steps", "ramp_steps"):
        if name in t:
            value = t[name]
        else:
            base = (selection or {}).get(name, config.get(name))
            if base is None:
                raise ValueError(f"trajectory requires resolved {name} or selection/config default")
            value = round(_number(base, name, 1) * t["schedule_factor"])
        resolved.append(_integer(value, name, 1))
    return tuple(resolved)


def validate_trajectories(trajectories, selection=None, config=None):
    """Validate grids, initial measurements, packages, prefixes and frozen blocks."""
    specs = _selection(selection)
    config = config or {}
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    if not isinstance(trajectories, list) or len(trajectories) > MAX_TRAJECTORIES:
        raise ValueError(f"trajectories must be a list of at most {MAX_TRAJECTORIES} paths")
    units, identities, blocks = set(), set(), defaultdict(list)
    for t in trajectories:
        if not isinstance(t, dict):
            raise ValueError("trajectory must be an object")
        for name in ("unit_id", "cohort", "arm", "condition"):
            if not isinstance(t.get(name), str) or not t[name]:
                raise ValueError(f"trajectory requires {name}")
        if t["unit_id"] in units:
            raise ValueError("duplicate unit_id")
        units.add(t["unit_id"])
        if t.get("stage") not in ("development", "fresh") or t.get("schedule") not in ("main", "stationary"):
            raise ValueError("trajectory stage/schedule is incompatible")
        replica = _integer(t.get("replica"), "replica")
        if type(t.get("completed")) is not bool:
            raise ValueError("completed must be boolean")
        if type(t.get("sign")) is not int or t["sign"] not in (-1, 1):
            raise ValueError("sign must be -1 or +1")
        if _number(t.get("schedule_factor"), "schedule_factor") not in (1., 1.25):
            raise ValueError("schedule_factor must be 1 or 1.25")
        ridge = _number(t.get("ridge"), "ridge", 0)
        if _number(t.get("c"), "c") != .025:
            raise ValueError("fixed reference c must remain .025")
        if t["stage"] == "fresh":
            if specs is None:
                raise ValueError("fresh trajectories require frozen selection")
            spec = specs.get(t["arm"])
            if spec is None or spec["condition"] != t["condition"] or spec["ridge"] != ridge:
                raise ValueError("fresh trajectory differs from its selected arm")
            if replica >= selection[t["schedule"] + "_replicas"]:
                raise ValueError("replica is outside the frozen allocation")
        elif t["arm"] not in (*ARMS, *MATCHED_ARMS):
            raise ValueError("unrecognized development arm")
        key = (t["stage"], t["cohort"], t["schedule"], replica)
        identity = (*key, t["arm"], t["condition"], ridge) if t["stage"] == "development" else (*key, t["arm"])
        if identity in identities:
            raise ValueError("duplicate trajectory block/arm identity")
        identities.add(identity)
        _rows(t.get("metrics"))
        h, r = _horizon(t, selection, config)
        final_step = _boundaries(h, r)[-1]
        if t.get("final_step", final_step) != final_step:
            raise ValueError("final_step disagrees with resolved horizon")
        if t["metrics"] and t["metrics"][-1]["step"] > final_step:
            raise ValueError("partial trajectory exceeds resolved horizon")
        if t["completed"]:
            summarize_path(t["metrics"], h, r)
        summary = t.get("summary", {})
        if not isinstance(summary, dict) or not isinstance(summary.get("costs", {}), dict):
            raise ValueError("summary and costs must be objects")
        for name, value in summary.get("costs", {}).items():
            if name.endswith("_seconds") and value is not None:
                _number(value, name, 0)
        for name in ("learning_seconds", "online_wall_seconds"):
            if summary.get(name) is not None:
                _number(summary[name], name, 0)
        blocks[key].append(t)
    for block in blocks.values():
        first = block[0]
        for t in block[1:]:
            for name in ("sign", "schedule_factor", "stream_hash", "initial_unit", "asset_unit", "pretrain_unit"):
                if t.get(name) != first.get(name):
                    raise ValueError(f"paired block has mismatched {name}")
            if _horizon(t, selection, config) != _horizon(first, selection, config):
                raise ValueError("paired block has mismatched resolved horizon")
            a, b = t["metrics"], first["metrics"]
            count = min(len(a), len(b))
            if t["completed"] and first["completed"] and len(a) != len(b):
                raise ValueError("paired completed grids differ")
            if [(x["step"], x["arrivals"], x["segment"]) for x in a[:count]] != [(x["step"], x["arrivals"], x["segment"]) for x in b[:count]]:
                raise ValueError("paired checkpoint grids/prefixes differ")
            if count:
                for name in ("old_risk", "shifted_risk", "old_accuracy", "shifted_accuracy", "old_nll", "shifted_nll"):
                    if a[0].get(name) != b[0].get(name):
                        raise ValueError(f"paired initialization has mismatched {name}")


def _costs(t):
    summary = t.get("summary", {})
    costs = summary.get("costs", {})
    parts = [costs.get(name) for name in ("solver_seconds", "curvature_seconds", "controller_seconds", "render_seconds")]
    learning = summary.get("learning_seconds")
    if learning is None and all(p is not None for p in parts):
        learning = float(sum(parts))
    return {"learning_seconds": learning,
            "online_wall_seconds": summary.get("online_wall_seconds")}


def _indices(blocks, draws, seed):
    n = len(blocks)
    strata = defaultdict(list)
    for i, block in enumerate(blocks):
        t = next(iter(block.values()))
        strata[(t["sign"], t["schedule_factor"])].append(i)
    stratified = bool(strata) and all(len(s) >= 2 for s in strata.values())
    rng = np.random.default_rng(seed)
    if not n or not draws:
        indices = np.empty((0, n), dtype=int)
    elif stratified:
        indices = np.concatenate([rng.choice(s, size=(draws, len(s)), replace=True) for _, s in sorted(strata.items())], axis=1)
    else:
        indices = rng.integers(n, size=(draws, n))
    return indices, "stratified_sign_factor" if stratified else "ordinary_paired_replica"


def _estimate(values, indices):
    present = all(v is not None for v in values)
    n = len(values)
    mean = float(np.mean(values)) if n and present else None
    reliable = bool(n >= 2 and len(indices) and present)
    ci = np.quantile(np.asarray(values)[indices].mean(axis=1), [.025, .975]).tolist() if reliable else None
    return {"mean": mean, "ci95": ci, "n": n if present else sum(v is not None for v in values),
            "interval_reliable": reliable}


def _contrast(blocks, paths, arm, control, indices):
    values = defaultdict(list)
    for block in blocks:
        a, b = (paths[block[name]["unit_id"]] for name in (arm, control))
        for domain, metric in (("old", "Aold"), ("shifted", "Ashifted")):
            values[domain + "_risk_gain"].append(b[metric] - a[metric])
            values[domain + "_rms_gap"].append(a[domain + "_rms"] - b[domain + "_rms"])
            av, bv = a[domain + "_accuracy"], b[domain + "_accuracy"]
            values[domain + "_accuracy_gain_pp"].append(100 * (av - bv) if av is not None and bv is not None else None)
        for name in ("Ashifted_whole", "endpoint_old_risk", "endpoint_shifted_risk", "post_return_shifted_risk"):
            values[name + "_gain"].append(b[name] - a[name])
        for regime in SEGMENTS:
            for domain in ("old", "shifted"):
                values[regime + "_" + domain + "_risk_gain"].append(b["regimes"][regime][domain] - a["regimes"][regime][domain])
        for name in ("learning_seconds", "online_wall_seconds"):
            av, bv = (_costs(block[x])[name] for x in (arm, control))
            values[name + "_ratio"].append(av / bv if av is not None and bv is not None and bv > 0 else None)
    names = ("old_risk_gain", "shifted_risk_gain", "old_rms_gap", "shifted_rms_gap",
             "old_accuracy_gain_pp", "shifted_accuracy_gain_pp", "learning_seconds_ratio", "online_wall_seconds_ratio")
    return {"arm": arm, "control": control, "n_pairs": len(blocks),
            "effects": {name: _estimate(values[name], indices) for name in (*names, *sorted(set(values) - set(names)))}}


def summarize(trajectories, selection=None, bootstrap_draws=2000, seed=DEFAULT_SEED, config=None):
    """Return complete-block paired effects, never checkpoint pseudo-replicates.

    All four common-ridge arms (and both companions when selected) must finish
    before any fresh block contributes. Bootstrap draws are shared across every
    endpoint and contrast within a cohort/schedule, with deterministic unit
    ordering. The persisted seed and bounded draw count are returned verbatim.
    """
    if type(bootstrap_draws) is not int:
        raise ValueError("bootstrap_draws must be an integer")
    _integer(seed, "seed", 0, 2**63 - 1)
    draws = max(0, min(bootstrap_draws, MAX_BOOTSTRAP_DRAWS))
    validate_trajectories(trajectories, selection, config)
    config = config or {}
    specs = _selection(selection)
    required = list(specs) if specs else list(ARMS)
    paths, groups, complete = {}, defaultdict(lambda: defaultdict(dict)), []
    for t in sorted(trajectories, key=lambda t: t["unit_id"]):
        if not t["completed"]:
            continue
        complete.append(t["unit_id"])
        paths[t["unit_id"]] = {**summarize_path(t["metrics"], *_horizon(t, selection, config)),
                               **{name: t[name] for name in ("unit_id", "stage", "cohort", "replica", "schedule", "arm", "condition", "ridge", "sign", "schedule_factor")},
                               "costs": _costs(t)}
        if t["stage"] == "fresh":
            groups[(t["cohort"], t["schedule"])][t["replica"]][t["arm"]] = t
    results, eligible_units = [], set()
    for (cohort, schedule), replicas in sorted(groups.items()):
        ids = sorted(r for r, block in replicas.items() if set(required) <= block.keys())
        blocks = [replicas[r] for r in ids]
        eligible_units.update(t["unit_id"] for block in blocks for t in block.values())
        group_seed = int(_hash([seed, cohort, schedule])[:16], 16)
        indices, mode = _indices(blocks, draws, group_seed)
        contrasts = [(a, "fixed") for a in ARMS if a != "fixed"]
        if specs and "kalman_matched" in specs:
            contrasts += [("kalman_matched", "fixed"), ("kalman_matched", "fixed_matched"), ("fixed_matched", "fixed")]
        results.append({"cohort": cohort, "schedule": schedule, "eligible_replicas": ids,
                        "n_blocks": len(blocks), "bootstrap_mode": mode,
                        "bootstrap_indices_sha256": _hash(indices.tolist()),
                        "uncertainty_note": "Sign/length strata preserved." if mode.startswith("stratified") else
                        "Ordinary paired-replica bootstrap; between-stratum variation enters uncertainty.",
                        "comparisons": [_contrast(blocks, paths, a, b, indices) for a, b in contrasts]})
    return {"schema": SCHEMA, "kind": KIND, "selection": deepcopy(selection),
            "trajectories_sha256": _hash(trajectories), "required_arms": required,
            "bootstrap": {"seed": seed, "draws": draws, "unit": "whole_replica", "shared_across_endpoints_and_contrasts": True},
            "averaging": "normalized trapezoid on actual pair arrivals; equal weight per whole replica",
            "risk_units": "squared pixels per landmark", "rms_units": "pixels",
            "path_summaries": paths, "groups": results,
            "counts": {"complete": len(complete), "incomplete": len(trajectories) - len(complete),
                       "paired_blocks": sum(g["n_blocks"] for g in results)},
            "complete_units": complete,
            "incomplete_units": sorted(t["unit_id"] for t in trajectories if not t["completed"]),
            "awaiting_full_block_units": sorted(t["unit_id"] for t in trajectories if t["stage"] == "fresh" and t["completed"] and t["unit_id"] not in eligible_units)}
