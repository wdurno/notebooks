"""Scalar-only Phase 7 tracking summaries and whole-replica paired inference.

Risk is mean squared Euclidean landmark distance in physical pixels. Integrate
squared risk on saved incoming-pair counts and take roots within each replica.
No training, model, dataset, or torch imports belong in this module.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

import numpy as np

from mnist_experiment.kalman_pi.registration.analysis import (
    MAX_BOOTSTRAP_DRAWS, MAX_METRICS, MAX_TRAJECTORIES,
    _area, _costs, _estimate, _hash, _indices, _integer, _number,
    landmark_metrics,
)

SCHEMA = 1
KIND = "plan17-discrepancy-analysis"
DEFAULT_SEED = 170717
ARMS = ("fixed", "covariance", "velocity", "discrepancy")
SEGMENTS = ("initial_hold", "rise", "return", "final_hold")
DOMAINS = ("current", "old", "shifted", "shifted25", "shifted45")
CONTRASTS = (("covariance", "fixed"), ("velocity", "fixed"),
             ("discrepancy", "fixed"), ("discrepancy", "velocity"),
             ("discrepancy", "covariance"))


def _boundaries(hold_steps, ramp_steps):
    h, r = (_integer(v, name, 1) for v, name in
            ((hold_steps, "hold_steps"), (ramp_steps, "ramp_steps")))
    return (0, h, h + r, h + 2 * r, 2 * h + 2 * r)


def _rows(metrics):
    if not isinstance(metrics, list) or len(metrics) > MAX_METRICS:
        raise ValueError(f"metrics must be a list of at most {MAX_METRICS} checkpoints")
    previous = (-1, -1)
    for row in metrics:
        if not isinstance(row, dict):
            raise ValueError("metric checkpoint must be an object")
        step = _integer(row.get("step"), "step")
        arrivals = _number(row.get("arrivals"), "arrivals", 0)
        if step <= previous[0] or arrivals <= previous[1]:
            raise ValueError("checkpoint steps and arrivals must be strictly increasing")
        if previous[0] == -1 and (step != 0 or arrivals != 0):
            raise ValueError("metrics require the initial checkpoint at zero")
        previous = step, arrivals
        if not isinstance(row.get("segment"), str) or not row["segment"]:
            raise ValueError("checkpoint requires a segment")
        for domain in ("current", "old", "shifted"):
            _number(row.get(domain + "_risk"), domain + "_risk", 0)
        for domain in DOMAINS:
            for prefix in ("", "frozen_"):
                risk = prefix + domain + "_risk"
                if row.get(risk) is not None:
                    _number(row[risk], risk, 0)
                accuracy = prefix + domain + "_accuracy"
                if row.get(accuracy) is not None:
                    _number(row[accuracy], accuracy, 0, 1)
                nll = prefix + domain + "_nll"
                if row.get(nll) is not None:
                    _number(row[nll], nll)
        for name in ("raw_pi", "applied_pi"):
            if row.get(name) is not None:
                _number(row[name], name, 0, 1)
        if row.get("applied_pi") is not None and not .0001 <= row["applied_pi"] <= .2:
            raise ValueError("applied_pi is outside the approved action limits")
        for name in ("motion_energy", "uncertainty_energy", "discrepancy_energy",
                     "level_uncertainty_energy", "noise_scale", "innovation_ratio"):
            if row.get(name) is not None:
                _number(row[name], name, 0)


def summarize_path(metrics, hold_steps, ramp_steps):
    """Return Acurrent (selection target), Ecurrent/current_rms, and all panels.

    Complete paths must save every regime boundary. Endpoint panels are averaged
    over the whole path, just like the primary current-law panel. Frozen-current
    risk is integrated as a changing curve, never an initialization constant.
    """
    _rows(metrics)
    boundaries = _boundaries(hold_steps, ramp_steps)
    if not metrics or metrics[-1]["step"] != boundaries[-1]:
        raise ValueError("complete path must reach its resolved final step")
    if not set(boundaries) <= {r["step"] for r in metrics}:
        raise ValueError("complete path requires exact saved regime boundaries")
    result = {"final_step": boundaries[-1], "arrivals": metrics[-1]["arrivals"],
              "ramp_steps": [boundaries[1], boundaries[3]], "regimes": {}}
    for domain in DOMAINS:
        area = _area(metrics, domain + "_risk")
        if area is None:
            continue
        rms = math.sqrt(area)
        result.update({"A" + domain: area, "E" + domain: rms,
                       domain + "_rms": rms,
                       domain + "_accuracy": _area(metrics, domain + "_accuracy"),
                       domain + "_nll": _area(metrics, domain + "_nll"),
                       "endpoint_" + domain + "_risk": metrics[-1][domain + "_risk"],
                       "endpoint_" + domain + "gap_frominitial":
                           math.sqrt(metrics[-1][domain + "_risk"]) - math.sqrt(metrics[0][domain + "_risk"])})
        frozen = _area(metrics, "frozen_" + domain + "_risk")
        result["Afrozen_" + domain] = frozen
        result["frozen_" + domain + "_rms"] = math.sqrt(frozen) if frozen is not None else None
        for name, start, end in zip(SEGMENTS, boundaries, boundaries[1:]):
            segment = result["regimes"].setdefault(name, {})
            segment[domain] = _area(metrics, domain + "_risk", start, end)
            segment[domain + "_accuracy"] = _area(metrics, domain + "_accuracy", start, end)
    result["oldgap_frominitial"] = result["old_rms"] - math.sqrt(metrics[0]["old_risk"])
    # Compatibility with the scalar selector; no endpoint normalization enters it.
    result["normalized_score"] = result["Acurrent"]
    result["score_basis"] = "current_law_budget_squared_error"
    return result


def _selection(selection):
    if selection is None or selection == {}:
        return None
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object or null")
    ridge = _number(selection.get("common_ridge"), "common_ridge", 0)
    for name in ("velocity_condition", "discrepancy_condition"):
        if not isinstance(selection.get(name), str) or not selection[name]:
            raise ValueError(f"selection requires {name}")
    for name, cap in (("main_replicas", 8), ("stationary_replicas", 4)):
        _integer(selection.get(name), name, 0, cap)
    arms = selection.get("arms")
    if not isinstance(arms, list) or not all(isinstance(a, dict) for a in arms):
        raise ValueError("selection.arms must list arm/condition/ridge objects")
    specs = {a.get("arm"): a for a in arms}
    if set(specs) != set(ARMS) or len(specs) != len(arms):
        raise ValueError("selection requires all four common-ridge policies exactly once")
    for arm, spec in specs.items():
        if (spec.get("condition") != selection.get(arm + "_condition", arm)
                or _number(spec.get("ridge"), "arm ridge", 0) != ridge):
            raise ValueError("selected condition/ridge differs from its policy package")
    return specs


def _horizon(t, selection=None, config=None):
    values = []
    for name in ("hold_steps", "ramp_steps"):
        value = t.get(name)
        if value is None:
            base = (selection or {}).get(name, (config or {}).get(name))
            if base is None:
                raise ValueError(f"trajectory requires resolved {name} or a configuration default")
            value = math.ceil(_number(base, name, 1) * t["schedule_factor"])
        values.append(_integer(value, name, 1))
    return tuple(values)


def _block_key(t):
    return t["stage"], t["cohort"], t["schedule"], t["angle_max"], t["replica"]


def validate_trajectories(trajectories, selection=None, config=None):
    specs = _selection(selection)
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be an object")
    if not isinstance(trajectories, list) or len(trajectories) > MAX_TRAJECTORIES:
        raise ValueError(f"trajectories must be a list of at most {MAX_TRAJECTORIES} paths")
    units, identities, blocks = set(), set(), defaultdict(list)
    for t in trajectories:
        if not isinstance(t, dict):
            raise ValueError("trajectory must be an object")
        for name in ("unit_id", "cohort", "condition"):
            if not isinstance(t.get(name), str) or not t[name]:
                raise ValueError(f"trajectory requires {name}")
        if t.get("arm") not in ARMS or t.get("stage") not in ("development", "fresh") or t.get("schedule") not in ("main", "stationary"):
            raise ValueError("incompatible trajectory arm/stage/schedule")
        angle = _number(t.get("angle_max"), "angle_max", 0)
        if angle not in ((0.,) if t["schedule"] == "stationary" else (25., 45.)):
            raise ValueError("changing amplitudes are 25/45; the single stationary cohort uses angle_max=0")
        if t["unit_id"] in units:
            raise ValueError("duplicate unit_id")
        units.add(t["unit_id"])
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
            spec = specs[t["arm"]]
            if spec["condition"] != t["condition"] or spec["ridge"] != ridge:
                raise ValueError("fresh trajectory differs from frozen policy package")
            if replica >= selection[t["schedule"] + "_replicas"]:
                raise ValueError("replica exceeds frozen allocation")
        key = _block_key(t)
        identity = (*key, t["arm"], t["condition"], ridge) if t["stage"] == "development" else (*key, t["arm"])
        if identity in identities:
            raise ValueError("duplicate block/arm identity")
        identities.add(identity)
        _rows(t.get("metrics"))
        h, r = _horizon(t, selection, config)
        final = _boundaries(h, r)[-1]
        if t.get("final_step", final) != final or t["metrics"] and t["metrics"][-1]["step"] > final:
            raise ValueError("trajectory exceeds or disagrees with resolved horizon")
        if t["completed"]:
            summarize_path(t["metrics"], h, r)
        summary = t.get("summary", {})
        if not isinstance(summary, dict) or not isinstance(summary.get("costs", {}), dict):
            raise ValueError("summary and costs must be objects")
        for name, value in {**summary.get("costs", {}), **{k: summary[k] for k in ("learning_seconds", "online_wall_seconds") if k in summary}}.items():
            if name.endswith("_seconds") and value is not None:
                _number(value, name, 0)
        blocks[key].append(t)
    for block in blocks.values():
        first = block[0]
        for t in block[1:]:
            # Development candidates with different physical settings still share data.
            for name in ("sign", "schedule_factor", "stream_hash", "initial_unit", "asset_unit", "pretrain_unit"):
                if first.get(name) != t.get(name):
                    raise ValueError(f"paired block has mismatched {name}")
            if _horizon(t, selection, config) != _horizon(first, selection, config):
                raise ValueError("paired block has mismatched horizon")
            a, b = first["metrics"], t["metrics"]
            if first["completed"] and t["completed"] and len(a) != len(b):
                raise ValueError("paired completed grids differ")
            n = min(len(a), len(b))
            if [(r["step"], r["arrivals"], r["segment"]) for r in a[:n]] != [(r["step"], r["arrivals"], r["segment"]) for r in b[:n]]:
                raise ValueError("paired checkpoint grids/prefixes differ")
            if n:
                for name in (k for k in set(a[0]) | set(b[0]) if k.endswith(("_risk", "_accuracy", "_nll"))):
                    if a[0].get(name) != b[0].get(name):
                        raise ValueError("paired initialization differs")
    # Paired amplitudes may have different stream hashes because their angles differ.
    amplitude_pairs = defaultdict(dict)
    for t in trajectories:
        if t["schedule"] == "main":
            amplitude_pairs[(t["stage"], t["cohort"], t["replica"], t["arm"], t["condition"], t["ridge"])][t["angle_max"]] = t
    for pair in amplitude_pairs.values():
        if not {25., 45.} <= pair.keys():
            continue
        a, b = pair[25.], pair[45.]
        for name in ("sign", "schedule_factor", "initial_unit", "pretrain_unit",
                     "source_hash", "uniforms_hash"):
            if a.get(name) != b.get(name):
                raise ValueError(f"cross-angle pairing has mismatched {name}")
        if _horizon(a, selection, config) != _horizon(b, selection, config):
            raise ValueError("cross-angle pairing has mismatched horizon")
        if a["metrics"] and b["metrics"]:
            for name in ("current_risk", "old_risk", "current_accuracy", "old_accuracy"):
                if a["metrics"][0].get(name) != b["metrics"][0].get(name):
                    raise ValueError("cross-angle initialization differs on the nominal panel")


def _contrast_values(blocks, paths, arm, control):
    values = defaultdict(list)
    for block in blocks:
        a, b = (paths[block[k]["unit_id"]] for k in (arm, control))
        for domain in DOMAINS:
            if "A" + domain not in a or "A" + domain not in b:
                continue
            values[domain + "_risk_gain"].append(b["A" + domain] - a["A" + domain])
            values[domain + "_rms_gap"].append(a[domain + "_rms"] - b[domain + "_rms"])
            av, bv = a[domain + "_accuracy"], b[domain + "_accuracy"]
            values[domain + "_accuracy_gain_pp"].append(100 * (av - bv) if av is not None and bv is not None else None)
            values["endpoint_" + domain + "_risk_gain"].append(b["endpoint_" + domain + "_risk"] - a["endpoint_" + domain + "_risk"])
            for regime in SEGMENTS:
                av, bv = a["regimes"][regime][domain], b["regimes"][regime][domain]
                values[regime + "_" + domain + "_risk_gain"].append(bv - av)
                values[regime + "_" + domain + "_rms_gap"].append(math.sqrt(av) - math.sqrt(bv))
        for name in ("learning_seconds", "online_wall_seconds"):
            av, bv = (_costs(block[k])[name] for k in (arm, control))
            values[name + "_ratio"].append(av / bv if av is not None and bv is not None and bv > 0 else None)
    return values


def _contrast(blocks, paths, arm, control, indices):
    values = _contrast_values(blocks, paths, arm, control)
    for name in ("current_rms_gap", "old_rms_gap", "shifted_rms_gap", "current_risk_gain",
                 "current_accuracy_gain_pp", "learning_seconds_ratio", "online_wall_seconds_ratio"):
        values.setdefault(name, [])
    return {"arm": arm, "control": control, "n_pairs": len(blocks),
            "effects": {name: _estimate(v if len(v) == len(blocks) else [None] * len(blocks), indices)
                        for name, v in sorted(values.items())}}


def summarize(trajectories, selection=None, bootstrap_draws=2000, seed=DEFAULT_SEED, config=None):
    """Summarize complete four-policy blocks, with paired-amplitude interactions.

    The same bootstrap index draws serve all outcomes and contrasts within a
    cohort. Fully paired amplitude groups use the same draws; cross-angle
    differences always resample the shared replica identities together.
    """
    if type(bootstrap_draws) is not int:
        raise ValueError("bootstrap_draws must be an integer")
    _integer(seed, "seed", 0, 2**63 - 1)
    draws = max(0, min(bootstrap_draws, MAX_BOOTSTRAP_DRAWS))
    validate_trajectories(trajectories, selection, config)
    specs = _selection(selection)
    required = list(specs) if specs else list(ARMS)
    paths, groups, complete = {}, defaultdict(lambda: defaultdict(dict)), []
    for t in sorted(trajectories, key=lambda t: t["unit_id"]):
        if not t["completed"]:
            continue
        complete.append(t["unit_id"])
        fields = ("unit_id", "stage", "cohort", "replica", "schedule", "angle_max", "arm", "condition", "ridge", "sign", "schedule_factor")
        paths[t["unit_id"]] = {**summarize_path(t["metrics"], *_horizon(t, selection, config)),
                               **{k: t[k] for k in fields}, "costs": _costs(t)}
        if t["stage"] == "fresh":
            groups[(t["cohort"], t["schedule"], t["angle_max"])][t["replica"]][t["arm"]] = t
    results, eligible, complete_groups = [], set(), {}
    for (cohort, schedule, angle), replicas in sorted(groups.items()):
        ids = sorted(r for r, block in replicas.items() if set(required) <= block.keys())
        blocks = [replicas[r] for r in ids]
        complete_groups[(cohort, schedule, angle)] = dict(zip(ids, blocks))
        eligible.update(t["unit_id"] for block in blocks for t in block.values())
        group_seed = int(_hash([seed, cohort, schedule])[:16], 16)
        indices, mode = _indices(blocks, draws, group_seed)
        results.append({"cohort": cohort, "schedule": schedule, "angle_max": angle,
                        "eligible_replicas": ids, "n_blocks": len(blocks), "bootstrap_mode": mode,
                        "bootstrap_indices_sha256": _hash(indices.tolist()),
                        "uncertainty_note": "Sign/length strata preserved." if mode.startswith("stratified") else
                            "Ordinary paired-replica bootstrap; fewer than two replicas per sign/length stratum.",
                        "comparisons": [_contrast(blocks, paths, a, b, indices) for a, b in CONTRASTS]})
    interactions = []
    for cohort in sorted({key[0] for key in complete_groups if key[1] == "main"}):
        low, high = (complete_groups.get((cohort, "main", angle), {}) for angle in (25., 45.))
        ids = sorted(low.keys() & high.keys())
        if not ids:
            continue
        blocks = [low[r] for r in ids]
        indices, mode = _indices(blocks, draws, int(_hash([seed, cohort, "main"])[:16], 16))
        comparisons = []
        for arm, control in CONTRASTS:
            a, b = (_contrast_values([group[r] for r in ids], paths, arm, control) for group in (low, high))
            differences = {name: [y - x if x is not None and y is not None else None for x, y in zip(a[name], b[name])]
                           for name in a.keys() & b.keys()}
            comparisons.append({"arm": arm, "control": control, "n_pairs": len(ids),
                                "effects": {name: _estimate(v, indices) for name, v in sorted(differences.items())}})
        interactions.append({"cohort": cohort, "eligible_replicas": ids, "n_blocks": len(ids),
                             "direction": "45-degree contrast minus 25-degree contrast",
                             "bootstrap_mode": mode, "bootstrap_indices_sha256": _hash(indices.tolist()),
                             "comparisons": comparisons})
    return {"schema": SCHEMA, "kind": KIND, "selection": deepcopy(selection),
            "trajectories_sha256": _hash(trajectories), "required_arms": required,
            "bootstrap": {"seed": seed, "draws": draws, "unit": "whole_replica",
                          "shared_across_endpoints_and_contrasts": True, "paired_amplitudes_preserved": True},
            "averaging": "normalized trapezoid of squared error on saved incoming-pair counts; root within replica",
            "primary": "current_rms", "risk_units": "squared pixels per landmark", "rms_units": "pixels",
            "path_summaries": paths, "groups": results, "cross_angle": interactions,
            "counts": {"complete": len(complete), "incomplete": len(trajectories) - len(complete),
                       "paired_blocks": sum(g["n_blocks"] for g in results)},
            "complete_units": complete,
            "incomplete_units": sorted(t["unit_id"] for t in trajectories if not t["completed"]),
            "awaiting_full_block_units": sorted(t["unit_id"] for t in trajectories
                if t["stage"] == "fresh" and t["completed"] and t["unit_id"] not in eligible)}
