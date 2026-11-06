"""Bounded Plan 17 analysis of sealed scalar artifacts, never learner state.

Accuracies are fractions; arrivals count incoming digits, not optimizer steps.
Inference uses complete paired trajectories within (stage, schedule, c, ridge).
Full condition IDs are protocol identities, including bounded fallback policies.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np

from mnist_experiment.diagonal_lfu.analysis import (
    ACCURACIES, AVERAGING, MAX_BOOTSTRAP_DRAWS, _estimate, _number,
    _paired_path, _path_summary, _seed, validate_trajectories as _validate_base,
)

SCHEMA = 1
DEFAULT_SEED = 170017
KIND = "plan17-paired-analysis"
POLICY_FLAGS = ("boundary_low", "boundary_high", "cold_start", "measurement_variance_fallback", "anchor_refresh")
POLICY_FRACTIONS = (
    "raw_pi", "applied_pi", "q", "replacement_mass", *POLICY_FLAGS,
    "observer_floor_fraction", "covariance_repair_fraction",
)
POLICY_NONNEGATIVE = (
    "effective_size", "motion_energy", "uncertainty_energy", "noise_scale", "rho",
    "mapping_error_ratio", "innovation_ratio", "mapping_error_energy", "mapping_error_noise",
)


def condition_name(value):
    """Do not alias distinct half-lives, noise scales, or fallback variants."""
    return value


def is_movement(condition):
    return condition.startswith(("ema_h", "kalman_"))


def trajectories_hash(trajectories):
    return hashlib.sha256(json.dumps(trajectories, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _policy_rows(rows):
    for row in rows:
        if row.get("fallback") is not None and not isinstance(row["fallback"], (str, bool)):
            raise ValueError("fallback must be a string reason or boolean")
        for name in POLICY_FRACTIONS:
            if row.get(name) is not None:
                value = row[name]
                if name in POLICY_FLAGS and isinstance(value, bool):
                    continue
                if not 0 <= _number(value, name) <= 1:
                    raise ValueError(f"{name} must be in [0, 1]")
        for name in POLICY_NONNEGATIVE:
            if row.get(name) is not None and _number(row[name], name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if row.get("log_replacement_mass") is not None and _number(row["log_replacement_mass"], "log_replacement_mass") > 0:
            raise ValueError("log_replacement_mass must be nonpositive")
        if row.get("innovation_log_score") is not None:
            _number(row["innovation_log_score"], "innovation_log_score")
        if row.get("step") is not None:
            if type(row["step"]) is not int or row["step"] < 0:
                raise ValueError("step must be a nonnegative integer")


def validate_trajectories(trajectories):
    _validate_base(trajectories)
    for t in trajectories:
        if t["condition"] not in ("fixed", "covariance") and not is_movement(t["condition"]):
            raise ValueError(f"unknown Plan 17 policy: {t['condition']}")
        _policy_rows(t["metrics"])


def validate_shadows(shadows):
    """Only complete summaries are calibration evidence; prefixes stay explicit."""
    if not isinstance(shadows, list):
        raise ValueError("shadows must be a list")
    seen, units = set(), set()
    for shadow in shadows:
        if not isinstance(shadow, dict):
            raise ValueError("shadow must be an object")
        for name in ("unit_id", "replica", "schedule", "condition", "completed", "metrics", "summary"):
            if name not in shadow:
                raise ValueError(f"shadow missing {name}")
        for name in ("unit_id", "schedule", "condition"):
            if not isinstance(shadow[name], str) or not shadow[name]:
                raise ValueError(f"shadow {name} must be a nonempty string")
        if isinstance(shadow["replica"], bool) or not isinstance(shadow["replica"], (int, str)):
            raise ValueError("shadow replica must be integer or string")
        key = (shadow["replica"], shadow["schedule"], shadow["condition"])
        if key in seen or shadow["unit_id"] in units:
            raise ValueError("duplicate shadow identity")
        seen.add(key)
        units.add(shadow["unit_id"])
        if type(shadow["completed"]) is not bool or not isinstance(shadow["summary"], dict):
            raise ValueError("shadow completed must be bool and summary must be object")
        rows = shadow["metrics"]
        if not isinstance(rows, list):
            raise ValueError("shadow metrics must be a list")
        previous = -1
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("shadow metrics must contain objects")
            arrival = _number(row.get("arrivals"), "shadow arrivals")
            if arrival < 0 or arrival <= previous:
                raise ValueError("shadow arrivals must be nonnegative and increasing")
            previous = arrival
        _policy_rows(rows)
        if shadow["completed"]:
            _number(shadow["summary"].get("log_score"), "shadow summary.log_score")
            regimes = shadow["summary"].get("regime_log_scores")
            if not isinstance(regimes, dict):
                raise ValueError("complete shadow requires summary.regime_log_scores object")
            for name, value in regimes.items():
                _number(value, f"shadow regime_log_scores.{name}")
            if shadow["summary"].get("pi_mean") is not None:
                if not 0 <= _number(shadow["summary"]["pi_mean"], "shadow pi_mean") <= 1:
                    raise ValueError("shadow pi_mean must be in [0, 1]")


def _contrast(key, treatment, control, by_condition, summaries, *, draws, seed, eligible_replicas=None):
    controls, targets = by_condition[control], by_condition[treatment]
    paired = set(controls) & set(targets)
    if eligible_replicas is not None:
        paired &= eligible_replicas
    paired = sorted(paired, key=lambda r: (type(r).__name__, str(r)))
    for replica in paired:
        _paired_path(controls[replica], targets[replica])
        for name in ("stream_hash", "initial_unit", "asset_unit"):
            if (name in controls[replica]) != (name in targets[replica]):
                raise ValueError(f"paired trajectories have mismatched {name} availability")
    pairs = [(summaries[controls[r]["unit_id"]], summaries[targets[r]["unit_id"]]) for r in paired]
    # Identical cohorts reuse identical bootstrap blocks across metrics/contrasts.
    block_seed = _seed(seed, [key, paired])
    def estimate(values, replicas=paired):
        return _estimate(values, draws=draws, seed=block_seed if replicas == paired else _seed(seed, [key, replicas]))
    result = {
        "condition": treatment, "control": control, "replicas": paired, "n_pairs": len(paired),
        "control_units": [controls[r]["unit_id"] for r in paired],
        "treatment_units": [targets[r]["unit_id"] for r in paired],
        "unpaired_control_units": [t["unit_id"] for r, t in controls.items() if r not in paired],
        "unpaired_treatment_units": [t["unit_id"] for r, t in targets.items() if r not in paired],
    }
    for scope in ("observation_mean", "endpoint"):
        result[scope + "_pp"] = {name: estimate([100 * (b[scope][name] - a[scope][name]) for a, b in pairs]) for name in ACCURACIES}
    result["segments"] = {}
    for segment in sorted({s for a, _ in pairs for s in a["segments"]}):
        result["segments"][segment] = {scope + "_pp": {
            name: estimate([100 * (b["segments"][segment][scope][name] - a["segments"][segment][scope][name]) for a, b in pairs])
            for name in ACCURACIES} for scope in ("observation_mean", "endpoint")}
    result["costs"] = {}
    for name in sorted({n for a, b in pairs for n in set(a["costs"]) & set(b["costs"])}):
        cost_pairs = [(r, a["costs"][name], b["costs"][name]) for r, (a, b) in zip(paired, pairs)
                      if name in a["costs"] and name in b["costs"]]
        positive = [(r, a, b) for r, a, b in cost_pairs if a > 0]
        result["costs"][name] = {
            "control_mean": float(np.mean([a for _, a, _ in cost_pairs])),
            "treatment_mean": float(np.mean([b for _, _, b in cost_pairs])),
            "difference": estimate([b - a for _, a, b in cost_pairs], [r for r, _, _ in cost_pairs]),
            "ratio": estimate([b / a for _, a, b in positive], [r for r, _, _ in positive]),
        }
    return result


def summarize(trajectories: list, *, bootstrap_draws=2000, seed=DEFAULT_SEED, required_conditions=None) -> dict:
    """All adaptive policies vs fixed, movement policies also vs covariance.

    A selected fixed/covariance/EMA/Kalman cohort has five contrasts per schedule.
    Candidate IDs are never collapsed, and incomplete prefixes are never paired.
    Requested draws are bounded to avoid unbounded report-time computation.
    When required_conditions is provided, fresh contrasts require all those arms
    completed in the same replica/stage/schedule/c/ridge block. Completed rows
    stay intact; withheld arms remain unmatched and explicitly await the block.
    None preserves the legacy pairwise output exactly, without new schema keys.
    """
    if type(bootstrap_draws) is not int:
        raise ValueError("bootstrap_draws must be an integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if required_conditions is not None:
        if (not isinstance(required_conditions, (list, tuple)) or not required_conditions
                or any(not isinstance(c, str) or not c for c in required_conditions)
                or len(set(required_conditions)) != len(required_conditions)):
            raise ValueError("required_conditions must be a nonempty sequence of distinct condition IDs")
        required_conditions = list(required_conditions)
    draws = max(0, min(bootstrap_draws, MAX_BOOTSTRAP_DRAWS))
    validate_trajectories(trajectories)
    grouped = defaultdict(list)
    for t in trajectories:
        grouped[(t["stage"], t["schedule"], float(t["c"]), float(t["ridge"]))].append(t)
    groups = []
    for key, rows in sorted(grouped.items()):
        complete = sorted((t for t in rows if t["completed"]), key=lambda t: t["unit_id"])
        by_condition = defaultdict(dict)
        for t in complete:
            by_condition[t["condition"]][t["replica"]] = t
        summaries = {t["unit_id"]: _path_summary(t) for t in complete}
        conditions = sorted({t["condition"] for t in rows})
        contrasts = [(name, "fixed") for name in conditions if name != "fixed"]
        contrasts += [(name, "covariance") for name in conditions if is_movement(name)]
        group = dict(zip(("stage", "schedule", "c", "ridge"), key))
        eligible = None
        if required_conditions is not None and (group["stage"] == "fresh" or group["stage"].startswith("fresh_")):
            eligible = set.intersection(*(set(by_condition[c]) for c in required_conditions))
            group.update(eligible_replicas=sorted(eligible, key=lambda r: (type(r).__name__, str(r))),
                         awaiting_full_block_units=[t["unit_id"] for t in complete if t["replica"] not in eligible])
        group.update(conditions=conditions, complete_units=[t["unit_id"] for t in complete],
                     incomplete_units=sorted(t["unit_id"] for t in rows if not t["completed"]),
                     replicas=list(summaries.values()), comparisons=[
                         _contrast(key, treatment, control, by_condition, summaries, draws=draws, seed=seed,
                                   eligible_replicas=eligible)
                         for treatment, control in contrasts])
        groups.append(group)
    result = {
        "schema": SCHEMA, "kind": KIND, "averaging": AVERAGING,
        "trajectories_sha256": trajectories_hash(trajectories),
        "bootstrap": {"requested_draws": bootstrap_draws, "draws": draws, "seed": seed, "confidence": .95,
                      "unit": "complete paired replica trajectories", "method": "percentile",
                      "coupling": "identical whole-replica draws for all outcomes and contrasts sharing a cohort"},
        "counts": {"sealed_trajectories": len(trajectories), "complete": sum(t["completed"] for t in trajectories),
                   "incomplete": sum(not t["completed"] for t in trajectories)},
        "groups": groups,
    }
    if required_conditions is not None:
        result["required_conditions"] = required_conditions
    return result
