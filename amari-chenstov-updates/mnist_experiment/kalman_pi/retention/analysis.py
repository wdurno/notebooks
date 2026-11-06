"""Bounded Phase 5 retention calibration and selected-package confirmation.

Accuracies are fractions; differences are percentage points (pp). Full paths
use trapezoidal arrival-clock averages, then equal weight per replica. Only
development rows select ridge. The historical common-ridge analyzer is unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

import numpy as np

from mnist_experiment.diagonal_lfu.analysis import (
    ACCURACIES, AVERAGING, MAX_BOOTSTRAP_DRAWS, _estimate, _number,
    _paired_path, _path_summary, _seed,
)
from mnist_experiment.kalman_pi.analysis import (
    _contrast, trajectories_hash, validate_trajectories,
)

SCHEMA = 1
KIND = "plan17-retention-analysis"
DEFAULT_SEED = 170517
CONDITIONS = ("fixed", "covariance", "ema_h16", "kalman_eta0.0001")
SCHEDULES = ("main", "stationary")
FIXED_RIDGE = .1
C = .025
TOLERANCE_PP = .5
MAX_RIDGE = 3.


def _replicas(value):
    """A count denotes range(count); explicit replica IDs are also accepted."""
    if type(value) is int:
        if value < 0:
            raise ValueError("replicas must be nonnegative")
        return set(range(value))
    if not isinstance(value, (list, tuple, set, range)):
        raise ValueError("replicas must be a count or a collection of IDs")
    if any(isinstance(r, bool) or not isinstance(r, (int, str)) for r in value):
        raise ValueError("replicas must contain integer or string IDs")
    if len(set(value)) != len(value):
        raise ValueError("replicas must contain distinct IDs")
    return set(value)


def _in_band(gap, tolerance):
    # Fraction-to-pp conversion can move a boundary by a few float64 ulps.
    return abs(gap) <= tolerance or math.isclose(abs(gap), tolerance, rel_tol=0, abs_tol=1e-12)


def _tolerance(value):
    value = _number(value, "tolerance_pp")
    if value < 0:
        raise ValueError("tolerance_pp must be nonnegative")
    return value


def _validate_phase(rows):
    validate_trajectories(rows)
    for t in rows:
        if t["stage"] not in ("development", "fresh"):
            raise ValueError("Phase 5 stage must be development or fresh")
        if t["schedule"] not in SCHEDULES or t["condition"] not in CONDITIONS:
            raise ValueError("unexpected Phase 5 schedule or condition")
        if t["c"] != C:
            raise ValueError("Phase 5 requires c=.025")
        if t["ridge"] > MAX_RIDGE:
            raise ValueError("Phase 5 ridge exceeds the bounded family r<=3")
        if t["condition"] == "fixed" and t["ridge"] != FIXED_RIDGE:
            raise ValueError("Phase 5 fixed reference requires ridge=.1")


def _pair(control, treatment):
    for name in ("stage", "schedule", "replica", "c"):
        if control[name] != treatment[name]:
            raise ValueError(f"paired trajectories have mismatched {name}")
    for name in ("stream_hash", "asset_unit", "initial_unit"):
        if any(not isinstance(t.get(name), str) or not t[name] for t in (control, treatment)):
            raise ValueError(f"paired trajectories require explicit {name}")
    _paired_path(control, treatment)
    if [r.get("step") for r in control["metrics"]] != [r.get("step") for r in treatment["metrics"]]:
        raise ValueError("paired trajectories have mismatched step grid")
    for name in ACCURACIES:
        if control["metrics"][0][name] != treatment["metrics"][0][name]:
            raise ValueError("paired trajectories have mismatched initial accuracies")


def _summary(t):
    result = _path_summary(t)
    costs = result["costs"]
    for name in ("learning_seconds", "online_wall_seconds"):
        if t["summary"].get(name) is not None:
            value = _number(t["summary"][name], name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            if name in costs and costs[name] != value:
                raise ValueError(f"conflicting measured {name}")
            costs[name] = value
    components = ("solver_seconds", "curvature_seconds", "controller_seconds")
    if "learning_seconds" not in costs and all(name in costs for name in components):
        costs["learning_seconds"] = sum(costs[name] for name in components)
    result["ridge"] = float(t["ridge"])
    return result


def _table(rows, schedule, condition, replicas, *, draws, seed):
    intended = None if replicas is None else _replicas(replicas)
    rows = [t for t in rows if t["schedule"] == schedule and t["completed"]
            and (intended is None or t["replica"] in intended)]
    controls = {t["replica"]: t for t in rows if t["condition"] == "fixed"}
    candidates = defaultdict(dict)
    for t in rows:
        if t["condition"] == condition:
            candidates[float(t["ridge"])][t["replica"]] = t
    summaries = {t["unit_id"]: _summary(t) for t in rows}
    table = []
    for ridge, targets in sorted(candidates.items()):
        paired = set(controls) & set(targets)
        if not paired or (intended is not None and paired != intended):
            continue
        paired = sorted(paired, key=lambda r: (type(r).__name__, str(r)))
        for r in paired:
            _pair(controls[r], targets[r])
        fixed = [summaries[controls[r]["unit_id"]] for r in paired]
        treatment = [summaries[targets[r]["unit_id"]] for r in paired]
        estimates = {name: _estimate([
            100 * (b["observation_mean"][name] - a["observation_mean"][name])
            for a, b in zip(fixed, treatment)], draws=draws,
            seed=_seed(seed, ["development", schedule, paired])) for name in ACCURACIES}
        times = [t["costs"].get("learning_seconds") for t in treatment]
        old_gap = estimates["old_accuracy"]["mean"]
        table.append({
            "ridge": ridge, "n_pairs": len(paired),
            "old_gap_pp": old_gap, "nine_gap_pp": estimates["nine_recall"]["mean"],
            "nine_recall": float(np.mean([t["observation_mean"]["nine_recall"] for t in treatment])),
            "old_accuracy": float(np.mean([t["observation_mean"]["old_accuracy"] for t in treatment])),
            "learning_seconds": float(np.mean(times)) if all(t is not None for t in times) else None,
            "matched": _in_band(old_gap, TOLERANCE_PP),
            "old_gap_pp_ci95": estimates["old_accuracy"]["ci95"],
            "nine_gap_pp_ci95": estimates["nine_recall"]["ci95"],
        })
    return table


def candidate_table(trajectories, schedule, condition, replicas=None):
    """Paired development-only candidate rows, sorted by measured ridge.

    With ``replicas`` (a count or IDs), omit a candidate until all intended
    fixed/treatment paths are complete. Without it, report available complete
    pairs explicitly via n_pairs. Never inspect confirmation outcomes.
    """
    if schedule not in SCHEDULES or condition not in CONDITIONS:
        raise ValueError("unexpected Phase 5 schedule or condition")
    if not isinstance(trajectories, list) or any(not isinstance(t, dict) for t in trajectories):
        raise ValueError("trajectories must be a list of objects")
    rows = [t for t in trajectories if t.get("stage") == "development"]
    _validate_phase(rows)
    return _table(rows, schedule, condition, replicas, draws=2000, seed=DEFAULT_SEED)


def _finite_candidates(table):
    if not isinstance(table, list):
        raise ValueError("candidate table must be a list")
    valid, seen = [], set()
    for row in table:
        if not isinstance(row, dict):
            raise ValueError("candidate rows must be objects")
        if type(row.get("n_pairs")) is not int or row["n_pairs"] <= 0:
            continue
        try:
            values = {name: _number(row.get(name), name) for name in
                      ("ridge", "old_gap_pp", "nine_gap_pp", "nine_recall", "old_accuracy")}
            time = row.get("learning_seconds")
            if time is not None and _number(time, "learning_seconds") < 0:
                continue
        except ValueError:
            continue
        if not 0 <= values["ridge"] <= MAX_RIDGE or any(
                not 0 <= values[name] <= 1 for name in ACCURACIES):
            continue
        if values["ridge"] in seen:
            raise ValueError("duplicate candidate ridge")
        seen.add(values["ridge"])
        valid.append(row)
    return valid


def select_candidate(table, tolerance_pp=.5):
    """Selected measured row with explicit matched status, or None if invalid.

    In-band: highest recall, smallest absolute old gap, least learning time,
    smallest ridge. Otherwise: smallest absolute old gap, highest recall, least
    learning time, smallest ridge. Missing timing cannot win a timing tie.
    """
    tolerance = _tolerance(tolerance_pp)
    valid = _finite_candidates(table)
    matched = [r for r in valid if _in_band(r["old_gap_pp"], tolerance)]
    if not valid:
        return None

    def key(row):
        acquisition, gap = -row["nine_recall"], abs(row["old_gap_pp"])
        time = row.get("learning_seconds")
        accuracy_key = (acquisition, gap) if matched else (gap, acquisition)
        return (*accuracy_key, math.inf if time is None else time, row["ridge"])

    selected = deepcopy(min(matched or valid, key=key))
    selected["matched"] = bool(matched)
    return selected


def refinement_candidate(table, tolerance_pp=.5):
    """One deterministic measured-neighbor refinement, or None.

    The caller persists decisions and enforces the two-additional-value limit.
    No predicted outcome is a candidate; existing ridges are never proposed.
    """
    tolerance = _tolerance(tolerance_pp)
    rows = sorted(_finite_candidates(table), key=lambda r: r["ridge"])
    if len(rows) < 2 or any(_in_band(r["old_gap_pp"], tolerance) for r in rows):
        return None
    brackets = [(a, b) for a, b in zip(rows, rows[1:]) if
                (a["old_gap_pp"] < 0) != (b["old_gap_pp"] < 0)]
    if brackets:
        a, b = min(brackets, key=lambda ab: (abs(ab[0]["old_gap_pp"]) +
                                            abs(ab[1]["old_gap_pp"]), ab[0]["ridge"]))
        proposed = (a["ridge"] + b["ridge"]) / 2 if a["ridge"] == 0 else math.sqrt(a["ridge"] * b["ridge"])
    else:
        closest = min(range(len(rows)), key=lambda i: (abs(rows[i]["old_gap_pp"]), rows[i]["ridge"]))
        if closest == len(rows) - 1 and abs(rows[-1]["old_gap_pp"]) < abs(rows[-2]["old_gap_pp"]) and rows[-1]["ridge"] < MAX_RIDGE:
            proposed = min(MAX_RIDGE, 3 * rows[-1]["ridge"])
        else:
            neighbors = [rows[i] for i in (closest - 1, closest + 1) if 0 <= i < len(rows)]
            neighbor = min(neighbors, key=lambda r: (abs(r["old_gap_pp"]), r["ridge"]))
            a, b = sorted((rows[closest]["ridge"], neighbor["ridge"]))
            proposed = (a + b) / 2 if a == 0 else math.sqrt(a * b)
    tested = {r.get("ridge") for r in table}
    return float(proposed) if 0 <= proposed <= MAX_RIDGE and proposed not in tested else None


def _selection(selection):
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    conditions = selection.get("conditions", list(CONDITIONS))
    if (not isinstance(conditions, (list, tuple)) or len(conditions) != 4
            or any(not isinstance(c, str) for c in conditions) or set(conditions) != set(CONDITIONS)):
        raise ValueError("selection.conditions must retain all four Phase 5 policies")
    development = selection.get("development_replicas", {"main": 4, "stationary": 2})
    if not isinstance(development, dict) or set(development) != set(SCHEDULES):
        raise ValueError("selection.development_replicas must specify both schedules")
    for count in development.values():
        if type(count) is not int or count <= 0:
            raise ValueError("development_replicas counts must be positive integers")
    ridges = selection.get("ridges", {})
    if not isinstance(ridges, dict) or set(ridges) - set(SCHEDULES):
        raise ValueError("selection.ridges must map schedules to policies")
    resolved = {}
    for schedule, mapping in ridges.items():
        if not isinstance(mapping, dict) or set(mapping) - set(CONDITIONS):
            raise ValueError("selection.ridges contains unexpected policies")
        resolved[schedule] = {"fixed": FIXED_RIDGE}
        for condition, ridge in mapping.items():
            ridge = _number(ridge, "selected ridge")
            if not 0 <= ridge <= MAX_RIDGE or (condition == "fixed" and ridge != FIXED_RIDGE):
                raise ValueError("selected ridge violates Phase 5 fixed reference or bounded family")
            resolved[schedule][condition] = ridge
    frozen = selection.get("frozen_at")
    if frozen is not None:
        if not isinstance(frozen, str) or not frozen or frozen.lower() == "pending":
            raise ValueError("selection.frozen_at must identify the actual freeze")
        if any(set(resolved.get(s, {})) != set(CONDITIONS) for s in SCHEDULES):
            raise ValueError("frozen selection requires every policy/schedule ridge")
    for schedule in SCHEDULES:
        count = selection.get(schedule + "_replicas")
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("confirmation replica counts must be nonnegative integers")
        if frozen is not None and count is None:
            raise ValueError("frozen selection requires confirmation replica counts")
    return list(conditions), development, resolved, frozen


def summarize(trajectories, selection, *, bootstrap_draws=2000, seed=170517):
    """Development maps and full-block confirmation under the supplied selection.

    Pending development accepts a partial ridge map; any fresh rows require a
    fully frozen map. This function never selects or retunes from fresh data.
    Explicit selected ridges replace only this phase's common-ridge grouping.
    """
    if type(bootstrap_draws) is not int:
        raise ValueError("bootstrap_draws must be an integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    draws = max(0, min(bootstrap_draws, MAX_BOOTSTRAP_DRAWS))
    conditions, development, ridges, frozen = _selection(selection)
    _validate_phase(trajectories)
    dev = [t for t in trajectories if t["stage"] == "development"]
    fresh = [t for t in trajectories if t["stage"] == "fresh"]
    if fresh and frozen is None:
        raise ValueError("fresh trajectories require frozen selection")
    for t in fresh:
        if float(t["ridge"]) != ridges[t["schedule"]][t["condition"]]:
            raise ValueError("fresh trajectory does not have its expected selected ridge")
        if t["replica"] not in _replicas(selection[t["schedule"] + "_replicas"]):
            raise ValueError("fresh replica is outside the frozen allocation")
    groups = []
    for schedule in SCHEDULES:
        rows = [t for t in fresh if t["schedule"] == schedule]
        if not rows and schedule not in ridges:
            continue
        complete = sorted((t for t in rows if t["completed"]), key=lambda t: t["unit_id"])
        by_condition = defaultdict(dict)
        for t in complete:
            by_condition[t["condition"]][t["replica"]] = t
        # Validate shared assets even for finished arms awaiting the other arms.
        replicas = {t["replica"] for t in complete}
        for replica in replicas:
            block = [by_condition[c][replica] for c in conditions if replica in by_condition[c]]
            for t in block[1:]:
                _pair(block[0], t)
        eligible = set.intersection(*(set(by_condition[c]) for c in conditions))
        summaries = {t["unit_id"]: _summary(t) for t in complete}
        contrasts = [(c, "fixed") for c in conditions if c != "fixed"]
        contrasts += [(c, "covariance") for c in conditions if c.startswith(("ema_h", "kalman_"))]
        groups.append({
            "stage": "fresh", "schedule": schedule, "c": C,
            "ridges": deepcopy(ridges.get(schedule, {})), "conditions": conditions,
            "eligible_replicas": sorted(eligible, key=lambda r: (type(r).__name__, str(r))),
            "awaiting_full_block_units": [t["unit_id"] for t in complete if t["replica"] not in eligible],
            "complete_units": [t["unit_id"] for t in complete],
            "incomplete_units": sorted(t["unit_id"] for t in rows if not t["completed"]),
            "replicas": list(summaries.values()),
            "comparisons": [_contrast(["fresh", schedule, C, ridges.get(schedule, {})], treatment,
                                      control, by_condition, summaries, draws=draws, seed=seed,
                                      eligible_replicas=eligible) for treatment, control in contrasts],
        })
    return {
        "schema": SCHEMA, "kind": KIND, "averaging": AVERAGING,
        "trajectories_sha256": trajectories_hash(trajectories),
        "selection": deepcopy(selection), "selection_sha256": trajectories_hash(selection),
        "required_conditions": conditions,
        "bootstrap": {"requested_draws": bootstrap_draws, "draws": draws, "seed": seed,
                      "confidence": .95, "unit": "complete paired replica trajectories", "method": "percentile",
                      "coupling": "identical whole-replica draws for all outcomes and contrasts sharing a cohort"},
        "counts": {"sealed_trajectories": len(trajectories),
                   "complete": sum(t["completed"] for t in trajectories),
                   "incomplete": sum(not t["completed"] for t in trajectories)},
        "development": {s: {c: _table(dev, s, c, development[s], draws=draws, seed=seed)
                            for c in conditions} for s in SCHEDULES},
        "groups": groups,
    }
