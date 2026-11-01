"""Bounded, replica-level Plan 16 analysis; no execution-package dependencies.

Input accuracies are fractions. ``arrivals`` is cumulative incoming observation
count (step times fixed batch size), not the number of evaluation rows.
Each metric vector includes the initial (arrival zero) and final evaluations.
``expected_nines`` / ``observed_nines`` and ``wall_seconds`` are cumulative.
Optional ``summary.costs`` contains measured, whole-unit costs in seconds;
``summary.frozen_initial`` contains contextual initial-model accuracies.
Only ``completed is True`` trajectories enter inference. Incomplete rows may
contain sealed prefixes, but are never promoted to complete trajectories.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

import numpy as np

SCHEMA = 1
DEFAULT_SEED = 160016
MAX_BOOTSTRAP_DRAWS = 20_000
ACCURACIES = ("nine_recall", "old_accuracy")
FRACTIONS = (*ACCURACIES, "nine_fpr", "nine_precision", "negative_fraction", "p")
ALIASES = {"fd": "paired_fd", "analytic": "analytic_lfu"}
AVERAGING = (
    "Trapezoidal integration over recorded cumulative observations (arrivals), "
    "divided by the full observation span. "
    "Segment intervals belong to the segment of their right endpoint."
)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def condition_name(value):
    return ALIASES.get(value, value)


def validate_trajectories(trajectories):
    """Validate compact scalar rows without reading checkpoints or unit dirs."""
    if not isinstance(trajectories, list):
        raise ValueError("trajectories must be a list")
    seen, units = set(), set()
    for t in trajectories:
        if not isinstance(t, dict):
            raise ValueError("each trajectory must be an object")
        for key in ("unit_id", "stage", "replica", "schedule", "condition", "c", "ridge", "completed", "metrics", "summary"):
            if key not in t:
                raise ValueError(f"trajectory missing {key}")
        for key in ("unit_id", "stage", "schedule", "condition"):
            if not isinstance(t[key], str) or not t[key]:
                raise ValueError(f"trajectory {key} must be a nonempty string")
        if isinstance(t["replica"], bool) or not isinstance(t["replica"], (int, str)):
            raise ValueError("replica must be an integer or string")
        c, ridge = _number(t["c"], "c"), _number(t["ridge"], "ridge")
        if not 0 < c <= 1 or ridge < 0:
            raise ValueError("require 0 < c <= 1 and ridge >= 0")
        if not isinstance(t["completed"], bool) or not isinstance(t["summary"], dict):
            raise ValueError("completed must be bool and summary must be an object")
        key = (t["stage"], t["schedule"], c, ridge, t["replica"], condition_name(t["condition"]))
        if key in seen or t["unit_id"] in units:
            raise ValueError("duplicate trajectory / unit identity")
        seen.add(key)
        units.add(t["unit_id"])
        rows = t["metrics"]
        if not isinstance(rows, list) or (t["completed"] and len(rows) < 2):
            raise ValueError("complete trajectories require initial and final metrics")
        previous = -1.0
        cumulative = {}
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("metric rows must be objects")
            arrival = _number(row.get("arrivals"), "arrivals")
            if arrival < 0 or arrival <= previous:
                raise ValueError("arrivals must be nonnegative and strictly increasing")
            if i == 0 and t["completed"] and arrival != 0:
                raise ValueError("complete trajectory must include arrival zero")
            previous = arrival
            for name in ACCURACIES:
                _number(row.get(name), name)
            for name in FRACTIONS:
                if row.get(name) is not None and not 0 <= _number(row[name], name) <= 1:
                    raise ValueError(f"{name} must be a fraction in [0, 1]")
            for name in ("expected_nines", "observed_nines", "wall_seconds"):
                if row.get(name) is not None:
                    value = _number(row[name], name)
                    if value < cumulative.get(name, 0):
                        raise ValueError(f"{name} must be nonnegative and cumulative")
                    cumulative[name] = value
            for name in ("nll", "negative_mass", "closure_count", "grad_norm"):
                if row.get(name) is not None and _number(row[name], name) < 0:
                    raise ValueError(f"{name} must be nonnegative")
            if row.get("segment") is not None and not isinstance(row["segment"], str):
                raise ValueError("segment must be a string")
        costs = t["summary"].get("costs", {})
        if not isinstance(costs, dict):
            raise ValueError("summary.costs must be an object")
        for name, value in costs.items():
            if value is not None and _number(value, f"costs.{name}") < 0:
                raise ValueError("costs must be nonnegative")
        baseline = t["summary"].get("frozen_initial", {})
        if not isinstance(baseline, dict):
            raise ValueError("summary.frozen_initial must be an object")
        for name in ACCURACIES:
            if baseline.get(name) is not None and not 0 <= _number(baseline[name], name) <= 1:
                raise ValueError("frozen initial accuracies must be fractions")
            field = "frozen_" + name
            values = [r[field] for r in rows if r.get(field) is not None]
            if values and (any(not 0 <= _number(v, field) <= 1 for v in values) or len(set(values)) != 1):
                raise ValueError(f"{field} must be a constant fraction")


def _path_summary(t):
    rows = t["metrics"]
    x = np.asarray([r["arrivals"] for r in rows], dtype=float)
    dx = np.diff(x)
    segments = {}
    for segment in dict.fromkeys(r.get("segment", "unspecified") for r in rows[1:]):
        mask = np.asarray([r.get("segment", "unspecified") == segment for r in rows[1:]])
        indices = np.flatnonzero(mask) + 1
        segments[segment] = {"arrivals": float(dx[mask].sum()), "endpoint": {}, "observation_mean": {}}
        for name in ACCURACIES:
            y = np.asarray([r[name] for r in rows], dtype=float)
            segments[segment]["observation_mean"][name] = float(np.sum(dx[mask] * (y[:-1] + y[1:])[mask] / 2) / dx[mask].sum())
            segments[segment]["endpoint"][name] = float(y[indices[-1]])
    costs = {k: float(v) for k, v in t["summary"].get("costs", {}).items() if v is not None}
    wall = costs.get("wall_seconds", t["summary"].get("wall_seconds", rows[-1].get("wall_seconds")))
    if wall is not None:
        costs["wall_seconds"] = _number(wall, "wall_seconds")
        costs["seconds_per_arrival"] = costs["wall_seconds"] / float(x[-1])
    return {
        "unit_id": t["unit_id"], "replica": t["replica"], "condition": condition_name(t["condition"]),
        "arrivals": float(x[-1]),
        "observation_mean": {name: float(np.sum(dx * (np.asarray([r[name] for r in rows[:-1]]) +
            np.asarray([r[name] for r in rows[1:]])) / 2) / (x[-1] - x[0])) for name in ACCURACIES},
        "endpoint": {name: float(rows[-1][name]) for name in ACCURACIES},
        "segments": segments, "costs": costs,
    }


def _paired_path(control, treatment):
    """Never silently truncate or interpolate mismatched pairing artifacts."""
    for name in ("arrivals", "p", "segment", "expected_nines", "observed_nines"):
        if [r.get(name) for r in control["metrics"]] != [r.get(name) for r in treatment["metrics"]]:
            raise ValueError(f"paired trajectories have mismatched {name}: {control['unit_id']} / {treatment['unit_id']}")
    for name in ("stream_hash", "initial_unit", "asset_unit"):
        if name in control and name in treatment and control[name] != treatment[name]:
            raise ValueError(f"paired trajectories have mismatched {name}")


def _seed(seed, key):
    digest = hashlib.sha256(json.dumps([int(seed), key], sort_keys=True).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _estimate(values, *, draws, seed):
    values = np.asarray(values, dtype=float)
    n = len(values)
    interval = None
    if n >= 2 and draws:
        rng = np.random.default_rng(seed)
        means = np.empty(draws)
        # Only whole-replica scalars are resampled; bounded temporary storage.
        for offset in range(0, draws, 128):
            count = min(128, draws - offset)
            means[offset:offset + count] = values[rng.integers(0, n, (count, n))].mean(axis=1)
        interval = [float(v) for v in np.quantile(means, [.025, .975])]
    return {"n_pairs": n, "mean": float(values.mean()) if n else None,
            "ci95": interval, "individual": values.tolist(), "interval_reliable": n >= 4 and draws >= 200,
            "interval_note": "nominal percentile interval; unreliable with fewer than four pairs or 200 draws"
            if n < 4 or draws < 200 else "nominal, unadjusted for multiple comparisons / interim inspection"}


def summarize(trajectories: list, *, bootstrap_draws=2000, seed=DEFAULT_SEED) -> dict:
    """Return JSON-safe paired contrasts, costs and segment/endpoint summaries.

    Group keys are exactly (stage, schedule, c, ridge); revisions must have a
    distinct stage/schedule identity. FD/analytic aliases normalize to
    paired_fd/analytic_lfu. Both specified contrasts are returned, even if no
    pair exists. No development/fresh or schedule pooling occurs. Requested
    draws are clamped to [0, 20000] and the resolved count/seed are recorded.
    """
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise ValueError("bootstrap_draws must be an integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    draws = max(0, min(bootstrap_draws, MAX_BOOTSTRAP_DRAWS))
    validate_trajectories(trajectories)
    grouped = defaultdict(list)
    for t in trajectories:
        grouped[(t["stage"], t["schedule"], float(t["c"]), float(t["ridge"]))].append(t)
    groups = []
    for key in sorted(grouped):
        rows = grouped[key]
        complete = [t for t in rows if t["completed"]]
        by_condition = defaultdict(dict)
        for t in complete:
            by_condition[condition_name(t["condition"])][t["replica"]] = t
        summaries = {t["unit_id"]: _path_summary(t) for t in complete}
        group = dict(zip(("stage", "schedule", "c", "ridge"), key))
        group.update(complete_units=[t["unit_id"] for t in complete],
                     incomplete_units=[t["unit_id"] for t in rows if not t["completed"]],
                     replicas=sorted(summaries.values(), key=lambda r: r["unit_id"]), comparisons=[])
        controls = by_condition["ema"]
        treatments = sorted({"paired_fd", "analytic_lfu"} | (set(by_condition) - {"ema", "frozen_initial"}))
        for treatment in treatments:
            targets = by_condition[treatment]
            paired = sorted(set(controls) & set(targets), key=lambda r: (type(r).__name__, str(r)))
            for replica in paired:
                _paired_path(controls[replica], targets[replica])
            contrast = {"condition": treatment, "control": "ema", "replicas": paired, "n_pairs": len(paired),
                        "control_units": [controls[r]["unit_id"] for r in paired],
                        "treatment_units": [targets[r]["unit_id"] for r in paired],
                        "unpaired_control_units": [controls[r]["unit_id"] for r in controls if r not in paired],
                        "unpaired_treatment_units": [targets[r]["unit_id"] for r in targets if r not in paired]}
            pairs = [(summaries[controls[r]["unit_id"]], summaries[targets[r]["unit_id"]]) for r in paired]
            for scope in ("observation_mean", "endpoint"):
                contrast[scope + "_pp"] = {name: _estimate(
                    [100 * (b[scope][name] - a[scope][name]) for a, b in pairs], draws=draws,
                    seed=_seed(seed, [key, treatment, scope, name])) for name in ACCURACIES}
            contrast["segments"] = {}
            segments = sorted({s for a, _ in pairs for s in a["segments"]})
            for segment in segments:
                segment_pairs = [(a["segments"][segment], b["segments"][segment]) for a, b in pairs
                                 if segment in a["segments"] and segment in b["segments"]]
                contrast["segments"][segment] = {scope + "_pp": {name: _estimate(
                    [100 * (b[scope][name] - a[scope][name]) for a, b in segment_pairs], draws=draws,
                    seed=_seed(seed, [key, treatment, segment, scope, name])) for name in ACCURACIES}
                    for scope in ("observation_mean", "endpoint")}
            cost_names = sorted({name for a, b in pairs for name in set(a["costs"]) & set(b["costs"])})
            contrast["costs"] = {}
            for name in cost_names:
                cost_pairs = [(a["costs"][name], b["costs"][name]) for a, b in pairs if name in a["costs"] and name in b["costs"]]
                contrast["costs"][name] = {
                    "control_mean": float(np.mean([a for a, _ in cost_pairs])),
                    "treatment_mean": float(np.mean([b for _, b in cost_pairs])),
                    "difference": _estimate([b - a for a, b in cost_pairs], draws=draws, seed=_seed(seed, [key, treatment, name, "difference"])),
                    "ratio": _estimate([b / a for a, b in cost_pairs if a > 0], draws=draws, seed=_seed(seed, [key, treatment, name, "ratio"])),
                }
            group["comparisons"].append(contrast)
        groups.append(group)
    return {"schema": SCHEMA, "kind": "plan16-paired-analysis", "averaging": AVERAGING,
            "trajectories_sha256": hashlib.sha256(json.dumps(trajectories, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "bootstrap": {"requested_draws": bootstrap_draws, "draws": draws, "seed": seed,
                          "confidence": .95, "unit": "complete paired replica trajectories", "method": "percentile"},
            "counts": {"sealed_trajectories": len(trajectories), "complete": sum(t["completed"] for t in trajectories),
                       "incomplete": sum(not t["completed"] for t in trajectories)}, "groups": groups}
