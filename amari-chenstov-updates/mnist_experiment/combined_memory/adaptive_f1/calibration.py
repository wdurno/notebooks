"""Bounded ridge proposals using only four paired development replicas."""

import math
import numpy as np
from .analysis import validate_record, summarize_metrics, TOLERANCE, CEILING

FIELDS = ("Havg", "H100", "novel_auc", "learning_seconds")


def candidates(records, members=(0, 1, 2, 3)):
    groups = {}
    references = {}
    for raw in records:
        row = validate_record(raw)
        if row["namespace"] != "development":
            raise ValueError("development-only calibration required")
        r = row["replica"]
        if r not in range(4):
            raise ValueError("unexpected development member")
        reference = references.setdefault(r, row)
        if (reference["plan_hash"] != row["plan_hash"] or
                any(reference["metrics"][0][f] != row["metrics"][0][f] for f in ("base_mse", "novel_mse"))):
            raise ValueError("unpaired development candidate")
        if r not in members:
            continue
        key = row["policy"], row["recipe"]["ridge"]
        group = groups.setdefault(key, {})
        if r in group:
            raise ValueError("duplicate development candidate/member")
        group[r] = summarize_metrics(row["metrics"], N=row["target"])
    out = []
    for (p, ridge), rows in groups.items():
        out.append(dict(policy=p, ridge=ridge, members=sorted(rows), complete=set(rows)==set(members),
            per_member={str(k): v for k, v in rows.items()},
            **{f: math.fsum(v[f] for v in rows.values())/len(rows) for f in FIELDS}))
    return sorted(out, key=lambda x: (x["policy"], x["ridge"]))


def selection(records, members=(0, 1, 2, 3), *, require_match=True):
    rows = candidates(records, members)
    ref = next((r for r in rows if r["policy"] == "R" and r["complete"]), None)
    if ref is None:
        raise ValueError("all fixed R development references required")
    choices = []
    for row in rows:
        if row["policy"] != "K" or not row["complete"]:
            continue
        gaps = {f: row[f]-ref[f] for f in ("Havg", "H100")}
        ceiling = max(row["Havg"], row["H100"], ref["Havg"], ref["H100"])
        violation = max(0., *(abs(v)-TOLERANCE for v in gaps.values()), ceiling-CEILING)
        choice = dict(ridge=row["ridge"], reference=ref, adaptive=row, gaps=gaps,
                      maximum_violation=violation, passed=violation <= 1e-12)
        if choice["passed"] or not require_match:
            choices.append(choice)
    if not choices:
        return None
    return min(choices, key=lambda x: (0. if x["passed"] else x["maximum_violation"],
        x["adaptive"]["novel_auc"], x["adaptive"]["learning_seconds"], x["ridge"]))


def propose(records, members=(0, 1)):
    rows = candidates(records, members)
    ref = next(r for r in rows if r["policy"] == "R" and r["complete"])
    trials = [r for r in rows if r["policy"] == "K" and r["complete"]]
    tried = {r["ridge"] for r in trials}
    gap = lambda r: np.asarray([r[f]-ref[f] for f in ("Havg", "H100")])
    best = min(trials, key=lambda r: (float(np.abs(gap(r)).max()), r["novel_auc"], r["ridge"]))
    proposals = []
    positives = sorted((r for r in trials if r["ridge"] > 0), key=lambda r: r["ridge"])
    for a, b in zip(positives, positives[1:]):
        ga, gb = gap(a), gap(b)
        if np.any(ga*gb <= 0) and np.linalg.norm(gb-ga) > 0:
            alpha = float(np.clip(-ga.dot(gb-ga)/np.square(gb-ga).sum(), .1, .9))
            c = math.exp((1-alpha)*math.log(a["ridge"])+alpha*math.log(b["ridge"]))
            proposals.append((float(np.abs((1-alpha)*ga+alpha*gb).max()), c, "joint-gap log interpolation"))
    if proposals:
        _, value, reason = min(proposals)
    elif float(gap(best).mean()) < 0:
        positive = min(tried - {0.})
        value, reason = positive/10, "stronger retention than R; adjacent lower decade"
    else:
        value, reason = min(1000., max(tried)*10), "weaker retention than R; adjacent upper decade"
    if value in tried or value > 1000:
        if 0. not in tried and all(v <= 0 for v in gap(best)):
            value, reason = 0., "positive lower boundary still overprotects"
        else:
            return None
    return dict(ridge=float(value), reason=reason, members=list(members))
