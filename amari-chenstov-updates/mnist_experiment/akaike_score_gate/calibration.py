"""Development-only policy selection, with fixed data and resource allowances."""

from __future__ import annotations

import itertools
import statistics
from collections import defaultdict
from typing import Any

from .config import canonical_hash


def candidate_grid() -> list[dict[str, Any]]:
    base = {"c": 1., "half_life": 8., "cadence": 1}
    result = [base]
    for key, values in (("c", (.5, 2., 4.)), ("half_life", (4., 16.)), ("cadence", (2, 4))):
        result.extend({**base, key: value} for value in values)
    return result


def candidate_id(candidate: dict) -> str:
    return canonical_hash(candidate)[:16]


def balanced_score(rows: list[dict]) -> dict[str, float]:
    """Average replicas, then Fisher cells, then separately named conditions."""
    if not rows:
        raise ValueError("cannot select from missing trajectories")
    cells = defaultdict(list)
    for row in rows:
        cells[(row["environment"], row["policy"])].append(row)
    envs = defaultdict(list)
    fields = ("nll_auc", "retention_change", "retention_nll", "online_seconds", "mean_active_parameters")
    for (env, _), values in cells.items():
        envs[env].append({name: statistics.mean(r[name] for r in values) for name in fields})
    condition_scores = {env: {name: statistics.mean(r[name] for r in values) for name in fields}
                        for env, values in envs.items()}
    result = {name: statistics.mean(v[name] for v in condition_scores.values()) for name in fields}
    result["conditions"] = condition_scores
    result["admissible"] = result["retention_change"] <= .02
    result["replicas_per_cell"] = {f"{env}/{policy}": len(values) for (env, policy), values in cells.items()}
    return result


def select_candidate(scored: dict[str, dict], *, tie_nats: float = .005) -> tuple[str, dict]:
    if not scored:
        raise ValueError("no candidate results")
    admissible = [key for key, score in scored.items() if score["admissible"]]
    pool = admissible or list(scored)
    def risk(key):
        score = scored[key]
        return score["nll_auc"] + (0 if admissible else score["retention_nll"])
    best = min(risk(key) for key in pool)
    tied = [key for key in pool if risk(key) <= best + tie_nats]
    fastest = min(scored[key]["online_seconds"] for key in tied)
    timing_tied = [key for key in tied if scored[key]["online_seconds"] <= 1.02 * fastest]
    selected = min(timing_tied, key=lambda key: (scored[key]["mean_active_parameters"], risk(key), key))
    return selected, {"constraint_failure": not admissible, "tie_nats": tie_nats,
                      "timing_tie_relative": .02, "scores": scored}


def out_of_fold_selection(candidate_rows: dict[str, list[dict]], *,
                          screening_replicas: tuple[int, ...] = ()) -> list[dict]:
    indices = sorted({r["replica_index"] for rows in candidate_rows.values() for r in rows})
    if len(indices) < 2:
        return []
    results = []
    for held_out in indices:
        # A replica used to form the shortlist is not an untouched scoring fold.
        if held_out in screening_replicas:
            continue
        train = {key: balanced_score([r for r in rows if r["replica_index"] != held_out])
                 for key, rows in candidate_rows.items()}
        selected, _ = select_candidate(train)
        test = balanced_score([r for r in candidate_rows[selected] if r["replica_index"] == held_out])
        results.append({"held_out_replica": held_out, "selected": selected, "score": test,
                        "screening_replicas_excluded_from_scoring": list(screening_replicas)})
    return results


def preassigned_steps(total_steps: int, count: int, *, nonterminal: bool = True) -> tuple[int, ...]:
    end = max(0, total_steps - (2 if nonterminal else 1))
    if count <= 1:
        return (end // 2,)
    return tuple(sorted({round(i * end / (count - 1)) for i in range(count)}))
