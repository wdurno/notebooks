"""Pure Phase 7 calibration with B/ridge-1 as the retention reference.

Artifact identity and pairing checks belong to the caller. The inherited H100
metric denotes H_1216; retention is rooted within members before aggregation.
Partial evidence can guide a proposal, but never certify a selected tuple.
"""

from __future__ import annotations

from itertools import combinations, product
import math

from ..config import ARMS, Recipe
from ..retention_match import calibration as phase6


MEMBERS = phase6.MEMBERS
STRENGTH = phase6.STRENGTH
EPSILON = phase6.EPSILON
RETENTION = phase6.RETENTION
aggregate = phase6.aggregate
matching = phase6.matching


def initial_recipes(strength=STRENGTH):
    """Return independent starting recipes, not an asserted retention match."""
    strength = phase6._number(strength, "strength", minimum=0)
    return {arm: Recipe(arm, ridge=ridge, strength=0. if arm == "A" else strength,
                       process_scale=1e-5 if arm in ("D", "E") else 1e-4)
            for arm, ridge in zip(ARMS, (1., 1., 1., .1, .1))}


def compatible(recipe, strength=STRENGTH):
    """Whether a well-formed recipe follows the fixed Phase 7 search policy.

    Zero is an explicit boundary probe; positive ridges lie in [1e-4, 1000].
    Malformed recipes raise, whereas well-formed incompatible policies return
    False. An explicit strength override is only for separate smoke settings.
    """
    recipe = phase6._recipe(recipe)
    start = initial_recipes(strength)[recipe.arm]
    return ((recipe.ridge == 0 or 1e-4 <= recipe.ridge <= 1000)
            and recipe.strength == start.strength
            and recipe.process_scale == start.process_scale
            and (recipe.arm != "B" or recipe.ridge == 1.))


def _candidates(records, members, strength):
    """Validate executed members even if a candidate is incomplete."""
    candidates = {}
    for recipe, group in phase6._groups(records, members).items():
        if not compatible(recipe, strength):
            continue
        covered = tuple(member for member in members
                        if group.get(member, {}).get("status") == "completed")
        row = phase6._aggregate(group, recipe, covered) if covered else None
        candidates[recipe] = (group, covered, row)
    return candidates


def select_tuple(records, tolerance=.1, members=MEMBERS, *, strength=STRENGTH):
    """Select only complete, jointly matched recipes with B always at ridge 1.

    Acquisition-risk ties are anchored at the global minimum with 1e-8 slack;
    total learning time, A--E ridges, and stable recipe IDs break those ties.
    A is tunable here, but remains the downstream sample-equivalence reference.
    """
    tolerance = phase6._number(tolerance, "tolerance", minimum=0)
    members = phase6._members(members)
    strength = phase6._number(strength, "strength", minimum=0)
    pools, incomplete = {arm: [] for arm in ARMS}, {}
    for recipe, (group, covered, row) in _candidates(records, members, strength).items():
        missing = phase6._missing(group, members)
        if missing:
            incomplete[recipe.key] = missing
        else:
            pools[recipe.arm].append(row)
    if any(not pool for pool in pools.values()):
        return None
    feasible = []
    for combination in product(*(sorted(pools[arm], key=lambda row: phase6._recipe(row["recipe"]).key)
                                 for arm in ARMS)):
        rows = dict(zip(ARMS, combination))
        check = matching(rows, tolerance)
        if not check["passed"]:
            continue
        score = math.fsum(row["novel_auc"] / len(ARMS) for row in combination)
        seconds = math.fsum(row["total_learning_seconds"] for row in combination)
        ridges = tuple(row["recipe"]["ridge"] for row in combination)
        ids = tuple(phase6._recipe(row["recipe"]).key for row in combination)
        feasible.append((score, seconds, ridges, ids, rows, check))
    if not feasible:
        return None
    minimum = min(item[0] for item in feasible)
    chosen = min((item for item in feasible if item[0] - minimum <= EPSILON),
                 key=lambda item: item[1:4])
    score, seconds, ridges, ids, rows, check = chosen
    return dict(recipes={arm: rows[arm]["recipe"] for arm in ARMS}, calibration=rows,
        matching=check, reference_arm="B", reference_ridge=1., strength=strength,
        members=list(members), mean_novel_auc=score, total_learning_seconds=seconds,
        ridge_tuple=list(ridges), stable_ids=list(ids),
        diagnostics=dict(feasible_tuples=len(feasible),
            complete_candidates={arm: len(pools[arm]) for arm in ARMS},
            incomplete_candidates=dict(sorted(incomplete.items())),
            minimum_mean_novel_auc=minimum, score_tie_epsilon=EPSILON))


def _mismatch(row, reference):
    return max(abs(row[name] - reference[name]) for name in RETENTION)


def _best_proposal(options):
    if not options:
        return None
    minimum = min(option["score"] for option in options)
    return min((option for option in options if option["score"] - minimum <= EPSILON),
               key=lambda option: (-len(option["members"]), option["ids"],
                                   option.get("priority", 0), option["ridge"]))


def ridge_proposal(arm, records, used_ridges, tolerance=.1, members=MEMBERS, *, strength=STRENGTH):
    """Propose an untried log-midpoint or boundary probe with auditable evidence.

    Every comparison uses the intersection of completed endpoint and B1 members.
    Among improving midpoints, minimize predicted worst absolute two-metric
    mismatch; ties prefer fuller evidence, then stable recipe IDs. If none
    improves its better endpoint, explore a measured edge without extrapolating
    a prediction. Failed/pending or explicitly used ridges are not retried;
    partially imported recipes can still be proposed for completion.
    """
    if arm not in ("A", "C", "D", "E"):
        raise ValueError("refinement supports A/C/D/E only; B is fixed at ridge 1")
    tolerance = phase6._number(tolerance, "tolerance", minimum=0)
    members = phase6._members(members)
    strength = phase6._number(strength, "strength", minimum=0)
    used = {phase6._number(ridge, "used ridge", minimum=0, maximum=1000)
            for ridge in used_ridges}
    if any(0 < ridge < 1e-4 for ridge in used):
        raise ValueError("positive used ridge must be at least 1e-4")
    candidates = _candidates(records, members, strength)
    reference_recipe = initial_recipes(strength)["B"]
    reference_group, reference_members, _ = candidates.get(reference_recipe, ({}, (), None))
    if not reference_members:
        return None
    evidence = []
    for recipe, (group, covered, _) in candidates.items():
        if recipe.arm != arm:
            continue
        if len(covered) == len(members) or any(row.get("status") != "completed" for row in group.values()):
            used.add(recipe.ridge)
        common = tuple(member for member in covered if member in reference_members)
        if common:
            evidence.append((recipe, group, common))
    evidence.sort(key=lambda item: (item[0].ridge, item[0].key))
    positive = [item for item in evidence if item[0].ridge > 0]

    def measured(item, covered):
        recipe, group, _ = item
        return phase6._aggregate(group, recipe, covered)

    def target(covered):
        return phase6._aggregate(reference_group, reference_recipe, covered)

    interiors = []
    for left, right in combinations(positive, 2):
        covered = tuple(member for member in left[2] if member in right[2])
        if not covered:
            continue
        low, high = left[0].ridge, right[0].ridge
        # Neighbors are defined on the matched subset, not on unrelated members.
        if any(low < item[0].ridge < high and set(covered).issubset(item[2]) for item in positive):
            continue
        ridge = math.sqrt(low * high)
        if ridge in used or not low < ridge < high:
            continue
        left_row, right_row, reference = measured(left, covered), measured(right, covered), target(covered)
        predicted = {name: math.fsum((left_row[name] / 2, right_row[name] / 2)) for name in RETENTION}
        score = _mismatch(predicted, reference)
        endpoint_score = min(_mismatch(row, reference) for row in (left_row, right_row))
        if score < endpoint_score - EPSILON:
            interiors.append(dict(ridge=ridge, score=score, members=covered,
                ids=(left[0].key, right[0].key), bracket=[low, high],
                predicted=predicted, endpoint_score=endpoint_score,
                reason="log_midpoint_predicted_improvement"))
    chosen = _best_proposal(interiors)
    if chosen is None:
        boundaries = []
        for item in positive:
            recipe, _, covered = item
            row, reference = measured(item, covered), target(covered)
            low_edge = not any(other[0].ridge < recipe.ridge and set(covered).issubset(other[2])
                               for other in positive)
            high_edge = not any(other[0].ridge > recipe.ridge and set(covered).issubset(other[2])
                                for other in positive)
            probes = []
            overprotective = (all(row[name] <= reference[name] + tolerance + EPSILON for name in RETENTION)
                              and any(row[name] < reference[name] - tolerance - EPSILON for name in RETENTION))
            if low_edge and overprotective:
                probes.append((0., "overprotective_lower_boundary_zero_probe", 0))
            if low_edge:
                probes.append((max(1e-4, recipe.ridge / 10), "adjacent_boundary_decade_probe", 1))
            if high_edge:
                probes.append((min(1000., recipe.ridge * 10), "adjacent_boundary_decade_probe", 1))
            for ridge, reason, priority in probes:
                if ridge == recipe.ridge or ridge in used:
                    continue
                boundaries.append(dict(ridge=ridge, score=_mismatch(row, reference), members=covered,
                    ids=(recipe.key,), bracket=sorted([recipe.ridge, ridge]), predicted=None,
                    endpoint_score=_mismatch(row, reference), reason=reason, priority=priority))
        chosen = _best_proposal(boundaries)
    if chosen is None:
        return None
    covered = chosen["members"]
    rows = [dict(ridge=item[0].ridge, stable_id=item[0].key, **measured(item, covered))
            for item in evidence if set(covered).issubset(item[2])]
    proposed_recipe = Recipe(arm, ridge=chosen["ridge"], strength=0. if arm == "A" else strength,
                             process_scale=1e-5 if arm in ("D", "E") else 1e-4)
    existing = candidates.get(proposed_recipe, ({}, (), None))[0]
    return dict(ridge=chosen["ridge"], recipe=proposed_recipe.mapping(),
        reference=target(covered), reference_arm="B", reference_ridge=1.,
        evidence=rows, reason=chosen["reason"], bracket=chosen["bracket"],
        covered_members=list(covered), n_members=len(covered), requested_members=list(members),
        partial_evidence=len(covered) != len(members),
        missing_members=phase6._missing(existing, members), stable_ids=list(chosen["ids"]),
        predicted_retention=chosen["predicted"],
        predicted_worst_mismatch=chosen["score"] if chosen["predicted"] is not None else None,
        endpoint_worst_mismatch=chosen["endpoint_score"],
        tolerance=tolerance, epsilon=EPSILON, used_ridges=sorted(used))
