"""Pure development-only calibration for Plan 18 Phase 6.

Callers verify artifact identities before supplying records. Havg/H100 are
already rooted within each replica; H100 is the inherited endpoint alias.
Nothing here trains, seals a decision, spends a budget, or authorizes a run.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
import math
from numbers import Real

from ..config import ARMS, Recipe


MEMBERS = (0, 1, 2, 3)
STRENGTH = .5691316767656495
EPSILON = 1e-8
RETENTION = ("Havg", "H100")
METRICS = (*RETENTION, "novel_auc", "learning_seconds")


def _number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite numeric data") from error
    if (not math.isfinite(value) or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"invalid {name}")
    return value


def _recipe(value):
    if isinstance(value, Recipe):
        value = value.mapping()
    if not isinstance(value, Mapping) or set(value) != {"arm", "ridge", "strength", "process_scale"}:
        raise ValueError("complete recipe mapping required")
    return Recipe(value["arm"], **{name: _number(value[name], name, minimum=0)
        for name in ("ridge", "strength", "process_scale")})


def _members(members):
    members = tuple(members)
    if (not members or any(type(member) is not int or member < 0 for member in members)
            or len(set(members)) != len(members)):
        raise ValueError("distinct nonnegative integer development members required")
    return tuple(sorted(members))


def initial_recipes(strength=STRENGTH):
    """Return the saved starting tuple as arm -> Recipe (fresh objects)."""
    strength = _number(strength, "strength", minimum=0)
    return {arm: Recipe(arm, ridge=ridge, strength=0. if arm == "A" else strength,
                       process_scale=1e-5 if arm in ("D", "E") else 1e-4)
            for arm, ridge in zip(ARMS, (10., 10., 10., 1., .1))}


def _groups(records, members):
    groups = {}
    for record in records.values() if isinstance(records, Mapping) else records:
        if not isinstance(record, Mapping) or not isinstance(record.get("metadata"), Mapping):
            raise ValueError("record metadata mapping required")
        metadata = record["metadata"]
        if (metadata.get("cohort") != "development"
                or metadata.get("kind", "trajectory") != "trajectory"):
            continue
        member = metadata.get("replica")
        if type(member) is not int or member < 0:
            raise ValueError("development replica must be a nonnegative integer")
        if member not in members:
            continue
        recipe = _recipe(metadata.get("recipe"))
        group = groups.setdefault(recipe, {})
        if member in group:
            raise ValueError(f"duplicate development member {member} for {recipe.key}")
        group[member] = record
    return groups


def _missing(group, members):
    return [member for member in members if group.get(member, {}).get("status") != "completed"]


def _aggregate(group, recipe, members):
    missing = _missing(group, members)
    if missing:
        raise ValueError(f"missing or incomplete development members for {recipe.key}: {missing}")
    per_member = {}
    for member in members:
        summary = group[member].get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError(f"summary mapping required for {recipe.key}, member {member}")
        per_member[str(member)] = {name: _number(summary.get(name), name,
            minimum=0 if name in ("novel_auc", "learning_seconds") else None) for name in METRICS}
    means = {name: math.fsum(row[name] / len(members) for row in per_member.values())
             for name in METRICS}
    total = math.fsum(row["learning_seconds"] for row in per_member.values())
    return dict(recipe=recipe.mapping(), members=list(members), n_members=len(members),
                per_member=per_member, total_learning_seconds=total, **means)


def aggregate(records, recipe, members=MEMBERS):
    """Require exactly one completed development record per requested member.

    Accept an iterable of records or a unit-ID -> record mapping. Scalar fields
    are member means; total_learning_seconds additionally preserves the sum.
    Signed retention improvements are valid. Missing/invalid summaries raise.
    """
    recipe, members = _recipe(recipe), _members(members)
    return _aggregate(_groups(records, members).get(recipe, {}), recipe, members)


def matching(rows, tolerance=.1):
    """Check both full five-arm ranges, without rounding, with 1e-8 slack.

    rows is an arm -> summary mapping or an iterable of aggregate() outputs.
    """
    tolerance = _number(tolerance, "tolerance", minimum=0)
    if not isinstance(rows, Mapping):
        by_arm = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("arm summary mapping required")
            arm = _recipe(row.get("recipe")).arm
            if arm in by_arm:
                raise ValueError(f"duplicate matching arm {arm}")
            by_arm[arm] = row
        rows = by_arm
    if set(rows) != set(ARMS):
        raise ValueError("matching requires all five arms A--E")
    means = {}
    for arm in ARMS:
        row = rows[arm]
        if not isinstance(row, Mapping):
            raise ValueError("arm summary mapping required")
        if "recipe" in row and _recipe(row["recipe"]).arm != arm:
            raise ValueError("summary recipe arm differs from matching arm")
        means[arm] = {name: _number(row.get(name), name) for name in RETENTION}
    ranges = {name: max(means[arm][name] for arm in ARMS)
              - min(means[arm][name] for arm in ARMS) for name in RETENTION}
    return dict(passed=all(value <= tolerance + EPSILON for value in ranges.values()),
                ranges=ranges, means=means, tolerance=tolerance, epsilon=EPSILON)


def _admissible(recipe, reference_ridge=None, strength=STRENGTH):
    start = initial_recipes(strength)[recipe.arm]
    return (0 <= recipe.ridge <= 1000 and recipe.strength == start.strength
            and recipe.process_scale == start.process_scale
            and (recipe.arm != "A" or reference_ridge is None or recipe.ridge == reference_ridge))


def select_tuple(records, reference_ridge=10., tolerance=.1, members=MEMBERS, *, strength=STRENGTH):
    """Select a complete, jointly matched tuple, or None (never unmatched).

    A is fixed at reference_ridge. Only the declared strength/process policy is
    eligible. Incomplete candidates cannot be selected; malformed completed
    summaries raise. Score ties are measured against the global minimum, not
    rounded bins or a nontransitive pairwise comparator. Time/ridges/Recipe.key
    then break ties exactly. strength overrides the production policy only for
    explicit smoke settings. Fallback-reference order belongs to the scheduler.
    """
    reference_ridge = _number(reference_ridge, "reference_ridge", minimum=0, maximum=1000)
    tolerance, members = _number(tolerance, "tolerance", minimum=0), _members(members)
    strength = _number(strength, "strength", minimum=0)
    pools = {arm: [] for arm in ARMS}
    incomplete = {}
    for recipe, group in _groups(records, members).items():
        if not _admissible(recipe, reference_ridge, strength):
            continue
        # Validate all executed summaries, even when another member is missing.
        completed = tuple(member for member in members
                          if group.get(member, {}).get("status") == "completed")
        if completed:
            row = _aggregate(group, recipe, completed)
        missing = _missing(group, members)
        if missing:
            incomplete[recipe.key] = missing
        else:
            pools[recipe.arm].append(row)
    if any(not pool for pool in pools.values()):
        return None
    feasible = []
    for combination in product(*(sorted(pools[arm], key=lambda row: _recipe(row["recipe"]).key)
                                 for arm in ARMS)):
        rows = dict(zip(ARMS, combination))
        check = matching(rows, tolerance)
        if not check["passed"]:
            continue
        score = math.fsum(row["novel_auc"] / len(ARMS) for row in combination)
        seconds = math.fsum(row["total_learning_seconds"] for row in combination)
        ridges = tuple(row["recipe"]["ridge"] for row in combination)
        ids = tuple(_recipe(row["recipe"]).key for row in combination)
        feasible.append((score, seconds, ridges, ids, rows, check))
    if not feasible:
        return None
    minimum = min(item[0] for item in feasible)
    chosen = min((item for item in feasible if item[0] - minimum <= EPSILON),
                 key=lambda item: item[1:4])
    score, seconds, ridges, ids, rows, check = chosen
    return dict(recipes={arm: rows[arm]["recipe"] for arm in ARMS}, calibration=rows,
        matching=check, reference_ridge=reference_ridge, strength=strength, members=list(members),
        mean_novel_auc=score, total_learning_seconds=seconds, ridge_tuple=list(ridges),
        stable_ids=list(ids), diagnostics=dict(feasible_tuples=len(feasible),
            complete_candidates={arm: len(pools[arm]) for arm in ARMS},
            incomplete_candidates=dict(sorted(incomplete.items())),
            minimum_mean_novel_auc=minimum, score_tie_epsilon=EPSILON))


def ridge_proposal(arm, records, selected_recipe, reference_summary, used_ridges,
                   tolerance=.1, members=MEMBERS, *, strength=STRENGTH):
    """Return a ridge proposal with measured evidence, or None.

    Uses only complete development candidates under the fixed policy. Search
    neighboring observed ridges for a two-statistic bracket, then explore one
    adjacent boundary decade. Boundary priority uses observed distance, not an
    assumed globally monotone ridge response. This is not a performance estimate
    or a candidate allowance; the scheduler admits at most two recipes per arm.
    """
    selected_recipe, members = _recipe(selected_recipe), _members(members)
    strength = _number(strength, "strength", minimum=0)
    if (arm not in ARMS[1:] or selected_recipe.arm != arm
            or not _admissible(selected_recipe, strength=strength)):
        raise ValueError("refinement requires a fixed-policy B--E recipe for the requested arm")
    tolerance = _number(tolerance, "tolerance", minimum=0)
    if not isinstance(reference_summary, Mapping):
        raise ValueError("reference summary mapping required")
    target = {name: _number(reference_summary.get(name), name) for name in RETENTION}
    used = {_number(ridge, "used ridge", minimum=0, maximum=1000) for ridge in used_ridges}
    evidence = []
    for recipe, group in _groups(records, members).items():
        if recipe.arm != arm or not _admissible(recipe, strength=strength):
            continue
        used.add(recipe.ridge)
        if not _missing(group, members):
            row = _aggregate(group, recipe, members)
            evidence.append(dict(ridge=recipe.ridge, **{name: row[name] for name in RETENTION}))
    evidence.sort(key=lambda row: row["ridge"])
    if not evidence:
        return None

    def proposal(ridge, reason, bracket):
        return dict(ridge=ridge, reason=reason, bracket=bracket, evidence=evidence,
                    reference=target, tolerance=tolerance, epsilon=EPSILON,
                    used_ridges=sorted(used))

    positive = [row for row in evidence if row["ridge"] > 0]
    brackets = [(left, right) for left, right in zip(positive, positive[1:])
        if all(min(left[name], right[name]) <= target[name] <= max(left[name], right[name])
               for name in RETENTION)
        and any(left[name] != right[name] for name in RETENTION)]
    brackets.sort(key=lambda pair: (
        not pair[0]["ridge"] <= selected_recipe.ridge <= pair[1]["ridge"],
        math.log(pair[1]["ridge"]) - math.log(pair[0]["ridge"]), pair[0]["ridge"]))
    for left, right in brackets:
        low, high = left["ridge"], right["ridge"]
        midpoint = math.exp((math.log(low) + math.log(high)) / 2)
        if low < midpoint < high and midpoint not in used:
            return proposal(midpoint, "observed_two_statistic_bracket", [low, high])
    if not positive:
        return None
    low, high = positive[0], positive[-1]
    if (low["ridge"] <= .1 and 0. not in used
            and all(low[name] < target[name] - tolerance - EPSILON for name in RETENTION)):
        return proposal(0., "overprotective_lower_boundary_zero_probe", [0., low["ridge"]])
    boundaries = [(low, max(0., low["ridge"] / 10)), (high, min(1000., high["ridge"] * 10))]
    boundaries.sort(key=lambda pair: (
        max(abs(pair[0][name] - target[name]) for name in RETENTION),
        pair[1] < pair[0]["ridge"], pair[1]))
    for row, ridge in boundaries:
        if ridge not in used and ridge != row["ridge"]:
            return proposal(ridge, "adjacent_boundary_decade_probe", sorted([row["ridge"], ridge]))
    return None


def propose_ridge(arm, records, selected_recipe, reference_summary, used_ridges,
                  tolerance=.1, *, strength=STRENGTH):
    """Scalar convenience API; ridge_proposal() also returns auditable reasons."""
    proposal = ridge_proposal(arm, records, selected_recipe, reference_summary, used_ridges,
                              tolerance, strength=strength)
    return None if proposal is None else proposal["ridge"]
