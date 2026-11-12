"""Pure development calibration for Plan 18 Phase 9.

Inputs may be legacy records enriched with policy/namespace/representation, or
compact rows with those identities, replica, recipe (legacy Recipe mapping),
Havg, H100, novel_auc and learning_seconds. Compact rows are already rooted
within members. The caller verifies artifact seals and enforces wall-clock and
proposal allowances; nothing here trains, writes a selection seal or authorizes
confirmation. Fresh evidence is never used for calibration.

select_tuple(records) returns a passing pooled-four choice or None. Its
``recipes`` are legacy Recipe mappings keyed by policy; ``policies`` additionally
bind representation and process_scale. ``require_match=False`` returns the best
measured proposal, explicitly labelled unmatched when necessary. ``members``
can be (0, 1) for the first stage; such choices are never confirmation-ready.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations, product
import math
from numbers import Real

from mnist_experiment.fisher_memory_regression.analysis import flatten_record
from mnist_experiment.fisher_memory_regression.config import Recipe, content_hash

from .analysis import CEILING, POLICIES, STRENGTH, TOLERANCE, validate_record

MEMBERS = (0, 1, 2, 3)
ARMS = ("A", "B", "C", "D", "E")
TUNABLE = ("A", "C", "D", "E")
RETENTION = ("Havg", "H100")
EPSILON = 1e-8
RANGE_EPSILON = 1e-12
PROCESS_SCALES = (1e-5, 1e-4, 1e-3)


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"finite numeric {name} required")
    value = float(value)
    if (not math.isfinite(value) or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"invalid {name}")
    return value


def _members(values):
    values = tuple(values)
    if (not values or any(type(v) is not int or v not in MEMBERS for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("distinct development members from 0,1,2,3 required")
    return tuple(sorted(values))


def _recipe(value, policy=None, process_scale=None):
    if hasattr(value, "mapping"):
        value = value.mapping()
    if not isinstance(value, Mapping):
        raise ValueError("recipe mapping required")
    policy = policy or value.get("policy", value.get("arm"))
    if policy not in POLICIES:
        raise ValueError("explicit Phase 9 policy required with legacy recipes")
    method = "replay_ewc" if policy == "A" else "fd_replay_ewc"
    nested = value.get("recipe", value)
    if hasattr(nested, "mapping"):
        nested = nested.mapping()
    if not isinstance(nested, Mapping):
        raise ValueError("legacy recipe mapping required")
    if "method" in nested:
        recipe = Recipe.from_mapping(dict(nested))
    else:
        recipe = Recipe(method, 32, 2, ridge=_number(nested.get("ridge"), "ridge", minimum=0),
                        strength=_number(nested.get("strength", 0. if policy == "A" else STRENGTH),
                                         "strength", minimum=0))
    if (recipe.method != method or recipe.capacity != 32 or recipe.replay_count != 2
            or recipe.strength != (0. if policy == "A" else STRENGTH)):
        raise ValueError("recipe differs from the fixed method/capacity/strength policy")
    ridge = _number(recipe.ridge, "ridge", minimum=0, maximum=1000)
    representation = value.get("representation", "low_rank_diagonal" if policy == "R" else "diagonal")
    if representation != ("low_rank_diagonal" if policy == "R" else "diagonal"):
        raise ValueError("only R can use low_rank_diagonal")
    scale = value.get("process_scale", 1e-4) if process_scale is None else process_scale
    scale = _number(scale, "process_scale", minimum=0)
    if scale not in PROCESS_SCALES:
        raise ValueError("process scale must be a declared shadow-calibration candidate")
    return dict(policy=policy, representation=representation, process_scale=scale,
                ridge=ridge, strength=recipe.strength, recipe=recipe.mapping())


def recipe_id(recipe, *, policy=None, process_scale=None):
    """Stable treatment identity even when legacy recipe_key is shared."""
    row = _recipe(recipe, policy, process_scale)
    return f"{row['policy']}-{content_hash(row)[:16]}"


def initial_recipes(process_scale=1e-4):
    """Return canonical policy dictionaries with B/R fixed at c=1."""
    return {p: _recipe(dict(policy=p, ridge=math.sqrt(10) if p == "A" else 1.),
                       process_scale=process_scale) for p in POLICIES}


def summarize_metrics(metrics, *, N=100):
    """Verify a complete novel-count clock and root retention within one member.

    Returns Havg=sqrt(mean old MSE at n=1..N)-sqrt(old MSE at n=0),
    H100=sqrt(old endpoint MSE)-sqrt(old initial MSE), and novel_auc/time.
    No improvements are clipped. N>=2 permits separate smoke configurations.
    """
    if type(N) is not int or N < 2 or not isinstance(metrics, list) or len(metrics) != N + 1:
        raise ValueError("complete N+1 metric rows required")
    last_observations, last_seconds = -1, -1.
    for n, row in enumerate(metrics):
        try:
            if type(row["n"]) is not int or row["n"] != n:
                raise ValueError("contiguous n=0,...,N required")
            obs = row["observations"]
            if type(obs) is not int or obs < n or obs <= last_observations:
                raise ValueError("strictly increasing exposure clock required")
            seconds = _number(row["learning_seconds"], "learning_seconds", minimum=0)
            if seconds < last_seconds or (n == 0 and (obs != 0 or seconds != 0)):
                raise ValueError("invalid cumulative learning time/initial exposure")
            for key in ("base_mse", "novel_mse"):
                _number(row[key], key, minimum=0)
            last_observations, last_seconds = obs, seconds
        except (KeyError, TypeError) as error:
            raise ValueError(f"malformed scalar metrics: {error}") from error
    initial = math.sqrt(metrics[0]["base_mse"])
    return dict(Havg=math.sqrt(math.fsum(p["base_mse"] / N for p in metrics[1:])) - initial,
                H100=math.sqrt(metrics[-1]["base_mse"]) - initial,
                novel_auc=math.fsum(p["novel_mse"] / N for p in metrics[1:]),
                learning_seconds=float(metrics[-1]["learning_seconds"]))


def summarize_record(record):
    """Validate and summarize one completed legacy record on its actual N clock."""
    row, complete = validate_record(record)
    if not complete:
        raise ValueError("complete record required for retention summary")
    return summarize_metrics(row["metrics"], N=row["target"])


def _groups(records, members, process_scale):
    groups, pairing = {}, {}
    source = records.values() if isinstance(records, Mapping) else records
    for raw in source:
        try:
            row = flatten_record(raw)
            if row.get("namespace", row.get("cohort")) != "development":
                raise ValueError("development-only calibration evidence required")
            member, policy = row["replica"], row["policy"]
            if type(member) is not int or member not in MEMBERS:
                raise ValueError("development replica must be 0,1,2,3")
            canonical = _recipe(dict(policy=policy, recipe=row["recipe"],
                representation=row.get("representation", "low_rank_diagonal" if policy == "R" else "diagonal"),
                process_scale=row.get("process_scale", process_scale)))
            if policy in ("B", "R") and canonical["ridge"] != 1.:
                raise ValueError("B and R must remain fixed at ridge 1")
            if row.get("valid", True) is not True:
                raise ValueError("invalid candidate evidence cannot be hidden")
            for name in (*RETENTION, "novel_auc", "learning_seconds"):
                if name in row:
                    _number(row[name], name, minimum=0 if name in ("novel_auc", "learning_seconds") else None)
            complete = row.get("status", "completed") in ("completed", "complete")
            if "metrics" in row:
                _, complete = validate_record(raw)
                measured = summarize_metrics(row["metrics"], N=row["target"]) if complete else None
                if measured is not None:
                    for name, value in measured.items():
                        if name in row and not math.isclose(_number(row[name], name), value, rel_tol=1e-9, abs_tol=1e-9):
                            raise ValueError(f"candidate summary disagrees with metrics: {name}")
                reference = pairing.setdefault(member, row)
                for key in ("plan_hash", "pairing_id", "frozen_base_mse", "frozen_novel_mse"):
                    if row.get(key) != reference.get(key):
                        raise ValueError(f"unpaired development candidate {key}")
                count = min(len(row["metrics"]), len(reference["metrics"]))
                if [p["observations"] for p in row["metrics"][:count]] != [
                        p["observations"] for p in reference["metrics"][:count]]:
                    raise ValueError("unpaired development exposure clock")
            else:
                measured = {name: _number(row.get(name), name,
                            minimum=0 if name in ("novel_auc", "learning_seconds") else None)
                            for name in (*RETENTION, "novel_auc", "learning_seconds")} if complete else None
            if member not in members or (policy in ("D", "E") and canonical["process_scale"] != process_scale):
                continue
            key = recipe_id(canonical)
            group = groups.setdefault(key, dict(policy=canonical, members={}))
            if member in group["members"]:
                raise ValueError(f"duplicate development candidate/member: {key}/{member}")
            group["members"][member] = dict(completed=complete, measured=measured)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"malformed development record: {error}") from error
    return groups


def _aggregate(group, members):
    covered = tuple(m for m in members if group["members"].get(m, {}).get("completed"))
    if covered != members:
        raise ValueError(f"missing or incomplete development members: {sorted(set(members) - set(covered))}")
    canonical = group["policy"]
    per_member = {str(m): dict(group["members"][m]["measured"]) for m in members}
    means = {name: math.fsum(v[name] / len(members) for v in per_member.values())
             for name in (*RETENTION, "novel_auc", "learning_seconds")}
    return dict(policy=canonical["policy"], recipe=canonical["recipe"], policy_recipe=canonical,
                candidate_id=recipe_id(canonical), ridge=canonical["ridge"], members=list(members),
                n_members=len(members), per_member=per_member,
                total_learning_seconds=math.fsum(v["learning_seconds"] for v in per_member.values()), **means)


def aggregate(records, recipe, members=MEMBERS, *, policy=None, process_scale=1e-4):
    """Pool a single recipe over requested complete members; never pool RMS first."""
    members = _members(members)
    canonical = _recipe(recipe, policy, process_scale)
    key = recipe_id(canonical)
    groups = _groups(records, members, process_scale)
    if key not in groups:
        raise ValueError("missing development recipe")
    return _aggregate(groups[key], members)


def matching(rows, tolerance=TOLERANCE, *, ceiling=CEILING):
    """Check both pooled A--E ranges and the separate absolute ceiling.

    R must be excluded. A--E can improve over initialization; signed means are
    preserved. Passing individual distances to B does not replace the range.
    """
    tolerance = _number(tolerance, "tolerance", minimum=0)
    ceiling = _number(ceiling, "ceiling", minimum=0)
    if not isinstance(rows, Mapping) or set(rows) != set(ARMS):
        raise ValueError("matching requires exactly A--E; R is excluded")
    means = {p: {name: _number(rows[p].get(name), name) for name in RETENTION} for p in ARMS}
    ranges = {name: max(means[p][name] for p in ARMS) - min(means[p][name] for p in ARMS)
              for name in RETENTION}
    range_passed = all(v <= tolerance + RANGE_EPSILON for v in ranges.values())
    ceiling_passed = all(value <= ceiling + RANGE_EPSILON for row in means.values() for value in row.values())
    return dict(passed=range_passed and ceiling_passed, range_passed=range_passed,
                ceiling_passed=ceiling_passed, ranges=ranges, means=means, tolerance=tolerance,
                ceiling=ceiling, maximum_violation=max(0., *(v - tolerance for v in ranges.values()),
                    *(value - ceiling for row in means.values() for value in row.values())))


def choose_tuple(records, tolerance=TOLERANCE, members=MEMBERS, *, replicas=None,
                 process_scale=1e-4, require_match=True):
    """Choose joint risk, then learning time/stable IDs, never a preferred effect.

    All six policies require requested-member coverage; R is reported but is
    excluded from matching, risk and time ranking. The A--E equally weighted
    risk objective is anchored at its global minimum with 1e-8 tie slack.
    With require_match=False an unmatched proposal minimizes maximum violation
    first; it never grants permission for confirmation. Partial candidate
    coverage is published but cannot enter a tuple.
    """
    if type(require_match) is not bool:
        raise ValueError("require_match must be boolean")
    members = _members(members if replicas is None else replicas)
    process_scale = _number(process_scale, "process_scale", minimum=0)
    groups = _groups(records, members, process_scale)
    pools, incomplete = {p: [] for p in POLICIES}, {}
    for key, group in sorted(groups.items()):
        covered = tuple(m for m in members if group["members"].get(m, {}).get("completed"))
        if covered != members:
            incomplete[key] = sorted(set(members) - set(covered))
        else:
            pools[group["policy"]["policy"]].append(_aggregate(group, members))
    if any(not values for values in pools.values()):
        return None
    candidates = []
    for combination in product(*(pools[p] for p in ARMS)):
        rows = dict(zip(ARMS, combination))
        result = matching(rows, tolerance)
        if require_match and not result["passed"]:
            continue
        score = math.fsum(row["novel_auc"] / len(ARMS) for row in combination)
        seconds = math.fsum(row["total_learning_seconds"] for row in combination)
        ids = tuple(row["candidate_id"] for row in combination)
        candidates.append((result["maximum_violation"], score, seconds, ids, rows, result))
    if not candidates:
        return None
    smallest_violation = min(c[0] for c in candidates)
    candidates = [c for c in candidates if c[0] <= smallest_violation + RANGE_EPSILON]
    minimum_risk = min(c[1] for c in candidates)
    chosen = min((c for c in candidates if c[1] <= minimum_risk + EPSILON), key=lambda c: (c[2], c[3]))
    _, score, seconds, ids, rows, gate = chosen
    reference = pools["R"][0]
    return dict(recipes={p: row["recipe"] for p, row in rows.items()},
        policies={p: row["policy_recipe"] for p, row in rows.items()}, calibration=rows,
        representation_reference=reference, matching=gate, reference_policy="B", reference_ridge=1.,
        process_scale=process_scale, members=list(members), mean_novel_auc=score,
        total_learning_seconds=seconds, stable_ids=list(ids),
        calibration_passed=gate["passed"], selection_ready=require_match and gate["passed"] and members == MEMBERS,
        status="matched" if gate["passed"] else "proposed_unmatched",
        diagnostics=dict(complete_candidates={p: len(pool) for p, pool in pools.items()},
                         incomplete_candidates=incomplete, minimum_mean_novel_auc=minimum_risk,
                         score_tie_epsilon=EPSILON))


select_tuple = choose_tuple


def proposals(ridge, target, response, *, used=(), tolerance=TOLERANCE):
    """Suggest bounded adjacent half/whole decades from a measured response.

    These directions are exploration proposals, never a monotonicity claim.
    ``target`` and ``response`` contain Havg/H100. Zero is offered only when
    the positive lower boundary (1e-4) still overprotects both summaries.
    Empirical bracket interpolation is handled by ridge_proposal().
    """
    ridge = _number(ridge, "ridge", minimum=0, maximum=1000)
    tolerance = _number(tolerance, "tolerance", minimum=0)
    used = {_number(v, "used ridge", minimum=0, maximum=1000) for v in used} | {ridge}
    differences = [_number(response[name], name) - _number(target[name], name) for name in RETENTION]
    if all(abs(v) <= tolerance + RANGE_EPSILON for v in differences):
        return []
    overprotects = all(v <= 0 for v in differences) and min(differences) < -tolerance
    underprotects = all(v >= 0 for v in differences) and max(differences) > tolerance
    directions = (-1, 1) if not (overprotects or underprotects) else ((-1,) if overprotects else (1,))
    values = []
    if overprotects and 0 < ridge <= 1e-4 and 0 not in used:
        values.append(dict(ridge=0., reason="overprotective_positive_lower_boundary", predicted_retention=None))
    for direction in directions:
        for step in (.5, 1.):
            candidate = min(1000., max(1e-4, (ridge if ridge else 1e-4) * 10 ** (direction * step)))
            if candidate in used or any(math.isclose(candidate, row["ridge"], rel_tol=1e-12) for row in values):
                continue
            values.append(dict(ridge=candidate, reason="adjacent_half_decade" if step == .5 else "adjacent_decade",
                               predicted_retention=None))
    return values


def ridge_proposal(policy, records, used_ridges=(), tolerance=TOLERANCE, members=None, *,
                   replicas=None, process_scale=1e-4, stage="first_pair", proposals_used=0):
    """Propose one ridge with explicit measured coverage and bounded allowance.

    stage='first_pair' permits two additional proposals; stage='repair' permits
    one all-four proposal, including extension of an earlier partial candidate.
    The scheduler passes its persisted proposals_used count (attempts, not just
    successful runs). This pure function enforces that limit and does not spend
    it. Missing/incomplete members never substitute for measured evidence.
    """
    if policy not in TUNABLE or stage not in ("first_pair", "repair"):
        raise ValueError("only A/C/D/E ridge proposals in first_pair or repair stage are supported")
    if type(proposals_used) is not int or proposals_used < 0:
        raise ValueError("nonnegative integer proposals_used required")
    members = _members((MEMBERS if stage == "repair" else (0, 1))
                       if members is None and replicas is None else (members if replicas is None else replicas))
    if (stage == "first_pair" and members != (0, 1)) or (stage == "repair" and members != MEMBERS):
        raise ValueError("first_pair uses members 0,1; repair uses all four")
    if proposals_used >= (2 if stage == "first_pair" else 1):
        return None
    tolerance = _number(tolerance, "tolerance", minimum=0)
    groups = _groups(records, members, process_scale)
    reference = next((g for g in groups.values() if g["policy"]["policy"] == "B"), None)
    if reference is None:
        return None
    used = {_number(v, "used ridge", minimum=0, maximum=1000) for v in used_ridges}
    evidence = []
    for key, group in sorted(groups.items()):
        if group["policy"]["policy"] != policy:
            continue
        covered = tuple(m for m in members if group["members"].get(m, {}).get("completed")
                        and reference["members"].get(m, {}).get("completed"))
        if covered:
            evidence.append((key, group, covered))
        used.add(group["policy"]["ridge"])
    options = []

    def mismatch(row, target):
        return max(abs(row[name] - target[name]) for name in RETENTION)

    if stage == "repair":
        for key, group, covered in evidence:
            if covered == members or any(not v["completed"] for v in group["members"].values()):
                continue
            measured, target = _aggregate(group, covered), _aggregate(reference, covered)
            options.append(dict(ridge=group["policy"]["ridge"], score=mismatch(measured, target),
                covered=covered, ids=[key], reason="extend_measured_candidate_to_four",
                predicted_retention=None, extension=True, evidence=[measured], target=target))
    for left, right in combinations(evidence, 2):
        left, right = sorted((left, right), key=lambda item: item[1]["policy"]["ridge"])
        low, high = left[1]["policy"]["ridge"], right[1]["policy"]["ridge"]
        covered = tuple(m for m in left[2] if m in right[2])
        if not covered or low <= 0 or not low < high:
            continue
        if any(low < g["policy"]["ridge"] < high and set(covered).issubset(cov)
               for _, g, cov in evidence):
            continue
        a, b, target = _aggregate(left[1], covered), _aggregate(right[1], covered), _aggregate(reference, covered)
        weights = [.5]
        for name in RETENTION:
            if b[name] != a[name]:
                weight = (target[name] - a[name]) / (b[name] - a[name])
                if .05 <= weight <= .95:
                    weights.append(weight)
        for weight in weights:
            ridge = math.exp((1 - weight) * math.log(low) + weight * math.log(high))
            if any(math.isclose(ridge, value, rel_tol=1e-12) for value in used):
                continue
            predicted = {name: (1 - weight) * a[name] + weight * b[name] for name in RETENTION}
            score = mismatch(predicted, target)
            if score + RANGE_EPSILON >= min(mismatch(a, target), mismatch(b, target)):
                continue
            options.append(dict(ridge=ridge, score=score, covered=covered, ids=[left[0], right[0]],
                reason="empirical_log_interpolation", predicted_retention=predicted,
                extension=False, evidence=[a, b], target=target))
    if not options:
        for key, group, covered in evidence:
            measured, target = _aggregate(group, covered), _aggregate(reference, covered)
            for proposal in proposals(group["policy"]["ridge"], target, measured, used=used, tolerance=tolerance):
                options.append(dict(**proposal, score=mismatch(measured, target), covered=covered,
                                    ids=[key], extension=False, evidence=[measured], target=target))
    if not options:
        return None
    minimum = min(option["score"] for option in options)
    chosen = min((o for o in options if o["score"] <= minimum + EPSILON),
                 key=lambda o: (-len(o["covered"]), o["ids"], o["ridge"]))
    canonical = _recipe(dict(policy=policy, ridge=chosen["ridge"]), process_scale=process_scale)
    covered = chosen.pop("covered")
    return dict(**chosen, policy=policy, recipe=canonical["recipe"], policy_recipe=canonical,
                candidate_id=recipe_id(canonical), members=list(members), covered_members=list(covered),
                missing_members=sorted(set(members) - set(covered)), partial_evidence=covered != members,
                proposals_used=proposals_used, stage=stage, tolerance=tolerance, used_ridges=sorted(used))


def select_process_scale(measurements):
    """Select the shared D/E scale from two development stream diagnostics.

    Each compact measurement contains namespace='development', replica 0/1,
    process_scale and mean_log_score. The latter must already average equally
    over coordinates and post-cold-start groups for that replica. No holdout
    risks, pi values, or method effects participate in this selection.
    """
    groups = {scale: {} for scale in PROCESS_SCALES}
    for row in measurements:
        if row.get("namespace") != "development" or type(row.get("replica")) is not int or row["replica"] not in (0, 1):
            raise ValueError("first two development streams required for process calibration")
        scale = _number(row.get("process_scale"), "process_scale")
        if scale not in groups or row["replica"] in groups[scale]:
            raise ValueError("duplicate or unknown process-scale measurement")
        groups[scale][row["replica"]] = _number(row.get("mean_log_score"), "mean_log_score")
    if any(set(group) != {0, 1} for group in groups.values()):
        raise ValueError("all three process scales need both development streams")
    scores = {scale: math.fsum(group.values()) / 2 for scale, group in groups.items()}
    best = max(scores.values())
    selected = min((scale for scale in PROCESS_SCALES if scores[scale] >= best - EPSILON),
                   key=lambda scale: (abs(math.log10(scale / 1e-4)), scale))
    return dict(process_scale=selected, mean_log_score=scores[selected],
                scores={str(scale): value for scale, value in scores.items()}, members=[0, 1],
                interpretation="composite prequential innovation diagnostic, not a calibrated joint likelihood")
