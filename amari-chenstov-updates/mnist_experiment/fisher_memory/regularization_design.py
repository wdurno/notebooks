"""Pure Plan 15 Phase 7 protocol, development selection, and paired analysis.

Records are complete trajectory summaries augmented with ``treatment`` (a
Treatment.mapping()), ``rows`` (including event zero), and optionally metadata.
Artifact-style {summary, metadata, rows, completed} records are also accepted.
No helper reads artifacts, evaluates a network, or schedules work.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics

from .config import Treatment
from .metrics import paired_summary
from src.seeding import derive_component_seed


PROTOCOL = "plan15-regularization-v1"
DEV_FAMILY = "regularization_development"
CONF_FAMILY = "regularization_confirmation"
LAMBDA0 = 160.89
STREAMS = ("mixed", "base")
CAVEAT = ("The sign-flip test assumes exchangeable signs under the null, not merely "
          "a mean-zero null. Bootstrap intervals resample whole paired replicas; "
          "secondary comparisons are nominal. Nonsignificant retention harm is not "
          "noninferiority. Fresh stochastic replicas reuse the MNIST benchmark.")


def ledger(cfg):
    """Resolve the fixed design; smoke settings are software checks only."""
    policy = dict(inner_steps=cfg.inner_steps if cfg.smoke else 100,
                  max_evaluations=cfg.max_evaluations if cfg.smoke else 150)
    return dict(schema_version=1, protocol=PROTOCOL, smoke=cfg.smoke,
                development_family=DEV_FAMILY, confirmation_family=CONF_FAMILY,
                development_replicas=1 if cfg.smoke else 4,
                confirmation_replicas=2 if cfg.smoke else 12,
                event_target=2 if cfg.smoke else 100,
                ridge_grid=[.1, .3] if cfg.smoke else [.03, .1, .3, 1.],
                audit_events=[1, 2] if cfg.smoke else [10, 50, 90],
                default_policy=policy,
                repair_policy={key: 2 * value for key, value in policy.items()},
                audit_repair_threshold_count=2 if cfg.smoke else 12,
                audit_relative_gap_threshold=1e-3,
                lambda0=.269 if cfg.smoke else LAMBDA0, retention_margin=.01,
                bootstrap_samples=cfg.bootstrap_samples if cfg.smoke else 2000,
                inference_seed=cfg.seed_for(CONF_FAMILY, 0, "replica_bootstrap"))


def _number(value, name, *, minimum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (minimum is not None and value < minimum)):
        raise ValueError(f"{name}: expected finite number" + (
            f" >= {minimum}" if minimum is not None else ""))
    return float(value)


def _count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: expected integer >= {minimum}")
    return value


def _policy(value):
    if not isinstance(value, dict) or set(value) != {"inner_steps", "max_evaluations"}:
        raise ValueError("policy must specify inner_steps and max_evaluations")
    return {key: _count(value[key], key, 1) for key in ("inner_steps", "max_evaluations")}


def _recipe(value):
    try:
        choice = value if isinstance(value, Treatment) else Treatment(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid Phase 7 treatment mapping") from error
    if (choice.condition not in ("replay_ewc", "finite_difference")
            or (choice.condition == "replay_ewc" and choice.strength != 0)
            or (choice.condition == "finite_difference" and choice.strength <= 0)):
        raise ValueError("Phase 7 requires replay strength zero or positive FD strength")
    for key in ("strength", "ridge_multiplier", "fd_step", "replay_weight"):
        _number(getattr(choice, key), key, minimum=0)
    if choice.fd_step <= 0 or not 0 < choice.replay_weight < 1:
        raise ValueError("invalid Phase 7 FD step or replay weight")
    for key in ("inner_steps", "max_evaluations", "probes"):
        _count(getattr(choice, key), key, 1)
    return choice


def treatment(cfg, condition, ridge, strength, policy):
    return _recipe(Treatment.default(cfg, condition, strength,
                                    ridge_multiplier=ridge, **_policy(policy)))


def decision_id(name):
    return f"regularization-{name}-v1"


def _record(value):
    """Normalize without mutating the caller; reject conflicting identity fields."""
    if not isinstance(value, dict):
        raise ValueError("trajectory record must be an object")
    if "summary" in value:
        summary = value["summary"]
        if summary is None and value.get("completed") is False:
            summary = {}
        if not isinstance(summary, dict):
            raise ValueError("trajectory summary must be an object")
        result = dict(summary)
        for key in ("rows", "metadata", "completed", "unit_id", "treatment", "treatment_id"):
            if key in value:
                if key in result and result[key] != value[key]:
                    raise ValueError(f"conflicting trajectory {key}")
                result[key] = value[key]
    else:
        result = dict(value)
    meta = result.get("metadata", {})
    if not isinstance(meta, dict):
        raise ValueError("trajectory metadata must be an object")
    for key in ("schema_version", "replica", "stream", "condition", "treatment",
                "treatment_id", "event_target", "family", "protocol", "unit_id"):
        if key in meta:
            if key in result and result[key] != meta[key]:
                raise ValueError(f"trajectory metadata {key} disagrees with summary")
            result[key] = meta[key]
    choice = _recipe(result.get("treatment"))
    if result.get("treatment_id", choice.key) != choice.key:
        raise ValueError("trajectory treatment_id disagrees with recipe")
    result.update(treatment=choice.mapping(), treatment_id=choice.key)
    _count(result.get("replica"), "replica")
    if result.get("stream") not in STREAMS:
        raise ValueError("unknown Phase 7 stream")
    return result


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _healthy(record):
    return (record is not None and record.get("completed", True) is True
            and record.get("nonfinite") is False
            and type(record.get("solver_failures")) is int
            and record["solver_failures"] == 0
            and all(_finite(record.get(key)) and record[key] >= 0
                    for key in ("rotated_auc", "base_auc", "active_seconds")))


def build_table(records, recipes, reference, replicas):
    """Require healthy paired streams, but rank exclusively on mixed outcomes.

    ``replicas`` is a count (IDs 0..n-1), or an explicit sequence of replica IDs.
    Invalid/missing numeric aggregates are None, never NaN or partial means.
    """
    replica_ids = list(range(_count(replicas, "replicas", 1))) if type(replicas) is int else list(replicas)
    if not replica_ids or len(set(replica_ids)) != len(replica_ids):
        raise ValueError("development replica IDs must be nonempty and unique")
    for replica in replica_ids:
        _count(replica, "replica")
    choices = [_recipe(value) for value in recipes]
    if len({choice.key for choice in choices}) != len(choices):
        raise ValueError("duplicate development recipe")
    reference = _recipe(reference)
    if reference.condition != "replay_ewc" or reference.ridge_multiplier != .1:
        raise ValueError("development reference must be replay/EWC at c=0.1")
    for choice in choices:
        if any(getattr(choice, key) != getattr(reference, key)
               for key in ("inner_steps", "max_evaluations", "probes", "fd_step", "replay_weight")):
            raise ValueError("development candidate differs from frozen common policy")
    index = {}
    for value in records:
        row = _record(value)
        key = (row["replica"], row["stream"], row["treatment_id"])
        if key in index:
            raise ValueError(f"duplicate development trajectory: {key}")
        index[key] = row
    table = []
    for choice in choices:
        pairs = []
        for replica in sorted(replica_ids):
            candidate = index.get((replica, "mixed", choice.key))
            control = index.get((replica, "mixed", reference.key))
            candidate_base = index.get((replica, "base", choice.key))
            control_base = index.get((replica, "base", reference.key))
            pair = dict(replica=replica, valid=all(_healthy(record) for record in (
                            candidate, control, candidate_base, control_base)),
                        acquisition=None, base_harm=None, online_seconds=None)
            if candidate is not None:
                pair["acquisition"] = candidate.get("rotated_auc") if _finite(candidate.get("rotated_auc")) else None
                pair["online_seconds"] = candidate.get("active_seconds") if _finite(candidate.get("active_seconds")) else None
            if (candidate is not None and control is not None
                    and _finite(candidate.get("base_auc")) and _finite(control.get("base_auc"))):
                pair["base_harm"] = candidate["base_auc"] - control["base_auc"]
            pairs.append(pair)
        row = dict(treatment=choice.mapping(), valid=all(pair["valid"] for pair in pairs), pairs=pairs)
        for output, field in (("mean_acquisition", "acquisition"), ("mean_harm", "base_harm"),
                              ("online_seconds", "online_seconds")):
            values = [pair[field] for pair in pairs]
            row[output] = statistics.fmean(values) if all(_finite(v) for v in values) else None
        table.append(row)
    return table


def select(table, margin):
    """Lowest acquisition NLL among feasible candidates, with exact tie-breaks."""
    _number(margin, "retention margin", minimum=0)
    valid = [row for row in table if row["valid"]
             and all(_finite(row.get(key)) for key in ("mean_acquisition", "mean_harm", "online_seconds"))]
    if not valid:
        raise ValueError("all development candidates are invalid; repair/version before selection")
    feasible = [row for row in valid if row["mean_harm"] <= margin]
    chosen = min(feasible or valid, key=lambda row: (
        row["mean_acquisition"], row["online_seconds"],
        row["treatment"]["ridge_multiplier"], row["treatment"]["strength"]))
    return chosen, "retention_feasible" if feasible else "retention_warning_acquisition_fallback"


def shortlist(table, margin=.01):
    result, statuses = {}, {}
    for name, condition in (("replay", "replay_ewc"), ("fd", "finite_difference")):
        row, status = select([row for row in table if row["treatment"]["condition"] == condition], margin)
        result[name], statuses[name] = dict(row["treatment"]), status
    result.update(ridges=sorted({result[name]["ridge_multiplier"] for name in statuses}),
                  statuses=statuses,
                  warnings=[f"{name}: {status}" for name, status in statuses.items()
                            if status != "retention_feasible"])
    return result


def extension(rowtable_at_ridge, lambda0, margin):
    """Propose a single edge coefficient; the caller seals/adopts it at most once."""
    if _number(lambda0, "lambda0", minimum=0) == 0:
        raise ValueError("lambda0 must be positive")
    choices = [_recipe(row["treatment"]) for row in rowtable_at_ridge]
    if (any(choice.condition != "finite_difference" for choice in choices)
            or len({choice.ridge_multiplier for choice in choices}) > 1):
        raise ValueError("coefficient extension requires FD candidates at one ridge")
    selected, _ = select(rowtable_at_ridge, margin)
    strength = selected["treatment"]["strength"]
    if math.isclose(strength, lambda0 / 3, rel_tol=1e-12):
        return lambda0 / 9
    if math.isclose(strength, 3 * lambda0, rel_tol=1e-12):
        return 9 * lambda0
    return None


def confirmation_ledger(cfg, selection, source):
    """Freeze calibrated recipes, including a deduplicated same-ridge control."""
    replay, fd = (_recipe(selection[key]) for key in ("replay", "fd"))
    policy = _policy(selection["policy"])
    if replay.condition != "replay_ewc" or fd.condition != "finite_difference":
        raise ValueError("selection requires replay and FD recipes")
    for choice in (replay, fd):
        expected = treatment(cfg, choice.condition, choice.ridge_multiplier, choice.strength, policy)
        if choice.mapping() != expected.mapping():
            raise ValueError("selected recipe differs from frozen configuration/policy")
    common = treatment(cfg, "replay_ewc", fd.ridge_multiplier, 0., policy)
    recipes = [replay, fd] + ([common] if common.key != replay.key else [])
    design = ledger(cfg)
    result = dict(schema_version=1, protocol=PROTOCOL, family=CONF_FAMILY, smoke=cfg.smoke,
                  replicas=design["confirmation_replicas"],
                  confirmation_replicas=design["confirmation_replicas"],
                  config_hash=cfg.scientific_hash, event_target=design["event_target"],
                  treatments=[choice.mapping() for choice in recipes], baseline=replay.mapping(),
                  treatment=fd.mapping(), shared_ridge_baseline=common.mapping(), policy=policy,
                  selected_strength=fd.strength, retention_margin=.01, alpha=.05,
                  alternative="greater", statistic="mean paired acquisition gain",
                  bootstrap_samples=design["bootstrap_samples"], inference_seed=design["inference_seed"],
                  angle=30., ramp_batches=2 if cfg.smoke else 40, plateau=.5 if cfg.smoke else .1,
                  selection=selection, source=source,
                  note="Fresh stochastic replicas on reused MNIST; no development replicas or early inference.")
    result["trajectory_ids"] = [
        f"trajectory-{CONF_FAMILY}-{replica:04d}-{stream}-{choice.key}"
        for replica in range(result["replicas"]) for stream in STREAMS for choice in recipes]
    return json.loads(json.dumps(result, allow_nan=False))


def _planned(plan):
    if (not isinstance(plan, dict) or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("protocol") != PROTOCOL or plan.get("family") != CONF_FAMILY
            or type(plan.get("smoke")) is not bool):
        raise ValueError("incompatible Phase 7 confirmation ledger")
    if (type(plan.get("replicas")) is not int or type(plan.get("event_target")) is not int
            or (plan["replicas"], plan["event_target"]) != ((2, 2) if plan["smoke"] else (12, 100))):
        raise ValueError("confirmation requires the fixed 12 x 100 or smoke 2 x 2 cohort")
    if type(plan.get("confirmation_replicas")) is not int or plan["confirmation_replicas"] != plan["replicas"]:
        raise ValueError("confirmation replica count disagrees with frozen cohort")
    _count(plan.get("bootstrap_samples"), "bootstrap_samples", 1)
    _count(plan.get("inference_seed"), "inference_seed")
    if plan["alpha"] != .05 or (not plan["smoke"] and plan["bootstrap_samples"] != 2000):
        raise ValueError("incompatible frozen confirmation inference settings")
    policy = _policy(plan["policy"])
    replay, fd, common = (_recipe(plan[key]) for key in ("baseline", "treatment", "shared_ridge_baseline"))
    if (replay.condition != "replay_ewc" or fd.condition != "finite_difference"
            or common.condition != "replay_ewc" or common.ridge_multiplier != fd.ridge_multiplier):
        raise ValueError("incompatible calibrated or same-ridge confirmation recipes")
    fixed = {"condition", "strength", "ridge_multiplier"}
    for choice in (fd, common):
        if any(value != choice.mapping()[key] for key, value in replay.mapping().items() if key not in fixed):
            raise ValueError("confirmation recipes differ outside ridge and FD coefficient")
    if any(getattr(replay, key) != value for key, value in policy.items()):
        raise ValueError("confirmation recipe differs from frozen policy")
    recipes = [replay, fd] + ([common] if common.key != replay.key else [])
    if plan.get("treatments") != [choice.mapping() for choice in recipes]:
        raise ValueError("confirmation recipe ledger is missing, duplicated, or unplanned")
    expected = {(replica, stream, choice.key): choice for replica in range(plan["replicas"])
                for stream in STREAMS for choice in recipes}
    names = [f"trajectory-{CONF_FAMILY}-{replica:04d}-{stream}-{key}" for replica, stream, key in expected]
    if plan.get("trajectory_ids") != names:
        raise ValueError("confirmation trajectory ledger disagrees with frozen recipes/cohort")
    return recipes, expected


def _validate_metadata(record, plan):
    if "metadata" not in record:
        return
    meta = record["metadata"]
    wanted = dict(schema_version=1, kind="drip_trajectory", phase="7", family=CONF_FAMILY,
                  protocol=PROTOCOL, replica=record["replica"], stream=record["stream"],
                  treatment=record["treatment"], treatment_id=record["treatment_id"],
                  condition=record["treatment"]["condition"], event_target=plan["event_target"])
    source = plan.get("source", {})
    identity = source.get("identity", source)
    for field in ("source_hash", "config_hash"):
        expected = identity.get(field, plan.get(field))
        if expected is not None:
            wanted[field] = expected
    if (type(meta.get("schema_version")) is not int
            or any(meta.get(field) != value for field, value in wanted.items())):
        raise ValueError("confirmation metadata differs from frozen protocol/source/configuration")


def _validate_payload(record, plan):
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("incompatible confirmation summary schema")
    choice = _recipe(record["treatment"])
    for key, value in dict(condition=choice.condition, strength=choice.strength,
                           event_target=plan["event_target"]).items():
        if record.get(key) != value:
            raise ValueError(f"confirmation summary {key} mismatch")
    rows, target = record.get("rows"), plan["event_target"]
    if not isinstance(rows, list) or len(rows) != target + 1:
        raise ValueError("confirmation event grid incomplete")
    for event, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("paired_rotated_event")) is not int or row["paired_rotated_event"] != event:
            raise ValueError("confirmation event grid incomplete or duplicated")
        actual = row.get("actual_rotated_events")
        if type(actual) is not int or actual != (event if record["stream"] == "mixed" else 0):
            raise ValueError("confirmation actual event count mismatch")
        for domain in ("rotated", "base"):
            _number(row.get(f"{domain}_nll"), f"{domain}_nll", minimum=0)
            if _number(row.get(f"{domain}_accuracy"), f"{domain}_accuracy", minimum=0) > 1:
                raise ValueError("confirmation accuracy must lie in [0, 1]")
    for domain in ("rotated", "base"):
        expected = statistics.fmean(row[f"{domain}_nll"] for row in rows[1:])
        if not math.isclose(_number(record.get(f"{domain}_auc"), f"{domain}_auc", minimum=0),
                            expected, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError("confirmation AUC disagrees with post-event metrics")
        for metric in ("nll", "accuracy"):
            key = f"final_{domain}_{metric}"
            if not math.isclose(_number(record.get(key), key, minimum=0), rows[-1][f"{domain}_{metric}"],
                                rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError("confirmation endpoint disagrees with event metrics")
    _number(record.get("active_seconds"), "active_seconds", minimum=0)
    _count(record.get("solver_failures"), "solver_failures")
    _count(record.get("under_solved", 0), "under_solved")
    if type(record.get("nonfinite")) is not bool:
        raise ValueError("confirmation nonfinite diagnostic must be boolean")
    for key in ("closure_calls", "exposure", "batches", "evictions"):
        if key in record:
            _count(record[key], key)
    counts = record.get("per_class_events")
    if counts is not None:
        if not isinstance(counts, dict) or set(counts) != {str(k) for k in range(10)}:
            raise ValueError("invalid per-class confirmation event counts")
        if sum(_count(value, "class event count") for value in counts.values()) != target:
            raise ValueError("per-class event counts disagree with event target")
    # Optional diagnostics cannot smuggle NaNs into otherwise valid summaries.
    json.dumps(record, allow_nan=False)


def _exact_sign_flip(differences):
    observed = math.fsum(differences)
    tolerance = 1e-12 * max(math.fsum(abs(value) for value in differences), 1e-15)
    count = sum(math.fsum(sign * value for sign, value in zip(signs, differences)) >= observed - tolerance
                for signs in itertools.product((-1, 1), repeat=len(differences)))
    return count / 2 ** len(differences)


def _effect(values, inferential, seed, resamples):
    differences = list(values.values())
    mean = statistics.fmean(differences) if differences else None
    deviation = statistics.stdev(differences) if len(differences) > 1 else 0.
    variance = statistics.fmean((value - mean) ** 2 for value in differences) if differences else 0.
    skew = statistics.fmean((value - mean) ** 3 for value in differences) / variance ** 1.5 if variance else None
    result = dict(n_pairs=len(values), mean_gain=mean, ci=None, p_value=None,
                  paired_values=[dict(replica=key, gain=value) for key, value in values.items()],
                  favorable_replicas=sum(value > 0 for value in differences),
                  standardized_effect=mean / deviation if deviation else None,
                  sample_skewness=skew, asymmetry_warning=bool(skew is not None and abs(skew) > 1),
                  seed=seed, resamples=resamples)
    if inferential:
        result["ci"] = paired_summary(values, dict.fromkeys(values, 0.), seed=seed, resamples=resamples)["ci"]
    return result


def analyze(records, ledger):
    """Validate the frozen cohort before releasing any interval or primary test.

    Partial results use complete all-recipe/two-stream replicas, with no replica
    exclusions for solver health. Smoke and failure-flagged cohorts never release
    inference. Absolute references come from each replica's shared event zero.
    ``retention_harm`` is FD minus calibrated replay; positive is harmful.
    """
    recipes, expected = _planned(ledger)
    index, seen = {}, set()
    for value in records:
        row = _record(value)
        key = (row["replica"], row["stream"], row["treatment_id"])
        if key not in expected:
            raise ValueError(f"unplanned confirmation trajectory: {key}")
        if key in seen:
            raise ValueError(f"duplicate confirmation trajectory: {key}")
        seen.add(key)
        for field, wanted in (("family", CONF_FAMILY), ("protocol", PROTOCOL)):
            if field in row and row[field] != wanted:
                raise ValueError(f"foreign confirmation {field}")
        name = f"trajectory-{CONF_FAMILY}-{row['replica']:04d}-{row['stream']}-{row['treatment_id']}"
        if "unit_id" in row and row["unit_id"] != name:
            raise ValueError("confirmation unit_id disagrees with trajectory identity")
        if type(row.get("completed", True)) is not bool:
            raise ValueError("completed must be boolean")
        _validate_metadata(row, ledger)
        if row.get("completed", True):
            _validate_payload(row, ledger)
            index[key] = row
    for replica in range(ledger["replicas"]):
        available = [row for key, row in index.items() if key[0] == replica]
        if not available:
            continue
        initial = available[0]["rows"][0]
        for row in available[1:]:
            if any(not math.isclose(row["rows"][0][key], initial[key], rel_tol=1e-7, abs_tol=1e-8)
                   for key in ("rotated_nll", "base_nll", "rotated_accuracy", "base_accuracy")):
                raise ValueError("confirmation replicas do not share the frozen initial predictions")
        for field in ("asset_id", "plan_hash"):
            values = [row["metadata"][field] for row in available if field in row.get("metadata", {})]
            if values and any(value != values[0] for value in values):
                raise ValueError(f"confirmation replica has unpaired {field}")
    for field in ("source_hash", "config_hash"):
        values = [row["metadata"][field] for row in index.values() if field in row.get("metadata", {})]
        if values and any(value != values[0] for value in values):
            raise ValueError(f"confirmation cohort mixes scientific {field}")
    paired = [replica for replica in range(ledger["replicas"])
              if all((replica, stream, choice.key) in index for stream in STREAMS for choice in recipes)]
    health = dict(sealed_trajectories=len(index),
                  solver_failures=sum(row["solver_failures"] for row in index.values()),
                  nonfinite_trajectories=sum(row["nonfinite"] for row in index.values()),
                  under_solved_updates=sum(row.get("under_solved", 0) for row in index.values()))
    failed = bool(health["solver_failures"] or health["nonfinite_trajectories"])
    health.update(status="failures_recorded" if failed else "ok",
                  optimization_limited=ledger["selection"].get("optimization_limited", False))
    complete = len(index) == len(expected)
    ready = complete and not ledger["smoke"] and not failed
    replay, fd = (_recipe(ledger[key]) for key in ("baseline", "treatment"))
    common = _recipe(ledger["shared_ridge_baseline"])

    def effect(name, function):
        values = {replica: function(replica) for replica in paired}
        return _effect(values, ready, derive_component_seed(ledger["inference_seed"], f"bootstrap:{name}"),
                       ledger["bootstrap_samples"])

    def at(replica, stream, choice):
        return index[(replica, stream, choice.key)]

    def gain(replica, stream, control):
        return at(replica, stream, control)["rotated_auc"] - at(replica, stream, fd)["rotated_auc"]

    primary = effect("primary", lambda r: gain(r, "mixed", replay))
    if ready:
        primary["p_value"] = _exact_sign_flip([row["gain"] for row in primary["paired_values"]])
    same = common.key == replay.key
    matched = dict(primary, p_value=None) if same else effect("matched_ridge", lambda r: gain(r, "mixed", common))
    retention = effect("retention_harm", lambda r: at(r, "mixed", fd)["base_auc"] - at(r, "mixed", replay)["base_auc"])
    matched_retention = dict(retention) if same else effect(
        "matched_ridge_retention_harm", lambda r: at(r, "mixed", fd)["base_auc"] - at(r, "mixed", common)["base_auc"])
    base_only = effect("base_only_gain", lambda r: gain(r, "base", replay))
    interaction = effect("interaction", lambda r: gain(r, "mixed", replay) - gain(r, "base", replay))
    frozen, deterioration, arms = {}, {}, []
    for choice in recipes:
        frozen[choice.key], deterioration[choice.key] = {}, {}
        for stream in STREAMS:
            frozen[choice.key][stream] = effect(f"frozen:{choice.key}:{stream}", lambda r: (
                at(r, stream, choice)["rows"][0]["rotated_nll"] - at(r, stream, choice)["rotated_auc"]))
            deterioration[choice.key][stream] = effect(f"deterioration:{choice.key}:{stream}", lambda r: (
                at(r, stream, choice)["base_auc"] - at(r, stream, choice)["rows"][0]["base_nll"]))
            cohort = [at(replica, stream, choice) for replica in paired]
            arm = dict(stream=stream, condition=choice.condition, treatment=choice.mapping(),
                       treatment_id=choice.key, n_pairs=len(paired))
            for field in ("rotated_auc", "base_auc", "active_seconds", "closure_calls", "exposure",
                          "final_rotated_nll", "final_base_nll", "final_rotated_accuracy", "final_base_accuracy"):
                values = [row[field] for row in cohort if row.get(field) is not None]
                arm[field] = statistics.fmean(values) if len(values) == len(cohort) and values else None
            for domain in ("rotated", "base"):
                values = [row["rows"][0][f"{domain}_nll"] for row in cohort]
                arm[f"initial_{domain}_nll"] = statistics.fmean(values) if values else None
            arm["online_seconds"] = arm["active_seconds"]
            arms.append(arm)
    result = dict(schema_version=1, protocol=PROTOCOL, smoke=ledger["smoke"], complete=complete,
                  inference_ready=ready, completed=len(index), expected=len(expected), n_pairs=len(paired),
                  paired_replicas=paired, expected_replicas=ledger["replicas"], event_target=ledger["event_target"],
                  primary=primary, matched_ridge=matched, matched_ridge_is_primary=same,
                  retention_harm=retention, matched_ridge_retention_harm=matched_retention,
                  base_only_gain=base_only, interaction=interaction, frozen_gain=frozen,
                  base_deterioration=deterioration, arm_summaries=arms, health=health,
                  alpha=ledger["alpha"], bootstrap_samples=ledger["bootstrap_samples"],
                  inference_seed=ledger["inference_seed"], source_lineage=ledger.get("source", {}),
                  inference_caveat=CAVEAT)
    json.dumps(result, allow_nan=False)
    return result
