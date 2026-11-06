"""Scalar-only Phase 5E inference for three frozen zero-ridge packages.

Physical condition IDs remain unchanged. The extra ``arm`` identity separates
fixed/.1 from fixed/0. No development calibration or retention matching occurs.
Only complete three-arm replica blocks contribute to any paired contrast.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from itertools import combinations

from mnist_experiment.diagonal_lfu.analysis import AVERAGING, MAX_BOOTSTRAP_DRAWS, _number
from mnist_experiment.kalman_pi.analysis import (
    _contrast, trajectories_hash, validate_trajectories as _validate_base,
)
from mnist_experiment.kalman_pi.retention.analysis import _pair, _summary
from mnist_experiment.kalman_pi.retention.config import PROTOCOL as PARTITION_PROTOCOL
from src.seeding import derive_component_seed

from .config import ARMS, PROTOCOL

SCHEMA = 1
KIND = "plan17-zero-ridge-analysis"
DEFAULT_SEED = 170518
CONTRASTS = (("kalman_zero", "fixed_ridge"), ("fixed_zero", "fixed_ridge"),
             ("kalman_zero", "fixed_zero"))
INITIAL_METRICS = (
    "nine_recall", "old_accuracy", "nine_fpr", "nine_precision", "precision_undefined",
    "nll", "nine_nll", "old_nll", "class_accuracy", "initial_nines",
    "frozen_nine_recall", "frozen_old_accuracy", "q", "effective_size",
    "replacement_mass", "log_replacement_mass",
)


def _selection(selection):
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    if type(selection.get("schema")) is not int or selection["schema"] != SCHEMA:
        raise ValueError("selection requires schema 1")
    if selection.get("protocol") != PROTOCOL:
        raise ValueError("selection has the wrong supplemental protocol")
    if selection.get("frozen") is not True:
        raise ValueError("supplemental design must be frozen before trajectories")
    frozen_at = selection.get("frozen_at")
    if not isinstance(frozen_at, str) or not frozen_at:
        raise ValueError("selection requires its pre-confirmation frozen_at")
    try:
        instant = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("selection.frozen_at must be an ISO timestamp") from error
    if instant.tzinfo is None:
        raise ValueError("selection.frozen_at must include a timezone")
    arms = selection.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("selection.arms must retain exactly the three approved arms")
    for arm, expected in ARMS.items():
        value = arms[arm]
        if (not isinstance(value, dict) or set(value) != set(expected)
                or value["condition"] != expected["condition"]
                or _number(value["ridge"], "selected ridge") != expected["ridge"]):
            raise ValueError(f"selection changed the physical condition/ridge for {arm}")
    n = selection.get("main_replicas")
    if type(n) is not int or not 1 <= n <= 8:
        raise ValueError("selection.main_replicas must be an integer in [1,8]")
    if type(selection.get("stationary_replicas")) is not int or selection["stationary_replicas"] != 0:
        raise ValueError("supplement has no stationary replicas")
    target = selection.get("fresh_trajectory_target")
    if type(target) is not int or target != len(ARMS) * n:
        raise ValueError("selection.fresh_trajectory_target must include every three-arm block")
    factors = selection.get("schedule_factors")
    if (not isinstance(factors, list) or not factors
            or any(_number(f, "schedule factor") not in (1., 1.25) for f in factors)
            or len(set(factors)) != len(factors)):
        raise ValueError("selection.schedule_factors must be distinct existing horizon factors")
    partition = selection.get("data_partition")
    if not isinstance(partition, dict) or partition.get("protocol") != PARTITION_PROTOCOL:
        raise ValueError("selection must preserve the primary data_partition protocol")
    seed = partition.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("data_partition.seed must be an integer in [0,2**63)")
    for split in ("train", "test"):
        expected = derive_component_seed(seed, f"{PARTITION_PROTOCOL}:study-partition:0:{split}")
        actual = partition.get(split + "_seed")
        if type(actual) is not int or actual != expected:
            raise ValueError(f"data_partition.{split}_seed changes the primary role partition")
    return list(range(n)), factors


def validate_trajectories(trajectories, selection):
    """Validate physical packages and shared full grids or accepted prefixes.

    Scalar rows cannot independently prove raw holdout membership. The frozen
    partition seeds and paired asset identities bind this analysis to the
    separately validated, sealed data assets without reopening them.
    """
    intended, factors = _selection(selection)
    _validate_base(trajectories)
    seen = set()
    by_replica = defaultdict(list)
    for t in trajectories:
        arm = t.get("arm")
        if not isinstance(arm, str) or arm not in ARMS:
            raise ValueError("trajectory requires an approved supplemental arm")
        if t["stage"] != "fresh" or t["schedule"] != "main" or t.get("cohort", "fresh") != "fresh":
            raise ValueError("supplemental trajectories must be fresh/main, never development or stationary")
        if type(t["replica"]) is not int or t["replica"] not in intended:
            raise ValueError("trajectory replica is outside the frozen intended replicas")
        key = t["replica"], arm
        if key in seen:
            raise ValueError("duplicate supplemental replica/arm identity")
        seen.add(key)
        expected = ARMS[arm]
        if t["condition"] != expected["condition"] or t["ridge"] != expected["ridge"] or t["c"] != .025:
            raise ValueError(f"unexpected physical condition/c/ridge for {arm}")
        if t.get("anchor_refresh", 0) != 0:
            raise ValueError("supplement preserves the fixed initialization anchor")
        factor = _number(t.get("schedule_factor"), "schedule_factor")
        if factor != factors[t["replica"] % len(factors)]:
            raise ValueError("trajectory schedule_factor differs from its frozen assignment")
        for name in ("stream_hash", "initial_unit", "asset_unit"):
            if not isinstance(t.get(name), str) or not t[name]:
                raise ValueError(f"supplemental trajectories require explicit {name}")
        if t["metrics"] and t["metrics"][0]["arrivals"] != 0:
            raise ValueError("partial trajectories must retain the shared initial checkpoint")
        by_replica[t["replica"]].append(t)
    for block in by_replica.values():
        for a, b in combinations(block, 2):
            for name in ("stream_hash", "initial_unit", "asset_unit", "schedule_factor"):
                if a[name] != b[name]:
                    raise ValueError(f"paired trajectories have mismatched {name}")
            count = min(len(a["metrics"]), len(b["metrics"]))
            if count:
                if a["completed"] and b["completed"]:
                    _pair(a, b)
                else:
                    for full, prefix in ((a, b), (b, a)):
                        if full["completed"] and len(prefix["metrics"]) > len(full["metrics"]):
                            raise ValueError("partial trajectory exceeds its completed partner's grid")
                    _pair({**a, "metrics": a["metrics"][:count]},
                          {**b, "metrics": b["metrics"][:count]})
                for name in INITIAL_METRICS:
                    if a["metrics"][0].get(name) != b["metrics"][0].get(name):
                        raise ValueError(f"paired trajectories have mismatched initial {name}")


def summarize(trajectories, selection, *, bootstrap_draws=2000, seed=DEFAULT_SEED):
    """Return the three predeclared contrasts using shared whole-block draws.

    Completed arms awaiting a third arm remain visible, never paired early.
    Comparisons use arm IDs in the historical condition/control slots; physical
    policy IDs and both ridges are additionally explicit. No learner is loaded.
    """
    if type(bootstrap_draws) is not int:
        raise ValueError("bootstrap_draws must be an integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    draws = max(0, min(bootstrap_draws, MAX_BOOTSTRAP_DRAWS))
    validate_trajectories(trajectories, selection)
    intended = list(range(selection["main_replicas"]))
    complete = sorted((t for t in trajectories if t["completed"]), key=lambda t: t["unit_id"])
    by_arm = defaultdict(dict)
    for t in complete:
        by_arm[t["arm"]][t["replica"]] = t
    eligible = set.intersection(*(set(by_arm[arm]) for arm in ARMS))
    summaries = {t["unit_id"]: {**_summary(t), "arm": t["arm"], "schedule_factor": t["schedule_factor"]}
                 for t in complete}
    comparisons = []
    for treatment, control in CONTRASTS:
        contrast = _contrast([KIND, "fresh", "main"], treatment, control, by_arm,
                             summaries, draws=draws, seed=seed, eligible_replicas=eligible)
        contrast.update(arm=treatment, control_arm=control,
                        treatment_condition=ARMS[treatment]["condition"], control_condition=ARMS[control]["condition"],
                        treatment_ridge=ARMS[treatment]["ridge"], control_ridge=ARMS[control]["ridge"])
        comparisons.append(contrast)
    return {
        "schema": SCHEMA, "kind": KIND, "stage": "fresh", "schedule": "main", "c": .025,
        "averaging": AVERAGING, "trajectories_sha256": trajectories_hash(trajectories),
        "selection": deepcopy(selection), "selection_sha256": trajectories_hash(selection),
        "arms": deepcopy(ARMS), "required_arms": list(ARMS), "intended_replicas": intended,
        "eligible_replicas": sorted(eligible), "n_pairs": len(eligible),
        "complete_units": [t["unit_id"] for t in complete],
        "incomplete_units": sorted(t["unit_id"] for t in trajectories if not t["completed"]),
        "awaiting_full_block_units": [t["unit_id"] for t in complete if t["replica"] not in eligible],
        "replicas": list(summaries.values()), "comparisons": comparisons,
        "counts": {"sealed_trajectories": len(trajectories), "complete": len(complete),
                   "incomplete": len(trajectories) - len(complete),
                   "planned_trajectories": len(ARMS) * len(intended),
                   "complete_blocks": len(eligible), "planned_blocks": len(intended)},
        "bootstrap": {"requested_draws": bootstrap_draws, "draws": draws, "seed": seed,
                      "confidence": .95, "unit": "complete three-arm replica trajectories", "method": "percentile",
                      "coupling": "identical whole-replica draws for all outcomes and contrasts sharing a cohort"},
    }
