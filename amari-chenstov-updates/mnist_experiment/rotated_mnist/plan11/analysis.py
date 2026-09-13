"""Frozen Plan 11 development selection and precision gate."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import chi2

from ..artifacts import RotatedArtifactError, _read_json, _write_json
from .artifacts import UnitStore
from .config import ANCHORS, GAINS, Policy
from .run import TRAJECTORY_REQUIRED


def freeze_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise RotatedArtifactError(f"frozen Plan 11 decision differs: {path}")
        return
    _write_json(path, value)


def _summary(store: UnitStore, index: int, schedule: str, policy: Policy) -> dict[str, Any]:
    unit = store.unit("development", index, schedule, policy.mapping())
    path = store.completed(unit, TRAJECTORY_REQUIRED)
    if path is None:
        raise RotatedArtifactError(f"Plan 11 development is incomplete: {index}, {schedule}, {policy.name}")
    return _read_json(path / "summary.json")


def _mean(store: UnitStore, schedule: str, policy: Policy, field: str) -> float:
    return statistics.fmean(
        _summary(store, index, schedule, policy)[field] for index in range(1, 13)
    )


def linear_selection(store: UnitStore) -> dict[str, Any]:
    fixed = {anchor: Policy(anchor, 0.0) for anchor in ANCHORS}
    fixed_nll = {anchor: _mean(store, "linear", policy, "environment_nll_auc") for anchor, policy in fixed.items()}
    fixed_acc = {anchor: _mean(store, "linear", policy, "environment_accuracy_auc") for anchor, policy in fixed.items()}
    grid = []
    anchor_winners = []
    for anchor in ANCHORS:
        eligible = []
        for gain in GAINS:
            policy = Policy(anchor, gain)
            nll = _mean(store, "linear", policy, "environment_nll_auc")
            accuracy = _mean(store, "linear", policy, "environment_accuracy_auc")
            fallback = _mean(store, "linear", policy, "unsupported_scale_fallback_fraction")
            allowed = (
                gain not in (0.0, 1.0)
                and fallback <= 0.10
                and nll <= fixed_nll[anchor] + 0.10
                and accuracy >= fixed_acc[anchor] - 0.02
            )
            grid.append({"anchor": anchor, "gain": gain, "nll_auc": nll, "accuracy_auc": accuracy, "fallback_fraction": fallback, "eligible": allowed})
            if allowed:
                eligible.append((policy, nll))
        if eligible:
            best = min(nll for _, nll in eligible)
            winner = min(
                (policy for policy, nll in eligible if nll <= best + 0.01),
                key=lambda policy: policy.gain,
            )
            anchor_winners.append((winner, _mean(store, "linear", winner, "environment_nll_auc")))
    finalists = [policy.mapping() for policy, _ in sorted(anchor_winners, key=lambda item: (item[1], item[0].anchor))[:2]]
    result = {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "grid": grid,
        "fixed_linear_nll": {str(key): value for key, value in fixed_nll.items()},
        "fixed_linear_accuracy": {str(key): value for key, value in fixed_acc.items()},
        "finalists": finalists,
        "stop_reason": None if finalists else "no_eligible_interior_policy",
    }
    freeze_json(store.root / "decisions" / "linear_selection.json", result)
    return result


def sigmoid_policies(selection: dict[str, Any]) -> tuple[Policy, ...]:
    policies = {Policy(anchor, 0.0) for anchor in ANCHORS}
    for value in selection["finalists"]:
        policy = Policy(**value)
        policies.add(policy)
        policies.add(Policy(policy.anchor, 1.0))
    return tuple(sorted(policies, key=lambda policy: (policy.anchor, policy.gain)))


def development_decision(store: UnitStore, benchmark_seconds_per_trajectory: float) -> dict[str, Any]:
    selection = linear_selection(store)
    if not selection["finalists"]:
        result = {"status": "stopped", "reason": selection["stop_reason"], "study_hash": store.study.config_hash}
        freeze_json(store.root / "decisions" / "development_decision.json", result)
        return result
    fixed_sig_nll = min(_mean(store, "sigmoid", Policy(c, 0.0), "environment_nll_auc") for c in ANCHORS)
    fixed_sig_acc = max(_mean(store, "sigmoid", Policy(c, 0.0), "environment_accuracy_auc") for c in ANCHORS)
    vetoes = []
    survivors = []
    for item in selection["finalists"]:
        policy = Policy(**item)
        nll = _mean(store, "sigmoid", policy, "environment_nll_auc")
        acc = _mean(store, "sigmoid", policy, "environment_accuracy_auc")
        veto = nll > fixed_sig_nll + 0.10 or acc < fixed_sig_acc - 0.02
        vetoes.append({**item, "sigmoid_nll_auc": nll, "sigmoid_accuracy_auc": acc, "vetoed": veto})
        if not veto:
            survivors.append(policy)
    comparator = min(ANCHORS, key=lambda c: (selection["fixed_linear_nll"][str(c)], c))
    chosen = min(
        survivors,
        key=lambda policy: (_mean(store, "linear", policy, "environment_nll_auc"), policy.gain, policy.anchor),
    ) if survivors else None
    reason = None
    if chosen is None:
        reason = "all_finalists_vetoed"
    elif _mean(store, "linear", chosen, "environment_nll_auc") > selection["fixed_linear_nll"][str(comparator)] + 0.02:
        reason = "selected_blend_trails_best_fixed"
    standard_deviation_upper = None
    n_required = None
    projected_hours = None
    if reason is None:
        upper_bounds = []
        for policy in survivors:
            differences = [
                _summary(store, index, "linear", Policy(comparator, 0.0))["environment_nll_auc"]
                - _summary(store, index, "linear", policy)["environment_nll_auc"]
                for index in range(1, 13)
            ]
            upper_bounds.append(statistics.stdev(differences) * math.sqrt(11 / chi2.ppf(0.10, 11)))
        standard_deviation_upper = max(upper_bounds)
        raw_n = math.ceil((1.959963984540054 + 0.8416212335729143) ** 2 * standard_deviation_upper ** 2 / 0.02 ** 2)
        n_required = max(32, 16 * math.ceil(raw_n / 16))
        projected_hours = n_required * 10 * benchmark_seconds_per_trajectory / 3600
        if n_required > 256:
            reason = "precision_exceeds_256_replicas"
    result = {
        "study_hash": store.study.config_hash,
        "status": "candidate_ready_for_cost_review" if reason is None else "stopped",
        "reason": reason,
        "selected_policy": None if chosen is None else chosen.mapping(),
        "fixed_comparator": comparator,
        "sigmoid_vetoes": vetoes,
        "standard_deviation_upper": standard_deviation_upper,
        "n_required": n_required,
        "projected_confirmation_hours": projected_hours,
        "benchmark_seconds_per_trajectory": benchmark_seconds_per_trajectory,
        "confirmation_authorized": False,
    }
    freeze_json(store.root / "decisions" / "development_decision.json", result)
    return result
