"""Fixed-cohort integrity checks and replica-level evidence synthesis."""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from mnist_experiment.rotated_mnist.artifacts import _read_json

from .analysis import mean_interval, paired_effects, target_attainment, target_summary
from .artifacts import UnitStore, canonical_hash, file_hash
from .config import canonical_policy
from .trajectory import REQUIRED as TRAJECTORY_REQUIRED


def scalar_metric_rows(conclusions):
    rows = []
    for conclusion in conclusions:
        row = {}
        for field, value in conclusion.items():
            if isinstance(value, dict) and "mean" in value:
                row[field] = value["mean"]
                row.update({f"{field}_{statistic}": result for statistic, result in value.items()
                            if statistic != "mean"})
            else:
                row[field] = value
        rows.append(row)
    return rows


def ensure_closeout(store, paths, development, *, resume=True):
    paths = sorted(set(map(Path, paths)))
    ledger = _read_json(store.root / "final_ledger.json")
    integrity_store = UnitStore(store.root, store.study)
    rows, metrics = [], {}
    for path in paths:
        config = _read_json(path / "config.json")
        if integrity_store.completed(config, TRAJECTORY_REQUIRED) != path:
            raise ValueError("final trajectory is missing or at an incompatible path")
        row = _read_json(path / "summary.json")
        if row["phase"] != "phase5" or row["budget_tier"] != "principal":
            raise ValueError("closeout received a development or wrong-budget trajectory")
        key = (row["environment"], row["replica_index"], row["policy"])
        if key in metrics:
            raise ValueError("duplicate final replica/policy in closeout")
        rows.append(row)
        metrics[key] = _read_json(path / "metrics.json")
    expected = set()
    for env in ledger["environments"]:
        for replica in ledger["replica_indices"]:
            for policy in ledger["policies"]:
                name = canonical_policy(policy)["name"]
                expected.add((env["name"], replica, name))
    if set(metrics) != expected:
        raise ValueError(f"final cohort is incomplete or mismatched: missing {len(expected-set(metrics))}, extra {len(set(metrics)-expected)}")
    unit = store.unit("phase6", "closeout", 0, detail={
        "ledger_hash": canonical_hash(ledger),
        "parents": {str(p.relative_to(store.root)): file_hash(p / "config.json") for p in paths}})
    required = ("summary.json", "paired_effects.json", "factorial_effects.json", "metrics.json", "theory_note.md")
    session = store.begin(unit, required, resume=resume)
    if session is None:
        return store.completed(unit, required)
    effects, factorial, conclusions = [], [], []
    seed, samples = store.study.seed("phase6:bootstrap"), store.study.bootstrap_samples
    for env in ledger["environments"]:
        name = env["name"]
        comparisons = development["comparators"][name]
        subset = [r for r in rows if r["environment"] == name]
        treatment = comparisons["asg"]
        policy_map = {p["name"]: p for p in ledger["policies"]}
        controls = list(dict.fromkeys((comparisons["fixed"], f"depth6_{policy_map[treatment]['ewc']}")))
        for control in controls:
            for metric in ("nll_auc", "retention_nll", "online_seconds", "prequential_nll",
                           "early_25_nll_auc", "early_50_nll_auc"):
                effects.extend(paired_effects(subset, treatment, control, metric,
                    group_fields=("environment",), seed=seed, samples=samples))
        if development.get("promoted_repair"):
            for control in controls:
                effects.extend(paired_effects(subset, development["promoted_repair"]["name"],
                    control, "nll_auc", group_fields=("environment",), seed=seed, samples=samples))
        by_replica = defaultdict(dict)
        for row in subset:
            by_replica[row["replica_index"]][row["policy"]] = row
        for metric in ("nll_auc", "retention_nll", "online_seconds"):
            contrasts = {"historical_ewc": [], "current_gate": [], "interaction": []}
            for values in by_replica.values():
                dd, dl, ld, ll = (values[p][metric] for p in (
                    "asg_diag_diag", "asg_diag_lr", "asg_lr_diag", "asg_lr_lr"))
                contrasts["historical_ewc"].append((ld + ll - dd - dl) / 2)
                contrasts["current_gate"].append((dl + ll - dd - ld) / 2)
                contrasts["interaction"].append(ll - ld - dl + dd)
            for contrast, values in contrasts.items():
                factorial.append({"environment": name, "metric": metric, "contrast": contrast,
                                  **mean_interval(values, seed=seed, samples=samples)})
        for policy in sorted({r["policy"] for r in subset}):
            members = [r for r in subset if r["policy"] == policy]
            targets = {str(t): target_summary([target_attainment(metrics[(name, r["replica_index"], policy)],
                        t, origin=0.) for r in members]) for t in store.study.quality_targets}
            conclusions.append({"environment": name, "policy": policy, "replicas": len(members),
                **{field: mean_interval([r[field] for r in members], seed=seed, samples=samples)
                   for field in ("nll_auc", "retention_change", "online_seconds", "mean_depth",
                                 "mean_active_parameters", "early_25_nll_auc")},
                "retention_admissible_mean": statistics.mean(r["retention_change"] for r in members) <= .02,
                "unresolved_fits": sum(r["unresolved_fits"] for r in members),
                "fit_calls": sum(r["fit_calls"] for r in members),
                "gate_denials": sum(r["gate_denials"] for r in members), "quality_targets": targets})
    summary = {"schema_version": 1, "phase": "phase6", "status": "completed",
        "cohort_complete": True, "unique_trajectories": len(paths),
        "replicas_per_condition": len(ledger["replica_indices"]),
        "nominal_trajectory_assignments": ledger["nominal_trajectory_assignments"],
        "uncalibrated_aliases_disclosed": True,
        "primary_endpoint": "mean exposure-normalized current NLL-AUC",
        "effect_convention": "treatment minus control; negative NLL difference favors treatment",
        "inference": "nominal condition-specific paired 95% bootstrap intervals; no simultaneous coverage claim",
        "success_target": "useful learning at scarce data/compute, not full-network equivalence",
        "comparisons": development["comparators"], "conclusions": conclusions,
        "limitations": ["Synthetic MNIST shifts do not demonstrate an applied deployment benefit.",
            "Source pretraining and historical information are shared resources, not zero-cost evidence.",
            "Depth and ESS do not certify finite-sample adequacy.",
            "Dense small-network matrix realizations and full-score probes do not establish scalable savings.",
            "Development selection and threshold plots remain distinct from the frozen primary endpoint."]}
    session.write_json("summary.json", summary)
    session.write_json("paired_effects.json", effects)
    session.write_json("factorial_effects.json", factorial)
    session.write_json("metrics.json", scalar_metric_rows(conclusions))
    from mnist_experiment.rotated_mnist.artifacts import _atomic_write
    _atomic_write(session.path / "theory_note.md", (
        "# Plan 14: Theory And Evidence\n\n"
        "Schur profiling and the zero-penalty S_B > 2q reduction are algebraic checks. "
        "The optimism trace relies on a locally linear estimator response conditioned on "
        "the historical archive. Information equality, temporal stability, and using the "
        "likelihood Fisher as observed curvature remain empirical approximations.\n\n"
        "Use paired_effects.json for the separately reported fresh conditions, "
        "factorial_effects.json for attribution, and Phase 2/3 artifacts for reference "
        "uncertainty and forecast calibration. A matrix improvement alone cannot establish "
        "better decisions, predictive learning, or practical resource savings. "
        "No top-level mathematical primer has been changed.\n").encode())
    return store.finish(session, required)
