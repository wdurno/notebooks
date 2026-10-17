"""Closeout contracts on sealed synthetic artifacts; no training or CUDA work."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from mnist_experiment.akaike_score_gate import analysis, artifacts, closeout
from mnist_experiment.akaike_score_gate.artifacts import UnitStore, file_hash
from mnist_experiment.akaike_score_gate.config import Study
from mnist_experiment.akaike_score_gate.trajectory import REQUIRED


REPLICAS = (3, 17)
COEFFICIENTS = {
    "asg_diag_diag": 0., "asg_diag_lr": 2., "asg_lr_diag": -3., "asg_lr_lr": -2.,
    "depth1_lr": 4., "depth6_lr": 1., "asg_uncalibrated": -1.,
}


def read(path, name="summary.json"):
    return json.loads((path / name).read_text())


def seal_trajectory(store, row, *, variant="fixture"):
    unit = store.unit(row["phase"], "trajectory", row["replica_index"],
                      environment=row["environment"], policy=row["policy"],
                      detail={"fixture_variant": variant, "budget_tier": row["budget_tier"]})
    session = store.begin(unit, REQUIRED, resume=False)
    session.write_json("summary.json", row)
    loss = 0.5 if row["replica_index"] == REPLICAS[0] else 2.
    session.write_json("metrics.json", [
        {"step": -1, "observations": 0, "fixed_nll": loss,
         "current_nll": row["nll_auc"], "online_seconds": 0., "charged_seconds": 0.},
        {"step": 0, "observations": 2, "fixed_nll": loss,
         "current_nll": row["nll_auc"], "online_seconds": 1., "charged_seconds": 1.},
    ])
    session.write_json("gates.json", [])
    session.write_json("pr_curves.json", [])
    session.write_torch("pr_curves.pt", [])
    session.write_torch("snapshots.pt", [])
    session.write_torch("final_state.pt", {"synthetic_fixture": True})
    return store.finish(session, REQUIRED)


@pytest.fixture
def cohort_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "source_hashes", lambda: {"fixture.py": "test-closeout-v1"})
    monkeypatch.setattr(artifacts, "runtime_metadata", lambda *args: {"fixture": "cpu-only"})
    counter = 0

    def make(*, environments=("mixture_m2",), include_uncalibrated=False,
             separate_uncalibrated=False, calibrated_c=1., uncalibrated_half_life=8.):
        nonlocal counter
        raw = Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json").mapping()
        study = Study({**raw, "bootstrap_samples": 32, "quality_targets": [1.]})
        root = tmp_path / f"cohort-{counter}"
        counter += 1
        root.mkdir()
        store = UnitStore(root, study)
        common = {"c": calibrated_c, "half_life": 8., "cadence": 1, "rank": 8, "score_samples": 4}
        policies = [
            {**common, "name": f"asg_{ewc}_{gate}", "depth": None, "ewc": ewc, "gate": gate}
            for ewc in ("diag", "lr") for gate in ("diag", "lr")
        ] + [
            {**common, "name": f"depth{depth}_lr", "depth": depth, "ewc": "lr", "gate": None}
            for depth in (1, 6)
        ]
        if include_uncalibrated:
            policies.append({**common, "name": "asg_uncalibrated", "depth": None,
                             "ewc": "lr", "gate": "lr", "c": 1.,
                             "half_life": uncalibrated_half_life})
            if calibrated_c == 1 and uncalibrated_half_life == 8:
                policies[-1]["alias_of"] = "asg_lr_lr"
        ledger = {"schema_version": 1, "environments": [{"name": name} for name in environments],
                  "replica_indices": list(REPLICAS), "policies": policies,
                  "nominal_trajectory_assignments": len(environments) * len(REPLICAS) * len(policies)}
        (root / "final_ledger.json").write_text(json.dumps(ledger))
        development = {"comparators": {name: {"asg": "asg_lr_lr", "fixed": "depth1_lr"}
                                       for name in environments}}
        paths = []
        for env_index, environment in enumerate(environments):
            sign = 1. if env_index == 0 else -2.
            for replica_index, replica in enumerate(REPLICAS):
                base = 100. + 10 * env_index + 100 * replica_index
                for policy in policies:
                    name = policy["name"]
                    if name == "asg_uncalibrated" and not separate_uncalibrated:
                        continue
                    offset = sign * (replica_index + 1) * COEFFICIENTS[name]
                    row = {"phase": "phase5", "budget_tier": "principal", "environment": environment,
                           "replica_index": replica, "policy": name, "nll_auc": base + offset,
                           "retention_nll": base + offset / 2, "online_seconds": 2 * (base + offset),
                           "prequential_nll": base + offset, "early_25_nll_auc": base + offset,
                           "early_50_nll_auc": base + offset, "retention_change": .01,
                           "mean_depth": 2., "mean_active_parameters": 84., "fit_calls": 4,
                           "unresolved_fits": 2 * replica_index, "gate_denials": replica_index}
                    paths.append(seal_trajectory(store, row))
        return SimpleNamespace(store=store, paths=paths, ledger=ledger, development=development)

    return make


def test_incomplete_cohort_is_refused_before_creating_closeout(cohort_factory):
    cohort = cohort_factory()
    with pytest.raises(ValueError, match="incomplete.*missing 1, extra 0"):
        closeout.ensure_closeout(cohort.store, cohort.paths[:-1], cohort.development)
    assert not (cohort.store.root / "phase6").exists()


def test_extra_replica_is_not_silently_added_to_frozen_cohort(cohort_factory):
    cohort = cohort_factory()
    row = {**read(cohort.paths[0]), "replica_index": 99}
    extra = seal_trajectory(cohort.store, row)
    with pytest.raises(ValueError, match="missing 0, extra 1"):
        closeout.ensure_closeout(cohort.store, [*cohort.paths, extra], cohort.development)


@pytest.mark.parametrize("overrides", [{"phase": "phase4"}, {"budget_tier": "strict"}])
def test_wrong_phase_or_budget_cannot_enter_final_cohort(cohort_factory, overrides):
    cohort = cohort_factory()
    replacement = seal_trajectory(cohort.store, {**read(cohort.paths[0]), **overrides}, variant="wrong-cohort")
    with pytest.raises(ValueError, match="development or wrong-budget"):
        closeout.ensure_closeout(cohort.store, [replacement, *cohort.paths[1:]], cohort.development)


def test_two_distinct_artifacts_for_one_replica_policy_are_rejected(cohort_factory):
    cohort = cohort_factory()
    duplicate = seal_trajectory(cohort.store, read(cohort.paths[0]), variant="duplicate")
    with pytest.raises(ValueError, match="duplicate final replica/policy"):
        closeout.ensure_closeout(cohort.store, [*cohort.paths, duplicate], cohort.development)


def test_paired_and_factorial_signs_use_replica_differences_within_each_environment(cohort_factory):
    cohort = cohort_factory(environments=("mixture_m2", "rotation_linear"))
    path = closeout.ensure_closeout(cohort.store, list(reversed(cohort.paths)), cohort.development)
    effects = read(path, "paired_effects.json")
    factorial = read(path, "factorial_effects.json")
    for environment, multiplier in (("mixture_m2", 1.), ("rotation_linear", -2.)):
        for control, differences in (("depth1_lr", {3: -6., 17: -12.}),
                                     ("depth6_lr", {3: -3., 17: -6.})):
            effect = next(row for row in effects if row["environment"] == environment
                          and row["control"] == control and row["metric"] == "nll_auc")
            assert effect["treatment"] == "asg_lr_lr"
            assert effect["effect_convention"] == "treatment minus control"
            assert dict(zip(effect["paired_replicas"], effect["differences"])) == {
                replica: multiplier * value for replica, value in differences.items()}
            assert effect["mean"] == multiplier * sum(differences.values()) / 2
            assert effect["n"] == effect["n_total"] == 2
            assert effect["unpaired_control"] == effect["unpaired_treatment"] == 0
            if multiplier > 0:
                assert effect["ci95_high"] < 0
            else:
                assert effect["ci95_low"] > 0
        for metric, scale in (("nll_auc", 1.), ("retention_nll", .5), ("online_seconds", 2.)):
            for contrast, expected in (("historical_ewc", -5.25), ("current_gate", 2.25), ("interaction", -1.5)):
                effect = next(row for row in factorial if row["environment"] == environment
                              and row["metric"] == metric and row["contrast"] == contrast)
                assert effect["mean"] == expected * multiplier * scale
                assert effect["n"] == effect["n_total"] == 2
    summary = read(path)
    scalar_rows = analysis._rows(path / "metrics.json")
    assert len(scalar_rows) == len(summary["conclusions"])
    for scalar, rich in zip(scalar_rows, summary["conclusions"]):
        for field in ("nll_auc", "retention_change", "online_seconds"):
            assert scalar[field] == rich[field]["mean"]
            assert scalar[f"{field}_ci95_low"] == rich[field]["ci95_low"]
            assert scalar[f"{field}_ci95_high"] == rich[field]["ci95_high"]
    assert summary["cohort_complete"]
    assert summary["replicas_per_condition"] == 2
    assert summary["unique_trajectories"] == len(cohort.paths)
    for conclusion in summary["conclusions"]:
        assert conclusion["replicas"] == 2
        assert conclusion["unresolved_fits"] == 2
        assert conclusion["fit_calls"] == 8
        assert conclusion["gate_denials"] == 1
        assert conclusion["quality_targets"]["1.0"]["replicas"] == 2
        assert conclusion["quality_targets"]["1.0"]["attained"] == 1
        assert conclusion["quality_targets"]["1.0"]["censored"] == 1


def test_identical_uncalibrated_alias_reuses_artifacts_without_double_counting(cohort_factory):
    cohort = cohort_factory(include_uncalibrated=True)
    aliases = [path for path in cohort.paths if read(path)["policy"] == "asg_lr_lr"]
    path = closeout.ensure_closeout(cohort.store, [*cohort.paths, *aliases], cohort.development)
    summary = read(path)
    assert summary["uncalibrated_aliases_disclosed"]
    assert summary["unique_trajectories"] == 12
    assert summary["nominal_trajectory_assignments"] == 14
    assert {row["policy"] for row in summary["conclusions"]} == set(COEFFICIENTS) - {"asg_uncalibrated"}
    assert all(row["replicas"] == row["nll_auc"]["n"] == 2 for row in summary["conclusions"])
    units = cohort.store.limit.units
    checksums = {file.name: file_hash(file) for file in path.iterdir() if file.is_file()}
    assert closeout.ensure_closeout(cohort.store, cohort.paths, cohort.development) == path
    assert cohort.store.limit.units == units
    assert checksums == {file.name: file_hash(file) for file in path.iterdir() if file.is_file()}


@pytest.mark.parametrize("calibrated_c,half_life", [(2., 8.), (1., 4.)])
def test_nonidentical_uncalibrated_policy_is_a_distinct_final_condition(cohort_factory, calibrated_c, half_life):
    cohort = cohort_factory(include_uncalibrated=True, separate_uncalibrated=True,
                            calibrated_c=calibrated_c, uncalibrated_half_life=half_life)
    path = closeout.ensure_closeout(cohort.store, cohort.paths, cohort.development)
    summary = read(path)
    assert summary["unique_trajectories"] == summary["nominal_trajectory_assignments"] == 14
    assert {row["policy"] for row in summary["conclusions"]} == set(COEFFICIENTS)
    assert all(row["replicas"] == 2 for row in summary["conclusions"])
    effects = read(path, "factorial_effects.json")
    assert next(row for row in effects if row["metric"] == "nll_auc" and row["contrast"] == "interaction")["mean"] == -1.5


def test_chosen_full_depth_control_is_not_reported_twice(cohort_factory):
    cohort = cohort_factory()
    development = copy.deepcopy(cohort.development)
    development["comparators"]["mixture_m2"]["fixed"] = "depth6_lr"
    path = closeout.ensure_closeout(cohort.store, cohort.paths, development)
    effects = read(path, "paired_effects.json")
    assert len(effects) == 6
    assert {row["control"] for row in effects} == {"depth6_lr"}


def test_final_integrity_pass_rehashes_even_previously_validated_artifacts(cohort_factory):
    cohort = cohort_factory()
    first = cohort.paths[0]
    assert cohort.store.completed(read(first, "config.json"), REQUIRED) == first
    summary = read(first)
    summary["nll_auc"] += 1
    (first / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="corrupt immutable artifact"):
        closeout.ensure_closeout(cohort.store, cohort.paths, cohort.development)
    assert not (cohort.store.root / "phase6").exists()
