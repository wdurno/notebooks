"""No-training tests of production selection, isolation, and immutable resume."""

import copy
import dataclasses
import json
import math
from pathlib import Path

import pytest

from mnist_experiment.fisher_memory.artifacts import Store, ImmutableError
from mnist_experiment.fisher_memory.config import Config, METHODS, Treatment
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested
from mnist_experiment.fisher_memory.scheduler import (
    Scheduler, SchedulerContractError, select_candidate, validate_calibration,
    validate_summary, run_study,
)


class MemoryStore:
    """Tiny on-disk JSON facade; real Store integration has its own test below."""

    def __init__(self, root):
        self.root = Path(root)
        self.records = {}
        self.reconciliations = 0

    def check(self):
        pass

    def unit_path(self, unit):
        return self.root / unit

    def is_complete(self, unit):
        return self.records.get(unit, {}).get("complete", False)

    def begin(self, unit, metadata):
        if unit in self.records:
            assert self.records[unit]["metadata"] == metadata
            assert not self.is_complete(unit)
        else:
            self.records[unit] = {"metadata": metadata, "complete": False}
        self.unit_path(unit).mkdir(parents=True, exist_ok=True)

    def write_json(self, unit, name, value):
        assert not self.is_complete(unit)
        (self.unit_path(unit) / name).write_text(json.dumps(value, allow_nan=False))

    def complete(self, unit):
        assert not self.is_complete(unit)
        self.records[unit]["complete"] = True

    def reconcile(self):
        self.reconciliations += 1

    def iter_units(self, completed_only=False):
        for key, value in list(self.records.items()):
            if not completed_only or value["complete"]:
                yield {"unit_id": key, **value}

    def storage_status(self):
        return {"used_bytes": sum(p.stat().st_size for p in self.root.rglob("*.json")), "external_bytes": 0}


def summary(current=1., historical=.7, cost=12., **extra):
    return {"current_auc": current, "historical_auc": historical,
            "active_seconds": cost, "fit_seconds": cost - 2., "nonfinite": False,
            "acquisition_ok": True, "retention_ok": True, "solver_ok": True,
            "nine_precision": .6, "nine_recall": .5, "realized_nines": 10, **extra}


def calibration():
    return {"methods": {name: {"valid": True, "unit_gradient_ratio": 1.,
                                "gradient_ratio_ltm": .8, "gradient_ratio_stm": .6,
                                "probes": 2, "fd_step": .01}
                        for name in (*METHODS, "distillation")}}


class FakeRunner:
    def __init__(self, tmp_path, *, environments=None, confirmation=2):
        self.cfg = Config(device="cpu", environments=environments or Config().environments,
                          phase1_replicas=1, pilot_screen_replicas=2, pilot_validate_replicas=2,
                          confirmation_replicas=confirmation, bootstrap_samples=20)
        self.store = MemoryStore(tmp_path)
        self.calls = []
        self.calibration_calls = []
        self.transform = lambda result, family, phase, env, replica, treatment: result
        self.calibration_transform = lambda result, env, treatment: result
        self.confirmation_before = None

    def run_trajectory(self, family, phase, env, replica, treatment):
        if phase == "phase3" and self.confirmation_before:
            self.confirmation_before(self)
        self.calls.append((family, phase, env, replica, treatment))
        winner = {"rotation_linear": "jvp", "rotation_sigmoid": "finite_difference",
                  "digit9_mixture": "directional"}[env]
        cost = {"replay_ewc": 12., "jvp": 22., "finite_difference": 32.,
                "directional": 42., "distillation": 17., "replay_compute_matched": 42.}[treatment.condition]
        gain = 0.
        if treatment.condition in (*METHODS, "distillation"):
            peak = .1 if treatment.condition == winner else .03
            gain = peak - .05 * abs(math.log10(treatment.strength / .1))
        value = summary(current=1. + .01 * replica - gain, historical=.7 + .01 * replica,
                        cost=cost, realized_nines=10 + replica)
        return self.transform(value, family, phase, env, replica, treatment)

    def calibrate(self, family, env, replica, baseline_treatment):
        self.calibration_calls.append((family, env, replica, baseline_treatment))
        return self.calibration_transform(calibration(), env, baseline_treatment)


class Progress:
    def __init__(self):
        self.state = {}
        self.history = []

    def __call__(self, force=False):
        self.history.append(copy.deepcopy(self.state))


def test_full_production_fixed_twelve_round_robin_and_no_outcome_stopping(tmp_path):
    runner = FakeRunner(tmp_path, confirmation=12)
    def before_confirm(runner):
        frozen = [r for r in runner.store.records.values()
                  if r["metadata"]["name"] == "confirmation-ledger" and r["complete"]]
        assert len(frozen) == 1
    runner.confirmation_before = before_confirm
    progress = Progress()
    result = run_study(runner, progress=progress)
    calls = [call for call in runner.calls if call[1] == "phase3"]
    assert len(calls) == 12 * 3 * 4
    for replica in range(12):
        block = calls[replica * 12:(replica + 1) * 12]
        assert [item[2] for item in block] == [env for env in runner.cfg.environments for _ in range(4)]
        assert {item[3] for item in block} == {replica}
        for i, env in enumerate(runner.cfg.environments):
            group = block[4 * i:4 * (i + 1)]
            expected = {"rotation_linear": "jvp", "rotation_sigmoid": "finite_difference",
                        "digit9_mixture": "directional"}[env]
            assert [g[4].condition for g in group] == ["replay_ewc", expected, "distillation", "replay_compute_matched"]
    assert result["confirmation_complete"] and result["release_inference"]
    assert not result["optional_extensions_started"]
    assert set(result["environments"]) == set(runner.cfg.environments)
    assert progress.state["confirmation_replicas"] == 12
    assert len(progress.state["calibration"]) == 12
    assert progress.state["remaining_seconds"] == 0
    assert any(row.get("phase") == "phase3" and row.get("remaining_seconds", 0) > 0 for row in progress.history)
    assert progress.state["forecast_bytes"] > 0
    assert runner.store.reconciliations == 1


def test_validation_coefficients_are_screen_frozen_and_families_disjoint(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    scheduler = Scheduler(runner)
    result = scheduler.run(through_phase=2)
    screen = [call for call in runner.calls if call[0] == "pilot_screen"]
    validation = [call for call in runner.calls if call[0] == "pilot_validation"]
    assert validation and screen
    for call in validation:
        assert call[4].mapping() in [item[4].mapping() for item in screen]
    selected = result["environments"]["rotation_linear"]
    assert selected["winner"]["strength"] == .1
    assert selected["compute_match"]["fit_time_multiplier"] == 2.
    assert selected["compute_match"]["treatment"]["max_evaluations"] == 150
    assert {call[0] for call in runner.calls} == {"policy_development", "pilot_screen", "pilot_validation"}


def test_resume_skips_callbacks_reports_and_preserves_decisions(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    refreshes = []
    scheduler = Scheduler(runner, refresh=lambda: refreshes.append(1))
    first = scheduler.run()
    files = {p: p.read_bytes() for p in tmp_path.rglob("result.json")}
    calls, calibration_calls = len(runner.calls), len(runner.calibration_calls)
    refreshed = len(refreshes)
    repeated = scheduler.run()
    assert repeated == first
    assert len(runner.calls) == calls and len(runner.calibration_calls) == calibration_calls
    assert len(refreshes) == refreshed + 1  # One final report, no historical replay.
    assert all(path.read_bytes() == value for path, value in files.items())


def test_halt_and_resume_mid_confirmation_keeps_fixed_cohort(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",), confirmation=3)
    def stop_once(runner):
        if sum(call[1] == "phase3" for call in runner.calls) == 5:
            runner.confirmation_before = None
            raise StopRequested("borrow node")
    runner.confirmation_before = stop_once
    scheduler = Scheduler(runner)
    with pytest.raises(StopRequested):
        scheduler.run()
    frozen = next(p for p in tmp_path.rglob("result.json") if "confirmation-ledger" in str(p))
    frozen_bytes = frozen.read_bytes()
    result = Scheduler(runner).run()
    confirmation = [call for call in runner.calls if call[1] == "phase3"]
    assert len(confirmation) == 12
    assert len({(c[2], c[3], c[4].key) for c in confirmation}) == 12
    assert frozen.read_bytes() == frozen_bytes
    assert result["completed_units"] == 12


def test_max_units_counts_new_trajectories_and_does_not_advance_after_limit(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    with pytest.raises(ResourceLimit):
        run_study(runner, max_units=2)
    assert len(runner.calls) == 2
    with pytest.raises(ResourceLimit):
        run_study(runner, max_units=1)
    assert len(runner.calls) == 3
    result = run_study(runner)
    assert result["confirmation_complete"]


def test_common_policy_failure_calibrates_bounded_grid_and_freezes_policy(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    def change(value, family, phase, env, replica, treatment):
        if family == "policy_development":
            value["acquisition_ok"] = treatment.replay_weight == .25 and treatment.ridge_multiplier == .01
        return value
    runner.transform = change
    result = Scheduler(runner).run(through_phase=2)
    policy_calls = [c for c in runner.calls if c[0] == "policy_development"]
    assert len(policy_calls) == 4
    chosen = result["environments"]["rotation_linear"]["baseline"]
    assert chosen["replay_weight"] == .25 and chosen["ridge_multiplier"] == .01
    assert all(c[4].replay_weight == .25 and c[4].ridge_multiplier == .01
               for c in runner.calls if c[0] != "policy_development")


def test_solver_ceiling_repair_precedes_policy_search(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    def change(value, family, phase, env, replica, treatment):
        value["solver_ok"] = treatment.inner_steps >= 100
        return value
    runner.transform = change
    result = Scheduler(runner).run(through_phase=1)
    assert len(runner.calls) == 2
    assert result["environments"]["rotation_linear"]["baseline"]["inner_steps"] == 100


def test_invalid_method_is_isolated_and_never_receives_predictive_trials(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    def invalidate(value, env, treatment):
        value["methods"]["jvp"] = {"valid": False, "reason": "unsupported higher derivative"}
        return value
    runner.calibration_transform = invalidate
    result = Scheduler(runner).run()
    assert not any(c[4].condition == "jvp" for c in runner.calls)
    assert result["completed_units"] == 8


def test_policy_preserves_measured_errors_cost_and_missing_diagnostics(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    runner.cfg = dataclasses.replace(runner.cfg, phase1_replicas=2)
    calls = 0
    def measurements(value, env, treatment):
        nonlocal calls
        calls += 1
        value["methods"]["jvp"].update(relative_error=.01 * calls,
                                         gradient_error=.03 / calls, seconds=float(calls))
        value["methods"]["finite_difference"] = {
            "valid": False, "reason": "failed reference comparison", "relative_error": .2 * calls,
            "gradient_error": None, "seconds": 2. * calls}
        value["methods"]["distillation"].update(relative_error=None, gradient_error=.004)
        return value
    runner.calibration_transform = measurements
    progress = Progress()
    first = Scheduler(runner, progress=progress).run(through_phase=1)
    methods = first["environments"]["rotation_linear"]["methods"]
    assert methods["jvp"]["relative_error"] == .02
    assert methods["jvp"]["gradient_error"] == .03
    assert methods["jvp"]["seconds"] == 3.
    assert methods["finite_difference"]["relative_error"] == .4
    assert methods["finite_difference"]["gradient_error"] is None
    assert methods["finite_difference"]["seconds"] == 6.
    assert methods["distillation"]["relative_error"] is None
    assert methods["distillation"]["gradient_error"] == .004
    assert methods["distillation"]["seconds"] is None
    assert methods["directional"]["relative_error"] is None
    jvp = next(row for row in progress.state["calibration"] if row["method"] == "jvp")
    assert jvp["relative_error"] == .02 and jvp["gradient_error"] == .03 and jvp["seconds"] == 3.


def test_losing_methods_still_get_diagnostic_confirmation_and_one_extension(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    def lose(value, family, phase, env, replica, treatment):
        if treatment.condition in METHODS:
            value["current_auc"] = 1. + .01 * replica + treatment.strength
        return value
    runner.transform = lose
    scheduler = Scheduler(runner)
    first = scheduler.phase1()
    second = scheduler.phase2(first)
    selection = second["environments"]["rotation_linear"]
    assert selection["primary_selection"]["classification"] == "diagnostic_no_acquisition_gain"
    for method in METHODS:
        assert len(selection["screen"][method]["treatments"]) == 4
        assert selection["screen"][method]["extension"]["direction"] == "down"
    ledger = scheduler.freeze_confirmation(first, second)
    assert len(ledger["environments"]["rotation_linear"]["treatments"]) == 4


def test_retention_margin_and_missingness_are_not_significance_or_noninferiority():
    baseline = [summary(), summary(current=1.1)]
    candidates = {"harmful": [summary(current=.5, historical=.72), summary(current=.6, historical=.72)],
                  "safe": [summary(current=.95, historical=.705), summary(current=1.05, historical=.705)],
                  "unknown": [summary(current=.1, historical=None), summary(current=.2, historical=None)]}
    result = select_candidate(baseline, candidates, retention_margin=.01)
    assert result["selected"] == "safe"
    missing = select_candidate(baseline, {"unknown": candidates["unknown"]}, retention_margin=.01)
    assert missing["classification"] == "diagnostic_retention_unavailable"
    assert not missing["table"][0]["retention_feasible"]


def test_uncertain_current_gain_ties_favor_lower_cost():
    baseline = [summary(), summary()]
    result = select_candidate(baseline, {
        "expensive": [summary(current=.7, cost=100.), summary(current=1.1, cost=100.)],
        "cheap": [summary(current=.95, cost=10.), summary(current=.95, cost=10.)]}, retention_margin=.01)
    assert result["selected"] == "cheap"
    assert set(result["pareto_keys"]) == {"cheap", "expensive"}


def test_invalid_summary_and_calibration_fail_clearly():
    with pytest.raises(SchedulerContractError):
        validate_summary(summary(current=float("nan")))
    with pytest.raises(SchedulerContractError):
        validate_summary(summary(historical=float("nan")))
    with pytest.raises(SchedulerContractError):
        validate_calibration({"methods": {}}, Config())
    bad = calibration()
    bad["methods"]["jvp"]["unit_gradient_ratio"] = 0
    with pytest.raises(SchedulerContractError):
        validate_calibration(bad, Config())
    assert validate_summary(summary(historical=None))["historical_auc"] is None


def test_numerically_broken_environment_is_isolated_without_invalid_confirmations(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear", "digit9_mixture"))
    def broken(value, family, phase, env, replica, treatment):
        value["solver_ok"] = env != "rotation_linear"
        return value
    runner.transform = broken
    result = Scheduler(runner).run()
    assert result["environments"]["rotation_linear"]["design_validity"] == "common_learner_unavailable"
    assert result["environments"]["rotation_linear"]["comparisons"] == {}
    assert not any(c[1] != "phase1" and c[2] == "rotation_linear" for c in runner.calls)
    assert result["environments"]["digit9_mixture"]["comparisons"]


def test_confirmation_outcomes_and_cost_cannot_rewrite_frozen_allowances(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    def surprising(value, family, phase, env, replica, treatment):
        if phase == "phase3" and treatment.condition == "jvp":
            value.update(current_auc=200., active_seconds=1000., fit_seconds=900.)
        return value
    runner.transform = surprising
    scheduler = Scheduler(runner)
    result = scheduler.run()
    matching = [c[4] for c in runner.calls if c[1] == "phase3" and c[4].condition == "replay_compute_matched"]
    assert all(t.max_evaluations == 150 for t in matching)
    assert len(matching) == runner.cfg.confirmation_replicas
    assert result["environments"]["rotation_linear"]["comparisons"]["jvp"]["current_auc"]["mean_gain"] < 0


def test_closeout_pairs_mechanism_with_controls_without_claiming_consolidation(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    def observed(value, family, phase, env, replica, treatment):
        reduction = .1 if treatment.condition == "jvp" else 0.
        value["mechanism"] = {
            "mean_stm_removal_harm": .2 + replica - reduction,
            "mean_ltm_only_nll": .7 + replica - reduction,
            "mean_future_stm_swap_harm": .3 + replica - reduction,
        }
        if phase == "phase3" and treatment.condition == "jvp":
            value["current_auc"] = 2.
            value["historical_auc"] = 2.
            value["under_solved_fraction"] = .8
            if replica == 1:
                value["mechanism"]["mean_future_stm_swap_harm"] = None
        return value
    runner.transform = observed
    result = Scheduler(runner).run()
    evidence = result["environments"]["rotation_linear"]["comparisons"]["jvp"]
    for field in ("mean_stm_removal_harm", "mean_ltm_only_nll", "mean_future_stm_swap_harm"):
        assert evidence["mechanism"][field]["mean_gain"] == pytest.approx(.1)
    assert evidence["mechanism"]["mean_future_stm_swap_harm"]["n_pairs"] == 1
    assert evidence["unresolved"]["incomplete_mechanism_pairs"]
    assert evidence["unresolved"]["under_solved_units"] == 2
    assert evidence["practical_interpretation"] == "no favorable mean predictive balance"
    assert "consolidation is not inferred automatically" in evidence["mechanism_caution"]
    assert "incomplete mechanism comparisons" in result["conclusion"]
    assert "persistent under-solving warnings" in result["conclusion"]


def test_fresh_store_resume_completes_nested_incomplete_decisions(tmp_path):
    runner = FakeRunner(tmp_path / "unused", environments=("rotation_linear",))
    root, identity = tmp_path / "study", {"test": "nested-resume"}
    with Store(root, identity, max_bytes=16_000_000, free_reserve_bytes=0) as store:
        runner.store = store
        with pytest.raises(ResourceLimit):
            Scheduler(runner, max_units=0).run(through_phase=1)
        assert any(record["status"] == "incomplete" for record in store.iter_units())
    with Store(root, identity, max_bytes=16_000_000, free_reserve_bytes=0, resume=True) as store:
        runner.store = store
        result = Scheduler(runner).run(through_phase=1)
        assert result["environments"]["rotation_linear"]["methods"]["jvp"]["valid"]
        assert all(record["status"] == "completed" for record in store.iter_units())


def test_closeout_rejects_partial_or_wrong_confirmation_jobs(tmp_path):
    runner = FakeRunner(tmp_path, environments=("rotation_linear",))
    scheduler = Scheduler(runner)
    phase3 = scheduler.run(through_phase=3)
    with pytest.raises(SchedulerContractError, match="every predeclared"):
        scheduler.phase4(phase3["ledger"], phase3["results"][:-1])
    with pytest.raises(SchedulerContractError, match="frozen job"):
        scheduler.phase4(phase3["ledger"], list(reversed(phase3["results"])))


def test_real_store_decisions_survive_fresh_process_style_resume(tmp_path):
    runner = FakeRunner(tmp_path / "unused", environments=("rotation_linear",))
    root = tmp_path / "study"
    identity = {"config": runner.cfg.scientific_hash, "source": "scheduler-test-v1"}
    with Store(root, identity, max_bytes=16_000_000, free_reserve_bytes=0) as store:
        runner.store = store
        first = Scheduler(runner).run(through_phase=1)
        calls = len(runner.calls)
        completed = [u for u in store.iter_units(completed_only=True)]
        assert any(u["metadata"]["kind"] == "scheduler_decision" for u in completed)
        with pytest.raises(ImmutableError):
            store.write_json(completed[0]["unit_id"], "result.json", {})
    with Store(root, identity, resume=True, max_bytes=16_000_000, free_reserve_bytes=0) as store:
        runner.store = store
        second = Scheduler(runner).run(through_phase=1)
        assert first == second
        assert len(runner.calls) == calls
