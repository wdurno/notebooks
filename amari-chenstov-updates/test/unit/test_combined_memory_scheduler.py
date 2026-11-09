"""Fast scheduler contracts using sealed decisions and synthetic scalar paths."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mnist_experiment.combined_memory import report, run, runtime, trajectory
from mnist_experiment.combined_memory.config import ARMS, Config, Recipe
from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory.control import ResourceLimit, StopRequested


IDENTITY = {"protocol": "plan18-scheduler-unit-fixture", "schema": 1}


def _store(root, *, resume=False):
    return Store(root, IDENTITY, resume=resume, max_bytes=8 * 1024**2,
                 free_reserve_bytes=0, finalization_reserve_bytes=16 * 1024)


def _scheduler(store, *, cfg=None, records=None):
    scheduler = run.Scheduler.__new__(run.Scheduler)
    scheduler.cfg = cfg or Config(device="cpu")
    scheduler.store = store
    scheduler.budget = SimpleNamespace(remaining=100_000.)
    scheduler.records = deepcopy(records or {})
    scheduler.initial_cache = {("development", 0): {"wall_seconds": 1.}}
    scheduler.numeric = dict(lambda0=1., fd_dtype=scheduler.cfg.fd_dtype,
                             max_iter=scheduler.cfg.max_iter, max_eval=scheduler.cfg.max_eval)
    scheduler.last_report = 0.
    return scheduler


def _key(recipe, replica, cohort="development"):
    return f"{cohort}-{recipe.key}-{replica}"


def _add(scheduler, recipe, replica, *, harm=.2, risk=1., cohort="development", **policy):
    numeric = dict(fd_dtype=scheduler.cfg.fd_dtype if recipe.strength else "unused",
                   max_iter=scheduler.cfg.max_iter, max_eval=scheduler.cfg.max_eval, **policy)
    record = dict(metadata=dict(kind="trajectory", recipe=recipe.mapping(), replica=replica,
        cohort=cohort, numerical_policy=numeric), status="completed",
        summary=dict(novel_auc=risk, Havg=harm, H100=harm, learning_seconds=1., online_wall_seconds=1.))
    scheduler.records[_key(recipe, replica, cohort)] = record
    return record


def _pair(scheduler, recipe, **kwargs):
    for replica in range(2):
        _add(scheduler, recipe, replica, **kwargs)


@pytest.mark.parametrize("remaining", [0., 100_000.], ids=["no-new-search-budget", "search-affordable"])
def test_refinement_resumes_sealed_partial_replica_zero_without_new_recipe(tmp_path, remaining):
    root = tmp_path / "store"
    selected = {arm: Recipe(arm, strength=0. if arm == "A" else 1.) for arm in ARMS}
    with _store(root) as store:
        first = _scheduler(store)
        for arm, recipe in selected.items():
            _pair(first, recipe, harm=.6 if arm == "E" else .2)
        for ridge in (.1, 10.):
            _add(first, replace(selected["E"], ridge=ridge), 0, harm=.6)
        completed_before_stop = []

        def interrupted_path(cohort, replica, recipe):
            assert cohort == "development"
            assert replica == 0
            completed_before_stop.append(recipe)
            _add(first, recipe, replica, harm=.2)
            raise StopRequested("replica zero committed; replica one not started")

        first.trajectory = interrupted_path
        with pytest.raises(StopRequested, match="replica zero committed"):
            first.refine(deepcopy(selected))
        unit = "decision-ridge-proposal-0"
        assert store.is_complete(unit)
        decision = json.loads((store.unit_path(unit) / "decision.json").read_bytes())
        proposal = Recipe(**decision["recipe"])
        assert completed_before_stop == [proposal]
        assert proposal.arm == "E" and proposal not in selected.values()
        assert _key(proposal, 0) in first.records and _key(proposal, 1) not in first.records
        records = deepcopy(first.records)
        sealed = (store.unit_path(unit) / "COMPLETED").read_bytes()

    with _store(root, resume=True) as store:
        resumed = _scheduler(store, records=records)
        resumed.budget.remaining = remaining
        calls, admitted = [], []

        def complete_only_recorded_path(cohort, replica, recipe):
            assert cohort == "development"
            assert recipe == proposal, "resume admitted a different refinement recipe"
            calls.append((recipe, replica))
            key = _key(recipe, replica)
            if key not in resumed.records:
                admitted.append((recipe, replica))
                return _add(resumed, recipe, replica, harm=.2)
            return resumed.records[key]

        resumed.trajectory = complete_only_recorded_path
        resumed.decision = Mock(side_effect=AssertionError("sealed replay must not search or reseal"))
        chosen, history = resumed.refine(deepcopy(selected))
        assert calls == [(proposal, 0), (proposal, 1)]
        assert admitted == [(proposal, 1)]
        assert chosen["E"] == proposal
        assert [a["proposal"] for a in history["attempted"]] == [proposal.mapping()]
        assert (store.unit_path(unit) / "COMPLETED").read_bytes() == sealed
        assert not store.is_complete("decision-ridge-proposal-1")
        # Another process restart replays the same decision without new paths.
        calls.clear()
        admitted.clear()
        again, again_history = resumed.refine(deepcopy(selected))
        assert again == chosen and again_history == history
        assert admitted == []


def test_new_refinement_consumes_recipe_returned_by_sealing_decision(tmp_path):
    with _store(tmp_path / "store") as store:
        scheduler = _scheduler(store)
        selected = {a: Recipe(a, strength=0. if a == "A" else 1.) for a in ARMS}
        for arm, recipe in selected.items():
            _pair(scheduler, recipe, harm=.6 if arm == "E" else .2)
        for ridge in (.1, 10.):
            _add(scheduler, replace(selected["E"], ridge=ridge), 0, harm=.6)
        recorded = replace(selected["E"], ridge=.7)
        proposed = []

        def seal(name, make):
            proposed.append(Recipe(**make()["recipe"]))
            return {"recipe": recorded.mapping()}

        def accepted(cohort, replica, recipe):
            assert recipe == recorded
            return _add(scheduler, recipe, replica, harm=.2)

        scheduler.decision = seal
        scheduler.trajectory = accepted
        chosen, history = scheduler.refine(selected)
        assert proposed and proposed[0] != recorded
        assert chosen["E"] == recorded
        assert history["attempted"][0]["proposal"] == recorded.mapping()


def test_calibration_reranks_A_C_against_refined_B_using_only_complete_development_pairs(tmp_path):
    cfg = replace(Config(device="cpu"), strength_ratios=(1.,), process_scales=(1e-4, 1e-3))
    with _store(tmp_path / "store") as store:
        scheduler = _scheduler(store, cfg=cfg)
        calls, frozen, before_refinement = [], {}, {}

        def outcomes(recipe):
            if recipe.arm == "B":
                return {1.: (.1, 1.), .1: (.1, 2.), 10.: (.3, 3.)}[recipe.ridge]
            if recipe.arm in ("A", "C"):
                # Ridge 1 would be best at the new target, but stays replica-0
                # only. Ridge 10 is the best fully paired alternative.
                return {.1: (.1, 1.), 1.: (.3, .01), 10.: (.25, .2)}[recipe.ridge]
            return .3, .5

        def scalar_path(cohort, replica, recipe):
            calls.append((cohort, replica, recipe))
            harm, risk = outcomes(recipe)
            return _add(scheduler, recipe, replica, cohort=cohort, harm=harm, risk=risk)

        def refine(selected):
            before_refinement.update(selected)
            assert selected["A"].ridge == selected["C"].ridge == .1
            refined = replace(selected["B"], ridge=10.)
            _pair(scheduler, refined, harm=.3, risk=3.)
            # Better-looking nuisance evidence must not enter development ranking.
            _pair(scheduler, Recipe("A", ridge=.03), cohort="fresh", harm=.3, risk=0.)
            _pair(scheduler, Recipe("C", ridge=.03, strength=2.), harm=.3, risk=0.)
            return {**selected, "B": refined}, {"attempted": [refined.mapping()]}

        scheduler.trajectory = scalar_path
        scheduler.refine = refine
        scheduler.decision = lambda name, make: {"etas": list(cfg.process_scales)}
        scheduler.freeze = lambda selected, shortlist: frozen.update(selected=deepcopy(selected), shortlist=shortlist)
        scheduler.calibration()
        chosen = frozen["selected"]
        assert chosen["B"].ridge == 10.
        assert chosen["B"].strength == before_refinement["B"].strength
        for arm in ("A", "C"):
            assert chosen[arm].ridge == 10.
            assert scheduler.rank(chosen[arm], [0, 1], chosen["B"])[0] == 0.
            assert scheduler.rank(before_refinement[arm], [0, 1], chosen["B"])[0] > 0.
            assert _key(replace(chosen[arm], ridge=1.), 1) not in scheduler.records
        assert chosen["C"].strength == chosen["B"].strength
        assert chosen["D"].process_scale == chosen["E"].process_scale
        assert not any(c == "fresh" for c, _, _ in calls)


@pytest.mark.parametrize("smoke", [True, False], ids=["smoke-isolated", "production-public-default"])
def test_report_only_never_opens_science_and_uses_correct_notebook_path(tmp_path, monkeypatch, smoke):
    cfg = Config.smoke_config() if smoke else Config(device="cpu")
    root = tmp_path / "run"
    build = Mock(return_value={"generation": "unit-test"})
    forbidden = Mock(side_effect=AssertionError("report-only attempted scientific execution"))
    monkeypatch.setattr(run.Config, "read", lambda _: cfg)
    monkeypatch.setattr(run, "open_store", forbidden)
    monkeypatch.setattr(run, "load_data", forbidden)
    monkeypatch.setattr(report, "build_report", build)
    assert run.main(["--config", str(tmp_path / "config.json"), "--root", str(root), "--report-only"]) == 0
    build.assert_called_once_with(root, cfg=cfg,
        notebook_path=root / "smoke_results.ipynb" if smoke else None)
    forbidden.assert_not_called()
    assert not root.exists()


@pytest.mark.parametrize("error,status", [(StopRequested("user pause"), "paused"),
    (ResourceLimit("invocation wall-clock allocation exhausted"), "resource_limited"),
    (ResourceLimit("cumulative allocation exhausted"), "resource_limited")],
    ids=["stop", "wall-limit", "cumulative-limit"])
def test_controlled_exit_requests_status_without_figures_and_closes_ledgers(tmp_path, monkeypatch, error, status):
    cfg = Config.smoke_config()
    store = SimpleNamespace(root=tmp_path / "run", control=SimpleNamespace(signal_handlers=nullcontext), close=Mock())
    budget = SimpleNamespace(used=3., close=Mock())
    scheduler = SimpleNamespace(run=Mock(side_effect=error), publish=Mock())
    monkeypatch.setattr(run.Config, "read", lambda _: cfg)
    monkeypatch.setattr(run, "open_store", lambda *a, **kw: (store, budget))
    monkeypatch.setattr(run, "load_data", lambda _: {})
    monkeypatch.setattr(run, "Scheduler", lambda *a, **kw: scheduler)
    build = Mock(side_effect=AssertionError("controlled pause attempted figures"))
    monkeypatch.setattr(report, "build_report", build)
    assert run.main(["--config", str(tmp_path / "cfg.json")]) == 0
    scheduler.publish.assert_called_once_with(status, figures=False)
    budget.close.assert_called_once_with()
    store.close.assert_called_once_with()
    build.assert_not_called()


@pytest.mark.parametrize("status", ["paused", "resource_limited"])
def test_publish_without_figures_flushes_status_without_rechecking_expired_budget(tmp_path, monkeypatch, status):
    store = SimpleNamespace(root=tmp_path, control=SimpleNamespace(stop_requested=False), write_json=Mock())
    scheduler = _scheduler(store)
    scheduler.status = lambda state: {"status": state, "phase": "unit-test"}
    scheduler.budget.check = Mock(side_effect=ResourceLimit("already expired"))
    build = Mock(side_effect=AssertionError("figures=False invoked report construction"))
    monkeypatch.setattr(report, "build_report", build)
    scheduler.publish(status, figures=False)
    store.write_json.assert_called_once_with(None, "status.json", {"status": status, "phase": "unit-test"}, finalizing=True)
    scheduler.budget.check.assert_not_called()
    build.assert_not_called()


def test_scientific_source_manifest_pins_actual_analysis_bytes_but_not_presentation():
    checks = []
    manifest = runtime.sources(lambda: checks.append(True))
    key = "mnist_experiment/combined_memory/analysis.py"
    expected = hashlib.sha256((runtime.REPO / key).read_bytes()).hexdigest()
    assert manifest[key] == expected
    assert "mnist_experiment/combined_memory/report.py" not in manifest
    assert checks and all(not Path(name).is_absolute() for name in manifest)


def test_plan18_observer_scores_innovation_with_prior_noise_then_updates_variance():
    cfg = Config.smoke_config()
    observer = trajectory.Controller(trajectory.PolicySpec("kalman", 1e-4), cfg, cfg.fixed_pi)
    diagonal = torch.tensor([2., 3.], dtype=torch.float64)
    observer.initialize(torch.zeros(2, dtype=torch.float64), diagonal,
                        torch.tensor([4., 9.], dtype=torch.float64))
    before = deepcopy(observer.state)
    action = observer.choose(1 / cfg.initialization_size, diagonal, 4, cfg.cold_start)
    action_copy = deepcopy(action)
    scores = torch.tensor([[100., 200.], [-100., -200.], [100., 200.], [-100., -200.]], dtype=torch.float64)
    transition = torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64)
    prior = transition @ before["covariance"] @ transition.T + observer.process_covariance()
    noise = before["raw_variance"] / before["mapping"].square() / len(scores)
    noise = noise.clamp_min(max(1e-30, float(noise.mean()) * 1e-12))
    variance = prior[:, 0, 0] + noise
    expected_log_score = float((-.5 * (math.log(2 * math.pi) + variance.log())).mean())
    diagnostics = observer.assimilate(scores, 0)
    assert diagnostics["innovation_log_score"] == pytest.approx(expected_log_score)
    weight = 1 - 2**(-1 / cfg.noise_half_life)
    torch.testing.assert_close(observer.state["raw_variance"],
        (1 - weight) * before["raw_variance"] + weight * scores.var(0, unbiased=True))
    assert observer.state["assimilations"] == observer.state["noise_updates"] == 1
    assert action == action_copy
    next_action = observer.choose(1 / cfg.initialization_size, diagonal, 4, cfg.cold_start + 1)
    assert next_action["noise_scale"] > action["noise_scale"]


def test_report_qualifies_partial_pairs_and_uses_completed_closeout_wording():
    metric = {"mean": -.1, "ci": [-.2, 0.]}
    effects = {name: {key: metric for key in ("novel_auc", "Havg", "H100", "current_whole_rms")}
               for name in ("E-B", "E-A")}
    result = dict(status="pending_or_partial", comparisons=effects, efficiency={},
                  cohorts={"fresh": {"n_pairs": 2}})
    partial = report._conclusions(result)
    assert "descriptive progress only" in partial and "degenerate stratified bootstrap" in partial
    assert "must finish before closeout claims" in partial
    complete = report._conclusions({**result, "status": "completed", "cohorts": {"fresh": {"n_pairs": 4}}})
    assert "frozen allocation is complete" in complete
    assert "must finish before closeout claims" not in complete
    assert "degenerate stratified bootstrap" not in complete
    assert "Stationary replicas remain descriptive" in complete


def test_readiness_prose_is_compact_and_links_repair_log_without_absolute_root(tmp_path):
    (tmp_path / "numerics.json").write_text(json.dumps(dict(fd_dtype="float64", max_iter=200, max_eval=300)))
    readiness = dict(initializations=[dict(history=[dict(epoch=15, validation_nll=5.),
                                                       dict(epoch=30, validation_nll=4.),
                                                       dict(epoch=45, validation_nll=4.5)])])
    (tmp_path / "readiness.json").write_text(json.dumps(readiness))
    notes = [dict(summary="Repaired a diagnostic export under a new source identity.")]
    (tmp_path / "execution_notes.json").write_text(json.dumps(notes))
    snapshot = report._snapshot(tmp_path, Config(device="cpu"), lambda: None)
    snapshot["artifact_hrefs"] = {n: "../cache/study/" + n for n in ("readiness.json", "execution_notes.json")}
    prose = report._readiness_text(snapshot)
    assert "common initialization budget is 45 epochs" in prose
    assert "optional 60-epoch extension is not selected" in prose
    assert "Selected development checkpoint epochs: 30" in prose
    assert "float64" in prose and "200" in prose and "300" in prose
    assert "Execution notes" in prose and "Full append-only log" in prose
    assert str(tmp_path) not in prose
    assert "execution_notes.json" in snapshot["inputs"]["files"]
    before = snapshot["source_digest"]
    (tmp_path / "execution_notes.json").write_text(json.dumps(notes + ["A second recorded repair."]))
    assert report._snapshot(tmp_path, Config(device="cpu"), lambda: None)["source_digest"] != before


def test_progress_separates_development_from_fresh_stationary_planned_denominator():
    snapshot = dict(status=dict(completed=dict(development=34, fresh=10, stationary=0), total_paths=40), records=[])
    result = dict(cohorts={"fresh": dict(n_pairs=2, planned_pairs=6), "stationary": dict(n_pairs=0, planned_pairs=2)})
    progress = report._progress(snapshot, result)
    assert "Planned fresh + stationary paths: 40 (development excluded)" in progress
    assert "descriptive progress only" in progress
