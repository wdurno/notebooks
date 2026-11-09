"""Scalar-only regressions for sealed Phase 6 refinement and reference replay."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mnist_experiment.combined_memory import run as base_run
from mnist_experiment.combined_memory.config import ARMS, Recipe
from mnist_experiment.combined_memory.retention_match import calibration, run
from mnist_experiment.combined_memory.retention_match.config import Config


REFERENCES = (10., 1., .1, 100.)


class MemoryStore:
    """Decision-only Store double; all payloads and completion state stay in RAM."""

    def __init__(self, root):
        self.root = root
        self.identity_hash = "scalar-refinement-source-identity"
        self.units, self.public, self.events = {}, {}, []

    def unit_path(self, uid):
        return self.root / "units" / uid

    def is_complete(self, uid):
        return self.units.get(uid, {}).get("completed", False)

    def begin(self, uid, metadata):
        self.events.append(("begin", uid))
        if self.is_complete(uid):
            return None
        assert uid not in self.units, "unexpected incomplete decision rewrite"
        self.units[uid] = dict(metadata=deepcopy(metadata), files={}, completed=False)
        return self.unit_path(uid)

    def write_json(self, uid, name, payload, **kwargs):
        self.events.append(("write", uid, name))
        if uid is None:
            self.public[name] = deepcopy(payload)
        else:
            assert not self.is_complete(uid), "completed decisions are immutable"
            self.units[uid]["files"][name] = deepcopy(payload)

    def complete(self, uid):
        assert not self.is_complete(uid), "completed decisions cannot be resealed"
        self.units[uid]["completed"] = True
        self.events.append(("complete", uid))

    def iter_units(self, *, completed_only=False):
        return [dict(unit_id=uid, metadata=deepcopy(unit["metadata"]))
                for uid, unit in self.units.items() if not completed_only or unit["completed"]]

    def read_json(self, path):
        path = Path(path)
        if path.parent == self.root:
            return deepcopy(self.public[path.name])
        uid = path.parent.name
        self.events.append(("read", uid, path.name))
        assert self.is_complete(uid), "only sealed decisions may be replayed"
        return deepcopy(self.units[uid]["files"][path.name])

    def checked_scalar(self, root, uid, name, check):
        assert root == self.root
        check()
        return self.read_json(self.unit_path(uid) / name), "in-memory-checked-scalar"


def scalar_record(recipe, member, *, matched):
    havg, h100 = (2., 4.) if matched or recipe.arm == "A" else (5., 7.)
    return dict(unit_id=f"development-{member}-{recipe.key}", status="completed",
        metadata=dict(kind="trajectory", cohort="development", replica=member,
                      recipe=recipe.mapping()),
        summary=dict(Havg=havg, H100=h100, novel_auc=8., learning_seconds=.01,
                     online_wall_seconds=.01))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "no-artifacts-written")
    monkeypatch.setattr(run, "read_json", store.read_json)
    monkeypatch.setattr(base_run, "read_json", store.read_json)
    monkeypatch.setattr(run, "checked_scalar", store.checked_scalar)
    monkeypatch.setattr(run, "utc_now", lambda: "2026-11-08T00:00:00Z")

    def make(*, records=None, matched=True, cfg=None, remaining=100_000.):
        obj = run.Scheduler.__new__(run.Scheduler)
        obj.cfg = cfg or replace(Config.smoke_config(), retention_tolerance=.1)
        obj.store = store
        obj.budget = SimpleNamespace(check=lambda: None, remaining=remaining, used=0.)
        obj.selection, obj.numeric = None, dict(fd_dtype=obj.cfg.fd_dtype,
            max_iter=obj.cfg.max_iter, max_eval=obj.cfg.max_eval)
        obj.initial_cache = {("development", 0): dict(wall_seconds=.01)}
        obj.reference_ridge, obj.current_check = 10., None
        obj.member_summaries, obj.rounds, obj.current = [], [], {}
        obj.calibration_status, obj.phase = "pending", "fixture"
        if records is None:
            rows = [scalar_record(recipe, member, matched=matched)
                    for member in range(4)
                    for recipe in calibration.initial_recipes(obj.cfg.fd_strength).values()]
            records = {row["unit_id"]: row for row in rows}
        obj.records = deepcopy(records)
        obj.fake_matched = matched

        def trajectory(cohort, member, recipe, **kwargs):
            store.events.append(("trajectory", cohort, member, recipe.key))
            assert cohort == "development", "unexpected confirmation trajectory launch"
            row = scalar_record(recipe, member, matched=matched)
            uid = row["unit_id"]
            if uid not in obj.records:
                obj.records[uid] = row
                store.events.append(("execute", cohort, member, recipe.key))
            return obj.records[uid]

        obj.trajectory = trajectory
        obj.import_history = lambda: store.events.append(("import_history",))
        obj.publish = lambda state="running", **kwargs: store.events.append(("publish", state))
        can_admit = obj.can_admit

        def admission(recipes):
            store.events.append(("admit", obj.reference_ridge, tuple(r.key for r in recipes)))
            return can_admit(recipes)

        obj.can_admit = admission
        return obj

    return SimpleNamespace(store=store, make=make)


def seal_round(scheduler, reference_index, round_index, recipes, *, completed=None):
    payload = scheduler.decision(f"refinement-{reference_index}-{round_index}", lambda: dict(
        reference_ridge=REFERENCES[reference_index], recipes=[recipe.mapping() for recipe in recipes],
        prior_allowances=scheduler.used_allowances(), evidence=[], reason="fixture admitted round"))
    pairs = [(member, recipe) for member in range(4) for recipe in recipes]
    for member, recipe in pairs[:len(pairs) if completed is None else completed]:
        row = scalar_record(recipe, member, matched=scheduler.fake_matched)
        scheduler.records[row["unit_id"]] = row
    return payload, pairs


def close_reference(scheduler, index):
    return scheduler.decision(f"reference-closed-{index}", lambda: dict(
        reference_ridge=REFERENCES[index], matching=dict(passed=False),
        reason="fixture completed bounded reference attempt"))


@pytest.mark.parametrize("round_index", [0, 1], ids=["first-round", "second-round"])
@pytest.mark.parametrize("completed", [1, 7, 15], ids=["one-path", "mid-member", "last-path-pending"])
def test_resume_finishes_the_entire_sealed_round_before_selection_or_freezing(
        harness, monkeypatch, round_index, completed):
    before_restart = harness.make()
    starting = calibration.initial_recipes(before_restart.cfg.fd_strength)
    if round_index:
        seal_round(before_restart, 0, 0, [replace(starting[arm], ridge=2.) for arm in ARMS[1:]])
    recipes = [replace(starting[arm], ridge=3.) for arm in ARMS[1:]]
    payload, pairs = seal_round(before_restart, 0, round_index, recipes, completed=completed)
    sealed_before = deepcopy(harness.store.units)
    # No unreserved refinement-round budget remains, but this round was admitted.
    resumed = harness.make(records=before_restart.records, remaining=15.)
    resumed.proposal_round = Mock(side_effect=AssertionError("sealed round was searched again"))
    real_select, real_payload = calibration.select_tuple, resumed.selection_payload

    def assert_round_complete():
        for recipe in recipes:
            assert calibration.aggregate(resumed.records, recipe)["members"] == [0, 1, 2, 3]

    def select(records, **kwargs):
        assert_round_complete()
        harness.store.events.append(("select", kwargs["reference_ridge"]))
        return real_select(records, **kwargs)

    def freeze(chosen):
        assert_round_complete()
        return real_payload(chosen)

    monkeypatch.setattr(calibration, "select_tuple", select)
    resumed.selection_payload = freeze
    harness.store.events.clear()
    assert resumed.calibration() is True

    executed = [event[2:] for event in harness.store.events if event[0] == "execute"]
    assert executed == [(member, recipe.key) for member, recipe in pairs[completed:]]
    selection_index = harness.store.events.index(("select", 10.))
    freeze_index = harness.store.events.index(("begin", "decision-selection"))
    assert all(index < selection_index < freeze_index for index, event in enumerate(harness.store.events)
               if event[0] == "execute")
    assert resumed.rounds[-1] == payload
    assert resumed.used_allowances() == dict.fromkeys(ARMS[1:], round_index + 1)
    assert resumed.selection["matching"]["passed"] is True
    assert harness.store.is_complete("decision-selection")
    assert all(harness.store.units[uid] == unit for uid, unit in sealed_before.items())
    assert len([event for event in harness.store.events if event[0] == "admit"]) == 1
    resumed.proposal_round.assert_not_called()


@pytest.mark.parametrize("closed_count", [1, 2, 3], ids=["primary", "first-fallback", "second-fallback"])
def test_resume_never_revisits_closed_references_even_when_they_are_now_feasible(
        harness, monkeypatch, closed_count):
    before_restart = harness.make()
    for index in range(closed_count):
        close_reference(before_restart, index)
    sealed_before = deepcopy(harness.store.units)
    resumed = harness.make(records=before_restart.records)
    resumed.proposal_round = Mock(side_effect=AssertionError("feasible fallback should not refine"))
    attempted = []
    real_select = calibration.select_tuple

    def select(records, **kwargs):
        attempted.append(kwargs["reference_ridge"])
        return real_select(records, **kwargs)

    monkeypatch.setattr(calibration, "select_tuple", select)
    harness.store.events.clear()
    assert resumed.calibration() is True
    expected = REFERENCES[closed_count]
    assert attempted == [expected]
    assert resumed.selection["reference_ridge"] == expected
    assert resumed.selection["recipes"]["A"]["ridge"] == expected
    assert [event[1] for event in harness.store.events if event[0] == "admit"] == [expected]
    # The common starting tuple is replayed once, not as another reference attempt.
    initial_A = calibration.initial_recipes(resumed.cfg.fd_strength)["A"]
    assert sum(event[0] == "trajectory" and event[3] == initial_A.key
               for event in harness.store.events) == 4
    assert all(harness.store.units[uid] == unit for uid, unit in sealed_before.items())
    assert not any(event[0] == "begin" and event[1] in sealed_before for event in harness.store.events)
    resumed.proposal_round.assert_not_called()


def test_two_recipe_limits_are_per_arm_and_global_across_resume_and_all_fallbacks(harness):
    before_restart = harness.make(matched=False)
    starting = calibration.initial_recipes(before_restart.cfg.fd_strength)
    first = [replace(starting[arm], ridge=ridge)
             for arm, ridge in zip(ARMS[1:], (100., 100., 10., 1.))]
    seal_round(before_restart, 0, 0, first)
    seal_round(before_restart, 0, 1, [replace(starting["B"], ridge=1000.)])
    close_reference(before_restart, 0)
    assert before_restart.used_allowances() == dict(B=2, C=1, D=1, E=1)
    sealed_before = deepcopy(harness.store.units)
    resumed = harness.make(records=before_restart.records, matched=False)
    assert resumed.calibration() is False

    new_round = harness.store.units["decision-refinement-1-0"]["files"]["decision.json"]
    assert new_round["reference_ridge"] == 1.
    assert new_round["prior_allowances"] == dict(B=2, C=1, D=1, E=1)
    assert [recipe["arm"] for recipe in new_round["recipes"]] == ["C", "D", "E"]
    for mapping in new_round["recipes"]:
        assert calibration.aggregate(resumed.records, Recipe(**mapping))["n_members"] == 4
    assert resumed.used_allowances() == dict.fromkeys(ARMS[1:], 2)
    assert all(harness.store.units[uid] == unit for uid, unit in sealed_before.items())
    assert all(harness.store.is_complete(f"decision-reference-closed-{index}") for index in range(4))
    rounds = [unit for unit in harness.store.iter_units(completed_only=True)
              if unit["metadata"]["name"].startswith("refinement-")]
    assert {unit["metadata"]["name"] for unit in rounds} == {
        "refinement-0-0", "refinement-0-1", "refinement-1-0"}
    again = harness.make(records=resumed.records, matched=False)
    for ridge in REFERENCES:
        again.reference_ridge = ridge
        assert again.proposal_round() == []
        assert again.used_allowances() == dict.fromkeys(ARMS[1:], 2)


@pytest.mark.parametrize("failure", ["forced", "infeasible", "budget_limited"])
def test_failed_calibration_launches_zero_fresh_or_stationary_trajectories(harness, failure):
    cfg = replace(Config.smoke_config(), retention_tolerance=.1,
                  force_calibration_failure=failure == "forced")
    scheduler = harness.make(cfg=cfg, matched=failure == "forced",
                             remaining=0. if failure == "budget_limited" else 100_000.)
    scheduler.run()
    assert scheduler.selection is None
    assert not harness.store.is_complete("decision-selection")
    assert not harness.store.is_complete("decision-final-integrity")
    assert "study_completed.json" not in harness.store.public
    completion = harness.store.public["calibration_completed.json"]
    assert completion["status"] == (
        "calibration_budget_limited" if failure == "budget_limited" else "calibration_infeasible")
    assert completion["launched_confirmation"] is False
    assert all(row["metadata"]["cohort"] == "development" for row in scheduler.records.values())
    assert not any(event[0] == "trajectory" and event[1] in ("fresh", "stationary")
                   for event in harness.store.events)
    if failure == "infeasible":
        assert scheduler.used_allowances() == dict.fromkeys(ARMS[1:], 2)
        assert all(harness.store.is_complete(f"decision-reference-closed-{index}") for index in range(4))
    else:
        assert scheduler.used_allowances() == dict.fromkeys(ARMS[1:], 0)
