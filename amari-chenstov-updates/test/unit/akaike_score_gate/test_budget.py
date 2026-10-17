from __future__ import annotations

import copy
import json
import math

import pytest

from mnist_experiment.akaike_score_gate.budget import WorkBudget


def test_credit_requires_arrival_and_carries_forward() -> None:
    budget = WorkBudget(3, {"fit": 2, "probe": 4})
    assert not budget.can_spend("fit")
    assert not budget.spend("fit")
    assert budget.spent == budget.remaining == 0
    assert budget.grant_arrival() == 3
    assert budget.spend("fit")
    assert not budget.spend("probe")
    assert budget.grant_arrival() == 4
    assert budget.spend("probe")
    assert budget.spent == budget.granted == 6
    assert budget.elapsed_arrivals == 2
    assert budget.remaining == 0
    assert budget.denied_operations == 2


def test_fractional_charges_cannot_borrow_or_lose_last_operation() -> None:
    budget = WorkBudget(0.3, {"fit": 0.1}, elapsed_arrivals=1)
    for _ in range(3):
        assert budget.spend("fit")
    assert budget.spent == 0.3
    assert budget.remaining == 0
    assert not budget.spend("fit", units=1e-15)


def test_queries_are_pure_and_additional_reserve_is_honored() -> None:
    budget = WorkBudget(10, {"fit": 3, "archive": 4}, elapsed_arrivals=1)
    state = budget.state_dict()
    assert budget.can_spend("fit", units=2, reserve=4)
    assert not budget.can_spend("fit", units=3, reserve=4)
    assert budget.state_dict() == state


def test_archive_reservation_is_protected_and_spent_exactly_once() -> None:
    budget = WorkBudget(10, {"fit": 3, "archive": 4}, elapsed_arrivals=1)
    assert budget.reserve("episode-0", "archive")
    assert budget.spent == 0
    assert budget.reserved == 4
    assert budget.remaining == 6
    assert budget.spend("fit", units=2)
    assert not budget.spend("fit")
    assert budget.spend_reserved("episode-0")
    assert budget.spent == 10
    assert budget.reserved == budget.remaining == 0
    with pytest.raises(KeyError):
        budget.spend_reserved("episode-0")


def test_reservation_denial_collision_and_release_do_not_charge() -> None:
    budget = WorkBudget(5, {"archive": 3}, elapsed_arrivals=1)
    assert budget.reserve("one", "archive")
    with pytest.raises(ValueError, match="already exists"):
        budget.reserve("one", "archive")
    assert not budget.reserve("two", "archive")
    assert budget.release("one") == 3
    assert budget.remaining == 5
    assert budget.spent == budget.reserved == 0
    assert budget.reserve("two", "archive")


@pytest.mark.parametrize("allowance", [None, math.inf])
def test_unlimited_audit_still_accounts_for_all_operations(allowance: float | None) -> None:
    budget = WorkBudget(allowance, {"fit": 2, "archive": 5})
    assert budget.unlimited
    assert budget.spend("fit", units=100)
    assert budget.reserve("end", "archive", units=4)
    assert budget.spend_reserved("end")
    assert budget.spent == 220
    assert budget.remaining == math.inf
    state = json.loads(json.dumps(budget.state_dict(), allow_nan=False))
    restored = WorkBudget.from_state_dict(state, budget_per_arrival=allowance)
    assert restored.state_dict() == state


def test_json_checkpoint_restores_credit_reservations_and_denials() -> None:
    charges = {"fit": 0.1, "archive": 0.4}
    identity = {"protocol": "plan14-test", "replica": 3, "profile": "cpu64"}
    budget = WorkBudget(0.3, charges, identity=identity)
    budget.grant_arrival()
    budget.grant_arrival()
    assert budget.reserve("end", "archive")
    assert budget.spend("fit")
    assert not budget.spend("fit", units=2)
    state = json.loads(json.dumps(budget.state_dict(), allow_nan=False))
    resumed = WorkBudget.from_state_dict(
        state, budget_per_arrival=0.3, charges=charges, identity=identity,
    )
    assert resumed.state_dict() == state
    assert resumed.remaining == budget.remaining == 0.1
    assert resumed.elapsed_arrivals == 2
    for account in (budget, resumed):
        assert account.spend("fit")
        assert not account.spend("fit")
        assert account.spend_reserved("end")
        account.grant_arrival()
        assert account.spend("fit", units=3)
    assert resumed.state_dict() == budget.state_dict()
    assert resumed.spent == 0.9


@pytest.mark.parametrize("override", [
    {"charges": {"fit": 2.1}},
    {"budget_per_arrival": 11},
    {"identity": {"replica": 2}},
])
def test_resume_rejects_changed_frozen_identity(override: dict) -> None:
    state = WorkBudget(10, {"fit": 2}, identity={"replica": 1}).state_dict()
    with pytest.raises(ValueError):
        WorkBudget.from_state_dict(state, **override)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 99), ("charge_table_hash", "corrupt"),
    ("spent", "2"), ("units", {"fit": "-1"}),
    ("denials", {"fit": -1}), ("elapsed_arrivals", 0),
    ("reservations", {"end": {"key": "fit", "units": "100"}}),
])
def test_resume_rejects_corrupt_accounting(field: str, value: object) -> None:
    budget = WorkBudget(10, {"fit": 2}, elapsed_arrivals=1)
    assert budget.spend("fit", units=2)
    state = budget.state_dict()
    state[field] = value
    with pytest.raises(ValueError):
        WorkBudget.from_state_dict(state)


def test_charge_table_and_identity_cannot_be_changed_through_aliases() -> None:
    charges = {"fit": 2}
    identity = {"protocol": {"version": 1}}
    budget = WorkBudget(10, charges, identity=identity)
    original = copy.deepcopy(budget.state_dict())
    charges["fit"] = 8
    identity["protocol"]["version"] = 2
    budget.identity["protocol"]["version"] = 3
    budget.state_dict()["charges"]["fit"] = "9"
    with pytest.raises(TypeError):
        budget.charges["fit"] = 5
    assert budget.state_dict() == original
    assert WorkBudget(None, {"fit": 2.0}).charge_table_hash == budget.charge_table_hash


@pytest.mark.parametrize("kwargs", [
    {"budget_per_arrival": -1}, {"budget_per_arrival": float("nan")},
    {"charges": {"fit": 0}}, {"charges": {"fit": float("inf")}},
    {"elapsed_arrivals": -1}, {"elapsed_arrivals": 1.5},
    {"spent": 1}, {"spent": -1},
])
def test_invalid_budget_configuration_fails(kwargs: dict) -> None:
    values = {"budget_per_arrival": 3, "charges": {"fit": 2}}
    values.update(kwargs)
    with pytest.raises(ValueError):
        WorkBudget(**values)


def test_unknown_operations_and_invalid_units_fail_without_spending() -> None:
    budget = WorkBudget(10, {"fit": 2}, elapsed_arrivals=1)
    with pytest.raises(KeyError, match="frozen work charge"):
        budget.spend("hidden-work")
    for units in (-1, math.inf, math.nan, True):
        with pytest.raises(ValueError):
            budget.spend("fit", units=units)
    assert budget.spent == 0


def test_existing_spent_credit_is_not_refilled_on_resume() -> None:
    budget = WorkBudget(10, {"fit": 2}, elapsed_arrivals=3, spent=28)
    restored = WorkBudget.from_state_dict(budget.state_dict())
    assert restored.remaining == 2
    assert restored.spend("fit")
    assert not restored.spend("fit")
    assert restored.elapsed_arrivals == 3
    assert restored.spent == 30
