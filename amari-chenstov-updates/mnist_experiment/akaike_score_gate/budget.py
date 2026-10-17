"""Deterministic, frozen-cost accounting for Plan 14 online work.

Charges are supplied by timing calibration, never inferred from fit quality.
Decimal arithmetic prevents fractional charges from permitting future borrowing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any


BUDGET_SCHEMA_VERSION = 1
_UNSET = object()


def _amount(value: Any, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class WorkBudget:
    """An arrival-funded account with protected, named operation reservations.

    ``None`` or positive infinity means unlimited *accounted* audit work.
    ``spend`` and ``reserve`` return False on denial, without partial spending.
    A denied admission is recorded; ``can_spend`` is a read-only query.
    ``reserve`` on ``can_spend`` is additional uncommitted headroom, while
    ``reserve(name, key)`` persists a reservation across operations and resumes.
    """

    def __init__(
        self,
        budget_per_arrival: float | None,
        charges: Mapping[str, float],
        elapsed_arrivals: int = 0,
        spent: float = 0,
        *,
        identity: Any = None,
    ) -> None:
        self._allowance = (
            None
            if budget_per_arrival is None or budget_per_arrival == math.inf
            else _amount(budget_per_arrival, "budget_per_arrival")
        )
        self._charges = {}
        for key, value in charges.items():
            if not isinstance(key, str) or not key:
                raise ValueError("charge keys must be nonempty strings")
            charge = _amount(value, f"charge {key}")
            if charge == 0:
                raise ValueError("profiled charges must be positive")
            self._charges[key] = charge
        self._identity_json = _canonical(identity)
        self._elapsed_arrivals = _count(elapsed_arrivals, "elapsed_arrivals")
        self._spent = _amount(spent, "spent")
        self._initial_spent = self._spent
        self._units = {key: Decimal(0) for key in self._charges}
        self._denials = {key: 0 for key in self._charges}
        self._reservations: dict[str, tuple[str, Decimal]] = {}
        if self._allowance is not None and self._spent > self._allowance * elapsed_arrivals:
            raise ValueError("spent work exceeds arrived allowance")

    @property
    def budget_per_arrival(self) -> float | None:
        return None if self._allowance is None else float(self._allowance)

    @property
    def elapsed_arrivals(self) -> int:
        return self._elapsed_arrivals

    @property
    def spent(self) -> float:
        return float(self._spent)

    @property
    def charges(self) -> Mapping[str, float]:
        return MappingProxyType({key: float(value) for key, value in self._charges.items()})

    @property
    def charge_table_hash(self) -> str:
        payload = {key: str(value.normalize()) for key, value in self._charges.items()}
        return hashlib.sha256(_canonical(payload).encode("ascii")).hexdigest()

    @property
    def identity(self) -> Any:
        return json.loads(self._identity_json)

    @property
    def unlimited(self) -> bool:
        return self._allowance is None

    @property
    def granted(self) -> float:
        if self._allowance is None:
            return math.inf
        return float(self._allowance * self._elapsed_arrivals)

    def _reserved_amount(self) -> Decimal:
        return sum(
            (self._charges[key] * units for key, units in self._reservations.values()),
            Decimal(0),
        )

    @property
    def reserved(self) -> float:
        return float(self._reserved_amount())

    @property
    def remaining(self) -> float:
        """Unspent credit excluding protected reservations."""
        if self._allowance is None:
            return math.inf
        return float(self._allowance * self._elapsed_arrivals - self._spent - self._reserved_amount())

    @property
    def denied_operations(self) -> int:
        return sum(self._denials.values())

    def grant_arrival(self) -> float:
        self._elapsed_arrivals += 1
        return self.remaining

    def _operation(self, key: str, units: float) -> tuple[Decimal, Decimal]:
        if key not in self._charges:
            raise KeyError(f"operation has no frozen work charge: {key}")
        count = _amount(units, "units")
        return count, self._charges[key] * count

    def can_spend(self, key: str, units: float = 1, reserve: float = 0) -> bool:
        _, cost = self._operation(key, units)
        headroom = _amount(reserve, "reserve")
        return self._allowance is None or (
            self._spent + self._reserved_amount() + cost + headroom
            <= self._allowance * self._elapsed_arrivals
        )

    def spend(self, key: str, units: float = 1) -> bool:
        count, cost = self._operation(key, units)
        if not self.can_spend(key, units):
            self._denials[key] += 1
            return False
        self._spent += cost
        self._units[key] += count
        return True

    def reserve(self, name: str, key: str, units: float = 1) -> bool:
        """Protect arrived credit for an operation, without charging it yet."""
        if not isinstance(name, str) or not name:
            raise ValueError("reservation name must be a nonempty string")
        if name in self._reservations:
            raise ValueError(f"reservation already exists: {name}")
        count, _ = self._operation(key, units)
        if not self.can_spend(key, units):
            self._denials[key] += 1
            return False
        self._reservations[name] = (key, count)
        return True

    def spend_reserved(self, name: str) -> bool:
        """Charge the protected operation exactly once; unknown names fail."""
        key, units = self._reservations.pop(name)
        self._spent += self._charges[key] * units
        self._units[key] += units
        return True

    def release(self, name: str) -> float:
        """Cancel a reservation without charging it."""
        key, units = self._reservations.pop(name)
        return float(self._charges[key] * units)

    def state_dict(self) -> dict[str, Any]:
        """JSON-safe exact checkpoint; numeric strings avoid round-trip drift."""
        return {
            "schema_version": BUDGET_SCHEMA_VERSION,
            "budget_per_arrival": None if self._allowance is None else str(self._allowance),
            "charges": {key: str(value) for key, value in self._charges.items()},
            "charge_table_hash": self.charge_table_hash,
            "identity": self.identity,
            "elapsed_arrivals": self._elapsed_arrivals,
            "spent": str(self._spent),
            "initial_spent": str(self._initial_spent),
            "units": {key: str(value) for key, value in self._units.items()},
            "denials": dict(self._denials),
            "reservations": {
                name: {"key": key, "units": str(units)}
                for name, (key, units) in self._reservations.items()
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        budget_per_arrival: Any = _UNSET,
        charges: Mapping[str, float] | None = None,
        identity: Any = _UNSET,
    ) -> WorkBudget:
        """Restore credit, validating checkpoint and optional protocol identity."""
        if state.get("schema_version") != BUDGET_SCHEMA_VERSION:
            raise ValueError("incompatible work-budget schema")
        restored = cls(
            state["budget_per_arrival"], state["charges"],
            state["elapsed_arrivals"], state["spent"], identity=state["identity"],
        )
        if restored.charge_table_hash != state["charge_table_hash"]:
            raise ValueError("checkpoint charge-table hash mismatch")
        if charges is not None:
            expected = cls(None, charges)
            if expected.charge_table_hash != restored.charge_table_hash:
                raise ValueError("frozen work charges changed across resume")
        if budget_per_arrival is not _UNSET:
            expected = cls(budget_per_arrival, {})
            if expected._allowance != restored._allowance:
                raise ValueError("arrival allowance changed across resume")
        if identity is not _UNSET and _canonical(identity) != restored._identity_json:
            raise ValueError("work-budget checkpoint identity mismatch")
        if set(state["units"]) != set(restored._charges) or set(state["denials"]) != set(restored._charges):
            raise ValueError("operation accounting does not match charge table")
        restored._units = {key: _amount(value, "units") for key, value in state["units"].items()}
        restored._denials = {key: _count(value, "denials") for key, value in state["denials"].items()}
        restored._initial_spent = _amount(state["initial_spent"], "initial_spent")
        accounted = restored._initial_spent + sum(
            (restored._charges[key] * units for key, units in restored._units.items()), Decimal(0)
        )
        if accounted != restored._spent:
            raise ValueError("checkpoint spent work disagrees with operation accounting")
        for name, reservation in state["reservations"].items():
            if not restored.reserve(name, reservation["key"], reservation["units"]):
                raise ValueError("checkpoint reservations exceed remaining credit")
        return restored
