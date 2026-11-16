"""The supervisor changes failure reporting, never finite numerical results."""

import copy
import math

import pytest

from numerical_experiment.picar.config import canonical_json
from numerical_experiment.picar_solver_supervisor import (
    GuardedRunStore, guarded_recorded, sanitize_diagnostics,
)


def test_nonfinite_diagnostic_is_null_with_signed_field_identity():
    raw = {"rows": [{"ratio": float("inf"), "value": 3.}, {"ratio": float("nan")}],
        "status": "numerical_failure", "negative": -float("inf")}
    safe, missing = sanitize_diagnostics(raw)
    assert safe["rows"][0] == {"ratio": None, "value": 3.}
    assert {r["value"] for r in missing} == {"positive_infinity", "negative_infinity", "nan"}
    assert math.isinf(raw["rows"][0]["ratio"])
    canonical_json(safe)


def test_finite_values_are_exactly_preserved():
    values = {"parameters": [1., -4., 1e30], "timing": .041, "count": 7, "optional": None}
    safe, missing = sanitize_diagnostics(values)
    assert safe == values and missing == []


def test_completion_is_immutable_and_records_guard(tmp_path):
    with GuardedRunStore(tmp_path / "unit", {"test": 1}, checkpoint_policy="journal-v1") as store:
        store.checkpoint({"accepted": [1., 2.]})
        store.complete({"ratio": float("inf"), "status": "numerical_failure"})
        with pytest.raises(RuntimeError):
            store.complete({"ratio": 0.})
    import json
    summary = json.loads((tmp_path / "unit/summary.json").read_text())
    assert summary["ratio"] is None
    assert summary["supervisor"]["changes_numerical_updates"] is False
    assert summary["unavailable_diagnostic_fields"][0]["path"] == "ratio"


@pytest.mark.parametrize("error,guarded", [(ValueError("model output must be finite"), True),
                                         (ValueError("wrong history shape"), False)])
def test_failed_candidate_restores_committed_boundary_and_schema_errors_raise(error, guarded):
    class Store:
        def load_checkpoint(self, **kwargs):
            return {"cursor": 8, "status": "running", "rows": [1, 2]}
    class Recorded:
        def __init__(self):
            self.s, self.store, self.saves = {"cursor": 9, "status": "running"}, Store(), 0
        def run(self):
            raise error
        def save(self):
            self.saves += 1
        def summary(self):
            return copy.deepcopy(self.s)
    run = guarded_recorded(Recorded)()
    if guarded:
        result = run.run()
        assert result["cursor"] == 8 and result["status"] == "numerical_failure"
        assert run.saves == 1
        assert run.run() == result
    else:
        with pytest.raises(ValueError, match="wrong history shape"):
            run.run()
