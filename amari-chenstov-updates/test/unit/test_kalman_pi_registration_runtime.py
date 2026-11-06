"""Allocation isolation and conservative crash recovery for Phase 6."""

import json
from pathlib import Path

import pytest

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.kalman_pi.registration import runtime
from mnist_experiment.kalman_pi.registration.config import smoke_config


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_phase_allocation_preserves_historical_budget(tmp_path, monkeypatch):
    family = tmp_path / "plan17"
    monkeypatch.setattr(runtime, "FAMILY", family)
    monkeypatch.setattr(runtime, "ACCEPTANCE_LEDGER", family / "phase6_acceptance_compute.json")
    for name, allocation, seconds in (("old", "plan17", 51000),
                                      ("prior_phase6", "plan17-phase6", 100)):
        _json(family / name / "identity.json", {"identity": {"allocation_family": allocation}})
        _json(family / name / "runtime.json", {"cumulative_active_seconds": seconds})
    _json(family / "prior_phase6/timing_lease.json", {"running": True, "charged_until": 140})
    _json(runtime.ACCEPTANCE_LEDGER, {"entries": [{"seconds": 20}]})
    assert runtime.phase_seconds(family / "current") == 160
    assert json.loads((family / "old/runtime.json").read_text())["cumulative_active_seconds"] == 51000


def test_phase_compute_ledger_rejects_invalid_seconds(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "FAMILY", tmp_path)
    monkeypatch.setattr(runtime, "ACCEPTANCE_LEDGER", tmp_path / "acceptance.json")
    _json(runtime.ACCEPTANCE_LEDGER, {"entries": [{"seconds": -1}]})
    with pytest.raises(Exception, match="invalid"):
        runtime.phase_seconds(tmp_path / "root")


def test_crash_reservation_is_recovered_and_invocation_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "external_bytes", lambda *a, **kw: 0)
    root = tmp_path / "study"
    store = Store(root, {"protocol": "tiny"}, free_reserve_bytes=0,
                  max_active_seconds=100, max_wall_seconds=20)
    try:
        budget = runtime.RuntimeBudget(store, other_seconds=3, allocation_seconds=100)
        lease = json.loads((root / "timing_lease.json").read_text())
        assert lease["charged_until"] <= store.control.base_seconds + 20
        budget.close()
    finally:
        store.close()
    _json(root / "timing_lease.json", {"running": True, "charged_until": 30})
    resumed = Store(root, {"protocol": "tiny"}, resume=True, free_reserve_bytes=0)
    try:
        budget = runtime.RuntimeBudget(resumed, other_seconds=3, allocation_seconds=100)
        assert budget.used >= 33
        assert budget.used < 35
        budget.close()
    finally:
        resumed.close()


@pytest.mark.parametrize("limits,pattern", [({"max_bytes": 1}, "byte allocation"),
                                           ({"min_free_bytes": 10**15}, "filesystem space")])
def test_configured_storage_limits_are_enforced(tmp_path, monkeypatch, limits, pattern):
    monkeypatch.setattr(runtime, "FAMILY", tmp_path / "family")
    monkeypatch.setattr(runtime, "phase_seconds", lambda *args: 0.)
    monkeypatch.setattr(runtime, "external_bytes", lambda *args: 0)
    monkeypatch.setattr(runtime.historical, "setup", lambda cfg: None)
    monkeypatch.setattr(runtime, "provenance", lambda *args: {"protocol": "storage-test"})
    with pytest.raises(Exception, match=pattern):
        runtime.open_store(smoke_config(**limits), tmp_path / "root", tmp_path / "data")


def test_source_identity_covers_state_restoration_dependency():
    sources = runtime.scientific_sources()
    assert "mnist_experiment/kalman_pi/trajectory.py" in sources
