"""Allocation separation and serialized acceptance settlement."""

import json
from pathlib import Path
import sys

import pytest

from mnist_experiment.kalman_pi.discrepancy import acceptance, runtime


def test_acceptance_holds_family_lock_until_time_is_settled(tmp_path, monkeypatch):
    family = tmp_path / "family"
    monkeypatch.setattr(acceptance, "FAMILY", family)
    monkeypatch.setattr(acceptance, "ACCEPTANCE_LEDGER", family / "acceptance.json")
    monkeypatch.setattr(acceptance, "phase_seconds", lambda root: 0.)
    program = ("import fcntl,sys; f=open(sys.argv[1],'a+b');\n"
               "try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
               "except BlockingIOError: sys.exit(0)\n"
               "sys.exit(1)")
    assert acceptance.main(["--timeout", "5", "--", sys.executable, "-c", program,
                            str(family / ".allocation.lock")]) == 0
    entry = json.loads((family / "acceptance.json").read_text())["entries"][0]
    assert entry["status"] == "completed" and entry["exit_code"] == 0
    assert 0 < entry["seconds"] < 5


def test_explicit_conservative_overhead_is_separate_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "FAMILY", tmp_path)
    monkeypatch.setattr(acceptance, "ACCEPTANCE_LEDGER", tmp_path / "acceptance.json")
    monkeypatch.setattr(acceptance, "phase_seconds", lambda root: 0.)
    assert acceptance.main(["--overhead-seconds", "30", "--description", "preflight"]) == 0
    entry = json.loads((tmp_path / "acceptance.json").read_text())["entries"][0]
    assert entry["status"] == "conservative-overhead"
    assert entry["seconds"] == 30 and entry["description"] == "preflight"
    monkeypatch.setattr(acceptance, "phase_seconds", lambda root: 71999.)
    with pytest.raises(RuntimeError, match="cannot reserve"):
        acceptance.main(["--overhead-seconds", "30"])


def test_phase_seven_ledger_does_not_reuse_old_allowances(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "FAMILY", tmp_path)
    monkeypatch.setattr(runtime, "ACCEPTANCE_LEDGER", tmp_path / "acceptance.json")
    own = tmp_path / "own"
    for name, family, seconds in (("own", "plan17-phase7", 17.),
                                  ("new-peer", "plan17-phase7", 13.),
                                  ("old-peer", "plan17-phase6", 99.)):
        path = tmp_path / name
        path.mkdir()
        (path / "identity.json").write_text(json.dumps({"identity": {"allocation_family": family}}))
        (path / "runtime.json").write_text(json.dumps({"cumulative_active_seconds": seconds}))
    (tmp_path / "acceptance.json").write_text(json.dumps({"entries": [{"seconds": 3.}]}))
    assert runtime.phase_seconds(own) == 16.
    assert runtime.phase_seconds(tmp_path / "other") == 33.


def test_scientific_manifest_excludes_only_reporting_and_acceptance():
    sources = runtime.scientific_sources()
    prefix = "mnist_experiment/kalman_pi/discrepancy/"
    assert prefix + "policies.py" in sources and prefix + "analysis.py" in sources
    assert prefix + "report.py" not in sources and prefix + "acceptance.py" not in sources
    assert "mnist_experiment/kalman_pi/registration/model.py" in sources


@pytest.mark.parametrize("case", ["outside", "incomplete", "symlink"])
def test_closeout_rejects_unfinished_or_redirected_roots(tmp_path, monkeypatch, case):
    family = tmp_path / "family"
    family.mkdir()
    monkeypatch.setattr(acceptance, "FAMILY", family)
    root = (tmp_path if case == "outside" else family) / "study"
    root.mkdir()
    if case != "incomplete":
        (root / "study_completed.json").write_text("{}")
    if case == "symlink":
        link = family / "redirect"
        link.symlink_to(root, target_is_directory=True)
        root = link
    with pytest.raises(ValueError, match="symlink|completed Phase 7"):
        acceptance.audit(root)
    assert not list(family.glob("**/closeout_validation-*.json"))
