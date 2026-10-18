"""Entry-point safeguards; no experiment execution or dataset access."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mnist_experiment.fisher_memory.config import PACKAGE
from mnist_experiment.fisher_memory.run import budget_lock, external_bytes, file_bytes, main
from mnist_experiment.fisher_memory.report import MAX_NOTEBOOK_BYTES
from mnist_experiment.fisher_memory.runner import common_fd_step
from mnist_experiment.fisher_memory import acceptance


def test_production_needs_explicit_allocation(tmp_path):
    with pytest.raises(ValueError, match="positive compute allocation"):
        main(["--config", str(PACKAGE / "configs/default.json"), "--root", str(tmp_path / "study")])
    assert not (tmp_path / "study").exists()


def test_smoke_cannot_be_mislabeled_as_production(tmp_path):
    with pytest.raises(ValueError, match="smoke=true"):
        main(["--config", str(PACKAGE / "configs/smoke.json"), "--root", str(tmp_path / "study")])


def test_budget_lock_prevents_cross_run_writers(tmp_path):
    with budget_lock(tmp_path):
        with pytest.raises(RuntimeError, match="shared storage budget"):
            with budget_lock(tmp_path):
                pytest.fail("second writer acquired the study budget")
    with budget_lock(tmp_path):
        pass


def test_dependencies_and_orphan_exports_are_charged(tmp_path):
    dataset, budget, exports = [tmp_path / name for name in ("data", "budget", "reports")]
    for path in (dataset, budget / "current", budget / "previous", exports):
        path.mkdir(parents=True)
    (dataset / "data.bin").write_bytes(b"123")
    (budget / "current" / "state.bin").write_bytes(b"12345")
    (budget / "previous" / "state.bin").write_bytes(b"1234567")
    (exports / ".plan15-report-old").write_bytes(b"123456789")
    args = SimpleNamespace(data_root=dataset, budget_root=budget, root=budget / "current",
                           notebook=exports / "report.ipynb")
    assert external_bytes(args) == MAX_NOTEBOOK_BYTES + 3 + 7 + 9
    assert file_bytes(budget) == 12


def test_fd_step_must_be_supported_at_every_checkpoint():
    def audit(*steps):
        return {"methods": {"finite_difference": {"candidates": [dict(h=h, valid=True) for h in steps]}}}
    assert common_fd_step([audit(.005, .01), audit(.01, .02)], .01) == .01
    assert common_fd_step([audit(.005), audit(.005, .01)], .01) == .005
    assert common_fd_step([audit(.005), audit(.02)], .01) is None
    assert common_fd_step([audit(.005), audit()], .01) is None


def test_acceptance_requires_actual_passing_test_evidence(tmp_path):
    path = tmp_path / "results.xml"
    path.write_text('<testsuite><testcase name="a" time="0.5"/><testcase name="b"><skipped/></testcase></testsuite>')
    assert acceptance.test_result(path, minimum=1, allow_skips=True)["passed"] == 1
    with pytest.raises(ValueError, match="incomplete or failed"):
        acceptance.test_result(path, minimum=1)
    path.write_text('<testsuite><testcase><failure>broken</failure></testcase></testsuite>')
    with pytest.raises(ValueError, match="incomplete or failed"):
        acceptance.test_result(path, minimum=1)
