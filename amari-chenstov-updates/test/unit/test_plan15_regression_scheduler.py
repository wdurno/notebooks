"""Calibration retries must preserve decisions and never approve failed audits."""

from types import SimpleNamespace

import pytest

from mnist_experiment.fisher_memory.artifacts import Store
from mnist_experiment.fisher_memory_regression.run import Study
from mnist_experiment.fisher_memory_regression import run
from mnist_experiment.fisher_memory_regression.config import Config, PACKAGE


def test_calibration_attempt_decisions_are_independent_and_final_global(tmp_path):
    with Store(tmp_path, {"test": "calibration_decisions"}, free_reserve_bytes=0) as store:
        study = Study.__new__(Study)
        study.store = store
        study.calibration_namespace = "calibration-attempt-0"
        study.seal("ridge", {"value": 1})
        assert study.decision("ridge") == {"value": 1}
        study.calibration_namespace = "calibration-attempt-1"
        assert study.decision("ridge") is None
        study.seal("ridge", {"value": 2})
        study.seal("calibration", {"value": 3})
        study.calibration_namespace = None
        assert study.decision("calibration") == {"value": 3}
        assert study.decision("ridge") is None
        study.calibration_namespace = "calibration-attempt-0"
        assert study.decision("ridge") == {"value": 1}
        with pytest.raises(ValueError, match="changed"):
            study.seal("ridge", {"value": 9})


@pytest.mark.parametrize("gap,failed,adequate", [(0., False, True), (.02, False, False),
                                                 (0., True, False), (None, True, False)])
def test_solver_audit_rejects_material_or_failed_comparison(tmp_path, gap, failed, adequate):
    with Store(tmp_path, {"test": "solver_audit"}, free_reserve_bytes=0) as store:
        study = Study.__new__(Study)
        study.store = store
        study.cfg = SimpleNamespace(audit_objective_tolerance=.001, smoke=False)
        store.begin("example", {})
        store.write_json("example", "audits.json", [{"relative_objective_gain": gap, "enlarged_failed": failed}])
        result = study._solver_audit([{"unit_id": "example"}])
        assert result["adequate"] is adequate
        assert not result["smoke_gate_bypassed"]


def test_resume_restores_visible_progress_even_when_every_stage_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "MNISTSource", lambda path: None)
    monkeypatch.setattr(run, "Runner", lambda *args, **kwargs: None)
    with Store(tmp_path, {"test": "resume_progress"}, free_reserve_bytes=0) as store:
        store.write_json(None, "status.json", dict(changes=[], family="confirmation", stage="8E complete",
                                                 completed=6, planned=6, forecast_seconds=0))
        args = SimpleNamespace(stage="all", data_root=tmp_path, smoke_stop_at=None)
        study = Study(Config(), args, store, None)
        assert study.status["completed"] == study.status["planned"] == 6
        assert study.status["family"] == "confirmation"
        assert study.status["stage"] == "8E complete"


@pytest.mark.parametrize("relative", [".", "nested/study"])
def test_root_layout_cannot_evade_cross_version_compute_accounting(tmp_path, relative):
    with pytest.raises(ValueError, match="direct child"):
        run.main(["--config", str(PACKAGE / "configs/smoke.json"),
                  "--budget-root", str(tmp_path), "--root", str(tmp_path / relative),
                  "--allocation-hours", ".5"])
