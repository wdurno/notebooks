import sys
import subprocess
import os

from numerical_experiment.picar import phase1_fix_report as report


def unit(kind, source, stamp, **fields):
    return {"stored_config": {"source_sha256": source}, "manifest": {"completed_at": stamp},
            "summary": {"kind": f"phase1_fix1_{kind}", **fields}}


def test_reporting_does_not_pool_changed_repair_revisions():
    old = unit("interface", {"roles": "old"}, "1")
    new = unit("interface", {"roles": "new"}, "2")
    assert report.repair_units({"complete_units": [old, new]}) == [new]


def test_repair_conclusion_names_unresolved_gates_without_claiming_learning():
    campaign = unit("campaign", {"x": "a"}, "1", parity_pass=True, control_pass=True,
        teacher_validation_successes=0, teacher_validation_count=0,
        blockers=["Teacher competence remains unresolved."])
    text = report.repair_conclusion({"complete_units": [campaign]})
    assert "Teacher competence remains unresolved" in text and "not run because development readiness failed" in text
    assert "no scientific learning phases have started" in text
    assert "No sample savings" in text


def test_report_import_and_empty_outputs_never_import_model_stack():
    code = ("import sys; from numerical_experiment.picar.phase1_fix_report import repair_outputs; "
            "assert repair_outputs({'complete_units': []}); "
            "assert not any(x in sys.modules for x in ('torch','transformers','numerical_experiment.picar.phase1_fix'))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_artifact_renderer_defaults_to_headless_backend():
    environment = dict(os.environ)
    environment.pop("MPLBACKEND", None)
    code = ("from numerical_experiment.picar import report; import matplotlib,sys; "
            "assert matplotlib.get_backend().lower() == 'agg'; assert 'tkinter' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], env=environment,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
