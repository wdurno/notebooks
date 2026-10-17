"""Tiny scalar fixtures for the standalone Phase 3 supplement; no experiments."""

import ast
import base64
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from mnist_experiment import akaike_score_gate_reporting as view
from mnist_experiment.akaike_score_gate import analysis

if __package__:
    from .test_analysis import make_root, reseal, seal_unit, write_json
else:
    from test_analysis import make_root, reseal, seal_unit, write_json


def matrix_errors(value):
    # Real fisher_audit.matrix_metrics scalar names, not the old notebook names.
    return {"frobenius_error": value * 2, "relative_frobenius_error": value,
            "relative_diagonal_error": value / 2, "trace_error": -value,
            "leading_subspace_overlap": .75, "reference_is_finite": True}


def audit_rows(*, converged=True, gain=.2, repeats=32):
    rows = [{"metric_schema_version": 1, "category": "history_fit", "repeat": repeat,
             "seed": 1400 + repeat, "converged": converged,
             "restricted_residual_max": .01 if converged else .5,
             "realized_nines": 2, "sample_count": 64, "ess": 60.,
             "observed_covariance_status": "available"} for repeat in range(repeats)]
    for estimator in view.ESTIMATORS:
        rows.append({"metric_schema_version": 1, "category": "covariance_calibration",
                     "covariance": estimator, "status": "available", "step": 7, "frontier": 1,
                     "gradient_noise_error": matrix_errors(0. if estimator == "repeated_history" else gain),
                     "parameter_covariance_error": matrix_errors(gain * 2),
                     "restricted_reference_converged": converged,
                     "all_history_fits_converged": converged,
                     "includes_all_assigned_history_fits": True,
                     "finite_covariance_reference": True, "supported_subspace_only": True})
    for layers in (1, 2):
        for estimator in ("model", "observed", "repeated_history"):
            rows.append({"metric_schema_version": 1, "category": "expansion_shadow",
                         "step": 7, "frontier": 1, "expansion_layers": layers, "covariance": estimator,
                         "gate_status": "closed" if converged else "restricted_fit_unresolved",
                         "gate_open": False, "restricted_fit_converged": converged,
                         "expanded_fit_converged": converged,
                         "forecast_gain": gain * layers, "forecast_risk_reduction": gain * layers - .1,
                         "realized_weighted_training_gain": gain * layers + .05,
                         "realized_next_gain": gain * layers - .04,
                         "optimism_scope": "next_condition_drift_and_selection_not_a_pure_optimism_estimate"})
    return rows


def add_audit(root, condition, *, index=0, converged=True, gain=.2, kind="calibration_audit", rows=None):
    case = {"environment": {"name": condition}, "step": 7, "frontier": 1}
    if rows is None:
        rows = audit_rows(converged=converged, gain=gain)
    repeats = 32 if kind == "calibration_audit" else 0
    if kind == "shadow_audit":
        rows = [row for row in rows if row["category"] == "expansion_shadow" and row["covariance"] != "repeated_history"]
    return seal_unit(root, phase="phase3", kind=kind, policy=None, environment=condition, index=index,
                     summary={"phase": "phase3", "case": case, "history_repeats": repeats,
                              "unresolved_history_fits": 0 if converged else repeats,
                              "restricted_reference_converged": converged}, detail={"case": case}, audit=rows)


def scalar_root(tmp_path, *, states=4):
    root = make_root(tmp_path / "root")
    write_json(root / "study.json", {"schema_version": 1, "smoke": True,
                                     "history_variance_states": states, "history_variance_repeats": 32})
    return root


def one(rows, **criteria):
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    assert len(matches) == 1
    return matches[0]


def notebook():
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"keep": "original"}, "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "# Original"},
        {"cell_type": "markdown", "id": "phase3", "metadata": {}, "source": ["## Phase 3: Risk calibration\n"]},
        {"cell_type": "code", "id": "phase3-existing", "metadata": {"plan14_phase": "phase3", "keep": True},
         "source": "existing = 7", "execution_count": 23, "outputs": [{"output_type": "stream", "name": "stdout", "text": "keep\n"}]},
        {"cell_type": "markdown", "id": "reader-note", "metadata": {}, "source": "Existing note"},
        {"cell_type": "markdown", "id": "phase4", "metadata": {}, "source": "## Phase 4: Development"},
        {"cell_type": "code", "id": "phase4-existing", "metadata": {"plan14_phase": "phase4"},
         "source": "later = 8", "execution_count": None, "outputs": []}]}


def complete_root(tmp_path):
    root = scalar_root(tmp_path)
    paths = [add_audit(root, condition, index=index) for index, condition in enumerate(view.CONDITIONS)]
    decision = {"schema_version": 1, "audit_paths": [str(path) for path in paths], "selected": {"c": 2.}}
    path = seal_unit(root, phase="phase3", kind="decision", policy="phase3_calibration", environment="shared",
                     summary={"phase": "phase3", "decision": "phase3_calibration", **decision})
    write_json(path / "decision.json", decision)
    reseal(path)
    write_json(root / "progress.json", {"status": "interrupted", "phases": {
        "phase3": {"status": "completed"}, "phase4": {"status": "running"}}})
    # Gracefully paused later work must not prevent a completed Phase 3 supplement.
    (root / "phase4" / ".incomplete" / "paused").mkdir(parents=True)
    target = tmp_path / "report.ipynb"
    write_json(target, notebook())
    return root, target


def snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file() and path.name != ".run.lock"}


def test_all_unresolved_keeps_all_assignments_and_excludes_numeric_forecasts(tmp_path):
    root = scalar_root(tmp_path)
    for index, condition in enumerate(view.CONDITIONS):
        add_audit(root, condition, index=index, converged=False)
    report = analysis.load_report(root)
    summary = view.summarize_phase3(report)
    for condition in view.CONDITIONS:
        counts = one(summary["fit_counts"], condition=condition, role="history")
        assert (counts["assigned"], counts["resolved"], counts["unresolved"], counts["missing"]) == (32, 0, 32, 0)
        for result in [row for row in summary["forecasts"] if row["condition"] == condition]:
            assert result["assigned"] == result["unresolved"] == 1
            assert result["resolved"] == 0
            assert result["pairs"]["forecast_gain"]["forecast"]["median"] is None
        errors = one(summary["covariances"], condition=condition, covariance="model", domain="parameter_covariance_error")
        assert errors["available"] == errors["uncertified_available"] == 1
        assert errors["errors"]["relative_frobenius_error"]["median"] == .4
    figures = view.phase3_figures(report)
    try:
        for name, figure in figures:
            try:
                if name in {"forecast_gain", "forecast_risk_reduction"}:
                    assert all(not axis.collections for axis in figure.axes)
                    assert all("No resolved paired" in axis.texts[0].get_text() for axis in figure.axes)
                if name == "conditional-covariance-errors":
                    assert all(len(axis.collections) == 5 for axis in figure.axes)
            finally:
                plt.close(figure)
    finally:
        figures.close()
    assert "Unresolved" in view.phase3_markdown(report)


def test_missing_rows_flags_and_terminal_gains_never_zero_fill(tmp_path):
    root = scalar_root(tmp_path)
    rows = audit_rows()
    rows = [row for row in rows if not (row["category"] == "history_fit" and row["repeat"] == 31)
            and not (row["category"] == "expansion_shadow" and row["covariance"] == "observed" and row["expansion_layers"] == 2)]
    rows[30]["converged"] = None
    for row in rows:
        if row["category"] == "expansion_shadow":
            row["realized_next_gain"] = None
            if row["covariance"] == "model":
                row["forecast_gain"] = None
        if row["category"] == "covariance_calibration" and row["covariance"] == "observed":
            row["status"] = "singleton_strata_unresolved"
            row.pop("gradient_noise_error")
            row.pop("parameter_covariance_error")
    add_audit(root, view.CONDITIONS[0], rows=rows)
    summary = view.summarize_phase3(analysis.load_report(root))
    health = summary["fit_health"][0]
    assert health["history_rows"] == 31 and health["covariance_denominator"] == 31
    assert health["counts"]["history"] == {"assigned": 32, "resolved": 30, "unresolved": 0, "missing": 2}
    result = one(summary["forecasts"], condition="rotation_linear", covariance="model", expansion_layers=1)
    assert result["pairs"]["forecast_gain"]["missing_values"] == 1
    assert result["pairs"]["forecast_gain"]["forecast"]["median"] is None
    assert result["pairs"]["forecast_risk_reduction"]["paired"] == 0
    absent = one(summary["forecasts"], condition="rotation_linear", covariance="observed", expansion_layers=2)
    assert absent["missing_rows"] == 1 and absent["unresolved"] == 0
    covariance = one(summary["covariances"], condition="rotation_linear", covariance="observed", domain="gradient_noise_error")
    assert covariance["available"] == 0 and covariance["unavailable"] == 1
    assert covariance["errors"]["relative_frobenius_error"]["median"] is None
    covariance = one(summary["covariances"], condition="rotation_linear", covariance="model", domain="gradient_noise_error")
    assert covariance["available"] == covariance["uncertified_available"] == 1
    missing_condition = one(summary["fit_counts"], condition="mixture_m18", role="history")
    assert missing_condition["assigned"] == missing_condition["missing"] == 32


def test_conditions_separate_and_fit_denominators_do_not_count_estimators(tmp_path):
    root = scalar_root(tmp_path)
    for index, condition in enumerate(view.CONDITIONS):
        add_audit(root, condition, index=index, gain=index + 1.)
    add_audit(root, "rotation_linear", index=9, kind="shadow_audit")
    report = analysis.load_report(root)
    before = copy.deepcopy(report)
    summary = view.summarize_phase3(report)
    assert report == before
    for index, condition in enumerate(view.CONDITIONS):
        counts = one(summary["fit_counts"], condition=condition, role="history")
        assert counts["assigned"] == counts["resolved"] == 32
        for role in ("restricted", "expanded_1", "expanded_2"):
            counts = one(summary["fit_counts"], condition=condition, role=role)
            assert counts["assigned"] == counts["resolved"] == (2 if index == 0 else 1)
        result = one(summary["forecasts"], condition=condition, covariance="repeated_history", expansion_layers=2)
        assert result["pairs"]["forecast_gain"]["forecast"]["median"] == 2 * (index + 1.)
        result = one(summary["covariances"], condition=condition, covariance="model", domain="gradient_noise_error")
        assert result["errors"]["relative_frobenius_error"]["median"] == index + 1.
        assert result["errors"]["trace_error"]["median"] == -(index + 1.)
    figures = view.phase3_figures(report)
    try:
        for name, figure in figures:
            try:
                if name == "forecast_gain":
                    assert len(figure.axes) == 8
                    for index, condition in enumerate(view.CONDITIONS):
                        assert condition in figure.axes[2 * index].get_title()
                        assert any(label.get_text() == "Repeated history" for label in figure.legends[0].texts)
                        assert all(len(axis.containers) == 0 for axis in figure.axes)
            finally:
                plt.close(figure)
    finally:
        figures.close()


def test_figure_labels_and_legends_fit_the_cpu_canvas(tmp_path):
    root = scalar_root(tmp_path)
    for index, condition in enumerate(view.CONDITIONS):
        add_audit(root, condition, index=index)
    figures = view.phase3_figures(analysis.load_report(root))
    try:
        for name, figure in figures:
            try:
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                boxes = [axis.get_tightbbox(renderer) for axis in figure.axes]
                assert all(not left.overlaps(right) for index, left in enumerate(boxes)
                           for right in boxes[index + 1:])
                for legend in figure.legends:
                    box = legend.get_window_extent(renderer)
                    assert not box.overlaps(figure._suptitle.get_window_extent(renderer))
                    assert figure.bbox.contains(box.x0, box.y0) and figure.bbox.contains(box.x1, box.y1)
                figure.savefig(tmp_path / f"{name}.png", dpi=100)
            finally:
                plt.close(figure)
    finally:
        figures.close()


def test_identity_guide_preserves_forecast_and_realized_data_ranges(tmp_path):
    root = scalar_root(tmp_path, states=1)
    rows = audit_rows()
    for row in rows:
        if row["category"] == "expansion_shadow":
            row["forecast_gain"] = 1000.
            row["forecast_risk_reduction"] = -1000.
            row["realized_weighted_training_gain"] = .01
            row["realized_next_gain"] = .02
    add_audit(root, "rotation_linear", rows=rows)
    figures = view.phase3_figures(analysis.load_report(root))
    try:
        for name, figure in figures:
            try:
                if name in {"forecast_gain", "forecast_risk_reduction"}:
                    for axis in figure.axes[:2]:
                        assert max(abs(value) for value in axis.get_ylim()) < 1.
                        expected = 1000. if name == "forecast_gain" else -1000.
                        assert axis.get_xlim()[0] < expected < axis.get_xlim()[1]
                        assert "paired rows=" in axis.get_title()
            finally:
                plt.close(figure)
    finally:
        figures.close()


def test_fit_unresolved_even_when_gate_status_is_closed(tmp_path):
    root = scalar_root(tmp_path, states=1)
    rows = audit_rows()
    for row in rows:
        if row["category"] == "expansion_shadow" and row["expansion_layers"] == 2:
            row["expanded_fit_converged"] = False
    add_audit(root, "rotation_linear", rows=rows)
    summary = view.summarize_phase3(analysis.load_report(root))
    result = one(summary["forecasts"], condition="rotation_linear", covariance="model", expansion_layers=2)
    assert result["resolved"] == 0 and result["unresolved"] == 1
    assert one(summary["fit_counts"], condition="rotation_linear", role="expanded_2")["unresolved"] == 1


def test_nonfinite_and_boolean_gains_are_missing_not_numeric(tmp_path):
    root = scalar_root(tmp_path, states=1)
    rows = audit_rows()
    for row in rows:
        if row["category"] == "expansion_shadow":
            row["forecast_gain"] = True
            row["realized_next_gain"] = float("nan")
    add_audit(root, "rotation_linear", rows=rows)
    summary = view.summarize_phase3(analysis.load_report(root))
    result = one(summary["forecasts"], condition="rotation_linear", covariance="model", expansion_layers=1)
    assert result["resolved"] == 1
    assert result["pairs"]["forecast_gain"]["paired"] == 0
    assert result["pairs"]["forecast_risk_reduction"]["paired"] == 0


def test_conflicting_history_denominator_rejected(tmp_path):
    root = scalar_root(tmp_path, states=1)
    path = add_audit(root, "rotation_linear")
    summary = json.loads((path / "summary.json").read_text())
    summary["history_repeats"] = 16
    write_json(path / "summary.json", summary)
    reseal(path)
    with pytest.raises(analysis.ArtifactError, match="denominator disagrees"):
        view.summarize_phase3(analysis.load_report(root))


def test_missing_calibration_condition_stays_in_configured_denominator(tmp_path):
    report = analysis.load_report(scalar_root(tmp_path))
    summary = view.summarize_phase3(report)
    for condition in view.CONDITIONS:
        row = one(summary["coverage"], condition=condition)
        assert row["calibration_states_assigned"] == row["calibration_states_missing"] == 1
        counts = one(summary["fit_counts"], condition=condition, role="history")
        assert counts["assigned"] == counts["missing"] == 32 and counts["resolved"] == 0
        covariance = one(summary["covariances"], condition=condition, covariance="model", domain="gradient_noise_error")
        assert covariance["assigned"] == covariance["missing_rows"] == 1
        assert covariance["errors"]["relative_frobenius_error"]["median"] is None


def test_covariance_glossary_and_recorded_self_comparison_not_replica_evidence(tmp_path):
    root = scalar_root(tmp_path, states=1)
    add_audit(root, "rotation_linear", converged=False)
    report = analysis.load_report(root)
    summary = view.summarize_phase3(report)
    result = one(summary["covariances"], condition="rotation_linear", covariance="repeated_history", domain="gradient_noise_error")
    assert result["errors"]["relative_frobenius_error"]["median"] == 0.
    assert result["all_assigned_states_included"] is True
    assert result["uncertified_available"] == 1
    cells = view.build_phase3_supplement(report, render=False)
    text = cells[0]["source"]
    for term in ("finite reference", "frozen archive", "not fresh experimental replicas", "final-replica efficacy",
                 "R - 1", "against itself", "drift and selection", "no iid or replica", "accepted states"):
        assert term in text
    assert "observations and their actual labels" in text
    assert "not certified MLE sampling variance" in text
    assert "conditional" in text.lower() and "Warnings" in text
    assert cells[0]["metadata"]["fit_counts"] == summary["fit_counts"]
    assert cells[0]["metadata"]["phase3_supplement_source_sha256"] == hashlib.sha256(Path(view.__file__).read_bytes()).hexdigest()
    ast.parse(cells[1]["source"])


@pytest.mark.parametrize("bad", ["duplicate_history", "conflicting_fit", "duplicate_covariance"])
def test_incompatible_scalar_slots_fail_clearly(tmp_path, bad):
    root = scalar_root(tmp_path, states=1)
    rows = audit_rows()
    if bad == "duplicate_history":
        rows.append(dict(rows[0]))
    elif bad == "conflicting_fit":
        next(row for row in rows if row["category"] == "expansion_shadow")["restricted_fit_converged"] = False
    else:
        rows.append(dict(next(row for row in rows if row["category"] == "covariance_calibration")))
    add_audit(root, "rotation_linear", rows=rows)
    with pytest.raises(analysis.ArtifactError):
        view.summarize_phase3(analysis.load_report(root))


def test_insertion_preserves_every_existing_cell_and_is_idempotent(tmp_path):
    root = scalar_root(tmp_path, states=1)
    add_audit(root, "rotation_linear")
    cells = view.build_phase3_supplement(analysis.load_report(root), render=False)
    original = notebook()
    before = copy.deepcopy(original)
    updated = view.insert_phase3_supplement(original, cells)
    assert original == before
    assert [cell for cell in updated["cells"] if view.SUPPLEMENT_TAG not in cell.get("metadata", {}).get("tags", [])] == before["cells"]
    assert updated["metadata"] == before["metadata"]
    ids = [cell["id"] for cell in updated["cells"]]
    assert ids.index("phase3-existing") < ids.index(cells[0]["id"]) < ids.index("phase4")
    assert view.insert_phase3_supplement(updated, cells) == updated
    updated["cells"][4]["source"] = "old supplement version"
    assert view.insert_phase3_supplement(updated, cells)["cells"][4]["source"] == cells[0]["source"]


@pytest.mark.parametrize("bad", ["missing_heading", "wrong_order", "colliding_id"])
def test_bad_notebook_order_or_id_is_rejected(bad):
    document = notebook()
    cell = {"id": "supplement", "cell_type": "markdown", "source": "new", "metadata": {"tags": [view.SUPPLEMENT_TAG]}}
    if bad == "missing_heading":
        document["cells"][4]["source"] = "No phase heading"
    elif bad == "wrong_order":
        document["cells"][1], document["cells"][4] = document["cells"][4], document["cells"][1]
    else:
        cell["id"] = "intro"
    with pytest.raises(analysis.ArtifactError):
        view.insert_phase3_supplement(document, [cell])


def test_cli_active_writer_rejected_before_report_loading(tmp_path, monkeypatch):
    from mnist_experiment.akaike_score_gate.artifacts import run_lock

    root, target = complete_root(tmp_path)
    previous = target.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("Active writer must be rejected before artifact validation")
    monkeypatch.setattr(analysis, "load_report", forbidden)
    with run_lock(root):
        lock_contents = (root / ".run.lock").read_bytes()
        with pytest.raises(RuntimeError, match="writer already owns"):
            view.main(["--output-root", str(root), "--notebook", str(target)])
        assert (root / ".run.lock").read_bytes() == lock_contents
    assert target.read_bytes() == previous


def test_cli_paused_later_phase_renders_idempotently_without_tensor_loads_or_mutation(tmp_path, monkeypatch, capsys):
    import torch

    root, target = complete_root(tmp_path)
    # A tensor artifact can be checksum-verified, but must never be deserialized.
    unit = next(root.glob("phase3/calibration_audit-*"))
    (unit / "diagnostics.pt").write_bytes(b"not a tensor serialization")
    integrity = json.loads((unit / "integrity.json").read_text())
    integrity["diagnostics.pt"] = hashlib.sha256((unit / "diagnostics.pt").read_bytes()).hexdigest()
    write_json(unit / "integrity.json", integrity)
    before = snapshot(root)
    original = json.loads(target.read_text())
    def forbidden(*args, **kwargs):
        pytest.fail("No tensor load or CUDA call is permitted")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    for _ in range(2):
        view.main(["--output-root", str(root), "--notebook", str(target)])
        data = target.read_bytes()
        if _ == 0:
            first = data
        else:
            assert data == first
    assert str(target) in capsys.readouterr().out
    assert snapshot(root) == before
    document = json.loads(first)
    kept = [cell for cell in document["cells"] if view.SUPPLEMENT_TAG not in cell.get("metadata", {}).get("tags", [])]
    assert kept == original["cells"] and document["metadata"] == original["metadata"]
    own = [cell for cell in document["cells"] if view.SUPPLEMENT_TAG in cell.get("metadata", {}).get("tags", [])]
    assert len(own) == 2 and len(own[1]["outputs"]) == 4
    assert all(base64.b64decode(output["data"]["image/png"]).startswith(b"\x89PNG\r\n\x1a\n") for output in own[1]["outputs"])
    assert plt.get_fignums() == []


@pytest.mark.parametrize("bad", ["not_completed", "incomplete_phase3", "missing_audit_assignment", "corrupt_artifact", "missing_state"])
def test_cli_incomplete_or_invalid_phase3_does_not_replace_notebook(tmp_path, bad):
    root, target = complete_root(tmp_path)
    if bad == "not_completed":
        write_json(root / "progress.json", {"phases": {"phase3": {"status": "running"}}})
    elif bad == "incomplete_phase3":
        (root / "phase3" / ".incomplete" / "paused").mkdir(parents=True)
    elif bad == "missing_audit_assignment":
        unit = next(root.glob("phase3/decision-*"))
        decision = json.loads((unit / "decision.json").read_text())
        decision["audit_paths"].pop()
        write_json(unit / "decision.json", decision)
        reseal(unit)
    elif bad == "missing_state":
        study = json.loads((root / "study.json").read_text())
        study["history_variance_states"] = 8
        write_json(root / "study.json", study)
    else:
        unit = next(root.glob("phase3/calibration_audit-*"))
        write_json(unit / "summary.json", {"corrupt": True})
    before, previous = snapshot(root), target.read_bytes()
    with pytest.raises(analysis.ArtifactError):
        view.supplement_notebook(root, target)
    assert snapshot(root) == before and target.read_bytes() == previous


def test_cli_rejects_artifact_destination_before_acquiring_lock(tmp_path):
    root, _ = complete_root(tmp_path)
    before = snapshot(root)
    with pytest.raises(analysis.ArtifactError, match="outside study artifacts"):
        view.supplement_notebook(root, root / "study.json")
    assert snapshot(root) == before and not (root / ".run.lock").exists()


def test_helpers_consume_report_without_reopening_artifacts(tmp_path, monkeypatch):
    root = scalar_root(tmp_path, states=1)
    add_audit(root, "rotation_linear")
    report = analysis.load_report(root)
    original = Path.open
    def guard(path, *args, **kwargs):
        if path.resolve().is_relative_to(root.resolve()):
            pytest.fail("Scalar helper reopened a study artifact")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    view.build_phase3_supplement(report, render=False)
    assert view.summarize_phase3(report)["fit_health"][0]["covariance_denominator"] == 31


def test_helper_import_does_not_import_torch_or_execution_modules():
    script = ("import sys\n"
              "from mnist_experiment import akaike_score_gate_reporting\n"
              "assert 'torch' not in sys.modules\n"
              "assert 'mnist_experiment.akaike_score_gate.calibration_audit' not in sys.modules\n"
              "assert 'mnist_experiment.akaike_score_gate.trajectory' not in sys.modules\n"
              "assert 'mnist_experiment.akaike_score_gate.artifacts' not in sys.modules\n")
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
