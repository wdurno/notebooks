"""Publication preservation and resume of nonlinear efficiency inference."""

from copy import deepcopy
import json

import nbformat
import pytest

from mnist_experiment.fisher_memory_shift import analysis, report
from mnist_experiment.fisher_memory_shift.config import Config
from test_plan15_shift_analysis import cohort


def test_bootstrap_blocks_resume_identically_and_skip_completed_inversions(monkeypatch):
    records, manifest = cohort()
    saved = {}
    ordinary = analysis.analyze(records, manifest, seed=51, draws=250)
    first = analysis.analyze(records, manifest, seed=51, draws=250,
        load_bootstrap=saved.get, save_bootstrap=lambda start, value: saved.update({start:value}))
    assert ordinary == first
    assert sorted(saved) == [0, 100, 200]
    calls, original = [], analysis._point
    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(analysis, "_point", measured)
    again = analysis.analyze(records, manifest, seed=51, draws=250,
                             load_bootstrap=saved.get)
    assert again == ordinary and len(calls) == 1
    damaged = deepcopy(saved)
    damaged[100]["indices"][0][0] = (damaged[100]["indices"][0][0] + 1) % 12
    with pytest.raises(ValueError, match="checkpoint"):
        analysis.analyze(records, manifest, seed=51, draws=250, load_bootstrap=damaged.get)


def pending():
    return dict(config=Config().mapping(),
        progress=dict(status="implementation", stage="9A", completed=0, planned=410,
                      active_seconds=0., heartbeat="pending"),
        contexts=[], initialization=None, readiness=None, development={},
        selection=None, analysis=None, confirmation=[], changes=[])


def test_report_preserves_historical_cells_and_is_idempotent(tmp_path):
    path = tmp_path/"results.ipynb"
    historical = [
        nbformat.v4.new_markdown_cell("# Original results", id="original-title"),
        nbformat.v4.new_code_cell("preserved = 3", execution_count=7,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="original\n")],
            id="original-output"),
        nbformat.v4.new_markdown_cell("Phase 8.1E conclusion", id="phase81e-original")]
    book = dict(nbformat.v4.new_notebook(cells=historical))
    path.write_text(json.dumps(book))
    report.publish(pending(), path)
    first = json.loads(path.read_text())
    assert [c for c in first["cells"] if not report.owned(c)] == historical
    assert first["cells"][0]["id"] == "original-title"
    assert first["cells"][1]["id"] == "plan15-phase9-progress"
    assert next(i for i,c in enumerate(first["cells"]) if c["id"] == "phase81e-original") < next(
        i for i,c in enumerate(first["cells"]) if c["id"] == "plan15-phase9-intro")
    report.publish(pending(), path)
    assert json.loads(path.read_text()) == first
    first["metadata"]["plan15_phase8_presentation"] = {"role":"frozen_archive"}
    path.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="frozen"):
        report.publish(pending(), path)


def test_significance_is_not_implied_when_matched_retention_is_unresolved():
    records, manifest = cohort()
    result = analysis.analyze(records, manifest, seed=7, draws=50)
    for item in result["methods"].values():
        for domain in ("base","novel"):
            curves = item["curves"][domain]
            curves["mean_mae"] = [value/2 for value in curves["mean_rmse"]]
            curves["mean_accuracy10"] = [.4]*len(curves["mean_rmse"])
    result["efficiency"]["retention_qualified"] = False
    snapshot = pending()
    snapshot["analysis"] = result
    cells = report.build_cells(snapshot)
    text = "\n".join(str(c["source"]) for c in cells)
    assert "retention-unqualified" in text
    assert "within" in text.lower() and "10 degrees" in text
    assert any(c.get("attachments") for c in cells)


def test_report_exposes_small_improvement_behind_a_savings_score():
    records, manifest = cohort()
    result = analysis.analyze(records, manifest, seed=7, draws=20)
    for item in result["methods"].values():
        for domain in ("base", "novel"):
            curves = item["curves"][domain]
            curves["mean_mae"] = [value/2 for value in curves["mean_rmse"]]
            curves["mean_accuracy10"] = [.4]*len(curves["mean_rmse"])
    result["efficiency"]["point"].update(useful_target=False, target_mse_gain_fraction=.01)
    snapshot = pending()
    snapshot["analysis"] = result
    text = "\n".join(str(c["source"]) for c in report.build_cells(snapshot))
    assert "1.00%" in text and "does not pass the prespecified 5%" in text
    assert "not an established practical-efficiency win" in text


def test_unresolved_inverse_does_not_hide_supported_full_budget_retention():
    records, manifest = cohort()
    result = analysis.analyze(records, manifest, seed=7, draws=20)
    for item in result["methods"].values():
        for domain in ("base", "novel"):
            curves = item["curves"][domain]
            curves["mean_mae"] = [value/2 for value in curves["mean_rmse"]]
            curves["mean_accuracy10"] = [.4]*len(curves["mean_rmse"])
    result["efficiency"]["point"].update(status="target_not_reached_at_observed_endpoint", S=None)
    result["efficiency"].update(retention_qualified=False, full_budget_retention_supported=True)
    result["efficiency"]["bootstrap"].update(invalid=20, ci=None)
    result["calibration_cost"] = {"development_trajectories": 304}
    snapshot = pending()
    snapshot["analysis"] = result
    text = "\n".join(str(c["source"]) for c in report.build_cells(snapshot))
    assert "20/20 bootstrap matches are unresolved" in text
    assert "Full-budget retention is supported; matched-budget retention is unresolved" in text
    assert "raw score is retention-unqualified" not in text
    assert "not distinct label budgets" in text
