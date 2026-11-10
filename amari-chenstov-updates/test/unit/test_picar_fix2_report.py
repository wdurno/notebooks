import json

import nbformat
import pytest

from numerical_experiment.picar import phase1_fix2_report, report


def test_settings_groups_units_but_keeps_distinct_versions_and_science_status():
    def row(version, smoke):
        return {"config": {"mode": "production", "version": version,
                           "model": {"kind": "qwen"}}, "smoke_only": smoke}
    rows = report.settings_rows({"complete_units": [row("v1", True), row("v1", True),
                                                   row("v2", True), row("v2", False)]})
    assert [r["completed_units"] for r in rows] == [2, 1, 1]
    assert [r["version"] for r in rows] == ["v1", "v2", "v2"]
    assert [r["smoke_only"] for r in rows] == [True, True, False]


def test_earlier_compute_avoids_double_charging_repair_campaigns():
    snapshot = {"complete_units": [
        {"summary": {"kind": "phase1_resource", "active_seconds": 400.}},
        {"summary": {"kind": "phase1_fix_campaign", "active_seconds": 20.}},
        {"summary": {"kind": "phase1_fix2_forecast", "active_seconds": 2.}},
        {"summary": {"kind": "phase0_software", "active_seconds": 100.}}],
        "diagnostics": [{"summary": {"active_seconds": 60.}}]}
    assert phase1_fix2_report.earlier_compute_seconds(snapshot) == 560.


def test_incomplete_fix2_has_a_pending_conclusion_and_chronological_section():
    snapshot = {"complete_units": [], "live": [{"path": "phase1/fix2-v2/orchestration", "status": "running"}]}
    book = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("### Phase 1 Fix 1"),
        nbformat.v4.new_markdown_cell("## Development"),
        nbformat.v4.new_markdown_cell("## Conclusion")])
    phase1_fix2_report.append_fix2(book, snapshot)
    assert "Fix 1" in book.cells[0].source
    assert "Fix 2" in book.cells[1].source
    assert book.cells[-1].source == "## Conclusion"
    assert "not yet complete" in phase1_fix2_report.conclusion(snapshot)


def test_failed_notebook_refresh_preserves_previous_readable_file(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    output = tmp_path / "results.ipynb"
    output.write_text(json.dumps({"previous": "reader can still open this"}))
    before = output.read_bytes()
    def failed_write(book, path):
        path.write_text("incomplete temporary document")
        raise OSError("simulated write failure")
    monkeypatch.setattr(nbformat, "write", failed_write)
    with pytest.raises(OSError, match="simulated"):
        report.build_notebook(root, output)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".plan19-report-*"))


def test_stopped_attempt_is_not_displayed_as_an_active_job(tmp_path):
    from numerical_experiment.picar.config import PicarConfig
    from numerical_experiment.picar.storage import RunStore
    campaign = tmp_path / "fix2-old"
    with RunStore(campaign / "orchestration", {"config": PicarConfig().to_dict()}) as store:
        store.checkpoint({}, progress={"phase": 1, "status": "old timing unit", "counts": {"step": 2}})
    (campaign / "STOP").touch()
    snapshot = report.load_report(tmp_path)
    phase = report.progress_rows(snapshot)[1]
    assert phase["stopped_units"] == phase["incomplete_units"] == 1
    assert "stopped attempts retained" in phase["status"]
