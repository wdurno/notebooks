"""Small scalar/media fixtures only: no runner, physics, models, or GPU."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
import pytest

from numerical_experiment.picar import phase1_fix3_report as report
from numerical_experiment.picar.schemas import METRIC_COLUMNS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def seal(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def unit(path, kind, **fields):
    config = {"config": {"schema_version": 1, "version": "fixture", "mode": "smoke"},
              "campaign_id": path.parent.name}
    write_json(path / "config.json", config)
    write_json(path / "summary.json", {"kind": "phase1_fix3_" + kind, **fields})
    (path / "metrics.csv").write_text(",".join(METRIC_COLUMNS) + "\n")
    manifest = {"artifact_schema_version": 1, "metric_schema_version": 1, "storage_schema_version": 1,
                "status": "completed", "completed_at": "2026-11-11T10:00:00Z",
                "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "final_files": {name: seal(path / name) for name in ("summary.json", "metrics.csv")}}
    write_json(path / "manifest.json", manifest)
    write_json(path / "COMPLETED", {"manifest_sha256": seal(path / "manifest.json")["sha256"]})
    return path


def rendered(cells):
    parts = []
    for cell in cells:
        parts.append(cell.source)
        for output in cell.get("outputs", []):
            parts.append(output.get("text", output.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


def test_pending_root_produces_valid_refreshable_cells_without_creating_it(tmp_path, monkeypatch):
    root = tmp_path / "absent"
    cells = report.cells(root)
    assert "Phase 1 Fix 3 pending" in rendered(cells)
    assert "teacher and evaluator remain frozen" in rendered(cells)
    assert not root.exists()
    nbformat.validate(nbformat.v4.new_notebook(cells=cells))
    assert all(cell.metadata["plan19_fix3"] for cell in cells)
    calls = []
    monkeypatch.setattr(report, "display_fix3", lambda path: calls.append(path))
    exec(cells[-1].source, {})
    assert calls == [root]


def test_live_progress_does_not_promote_partial_summaries_or_touch_checkpoints(tmp_path):
    campaign = tmp_path / "phase1" / "fix3-v1"
    write_json(campaign / "orchestration" / "progress.json", {
        "status": "collecting", "pending_units": ["pilot"], "active_seconds": 12,
        "remaining_estimate_seconds": 90, "settings": {"eta": .5, "exposure_ramp": [32, 256]}})
    write_json(campaign / "unfinished" / "summary.json", {
        "kind": "phase1_fix3_pilot", "seed": "MUST_NOT_REPORT", "acquired_count": 999})
    (campaign / "unfinished" / "checkpoint.pt").write_bytes(b"never unpickle me")
    before = {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    text = rendered(report.cells(tmp_path))
    assert "collecting" in text and "90" in text and "Curriculum ramp" in text and "32--256 intervals" in text
    assert "MUST_NOT_REPORT" not in text and "999" not in text
    assert "Student pilot / fixed-panel evaluations / before-after clips: pending" in text
    assert before == {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}


def test_complete_diagnostics_render_geometry_components_and_paired_difficulty_counts(tmp_path):
    campaign = tmp_path / "phase1" / "fix3-v1"
    write_json(campaign / "orchestration" / "progress.json", {
        "status": "pilot complete", "active_seconds": 120, "blockers": ["confirmation pending"]})
    unit(campaign / "geometry", "geometry_audit", rows=[
        {"exposure": 0, "sigma": .8, "clear_fraction": .5, "blocked_fraction": .5, "alpha": 0},
        {"exposure": 160, "sigma": .2, "clear_fraction": .2, "blocked_fraction": .8, "alpha": 1},
        {"exposure": 320, "sigma": .1, "clear_fraction": .05, "blocked_fraction": .95, "alpha": 0}],
        densities=[{"y": y, "early": .5, "middle": .6, "late": .8} for y in (-1, 0, 1)])
    reward = unit(campaign / "rewards", "rewards", rows=[
        {"state": "forward", "seed": 1, "split": "heldout", "frame_ref": "frame.npy", "visible": True,
         "proximity": 5, "centering": 1, "pan_rad": 0, "centering_bonus": .5, "score": 5.5},
        {"state": "pan-only", "seed": 1, "split": "heldout", "frame_ref": "missing.npy", "visible": True,
         "proximity": 5, "centering": 1, "pan_rad": .2, "centering_bonus": 0, "score": 5}])
    np.save(reward / "frame.npy", np.zeros((4, 6, 3), dtype=np.uint8), allow_pickle=False)
    unit(campaign / "teacher", "teacher", seed=7, variant="revised", success=False,
         end_reason="wall_collision", commands=["left", "forward"], steering_corrections=1)
    unit(campaign / "bootstrap", "bootstrap", transitions=30, distillation={"status": "completed"})
    pilot = unit(campaign / "pilot", "pilot", seed=7, acquired_count=320, episodes=20,
        evaluations=[
            {"checkpoint": "initial", "difficulty": "easy", "rows": [
                {"success": False, "end_reason": "wall_collision"},
                {"success": False, "end_reason": "timeout"}]},
            {"checkpoint": "final", "difficulty": "easy", "rows": [
                {"success": True, "end_reason": "target_contact"},
                {"success": False, "end_reason": "timeout"}]},
            {"checkpoint": "final", "difficulty": "hard", "rows": [
                {"success": False, "collision": True}]}],
        videos=[{"checkpoint": checkpoint, "layout_id": "common-layout", "view": "ego/top-down",
                 "path": f"{checkpoint}.mp4"} for checkpoint in ("initial", "final")])
    for checkpoint in ("initial", "final"):
        (pilot / f"{checkpoint}.mp4").write_bytes(b"links only; no video decoding")
    cells = report.cells(tmp_path)
    text = rendered(cells)
    assert "clear_fraction" in text and "sigma" in text
    assert "separate texture/gain alpha" in text
    assert all(field in text for field in ("pan-only", "proximity", "centering", "pan_rad", "centering_bonus"))
    assert "revised" in text and "wall_collision" in text and "Distillation initial KL" in text
    assert "0/2 (0.0%)" in text and "1/2 (50.0%)" in text and "1/1 (100.0%)" in text
    assert "middle" in text and "pending / not recorded" in text
    assert "confirmation pending" in text and "No STM efficacy" in text
    outputs = cells[-1].outputs
    assert any("image/png" in o.get("data", {}) for o in outputs)
    markup = "\n".join(o.get("data", {}).get("text/html", "") for o in outputs)
    assert "data:image/png;base64," in markup
    assert "initial.mp4" in markup and "final.mp4" in markup and "common-layout" in markup
    nbformat.validate(nbformat.v4.new_notebook(cells=cells))


def test_missing_results_are_unknown_not_failures_and_seeds_are_not_pooled(tmp_path):
    campaign = tmp_path / "fix3-v1"
    unit(campaign / "pilot", "pilot", seed=11, evaluations=[
        {"checkpoint": "initial", "difficulty": "easy", "rows": [{"success": True}, {}]},
        {"checkpoint": "final", "difficulty": "hard", "rows": [{"end_reason": "interrupted"}]}])
    unit(campaign / "pilot2", "pilot", seed=12, evaluations=[
        {"checkpoint": "initial", "difficulty": "easy", "rows": [{"success": False}]}])
    text = rendered(report.cells(tmp_path))
    assert "1/1 (100.0%); 1 unknown" in text
    assert "0/1 (0.0%)" in text and "1/3" not in text
    assert "initial: pending" in text and "final: pending" in text
    assert "pairing unverified" in text
    assert report._rate([{"end_reason": "interrupted"}], "collision") == report.PENDING
    assert report._rate([{"success": "false"}], "success") == report.PENDING


def test_campaign_revisions_are_separate_and_stopped_attempt_is_not_running(tmp_path):
    for version, count in (("old", 3), ("new", 4)):
        campaign = tmp_path / f"fix3-{version}"
        unit(campaign / "bootstrap", "bootstrap", transitions=count)
        write_json(campaign / "orchestration" / "progress.json", {"status": "running"})
    (tmp_path / "fix3-old" / "STOP").touch()
    text = rendered(report.cells(tmp_path))
    assert "Campaign: fix3-old" in text and "Campaign: fix3-new" in text
    assert text.count("revisions are not pooled") == 2
    assert "stopped" in text
    assert text.count("Completed units (sealed)") == 2


def test_corrupt_or_incompatible_completed_scalar_artifacts_fail_without_repair(tmp_path):
    from numerical_experiment.picar.report import ArtifactError

    path = unit(tmp_path / "fix3-v1" / "teacher", "teacher", success=True)
    (path / "summary.json").write_text('{"kind":"phase1_fix3_teacher","schema_version":999}')
    with pytest.raises(ArtifactError, match="summary schema"):
        report.cells(tmp_path)
    (path / "summary.json").write_text('{"kind":"phase1_fix3_teacher","success":false}')
    with pytest.raises(ArtifactError, match="hash/size mismatch"):
        report.cells(tmp_path)


def test_contact_sheet_handles_missing_fields_safe_frames_and_escaped_labels(tmp_path):
    campaign = tmp_path / "fix3-v1"
    frame_unit = unit(campaign / "panels", "panels")
    (frame_unit / "frames").mkdir()
    frame = frame_unit / "frames" / "temporary.npy"
    np.save(frame, np.ones((3, 4, 3), dtype=np.uint8), allow_pickle=False)
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    frame.rename(frame.with_name(digest + ".npy"))
    unit(campaign / "rewards", "rewards", rows=[
        {"state": "<script>bad</script>", "frame_ref": "sha256:" + digest},
        {"state": "outside", "frame_ref": str(tmp_path / "outside.npy")}])
    np.save(tmp_path / "outside.npy", np.zeros((2, 2, 3), dtype=np.uint8), allow_pickle=False)
    cells = report.cells(campaign)
    markup = "\n".join(o.get("data", {}).get("text/html", "") for o in cells[-1].outputs)
    assert "<script>" not in markup and "&lt;script&gt;" in markup
    assert markup.count("data:image/png;base64,") == 1
    assert report.PENDING in markup


def test_full_artifact_render_never_imports_model_stack_or_loads_pickle(tmp_path):
    campaign = tmp_path / "fix3-v1"
    unit(campaign / "geometry", "geometry_audit", rows=[], densities=[])
    reward = unit(campaign / "rewards", "rewards", rows=[{"frame_ref": "frame.npy"}])
    np.save(reward / "frame.npy", np.zeros((3, 4, 3), dtype=np.uint8), allow_pickle=False)
    code = (
        "import sys, pickle\n"
        "def forbidden(*args, **kwargs): raise AssertionError('pickle loading forbidden')\n"
        "pickle.load = pickle.loads = forbidden\n"
        "from pathlib import Path\n"
        "import nbformat\n"
        "from numerical_experiment.picar.phase1_fix3_report import cells, append_fix3, conclusion\n"
        "from numerical_experiment.picar.report import load_report\n"
        f"assert cells(Path({str(tmp_path)!r}))\n"
        f"snapshot = load_report(Path({str(tmp_path)!r}))\n"
        "assert append_fix3(nbformat.v4.new_notebook(), snapshot).cells\n"
        "assert conclusion(snapshot)\n"
        "assert not any(x in sys.modules for x in ('torch', 'transformers', 'pybullet', 'tkinter', "
        "'numerical_experiment.picar.storage', 'numerical_experiment.picar.phase1_fix3'))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")}, timeout=30)
    assert result.returncode == 0, result.stderr


def test_snapshot_append_is_idempotent_chronological_and_independently_executable(tmp_path, monkeypatch):
    from numerical_experiment.picar.report import load_report

    campaign = tmp_path / "phase1" / "fix3-v1"
    unit(campaign / "teacher", "teacher", seed=1, variant="revised", success=False)
    snapshot = load_report(tmp_path)
    book = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("### Phase 1 Fix 2"),
        nbformat.v4.new_markdown_cell("## Development"),
        nbformat.v4.new_markdown_cell("## Conclusion")])
    assert report.append_fix3(book, snapshot) is book
    report.append_fix3(book, snapshot)
    assert len(book.cells) == 5
    assert "Fix 2" in book.cells[0].source and "Fix 3" in book.cells[1].source
    assert book.cells[3].source == "## Development" and book.cells[4].source == "## Conclusion"
    calls = []
    monkeypatch.setattr(report, "display_fix3", lambda value: calls.append(value))
    exec(book.cells[2].source, {"snapshot": snapshot})
    assert calls == [snapshot]
    nbformat.validate(book)
    assert "pending: geometry, rewards, bootstrap, pilot" in report.conclusion(snapshot)
    assert "No STM efficacy or sample-efficiency result" in report.conclusion(snapshot)


def test_snapshot_only_live_campaign_and_absent_fix3_conclusions(tmp_path):
    empty = {"complete_units": [], "live": []}
    book = nbformat.v4.new_notebook()
    assert report.append_fix3(book, empty) is book and not book.cells
    assert report.conclusion(empty) is None
    live = {"complete_units": [], "live": [
        {"path": str(tmp_path / "fix3-v2" / "orchestration"), "status": "running", "stopped": True}]}
    report.append_fix3(book, live)
    assert "stopped" in rendered(book.cells)
    assert "stopped" in report.conclusion(live)
    assert "0 sealed diagnostic units" in report.conclusion(live)


def test_snapshot_entrypoint_uses_validated_units_without_reopening_summary(tmp_path):
    path = tmp_path / "fix3-v1" / "teacher"
    snapshot = {"complete_units": [{"path": str(path), "summary": {
        "kind": "phase1_fix3_teacher", "seed": 3, "variant": "revised", "success": True}}], "live": []}
    assert "revised" in rendered(report.append_fix3(nbformat.v4.new_notebook(), snapshot).cells)
    assert not path.exists()


def test_fixed_panel_identity_is_checked_instead_of_assumed():
    def evaluation(checkpoint, ids):
        return {"checkpoint": checkpoint, "difficulty": "easy", "rows": [
            {"layout_id": layout, "success": True, "collision": False} for layout in ids]}
    matched = report._evaluations({"evaluations": [evaluation("initial", [1, 2]), evaluation("final", [2, 1])]})
    assert matched[0]["Panel identity"] == "matched layout IDs"
    changed = report._evaluations({"evaluations": [evaluation("initial", [1, 2]), evaluation("final", [1, 3])]})
    assert changed[0]["Panel identity"] == "layout IDs differ"


def test_actual_geometry_histograms_use_early_middle_late_rows_without_recomputing(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from numerical_experiment.picar.report import load_report

    campaign = tmp_path / "fix3-v1"
    rows = [{"exposure": exposure, "sigma": .8 - index * .1,
             "clear_fraction": .5 - index * .05, "blocked_fraction": .5 + index * .05,
             "lateral_density_per_m": [index + 1, index + 2], "lateral_histogram_counts": [10, 20]}
            for index, exposure in enumerate((0, 32, 80, 144, 192, 256, 320))]
    unit(campaign / "geometry", "geometry", rows=rows, lateral_histogram_edges_m=[-1, 0, 1],
         settings={"curriculum_total_intervals": 320, "barrier_sigma_easy": 1., "barrier_sigma_hard": .08})
    calls = []
    original = Axes.stairs

    def stairs(self, values, edges, **kwargs):
        calls.append((values, edges))
        return original(self, values, edges, **kwargs)

    monkeypatch.setattr(Axes, "stairs", stairs)
    snapshot = load_report(tmp_path)
    book = report.append_fix3(nbformat.v4.new_notebook(), snapshot)
    assert calls == [(rows[i]["lateral_density_per_m"], [-1, 0, 1]) for i in (0, 3, 6)]
    text = rendered(book.cells)
    assert "Curriculum ramp" in text and "320 total intervals" in text
    assert "lateral_histogram_counts" not in text
    assert "pending: rewards, teacher, bootstrap, pilot" in report.conclusion(snapshot)


def test_progress_schema_and_optional_corrupt_image_are_not_silently_accepted(tmp_path):
    from numerical_experiment.picar.report import ArtifactError

    campaign = tmp_path / "fix3-v1"
    progress = campaign / "orchestration" / "progress.json"
    write_json(progress, {"schema_version": 999})
    with pytest.raises(ArtifactError, match="Fix 3 progress schema"):
        report.cells(tmp_path)
    write_json(progress, {"schema_version": 1, "counts": {"step": 2, "target": 23}})
    reward = unit(campaign / "rewards", "rewards", rows=[{"frame_ref": "empty.npy"}])
    (reward / "empty.npy").touch()
    text = rendered(report.cells(tmp_path))
    assert "Pending units" in text and "21" in text
    assert "unavailable frames remain pending" in text


def test_human_report_aggregates_large_payloads_and_bounds_settings_geometry(tmp_path, monkeypatch):
    tables = []
    original = report._table

    def table(rows):
        tables.append(rows)
        assert not any(isinstance(value, (dict, list, tuple)) for row in rows for value in row.values())
        return original(rows)

    monkeypatch.setattr(report, "_table", table)
    monkeypatch.setattr(report, "_geometry", lambda summary: report._note("Recorded geometry figure"))
    campaign = tmp_path / "fix3-v1"
    def summary(kind, **fields):
        return {"path": str(campaign / kind), "summary": {"kind": "phase1_fix3_" + kind, **fields},
                "stored_config": {"config": {"model": {"context_length": 4, "return_bins": 161},
                                             "learning": {"replay_capacity": 200}}}}
    teacher = [summary("teacher", variant=variant, success=index < 2, end_reason="timeout",
        commands=[{"raw_text": "RAW_COMMAND_SENTINEL", "probabilities": [0.] * 8}] * 100,
        action_counts={"0": 90, "1": 10}) for variant in ("ported", "centering") for index in range(6)]
    units = [summary("geometry", settings={**{f"unused_{i}": i for i in range(25)},
        "curriculum_total_intervals": 320}, rows=[{"exposure": i} for i in range(7)]), *teacher,
        summary("bootstrap", transitions=300, distillation={"initial_kl": 1.2, "final_kl": .3,
            "history": [{"data": "DISTILL_HISTORY_SENTINEL"}] * 1000}),
        summary("pilot", seed=1, acquired_count=320,
                episodes=[{"data": "RAW_EPISODE_SENTINEL"}] * 40)]
    snapshot = {"complete_units": units, "live": []}
    text = rendered(report.append_fix3(nbformat.v4.new_notebook(), snapshot).cells)
    assert not any(marker in text for marker in ("RAW_COMMAND_SENTINEL", "DISTILL_HISTORY_SENTINEL", "RAW_EPISODE_SENTINEL"))
    assert "unused_" not in text
    assert len(tables[0]) == len(tables[1]) == 1
    assert tables[1] == [{"Context": 4, "Replay capacity": 200, "Return bins": 161, "Curriculum ramp": "320 total intervals"}]
    teacher_rows = next(rows for rows in tables if rows and "Variant" in rows[0])
    assert len(teacher_rows) == 2
    assert all(r["Success count / rate"] == "2/6 (33.3%)" and r["Commands total"] == 600
               and r["Unique actions"] == 2 for r in teacher_rows)
    boot = next(rows[0] for rows in tables if rows and "Transitions" in rows[0])
    assert boot == {"Transitions": 300, "Distillation initial KL": 1.2, "Distillation final KL": .3}
    pilot = next(rows[0] for rows in tables if rows and "Training episodes" in rows[0])
    assert pilot["Training episodes"] == 40 and pilot["Acquired transitions"] == 320
    assert report._compact([{"private": "never render"}]) == "1 archived entries"


def test_conclusion_describes_actual_fixed_panel_rates_and_final_blockers(tmp_path):
    campaign = tmp_path / "fix3-v1"
    def evaluation(checkpoint, successes):
        return {"checkpoint": checkpoint, "difficulty": "easy", "rows": [
            {"layout_id": i, "success": success, "collision": not success} for i, success in enumerate(successes)]}
    snapshot = {"complete_units": [
        {"path": str(campaign / "pilot"), "summary": {"kind": "phase1_fix3_pilot", "seed": 9,
            "evaluations": [evaluation("initial", [False, False]), evaluation("final", [True, False])]}},
        {"path": str(campaign / "orchestration"), "summary": {"kind": "phase1_fix3_campaign",
            "status": "ready_for_checkin", "blockers": ["Fresh confirmation required", "Runtime forecast pending"]}}], "live": []}
    text = report.conclusion(snapshot)
    assert "ready_for_checkin" in text and "Student seed 9" in text
    assert "easy success 0/2 (0.0%) to 1/2 (50.0%)" in text
    assert "collisions 2/2 (100.0%) to 1/2 (50.0%)" in text
    assert "middle pending" in text and "hard pending" in text
    assert "Fresh confirmation required; Runtime forecast pending" in text
