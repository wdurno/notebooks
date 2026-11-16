"""Small sealed scalar fixtures; no model, Torch, downloads or GPU work."""

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

from numerical_experiment.picar import report as shared
from numerical_experiment.picar.schemas import METRIC_COLUMNS
from numerical_experiment.picar_solver_repair import report


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def seal(path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def unit(root, kind, *, name=None, campaign=report.CAMPAIGN, **fields):
    path = root / "phase3" / campaign / "units" / (name or kind)
    path.mkdir(parents=True)
    config = {"config": {"schema_version": 1, "version": report.CAMPAIGN_VERSIONS[campaign], "mode": "production"}}
    write_json(path / "config.json", config)
    write_json(path / "summary.json", {"schema_version": 1, "kind": "solver_" + kind,
        "phase": 3, "smoke_only": True, "development_only": True, **fields})
    with (path / "metrics.csv").open("w") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_COLUMNS).writeheader()
    manifest = {"artifact_schema_version": 1, "metric_schema_version": 1, "storage_schema_version": 1,
        "status": "completed", "completed_at": "2026-11-14T12:00:00Z",
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "final_files": {key: seal(path / key) for key in ("summary.json", "metrics.csv")}}
    write_json(path / "manifest.json", manifest)
    write_json(path / "COMPLETED", {"manifest_sha256": seal(path / "manifest.json")["sha256"]})
    return path


def live(root, *, name="orchestration", schema=True, campaign=report.CAMPAIGN, **fields):
    path = root / "phase3" / campaign / name
    path.mkdir(parents=True)
    version = report.CAMPAIGN_VERSIONS[campaign]
    write_json(path / "config.json", {"schema_version": 1, "version": version})
    write_json(path / "manifest.json", {"artifact_schema_version": 1, "metric_schema_version": 1, "status": "running"})
    transaction = path / "transactions" / "commit-1"
    transaction.mkdir(parents=True)
    value = {**({"schema_version": 1} if schema else {}), "version": version, "phase": 3, **fields}
    write_json(transaction / "progress.json", value)
    write_json(transaction / "manifest.json", {"files": {"progress.json": seal(transaction / "progress.json")}})
    write_json(path / "CURRENT.json", {"storage_schema_version": 1, "commit_id": "commit-1", "manifest": seal(transaction / "manifest.json")})
    write_json(path / "progress.json", {"status": "STALE uncommitted mirror", "repair_active_seconds": 999999.})
    return path


def recorded_fields(recipe="proximal_1", condition="replay_ewc"):
    validations, events, segments = [], [], []
    for index, segment in enumerate(("early", "late")):
        panels = [{"segment": segment, "events": n, "cumulative_fit_seconds": index * 4. + n * 2.,
                   "old_nll": 2. + n * .1, "live_nll": 4. - n * .2} for n in (0, 1, 2)]
        validations.extend(panels)
        segments.append({"segment": segment, "events": 2, "validation_initial": panels[0],
                         "validation_final": panels[-1], "start_state": "common_pre_stream_asset",
                         "historical_policy_continuation": False})
        events.extend({"segment": segment, "event_id": n, "update_seconds": 2.,
                       "first_step_ewc_suppression": .25 + n * .1,
                       "inner_steps": [{"step_cosine": -.2, "stiffness_max": 1.2}],
                       "post_centered_scaled_gradient_residual": .5 / n,
                       "private_trace_payload": "trace-not-narrative", "gpu_seconds": .3}
                      for n in (1, 2))
    return {"status": "complete", "solver_recipe": recipe, "condition": condition,
            "settings": {"solver_recipe": recipe, "condition": condition, "learning_rate": .003},
            "steps_per_event": report.RECIPES[recipe], "completed_events": 4, "fit_seconds": 8.,
            "validation": {"initial": {"old_nll": 2., "live_nll": 4.}, "final": {"old_nll": 2.2, "live_nll": 3.6}},
            "timing": {"mean_seconds": 2., "median_seconds": 2., "p95_seconds": 2., "fit_seconds": 6.,
                       "diagnostic_fit_seconds": 2., "gpu_seconds": 1.2, "peak_vram_bytes": 1024,
                       "gradient_evaluations": 3, "timing_policy": "exclude_warmup_and_diagnostics"},
            "segments": segments, "rows": events, "validation_rows": validations}


def test_live_forecast_uses_checks_then_completed_pilots(monkeypatch):
    monkeypatch.setattr(report, "_prior_panel_seconds", lambda units: 100.)
    units = [{"summary": {"kind": "solver_readiness", "condition": c,
                          "active_seconds": 48., "acquired_count": 48}}
             for c in ("replay_ewc", "stm_fd")]
    assert report._forecast_seconds(units) == [2220., 4440.]
    units.append({"summary": {"kind": "solver_pilot", "condition": "replay_ewc", "active_seconds": 500.}})
    assert report._forecast_seconds(units) == [1485., 2970.]
    units.append({"summary": {"kind": "solver_campaign"}})
    assert report._forecast_seconds(units) == [0., 0.]


def test_resumed_pilot_forecast_uses_cumulative_runtime(monkeypatch):
    monkeypatch.setattr(report, "_prior_panel_seconds", lambda units: 100.)
    units = [{"summary": {"kind": "solver_readiness", "condition": c,
                          "active_seconds": 48., "acquired_count": 48}}
             for c in ("replay_ewc", "stm_fd")]
    units.append({"summary": {"kind": "solver_pilot", "condition": "replay_ewc",
                              "active_seconds": 5., "timings": {"active_seconds": 500.}}})
    assert report._forecast_seconds(units) == [1485., 2970.]


def panel(checkpoint, difficulty, n, *, seeds=(11, 12), successes=(True, False)):
    return {"checkpoint": checkpoint, "difficulty": difficulty, "acquired_count": n,
            "seeds": list(seeds), "rows": [{"seed": s, "success": v} for s, v in zip(seeds, successes)]}


def pilot(root, condition, *, recipe="proximal_1", seeds=(11, 12), asset="common", seed=101):
    return unit(root, "pilot", name=condition + recipe + str(seed), solver_recipe=recipe, condition=condition,
                settings={"name": condition, "seed": seed, "initial_asset": asset},
                fit_metrics=[{"first_step_ewc_suppression": .2, "update_seconds": .1}],
                evaluations=[panel(cp, d, n, seeds=seeds) for cp, n in (("initial", 0), ("midpoint", 10), ("final", 20))
                             for d in ("easy", "hard")], timings={"active_seconds": 60., "fit_seconds": 10.}, acquired_count=20)


def text_outputs(outputs):
    return "\n".join(o.get("text", "") + o.get("data", {}).get("text/plain", "") for o in outputs)


def rendered(book):
    return "\n".join(c.source + "\n" + text_outputs(c.get("outputs", [])) for c in book.cells)


def base_book():
    return nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Historical results"),
        nbformat.v4.new_code_cell("historical_progress()", metadata={"report_section": "progress"}, outputs=[shared_output()]),
        nbformat.v4.new_markdown_cell("### Phase 3 Fix 1", metadata={"plan19_phase3_fix1": True}),
        nbformat.v4.new_markdown_cell("## Conclusion\n\nHistorical steering-priority recommendation.")])


def shared_output():
    return nbformat.v4.new_output("stream", name="stdout", text="Historical counters\n")


def test_missing_root_is_pending_read_only_and_chronological(tmp_path):
    root = tmp_path / "missing"
    book = base_book()
    previous = copy.deepcopy(book.cells)
    report.augment_notebook(book, root=root)
    nbformat.validate(book)
    assert not root.exists()
    assert book.cells[0] == previous[0] and book.cells[2] == previous[2]
    assert book.cells[1].source.startswith(previous[1].source)
    assert book.cells[1].outputs[0] == previous[1].outputs[0]
    text = rendered(book)
    assert text.index("Phase 3 Fix 1") < text.index("### Phase 3 Fix 2")
    assert "Historical steering-priority recommendation" in text
    assert "Phase 3 Fix 2" in book.cells[-1].source and "steering-priority" not in book.cells[-1].source
    assert "pending / not recorded" in text and "not yet started" in text
    assert "late segment is not a historical continuation" in text
    assert report.progress_rows(report.load_report(root))[0]["Selected solver"] == report.PENDING
    assert sum("image/png" in o.get("data", {}) for c in book.cells for o in c.get("outputs", [])) == 0
    report.augment_notebook(book, root=root)
    assert len([c for c in book.cells if c.id == "plan19-phase3-fix2-0"]) == 1
    assert sum(o.get("metadata", {}).get(report._TAG + "_progress", False) for o in book.cells[1].outputs) == 1
    assert book.cells[1].source.count("solver_report.display_progress") == 1


def test_committed_progress_not_mirror_and_no_partial_outcomes(tmp_path):
    path = live(tmp_path, status="recorded comparison", repair_active_seconds=360., cumulative_study_seconds=39600.,
                remaining_budget_seconds=32040., completed_trajectories=0,
                remaining_time_range_seconds=[7200., 10800.],
                active_unit_progress={"counts": {"step": 16, "target": 128}, "validation": {"old_nll": -999.}})
    snapshot = report.load_report(tmp_path)
    assert snapshot["complete_units"] == []
    row = report.progress_rows(snapshot)[0]
    assert row["Stage"] == "incomplete: recorded comparison"
    assert row["Charged repair / study h"] == "0.100 / 11.000"
    assert row["Forecast / remaining ceiling h"] == "2.00--3.00 / 8.900"
    assert row["Active counter / target"] == "16 / 128"
    assert row["Pilots complete / remaining (four planned)"] == "0 / 4"
    text = text_outputs(report.diagnostic_outputs(snapshot))
    assert "-999" not in text and "STALE" not in text
    assert report.PENDING in text
    assert (path / "CURRENT.json").exists()


def test_child_progress_without_schema_and_uncommitted_attempt_are_incomplete(tmp_path):
    live(tmp_path, name="units/recorded-active", schema=False, counts={"step": 1, "target": 2})
    empty = tmp_path / "phase3" / report.CAMPAIGN / "units" / "not-committed"
    empty.mkdir()
    write_json(empty / "config.json", {"schema_version": 1})
    write_json(empty / "progress.json", {"completed_trajectories": 99})
    snapshot = report.load_report(tmp_path)
    assert len(snapshot["live"]) == 2
    assert all(u["status"] == "incomplete" for u in snapshot["live"])
    assert report.progress_rows(snapshot)[0]["Pilots complete / remaining (four planned)"] == report.PENDING + " / " + report.PENDING


def test_actual_recorded_schema_segment_times_and_plots(tmp_path, monkeypatch):
    unit(tmp_path, "recorded", **recorded_fields())
    snapshot = report.load_report(tmp_path)
    rows = report.recorded_rows(snapshot)
    assert [r["Segment"] for r in rows] == ["early", "late"]
    assert [r["Accepted events"] for r in rows] == [2, 2]
    assert [r["Fit seconds"] for r in rows] == [4., 4.]
    assert rows[0]["Old NLL before / after"] == "2.0 / 2.2"
    assert len(report.timing_rows(snapshot)) == 1
    timing = report.timing_rows(snapshot)[0]
    assert timing["mean_seconds"] == 2.
    assert timing["Charged fit seconds (all events)"] == 8.
    assert timing["Timed fit seconds (diagnostics excluded)"] == 6.
    assert timing["diagnostic_fit_seconds"] == 2.
    figures = []
    original = report.initialreport._figure

    def capture(figure, description):
        figures.append([[(list(l.get_xdata()), list(l.get_ydata())) for l in a.lines] for a in figure.axes])
        return original(figure, description)

    monkeypatch.setattr(report.initialreport, "_figure", capture)
    outputs = report.diagnostic_outputs(snapshot)
    assert len(figures) == 2
    assert ([0., 2., 4.], [2., 2.1, 2.2]) in figures[0][1]
    assert ([1, 2], [.35, .45]) in figures[1][0]
    assert ([1, 2], [-.2, -.2]) in figures[1][1]
    assert ([1, 2], [.5, .25]) in figures[1][3]
    assert "trace-not-narrative" not in text_outputs(outputs)
    assert "include charged diagnostics" in text_outputs(outputs)


def test_failed_zero_event_recorded_unit_remains_visible_not_success(tmp_path):
    fields = recorded_fields("lagged_1")
    fields.update(status="numerical_failure", completed_events=0, fit_seconds=0., segments=[], rows=[],
                  validation_rows=[], timing={"events": 0}, validation={"initial": {"old_nll": None, "live_nll": None},
                                                                               "final": {"old_nll": None, "live_nll": None}})
    unit(tmp_path, "recorded", **fields)
    snapshot = report.load_report(tmp_path)
    assert report.recorded_rows(snapshot)[0]["Status"] == "numerical_failure"
    assert "1 numerical failures" in report.conclusion(snapshot)
    assert "selection remains unresolved" in report.conclusion(snapshot)
    assert report.timing_rows(snapshot)[0]["cost_ratio_to_proximal_1"] is None


def test_selection_and_candidate_measured_timing_render_without_reselection(tmp_path):
    unit(tmp_path, "selection", selected_solver="proximal_3", selected_learning_rate=.0015,
         reason="Recorded quality and cost trade-off", candidates=[
             {"solver_recipe": "proximal_1", "stable": False, "mean_seconds": None},
             {"solver_recipe": "proximal_3", "stable": True, "mean_seconds": .82, "old_gap": .01, "live_nll": 2.}])
    snapshot = report.load_report(tmp_path)
    text = text_outputs(report.diagnostic_outputs(snapshot))
    assert "mean_seconds" in text and "0.82" in text
    assert "Recorded quality and cost trade-off" in text
    assert report.progress_rows(snapshot)[0]["Selected solver"] == "proximal_3"
    assert "Recorded selection: proximal_3" in report.conclusion(snapshot)


def test_conclusion_separates_curvature_use_from_pending_live_learning(tmp_path):
    fields = recorded_fields("proximal_3")
    for event in fields["rows"]:
        event["first_step_ewc_proposal_difference_norm"] = .1
    unit(tmp_path, "recorded", **fields)
    unit(tmp_path, "selection", selected_solver="proximal_3", selected_learning_rate=.003,
         reason="Recorded development guide", candidates=[])
    text = report.conclusion(report.load_report(tmp_path))
    assert "4/4 recorded events" in text
    assert "not successful retention" in text
    assert "Execution is still in progress" in text
    assert "No completed pilot succeeded" not in text


def test_runtime_rows_do_not_count_policy_decisions_as_fallbacks(tmp_path):
    path = pilot(tmp_path, "replay_ewc")
    fixture = shared.read_completed_unit(path)
    fixture["summary"].update(active_seconds=3600., fallback_counts={"policy": 18, "startup": 1, "inference_pending": 1},
        episodes=[{"intervals": [{"timing": {"model_lease_wait_s": .2, "inference_deadline_miss": 1.}}]}])
    fixture["summary"]["timings"]["active_seconds"] = 7200.
    snapshot = {"root": str(path.parents[1]), "complete_units": [fixture], "live": []}
    row = report.operational_rows(snapshot)[0]
    assert row["Non-startup fallback commands"] == 1
    assert row["Deadline misses"] == 1
    assert row["Max actor lease wait s"] == "0.200"
    assert row["Acquired / pilot h"] == "10.0"


def test_completed_budget_uses_sealed_limit_and_conclusion_qualifies_comparison(tmp_path):
    path = unit(tmp_path, "campaign", repair_active_seconds=3600.,
                completed_trajectories=4, status="complete_for_review")
    fixture = shared.read_completed_unit(path)
    fixture["stored_config"]["max_hours"] = 5.
    snapshot = {"root": str(path.parents[1]), "complete_units": [fixture], "live": []}
    row = report.progress_rows(snapshot)[0]
    assert row["Forecast / remaining ceiling h"] == "0.00--0.00 / 4.000"
    assert row["Active counter / target"] == "Not active"
    p = pilot(tmp_path, "stm_fd")
    snapshot["complete_units"].append(shared.read_completed_unit(p))
    text = report.conclusion(snapshot)
    assert "share trainable STM/LTM banks" in text
    assert "not freshly retention-matched" in text
    assert "do not isolate FD efficacy at equal retention" in text


def test_missing_gpu_telemetry_is_not_zero_and_ratio_uses_measured_means(tmp_path):
    for name, mean in (("proximal_1", .1), ("proximal_3", .25)):
        fields = recorded_fields(name)
        fields["timing"].update(mean_seconds=mean, gpu_seconds=None, peak_vram_bytes=None)
        unit(tmp_path, "recorded", name=name, **fields)
    rows = report.timing_rows(report.load_report(tmp_path))
    assert [r["cost_ratio_to_proximal_1"] for r in rows] == [1., 2.5]
    assert all("gpu_seconds" not in r and "peak_vram_bytes" not in r for r in rows)


def test_pilot_pairing_midpoint_curve_and_compact_outcomes(tmp_path, monkeypatch):
    pilot(tmp_path, "replay_ewc")
    pilot(tmp_path, "stm_fd")
    snapshot = report.load_report(tmp_path)
    figures = []
    original = report.initialreport._figure

    def capture(figure, description):
        figures.append([[list(l.get_xdata()) for l in a.lines] for a in figure.axes])
        return original(figure, description)

    monkeypatch.setattr(report.initialreport, "_figure", capture)
    report._capability(snapshot)
    assert figures == [[[[0, 10, 20], [0, 10, 20]], [[0, 10, 20], [0, 10, 20]]]]
    assert report.pilot_rows(snapshot)[0]["A midpoint"] == "1/2"
    assert report.pilot_rows(snapshot)[0]["B final"] == "1/2"


@pytest.mark.parametrize("mismatch", ["seeds", "asset", "recipe", "seed"])
def test_unpaired_pilots_do_not_get_paired_curves(tmp_path, monkeypatch, mismatch):
    pilot(tmp_path, "replay_ewc")
    kwargs = {"seeds": (13, 14)} if mismatch == "seeds" else {mismatch: {"asset": "other", "recipe": "lagged_1", "seed": 102}[mismatch]}
    pilot(tmp_path, "stm_fd", **kwargs)
    lines = []
    monkeypatch.setattr(report.initialreport, "_figure", lambda f, d: lines.extend(len(a.lines) for a in f.axes))
    assert report._capability(report.load_report(tmp_path)) is None
    assert lines == []
    assert len(report.pilot_rows(report.load_report(tmp_path))) == 2


@pytest.mark.parametrize("fields,match", [
    ({"solver_recipe": "adam"}, "recipe"),
    ({"steps_per_event": 3}, "steps_per_event"),
    ({"condition": "unknown"}, "condition"),
    ({"completed_events": -1, "segments": []}, "completed_events"),
    ({"schema_version": 99}, "summary schema"),
    ({"version": "future"}, "version"),
    ({"per_method": [42]}, "per_method"),
    ({"fit_seconds": -1, "segments": []}, "fit_seconds"),
])
def test_incompatible_completed_summary_fails_clearly(tmp_path, fields, match):
    unit(tmp_path, "recorded", **{**recorded_fields(), **fields})
    with pytest.raises(shared.ArtifactError, match=match):
        report.load_report(tmp_path)


@pytest.mark.parametrize("kind", ["software", "provenance", "integrity", "dataset", "readiness"])
def test_auxiliary_kinds_do_not_invent_outcomes(tmp_path, kind):
    unit(tmp_path, kind, passed=True)
    snapshot = report.load_report(tmp_path)
    assert report.recorded_rows(snapshot) == [] and report.pilot_rows(snapshot) == []
    assert report.progress_rows(snapshot)[0]["Selected solver"] == report.PENDING


def test_tampering_completion_and_progress_are_rejected_without_repair(tmp_path):
    path = unit(tmp_path, "recorded", **recorded_fields())
    (path / "summary.json").write_text((path / "summary.json").read_text() + " ")
    with pytest.raises(shared.ArtifactError, match="hash/size"):
        report.load_report(tmp_path)
    assert (path / "COMPLETED").exists()
    other = tmp_path / "live"
    path = live(other, status="fitting")
    write_json(path / "transactions/commit-1/progress.json", {"schema_version": 1})
    with pytest.raises(shared.ArtifactError, match="hash/size"):
        report.load_report(other)


def test_incompatible_orchestration_progress_fails(tmp_path):
    live(tmp_path, schema=False, status="fitting")
    with pytest.raises(shared.ArtifactError, match="progress schema"):
        report.load_report(tmp_path)


def test_selected_clip_links_are_bounded_without_opening_media(tmp_path, monkeypatch):
    path = pilot(tmp_path, "replay_ewc")
    fixture = shared.read_completed_unit(path)
    videos = [{"path": "chosen.mp4", "checkpoint": "final", "difficulty": "easy", "selected": True},
              {"path": "../../../../escape.mp4", "checkpoint": "final", "difficulty": "hard"}]
    fixture["summary"]["videos"] = videos
    (path / "chosen.mp4").write_bytes(b"not decoded by reporter")
    snapshot = {"root": str(path.parents[1]), "complete_units": [fixture], "live": []}
    outputs = report.diagnostic_outputs(snapshot, notebook_dir=tmp_path)
    html = "\n".join(o.get("data", {}).get("text/html", "") for o in outputs)
    assert "chosen.mp4" in html and "escape.mp4" not in html


def test_clip_links_prefer_preselected_endpoint_over_reused_initialization(tmp_path):
    path = pilot(tmp_path, "replay_ewc")
    fixture = shared.read_completed_unit(path)
    fixture["summary"]["videos"] = [
        {"path": name + ".mp4", "checkpoint": cp, "difficulty": domain, "selection": "preselected"}
        for name, cp, domain in (("initial", "initial", "easy"),
                                 ("endpoint-easy", "final", "easy"), ("endpoint-hard", "final", "hard"))]
    for video in fixture["summary"]["videos"]:
        (path / video["path"]).write_bytes(b"report must not decode media")
    snapshot = {"root": str(path.parents[1]), "complete_units": [fixture], "live": []}
    outputs = report.diagnostic_outputs(snapshot, notebook_dir=tmp_path)
    html = "\n".join(o.get("data", {}).get("text/html", "") for o in outputs)
    assert "endpoint-easy.mp4" in html and "endpoint-hard.mp4" in html
    assert "initial.mp4" not in html


def test_build_cache_excludes_solver_completions_deepcopies_and_invalidates(tmp_path, monkeypatch):
    calls = []
    report._BASE_CACHE.clear()

    def build(root, output):
        calls.append(root)
        nbformat.write(base_book(), output)
        return Path(output)

    monkeypatch.setattr(report.initialreport, "build_notebook", build)
    output = tmp_path / "out.ipynb"
    report.build_notebook(tmp_path, output)
    report.build_notebook(tmp_path, output)
    assert len(calls) == 1
    unit(tmp_path, "selection", campaign="solver-fix2-v1", selected_solver="lagged_1", reason="superseded")
    report.build_notebook(tmp_path, output)
    assert len(calls) == 1
    assert "v1 diagnostic halt" in rendered(nbformat.read(output, as_version=4))
    unit(tmp_path, "selection", selected_solver="proximal_1", reason="settled")
    report.build_notebook(tmp_path, output)
    assert len(calls) == 1
    assert "Recorded selection: proximal_1" in nbformat.read(output, as_version=4).cells[-1].source
    cached = next(iter(report._BASE_CACHE.values()))
    assert "Phase 3 Fix 2:" not in rendered(cached)
    marker = tmp_path / "phase2" / "old" / "COMPLETED"
    marker.parent.mkdir(parents=True)
    write_json(marker, {"manifest_sha256": "new historical completion"})
    report.build_notebook(tmp_path, output)
    assert len(calls) == 2
    report.build_notebook(tmp_path, output, use_cache=False)
    assert len(calls) == 3
    assert not list(tmp_path.glob(".plan19-solver-report-*.ipynb"))


def test_build_is_atomic_and_refuses_completed_output(tmp_path, monkeypatch):
    output = tmp_path / "out.ipynb"
    output.write_text("existing report")

    def fail(*args, **kwargs):
        raise RuntimeError("historical build failed")

    monkeypatch.setattr(report.initialreport, "build_notebook", fail)
    with pytest.raises(RuntimeError, match="historical build failed"):
        report.build_notebook(tmp_path, output, use_cache=False)
    assert output.read_text() == "existing report"
    write_json(tmp_path / "COMPLETED", {})
    with pytest.raises(shared.ArtifactError, match="completed unit"):
        report.build_notebook(tmp_path, output)


def test_full_pending_builder_uses_reporting_only_python(tmp_path):
    script = """
import importlib.abc
import pathlib
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'safetensors', 'huggingface_hub'}:
            raise AssertionError('forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, Block())
from numerical_experiment.picar_solver_repair import report
root, output = map(pathlib.Path, sys.argv[1:])
report.build_notebook(root, output, use_cache=False)
assert not root.exists()
"""
    check = subprocess.run([sys.executable, "-c", script, str(tmp_path / "missing"), str(tmp_path / "report.ipynb")],
                           cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
                           env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")}, timeout=120)
    assert check.returncode == 0, check.stderr
    book = nbformat.read(tmp_path / "report.ipynb", as_version=4)
    nbformat.validate(book)
    assert "Phase 3 Fix 2:" in book.cells[-1].source


def test_current_default_excludes_v1_science_but_notes_halt_and_shared_budget(tmp_path):
    old = unit(tmp_path, "recorded", campaign="solver-fix2-v1", **recorded_fields("lagged_1"))
    old_selection = unit(tmp_path, "selection", campaign="solver-fix2-v1",
                         selected_solver="lagged_1", reason="INVALID_OLD_SELECTION")
    new = unit(tmp_path, "software", passed=True)
    snapshot = report.load_report(tmp_path)
    assert report.CAMPAIGN == "solver-fix2-v3"
    assert report.VERSION == "plan19-phase3-fix2-v3"
    assert [u["path"] for u in snapshot["complete_units"]] == [str(new)]
    assert report.recorded_rows(snapshot) == []
    assert report.progress_rows(snapshot)[0]["Selected solver"] == report.PENDING
    text = text_outputs(report.diagnostic_outputs(snapshot))
    assert "v1 diagnostic halt before live admission" in text
    assert "frozen full holdout" in text and "not pooled with v2 science" in text
    assert "same nine-hour Fix 2 ceiling" in text
    assert "INVALID_OLD_SELECTION" not in text
    shared_snapshot = {"root": str(tmp_path), "complete_units": [shared.read_completed_unit(p)
                       for p in (old, old_selection, new)], "live": []}
    assert [u["path"] for u in report._units(shared_snapshot)] == [str(new)]
    assert report.recorded_rows(shared_snapshot) == []
    assert (old / "COMPLETED").exists() and (old_selection / "COMPLETED").exists()


def test_explicit_historical_v1_root_and_progress_remain_readable(tmp_path, monkeypatch):
    old = unit(tmp_path, "recorded", campaign="solver-fix2-v1", **recorded_fields())
    live(tmp_path, campaign="solver-fix2-v1", status="halted before live", repair_active_seconds=120.)
    root = tmp_path / "phase3" / "solver-fix2-v1"
    snapshot = report.load_report(root)
    assert snapshot["root"] == str(root)
    assert [u["path"] for u in snapshot["complete_units"]] == [str(old)]
    assert report.progress_rows(snapshot)[0]["Stage"] == "incomplete: halted before live"
    assert report.progress_rows(snapshot)[0]["Charged repair / study h"].startswith("0.033 /")
    assert "v1 diagnostic halt" in report._correction_note(snapshot)
    calls = []

    def build(root, output):
        calls.append(root)
        nbformat.write(base_book(), output)

    monkeypatch.setattr(report.initialreport, "build_notebook", build)
    report.build_notebook(root, tmp_path / "historical.ipynb", use_cache=False)
    assert calls == [tmp_path]


def test_campaign_versions_cannot_be_misfiled_or_progress_mixed(tmp_path):
    unit(tmp_path, "recorded", version="plan19-phase3-fix2-v1", **recorded_fields())
    with pytest.raises(shared.ArtifactError, match="do not pool campaigns"):
        report.load_report(tmp_path)
    other = tmp_path / "progress"
    live(other, version="plan19-phase3-fix2-v1", status="wrong version")
    with pytest.raises(shared.ArtifactError, match="do not pool campaigns"):
        report.load_report(other)
