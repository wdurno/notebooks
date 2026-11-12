"""Small validated-snapshot fixtures; never load runners, Torch or checkpoints."""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

from numerical_experiment.picar import phase2_report as report


def unit(campaign, kind, *, name=None, completed_at="2026-11-12T12:00:00Z", **summary):
    return {"path": str(campaign / "units" / (name or kind)),
            "manifest": {"completed_at": completed_at},
            "summary": {"kind": "phase2_" + kind, **summary}}


def snapshot(root, *units, live=None):
    return {"root": str(root), "complete_units": list(units), "live": live or []}


def panel(n=20, successes=12):
    return [{"seed": 192501 + i, "success": i < successes,
             "end_reason": "target_contact" if i < successes else "wall_collision",
             "physical_seconds": 4 + i,
             "layout": {"direct_path_blocked": bool(i % 2), "layout_id": i}}
            for i in range(n)]


def evaluation(checkpoint, difficulty="easy", rows=None):
    return {"checkpoint": checkpoint, "difficulty": difficulty,
            "rows": panel() if rows is None else rows}


def pilot(campaign, *, seed=192101, name="pilot", evaluations=None, **extra):
    return unit(campaign, "pilot", name=name, seed=seed,
                settings={"name": "common-baseline", "hyperparams": {"lr": .001, "fd_strength": 0}},
                evaluations=evaluations if evaluations is not None else [
                    evaluation("initial"), evaluation("final", rows=panel(successes=14)),
                    evaluation("initial", "hard", panel(10, 1)),
                    evaluation("final", "hard", panel(10, 3))], **extra)


def rendered(book):
    parts = []
    for cell in book.cells:
        parts.append(cell.source)
        for output in cell.get("outputs", []):
            parts.append(output.get("text", output.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


def book_for(data, **kwargs):
    return report.append_phase2(nbformat.v4.new_notebook(), data, **kwargs)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_absent_phase2_is_noop_without_creating_cache(tmp_path):
    root = tmp_path / "missing"
    data = snapshot(root)
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("## Development")])
    assert report.append_phase2(book, data) is book
    assert len(book.cells) == 1
    assert report.conclusion(data) is None
    assert not root.exists()
    assert "Phase 2 pending" in report.phase2_outputs(data)[0].text


def test_append_is_idempotent_after_fix3_before_development_and_code_refreshes(tmp_path, monkeypatch):
    campaign = tmp_path / "phase2" / "readiness-v1"
    data = snapshot(tmp_path, pilot(campaign))
    book = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("### Phase 1 Fix 3"),
        nbformat.v4.new_code_cell("# retained fix3", metadata={"plan19_fix3": True}),
        nbformat.v4.new_markdown_cell("## Development"),
        nbformat.v4.new_markdown_cell("## Conclusion")])
    report.append_phase2(book, data, notebook_dir=tmp_path)
    report.append_phase2(book, data, notebook_dir=tmp_path)
    assert len(book.cells) == 6
    assert book.cells[0].source == "### Phase 1 Fix 3"
    assert book.cells[1].metadata.plan19_fix3
    assert book.cells[2].source.startswith("### Phase 2")
    assert book.cells[4].source == "## Development"
    assert book.cells[5].source == "## Conclusion"
    assert all(c.metadata.plan19_phase2 for c in book.cells[2:4])
    assert len({c.id for c in book.cells}) == len(book.cells)
    nbformat.validate(book)
    calls = []
    monkeypatch.setattr(report, "display_phase2", lambda value, **kw: calls.append((value, kw)))
    exec(book.cells[3].source, {"snapshot": data})
    assert calls == [(data, {"notebook_dir": str(tmp_path)})]
    report.append_phase2(book, snapshot(tmp_path))
    assert len(book.cells) == 4


@pytest.mark.parametrize("root_kind", ["cache", "phase2", "campaign"])
def test_direct_or_nested_cache_progress_is_operational_only(tmp_path, root_kind):
    campaign = tmp_path / "phase2" / "readiness-v2"
    root = {"cache": tmp_path, "phase2": campaign.parent, "campaign": campaign}[root_kind]
    progress = campaign / "orchestration" / "progress.json"
    write_json(progress, {"schema_version": 1, "status": "collecting", "counts": {"step": 2, "target": 7},
                          "settings": {"phase2_cap_hours": 5}, "active_seconds": 23,
                          "remaining_estimate_seconds": 90})
    write_json(campaign / "unfinished" / "summary.json", {
        "kind": "phase2_pilot", "seed": "MUST_NOT_REPORT", "acquired_count": 999})
    (campaign / "unfinished" / "checkpoint.pt").write_bytes(b"must never be unpickled")
    before = {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    text = rendered(book_for(snapshot(root)))
    assert "collecting" in text and "phase2_cap_hours" in text and "90" in text
    assert "MUST_NOT_REPORT" not in text and "999" not in text
    assert "outcomes: pending / not recorded" in text
    assert "pending or stopped" in report.conclusion(snapshot(root))
    assert before == {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}


def test_live_only_phase2_and_stop_are_reported_without_old_fix3(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v3"
    data = snapshot(tmp_path, live=[
        {"path": str(tmp_path / "phase1" / "fix3-v2" / "orchestration"), "status": "OLD_FIX3"},
        {"path": str(campaign / "orchestration"), "status": "running", "stopped": True}])
    text = rendered(book_for(data))
    assert "stopped" in text and "OLD_FIX3" not in text
    assert "pending or stopped" in report.conclusion(data)


def test_validated_snapshot_does_not_require_unit_files_and_keeps_reuse_counts_separate(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v1"
    data = snapshot(tmp_path,
        unit(campaign, "initial_asset", source="phase1/fix3-v2/bootstrap", source_marker_sha256="a" * 64,
             acquired_count=292, distinct_bootstrap_samples=247, retired_count=239, eligible_count=8,
             reused=True, original_compute_excluded=True),
        pilot(campaign, acquired_count=640, episodes=[{}, {}],
              online_updates=639, sleep_updates=4, fallback_counts={"startup": 1, "nonstartup": 0},
              timings={"online_seconds": 12.5, "sleep_seconds": 3.5},
              distill_receipt={"continuation": "initial_asset", "repeated": False}))
    text = rendered(book_for(data))
    for field in ("292", "247", "239", "eligible_count", "source_marker_sha256", "original_compute_excluded",
                  "New acquired", "640", "common-baseline", "hyperparams", "Online updates", "Sleep updates",
                  "639", "Fallback counts", "nonstartup", "online_seconds", "receipt archived"):
        assert field in text
    assert "KL" not in text
    assert "original counts and compute are excluded" in text
    assert not tmp_path.joinpath("phase2").exists()


def test_full_easy_panel_clear_blocked_counts_and_pilots_are_not_pooled(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v1"
    teacher = unit(campaign, "teacher_readiness", rows=panel(successes=15))
    initial = unit(campaign, "initial_readiness", evaluations=[evaluation("initial")])
    second = pilot(campaign, seed=192102, name="second", evaluations=[
        evaluation("initial", rows=panel(successes=0)), evaluation("final", rows=panel(successes=0))])
    data = snapshot(tmp_path, teacher, initial, pilot(campaign), second)
    rows = report._outcome_rows(teacher)
    assert rows[0]["Rows"] == 20
    assert rows[0]["Success"] == "15/20 (75.0%)"
    assert rows[0]["Easy clear success"] == "8/10 (80.0%)"
    assert rows[0]["Easy blocked success"] == "7/10 (70.0%)"
    assert rows[0]["Collisions"] == "5/20 (25.0%)"
    assert rows[0]["Physical seconds (mean / n)"] == "13.5 / 20"
    text = rendered(book_for(data))
    assert "initial_readiness" not in text  # Evidence uses controller/checkpoint labels.
    assert "12/20 (60.0%)" in text and "14/20 (70.0%)" in text and "0/20 (0.0%)" in text
    assert "192101" in text and "192102" in text and "12/40" not in text
    assert "independent training trajectories" in text
    assert "20 fixed layouts" in text and "first 10" in text


def test_missing_results_and_layouts_are_unknown_not_failure_or_clear(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v1"
    teacher = unit(campaign, "teacher_readiness", rows=[
        {"success": True, "layout": {"direct_path_blocked": False}},
        {"success": "false", "end_reason": "interrupted"}, {}])
    row = report._outcome_rows(teacher)[0]
    assert row["Success"] == "1/1 (100.0%); 2 unknown"
    assert row["Collisions"] == report.PENDING
    assert row["Easy clear success"] == "1/1 (100.0%)"
    assert row["Easy blocked success"] == report.PENDING
    assert row["Easy path unknown"] == 2
    assert report._rate([{"success": "false"}]) == report.PENDING
    text = rendered(book_for(snapshot(tmp_path, teacher)))
    assert "Language responsiveness: pending" in text and "Forecast / FD overhead" in text


def test_nested_layout_metadata_terminal_diagnostics_and_partial_panels(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v1"
    rows = panel(2, 1)
    for row in rows:
        row["layout"] = {"curriculum": row["layout"]}
        row["terminal_diagnostics"] = {"clearance_m": .25, "heading_error_rad": .5,
                                       "target_contact": False, "physical_seconds": 3}
    outcome = report._outcome_rows(unit(campaign, "initial_readiness", evaluations=[evaluation("initial", rows=rows)]))
    assert outcome[0]["Easy clear success"] == "1/1 (100.0%)"
    assert outcome[0]["Easy blocked success"] == "0/1 (0.0%)"
    assert outcome[0]["Panel coverage"] == "2/20 expected seed IDs; 0 unexpected; 0 duplicates"
    assert outcome[0]["Terminal geometry (mean)"] == "clearance_m=0.25 (n=2/2); heading_error_rad=0.5 (n=2/2)"
    assert outcome[1]["Difficulty"] == "hard" and outcome[1]["Success"] == report.PENDING


@pytest.mark.parametrize("change,expected", [
    ("reverse", "matched seed IDs"), ("missing", "missing seed IDs"),
    ("duplicate", "duplicate seed IDs"), ("different", "seed IDs differ"),
    ("geometry", "layout metadata differ"), ("bool", "missing seed IDs")])
def test_pairing_requires_unique_matching_seed_ids_and_consistent_recorded_layouts(change, expected):
    before, after = panel(2), panel(2)
    if change == "reverse":
        after.reverse()
    elif change == "missing":
        del after[0]["seed"]
    elif change == "duplicate":
        after[1]["seed"] = after[0]["seed"]
    elif change == "different":
        after[0]["seed"] = 88
    elif change == "geometry":
        after[0]["layout"]["layout_id"] = 88
    else:
        after[0]["seed"] = True
    paired, reason = report._pairing(before, after)
    assert paired == (change == "reverse")
    assert expected in reason


def test_plot_connects_only_verified_complete_pairs(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    campaign = tmp_path / "phase2" / "readiness-v1"
    calls = []
    original = Axes.plot

    def plot(axis, *args, **kwargs):
        calls.append(args)
        return original(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot)
    changed = panel(2)
    changed[0]["seed"] = 1
    missing = panel(2)
    del missing[0]["success"]
    output = report._plot([pilot(campaign, evaluations=[
        evaluation("initial", "easy", panel(2)), evaluation("final", "easy", list(reversed(panel(2)))),
        evaluation("initial", "hard", panel(2)), evaluation("final", "hard", changed),
        evaluation("initial", "middle", panel(2)), evaluation("final", "middle", missing)])])
    assert len(calls) == 1
    assert list(calls[0][1]) == [1., 1.]
    assert "seed IDs differ" in output.data["text/plain"]
    assert base64.b64decode(output.data["image/png"]).startswith(b"\x89PNG\r\n\x1a\n")


def test_optional_terminal_geometry_reports_units_and_coverage():
    assert report._approach([{}]) == report.PENDING
    assert report._approach([{"terminal_geometry": {"target_distance_m": .4}}, {}]) == "target_distance_m=0.4 (n=1/2)"
    assert report._approach([{"terminal_geometry": {"target_distance_m": float("nan")}}]) == report.PENDING


def test_heterogeneous_scope_fields_remain_pending_not_nan_or_zero():
    output = report._table([{"scope": "small", "pairs": 4}, {"scope": "unpriced", "total_hours": 12}])
    text = output.data["text/plain"]
    assert text.count(report.PENDING) == 2
    assert "NaN" not in text


def test_video_links_label_actual_controller_checkpoint_difficulty_and_escape_paths(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v1"
    source = campaign / "units" / "initial_readiness"
    source.mkdir(parents=True)
    (source / "before clip.mp4").write_bytes(b"link only; never decode")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not a campaign video")
    (source / "escaped.mp4").symlink_to(outside)
    initial = unit(campaign, "initial_readiness", evaluations=[evaluation("initial")], videos=[
        {"controller": "LSTM <student>", "checkpoint": "initial", "difficulty": "easy", "path": "before clip.mp4"},
        {"controller": "Qwen", "checkpoint": "frozen", "difficulty": "hard", "path": "escaped.mp4"},
        {"actor": "learned_student", "checkpoint": "initial", "difficulty": "hard", "path": "before clip.mp4"},
        {"checkpoint": "final", "path": "missing.mp4"}])
    outputs = book_for(snapshot(tmp_path, initial), notebook_dir=tmp_path).cells[-1].outputs
    markup = "\n".join(o.get("data", {}).get("text/html", "") for o in outputs)
    assert "before%20clip.mp4" in markup
    assert "LSTM &lt;student&gt;" in markup and "<student>" not in markup
    assert "checkpoint=initial; difficulty=easy" in markup
    assert "Qwen; checkpoint=frozen; difficulty=hard" in markup
    assert "controller=learned_student; checkpoint=initial; difficulty=hard" in markup
    assert "controller=pending / not recorded" in markup
    assert 'href="escaped.mp4"' not in markup and "outside.mp4" not in markup
    assert "not the old scripted-controller demos" in markup


def test_forecasts_show_recorded_totals_and_definitions_not_derived_or_fd_measurements(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v1"
    data = snapshot(tmp_path, unit(campaign, "forecast", scope_rows=[
        {"scope": "small", "pairs": 4, "transitions_per_arm": 1200, "total_hours": 12.75}],
        totals={"prior_study_hours": 2.25, "phase2_cap_hours": 5, "total_hours": 15.125},
        fd_unmeasured=True, cost_definitions={"phase2": "new acquisition and updates only; reused bootstrap excluded"},
        readiness_conclusions=["easy A useful; common B acquisition unresolved"],
        blockers=["initial Fisher audit pending"], next_checkin="review recipe and forecast"),
        unit(campaign, "software", passed=True, test_result="small deterministic fixtures"))
    text = rendered(book_for(data))
    for field in ("12.75", "15.125", "2.25", "phase2_cap_hours", "fd_unmeasured", "cost_definitions",
                  "initial Fisher audit pending", "review recipe and forecast", "small deterministic fixtures"):
        assert field in text
    assert "not FD or sample-efficiency measurements" in text
    assert "p-value" not in text


def test_scalar_forecast_cost_schema_is_preserved_without_inventing_total_study_cost(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v1"
    forecast = unit(campaign, "forecast", scope_rows=[
        {"pairs": 4, "observations_per_arm": 1200, "fd_cost_multiplier_scenario": 2.,
         "training_processing_hours": 8.125, "not_total_study_forecast": True}],
        phase2_active_seconds=3600, prior_study_seconds=8100, study_charged_seconds=11700,
        remaining_to_20h_seconds=60300, trainability_observed=False, fd_unmeasured=True,
        limitations=["FD multipliers are scenarios, not measurements"], next_phase_authorized=False)
    data = snapshot(tmp_path, forecast, unit(campaign, "campaign", total_active_seconds=3600))
    text = rendered(book_for(data))
    for field in ("8100", "11700", "60300", "not_total_study_forecast", "trainability_observed",
                  "next_phase_authorized", "FD multipliers are scenarios, not measurements"):
        assert field in text
    assert "common-student trainability observed: no" in report.conclusion(data)
    assert "Phase 2 charged 1.00 h" in report.conclusion(data)
    assert "conservative study charge is 3.25 h" in report.conclusion(data)
    assert "16.75 h of the 20 h target" in report.conclusion(data)


def test_language_collapses_32_archived_rows_to_four_instruction_summaries():
    probabilities = {"forward": [.1, .1, .8, 0, 0, 0, 0, 0],
                     "left": [.7, .1, .2, 0, 0, 0, 0, 0],
                     "right": [.1, .7, .2, 0, 0, 0, 0, 0],
                     "stop": [0, 0, 0, 0, 0, 0, 0, 1]}
    raw = [{"condition": name, "probabilities": values, "source_history_digest": str(i), "logits": [0] * 8}
           for i in range(8) for name, values in probabilities.items()]
    before = json.dumps(raw)
    rows = report._language_rows(raw)
    assert len(rows) == 4
    assert [r["Instruction"] for r in rows] == ["forward", "left", "right", "stop"]
    assert all(r["Rows"] == r["Valid probability rows"] == 8 for r in rows)
    assert rows[0]["Mean forward P"] == pytest.approx(.8)
    assert rows[1]["Mean left P"] == pytest.approx(.7)
    assert rows[2]["Mean right P"] == pytest.approx(.7)
    assert rows[0]["Argmax counts"] == "drive-forward: 8"
    assert rows[3]["Argmax counts"] == "look-forward: 8"
    assert json.dumps(raw) == before
    invalid = report._language_rows([{"condition": "forward", "probabilities": [1, 2]}])
    assert invalid[0]["Rows"] == 1 and invalid[0]["Valid probability rows"] == 0
    assert invalid[0]["Mean forward P"] is None and invalid[1]["Rows"] == 0


def test_forecast_summary_is_eight_meaningful_scalars_without_metadata_or_none_rows():
    summary = {"seconds_per_captured_interval": 12., "phase2_active_seconds": 3600,
               "prior_study_seconds": 8100, "study_charged_seconds": 11700,
               "remaining_to_20h_seconds": 0, "trainability_observed": False,
               "fd_unmeasured": True, "next_phase_authorized": False,
               "frame_sha256": ["hash"] * 100, "cost_definitions": None, "next_checkin": None}
    rows = report._forecast_rows(summary)
    assert len(rows) == 8
    assert all(type(r["Recorded"]) in (int, float, bool) for r in rows)
    assert {r["Field"]: r["Recorded"] for r in rows}["remaining_to_20h_seconds"] == 0
    assert "frame_sha256" not in str(rows) and report.PENDING not in str(rows)


def test_latest_rms_family_preserves_historical_baselines_as_reference_rows_not_new_pilots(tmp_path):
    old = tmp_path / "phase2" / "readiness-v1"
    new = tmp_path / "phase2" / "rms-v1"
    references = [{"model": "original baseline", "seed": seed, "easy_initial": "12/20", "easy_final": "14/20",
                   "hard_initial": "0/10", "hard_final": "1/10", "source": f"original/pilot-{seed}",
                   "new_trajectory": False, "charged_again": False, "frame_sha256": ["REFERENCE_METADATA_SENTINEL"]}
                  for seed in (192101, 192102)]
    data = snapshot(tmp_path,
        unit(old, "campaign", completed_at="2026-11-12T10:00:00Z", status="completed"),
        unit(new, "reference_baselines", rows=references))
    before = json.dumps(data, sort_keys=True)
    book = book_for(data)
    text = rendered(book)
    for value in ("Historical baselines shown for comparison", "not new runs", "not pooled with rescue trajectories",
                  "original baseline", "192101", "192102", "12/20", "14/20", "0/10", "1/10",
                  "original/pilot-192101", "new_trajectory", "charged_again"):
        assert value in text
    assert "REFERENCE_METADATA_SENTINEL" not in text and "reference_baselines" not in text
    tables = [o for o in book.cells[-1].outputs if "original baseline" in o.get("data", {}).get("text/plain", "")]
    assert len(tables) == 1 and tables[0].data["text/plain"].count("False") == 4
    assert not any("image/png" in o.get("data", {}) for o in book.cells[-1].outputs)
    assert json.dumps(data, sort_keys=True) == before


def test_empty_reference_baselines_stay_pending_not_zero(tmp_path):
    campaign = tmp_path / "phase2" / "rms-v1"
    text = rendered(book_for(snapshot(tmp_path, unit(campaign, "reference_baselines", rows=[]))))
    assert "Historical baseline outcomes: pending / not recorded" in text


def test_conditioning_and_navigation_conclusions_are_distinct(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-rms-v1"
    health = unit(campaign, "feature_health", normalization="feature_and_projection_rms",
                  saturated_fraction=0., derivative_median=.46)
    data = snapshot(tmp_path, health,
                    unit(campaign, "forecast", trainability_observed=False, fd_unmeasured=True))
    text = report.conclusion(data)
    assert "unsaturated initialization probes" in text
    assert "not evidence of correct instruction following or navigation learning" in text
    assert "Common-controller learning remains unresolved" in text
    assert "larger FD comparison is not yet justified" in text
    assert "All Phase 2 versions share the same five-hour cap" in rendered(book_for(data))
    health["summary"]["derivative_median"] = 0.
    assert "unsaturated initialization probes" not in report.conclusion(data)


def test_closeout_ledger_supersedes_forecast_time_without_changing_it(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-rms-v1"
    forecast = unit(campaign, "forecast", phase2_active_seconds=3600,
                    study_charged_seconds=11700, remaining_to_20h_seconds=60300)
    original = json.dumps(forecast, sort_keys=True)
    closeout = unit(campaign, "closeout", completed_at="2026-11-12T13:00:00Z",
                    phase2_active_seconds=3900, study_charged_seconds=12000,
                    remaining_to_20h_seconds=60000, closeout_reserve_seconds=300)
    text = report.conclusion(snapshot(tmp_path, forecast, closeout))
    assert "Phase 2 charged 1.08 h" in text
    assert "5 min conservative closeout allowance" in text
    assert "Phase 2 charged 1.00 h" not in text
    assert json.dumps(forecast, sort_keys=True) == original
    assert "reference_baselines" not in text


def health_summary(normalization="none"):
    return {"normalization": normalization, "eps": 1e-6, "abs_preactivation_median": 11.588985,
        "saturated_fraction": 1., "derivative_median": 0., "derivative_max": 1.22786e-5,
        "distributions": {"preactivation": {"edges": [-20., 0., 20.], "counts": [10, 10]},
                          "derivative": {"edges": [0., .00001, .00002], "counts": [19, 1]}}}


def test_feature_health_all_versions_render_four_numbers_and_scalar_only_plots(tmp_path):
    old, new = (tmp_path / "phase2" / f"readiness-v{i}" for i in (1, 2))
    data = snapshot(tmp_path,
        unit(old, "feature_health", name="health-original", completed_at="2026-11-12T10:00:00Z", **health_summary()),
        unit(new, "feature_health", name="health-normalized", **health_summary("feature_rms")))
    data["complete_units"][0]["summary"]["original_digests"] = ["DO_NOT_RENDER_DIGESTS"]
    book = book_for(data)
    text = rendered(book)
    for value in ("health-original", "health-normalized", "feature_rms", "11.589", "1.22786e-05", "Median derivative",
                  "not causal proof", "local tanh slope", "epsilon"):
        assert value in text
    assert "DO_NOT_RENDER_DIGESTS" not in text
    assert text.count("Median |z|") == 2
    outputs = book.cells[-1].outputs
    assert sum("image/png" in o.get("data", {}) for o in outputs) == 2
    nbformat.validate(book)


def test_feature_health_missing_optional_evidence_stays_pending_and_malformed_histograms_fail():
    outputs = report._feature_health({}, "missing")
    assert report.PENDING in outputs[0].data["text/plain"]
    assert "distributions: pending" in outputs[1].text
    summary = health_summary()
    summary["distributions"]["preactivation"]["edges"] = [0, 1]
    with pytest.raises(ValueError, match="feature-health histogram"):
        report._feature_health(summary, "bad")


def test_default_render_omits_metadata_arrays_receipts_and_raw_language_without_mutating_snapshot(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v1"
    language = [{"condition": condition, "probabilities": [.1, .1, .8, 0, 0, 0, 0, 0],
                 "logits": ["RAW_LOGIT_SENTINEL"] * 8, "features": ["RAW_FEATURE_SENTINEL"] * 100,
                 "source_history_digest": "HISTORY_METADATA_SENTINEL"}
                for _ in range(8) for condition in ("forward", "left", "right", "stop")]
    units = [unit(campaign, "initial_asset", acquired_count=292, distinct_bootstrap_samples=247,
                  retired_count=239, eligible_count=8, source="fix3/bootstrap", reused=True),
             pilot(campaign), unit(campaign, "language", rows=language),
             unit(campaign, "forecast", phase2_active_seconds=3600, fd_unmeasured=True, scope_rows=[
                 {"pairs": 4, "training_processing_hours": 8., "frame_sha256": ["SCOPE_METADATA_SENTINEL"]}]),
             unit(campaign, "provenance", archive_sha256="archive-seal"),
             unit(campaign, "campaign", status="ready_for_checkin")]
    for value in units:
        value["summary"].update(frame_sha256=["FRAME_METADATA_SENTINEL"] * 100,
            media_sha256={"frame.png": "MEDIA_METADATA_SENTINEL"},
            distill_receipt={"history_digests": ["RECEIPT_METADATA_SENTINEL"] * 100})
    data = snapshot(tmp_path, *units)
    before = json.dumps(data, sort_keys=True)
    book = book_for(data)
    text = rendered(book)
    markup = "\n".join(o.get("data", {}).get("text/html", "") for c in book.cells for o in c.get("outputs", []))
    for excluded in ("frame_sha256", "media_sha256", "distill_receipt", "source_history_digest", "logits", "features",
                     "FRAME_METADATA_SENTINEL", "MEDIA_METADATA_SENTINEL", "HISTORY_METADATA_SENTINEL",
                     "RECEIPT_METADATA_SENTINEL", "SCOPE_METADATA_SENTINEL", "RAW_LOGIT_SENTINEL", "RAW_FEATURE_SENTINEL"):
        assert excluded not in text and excluded not in markup
    assert "receipt archived" in text and "archive-seal" in text
    assert len(report._asset_rows(units[:1])) == 1
    assert json.dumps(data, sort_keys=True) == before


def test_latest_campaign_uses_timestamps_not_lexical_version_and_does_not_reuse_old_success(tmp_path):
    old = tmp_path / "phase2" / "readiness-v9"
    new = tmp_path / "phase2" / "readiness-v10"
    data = snapshot(tmp_path,
        unit(old, "campaign", completed_at="2026-11-11T12:00:00Z", status="completed"),
        unit(old, "teacher_readiness", completed_at="2026-11-11T11:00:00Z", rows=panel(successes=20)),
        live=[{"path": str(new / "orchestration"), "updated_at": "2026-11-12T12:00:00Z", "status": "running"}])
    text = report.conclusion(data)
    assert "readiness-v10" in text and "pending or stopped" in text
    assert "20/20" not in text and "Frozen Qwen easy-A success" not in text
    rendered_text = rendered(book_for(data))
    assert rendered_text.index("campaign: readiness-v9") < rendered_text.index("campaign: readiness-v10")
    assert "20/20" not in rendered_text
    assert "revisions are not pooled" in rendered_text


def test_completed_campaign_conclusion_qualifies_readiness_and_stops_at_checkin(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v1"
    data = snapshot(tmp_path, unit(campaign, "teacher_readiness", rows=panel(successes=15)),
        unit(campaign, "initial_readiness", evaluations=[evaluation("initial")]),
        pilot(campaign), unit(campaign, "campaign", status="readiness unresolved"),
        unit(campaign, "forecast", fd_unmeasured=True, readiness_conclusions=["common B learner still unresolved"]))
    text = report.conclusion(data)
    for phrase in ("complete as an artifact campaign", "completion alone is not a readiness pass",
                   "15/20 (75.0%)", "Initial student", "matched seed IDs", "common B learner still unresolved",
                   "frozen Qwen need not solve the hard task", "Next check-in", "no automatic Phase 3"):
        assert phrase in text
    assert "not independent training trajectories" in text
    assert "| Controller / seed | Level | Before | After | Pairing |" in text
    assert len(text.split("\n\n")) == 4
    assert len(text.splitlines()) <= 15


def test_incompatible_progress_fails_clearly_without_repair(tmp_path):
    from numerical_experiment.picar.report import ArtifactError

    path = tmp_path / "phase2" / "readiness-v1" / "orchestration" / "progress.json"
    write_json(path, {"schema_version": 999, "status": "running"})
    before = path.read_bytes()
    with pytest.raises(ArtifactError, match="Phase 2 progress schema"):
        book_for(snapshot(tmp_path))
    assert path.read_bytes() == before


def test_notebook_cell_executes_with_torch_models_and_simulator_imports_forbidden(tmp_path):
    campaign = tmp_path / "phase2" / "readiness-v1"
    data = snapshot(tmp_path, unit(campaign, "teacher_readiness", rows=panel(2)), pilot(campaign),
                    unit(campaign, "language", rows=[{"condition": "forward", "probabilities": [0, 0, 1, 0, 0, 0, 0, 0]}]),
                    unit(campaign, "forecast", fd_unmeasured=True, phase2_active_seconds=3600),
                    unit(campaign, "feature_health", **health_summary()))
    script = """
import builtins, contextlib, io, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    forbidden = ('torch', 'transformers', 'pybullet', 'numerical_experiment.picar_phase2',
                 'numerical_experiment.picar_feature_health',
                 'numerical_experiment.picar.learner',
                 'numerical_experiment.picar.phase2', 'numerical_experiment.picar.phase1_fix3')
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden):
        raise AssertionError('runtime import in artifact report: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import nbformat
from numerical_experiment.picar.phase2_report import append_phase2, conclusion
snapshot = json.loads(sys.argv[1])
book = append_phase2(nbformat.v4.new_notebook(), snapshot)
with contextlib.redirect_stdout(io.StringIO()):
    for cell in book.cells:
        if cell.cell_type == 'code':
            exec(cell.source, {'snapshot': snapshot})
assert conclusion(snapshot).startswith('## Conclusion')
assert 'torch' not in sys.modules
nbformat.validate(book)
print('artifact-only cell executed')
"""
    result = subprocess.run([sys.executable, "-c", script, json.dumps(data)], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[2],
                            env={**os.environ, "MPLBACKEND": "Agg"}, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "artifact-only cell executed" in result.stdout
