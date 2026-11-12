"""Small artifact snapshots only; no training, model loading, or GPU work."""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

from numerical_experiment.picar import phase3_report as report


def unit(campaign, kind, *, name=None, completed_at="2026-11-13T12:00:00Z", **summary):
    return {"path": str(campaign / "units" / (name or kind)), "manifest": {"completed_at": completed_at},
            "summary": {"kind": "phase3_" + kind, **summary}}


def snapshot(root, *units, live=None):
    return {"root": str(root), "complete_units": list(units), "live": live or []}


def panel(n, successes):
    return [{"seed": 300501 + i, "success": i < successes,
             "end_reason": "target_contact" if i < successes else "wall_collision",
             "layout": {"direct_path_blocked": bool(i % 2)}} for i in range(n)]


def evaluation(checkpoint, difficulty, rows):
    return {"checkpoint": checkpoint, "difficulty": difficulty, "rows": rows}


def pilot(campaign, *, name="control", seed=193101, strength=0., easy_final=11, hard_final=4, **extra):
    return unit(campaign, "pilot", name=f"{name}-{seed}", seed=seed,
        settings={"name": name, "fd_strength": strength, "ridge": .1, "learning_rate": .001,
                  "optimizer_steps": 2, "temperature": 1.2, "action_variance": .04, "exploration_std": 1.},
        acquired_count=320, timings={"active_seconds": 110, "fd_seconds": 12},
        evaluations=[evaluation("initial", "easy", panel(20, 12)), evaluation("final", "easy", panel(20, easy_final)),
                     evaluation("initial", "hard", panel(10, 1)), evaluation("final", "hard", panel(10, hard_final))], **extra)


def rendered(book):
    return "\n".join(part for cell in book.cells for part in [cell.source, *[
        o.get("text", o.get("data", {}).get("text/plain", "")) for o in cell.get("outputs", [])]])


def book_for(data, **kwargs):
    return report.append_phase3(nbformat.v4.new_notebook(), data, **kwargs)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def selection(campaign, **extra):
    return unit(campaign, "selection", selected_control={"name": "control", "fd_strength": 0.},
                selected_fd={"name": "fd", "fd_strength": .1}, next_phase_authorized=False, **extra)


def test_retention_handover_references_are_not_new_trajectories_or_raw_metadata(tmp_path):
    campaign=tmp_path/"phase3"/"development-retention-v1"
    reference=unit(campaign,"reference_controls",candidate_rows=[
        {"name":"control_ridge_0.1","seed":193101,"fd_strength":0.,"ridge":.1,
         "old_gap_endpoint":.25,"new_gain_endpoint":0.,"raw_features":["MUST_NOT_RENDER"]}])
    text=rendered(book_for(snapshot(tmp_path,reference)))
    assert "Historical ridge screen, already charged" in text
    assert "not new trajectories" in text and "unexecuted plasticity repair" in text
    assert "control_ridge_0.1" in text and "0.25" in text
    assert "MUST_NOT_RENDER" not in text


def test_no_phase3_is_noop_and_does_not_create_cache(tmp_path):
    root = tmp_path / "absent"
    data = snapshot(root)
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("## Fresh Learning")])
    assert report.append_phase3(book, data) is book and len(book.cells) == 1
    assert report.conclusion(data) is None and report.progress_rows(data) == []
    assert "Phase 3 pending" in report.phase3_outputs(data)[0].text
    assert not root.exists()


@pytest.mark.parametrize("placeholder", ["## Development", "## Fresh Learning"])
def test_two_refreshable_cells_are_idempotent_after_phase2_before_placeholder(tmp_path, monkeypatch, placeholder):
    data = snapshot(tmp_path, pilot(tmp_path / "phase3" / "development-v1"))
    book = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("### Phase 2", metadata={"plan19_phase2": True}),
        nbformat.v4.new_code_cell("# retained phase2", metadata={"plan19_phase2": True}),
        nbformat.v4.new_markdown_cell(placeholder), nbformat.v4.new_markdown_cell("## Conclusion")])
    report.append_phase3(book, data, notebook_dir=tmp_path)
    report.append_phase3(book, data, notebook_dir=tmp_path)
    assert len(book.cells) == 6
    assert book.cells[1].source == "# retained phase2" and book.cells[4].source == placeholder
    assert book.cells[2].source.startswith("### Phase 3")
    assert all(c.metadata.plan19_phase3 for c in book.cells[2:4])
    assert len({c.id for c in book.cells}) == len(book.cells)
    nbformat.validate(book)
    calls = []
    monkeypatch.setattr(report, "display_phase3", lambda value, **kwargs: calls.append((value, kwargs)))
    exec(book.cells[3].source, {"snapshot": data})
    assert calls == [(data, {"notebook_dir": str(tmp_path)})]
    report.append_phase3(book, snapshot(tmp_path))
    assert len(book.cells) == 4 and book.cells[1].metadata.plan19_phase2


@pytest.mark.parametrize("root_kind", ["cache", "phase3", "campaign"])
def test_live_progress_only_never_promotes_partial_scientific_summaries(tmp_path, root_kind):
    campaign = tmp_path / "phase3" / "development-v1"
    root = {"cache": tmp_path, "phase3": campaign.parent, "campaign": campaign}[root_kind]
    write_json(campaign / "orchestration/progress.json", {"schema_version": 1, "status": "calibrating",
        "active_seconds": 72, "cumulative_study_seconds": 2000, "remaining_estimate_seconds": 3600,
        "active_unit": str(campaign / "units/current-probe"), "counts": {"step": 3, "target": 9}})
    write_json(campaign / "units/current-probe/summary.json", {"kind": "phase3_pilot", "seed": "UNSEALED_SENTINEL"})
    (campaign / "units/current-probe/checkpoint.pt").write_bytes(b"never load")
    before = {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    data = snapshot(root)
    text = rendered(book_for(data))
    for value in ("calibrating", "current-probe", "72", "2000", "3600", "3/9", "pending / not recorded"):
        assert value in text
    assert "UNSEALED_SENTINEL" not in text
    assert report.progress_rows(data)[0]["Sealed units"] == 0
    assert "Common-control learning and FD efficacy remain unverified" in report.conclusion(data)
    assert before == {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}


def test_phase2_progress_is_ignored_and_phase3_stop_is_explicit(tmp_path):
    campaign = tmp_path / "phase3" / "development-v1"
    data = snapshot(tmp_path, live=[
        {"path": str(tmp_path / "phase2/readiness-v1/orchestration"), "status": "OLD_PHASE2_SENTINEL"},
        {"path": str(campaign / "orchestration"), "status": "running", "stopped": True}])
    text = rendered(book_for(data))
    assert "stopped" in text and "OLD_PHASE2_SENTINEL" not in text
    assert "stopped" in report.conclusion(data)


def test_progress_prioritizes_child_steps_and_trajectories_and_separates_budget_from_eta(tmp_path):
    campaign = tmp_path / "phase3/development-v1"
    progress = {"path": str(campaign / "orchestration"), "status": "running",
        "active_seconds": 600, "remaining_budget_seconds": 21000, "remaining_estimate_seconds": 1200,
        "completed_trajectories": 2, "counts": {"step": 8, "target": 19},
        "active_unit": str(campaign / "units/pilot-control-193102"),
        "active_unit_progress": {"status": "training", "counts": {"step": 160, "target": 640}}}
    data = snapshot(tmp_path, live=[progress])
    before = json.dumps(data, sort_keys=True)
    row = report.progress_rows(data)[0]
    assert row["Completed trajectories"] == 2
    assert row["Active progress"] == "training: 160/640"
    assert row["Active unit"] == "pilot-control-193102"
    assert row["Budget remaining (s)"] == 21000 and row["Remaining estimate (s)"] == 1200
    assert row["Unit progress"] == "8/19" and row["Sealed units"] == 0
    assert list(row).index("Completed trajectories") < list(row).index("Unit progress")
    text = rendered(book_for(data))
    for value in ("Budget remaining (s)", "Remaining estimate (s)", "Completed trajectories", "training: 160/640"):
        assert value in text
    assert json.dumps(data, sort_keys=True) == before
    del progress["remaining_estimate_seconds"]
    progress["remaining_seconds"] = 21000
    row = report.progress_rows(data)[0]
    assert row["Budget remaining (s)"] == 21000 and row["Remaining estimate (s)"] is None


def test_missing_child_progress_and_trajectory_counts_stay_unknown_not_zero(tmp_path):
    campaign = tmp_path / "phase3/development-v1"
    progress = {"path": str(campaign / "orchestration"), "status": "calibrating",
                "counts": {"step": 3, "target": 19}, "active_unit_progress": {"status": "checking"}}
    row = report.progress_rows(snapshot(tmp_path, live=[progress]))[0]
    assert row["Active progress"] == "checking" and row["Completed trajectories"] is None
    assert row["Budget remaining (s)"] is None and row["Remaining estimate (s)"] is None
    progress["completed_trajectories"] = 0
    progress["active_unit_progress"]["counts"] = {"step": 0, "target": 640}
    row = report.progress_rows(snapshot(tmp_path, live=[progress]))[0]
    assert row["Completed trajectories"] == 0 and row["Active progress"] == "checking: 0/640"


def test_incompatible_progress_schema_fails_clearly_without_repair(tmp_path):
    from numerical_experiment.picar.report import ArtifactError

    path = tmp_path / "phase3/development-v1/orchestration/progress.json"
    write_json(path, {"schema_version": 999})
    before = path.read_bytes()
    with pytest.raises(ArtifactError, match="Phase 3 progress schema"):
        book_for(snapshot(tmp_path))
    assert path.read_bytes() == before


def test_full_report_is_scalar_only_and_preserves_unknown_calibration_and_selection(tmp_path):
    campaign = tmp_path / "phase3" / "development-v1"
    data = snapshot(tmp_path,
        unit(campaign, "calibration", rows=[
            {"name": "returns", "tau": 1.2, "return_min": -6., "return_max": 12.},
            {"name": "weights", "effective_count": 9.5, "status": None},
            {"name": "FD resolution", "resolvable": False, "passed": False}]),
        unit(campaign, "bootstrap", source="original demonstrations", acquired_count=292,
             distinct_bootstrap_samples=247, retired_count=239, eligible_count=8),
        pilot(campaign), pilot(campaign, name="fd", strength=.1, easy_final=12, hard_final=5),
        selection(campaign, control_learning_observed=True, retention_matched=None, retentionstatus="unverified",
                  fd_strengths=[.01, .1, 1.], reason="Resolution check remains incomplete"),
        unit(campaign, "forecast", phase3_active_seconds=3600, prior_study_seconds=12000,
             study_charged_seconds=15600, remaining_to_20h_seconds=56400, fd_cost_measured=True,
             scope_rows=[{"pairs": 4, "observations_per_arm": 1200, "fd_processing_hours": 4.5, "not_total_study_forecast": True}]),
        unit(campaign, "software", passed=True, test_result="tiny fixture checks"))
    for u in data["complete_units"]:
        u["summary"].update(frame_sha256=["FRAME_METADATA_SENTINEL"],
            fit_metrics=[{"row_total": 999, "vector": ["FIT_METADATA_SENTINEL"]}],
            history_digests=["HISTORY_SENTINEL"], distill_receipt={"steps": 247})
    before = json.dumps(data, sort_keys=True)
    book = book_for(data)
    text = rendered(book)
    for value in ("returns", "weights", "FD resolution", "resolvable", "False", "9.5", "tau",
                  "original demonstrations", "247", "239", "lr", "steps", "action_variance", "320",
                  "12/20 (60.0%)", "11/20 (55.0%)", "4/10 (40.0%)", "5/10 (50.0%)",
                  "matched seed IDs", "fd_strengths", "retention_matched", "unverified", "15600", "56400", "4.5"):
        assert value in text
    for value in ("FRAME_METADATA_SENTINEL", "FIT_METADATA_SENTINEL", "HISTORY_SENTINEL", "fit_metrics", "history_digests", "distill_receipt"):
        assert value not in text
    assert text.count("Development retention/acquisition frontier") == 1
    assert sum("image/png" in o.get("data", {}) for o in book.cells[-1].outputs) == 1
    assert "iid sample count" in text and "sample savings are unavailable" in text
    assert json.dumps(data, sort_keys=True) == before
    assert not campaign.exists()
    nbformat.validate(book)


def test_versions_are_chronological_separate_and_latest_pending_does_not_inherit_pass(tmp_path):
    old, new = (tmp_path / "phase3" / f"development-v{i}" for i in (9, 10))
    old_pilot = pilot(old, easy_final=20)
    old_pilot["manifest"]["completed_at"] = "2026-11-12T10:00:00Z"
    old_selection = selection(old, control_learning_observed=True, retention_matched=True)
    old_selection["manifest"]["completed_at"] = "2026-11-12T10:00:00Z"
    data = snapshot(tmp_path, old_pilot, old_selection,
        live=[{"path": str(new / "orchestration"), "status": "collecting", "updated_at": "2026-11-13T10:00:00Z"}])
    text = rendered(book_for(data))
    assert text.index("campaign: development-v9") < text.index("campaign: development-v10")
    assert "20/20 (100.0%)" in text and "20/40" not in text
    assert text.count("versions are not pooled") == 2
    end = report.conclusion(data)
    assert "development-v10" in end and "remain unverified" in end
    assert "ready for the development check-in" not in end


def test_calibration_collections_keep_recorded_resolution_and_common_recipe_names(tmp_path):
    campaign = tmp_path / "phase3/development-v1"
    data = snapshot(tmp_path, unit(campaign, "calibration", returns={
        "selected_temperature": 5., "return_mean": 2., "robust_spread": 10.,
        "temperature_rows": [{"temperature": 5., "weight_effective_fraction": .4, "maximum_weight_share": .05}],
        "advantage_quantiles": ["DO_NOT_RENDER_VECTOR"]}, exploration={
        "selected_exploration_std": 8., "exploration_rows": [{"exploration_std": 8., "forward_steering_fraction": .3}]}),
        unit(campaign, "probe", optimizer_rows=[{"learning_rate": .001, "optimizer_steps": 2,
            "data_nll_decrease": .3, "symmetric_model_kl": .01, "finite": True}],
            fd_convergence_pass=False, selected_fd_step=None, data_gradient_norm=7.2,
            fd_resolution_groups=[{"passed": False, "rows": [{"epsilon": .01, "penalty": 1e-5,
                "gradient_norm": .7, "cosine_to_previous": .8}]}],
            fd_strength_rows=[{"target_gradient_ratio": .1, "fd_strength": .001}]))
    text = rendered(book_for(data))
    for value in ("weight_effective_fraction", "selected_temperature", "forward_steering_fraction",
                  "data_nll_decrease", "fd_convergence_pass", "False", "FD resolution 1", "target_gradient_ratio"):
        assert value in text
    assert "DO_NOT_RENDER_VECTOR" not in text
    u = unit(campaign, "pilot", settings={"name": "recipe", "fd_step": .01, "pi": .025})
    assert report._settings(u)["fd_step"] == .01 and report._settings(u)["pi0"] == .025


def test_actual_selection_candidate_gaps_comparisons_and_measured_forecast_are_preserved(tmp_path):
    from numerical_experiment.picar_phase3 import analysis

    campaign = tmp_path / "phase3/development-v1"
    strength = 1.234567890123456e24
    pilots = [pilot(campaign, name=name, seed=seed, strength=fd, easy_final=easy, hard_final=hard)
              for seed in (193101, 193102)
              for name, fd, easy, hard in (("control", 0., 11, 4), ("fd", strength, 12, 5))]
    for u in pilots:
        u["summary"]["unit_path"] = u["path"]
        u["summary"]["timings"]["evaluation_seconds"] = 10.
    decision = analysis.selection([u["summary"] for u in pilots])
    forecast = analysis.forecast([u["summary"] for u in pilots], decision,
                                phase3_seconds=3600, prior_study_seconds=20719.62480906)
    decision["candidate_rows"][0]["features"] = ["NO_RAW_FEATURES"]
    data = snapshot(tmp_path, *pilots, unit(campaign, "selection", **decision), unit(campaign, "forecast", **forecast))
    before = json.dumps(data, sort_keys=True)
    book = book_for(data)
    text = rendered(book)
    for value in ("old_gap_average", "old_gap_endpoint", "new_gain_endpoint", "retention_average_difference",
                  "retention_endpoint_difference", "new_endpoint_difference", "ready_to_propose_confirmation",
                  "development_replica_count", "retention_tolerance", "control_retention_limit", "fd_processing_multiplier",
                  str(strength), "FD minus the same-seed control"):
        assert value in text
    assert "NO_RAW_FEATURES" not in text and "sample savings are unavailable" in text
    assert json.dumps(data, sort_keys=True) == before
    evidence = report._selection_evidence(decision)
    assert str(strength) in evidence[0].data["text/plain"]
    assert report._forecast_rows(forecast)[1]["Recorded"] == 20719.62480906


def test_actual_numerical_gate_and_fit_cost_rows_preserve_large_fd_scale(tmp_path):
    campaign = tmp_path / "phase3/development-v1"
    strength = 8.765432109876543e21
    probe = unit(campaign, "probe", fd_convergence_pass=False, selected_fd_step=None,
        fd_resolution_groups=[{"passed": False, "rows": [
            {"epsilon": eps, "penalty": 1e-9, "gradient_norm": 1e-10, "cosine_to_previous": None if i == 0 else .5}
            for i, eps in enumerate((.005, .01, .02))]}],
        fit_cost_rows=[{"fd_strength": fd, "optimizer_steps": 2, "median_update_seconds": seconds, "repetitions": 3}
                       for fd, seconds in ((0., .05), (strength, .08))],
        numeric_weight_safe=True, teacher_logit_variance_max=0., hypothetical_draws=16,
        coverage_only_not_reward_selection=True)
    for row in probe["summary"]["fd_resolution_groups"][0]["rows"]:
        row.update(ltm_gradient_norm=6e-11, stm_gradient_norm=8e-11)
    rows = report._diagnostics(probe)
    groups = [r for r in rows if "FD resolution" in r["Check"]]
    costs = [r for r in rows if "fit_cost_rows" in r["Check"]]
    assert len(groups) == 3 and all(r["Status"] is False for r in groups)
    assert all(r["Recorded"]["ltm_gradient_norm"] == 6e-11 and r["Recorded"]["stm_gradient_norm"] == 8e-11 for r in groups)
    assert len(costs) == 2 and costs[1]["Recorded"]["fd_strength"] == strength
    assert rows[0]["Status"] is False
    text = rendered(book_for(snapshot(tmp_path, probe)))
    for value in ("numeric_weight_safe", "teacher_logit_variance_max", "hypothetical_draws", "median_update_seconds",
                  "repetitions", "coverage_only_not_reward_selection", "ltm_gradient_norm", "stm_gradient_norm", str(strength)):
        assert value in text


def test_conclusion_respects_recorded_replication_and_fd_resolution_gates(tmp_path):
    campaign = tmp_path / "phase3/development-v1"
    decision = selection(campaign, control_learning_observed=True, retention_matched=True,
                         ready_to_propose_confirmation=False, development_replica_count=1)
    data = snapshot(tmp_path, decision)
    assert "still block a confirmation proposal" in report.conclusion(data)
    assert "ready for the development check-in" not in report.conclusion(data)
    data["complete_units"].append(unit(campaign, "probe", fd_convergence_pass=False))
    assert "FD numerical resolution failed" in report.conclusion(data)
    assert "not an FD efficacy result" in report.conclusion(data)


@pytest.mark.parametrize("change,expected", [("reverse", True), ("seed", False), ("duplicate", False),
                                           ("missing", False), ("geometry", False), ("partial", False)])
def test_panel_identity_and_full_panel_coverage_are_checked(change, expected, tmp_path):
    u = pilot(tmp_path / "phase3/development-v1")
    rows = u["summary"]["evaluations"][1]["rows"]
    if change == "reverse":
        rows.reverse()
    elif change == "seed":
        rows[0]["seed"] = 999
    elif change == "duplicate":
        rows[1]["seed"] = rows[0]["seed"]
    elif change == "missing":
        del rows[0]["seed"]
    elif change == "geometry":
        rows[0]["layout"]["direct_path_blocked"] = True
    else:
        u["summary"]["evaluations"][0]["rows"].pop()
        rows.pop()
    assert report._panel(u, "easy")[2] is expected


def test_frontier_omits_unverified_or_incomplete_pairs_and_unknown_success_is_not_zero(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    campaign = tmp_path / "phase3/development-v1"
    valid, invalid = pilot(campaign), pilot(campaign, name="missing")
    invalid["summary"]["evaluations"][1]["rows"][0]["success"] = None
    calls = []
    original = Axes.plot

    def plot(axis, *args, **kwargs):
        calls.append(args)
        return original(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot)
    output = report._frontier([valid, invalid])
    assert len(calls) == 1
    assert list(calls[0][0]) == [.6, .55] and list(calls[0][1]) == [.1, .4]
    assert base64.b64decode(output.data["image/png"]).startswith(b"\x89PNG")
    comparison = report._comparisons([invalid])[0]
    assert "1 unknown" in comparison["Easy A, before to after"]
    assert "11/20 (55.0%)" not in comparison["Easy A, before to after"]


def test_mid_checkpoints_and_missing_endpoints_are_visible_without_false_failures(tmp_path):
    campaign = tmp_path / "phase3/development-v1"
    u = pilot(campaign)
    u["summary"]["evaluations"] = [evaluation("mid-160", "easy", panel(20, 15))]
    text = rendered(book_for(snapshot(tmp_path, u)))
    assert "mid-160" in text and "15/20 (75.0%)" in text
    assert "pending / not recorded to pending / not recorded" in text
    assert "missing seed IDs" in text


@pytest.mark.parametrize("learning,matched,phrase", [
    (False, None, "not a negative FD result"), (None, None, "remain unverified"),
    (True, False, "qualified trade-off, not a matched win"), (True, None, "retention matching remains unverified"),
    (True, True, "ready for the development check-in, not a presumed FD win"),
    ("true", True, "remain unverified")])
def test_conclusion_distinguishes_common_learning_retention_and_fd_evidence(tmp_path, learning, matched, phrase):
    campaign = tmp_path / "phase3/development-v1"
    data = snapshot(tmp_path, selection(campaign, control_learning_observed=learning, retention_matched=matched,
                    reason="Recorded gate assessment"), unit(campaign, "campaign", status="ready_for_checkin"))
    text = report.conclusion(data)
    assert phrase in text and "Recorded gate assessment" in text
    assert "sample savings are unavailable until fresh paired Phase 4 trajectories" in text
    assert "no automatic Phase 4" in text and "next_phase_authorized | False" in text


def test_completed_gates_with_missing_recipe_references_do_not_claim_ready(tmp_path):
    campaign = tmp_path / "phase3/development-v1"
    data = snapshot(tmp_path, unit(campaign, "selection", control_learning_observed=True, retention_matched=True,
                                   selected_control=True, selected_fd=True))
    assert "sealed recipe selection remains incomplete" in report.conclusion(data)
    assert "ready for the development check-in" not in report.conclusion(data)


def test_preselected_video_labels_and_safe_links_use_actual_controller_and_candidate(tmp_path):
    campaign = tmp_path / "phase3/development-v1"
    u = pilot(campaign)
    path = Path(u["path"])
    path.mkdir(parents=True)
    (path / "before clip.mp4").write_bytes(b"link only")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside campaign")
    (path / "escape.mp4").symlink_to(outside)
    u["summary"]["videos"] = [
        {"actor": "learned_student", "checkpoint": "initial", "difficulty": "hard", "path": "before clip.mp4"},
        {"controller": "<unsafe>", "checkpoint": "final", "difficulty": "easy", "path": "escape.mp4"}]
    book = book_for(snapshot(tmp_path, u), notebook_dir=tmp_path)
    markup = "\n".join(o.get("data", {}).get("text/html", "") for o in book.cells[-1].outputs)
    assert "before%20clip.mp4" in markup and "learned_student; checkpoint=initial; difficulty=hard" in markup
    assert "&lt;unsafe&gt;" in markup and "<unsafe>" not in markup
    assert "outside.mp4" not in markup and 'href="escape.mp4"' not in markup
    assert "Preselected candidate clips: control / 193101" in rendered(book)


def test_code_cell_executes_with_all_model_checkpoint_and_gpu_imports_forbidden(tmp_path):
    campaign = tmp_path / "phase3/development-v1"
    data = snapshot(tmp_path, pilot(campaign), unit(campaign, "calibration", rows=[{"name": "weights", "value": .3}]))
    script = """
import builtins, contextlib, io, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    forbidden = ('torch', 'transformers', 'pybullet', 'numerical_experiment.picar.models',
                 'numerical_experiment.picar.storage', 'numerical_experiment.picar.learner',
                 'numerical_experiment.picar_phase3', 'numerical_experiment.picar_phase2',
                 'numerical_experiment.picar_phase2_rescue', 'numerical_experiment.picar_feature_health')
    if any(name == p or name.startswith(p + '.') for p in forbidden):
        raise AssertionError('runtime import in report: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import nbformat
from numerical_experiment.picar.phase3_report import append_phase3, conclusion
snapshot = json.loads(sys.argv[1])
book = append_phase3(nbformat.v4.new_notebook(), snapshot)
with contextlib.redirect_stdout(io.StringIO()):
    exec(book.cells[-1].source, {'snapshot': snapshot})
assert 'torch' not in sys.modules
assert conclusion(snapshot).startswith('## Conclusion')
nbformat.validate(book)
print('artifact-only Phase 3 cell executed')
"""
    result = subprocess.run([sys.executable, "-c", script, json.dumps(data)], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[2], env={**os.environ, "MPLBACKEND": "Agg"}, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "artifact-only Phase 3 cell executed" in result.stdout
