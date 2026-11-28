"""Bounded JSON fixtures for Fix 6; no GPU, learner or simulator execution."""

import json
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

from numerical_experiment.picar_coaching import report


TASK_ONLY_PROTOCOL = {"reward_mode": "task_only", "adherence_enabled": False, "compliance_raw_magnitude": 0}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return path


def progress(root, **fields):
    return write(root / "progress.json", {"status": "running", "stage": "collection", "completed_units": 0,
                 "planned_units": 12, "charged_seconds": 720, "forecast": {"remaining_hours": [10, 13]},
                 "active_progress": {"condition": "G", "seed": 7, "n": 23}, **fields})


def episode(value=2., *, seed=0, **fields):
    return {"success": False, "raw_return": value, "task_return": value, "guidance_return": 0.,
            "episode_id": f"e-{seed}", "seed": seed, "n": 0, "end_reason": "deadline", "videos": [], **fields}


def panel(n=1024, value=2., environment="initial_A", count=8):
    return {"n": n, "acquired_count": n + 3, "environment": environment,
            "episodes": [episode(value, seed=i, n=n) for i in range(count)]}


def unit(root, name, *, condition="U", seed=7, phase="assessment", operation="trajectory", sealed=True, **fields):
    settings = {"condition": condition, "seed": seed, "phase": phase, "total": 1024, "recipe": {"ridge": .1}}
    settings.update(fields.pop("settings", {}))
    value = {"schema_version": 1, "operation": operation, "settings": settings, "active_seconds": 30., **fields}
    path = root / "units" / name
    write(path / "summary.json", value)
    if sealed:
        (path / "COMPLETED").touch()
    return path


def text(outputs, mime="text/plain"):
    return "\n".join(o["data"].get(mime, "") for o in outputs)


@pytest.fixture(autouse=True)
def no_historical_notebook(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "DEFAULT_NOTEBOOK", tmp_path / "missing.ipynb")


def test_absent_root_is_read_only_and_can_build_provisional_notebook(tmp_path):
    root = tmp_path / "missing"
    snap = report.load_report(root)
    assert not snap["complete"] and not snap["units"]
    assert "Not started / incomplete" in text(report.progress_outputs(snap))
    report.build_notebook(root, tmp_path / "result.ipynb")
    assert not root.exists()
    book = nbformat.read(tmp_path / "result.ipynb", as_version=4)
    nbformat.validate(book)
    assert "Provisional / incomplete" in book.cells[-1].source


def test_failed_readiness_does_not_claim_unpassed_geometry_or_show_pending_learning_tables(tmp_path):
    progress(tmp_path, status="readiness_failed")
    unit(tmp_path, "geometry", operation="readiness_geometry", phase="readiness", ready=False,
         rows=[{"collision": True, "minimum_clearance_m": -.1}])
    unit(tmp_path, "compliance", operation="readiness_compliance", phase="readiness", ready=False,
         rows=[], audit_clear_correct=1, audit_clear_total=2)
    snapshot = report.load_report(tmp_path)
    conclusion = report.conclusion(snapshot)
    assert "No sealed online-learning trajectories" in conclusion
    assert "checks passed" not in conclusion and "GPU is released" not in conclusion
    outputs = text(report.result_outputs(snapshot))
    assert "Paired Endpoints" not in outputs and "Outcome-Selected Gallery" not in outputs


def test_readiness_images_are_original_saved_arrays_and_need_no_checkpoint(tmp_path):
    import numpy as np
    from numerical_experiment.picar.storage import FrameStore
    path = tmp_path / "units/compliance"
    frames = FrameStore(path / "frames")
    before, after = frames.put(np.zeros((8, 8, 3), dtype=np.uint8)), frames.put(np.full((8, 8, 3), 255, dtype=np.uint8))
    rows = [{"index": i, "panel": "audit", "correct": bool(i), "expected": "adherent",
             "judgment": "adherent" if i else "violated", "reason": "recorded test judgment",
             "before_ref": before, "after_ref": after, "evidence": {"instructions": [{"text": "Turn left."}],
                 "controls": [{"executed": [0., 0., -.4, .2]}]}} for i in range(2)]
    unit(tmp_path, "compliance", operation="readiness_compliance", phase="readiness", rows=rows, ready=False,
         audit_clear_correct=1, audit_clear_total=2)
    outputs = report.readiness_image_outputs(report.load_report(tmp_path))
    assert len(outputs) == 1 and outputs[0]["data"]["image/png"]
    assert "first incorrect and first correct" in text(outputs)


def test_history_metadata_outputs_and_user_additions_survive_refresh(tmp_path):
    root, output = tmp_path / "run", tmp_path / "result.ipynb"
    book = nbformat.v4.new_notebook(metadata={"annotation": {"keep": 1}}, cells=[
        nbformat.v4.new_markdown_cell("# Saved history", id="title"),
        nbformat.v4.new_code_cell("print('old')", execution_count=19, id="plan19-phase3-fix6-progress",
            metadata={"plan19_phase3_fix5a": True, "custom": [1]},
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="historical output\n")]),
        nbformat.v4.new_markdown_cell("Old conclusion", id="prior-conclusion")])
    original = json.loads(nbformat.writes(book))
    history = write(root / "history.ipynb", original)
    progress(root)
    report.build_notebook(root, output, history)
    first = json.loads(output.read_text())
    assert first["cells"][1]["metadata"][report._TAG]
    assert [c for c in first["cells"] if report._TAG not in c["metadata"]] == original["cells"]
    addition = dict(nbformat.v4.new_markdown_cell("New user note", id="user-note"))
    first["cells"].append(addition)
    write(output, first)
    progress(root, charged_seconds=1440)
    report.build_notebook(root, output, history)
    updated = json.loads(output.read_text())
    assert [c for c in updated["cells"] if report._TAG not in c["metadata"]] == original["cells"] + [addition]
    assert updated["metadata"] == original["metadata"]
    assert len({c["id"] for c in updated["cells"]}) == len(updated["cells"])
    assert len(first["cells"]) == len(updated["cells"])
    assert "0.4" in text(updated["cells"][1]["outputs"])
    assert "Phase 3 Fix 6" in updated["cells"][-3]["source"]
    assert json.loads(history.read_text()) == original


@pytest.mark.parametrize("source", ["progress", "orchestration", "root"])
def test_task_only_protocol_disables_adherence_without_pending_gate(tmp_path, source):
    root = tmp_path / ("coaching-fix6-v4" if source == "root" else "renamed-run")
    if source == "progress":
        progress(root, protocol=TASK_ONLY_PROTOCOL, planned_units=23)
    elif source == "orchestration":
        write(root / "orchestration/config.json", {"version": report.VERSION, "protocol": TASK_ONLY_PROTOCOL})
    snapshot = report.load_report(root)
    gate = next(r for r in report.readiness_rows(snapshot) if r["Gate"] == "Qwen compliance")
    assert gate["Status"] == "disabled by design"
    assert gate["Evidence"] == TASK_ONLY_PROTOCOL
    assert "No compliance readiness gate" in gate["Reason / qualification"]
    assert "disabled by design" in text(report.progress_outputs(snapshot))
    settings = report.settings_rows(snapshot)
    assert all(r["Adherence"] == "disabled by design" and r["Compliance bonus (raw)"] == 0 for r in settings)
    assert {r["Training reward"] for r in settings} == {"Task-only; identical task reward for U and G"}
    assert settings[1]["Online guidance"] == "50% to zero by n/N=.75"
    conclusion = report.conclusion(snapshot)
    assert "identical task-only reward" in conclusion and "hypothesis is untested" in conclusion
    assert "next required repair is reliable adherence evaluation" not in conclusion
    assert report.VERSION == "plan19-phase3-fix6-v1"


def test_task_only_history_preserves_all_64_v3_cells_and_refreshes_only_v4(tmp_path):
    legacy, root, output = tmp_path / "coaching-fix6-v3", tmp_path / "coaching-fix6-v4", tmp_path / "result.ipynb"
    book = nbformat.v4.new_notebook(metadata={"history": "keep"}, cells=[
        nbformat.v4.new_markdown_cell(f"Historical cell {i}", id=f"history-{i}") for i in range(60)])
    base = write(legacy / "history.ipynb", json.loads(nbformat.writes(book)))
    progress(legacy, status="readiness_failed")
    unit(legacy, "compliance", operation="readiness_compliance", phase="readiness", ready=False,
         audit_clear_correct=1, audit_clear_total=2, rows=[])
    report.build_notebook(legacy, output, base)
    original_bytes = output.read_bytes()
    original = json.loads(original_bytes)
    assert len(original["cells"]) == 64
    old_cells = [c for c in original["cells"] if report._TAG in c["metadata"]]
    assert len(old_cells) == 4 and "1/2" in old_cells[-1]["source"]
    history = write(root / "history.ipynb", original)
    history_bytes = history.read_bytes()
    progress(root, protocol=TASK_ONLY_PROTOCOL, planned_units=23)
    report.build_notebook(root, output)
    current = json.loads(output.read_bytes())
    assert len(current["cells"]) == 68
    assert [c for c in current["cells"] if report._TASK_ONLY_TAG not in c["metadata"]] == original["cells"]
    new_cells = [c for c in current["cells"] if report._TASK_ONLY_TAG in c["metadata"]]
    assert len(new_cells) == 4
    assert "Task-Reward-Only" in new_cells[1]["source"]
    assert "disabled by design" in text(new_cells[2]["outputs"])
    assert "task-only" in new_cells[0]["id"]
    assert "coaching-fix6-v3" in old_cells[0]["source"]  # Old live cells still load their own failed campaign.
    addition = dict(nbformat.v4.new_markdown_cell("User annotation", id="user-note"))
    current["cells"].append(addition)
    write(output, current)
    progress(root, protocol=TASK_ONLY_PROTOCOL, planned_units=23, charged_seconds=1440)
    report.build_notebook(root, output, history)
    updated = json.loads(output.read_bytes())
    assert [c for c in updated["cells"] if report._TASK_ONLY_TAG not in c["metadata"]] == original["cells"] + [addition]
    assert [c for c in updated["cells"] if report._TAG in c["metadata"]] == old_cells
    assert len(updated["cells"]) == 69 and updated["metadata"] == original["metadata"]
    assert history.read_bytes() == history_bytes
    assert "1/2 clear audit cases" in report.conclusion(report.load_report(legacy))


def test_task_only_diagnostics_keep_cues_rewards_and_pilot_failures_without_judge_expectations(tmp_path):
    legacy, root = tmp_path / "coaching-fix6-v3", tmp_path / "coaching-fix6-v4"
    unit(legacy, "compliance", operation="readiness_compliance", phase="readiness", ready=False,
         audit_clear_correct=1, audit_clear_total=2, rows=[])
    progress(root, protocol=TASK_ONLY_PROTOCOL)
    unit(root, "g", condition="G", n=10, reward_mode="task_only", adherence_enabled=False,
         episodes=[episode(n=1, coded_return=.2)],
         coaching_metrics=[{"windows": 2, "enabled": 1, "cues": 1, "active_instruction_seconds": 1.5,
                            "adherence_enabled": False, "assessable": 0, "scored": 0, "abstentions": 2}])
    snapshot = report.load_report(root)
    diagnostic = text(report.diagnostic_outputs(snapshot))
    assert "disabled by design" in diagnostic and "cues: 1" in diagnostic
    assert "active_instruction_seconds: 1.5" in diagnostic
    assert "abstentions: 2" in diagnostic and "geometric-coach abstentions" in diagnostic
    assert "Task raw return: 2" in diagnostic and "Guidance raw return: 0" in diagnostic
    assert "scored: 0" not in diagnostic and "assessable: 0" not in diagnostic and "+/-2" not in diagnostic
    rendered = text(report.result_outputs(snapshot))
    assert "Compliance gate: False" in rendered and "1/2" in rendered
    assert "Their compliance gates do not apply" in rendered
    assert "Compliance Confusion" not in rendered and "Compliance Audit Examples" not in rendered


def test_task_only_failed_geometry_does_not_require_adherence_repair(tmp_path):
    progress(tmp_path, protocol=TASK_ONLY_PROTOCOL, status="readiness_failed")
    unit(tmp_path, "geometry", operation="readiness_geometry", phase="readiness", ready=False,
         rows=[{"collision": True}])
    snapshot = report.load_report(tmp_path)
    conclusion = report.conclusion(snapshot)
    assert "Stopped at readiness" in conclusion and "Geometry gate failed" in conclusion
    assert "adherence judge got" not in conclusion and "reliable adherence evaluation" not in conclusion
    assert "disabled by design" in conclusion
    assert "Paired Endpoints" not in text(report.result_outputs(snapshot))


def test_initialization_coverage_uses_saved_turn_and_success_denominators(tmp_path):
    unit(tmp_path, "collection", operation="collection", phase="initialization", condition="U",
         n=769, records=655,
         episodes=[{"success": False}, {"success": True}, {"success": False}],
         action_metrics=[{"time_above_turn_threshold_s": 0.}, {"time_above_turn_threshold_s": 2.}, {}])
    rendered = text(report._initialization_outputs(report.load_report(tmp_path)))
    assert "Captured intervals: 769" in rendered and "Usable histories: 655" in rendered
    assert "Successes: 1" in rendered and "Turn episodes / measured: 1/2" in rendered
    assert "not held-out results" in rendered


@pytest.mark.parametrize("location", ["progress", "orchestration"])
def test_invalid_protocol_metadata_fails_clearly(tmp_path, location):
    if location == "progress":
        progress(tmp_path, protocol=[])
    else:
        write(tmp_path / "orchestration/config.json", {"protocol": []})
    with pytest.raises(report.ArtifactError, match="protocol must be an object"):
        report.load_report(tmp_path)


def test_never_opens_unsealed_summary_or_any_checkpoint(tmp_path, monkeypatch):
    path = unit(tmp_path, "active", sealed=False)
    (path / "summary.json").write_text("not parseable")
    (path / "checkpoint.pt").write_bytes(b"never load")
    write(path / "progress.json", {"stage": "coaching", "n": 51, "settings": {"condition": "G", "seed": 7}})
    progress(tmp_path, active_unit="active", active_progress={})
    opened, original = [], report._read

    def read(path, limit):
        opened.append(Path(path).name)
        return original(path, limit)

    monkeypatch.setattr(report, "_read", read)
    snap = report.load_report(tmp_path)
    assert opened == ["progress.json", "progress.json"]
    assert not snap["units"] and not snap["complete"]
    assert "51" in text(report.progress_outputs(snap))
    assert report.evaluation_rows(snap) == report.paired_endpoint_rows(snap) == []


@pytest.mark.parametrize("operation", sorted(report.OPERATIONS))
def test_all_contract_operations_accepted_without_version_field(tmp_path, operation):
    unit(tmp_path, operation, operation=operation)
    assert report.load_report(tmp_path)["units"][0]["summary"]["operation"] == operation


@pytest.mark.parametrize("changes,match", [({"schema_version": True}, "schema_version"),
    ({"schema_version": 2}, "schema_version"), ({"operation": "old"}, "operation"),
    ({"settings": {"condition": "B"}}, "condition"), ({"settings": {"phase": "confirmation"}}, "phase"),
    ({"settings": {"seed": True}}, "seed"), ({"settings": {"total": 0}}, "total"),
    ({"evaluations": [{"n": 0, "environment": "B", "episodes": []}]}, "environment"),
    ({"episodes": [{"success": 1}]}, "Boolean"),
    ({"episodes": [episode(videos=[{"path": ""}])]}, "video"),
    ({"curves": [{"epoch": 0, "train_kl": "wrong"}]}, "train_kl")])
def test_incompatible_sealed_summary_fails_clearly(tmp_path, changes, match):
    unit(tmp_path, "bad", **changes)
    with pytest.raises(report.ArtifactError, match=match):
        report.load_report(tmp_path)


def test_nonfinite_json_overflow_and_total_size_are_rejected(tmp_path, monkeypatch):
    path = unit(tmp_path, "bad") / "summary.json"
    path.write_text(path.read_text()[:-1] + ', "diagnostic": 1e999}')
    with pytest.raises(report.ArtifactError, match="nonfinite"):
        report.load_report(tmp_path)
    unit(tmp_path, "bad")
    monkeypatch.setattr(report, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(report.ArtifactError, match="total scalar"):
        report.load_report(tmp_path)


def test_summary_required_for_completed_unit_but_unsealed_is_provisional(tmp_path):
    path = unit(tmp_path, "missing")
    (path / "summary.json").unlink()
    with pytest.raises(report.ArtifactError, match="missing"):
        report.load_report(tmp_path)
    (path / "COMPLETED").unlink()
    assert report.load_report(tmp_path)["pending"]


def test_empty_present_progress_and_completed_count_mismatch(tmp_path):
    write(tmp_path / "progress.json", {})
    with pytest.raises(report.ArtifactError, match="status/stage"):
        report.load_report(tmp_path)
    progress(tmp_path, status="completed", planned_units=2, completed_units=2)
    unit(tmp_path, "only-one")
    assert not report.load_report(tmp_path)["complete"]
    unit(tmp_path, "second")
    assert report.load_report(tmp_path)["complete"]


def test_curve_common_four_not_endpoint_eight_and_task_not_total(tmp_path):
    p = panel()
    for e in p["episodes"]:
        e["raw_return"] = 999
    for e in p["episodes"][4:]:
        e["task_return"], e["success"] = 100, True
    unit(tmp_path, "u", evaluations=[p])
    row = report.evaluation_rows(report.load_report(tmp_path))[0]
    assert row["task_return"] == 2 and row["success"] == 0
    assert row["Layouts"] == 4 and row["Actual acquisitions"] == 1027


def test_missing_return_is_never_zero_or_substituted_by_total(tmp_path):
    p = panel()
    p["episodes"][0]["task_return"] = None
    unit(tmp_path, "u", evaluations=[p])
    assert report.evaluation_rows(report.load_report(tmp_path))[0]["task_return"] is None


def test_pairing_and_auc_keep_phase_seed_and_training_replica_distinct(tmp_path):
    for phase, seed, improvement in (("development", 3, 99.), ("assessment", 7, 4.), ("assessment", 8, 8.)):
        for condition in ("U", "G"):
            panels = [panel(n, value=1 + (improvement * n / 1024 if condition == "G" else 0), environment=env)
                      for env in report.ENVIRONMENTS for n in (0, 256, 512, 768, 1024)]
            unit(tmp_path, f"{phase}-{seed}-{condition}", condition=condition, phase=phase, seed=seed, evaluations=panels)
    snap = report.load_report(tmp_path)
    rows = report.paired_endpoint_rows(snap)
    assert len(rows) == 6
    row = next(r for r in rows if r["Seed"] == 7 and r["Environment"] == "initial_A")
    assert row["G-U task return"] == 4
    assert row["G-U budget-averaged return"] == 2
    assert row["G initial-A change (common layouts)"] == 4
    assert row["G successes/episodes"] == "0/8"
    means = report.paired_mean_rows(snap)
    assert next(r for r in means if r["Cohort"] == "assessment")["G-U task return"] == 6
    assert next(r for r in means if r["Cohort"] == "assessment")["Pairs"] == 2


@pytest.mark.parametrize("mismatch", ["budget", "layouts", "recipe", "duplicate", "phase", "seed", "missing"])
def test_unmatched_pairs_are_not_silently_compared(tmp_path, mismatch):
    unit(tmp_path, "u", evaluations=[panel()])
    args = {"condition": "G", "evaluations": [panel(value=8)]}
    if mismatch == "budget":
        args["settings"] = {"total": 2048}
    elif mismatch == "layouts":
        args["evaluations"][0]["episodes"][0]["seed"] = 999
    elif mismatch == "recipe":
        args["settings"] = {"recipe": {"ridge": 1}}
    elif mismatch == "phase":
        args["phase"] = "development"
    elif mismatch == "seed":
        args["seed"] = 8
    if mismatch != "missing":
        unit(tmp_path, "g", **args)
    if mismatch == "duplicate":
        unit(tmp_path, "g2", **args)
    assert report.paired_endpoint_rows(report.load_report(tmp_path)) == []


def test_auc_requires_all_checkpoints_not_endpoint_substitution(tmp_path):
    unit(tmp_path, "u", evaluations=[panel(0), panel()])
    unit(tmp_path, "g", condition="G", evaluations=[panel(0), panel()])
    row = report.paired_endpoint_rows(report.load_report(tmp_path))[0]
    assert row["G-U task return"] == 0
    assert row["G-U budget-averaged return"] is None


def test_heldout_cannot_have_guidance_rewards(tmp_path):
    p = panel()
    p["episodes"][0]["guidance_return"] = 2
    unit(tmp_path, "bad", evaluations=[p])
    with pytest.raises(report.ArtifactError, match="held-out guidance"):
        report.load_report(tmp_path)


def test_gallery_hierarchy_first_three_distinct_outcomes_missing_counts_and_scene_difficulty(tmp_path):
    rows = []
    for i in range(5):
        filename = f"clip {i}.mp4"
        rows.append(episode(seed=i, n=512, success=True, difficulty_start=.5, difficulty_end=.61,
                            videos=[{"path": filename}, {"path": filename}],
                            layout={"curriculum": {"direct_path_blocked": True}}, guidance_status="active"))
    path = unit(tmp_path, "g", condition="G", episodes=rows)
    # The first eligible episode is missing; later recordings must not replace it.
    for i in range(1, 5):
        (path / f"clip {i}.mp4").write_bytes(b"fixture")
    p = panel(1024, count=4)
    p["episodes"][0].update(videos=[{"path": "eval.mp4", "selection": "preselected"}], success=True)
    u = unit(tmp_path, "u", evaluations=[p])
    (u / "eval.mp4").write_bytes(b"fixture")
    snap = report.load_report(tmp_path)
    output = report.recording_outputs(snap, tmp_path)[0]
    markup = output["data"]["text/html"]
    assert "<details><summary>Condition G</summary><details><summary>initial</summary>" in markup
    assert "2/3 selected recorded; 5 available outcomes; target 3; shortfall 1" in markup
    assert "clip%201.mp4" in markup and "clip%203.mp4" not in markup
    assert "Missing recording" in markup and "Pending runs" in markup and "No successes observed" in markup
    assert "same-cohort evaluated successes/episodes=1/4" in markup
    assert "learning checkpoint=1024; actual difficulty=0 to 0" in markup
    assert "cohort=assessment" in markup and "direct path=blocked" in markup and "guidance=active" in markup
    assert "Common Preselected Recordings" in markup and "Outcome-Selected Gallery" in markup
    assert "[" not in markup.split("Outcome-Selected Gallery")[1]  # No Markdown links.


@pytest.mark.parametrize("value,expected", [(0, "initial"), (.1, "initial"), (.11, None), (.4, "medium"),
                                         (.6, "medium"), (.61, None), (.9, "final"), (1, "final")])
def test_episode_start_difficulty_windows(value, expected):
    assert report.difficulty_group({"difficulty_start": value, "n": 1024}, None) == expected


def test_rendered_recording_cell_executes_html_display_not_raw_markdown(tmp_path, monkeypatch):
    outputs = []
    import IPython.display
    monkeypatch.setattr(IPython.display, "display", lambda value, **kwargs: outputs.append((value, kwargs)))
    report.display_results(report.load_report(tmp_path), tmp_path)
    assert any("<details>" in value.get("text/html", "") for value, _ in outputs)
    assert all(kwargs == {"raw": True} for _, kwargs in outputs)


def test_action_binning_excludes_partial_history_and_separates_measured_yaw(tmp_path):
    rows = [{"complete_history": True, "difficulty_start": .5, "mean_abs_steering_deg": 10.,
             "turn_bouts": 2, "time_above_turn_threshold_s": 3., "measured_yaw_change_deg": 20.},
            {"complete_history": True, "difficulty_start": .5, "mean_abs_steering_deg": 20.,
             "turn_bouts": 4, "time_above_turn_threshold_s": 5., "measured_yaw_change_deg": 40.},
            {"complete_history": False, "difficulty_start": .5, "turn_bouts": 1000}]
    unit(tmp_path, "u", action_metrics=rows)
    row = report.action_rows(report.load_report(tmp_path))[0]
    assert row["Complete episodes"] == 2 and row["turn_bouts"] == 3
    assert row["mean_abs_steering_deg"] == 15 and row["measured_yaw_change_deg"] == 30


def test_live_control_summary_contract_without_complete_history_flag(tmp_path):
    unit(tmp_path, "u", action_metrics=[{"difficulty": .5, "n": 512, "turn_bouts": 2,
                                         "moving_seconds": 6., "mean_abs_steering_deg": 14.}])
    row = report.action_rows(report.load_report(tmp_path))[0]
    assert row["Difficulty bin"] == "medium" and row["turn_bouts"] == 2
    assert row["moving_seconds"] == 6 and row["measured_yaw_change_deg"] is None


def test_readiness_failed_report_explains_gate_counts_confusion_and_no_learning_claim(tmp_path):
    progress(tmp_path, status="readiness_failed", stage="readiness_compliance")
    unit(tmp_path, "geometry", operation="readiness_geometry", phase="readiness", ready=False,
         rows=[{"collision": True, "minimum_clearance_m": -.02}], qualification="Bounded geometry check")
    audit = [{"index": 0, "panel": "audit", "expected": "adherent", "judgment": "violated", "correct": False,
              "reason": "wrong direction", "evidence": {"instructions": [{"text": "Turn left."}]}},
             {"index": 1, "panel": "audit", "expected": "violated", "judgment": "violated", "correct": True},
             {"index": 2, "panel": "development", "expected": "adherent", "judgment": "adherent", "correct": True}]
    unit(tmp_path, "compliance", operation="readiness_compliance", phase="readiness", ready=False,
         audit_clear_correct=1, audit_clear_total=2, rows=audit)
    unit(tmp_path, "teacher", operation="readiness_teacher", phase="readiness", ready=False,
         correct_left_turns=0, correct_right_turns=3)
    snap = report.load_report(tmp_path)
    top = text(report.progress_outputs(snap))
    assert "readiness_failed" in top and "1/2 clear audit cases" in top and "left/right turns 0/3" in top
    outputs = report.readiness_outputs(snap)
    confusion = text(outputs)
    assert "Compliance Confusion" in confusion and "Sealed audit panel only" in confusion
    assert "wrong direction" in confusion and "Turn left." in confusion
    markup = text(outputs, "text/html")
    assert "Judged uncertain" in markup
    assert "No sealed online-learning trajectories" in report.conclusion(snap)
    assert "Provisional / incomplete" not in report.conclusion(snap)
    assert "Stopped at readiness, before initialization or learner training" in report.conclusion(snap)
    assert not snap["complete"] and not report.paired_endpoint_rows(snap)


def test_software_summary_with_empty_settings_and_feature_preparation_are_compatible(tmp_path):
    path = unit(tmp_path, "software", operation="software")
    value = json.loads((path / "summary.json").read_text())
    value["settings"] = {}
    write(path / "summary.json", value)
    unit(tmp_path, "semantics", operation="distillation_features", phase="initialization")
    snap = report.load_report(tmp_path)
    assert len(snap["units"]) == 2
    assert report.diagnostic_outputs(snap)


@pytest.mark.parametrize("phase", [None, "readiness"])
def test_final_audit_accepts_empty_settings_and_readiness_phase(tmp_path, phase):
    path = unit(tmp_path, "closeout", operation="final_audit", checks={"integrity": True})
    value = json.loads((path / "summary.json").read_text())
    value["settings"] = {} if phase is None else {"phase": phase}
    write(path / "summary.json", value)
    snap = report.load_report(tmp_path)
    assert snap["units"][0]["summary"]["operation"] == "final_audit"
    assert report.diagnostic_outputs(snap)


def test_coaching_metrics_are_compact_counts_not_unbounded_episode_table(tmp_path):
    rows = [{"episode_id": f"ep-{i}", "n": i, "windows": 2, "enabled": 1, "cues": 1, "assessable": 1, "scored": 1}
            for i in range(30)]
    unit(tmp_path, "g", condition="G", n=100, coaching_metrics=rows)
    rendered = text(report.diagnostic_outputs(report.load_report(tmp_path)))
    assert "Enabled window fraction: 0.5" in rendered
    assert "Cues per acquired interval: 0.3" in rendered
    assert "ep-29" not in rendered


def test_distillation_epoch_zero_turns_and_saved_postfit_ewc(tmp_path, monkeypatch):
    curves = [{"epoch": i, "train_kl": 1 / (i + 1), "validation_kl": .2 + i * .1,
               "train_turn_kl": 2., "validation_turn_kl": 3., "train_turn_agreement": .5,
               "validation_turn_agreement": .25} for i in range(3)]
    unit(tmp_path, "fit", operation="distillation", phase="initialization", curves=curves, selected_epoch=0)
    unit(tmp_path, "u", metrics=[{"n": 1, "ewc": 0., "post_step_ewc": 4., "post_step_fd": 3.}])
    figures = []
    monkeypatch.setattr(report, "_figure", lambda f, d: figures.append(f) or report._html("", d))
    snap = report.load_report(tmp_path)
    report._initialization_outputs(snap)
    assert list(figures[0].axes[0].lines[0].get_xdata()) == [0, 1, 2]
    assert list(figures[0].axes[0].lines[-1].get_xdata()) == [0, 0]
    figures.clear()
    report.diagnostic_outputs(snap)
    assert list(figures[0].axes[0].lines[0].get_ydata()) == [4]


def test_estimator_figure_uses_actual_first_step_fd_and_normalized_age(tmp_path, monkeypatch):
    unit(tmp_path, "u", phase="development", metrics=[
        {"n": 20, "ewc": 0., "post_step_ewc": 4., "fd": 2., "post_step_fd": 99., "age_mean": .25},
        {"n": 10, "ewc": 0., "post_step_ewc": 3., "fd": 1., "post_step_fd": 98., "age_mean": .125}])
    figures, descriptions = [], []

    def figure(value, description):
        figures.append(value)
        descriptions.append(description)
        return report._html("", description)

    monkeypatch.setattr(report, "_figure", figure)
    report.diagnostic_outputs(report.load_report(tmp_path))
    assert len(figures) == 1 and len(figures[0].axes) == 3
    axes = figures[0].axes
    assert list(axes[0].lines[0].get_ydata()) == [3., 4.]
    assert list(axes[1].lines[0].get_xdata()) == [10, 20]
    assert list(axes[1].lines[0].get_ydata()) == [1., 2.]
    assert "First-step" in axes[1].get_title() and "weighted" in axes[1].get_title()
    assert list(axes[2].lines[0].get_ydata()) == [.125, .25]
    assert "capacity" in axes[2].get_ylabel()
    assert "not a post-fit value" in descriptions[0]


def test_estimator_health_is_one_figure_fresh_only_once_fresh_appears(tmp_path, monkeypatch):
    for phase, seeds in (("development", [3]), ("assessment", [7, 8])):
        for seed in seeds:
            for condition in ("U", "G"):
                unit(tmp_path, f"{phase}-{seed}-{condition}", phase=phase, seed=seed, condition=condition,
                     metrics=[{"n": 1, "post_step_ewc": seed, "fd": seed + 1, "age_mean": .5}])
    figures = []
    monkeypatch.setattr(report, "_figure", lambda f, d: figures.append(f) or report._html("", d))
    report.diagnostic_outputs(report.load_report(tmp_path))
    assert len(figures) == 1 and len(figures[0].axes) == 3
    for axis in figures[0].axes:
        assert len(axis.lines) == 4
        assert {line.get_label() for line in axis.lines} == {f"{c} seed {s}" for c in ("U", "G") for s in (7, 8)}


def test_estimator_development_only_until_fresh_with_missing_fields_left_unavailable(tmp_path, monkeypatch):
    for condition in ("U", "G"):
        unit(tmp_path, condition, condition=condition, phase="development",
             metrics=[{"n": 1, "post_step_ewc": 4., "post_step_fd": 100., "replay_age_mean": 20}])
    figures = []
    monkeypatch.setattr(report, "_figure", lambda f, d: figures.append(f) or report._html("", d))
    report._estimator_outputs(report.load_report(tmp_path))
    assert len(figures) == 1
    assert len(figures[0].axes[0].lines) == 2
    for axis in figures[0].axes[1:]:
        assert not axis.lines
        assert axis.texts[0].get_text() == "Not recorded"
    unit(tmp_path, "fresh-pending-metrics", phase="assessment", metrics=[])
    assert report._estimator_outputs(report.load_report(tmp_path)) == []


def test_escaped_html_and_no_media_binary_or_checkpoint_reads(tmp_path, monkeypatch):
    unit(tmp_path, "u", settings={"recipe": {"note": "<script>alert(1)</script>"}})
    markup = text(report.result_outputs(report.load_report(tmp_path)), "text/html")
    assert "&lt;script&gt;" in markup and "<script>" not in markup


def test_symlinks_and_completed_output_are_refused(tmp_path):
    root = tmp_path / "run"
    path = unit(root, "u")
    (root / "units/link").symlink_to(path, target_is_directory=True)
    with pytest.raises(report.ArtifactError, match="symlink"):
        report.load_report(root)
    (root / "units/link").unlink()
    (tmp_path / "COMPLETED").touch()
    with pytest.raises(report.ArtifactError, match="completed"):
        report.build_notebook(root, tmp_path / "result.ipynb")


def test_atomic_failure_leaves_existing_history_and_cleans_temp(tmp_path, monkeypatch):
    output = tmp_path / "result.ipynb"
    write(output, json.loads(nbformat.writes(nbformat.v4.new_notebook())))
    before = output.read_bytes()
    monkeypatch.setattr(report.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("test interruption")))
    with pytest.raises(OSError, match="interruption"):
        report.build_notebook(tmp_path / "absent", output)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".coaching-report-*"))


def test_import_and_report_do_not_import_runtime_torch_or_unpickle(tmp_path):
    code = """
import sys
class Guard:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith(('numerical_experiment.picar_head_awac', 'src.robotics')):
            raise RuntimeError('forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, Guard())
import pickle
pickle.load = pickle.loads = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('unpickle'))
from numerical_experiment.picar_coaching import report
snapshot = report.load_report(sys.argv[1])
report.build_notebook(sys.argv[1], sys.argv[2])
assert not snapshot['complete']
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "missing"), str(tmp_path / "result.ipynb")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
