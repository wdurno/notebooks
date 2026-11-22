"""Synthetic scalar envelopes only: no model, scoring, rollout or canonical writes."""

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
from numerical_experiment.picar_reward_repair import report


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def seal(path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


@pytest.fixture
def root(tmp_path):
    path = tmp_path / report.CAMPAIGN
    path.mkdir()
    return path


def unit(root, kind, *, name=None, condition=None, config=None, **fields):
    path = root / "units" / (name or kind)
    path.mkdir(parents=True)
    settings = {"condition": condition} if condition else {}
    settings.update(fields.pop("settings", {}))
    resolved = {"schema_version": 1, "version": report.VERSION, "mode": "production",
                "settings": settings, **(config or {})}
    stored = {"config": resolved}
    write_json(path / "config.json", stored)
    write_json(path / "summary.json", {"schema_version": 1, "kind": "reward_" + kind,
        "version": report.VERSION, "phase": 3, "development_only": True, "settings": settings, **fields})
    with (path / "metrics.csv").open("w") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_COLUMNS).writeheader()
    manifest = {"artifact_schema_version": 1, "metric_schema_version": 1, "storage_schema_version": 1,
        "status": "completed", "completed_at": "2026-11-17T12:00:00Z",
        "config_hash": hashlib.sha256(json.dumps(stored, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "final_files": {key: seal(path / key) for key in ("summary.json", "metrics.csv")}}
    write_json(path / "manifest.json", manifest)
    write_json(path / "COMPLETED", {"manifest_sha256": seal(path / "manifest.json")["sha256"]})
    return path


def progress(root, **fields):
    path = root / "orchestration"
    path.mkdir(exist_ok=True)
    value = {"schema_version": 1, "phase": 3, "version": report.VERSION,
             "status": "scoring", "max_hours": 12, **fields}
    write_json(path / "progress.json", value)
    return path


def commit_progress(path):
    transaction = path / "transactions" / "commit-1"
    transaction.mkdir(parents=True)
    (transaction / "progress.json").write_bytes((path / "progress.json").read_bytes())
    write_json(transaction / "manifest.json", {"files": {"progress.json": seal(transaction / "progress.json")}})
    write_json(path / "CURRENT.json", {"storage_schema_version": 1, "commit_id": "commit-1",
               "manifest": seal(transaction / "manifest.json")})


def incomplete_fit(root, *, condition=report.CONDITIONS[1], committed=True, operation="fit", curves=None):
    active = root / "units" / (operation + "-012345")
    active.mkdir(parents=True)
    write_json(active / "config.json", {"config": {"operation": operation, "settings": {"condition": condition}}})
    if curves is None:
        curves = [{"stage": "nll", "epoch": i, "train": {"joint_nll": 2-i*.1, "teacher_kl": .1},
                   "validation": {"joint_nll": 2.1-i*.1, "teacher_kl": .2},
                   "diagnostics": {"gradient_norm_mean": .3, "update_norm_mean": .01}} for i in (0, 1)]
    path = progress(root, status=operation, active_unit=str(active), completed_trajectories=0,
                    active_unit_progress={"stage": "nll", "counts": {"step": 7, "target": 8240},
                                          "diagnostic_curves": curves})
    if committed:
        commit_progress(path)
    return path, active


def episode(*, seed=11, score=4., n=2, condition=report.CONDITIONS[0], evaluation=True):
    scale = 10 if condition == report.CONDITIONS[1] else 1
    coded = [1. / scale] * n
    raw = [1.] * n
    return {"episode_id": f"episode-{seed}", "seed": seed, "complete": True,
        "evaluation": evaluation, "learning": False, "explore": False,
        "reward_scoring": True, "reward_condition": condition, "success": False,
        "collision": False, "end_reason": "deadline", "truncated": False,
        "reset_exposure": 0., "layout": {"domain": "A"},
        "intervals": [{"elapsed_s": .5, "frame_ref": f"rgb-{seed}-{i}"} for i in range(n)],
        "visual_scores": [{"score": score, "width_percent": 8 * score} for _ in range(n)],
        "stage_rewards": coded, "returns": [sum(coded[i:]) for i in range(n)],
        "raw_stage_rewards": raw, "raw_returns": [float(n-i) for i in range(n)],
        "terminal_bonuses": [0.] * (n-1) + [-1.], "return_bins": [1] * n,
        "counterfactual_rewards": {c: {"coded_stage_rewards": coded, "coded_returns": coded,
            "raw_stage_rewards": raw, "raw_returns": raw} for c in report.CONDITIONS}}


def panel(checkpoint="initial", difficulty="easy", *, condition=report.CONDITIONS[0], asset="asset-linear", rows=None):
    rows = [episode(condition=condition)] if rows is None else rows
    return {"checkpoint": checkpoint, "difficulty": difficulty, "initial_asset_id": asset,
            "acquired_count": {"initial": 0, "midpoint": 320, "final": 640}[checkpoint],
            "seeds": [r["seed"] for r in rows], "rows": rows}


def pilot(root, *, condition=report.CONDITIONS[0], seed=197101, episodes=None, evaluations=None,
          fit_metrics=None, **kwargs):
    return unit(root, "pilot", name=f"pilot-{condition}-{seed}", condition=condition,
                settings={"seed": seed, "initial_asset_id": "asset-linear"}, acquired_count=640,
                episodes=[episode(evaluation=False, condition=condition)] if episodes is None else episodes,
                evaluations=[panel(cp, condition=condition) for cp in ("initial", "midpoint", "final")] if evaluations is None else evaluations,
                fit_metrics=([{"first_step_ewc_suppression": .3, "update_seconds": .2}]
                             if fit_metrics is None else fit_metrics),
                timings={"active_seconds": 10., "fit_seconds": 1., "scoring_seconds": 2.}, **kwargs)


def closeout_condition(root, condition, *, final_a, final_b, initial_a=2, initial_b=0):
    def outcomes(checkpoint, domain, successes):
        rows = [episode(seed=seed, condition=condition) for seed in (11, 12)]
        for index, row in enumerate(rows):
            row["success"] = index < successes
        return panel(checkpoint, domain, condition=condition, rows=rows)

    initial = [outcomes("initial", "easy", initial_a), outcomes("initial", "hard", initial_b)]
    unit(root, "initial_evaluation", name="initial-" + condition, condition=condition, evaluations=initial)
    for index, seed in enumerate((197101, 197102)):
        if final_a[index] is None and final_b[index] is None:
            continue
        evaluations = copy.deepcopy(initial)
        for domain, successes in (("easy", final_a[index]), ("hard", final_b[index])):
            if successes is not None:
                evaluations.append(outcomes("final", domain, successes))
        pilot(root, condition=condition, seed=seed, evaluations=evaluations)


def history(root):
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Historical results"),
        nbformat.v4.new_code_cell("print('Historical progress')", metadata={"report_section": "progress"},
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="Historical progress\n")]),
        nbformat.v4.new_markdown_cell("### Phase 3 Fix 2\n\nHistorical finding."),
        nbformat.v4.new_markdown_cell("## Conclusion\n\nHistorical recommendation stays unchanged.")])
    path = root / "history.ipynb"
    nbformat.write(book, path)
    return path, book


def rendered(book):
    return "\n".join(c.source + "\n" + "\n".join(
        o.get("text", "") + o.get("data", {}).get("text/plain", "") for o in c.get("outputs", [])) for c in book.cells)


def test_pending_progress_is_read_only_and_not_zero(tmp_path):
    missing = tmp_path / "missing" / report.CAMPAIGN
    snapshot = report.load_report(missing)
    assert not missing.exists()
    rows = {r["Item"]: r["State"] for r in report.progress_rows(snapshot)}
    assert rows["Scoring done / remaining"] == report.PENDING + " / " + report.PENDING
    assert rows["Pilots completed / planned"] == report.PENDING + " / 4"
    assert "0/4 sealed" in report.conclusion(snapshot)
    assert not report.reward_curves(snapshot)


def test_operational_progress_never_enters_outcomes(root):
    path = progress(root, active_unit="scoring", completed_trajectories=0,
                    repair_active_seconds=360., remaining_budget_seconds=42840., forecast=[7200., 10800.],
                    active_unit_progress={"counts": {"scored": 3, "target": 8}, "success": 1.,
                        "evaluations": [{"checkpoint": "initial", "difficulty": "easy", "episodes": 20, "scored": 3}]})
    before = (path / "progress.json").read_bytes()
    snapshot = report.load_report(root)
    assert snapshot["complete_units"] == []
    rows = {r["Item"]: r["State"] for r in report.progress_rows(snapshot)}
    assert rows["Scoring done / remaining"] == "3 / 5"
    assert rows["Measured remaining h"] == "2.00--3.00"
    assert rows["Charged / remaining ceiling h (12 h cap)"] == "0.10 / 11.90"
    assert not report.endpoint_rows(snapshot)
    assert (path / "progress.json").read_bytes() == before


def test_sealed_current_progress_wins_over_mirror(root):
    path = progress(root, status="STALE", completed_trajectories=4)
    transaction = path / "transactions" / "commit-1"
    transaction.mkdir(parents=True)
    write_json(transaction / "progress.json", {"schema_version": 1, "phase": 3, "version": report.VERSION,
               "status": "fit", "completed_trajectories": 0})
    write_json(transaction / "manifest.json", {"files": {"progress.json": seal(transaction / "progress.json")}})
    write_json(path / "CURRENT.json", {"storage_schema_version": 1, "commit_id": "commit-1",
               "manifest": seal(transaction / "manifest.json")})
    snapshot = report.load_report(root)
    assert snapshot["progress"]["status"] == "fit"
    assert snapshot["progress"]["completed_trajectories"] == 0


def test_fit_progress_not_misread_as_scoring_and_hashed_active_unit(root):
    unit(root, "scoring", unique_images=8, scored_images=8, seconds=12.)
    active = root / "units" / "fit-012345"
    active.mkdir()
    write_json(active / "config.json", {"config": {"settings": {"condition": report.CONDITIONS[0]}}})
    progress(root, active_unit=str(active), active_unit_progress={"stage": "nll", "counts": {"step": 7, "target": 8240}},
             completed_trajectories=2, forecast={"four_pilots_hours": [2., 4.]})
    rows = {r["Item"]: r["State"] for r in report.progress_rows(report.load_report(root))}
    assert rows["Scoring done / remaining"] == "8 / 0"
    assert "running: nll 7/8240" in rows["Linear fit / Fisher"]
    assert rows["Measured remaining h"].startswith("1.00--2.00")
    assert "not an updated measurement" in rows["Measured remaining h"]


def test_committed_incomplete_fit_preview_before_any_sealed_fit(root, monkeypatch):
    parent, active = incomplete_fit(root)
    write_json(parent / "progress.json", {"status": "uncommitted mirror must not supply diagnostics"})
    write_json(active / "progress.json", {"diagnostic_curves": [{"train_nll": -999.}]})
    (active / "checkpoint.pt").write_bytes(b"Not opened by reporting")
    original_read = shared._read
    def read(path, limit):
        assert Path(path).suffix != ".pt"
        assert Path(path) != active / "progress.json"
        return original_read(path, limit)
    monkeypatch.setattr(shared, "_read", read)
    snapshot = report.load_report(root)
    assert snapshot["progress_committed"] is True
    preview = report._fit_preview(snapshot)
    assert preview["condition"] == report.CONDITIONS[1]
    assert [r["epoch"] for r in preview["curves"]] == [0, 1]
    assert preview["curves"][-1]["gradient_norm"] == .3
    assert snapshot["complete_units"] == []
    assert report.initialization_rows(snapshot) == []
    assert report.endpoint_rows(snapshot) == []
    assert "0/4 sealed pilot" in report.conclusion(snapshot)
    rows = {r["Item"]: r["State"] for r in report.progress_rows(snapshot)}
    assert rows["Pilots completed / planned"] == "0 / 4"
    assert "sealed" not in rows["Signed square fit / Fisher"]
    output = report.result_outputs(snapshot)
    assert sum("image/png" in o.get("data", {}) for o in output) == 1
    assert any("provisional/incomplete" in o.get("data", {}).get("text/plain", "").lower() for o in output)
    text = " ".join(o.get("text", "") for o in output)
    assert "committed NLL pass 1" in text and "No accepted checkpoint" in text
    assert "-999" not in text


def test_preview_coexists_with_sealed_curves_and_is_visibly_labeled(root, monkeypatch):
    unit(root, "fit", condition=report.CONDITIONS[0],
         curves=[{"stage": "nll", "epoch": 0, "train": {"joint_nll": 3.}, "validation": {"joint_nll": 4.}}],
         selection={"nll": {"epoch": 0}}, readiness={"status": "ready"})
    incomplete_fit(root, curves=[{"stage": "nll", "epoch": 0, "train": {"joint_nll": 1.},
                                  "validation": {"joint_nll": 2.}}])
    captured = []
    original = report._figure
    def figure(value, description):
        captured.append(([line.get_label() for line in value.axes[0].lines], value.axes[0].get_title()))
        return original(value, description)
    monkeypatch.setattr(report, "_figure", figure)
    snapshot = report.load_report(root)
    assert report._initialization_figure(snapshot) is not None
    labels, title = captured[0]
    assert "Linear train sealed" in labels
    assert "Signed square train provisional/incomplete" in labels
    assert "provisional/incomplete" in title
    assert len(report.initialization_rows(snapshot)) == 1


@pytest.mark.parametrize("committed, operation", [(False, "fit"), (True, "bootstrap")])
def test_uncommitted_or_nonfit_diagnostics_never_previewed(root, committed, operation):
    incomplete_fit(root, committed=committed, operation=operation)
    snapshot = report.load_report(root)
    assert report._fit_preview(snapshot) is None
    assert report._initialization_figure(snapshot) is None


def test_active_completed_fit_is_not_replotted_as_provisional(root):
    path = unit(root, "fit", name="fit-012345", condition=report.CONDITIONS[0],
                curves=[{"stage": "nll", "epoch": 0, "train": {"joint_nll": 3.}, "validation": {"joint_nll": 4.}}],
                selection={"nll": {"epoch": 0}}, readiness={"status": "ready"})
    parent = progress(root, status="fit complete", active_unit=str(path),
                      active_unit_progress={"diagnostic_curves": [{"invalid": True}]})
    commit_progress(parent)
    snapshot = report.load_report(root)
    assert report._fit_preview(snapshot) is None
    assert len(report.initialization_rows(snapshot)) == 1


@pytest.mark.parametrize("curves", [[{"stage": "nll", "epoch": 0}], {"epoch": 0}])
def test_committed_preview_rejects_incompatible_curves(root, curves):
    incomplete_fit(root, curves=curves)
    with pytest.raises(shared.ArtifactError, match="provisional fit"):
        report.load_report(root)


def test_preview_requires_admitted_active_settings_condition(root):
    incomplete_fit(root, condition="historical_replay_ewc")
    with pytest.raises(shared.ArtifactError, match="active_settings.condition"):
        report.load_report(root)


def test_empty_committed_fit_diagnostics_remain_pending(root):
    incomplete_fit(root, curves=[])
    snapshot = report.load_report(root)
    assert report._fit_preview(snapshot) is None
    assert report._initialization_figure(snapshot) is None


def test_preview_row_limit_and_parent_seal_are_checked(root, monkeypatch):
    parent, _ = incomplete_fit(root)
    monkeypatch.setattr(report, "MAX_PREVIEW_CURVES", 1)
    with pytest.raises(shared.ArtifactError, match="bounded row limit"):
        report.load_report(root)
    write_json(parent / "transactions" / "commit-1" / "progress.json", {})
    with pytest.raises(shared.ArtifactError, match="hash/size mismatch"):
        report.load_report(root)


@pytest.mark.parametrize("fields, match", [
    ({"schema_version": 2}, "schema"), ({"phase": 4}, "phase"),
    ({"version": "plan19-phase3-fix2-v3"}, "version"),
    ({"development_only": False}, "development_only"),
    ({"config": {"condition": "width6_signed_square"}}, "condition mismatch"),
    ({"fd_strength": .1}, "FD must be off"),
])
def test_invalid_summary_identity(root, fields, match):
    unit(root, "audit", condition=report.CONDITIONS[0], **fields)
    with pytest.raises(shared.ArtifactError, match=match):
        report.load_report(root)


def test_config_wrong_source_or_condition_rejected(root):
    unit(root, "audit", condition=report.CONDITIONS[0], config={"version": "historical-v1"})
    with pytest.raises(shared.ArtifactError, match="source version"):
        report.load_report(root)


@pytest.mark.parametrize("mutation, match", [
    (lambda p: p.update(initial_asset_id="wrong"), "source asset"),
    (lambda p: p.update(condition="width6_signed_square"), "condition mismatch"),
    (lambda p: p["rows"][0].update(reward_scoring=False), "complete, scored"),
    (lambda p: p["rows"][0].update(learning=True), "no-learning"),
    (lambda p: p["rows"][0].update(reward_condition="width6_signed_square"), "condition mismatch"),
    (lambda p: p["rows"][0]["visual_scores"][0].update(score=None), "invalid is not zero"),
    (lambda p: p["rows"][0].update(raw_returns=[]), "raw_returns"),
])
def test_evaluation_source_flags_and_arrays_checked(root, mutation, match):
    p = panel()
    mutation(p)
    pilot(root, evaluations=[p])
    with pytest.raises(shared.ArtifactError, match=match):
        report.load_report(root)


def test_shared_initial_deduplicated_and_conflicts_fail(root):
    initial = panel()
    unit(root, "initial_evaluation", condition=report.CONDITIONS[0], evaluations=[initial])
    pilot(root)
    pilot(root, seed=197102)
    snapshot = report.load_report(root)
    panels = report.evaluation_panels(snapshot)
    assert len([p for p in panels if p["panel"]["checkpoint"] == "initial"]) == 1
    assert len([p for p in panels if p["panel"]["checkpoint"] == "final"]) == 2
    curves = report.reward_curves(snapshot, grid=[.5])
    assert next(c for c in curves if c["label"] == "initial")["points"][0]["replicas"] == 1
    changed = copy.deepcopy(initial)
    changed["rows"][0]["success"] = True
    unit(root, "initial_evaluation", name="conflict", condition=report.CONDITIONS[0], evaluations=[changed])
    with pytest.raises(shared.ArtifactError, match="conflicting evaluation source"):
        report.load_report(root)


def test_different_initial_sources_are_not_pooled(root):
    unit(root, "initial_evaluation", name="initial-one", condition=report.CONDITIONS[0], evaluations=[panel()])
    unit(root, "initial_evaluation", name="initial-two", condition=report.CONDITIONS[0], evaluations=[panel(asset="different-asset")])
    with pytest.raises(shared.ArtifactError, match="different initial source assets"):
        report.load_report(root)


def test_mixed_frozen_evaluator_sources_fail(root):
    for condition in report.CONDITIONS:
        pilot(root, condition=condition, reward_repair={"version": report.VERSION, "condition": condition,
                                                      "scorer": {"identity": condition}})
    with pytest.raises(shared.ArtifactError, match="different frozen scorer sources"):
        report.load_report(root)


def test_equal_replica_weights_and_terminal_handling(root):
    def episodes(scores, n):
        return [episode(seed=i, score=v, n=n, evaluation=False) for i, v in enumerate(scores)]
    pilot(root, episodes=episodes([0., 0., 0.], 1))
    pilot(root, seed=197102, episodes=episodes([10.], 2))
    snapshot = report.load_report(root)
    curves = report.reward_curves(snapshot, grid=[.5, 1., 1.5])
    training = next(c for c in curves if c["mode"] == "training")
    assert training["points"] == [
        {"elapsed_s": .5, "mean": 5., "episodes": 4, "replicas": 2},
        {"elapsed_s": 1., "mean": 10., "episodes": 1, "replicas": 1}]
    cumulative = next(c for c in report.reward_curves(snapshot, grid=[.5, 1., 1.5], cumulative=True) if c["mode"] == "training")
    assert [p["mean"] for p in cumulative["points"]] == [1., 1.5, 1.5]
    assert all(p["replicas"] == 2 for p in cumulative["points"])


def test_post_action_scores_are_not_backfilled_or_zero_padded():
    row = episode(n=2)
    row["intervals"][0]["elapsed_s"] = .3
    row["visual_scores"][0]["score"] = 2.
    row["visual_scores"][1]["score"] = 8.
    assert report._episode_value(row, 0.) is None
    assert report._episode_value(row, .3) == 2.
    assert report._episode_value(row, .5) == 2.
    assert report._episode_value(row, .8) == 8.
    assert report._episode_value(row, 1.) is None
    assert report._episode_value(row, 1., cumulative=True) == 2.


def test_scoring_and_initializer_units(root):
    unit(root, "scoring", unique_images=8, scored_images=8, seconds=12.)
    for condition in report.CONDITIONS:
        unit(root, "fit", name="fit-" + condition, condition=condition,
            curves=[{"stage": "nll", "epoch": i, "train": {"joint_nll": 2-i*.1, "teacher_kl": .1},
                     "validation": {"joint_nll": 2.1-i*.1, "teacher_kl": .2}} for i in (0, 1)],
            selection={"nll": {"epoch": 1}}, readiness={"status": "ready"}, counts={"records": 515})
        unit(root, "bootstrap", name="fisher-" + condition, condition=condition,
            fisher_stability={"nested_draws": [{"draws_per_history": n, "relative_l2": 1/n} for n in (1, 2, 4)]},
            normalization={"history_count": 515, "draws_per_history": 4})
    snapshot = report.load_report(root)
    rows = {r["Item"]: r["State"] for r in report.progress_rows(snapshot)}
    assert rows["Scoring done / remaining"] == "8 / 0"
    assert rows["Linear fit / Fisher"] == "ready / sealed"
    outputs = report.result_outputs(snapshot)
    assert sum("image/png" in o.get("data", {}) for o in outputs) == 1
    assert len(report.initialization_rows(snapshot)) == 2


def test_readiness_can_have_no_panels_and_training_uses_saved_strata(root):
    unit(root, "readiness", condition=report.CONDITIONS[0], evaluations=[], episodes=[])
    row = episode(evaluation=False)
    row["layout"] = {"curriculum": {"direct_path_blocked": True}}
    row["reset_exposure"] = 500.
    pilot(root, episodes=[row])
    curves = report.reward_curves(report.load_report(root), grid=[.5])
    curve = next(c for c in curves if c["mode"] == "training")
    assert curve["difficulty"] == "hard" and curve["label"] == "late"


def test_endpoint_counts_and_raw_coded_returns(root):
    condition = report.CONDITIONS[1]
    initial = panel(condition=condition)
    final = panel("final", condition=condition)
    final["rows"][0]["success"] = True
    pilot(root, condition=condition, evaluations=[initial, final])
    rows = report.endpoint_rows(report.load_report(root))
    assert rows[0]["Success / episodes"] == "1 / 1"
    assert rows[0]["A change vs shared initial"] == 1.
    assert rows[0]["Raw earned / coded earned"] == "2 / 0.2"


def test_closeout_all_zero_b_summarizes_a_retention_and_shared_initial_once(root):
    closeout_condition(root, report.CONDITIONS[0], final_a=(2, 1), final_b=(0, 0))
    closeout_condition(root, report.CONDITIONS[1], initial_a=1, final_a=(1, 0), final_b=(0, 0))
    snapshot = report.load_report(root)
    text = report.conclusion(snapshot)
    assert "4/4 sealed pilot trajectories; 2/2 complete paired seeds" in text
    assert "Complete development panels." in text
    assert "Linear (final seeds 197101, 197102): A retention 2/2 -> 2/2, 1/2; B acquisition 0/2 -> 0/2, 0/2" in text
    assert "Signed square (final seeds 197101, 197102): A retention 1/2 -> 1/2, 0/2" in text
    assert "Hard-task acquisition remains unestablished for both conditions." in text
    assert "**Scope:** linear versus signed-square reward recipes" in text
    assert "not a test of FD-KL statistical efficiency" in text
    assert "**No demonstrated advantage of either reward recipe**" in text
    assert "not evidence of equivalence" in text
    assert "does not rule out an FD benefit" in text
    assert "Neither is ready for promotion" not in text
    assert "Linear: A declined in 1/2 seeds, B successes in 0/2 evaluated seeds" in text
    assert "shared initial panels count once" in text
    assert text.count("A retention 2/2 ->") == 1
    assert "A retention 4/4" not in text and "A retention 6/6" not in text
    assert len([p for p in report.evaluation_panels(snapshot) if p["panel"]["checkpoint"] == "initial"]) == 4
    assert "selection remains unresolved" not in text
    assert len(text.split()) < 220


def test_closeout_mixed_b_counts_and_provisional_followup_respect_a_loss(root):
    closeout_condition(root, report.CONDITIONS[0], final_a=(2, 2), final_b=(1, 0))
    closeout_condition(root, report.CONDITIONS[1], final_a=(1, 1), final_b=(2, 1))
    text = report.conclusion(report.load_report(root))
    assert "B acquisition 0/2 -> 1/2, 0/2" in text
    assert "B acquisition 0/2 -> 2/2, 1/2" in text
    assert "Linear: A declined in 0/2 seeds, B successes in 1/2 evaluated seeds" in text
    assert "Signed square: A declined in 2/2 seeds, B successes in 2/2 evaluated seeds" in text
    assert "Provisional follow-up candidates with B successes and no observed A decline: Linear." in text
    assert "B successes are development evidence only, not proof of learning beyond initialization" in text
    assert "Hard-task acquisition remains unestablished for both conditions" not in text
    assert "FD was off" in text and "no FD benefit, significance, causal proof" in text
    assert "confidence interval" not in text.lower()


def test_closeout_does_not_promote_b_successes_that_accompany_a_loss(root):
    for condition in report.CONDITIONS:
        closeout_condition(root, condition, final_a=(1, 1), final_b=(1, 1))
    text = report.conclusion(report.load_report(root))
    assert "B successes accompany A losses" in text
    assert "Provisional follow-up candidates" not in text


def test_closeout_pending_does_not_turn_missing_outcomes_into_zero(root):
    text = report.conclusion(report.load_report(root))
    assert "0/4 sealed pilot trajectories; 0/2 complete paired seeds" in text
    assert "Partial development results" in text
    assert "Linear: A retention pending -> pending; B acquisition pending -> pending" in text
    assert "Signed square: A retention pending -> pending; B acquisition pending -> pending" in text
    assert "Defer recipe selection" in text
    assert "Hard-task acquisition remains unestablished for both conditions" not in text
    assert "No demonstrated advantage" not in text
    assert "B acquisition 0/" not in text


def test_closeout_one_complete_pair_remains_partial(root):
    for condition in report.CONDITIONS:
        closeout_condition(root, condition, final_a=(2, None), final_b=(0, None))
    text = report.conclusion(report.load_report(root))
    assert "2/4 sealed pilot trajectories; 1/2 complete paired seeds" in text
    assert "Partial development results" in text
    assert "A retention 2/2 -> 2/2, pending; B acquisition 0/2 -> 0/2, pending" in text
    assert "not evidence of inability to learn" in text
    assert "Provisional follow-up candidates" not in text


def test_closeout_four_trajectories_with_missing_b_panels_are_not_complete_evidence(root):
    for condition in report.CONDITIONS:
        closeout_condition(root, condition, final_a=(2, 2), final_b=(None, None))
    text = report.conclusion(report.load_report(root))
    assert "4/4 sealed pilot trajectories; 2/2 complete paired seeds" in text
    assert "Partial development results" in text and "Complete development panels." not in text
    assert "B acquisition 0/2 -> pending, pending" in text
    assert "Hard-task acquisition remains unestablished for both conditions" not in text


def test_closeout_b_successes_already_present_at_initialization_are_not_learning_proof(root):
    for condition in report.CONDITIONS:
        closeout_condition(root, condition, initial_b=1, final_a=(2, 2), final_b=(1, 1))
    text = report.conclusion(report.load_report(root))
    assert text.count("B acquisition 1/2 -> 1/2, 1/2") == 2
    assert "not proof of learning beyond initialization" in text


def test_closeout_unmatched_a_panels_are_not_reported_as_zero_retention_loss(root):
    for condition in report.CONDITIONS:
        initial = [panel("initial", domain, condition=condition) for domain in ("easy", "hard")]
        for seed in (197101, 197102):
            evaluations = copy.deepcopy(initial)
            evaluations.append(panel("final", "easy", condition=condition,
                                     rows=[episode(seed=99, condition=condition)]))
            evaluations.append(panel("final", "hard", condition=condition))
            pilot(root, condition=condition, seed=seed, evaluations=evaluations)
    text = report.conclusion(report.load_report(root))
    assert "Partial development results" in text
    assert "Defer recipe selection until matched paired A/B panels are complete" in text
    assert "A declined in 0/2" not in text
    assert "Provisional follow-up candidates" not in text


def test_notebook_preserves_history_and_refresh_deduplicates(root, tmp_path):
    base, previous = history(root)
    before = base.read_bytes()
    output = tmp_path / "results.ipynb"
    pilot(root)
    report.build_notebook(root, output)
    book = nbformat.read(output, as_version=4)
    assert base.read_bytes() == before
    assert [c for c in book.cells if not c.metadata.get(report._TAG)] == previous.cells
    assert book.cells[1].id == "plan19-phase3-fix3-progress"
    text = rendered(book)
    assert text.index("### Phase 3 Fix 2") < text.index("### Phase 3 Fix 3")
    assert "FD was off" in book.cells[-1].source
    assert len(book.cells) == len(previous.cells) + 4
    assert sum("image/png" in o.get("data", {}) for c in book.cells for o in c.get("outputs", [])) == 2
    refreshed = tmp_path / "refresh.ipynb"
    report.build_notebook(root, refreshed, base_notebook=output)
    refreshed_book = nbformat.read(refreshed, as_version=4)
    assert len(refreshed_book.cells) == len(book.cells)
    assert len({c.id for c in refreshed_book.cells}) == len(refreshed_book.cells)
    assert [c for c in refreshed_book.cells if not c.metadata.get(report._TAG)] == previous.cells


def test_fix3_refresh_preserves_later_fix4_draft_and_chronology(root, tmp_path):
    _, previous = history(root)
    output = tmp_path / "results.ipynb"
    pilot(root)
    report.build_notebook(root, output)
    book = nbformat.read(output, as_version=4)
    draft = nbformat.v4.new_markdown_cell(
        "### Phase 3 Fix 4\n\nDraft for review; not executed.",
        metadata={report._DRAFT_TAG: True}, id="plan19-phase3-fix4-draft")
    book.cells.append(draft)
    nbformat.write(book, output)
    report.build_notebook(root, output)
    refreshed = nbformat.read(output, as_version=4)
    assert refreshed.cells[-1] == draft
    assert refreshed.cells[-2].source.startswith("## Phase 3 Fix 3 Conclusion")
    assert [c for c in refreshed.cells if not c.metadata.get(report._TAG)
            and not c.metadata.get(report._DRAFT_TAG)] == previous.cells
    assert sum(c.metadata.get(report._DRAFT_TAG, False) for c in refreshed.cells) == 1
    alternate = tmp_path / "alternate.ipynb"
    report.build_notebook(root, alternate, base_notebook=output)
    assert nbformat.read(alternate, as_version=4).cells[-1] == draft


def test_reward_plot_legends_are_concise_and_acquisition_axis_explicit(root, monkeypatch):
    pilot(root)
    snapshot = report.load_report(root)
    monkeypatch.setattr(report, "_figure", lambda figure, description: figure)
    reward = report._reward_figure(snapshot)
    labels = [label for axis in reward.axes for label in axis.get_legend_handles_labels()[1]]
    assert labels
    assert all("e=" not in label and "r=" not in label for label in labels)
    capability = report._capability_figure(snapshot)
    assert all(axis.get_xlabel() == "Captured transitions" for axis in capability.axes)
    reward.clear()
    capability.clear()


def training_metrics(values, *, origin=100):
    return [{"event_id": index, "acquired_count": origin + acquired, "data": value - .5,
             "total": 999., "path": "online" if index % 2 else "sleep",
             "inner_steps": [{"inner_step": step, "data": value - .25 * (step - 1)}
                             for step in (1, 2, 3)]}
            for index, (acquired, value) in enumerate(values)]


def test_training_nll_uses_first_step_data_equal_replica_weights_and_local_clock(root):
    unit(root, "dataset", acquired_count=100)
    pilot(root, seed=197101, fit_metrics=training_metrics([(1, 2.), (32, 4.), (33, 8.)]))
    pilot(root, seed=197102, fit_metrics=training_metrics([(16, 9.), (64, 10.), (65, 6.)]))
    pilot(root, condition=report.CONDITIONS[1], fit_metrics=training_metrics([(1, -2.)]))
    curves = report.training_nll_curves(report.load_report(root))
    assert curves[0]["condition"] == report.CONDITIONS[0]
    points = curves[0]["points"]
    assert [p["observations"] for p in points] == [16, 48, 80]
    assert [p["mean"] for p in points] == [6., 9., 6.]
    assert [p["events"] for p in points] == [3, 2, 1]
    assert [p["replicas"] for p in points] == [2, 2, 1]
    assert points[0]["window_start"] == 0 and points[0]["window_end"] == 32
    assert curves[1]["points"][0]["mean"] == -2.


def test_training_nll_no_missing_or_readiness_imputation(root):
    pilot(root)
    unit(root, "readiness", condition=report.CONDITIONS[0],
         fit_metrics=training_metrics([(1, 1000.)]))
    snapshot = report.load_report(root)
    assert report.training_nll_curves(snapshot) == []
    assert report._training_nll_figure(snapshot) is None
    text = " ".join(o.get("text", "") + o.get("data", {}).get("text/plain", "")
                    for o in report.result_outputs(snapshot))
    assert "Online training NLL is unavailable" in text
    assert "Held-out online NLL was not recorded" in text


@pytest.mark.parametrize("defect", ["origin", "clock", "nonfinite", "missing_steps", "duplicate_event"])
def test_training_nll_rejects_ambiguous_clock_or_invalid_metrics(root, defect):
    if defect != "origin":
        unit(root, "dataset", acquired_count=100)
    metrics = training_metrics([(1, 2.), (2, 3.)])
    if defect == "clock":
        metrics[0]["acquired_count"] = 100
    elif defect == "nonfinite":
        metrics[0]["inner_steps"][0]["data"] = float("nan")
    elif defect == "missing_steps":
        del metrics[0]["inner_steps"]
    elif defect == "duplicate_event":
        metrics[1]["event_id"] = metrics[0]["event_id"]
    pilot(root, fit_metrics=metrics)
    with pytest.raises(shared.ArtifactError, match="nonfinite JSON|training NLL"):
        report.training_nll_curves(report.load_report(root))


def test_training_nll_plot_and_visible_notebook_limitations(root, tmp_path, monkeypatch):
    unit(root, "dataset", acquired_count=100)
    pilot(root, fit_metrics=training_metrics([(1, 4.), (64, 3.)]))
    snapshot = report.load_report(root)
    original = report._figure
    monkeypatch.setattr(report, "_figure", lambda figure, description: figure)
    figure = report._training_nll_figure(snapshot)
    axis = figure.axes[0]
    assert axis.get_xlabel() == "Newly captured training intervals"
    assert axis.get_ylabel() == "Data-only joint NLL (lower is better)"
    assert list(axis.lines[0].get_ydata()) == [4., 3.]
    figure.clear()
    monkeypatch.setattr(report, "_figure", original)
    history(root)
    output = tmp_path / "nll.ipynb"
    report.build_notebook(root, output)
    book = nbformat.read(output, as_version=4)
    text = rendered(book)
    assert "**Held-out online NLL was not recorded.**" in text
    assert "different likelihood targets" in text.replace("\n", " ")
    assert "regularizers are not" in text and "not necessarily better navigation" in text
    assert "32-captured-interval windows" in text
    assert sum("image/png" in o.get("data", {}) for c in book.cells for o in c.get("outputs", [])) == 3


def test_capability_curve_averages_checkpoints_not_distinct_replica_counts(root, monkeypatch):
    first = [panel(cp) for cp in ("initial", "midpoint", "final")]
    second = copy.deepcopy(first)
    first[1]["acquired_count"], second[1]["acquired_count"] = 320, 330
    first[2]["acquired_count"], second[2]["acquired_count"] = 620, 630
    second[1]["rows"][0]["success"] = True
    first[2]["rows"][0]["success"] = True
    pilot(root, seed=197101, evaluations=first)
    pilot(root, seed=197102, evaluations=second)
    monkeypatch.setattr(report, "_figure", lambda figure, description: figure)
    figure = report._capability_figure(report.load_report(root))
    line = figure.axes[0].lines[0]
    assert list(line.get_xdata()) == [0, 325, 625]
    assert list(line.get_ydata()) == [0, .5, .5]
    figure.clear()


def test_endpoint_collisions_use_recorded_terminal_reason(root):
    row = episode()
    row.pop("collision")
    row["end_reason"] = "collision"
    pilot(root, evaluations=[panel(), panel("final", rows=[row])])
    endpoint = report.endpoint_rows(report.load_report(root))[0]
    assert endpoint["Collisions"] == 1
    assert endpoint["Timeouts"] == 0


def test_missing_history_wrong_output_or_seal_fails_without_writes(root, tmp_path):
    output = tmp_path / "results.ipynb"
    with pytest.raises(shared.ArtifactError, match="missing/nonregular"):
        report.build_notebook(root, output)
    assert not output.exists()
    base, _ = history(root)
    with pytest.raises(shared.ArtifactError, match="overwrite immutable history"):
        report.build_notebook(root, base)
    completed = unit(root, "audit")
    with pytest.raises(shared.ArtifactError, match="completed unit"):
        report.build_notebook(root, completed / "report.ipynb")
    (completed / "summary.json").write_text("{}")
    with pytest.raises(shared.ArtifactError, match="hash/size mismatch"):
        report.load_report(root)


def test_bounds_and_no_checkpoint_reads(root, monkeypatch):
    path = unit(root, "audit")
    (path / "checkpoint.pt").write_bytes(b"Not a model; must never be opened")
    original = shared._read
    def read(path, limit):
        assert Path(path).suffix != ".pt"
        return original(path, limit)
    monkeypatch.setattr(shared, "_read", read)
    assert len(report.load_report(root)["complete_units"]) == 1
    monkeypatch.setattr(report, "MAX_UNITS", 0)
    with pytest.raises(shared.ArtifactError, match="unit limit"):
        report.load_report(root)


def test_only_units_and_orchestration_scanned(root, monkeypatch):
    unit(root, "references")
    for name in ("score_cache", "reference_bundle", "historical"):
        path = root / name / "inputs" / "not-a-unit"
        path.mkdir(parents=True)
        (path / "COMPLETED").write_text("not a RunStore seal")
    monkeypatch.setattr(report, "MAX_SCAN_DIRS", 3)
    assert len(report.load_report(root)["complete_units"]) == 1


def test_inherited_assemble_video_metadata_survives_filter_and_shared_clip_dedup(root, tmp_path):
    # Scalar metadata returned by assemble_video plus inherited render_video fields.
    clips = [{"path": str(root / "units" / "clips" / "initial.mp4"), "seed": 11,
              "selection": "preselected", "frames": 2, "fps": 5., "topdown_reader_only": True,
              "checkpoint": "initial", "difficulty": "easy", "view": "ego+topdown"},
             {"path": str(root / "units" / "clips" / "failure.mp4"), "seed": 12,
              "selection": "failure_gallery", "frames": 2, "fps": 5., "topdown_reader_only": True}]
    unit(root, "initial_evaluation", condition=report.CONDITIONS[0], evaluations=[panel()], videos=clips)
    pilot(root, videos=clips)
    output = report._links(report.load_report(root), tmp_path)
    markup = output.data["text/html"]
    assert markup.count("initial.mp4") == 1 and markup.count("failure.mp4") == 1
    assert "preselected" in markup and "failure_gallery" in markup and "ego+topdown" in markup


def test_scalar_byte_and_scan_bounds_fail_clearly(root, monkeypatch):
    unit(root, "audit")
    monkeypatch.setattr(report, "MAX_SCALAR_BYTES", 1)
    with pytest.raises(shared.ArtifactError, match="scalar size limit"):
        report.load_report(root)
    monkeypatch.setattr(report, "MAX_SCAN_DIRS", 1)
    with pytest.raises(shared.ArtifactError, match="scan limit"):
        report.load_report(root)


@pytest.mark.parametrize("working_directory", ["checkout", "notebook"])
def test_notebook_cells_execute_with_gpu_model_imports_blocked(root, tmp_path, working_directory):
    base, book = history(root)
    book.cells[1].source = (
        "from pathlib import Path\nimport sys\n"
        "for parent in (Path.cwd(), *Path.cwd().parents):\n"
        "    if (parent / 'numerical_experiment/picar').is_dir():\n"
        "        sys.path.insert(0, str(parent))\n"
        "        break\n")
    nbformat.write(book, base)
    before = base.read_bytes()
    pilot(root)
    incomplete_fit(root)
    output = tmp_path / "artifact-report.ipynb"
    report.build_notebook(root, output)
    built = nbformat.read(output, as_version=4)
    assert built.cells[1].id == "plan19-phase3-fix3-progress"
    assert built.cells[2] == book.cells[1]
    assert built.cells[1].source.index("sys.path.insert") < built.cells[1].source.index("from numerical_experiment")
    script = r'''
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path
assert 'PYTHONPATH' not in os.environ
project = Path(sys.argv[2]).resolve()
sys.path = [p for p in sys.path if Path(p or '.').resolve() != project]
assert importlib.util.find_spec('numerical_experiment') is None
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'pybullet'} or fullname in {
            'numerical_experiment.picar_reward_repair.config',
            'numerical_experiment.picar_reward_repair.scoring',
            'numerical_experiment.picar_reward_repair.student',
            'numerical_experiment.picar_reward_repair.initialization',
            'numerical_experiment.picar.runtime'}:
            raise AssertionError('Forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, Guard())
import nbformat
from IPython import get_ipython
book = nbformat.read(sys.argv[1], as_version=4)
namespace = {}
for cell in book.cells:
    if cell.cell_type == 'code':
        exec(cell.source, namespace)
        assert 'reward_report' in namespace
assert 'torch' not in sys.modules
'''
    project = Path(__file__).resolve().parents[2]
    cwd = project if working_directory == "checkout" else project / "numerical_experiment" / "picar"
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, "-c", script, str(output), str(project)],
                            cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert base.read_bytes() == before
