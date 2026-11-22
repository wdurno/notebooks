"""Bounded synthetic scalar envelopes, with no training or canonical writes."""

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
from numerical_experiment.picar_curriculum_repair import report


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def seal(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def reseal(path):
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["final_files"] = {name: seal(path / name) for name in ("summary.json", "metrics.csv")}
    write_json(path / "manifest.json", manifest)
    write_json(path / "COMPLETED", {"manifest_sha256": seal(path / "manifest.json")["sha256"]})


@pytest.fixture
def root(tmp_path):
    path = tmp_path / report.CAMPAIGN
    path.mkdir()
    return path


def unit(root, operation, *, name=None, settings=None, **fields):
    path = root / "units" / (name or operation)
    path.mkdir(parents=True)
    config = {"schema_version": 1, "version": report.VERSION, "operation": operation, "settings": settings or {}}
    stored = {"config": config}
    write_json(path / "config.json", stored)
    write_json(path / "summary.json", {"schema_version": 1, "version": report.VERSION,
        "kind": "curriculum_" + operation, "settings": settings or {}, "active_seconds": 5., **fields})
    with (path / "metrics.csv").open("w") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_COLUMNS).writeheader()
    write_json(path / "manifest.json", {"artifact_schema_version": 1, "metric_schema_version": 1,
        "storage_schema_version": 1, "status": "completed", "completed_at": "2026-11-19T12:00:00Z",
        "config_hash": hashlib.sha256(json.dumps(stored, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    reseal(path)
    return path


def predictive(*, value=5., index=None):
    row = {"return_nll": 2., "action_nll": value - 2., "joint_nll": value, "weight": 1.,
           "label": 1, "action": [0.] * 7, "baseline": 1., "bin_values": [-1., 1.],
           "temperature": 2., "action_variance": 3., "return_logits": [0., 0.], "action_mean": [0.] * 7}
    if index is not None:
        row["index"] = index
    return row


def episode(*, seed=11, success=False, evaluation=True):
    row = {"complete": True, "truncated": False, "success": success, "reward_scoring": True,
        "seed": seed, "episode_id": f"ep{seed}", "evaluation": evaluation,
        "intervals": [{"elapsed_s": .5}, {"elapsed_s": .5}],
        "raw_stage_rewards": [1., 2.], "raw_returns": [2.5, 2.]}
    if evaluation:
        row.update(learning=False, explore=False, predictive_scoring=True, predictive_missing_intervals=0,
                   predictive_scores=[predictive(index=0), predictive(index=1)])
    return row


def panel(target=64, environment="A", *, actual=None, rows=None):
    rows = [episode(), episode(seed=12)] if rows is None else rows
    return {"target": target, "checkpoint": f"n{target}", "difficulty": environment,
        "acquired_count": target if actual is None else actual, "rows": rows,
        "initial_asset_id": "linear-initializer", "config": {"fixed_environment": environment},
        "seeds": [r["seed"] for r in rows]}


def references(target=64, value=5.):
    return {"target": target, "acquired_count": target, "rows": [
        {**predictive(value=value), "environment": domain, "episode_id": domain + ":ep0", "record_id": domain + ":0"}
        for domain in report.ENVIRONMENTS]}


def fit_metrics(points):
    return [{"event_id": index, "acquired_count": 100 + n, "data": 1000.,
             "inner_steps": [{"inner_step": 1, "data": value}, {"inner_step": 2, "data": -100.},
                             {"inner_step": 3, "data": -200.}]}
            for index, (n, value) in enumerate(points)]


def trajectory(root, *, method="replay_ewc", seed=198101, operation="trajectory", **fields):
    defaults = {"acquired_count": 64, "arrival_offset": 100, "episodes": [episode(evaluation=False)],
        "evaluations": [panel(n, domain) for n in (0, 64) for domain in report.ENVIRONMENTS],
        "reference_scores": [references(0), references(64, value=4.)],
        "fit_metrics": fit_metrics([(1, 2.), (32, 4.), (33, 8.)]), "optimizer_exposures": 18,
        "first_step_ewc_events": 2, "fd_strength": .1 if method == "stm_fd" else 0., "ridge": .1,
        "fix4": {"reference_id": "common-bank"}}
    return unit(root, operation, name=f"{operation}-{method}-{seed}", settings={"method": method, "seed": seed},
                **{**defaults, **fields})


def test_training_clock_can_be_verified_from_complete_captured_arrival_ledger(root):
    episodes = [episode(evaluation=False)]
    episodes[0]["intervals"] = [{"elapsed_s": .5, "arrival": 752+i} for i in range(64)]
    episodes[0]["raw_stage_rewards"] = [0.]*64
    episodes[0]["raw_returns"] = [0.]*64
    trajectory(root, arrival_offset=None, episodes=episodes)
    # Event clocks are source-acquisition clocks, including the stationary
    # initializer's captured images, not just its fitted training histories.
    summary = json.loads(next((root/"units").glob("trajectory-*/summary.json")).read_text())
    summary["fit_metrics"] = fit_metrics([(1, 2.), (32, 4.), (33, 8.)])
    for event in summary["fit_metrics"]:
        event["acquired_count"] += 652
    path = next((root/"units").glob("trajectory-*"))
    write_json(path/"summary.json", summary)
    reseal(path)
    curve = report.training_nll_curves(report.load_report(root))[0]
    assert [p["mean"] for p in curve["points"]] == [3., 8.]
    summary["episodes"][0]["intervals"][10]["arrival"] += 1
    write_json(path/"summary.json", summary)
    reseal(path)
    with pytest.raises(shared.ArtifactError, match="contiguous"):
        report.load_report(root)


def test_budget_average_uses_observed_overshoot_without_extrapolation():
    assert report._budget_average([(0, 4.), (10, 2.), (20, 0.)], 15) == pytest.approx(2.5)
    with pytest.raises(shared.ArtifactError, match="coverage"):
        report._budget_average([(0, 4.), (10, 2.)], 15)


def test_paired_effects_use_replica_pairs_and_common_observation_budgets(root):
    for seed in (198101, 198102):
        trajectory(root, seed=seed)
        trajectory(root, method="stm_fd", seed=seed,
            reference_scores=[references(0), references(64, value=3.)], active_seconds=10.)
    rows = report.paired_effect_rows(report.load_report(root))
    risk = next(r for r in rows if r["Paired Metric"] == "B_final: budget-averaged reference NLL")
    assert risk["Pairs"] == 2
    assert risk["Mean FD - Control / Ratio"] == pytest.approx(-.5)
    cost = next(r for r in rows if "multiplier" in r["Paired Metric"])
    assert cost["Mean FD - Control / Ratio"] == pytest.approx(2.)
    success = next(r for r in rows if "final success" in r["Paired Metric"])
    assert "discrete" in success["Exploratory 95% CI"]


def progress(root, **fields):
    path = root / "orchestration"
    path.mkdir(exist_ok=True)
    write_json(path / "progress.json", {"schema_version": 1, "version": report.VERSION,
        "status": "not yet run", "max_hours": 28, "charged_seconds": 60., **fields})
    return path


def committed_progress(path):
    transaction = path / "transactions" / "commit-1"
    transaction.mkdir(parents=True)
    (transaction / "progress.json").write_bytes((path / "progress.json").read_bytes())
    write_json(transaction / "manifest.json", {"files": {"progress.json": seal(transaction / "progress.json")}})
    write_json(path / "CURRENT.json", {"storage_schema_version": 1, "commit_id": "commit-1",
                                       "manifest": seal(transaction / "manifest.json")})


def history(root):
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Plan 19 Historical Results"),
        nbformat.v4.new_code_cell("print('Fix 3 historical counters')", id="plan19-phase3-fix3-progress",
            metadata={"plan19_phase3_fix3": True},
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="Old counters\n")]),
        nbformat.v4.new_markdown_cell("### Fix 3 conclusion\n\nNo demonstrated advantage."),
        nbformat.v4.new_markdown_cell("### Fix 4 draft\n\nUnexecuted draft.", metadata={report._DRAFT_TAG: True})])
    path = root / "history.ipynb"
    nbformat.write(book, path)
    return path, book


def rendered(book):
    return "\n".join(cell.source + "\n" + "\n".join(
        output.get("text", "") + output.get("data", {}).get("text/plain", "") for output in cell.get("outputs", []))
        for cell in book.cells)


def file_state(root):
    return {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}


def test_missing_root_is_explicit_and_readonly(tmp_path):
    missing = tmp_path / report.CAMPAIGN
    snapshot = report.load_report(missing)
    assert not missing.exists()
    assert report.progress_rows(snapshot)[0]["Status"] == "not yet run"
    assert report.outcome_rows(snapshot) == []
    assert "not collected" in report.conclusion(snapshot)
    with pytest.raises(shared.ArtifactError, match="missing/nonregular"):
        report.build_notebook(missing, tmp_path / "output.ipynb")
    assert not missing.exists()


def test_pre_first_unit_and_auxiliary_only_reporting_work(root):
    progress(root, status="provenance")
    assert report.load_report(root)["complete_units"] == []
    for operation in ("provenance", "software", "cpu_smoke", "audit", "readiness", "fd_probe", "references", "selection", "forecast", "integrity"):
        unit(root, operation)
    snapshot = report.load_report(root)
    assert report.training_nll_curves(snapshot) == report.reference_nll_curves(snapshot) == report.task_curves(snapshot) == []
    assert report.progress_rows(snapshot)[0]["Sealed / Planned Trajectories"] == "0 / 4"
    assert report.cost_rows(snapshot) == []
    assert report.setup_cost_rows(snapshot)[0]["Sealed Units"] == 10


def test_completed_and_incomplete_readers_never_open_checkpoint_or_media(root, monkeypatch):
    trajectory(root)
    incomplete = root / "units" / "trajectory-incomplete"
    incomplete.mkdir()
    write_json(incomplete / "config.json", {"settings": {"method": "stm_fd", "seed": 198101}, "operation": "trajectory"})
    (incomplete / "checkpoint.pt").write_bytes(b"must never be unpickled")
    (incomplete / "summary.json").write_text("not valid JSON")
    seen = []
    original = shared._read
    def read(path, limit):
        seen.append(Path(path))
        assert Path(path).suffix not in {".pt", ".png", ".mp4"}
        assert Path(path) != incomplete / "summary.json"
        return original(path, limit)
    monkeypatch.setattr(shared, "_read", read)
    before = file_state(root)
    snapshot = report.load_report(root)
    assert len(snapshot["complete_units"]) == 1
    assert snapshot["incomplete_units"] == [str(incomplete)]
    assert file_state(root) == before and seen


@pytest.mark.parametrize("field,value,match", [
    ("schema_version", 2, "summary schema"), ("version", "wrong", "wrong version"),
    ("kind", "curriculum_wrong", "unknown curriculum operation"), ("active_seconds", -1., "active_seconds"),
])
def test_wrong_summary_identity_fails_without_repair(root, field, value, match):
    path = unit(root, "software", **{field: value})
    before = file_state(root)
    with pytest.raises(shared.ArtifactError, match=match):
        report.load_report(root)
    assert file_state(root) == before


@pytest.mark.parametrize("field,value,match", [
    ("learning", True, "nonlearning"), ("explore", True, "nonlearning"),
    ("evaluation", False, "nonlearning"), ("complete", False, "complete"),
    ("truncated", True, "nontruncated"), ("reward_scoring", False, "reward scoring"),
    ("predictive_scoring", False, "predictive scoring"), ("predictive_missing_intervals", 1, "coverage"),
    ("raw_returns", [], "raw_returns"),
])
def test_evaluation_flags_and_coverage_are_strict(root, field, value, match):
    row = episode()
    row[field] = value
    trajectory(root, evaluations=[panel(rows=[row])])
    with pytest.raises(shared.ArtifactError, match=match):
        report.load_report(root)


def test_joint_nll_decomposition_checked_for_both_predictive_views(root):
    row = episode()
    row["predictive_scores"][0]["joint_nll"] = 999.
    path = trajectory(root, evaluations=[panel(rows=[row])])
    with pytest.raises(shared.ArtifactError, match="joint NLL"):
        report.load_report(root)
    summary = json.loads((path / "summary.json").read_text())
    summary["evaluations"] = []
    summary["reference_scores"][0]["rows"][0]["action_nll"] = 999.
    write_json(path / "summary.json", summary)
    reseal(path)
    with pytest.raises(shared.ArtifactError, match="joint NLL"):
        report.load_report(root)


def test_reference_conditioning_and_bank_identity_cannot_drift(root):
    checkpoints = [references(0), references(64)]
    checkpoints[1]["rows"][0]["baseline"] = 99.
    trajectory(root, reference_scores=checkpoints)
    with pytest.raises(shared.ArtifactError, match="conditioning differ"):
        report.load_report(root)


def test_equal_replica_32_observation_training_means_and_diagnostic_exclusion(root):
    trajectory(root)
    trajectory(root, seed=198102, fit_metrics=fit_metrics([(16, 9.), (64, 10.)]))
    trajectory(root, operation="timing", seed=198201, fit_metrics=fit_metrics([(1, 9999.)]))
    trajectory(root, operation="calibration_run", seed=198301, fit_metrics=fit_metrics([(1, 9999.)]))
    snapshot = report.load_report(root)
    points = report.training_nll_curves(snapshot)[0]["points"]
    assert [p["observations"] for p in points] == [16., 48.]
    assert [p["mean"] for p in points] == [6., 9.]
    assert [p["events"] for p in points] == [3, 2]
    assert all(p["replicas"] == 2 for p in points)
    assert report.progress_rows(snapshot)[0]["Sealed / Planned Trajectories"] == "2 / 4"


def test_training_clock_is_never_inferred(root):
    trajectory(root, arrival_offset=None)
    with pytest.raises(shared.ArtifactError, match="arrival_offset"):
        report.load_report(root)


def test_shared_initial_once_and_latest_task_equal_replica_weights(root):
    a = [panel(0, domain) for domain in report.ENVIRONMENTS]
    for p in a:
        p["shared_panel_id"] = "shared-initial-" + p["difficulty"]
    trajectory(root, evaluations=a + [panel(rows=[episode(success=True), episode(seed=12)])])
    trajectory(root, seed=198102, evaluations=a + [panel(rows=[episode()])])
    unit(root, "initial_evaluation", evaluations=a)
    snapshot = report.load_report(root)
    panels = report.evaluation_panels(snapshot)
    assert len([p for p in panels if p["panel"]["target"] == 0]) == 3
    points = report.task_curves(snapshot)
    latest = next(p for p in points if p["environment"] == "A" and p["target"] == 64)
    assert latest["success"] == .25 and latest["replicas"] == 2
    assert latest["episodes"] == 3 and latest["raw_reward"] == 3.
    assert all(p["replicas"] == 1 for p in points if p["target"] == 0)


def test_conflicting_shared_initial_panels_fail(root):
    shared_panels = [panel(0)]
    shared_panels[0]["shared_panel_id"] = "declared-shared"
    trajectory(root, evaluations=shared_panels)
    altered = panel(0)
    altered["shared_panel_id"] = "declared-shared"
    altered["rows"][0]["success"] = True
    unit(root, "initial_evaluation", evaluations=[altered])
    with pytest.raises(shared.ArtifactError, match="conflicting shared initial"):
        report.load_report(root)


def test_independently_collected_initial_panels_are_not_shared_by_asset_alone(root):
    trajectory(root, evaluations=[panel(0)])
    changed = panel(0)
    changed["rows"][0]["raw_stage_rewards"][0] += .5
    trajectory(root, seed=198102, evaluations=[changed])
    snapshot = report.load_report(root)
    assert len(report.evaluation_panels(snapshot)) == 2
    point = report.task_curves(snapshot)[0]
    assert point["replicas"] == 2 and point["episodes"] == 4


def test_reference_curves_stay_separate_from_policy_outputs(root):
    trajectory(root)
    trajectory(root, seed=198102, reference_scores=[references(0), references(64, value=10.)])
    snapshot = report.load_report(root)
    curves = report.reference_nll_curves(snapshot)
    p = next(p for p in curves if p["target"] == 64 and p["environment"] == "B_final")
    assert p["joint_nll"] == 7. and p["return_nll"] == 2. and p["action_nll"] == 5.
    assert p["replicas"] == 2 and p["records"] == p["episodes"] == 2


def test_scalar_progress_previews_are_provisional_and_committed_state_wins(root, monkeypatch):
    path = progress(root, status="stopped", charged_seconds=7200., forecast={"remaining_hours": [2., 4.]},
        active_unit_progress={"stage": "collect", "condition": "stm_fd", "acquired_count": 800,
            "curriculum": {"name": "B initial hold", "within": 32},
            "preview_training": [{"n": 798, "data": 7.}, {"n": 800, "data": 9.}],
            "checkpoint_preview": [{"target": 768, "environment": "A", "acquired_count": 780,
                                     "episodes": 8, "success": .5, "raw_return": 2.}],
            "reference_preview": [{"target": 768, "environment": "B_final", "joint_nll": 4., "records": 32}]})
    committed_progress(path)
    progress(root, status="stale mirror", charged_seconds=0.)
    snapshot = report.load_report(root)
    row = report.progress_rows(snapshot)[0]
    assert row["Status"] == "stopped" and row["Charged Active Hours"] == 2.
    assert row["Remaining Budget Hours"] == 26. and row["ETA Hours (Recorded)"] == [2., 4.]
    assert row["Stage"] == "B initial hold" and row["Within Stage"] == 32
    assert report.training_nll_curves(snapshot) == report.task_curves(snapshot) == report.reference_nll_curves(snapshot) == []
    figures = []
    def figure(value, description):
        figures.append((value, description))
        return nbformat.v4.new_output("stream", name="stdout", text=description + "\n")
    monkeypatch.setattr(report, "_figure", figure)
    outputs = report.result_outputs(snapshot, notebook_dir=root)
    assert any("Provisional" in o.get("text", "") for o in outputs)
    assert figures[0][0].axes[0].lines[0].get_xdata().tolist() == [780]
    assert any("provisional" in line.get_label() for f, _ in figures for axis in f.axes for line in axis.lines)


def test_partial_pairs_no_zero_imputation_or_samples_saved_claim(root):
    trajectory(root)
    trajectory(root, method="stm_fd")
    trajectory(root, seed=198102)
    snapshot = report.load_report(root)
    assert report.progress_rows(snapshot)[0]["Started / Completed Pairs"] == "2 / 1"
    text = report.conclusion(snapshot)
    assert "Partial" in text and "3/4" in text and "1/2" in text
    assert "No success was observed" in text and "flat curves are not extrapolated" in text
    assert "No samples-saved estimate" in text


def test_history_preserved_draft_removed_refresh_deduplicated_and_source_readonly(root, tmp_path):
    base, original = history(root)
    progress(root, status="software")
    before = file_state(root)
    output = tmp_path / "results.ipynb"
    report.build_notebook(root, output)
    book = nbformat.read(output, as_version=4)
    nbformat.validate(book)
    for old in original.cells:
        if old.metadata.get(report._DRAFT_TAG):
            assert not any(c.id == old.id for c in book.cells)
        else:
            assert next(c for c in book.cells if c.id == old.id) == old
    assert book.cells[1].id == "plan19-phase3-fix4-progress"
    assert "Fix 4 Conclusion" in book.cells[-1].source
    text = rendered(book)
    assert "Historical progress snapshot" in text and "transition.png" in text
    assert text.index("Fix 3 conclusion") < text.index("Staged A-To-B")
    assert "32-observation" in text and "not imitation accuracy" in text
    assert file_state(root) == before
    report.build_notebook(root, output)
    refreshed = nbformat.read(output, as_version=4)
    assert [c.id for c in refreshed.cells] == [c.id for c in book.cells]
    with pytest.raises(shared.ArtifactError, match="overwrite history"):
        report.build_notebook(root, base)


def test_scalar_and_scan_bounds_are_enforced(root, monkeypatch):
    trajectory(root)
    monkeypatch.setattr(report, "MAX_SCALAR_BYTES", 1)
    with pytest.raises(shared.ArtifactError, match="scalar byte bound"):
        report.load_report(root)
    monkeypatch.setattr(report, "MAX_SCALAR_BYTES", 128 * 1024 * 1024)
    monkeypatch.setattr(report, "MAX_SCAN_DIRS", 0)
    with pytest.raises(shared.ArtifactError, match="scan bound"):
        report.load_report(root)


def test_preparation_has_no_empty_figures_and_narrow_shared_costs(root, monkeypatch):
    for name in ("provenance", "software", "cpu_smoke"):
        unit(root, name)
    snapshot = report.load_report(root)
    def fail(*args):
        raise AssertionError("preparation must not render empty plots")
    monkeypatch.setattr(report, "_figure", fail)
    outputs = report.result_outputs(snapshot, notebook_dir=root)
    assert report.cost_rows(snapshot) == []
    assert len(report.setup_cost_rows(snapshot)) == 1
    assert len(report.setup_cost_rows(snapshot)[0]) == 3
    assert not any("image/png" in o.get("data", {}) for o in outputs)


def test_scalar_links_collapsed_and_only_actual_bounded_videos_surfaced(root, tmp_path):
    path = trajectory(root)
    videos = []
    for domain in report.ENVIRONMENTS:
        clip = path / (domain + ".mp4")
        clip.write_bytes(b"do not decode")
        videos.append({"path": clip.name, "difficulty": domain, "checkpoint": "n64"})
    external = tmp_path / "outside.mp4"
    external.write_bytes(b"outside campaign")
    videos.extend([{"path": str(external)}, {"path": "missing.mp4"}])
    summary = json.loads((path / "summary.json").read_text())
    summary["videos"] = videos
    write_json(path / "summary.json", summary)
    reseal(path)
    snapshot = report.load_report(root)
    outputs = report.result_outputs(snapshot, notebook_dir=root)
    text = "\n".join(o.get("data", {}).get("text/html", "") for o in outputs)
    assert "<details><summary>Scalar artifacts" in text
    assert "Preselected recorded clips" in text
    assert all(domain + ".mp4" in text for domain in report.ENVIRONMENTS)
    assert "outside.mp4" not in text and "missing.mp4" not in text
    assert len(report._video_links(snapshot, root, limit=2)) == 2


def test_rendered_progress_is_vertical_without_raw_forecast_or_full_path(root):
    active = root / "units" / "trajectory-abc123"
    progress(root, active_unit=str(active), forecast={"main_hours": [14., 19.], "basis": "long unrendered forecast text"})
    snapshot = report.load_report(root)
    output = report._progress_table(snapshot)
    text = output.data["text/plain"]
    assert "Item" in text and "Value" in text
    assert "trajectory-abc123" in text and str(active) not in text
    assert "long unrendered forecast text" not in text
    assert "Whole-Main Forecast Hours" in text
    progress(root)
    assert "not yet profiled" in report._progress_table(report.load_report(root)).data["text/plain"]


def test_notebook_own_cells_execute_without_torch_models_or_runtime_imports(root, tmp_path):
    history(root)
    trajectory(root)
    output = tmp_path / "results.ipynb"
    report.build_notebook(root, output)
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname.endswith(('.runtime', '.models', '.student', '.predictive')):
            raise AssertionError('report imported forbidden runtime module: ' + fullname)
sys.meta_path.insert(0, Block())
import nbformat
book = nbformat.read(sys.argv[1], as_version=4)
scope = {}
for cell in book.cells:
    if cell.cell_type == 'code' and cell.metadata.get('plan19_phase3_fix4'):
        exec(cell.source, scope)
print('REPORT_ONLY_OK')
"""
    project = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", script, str(output)], cwd=project,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")}, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REPORT_ONLY_OK" in result.stdout
