"""Phase 6 scalar inference and artifact-only publication, without learners/GPU."""

import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import nbformat
import numpy as np
from PIL import Image
import pytest

from mnist_experiment.kalman_pi.registration import analysis, report


@pytest.fixture(autouse=True)
def plotting_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def selection(n=2, stationary=0, matched=False):
    conditions = {"fixed": "fixed", "covariance": "covariance", "ema": "ema_h16", "kalman": "kalman_eta0.0001"}
    arms = [{"arm": a, "condition": c, "ridge": .1} for a, c in conditions.items()]
    if matched:
        arms += [{"arm": a + "_matched", "condition": conditions[a], "ridge": .01} for a in ("kalman", "fixed")]
    return {"common_ridge": .1, "ema_condition": conditions["ema"], "kalman_condition": conditions["kalman"],
            "calibrated_ridge": .01 if matched else None, "arms": arms, "main_replicas": n,
            "stationary_replicas": stationary, "bootstrap_seed": 833, "hold_steps": 4, "ramp_steps": 2}


def trajectory(arm="fixed", replica=0, *, schedule="main", completed=True, matched=False, stage="fresh"):
    spec = next(a for a in selection(matched=matched)["arms"] if a["arm"] == arm)
    effect = {"fixed": 0, "covariance": .1, "ema": .2, "kalman": .3, "fixed_matched": .05, "kalman_matched": .4}[arm]
    rows = []
    for i, (step, arrivals) in enumerate(zip((0, 4, 6, 10, 12, 16), (0, 8, 20, 44, 52, 80))):
        change = 0 if i == 0 else effect * (replica + 1)
        rows.append({"step": step, "arrivals": arrivals, "segment": "stationary" if schedule == "stationary" else ("initial", "rise", "high_hold", "return", "final_hold", "final_hold")[i],
                     "old_risk": 4 + i - change, "shifted_risk": 16 - i - change,
                     "old_accuracy": .8 - i / 100 + change / 100, "shifted_accuracy": .2 + i / 20 + change / 100,
                     "old_nll": 10., "shifted_nll": 12., "applied_pi": .025,
                     "raw_pi": .025, "motion_energy": float(i), "uncertainty_energy": float(i * 2),
                     "noise_scale": 1., "innovation_ratio": 1.})
    if not completed:
        rows = rows[:2]
    return {"unit_id": f"{stage}-{schedule}-{replica}-{arm}", "stage": stage, "cohort": stage, "replica": replica,
            "schedule": schedule, **spec, "c": .025, "sign": (-1, 1)[replica % 2],
            "schedule_factor": (1., 1.25)[(replica // 2) % 2], "hold_steps": 4, "ramp_steps": 2, "final_step": 16,
            "initial_unit": f"initial-{replica}", "stream_hash": f"stream-{schedule}-{replica}",
            "metrics": rows, "completed": completed, "summary": {"learning_seconds": 9., "online_wall_seconds": 12.,
                "costs": {"solver_seconds": 3., "curvature_seconds": 4., "controller_seconds": 1., "render_seconds": 1.}}}


def block(replica=0, **kwargs):
    arms = (*analysis.ARMS, *analysis.MATCHED_ARMS) if kwargs.get("matched") else analysis.ARMS
    return [trajectory(arm, replica, **kwargs) for arm in arms]


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (24, 24), "#428579").save(output, format="PNG")
    return output.getvalue()


def seal_history(directory, book, snapshot):
    data = report._json_bytes(snapshot)
    book.metadata.plan17_zero_ridge_report.snapshot_sha256 = report._hash(data)
    (directory / "report_snapshot.json").write_bytes(data)
    nbformat.write(book, directory / "report.ipynb")
    manifest = {"schema": 1, "kind": "plan17-zero-ridge-report", "study_id": "historical", "smoke": False,
                "report_dir": str(directory), "snapshot_sha256": report._hash(data),
                "files_sha256": {p.name: report._hash(p.read_bytes()) for p in directory.iterdir() if p.name not in ("manifest.json", "COMPLETED")}}
    encoded = report._json_bytes(manifest)
    (directory / "manifest.json").write_bytes(encoded)
    (directory / "COMPLETED").write_text(report._hash(encoded) + "\n")


@pytest.fixture
def historical(tmp_path):
    directory = tmp_path / "historical/report/generation-history"
    directory.mkdir(parents=True)
    cells = [nbformat.v4.new_markdown_cell("# All historical evidence", id="plan17-title")]
    for prefix in ("plan17", "plan17-retention", "plan17-zero-ridge"):
        cells.append(nbformat.v4.new_markdown_cell("Exact inherited conclusion and intervals", id=prefix + "-conclusion", metadata={"custom": [1, 2]}))
    png = png_bytes()
    for name in ("10_zero_ridge_effects", "11_zero_ridge_accuracy"):
        (directory / (name + ".png")).write_bytes(png)
        cells.append(nbformat.v4.new_code_cell("historical code; never executed", id="plan17-zero-ridge-" + name,
                     execution_count=17, metadata={"preserve": True}, outputs=[nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png).decode()})]))
    book = nbformat.v4.new_notebook(cells=cells, metadata={"custom": {"all": [1, 2, 3]},
        "plan17_report": {"schema": 1}, "plan17_retention_report": {"schema": 1},
        "plan17_zero_ridge_report": {"schema": 1, "smoke": False, "status": "Completed supplement", "study_id": "historical", "generation": directory.name}})
    (directory / "analysis.json").write_text("{}")
    seal_history(directory, book, {"schema": 1, "kind": "plan17-zero-ridge-snapshot", "study_id": "historical", "status": "completed"})
    return directory


def snapshot(history=None, rows=None, selected=None, *, smoke=False):
    return {"schema": 1, "kind": report.KIND, "study_id": "registration-v1", "status": "awaiting GPU repair/reboot",
            "phase": "6A: acceptance only", "config": {"smoke": smoke, "hold_steps": 4, "ramp_steps": 2, "bootstrap_draws": 32},
            "selection": selected, "trajectories": rows or [], "readiness": {}, "diagnostics": {},
            "progress": {"phase6_active_hours": .125, "historical_active_hours": 14.318, "total_family_active_hours": 14.443,
                         "active_hours": .125, "phase6_allocation_hours": 20., "heartbeat": "2026-11-04T10:00:00Z",
                         "active_unit": "CPU-acceptance", "last_accepted_checkpoint": "accepted-3", "forecast_seconds": 3600,
                         "storage": {"used_bytes": 1e9, "max_bytes": 500e9}},
            "historical_report_dir": str(history) if history else None,
            "historical_notebook_sha256": report._hash((history / "report.ipynb").read_bytes()) if history else None}


def write_snapshot(root, value):
    root.mkdir(parents=True, exist_ok=True)
    (root / "report_snapshot.json").write_bytes(report._json_bytes(value))


@pytest.fixture
def fast_renderers(monkeypatch):
    calls = []
    def render(snapshot, result, path):
        calls.append(path.name)
        path.write_bytes(png_bytes())
    for name in ("_context", "_development", "_fresh", "_effects"):
        monkeypatch.setattr(report, name, render)
    return calls


def test_landmark_not_coordinate_mse_and_inclusive_accuracy():
    prediction = [[1, 0, 0, 2, 3, 4], [0, 0, 0, 0, 0, 0]]
    result = analysis.landmark_metrics(prediction, np.zeros((2, 6)))
    assert result["risk"] == 5  # (1+4+25) / (3 landmarks * 2 observations)
    assert result["rms"] == pytest.approx(np.sqrt(5))
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["nll"] == pytest.approx(30 / 16 + 3 * np.log(8 * np.pi))
    assert analysis.landmark_metrics(prediction, np.zeros((2, 6)), [1, 0])["risk"] == 10


@pytest.mark.parametrize("predicted,target,weights", [([], [], None), ([[0] * 5], [[0] * 5], None),
    ([[float("nan")] * 6], [[0] * 6], None), ([[0] * 6], [[0] * 6], [-1])])
def test_landmark_input_bounds(predicted, target, weights):
    with pytest.raises(ValueError):
        analysis.landmark_metrics(predicted, target, weights)


def test_actual_arrival_areas_windows_rms_and_zero_denominator_fallback():
    rows = trajectory()["metrics"]
    result = analysis.summarize_path(rows, 4, 2)
    expected_old = sum((b["arrivals"] - a["arrivals"]) * (a["old_risk"] + b["old_risk"]) / 2 for a, b in zip(rows, rows[1:])) / 80
    assert result["Aold"] == expected_old
    assert result["Aold"] != np.mean([r["old_risk"] for r in rows])
    assert result["Ashifted"] == pytest.approx((12 * 14.5 + 24 * 13.5) / 36)
    assert result["Ashifted_whole"] != result["Ashifted"]
    assert result["old_rms"] == np.sqrt(expected_old)
    assert result["oldgap_frominitial"] == np.sqrt(expected_old) - 2
    assert result["normalized_score"] == .5 * (result["Aold"] / 4 + result["Ashifted"] / 16)
    rows[0]["old_risk"] = 0
    fallback = analysis.summarize_path(rows, 4, 2)
    assert fallback["normalized_score"] == .5 * (fallback["Aold"] + fallback["Ashifted"])
    assert fallback["score_basis"] == "absolute_risk_zero_initial"


def test_nested_readiness_probes_keep_unsuccessful_alternate_and_average_replicas():
    probes = [{"angle_max": angle, "frozen_old": {"risk": 2. + r}, "frozen_shifted": {"risk": shifted + r},
               "fitted_shifted": {"risk": fitted + r}, "readiness_flags": angle == 35}
              for angle, shifted, fitted in ((25, 4., 4.1), (35, 8., 5.)) for r in (0, 1)]
    result = report._probe_summary(probes)
    assert set(result) == {25., 35.}
    assert result[25.] == {"old": 2.5, "shifted": 4.5, "fitted": 4.6}


def test_stationary_virtual_window_and_per_trajectory_resolved_horizon():
    rows = block(schedule="stationary")
    result = analysis.summarize(rows, selection(stationary=1), bootstrap_draws=0, config={"hold_steps": 999, "ramp_steps": 999})
    assert result["groups"][0]["schedule"] == "stationary"
    assert result["groups"][0]["n_blocks"] == 1
    assert result["path_summaries"][rows[0]["unit_id"]]["acquisition_steps"] == [4, 10]


@pytest.mark.parametrize("mutate", [lambda t: t["metrics"].pop(2), lambda t: t["metrics"].pop(),
    lambda t: t["metrics"][1].update(arrivals=0), lambda t: t["metrics"][1].update(old_risk=-1),
    lambda t: t["metrics"][1].update(old_accuracy=1.01), lambda t: t["metrics"][1].update(applied_pi=.3),
    lambda t: t["metrics"][1].update(uncertainty_energy=-1), lambda t: t.update(c=.05),
    lambda t: t.update(final_step=17), lambda t: t.update(condition="other"), lambda t: t.update(sign=0)])
def test_trajectory_bounds_and_exact_boundary_validation(mutate):
    t = trajectory()
    mutate(t)
    with pytest.raises(ValueError):
        analysis.summarize([t], selection())


@pytest.mark.parametrize("name,value", [("old_risk", 5), ("old_accuracy", .7), ("old_nll", 15)])
def test_shared_initialization_is_required(name, value):
    rows = block()
    rows[-1]["metrics"][0][name] = value
    with pytest.raises(ValueError, match="initialization"):
        analysis.summarize(rows, selection())


def test_block_identity_grid_and_asset_pairing():
    for name, value in (("sign", 1), ("schedule_factor", 1.25), ("stream_hash", "other")):
        rows = block()
        rows[-1][name] = value
        with pytest.raises(ValueError, match="paired block"):
            analysis.summarize(rows, selection())
    rows = block()
    rows[-1]["metrics"][1]["arrivals"] += 1
    with pytest.raises(ValueError, match="grids"):
        analysis.summarize(rows, selection())
    with pytest.raises(ValueError, match="duplicate"):
        analysis.summarize([trajectory(), trajectory()], selection())


def test_partial_or_missing_fourth_arm_excludes_the_entire_block():
    rows = block()
    rows[-1] = trajectory("kalman", completed=False)
    result = analysis.summarize(rows, selection())
    assert result["counts"] == {"complete": 3, "incomplete": 1, "paired_blocks": 0}
    assert len(result["awaiting_full_block_units"]) == 3
    assert all(c["n_pairs"] == 0 for c in result["groups"][0]["comparisons"])
    assert analysis.summarize(rows[:-1], selection())["counts"]["paired_blocks"] == 0


def test_companions_both_required_and_all_six_complete_before_contrasts():
    s = selection(matched=True)
    with pytest.raises(ValueError, match="companions"):
        analysis.summarize([], {**s, "arms": s["arms"][:-1]})
    rows = block(matched=True)
    assert analysis.summarize(rows[:-1], s)["counts"]["paired_blocks"] == 0
    result = analysis.summarize(rows, s)
    contrasts = {(c["arm"], c["control"]) for c in result["groups"][0]["comparisons"]}
    assert {("kalman_matched", "fixed"), ("kalman_matched", "fixed_matched"), ("fixed_matched", "fixed")} <= contrasts


def test_paired_gains_within_replica_roots_cost_ratios_and_missing_costs():
    rows = block() + block(1)
    rows[0]["summary"]["learning_seconds"] = 18
    result = analysis.summarize(rows, selection(), bootstrap_draws=20)
    group = result["groups"][0]
    kalman = next(c for c in group["comparisons"] if c["arm"] == "kalman")
    paths = result["path_summaries"]
    a, b = ([paths[f"fresh-main-{r}-{arm}"] for r in (0, 1)] for arm in ("kalman", "fixed"))
    expected = np.mean([x["old_rms"] - y["old_rms"] for x, y in zip(a, b)])
    assert kalman["effects"]["old_rms_gap"]["mean"] == expected
    assert kalman["effects"]["old_risk_gain"]["mean"] > 0
    assert kalman["effects"]["learning_seconds_ratio"]["mean"] == .75
    assert group["bootstrap_mode"] == "ordinary_paired_replica"
    rows[-1]["summary"].pop("learning_seconds")
    rows[-1]["summary"]["costs"]["render_seconds"] = 4
    assert analysis._costs(rows[-1])["learning_seconds"] == 12
    rows[-1]["summary"]["costs"].pop("render_seconds")
    missing = analysis.summarize(rows, selection())["groups"][0]["comparisons"][-1]["effects"]["learning_seconds_ratio"]
    assert missing["mean"] is None and missing["ci95"] is None


def test_stratified_shared_bootstrap_and_mixed_schedules_are_separate():
    rows = [t for r in range(8) for t in block(r)] + [t for r in range(4) for t in block(r, schedule="stationary")]
    s = selection(n=8, stationary=4)
    result = analysis.summarize(rows, s, bootstrap_draws=50, seed=12)
    changing, stationary = result["groups"]
    assert changing["bootstrap_mode"] == "stratified_sign_factor"
    assert stationary["bootstrap_mode"] == "ordinary_paired_replica"
    assert [g["n_blocks"] for g in result["groups"]] == [8, 4]
    again = analysis.summarize(list(reversed(rows)), s, bootstrap_draws=50, seed=12)
    assert result["groups"] == again["groups"]
    assert result["bootstrap"]["shared_across_endpoints_and_contrasts"] is True
    # Each resampled sign/length stratum retains its original two members.
    blocks = [{t["arm"]: t for t in block(r)} for r in range(8)]
    indices, mode = analysis._indices(blocks, 50, 12)
    strata = [(r % 2, (r // 2) % 2) for r in range(8)]
    assert all(all(sum(strata[i] == s for i in draw) == 2 for s in set(strata)) for draw in indices)


def test_bootstrap_and_input_bounds_and_pending_selection():
    assert analysis.summarize([], None)["counts"]["paired_blocks"] == 0
    assert analysis.summarize([], bootstrap_draws=999999)["bootstrap"]["draws"] == analysis.MAX_BOOTSTRAP_DRAWS
    assert analysis.summarize([], bootstrap_draws=-1)["bootstrap"]["draws"] == 0
    for kwargs in ({"bootstrap_draws": True}, {"seed": -1}, {"seed": True}):
        with pytest.raises(ValueError):
            analysis.summarize([], **kwargs)
    with pytest.raises(ValueError, match="frozen selection"):
        analysis.summarize(block(), None)
    with pytest.raises(ValueError, match="at most"):
        analysis.summarize([{}] * (analysis.MAX_TRAJECTORIES + 1))


def test_pending_publication_preserves_every_historical_cell_output_metadata(tmp_path, historical, fast_renderers):
    original = nbformat.read(historical / "report.ipynb", as_version=4)
    before = {p: p.read_bytes() for p in historical.iterdir()}
    root, target = tmp_path / "registration", tmp_path / "canonical.ipynb"
    target.write_bytes((historical / "report.ipynb").read_bytes())
    write_snapshot(root, snapshot(historical))
    manifest = report.build_report(root, target)
    book = nbformat.read(target, as_version=4)
    assert book.cells[1].id == report.PREFIX + "progress"
    assert [c for c in book.cells if not c.id.startswith(report.PREFIX)] == original.cells
    assert {k: v for k, v in book.metadata.items() if k != report.MARKER} == original.metadata
    appendix = book.cells[len(original.cells) + 1:]
    assert len([c for c in appendix if c.cell_type == "code"]) == 4
    assert sum("| ---" in c.source for c in appendix if c.cell_type == "markdown") == 2
    assert fast_renderers == [n + ".png" for n, _ in report.FIGURES]
    text = "\n".join(c.source for c in book.cells if c.cell_type == "markdown")
    for fragment in ("awaiting GPU", "0.125 / 20.000", "14.318 / 14.443", "no efficacy conclusion", "no fresh matching"):
        assert fragment in text
    assert all(p.read_bytes() == data for p, data in before.items())
    assert report._verify_generation(Path(manifest["report_dir"])) == manifest
    assert str(Path(analysis.__file__).resolve()) in manifest["sources_sha256"]
    assert json.loads((root / "report/latest.json").read_text()) == manifest
    old_files = {p: p.read_bytes() for p in Path(manifest["report_dir"]).iterdir()}
    assert report.build_report(root, target) == manifest
    assert len(fast_renderers) == 4
    assert all(p.read_bytes() == data for p, data in old_files.items())


def test_default_production_notebook_path_and_historical_write_protection(tmp_path, historical, fast_renderers, monkeypatch):
    root = tmp_path / "registration"
    write_snapshot(root, snapshot(historical))
    fake = tmp_path / "project/mnist_experiment/kalman_pi/registration/report.py"
    sources = report._sources()
    monkeypatch.setattr(report, "_sources", lambda: sources)
    monkeypatch.setattr(report, "__file__", str(fake))
    manifest = report.build_report(root)
    assert manifest["notebook"] == str(fake.parents[2] / "plan17_results.ipynb")
    with pytest.raises(ValueError, match="historical root"):
        report.build_report(root, historical.parent.parent / "must-not-write.ipynb")


def test_pending_to_frozen_selection_then_completed_block_monotonic(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    value.update(selection=selection(n=1), trajectories=block(), status="completed")
    write_snapshot(root, value)
    final = report.build_report(root)
    assert final["status"] == "Completed Phase 6"
    assert final["report_dir"] != first["report_dir"]
    assert (Path(first["report_dir"]) / "COMPLETED").exists()
    value["trajectories"][0]["summary"]["learning_seconds"] += 1
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="rewrites completed"):
        report.build_report(root)


@pytest.mark.parametrize("change", ["omit", "rollback", "empty", "timing_replay", "metric_replay", "summary_replay", "complete_replay"])
def test_provisional_trajectories_can_be_omitted_or_replayed(tmp_path, fast_renderers, change):
    root = tmp_path / "smoke"
    partial = trajectory("kalman", completed=False)
    for i, row in enumerate(partial["metrics"]):
        row.update(wall_seconds=float(i + 1), solver_seconds=.1)
    value = snapshot(rows=[partial], selected=selection(), smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    directory = Path(first["report_dir"])
    old_files = {p.name: p.read_bytes() for p in directory.iterdir()}
    if change == "omit":
        value["trajectories"] = []
    elif change == "rollback":
        partial["metrics"] = partial["metrics"][:1]
    elif change == "empty":
        partial["metrics"] = []
    elif change == "timing_replay":
        partial["metrics"][-1].update(wall_seconds=3., solver_seconds=.2)
    elif change == "metric_replay":
        partial["metrics"][-1]["old_risk"] += .01
    elif change == "summary_replay":
        partial["summary"]["learning_seconds"] += 1
    else:
        completed = trajectory("kalman")
        completed["metrics"][:2] = deepcopy(partial["metrics"])
        completed["metrics"][1].update(wall_seconds=3., solver_seconds=.2)
        value["trajectories"] = [completed]
    write_snapshot(root, value)
    final = report.build_report(root)
    assert final["report_dir"] != first["report_dir"]
    sealed = json.loads((Path(final["report_dir"]) / "report_snapshot.json").read_text())
    assert sealed["trajectories"] == value["trajectories"]
    result = json.loads((Path(final["report_dir"]) / "analysis.json").read_text())
    assert result["counts"]["paired_blocks"] == 0
    assert result["complete_units"] == ([partial["unit_id"]] if change == "complete_replay" else [])
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == old_files
    assert report._verify_generation(directory) == first
    assert json.loads((root / "report/latest.json").read_text()) == final


@pytest.mark.parametrize("field,value", [
    ("stage", "development"), ("cohort", "other"), ("replica", 1), ("schedule", "stationary"),
    ("arm", "ema"), ("condition", "kalman_eta0.001"), ("ridge", .03), ("c", .03),
    ("sign", 1), ("schedule_factor", 1.25), ("hold_steps", 5), ("ramp_steps", 3),
    ("stream_hash", "other-stream"), ("initial_unit", "other-initial"), ("asset_unit", "other-assets"),
    ("final_step", 17), ("shadow_conditions", []),
])
def test_provisional_trajectory_identity_cannot_change(tmp_path, fast_renderers, field, value):
    root = tmp_path / "smoke"
    current = snapshot(rows=[trajectory("kalman", completed=False)], selected=selection(), smoke=True)
    current["trajectories"][0].update(asset_unit="assets-0", shadow_conditions=["ema_h16"])
    write_snapshot(root, current)
    first = report.build_report(root)
    current["trajectories"][0][field] = value
    with pytest.raises(ValueError, match="identity|rewrites|changed"):
        report._previous(root, current, first["historical_sha256"])
    assert json.loads((root / "report/latest.json").read_text()) == first


@pytest.mark.parametrize("change", ["remove_unit", "reopen_unit", "prefix", "timing", "summary"])
def test_completed_trajectory_evidence_cannot_change(tmp_path, fast_renderers, change):
    root = tmp_path / "smoke"
    value = snapshot(rows=[trajectory("kalman")], selected=selection(), smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    directory = Path(first["report_dir"])
    old_files = {p.name: p.read_bytes() for p in directory.iterdir()}
    if change == "remove_unit":
        value["trajectories"] = []
    elif change == "reopen_unit":
        value["trajectories"][0]["completed"] = False
    elif change == "prefix":
        value["trajectories"][0]["metrics"][1]["old_risk"] += .01
    elif change == "timing":
        value["trajectories"][0]["metrics"][1]["wall_seconds"] = 3.
    else:
        value["trajectories"][0]["summary"]["learning_seconds"] += 1
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="regresses|rewrites completed"):
        report.build_report(root)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == old_files
    assert report._verify_generation(directory) == first
    assert json.loads((root / "report/latest.json").read_text()) == first


@pytest.mark.parametrize("change", ["study_id", "config", "selection", "selection_missing",
    "phase6_active_hours", "active_hours", "total_family_active_hours", "history_time",
    "phase6_active_hours_missing", "active_hours_missing", "total_family_active_hours_missing"])
def test_monotonic_snapshot_guards(tmp_path, fast_renderers, change):
    root = tmp_path / "smoke"
    value = snapshot(rows=[trajectory("kalman", completed=False, stage="development")], selected=selection(), smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    if change == "study_id":
        value["study_id"] = "other-study"
    elif change == "config":
        value["config"]["bootstrap_draws"] += 1
    elif change == "selection":
        value["selection"]["bootstrap_seed"] += 1
    elif change == "selection_missing":
        value["selection"] = None
    elif change == "history_time":
        value["progress"]["historical_active_hours"] = 0
    elif change.endswith("_missing"):
        value["progress"].pop(change.removesuffix("_missing"))
    else:
        value["progress"][change] = 0
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="changed|changes|regresses|rewrites|mismatch"):
        report.build_report(root)
    assert json.loads((root / "report/latest.json").read_text()) == first


@pytest.mark.parametrize("rms", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_group_frozen_reference_is_mean_initial_transformed_metric(rms, reverse):
    paths = [trajectory(replica=i) for i in (0, 4, 2, 6)]
    for path, initial in zip(paths, (1., 9., 4., 16.), strict=True):
        path["metrics"][0]["old_risk"] = initial
    if reverse:
        paths.reverse()
    ax = Mock()
    report._curves(ax, paths, "old_risk", rms=rms)
    references = [call for call in ax.plot.call_args_list if call.kwargs.get("linestyle") == ":"]
    curves = [call for call in ax.plot.call_args_list if call.kwargs.get("linestyle") != ":"]
    assert len(references) == len(curves) == 2
    for reference, curve, expected in zip(references, curves, (2., 3.) if rms else (5., 10.), strict=True):
        np.testing.assert_allclose(reference.args[0], [0, 80])
        np.testing.assert_allclose(reference.args[1], [expected, expected])
        assert curve.args[1][0] == pytest.approx(expected)


def test_action_curves_skip_unavailable_initial_action_not_the_whole_path():
    path = trajectory("kalman")
    path["metrics"][0].pop("applied_pi")
    ax = Mock()
    report._curves(ax, [path], "applied_pi")
    assert ax.plot.call_count == 1
    np.testing.assert_array_equal(ax.plot.call_args.args[0], [8, 20, 44, 52, 80])
    np.testing.assert_allclose(ax.plot.call_args.args[1], .025)


def test_pixel_interval_precision_does_not_round_a_nonzero_bound_to_zero():
    effect = {"mean": .0088066454, "ci95": [.0032998677, .0143134231], "interval_reliable": True}
    assert report._effect(effect, signed=True) == "+0.0088 [0.0033, 0.0143]"


def test_descriptive_curves_pool_directions_but_never_distinct_grids():
    paths = [trajectory(replica=i) for i in (0, 1)]
    paths[1]["metrics"][0]["old_risk"] = 9.
    ax = Mock()
    report._curves(ax, paths, "old_risk")
    assert ax.plot.call_count == 2
    assert ax.plot.call_args_list[0].args[1][0] == pytest.approx(6.5)
    paths[1]["metrics"][1]["arrivals"] += 1
    ax = Mock()
    report._curves(ax, paths, "old_risk")
    assert ax.plot.call_count == 4


@pytest.mark.parametrize("selected,expected", [
    (None, "pending development calibration"), ({}, "pending development calibration"),
    ({"common_ridge": .1}, "development matching gap not recorded in this snapshot"),
    ({"common_ridge": .1, "calibrated_ridge": .03, "retention_candidates": [
        {"ridge": .1, "old_gap": 0., "acquisition_gap": 0.}]},
     "development matching gap not recorded in this snapshot"),
])
def test_retention_status_pending_or_missing_selected_candidate(selected, expected):
    assert report._retention_status(selected) == expected


@pytest.mark.parametrize("old_gap,acquisition_gap,expected", [
    (0., .2, "within development band"), (-.10, -.2, "within development band"),
    (.10, .2, "within development band"), (.1001, -.2, "outside development band / unmatched"),
    (-.1001, 0., "outside development band / unmatched"),
    (-.1001, -.2, "both development errors lower; matching not forced"),
])
def test_retention_status_common_ridge_band_boundaries(old_gap, acquisition_gap, expected):
    selected = {"common_ridge": .1, "calibrated_ridge": None, "retention_candidates": [
        {"ridge": .03, "old_gap": .5, "acquisition_gap": .2},
        {"ridge": .1, "old_gap": old_gap, "acquisition_gap": acquisition_gap}]}
    before = deepcopy(selected)
    assert report._retention_status(selected) == (
        f"{expected}; selected r=0.1, old RMS gap {old_gap:+.4f} pixels; no fresh matching")
    assert selected == before


@pytest.mark.parametrize("ridge,old_gap,expected", [
    (.03, .05, "within development band"), (.03, -.2, "outside development band / unmatched"),
    (0., .05, "within development band"),
])
def test_retention_status_uses_calibrated_ridge_including_zero(ridge, old_gap, expected):
    selected = {"common_ridge": .1, "calibrated_ridge": ridge, "retention_candidates": [
        {"ridge": .1, "old_gap": -.3, "acquisition_gap": -.2},
        {"ridge": ridge, "old_gap": old_gap, "acquisition_gap": -.2}]}
    assert report._retention_status(selected) == (
        f"{expected}; selected r={ridge:g}, old RMS gap {old_gap:+.4f} pixels; no fresh matching")


@pytest.mark.parametrize("corrupt", ["COMPLETED", "analysis.json", "report.ipynb", "12_registration_context.png"])
def test_corrupt_sealed_artifacts_rejected_without_repair(tmp_path, fast_renderers, corrupt):
    root = tmp_path / "smoke"
    write_snapshot(root, snapshot(smoke=True))
    manifest = report.build_report(root)
    path = Path(manifest["report_dir"]) / corrupt
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        report.build_report(root)
    assert path.read_bytes() == b"corrupt"


def test_cached_analysis_and_incompatible_selection_fail_before_render(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    value["analysis"] = {"schema": 1, "made_up": True}
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="differs from validated"):
        report.build_report(root)
    assert not fast_renderers
    value.pop("analysis")
    value["selection"] = ["invalid"]
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="selection must"):
        report.build_report(root)


@pytest.mark.parametrize("corrupt", ["output", "incomplete", "missing_hash"])
def test_semantically_corrupt_but_resealed_history_rejected(tmp_path, historical, fast_renderers, corrupt):
    book = nbformat.read(historical / "report.ipynb", as_version=4)
    sealed = json.loads((historical / "report_snapshot.json").read_text())
    if corrupt == "output":
        book.cells[-1].outputs[0].data["image/png"] = "bm90LXBvcnRhYmxlLWltYWdl"
    elif corrupt == "incomplete":
        sealed["status"] = "running"
    else:
        (historical / "11_zero_ridge_accuracy.png").unlink()
    seal_history(historical, book, sealed)
    root = tmp_path / "registration"
    write_snapshot(root, snapshot(historical))
    with pytest.raises(ValueError, match="inherited image|completed non-smoke|missing required"):
        report.build_report(root, tmp_path / "canonical.ipynb")


def test_atomic_publication_failure_restores_target_and_latest(tmp_path, fast_renderers, monkeypatch):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    manifest = report.build_report(root)
    target, latest = Path(manifest["notebook"]), root / "report/latest.json"
    before = {p: p.read_bytes() for p in (target, latest)}
    value["progress"]["heartbeat"] = "new snapshot"
    write_snapshot(root, value)
    from mnist_experiment.kalman_pi.retention import report as inherited_helpers
    original = inherited_helpers._atomic
    def fail(path, data):
        if path == target:
            raise OSError("simulated notebook replacement failure")
        original(path, data)
    monkeypatch.setattr(inherited_helpers, "_atomic", fail)
    with pytest.raises(OSError, match="replacement failure"):
        report.build_report(root)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_latest_and_source_changes_during_render_do_not_publish(tmp_path, fast_renderers, monkeypatch):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    target = Path(first["notebook"])
    before = target.read_bytes()
    value["progress"]["heartbeat"] = "second"
    write_snapshot(root, value)
    sources = report._sources()
    def change_sources(snapshot, result, path):
        path.write_bytes(png_bytes())
        monkeypatch.setattr(report, "_sources", lambda: {**sources, "extra": "changed"})
    monkeypatch.setattr(report, "_context", change_sources)
    with pytest.raises(ValueError, match="changed during rendering"):
        report.build_report(root)
    assert target.read_bytes() == before
    assert json.loads((root / "report/latest.json").read_text()) == first


def test_wrong_history_hash_unsealed_history_and_smoke_target_are_rejected(tmp_path, historical, fast_renderers):
    root, target = tmp_path / "registration", tmp_path / "canonical.ipynb"
    value = snapshot(historical)
    value["historical_notebook_sha256"] = "wrong"
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="pinned sealed"):
        report.build_report(root, target)
    value = snapshot(historical)
    write_snapshot(root, value)
    (historical / "COMPLETED").write_text("bad")
    with pytest.raises(ValueError, match="completion seal"):
        report.build_report(root, target)
    write_snapshot(root, snapshot(smoke=True))
    with pytest.raises(ValueError, match="inside its own root"):
        report.build_report(root, target)


def test_manual_newer_notebook_and_snapshot_render_races_are_not_overwritten(tmp_path, historical, fast_renderers, monkeypatch):
    root, target = tmp_path / "registration", tmp_path / "canonical.ipynb"
    value = snapshot(historical)
    write_snapshot(root, value)
    manifest = report.build_report(root, target)
    original = target.read_bytes()
    book = nbformat.read(target, as_version=4)
    book.cells[-1].source += " user note"
    nbformat.write(book, target)
    with pytest.raises(ValueError, match="newer, modified"):
        report.build_report(root, target)
    target.write_bytes(original)
    value["progress"]["heartbeat"] = "later"
    write_snapshot(root, value)
    def race(snapshot, result, path):
        path.write_bytes(png_bytes())
        changed = deepcopy(value)
        changed["progress"]["heartbeat"] = "newer"
        write_snapshot(root, changed)
    monkeypatch.setattr(report, "_context", race)
    with pytest.raises(ValueError, match="changed during rendering"):
        report.build_report(root, target)
    assert target.read_bytes() == original
    assert json.loads((root / "report/latest.json").read_text()) == manifest
    assert not list((root / "report").glob(".building-*"))


@pytest.mark.parametrize("bad", ["schema", "nonfinite", "oversized"])
def test_corrupt_or_incompatible_snapshot_fails_before_publication(tmp_path, monkeypatch, bad):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    if bad == "schema":
        value["schema"] = 2
    write_snapshot(root, value)
    if bad == "nonfinite":
        (root / "report_snapshot.json").write_text('{"value":NaN}')
    if bad == "oversized":
        monkeypatch.setattr(report.report_helpers, "MAX_JSON_BYTES", 10)
    with pytest.raises(ValueError):
        report.build_report(root)
    assert not (root / "report").exists()


def test_real_cpu_renderers_and_embedded_pngs(tmp_path):
    root = tmp_path / "smoke"
    rows = block() + block(stage="development") + block(schedule="stationary")
    value = snapshot(rows=rows, selected=selection(n=1, stationary=1), smoke=True)
    value["readiness"] = {"examples": [{"reference": np.zeros((28, 28)).tolist(), "query": np.ones((28, 28)).tolist(),
                          "reference_landmarks": [[-6, -4], [6, -4], [0, 8]], "query_landmarks": [[-5, -3], [7, -3], [1, 9]]}],
                          "schedule": [{"step": 0, "h": 0}, {"step": 6, "h": 1}, {"step": 16, "h": 0}],
                          "angle_law": {"angles": [-50, 0, 50], "proposed": {"h=0": [0, 1, 0]}, "accepted": {"h=0": [0, 1, 0]}},
                          "probes": [{"angle": 25, "old_risk": 2, "shifted_risk": 3, "capability_shifted_risk": 2.5}]}
    write_snapshot(root, value)
    manifest = report.build_report(root)
    book = nbformat.read(manifest["notebook"], as_version=4)
    for cell in book.cells:
        if cell.cell_type == "code":
            raw = base64.b64decode(cell.outputs[0].data["image/png"])
            image = np.asarray(Image.open(io.BytesIO(raw)))
            assert image.std() > 10
            assert image.shape[0] > 100


def test_reporting_imports_do_not_load_torch_or_execution_modules():
    code = """
import sys
from mnist_experiment.kalman_pi.registration import report, analysis
assert 'torch' not in sys.modules
prefix = 'mnist_experiment.kalman_pi.registration.'
assert not any(prefix + name in sys.modules for name in ('model','data','config','trajectory','runtime','run'))
assert analysis.summarize([], None)['schema'] == 1
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
