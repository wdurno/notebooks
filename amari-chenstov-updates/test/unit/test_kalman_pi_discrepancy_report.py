"""Phase 7 scalar inference and publication; all artifacts are tiny test fixtures."""

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

from mnist_experiment.kalman_pi.discrepancy import analysis, report


@pytest.fixture(autouse=True)
def plotting_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def selection(n=2, stationary=0):
    conditions = {"fixed": "fixed", "covariance": "covariance",
                  "velocity": "velocity_eta0.0001", "discrepancy": "discrepancy_eta0.001"}
    return {"common_ridge": .1, "velocity_condition": conditions["velocity"],
            "discrepancy_condition": conditions["discrepancy"],
            "arms": [{"arm": a, "condition": c, "ridge": .1} for a, c in conditions.items()],
            "main_replicas": n, "stationary_replicas": stationary, "bootstrap_seed": 833,
            "hold_steps": 4, "ramp_steps": 2}


def trajectory(arm="fixed", replica=0, *, angle=25., schedule="main", completed=True, stage="fresh"):
    angle = 0. if schedule == "stationary" else float(angle)
    spec = next(a for a in selection()["arms"] if a["arm"] == arm)
    effect = {"fixed": 0., "covariance": .1, "velocity": .2, "discrepancy": .3}[arm]
    rows = []
    for i, (step, arrivals) in enumerate(zip((0, 4, 6, 8, 12), (0, 8, 20, 36, 64))):
        change = 0 if i == 0 else effect * (replica + 1) * (2 if angle == 45 else 1)
        old = 3 + i / 4 - .3 * change
        current = old if schedule == "stationary" else 4 + i - change
        row = {"step": step, "arrivals": arrivals,
               "segment": "stationary" if schedule == "stationary" else ("initial_hold", "rise", "return", "final_hold", "final_hold")[i],
               "current_risk": current, "old_risk": old, "shifted_risk": 16 - i - change,
               "current_accuracy": .5 + i / 100 + change / 100,
               "old_accuracy": .8 - i / 100 + change / 100, "shifted_accuracy": .2 + i / 20 + change / 100,
               "current_nll": 10., "old_nll": 9., "shifted_nll": 12.,
               "frozen_current_risk": 3. if schedule == "stationary" else 4. + i,
               "frozen_current_accuracy": .5 + i / 100,
               "applied_pi": None if i == 0 else .025, "raw_pi": None if i == 0 else .025,
               "motion_energy": float(i), "uncertainty_energy": float(i * 2),
               "discrepancy_energy": float(i * 3), "level_uncertainty_energy": float(i * 4),
               "noise_scale": 1., "innovation_ratio": 1.}
        if schedule == "stationary":
            row.update(shifted25_risk=16 - i - change, shifted45_risk=25 - i - change,
                       shifted25_accuracy=.2, shifted45_accuracy=.1)
        rows.append(row)
    if not completed:
        rows = rows[:2]
    return {"unit_id": f"{stage}-{schedule}-{angle:g}-{replica}-{arm}", "stage": stage,
            "cohort": stage, "replica": replica, "schedule": schedule, "angle_max": angle,
            **spec, "c": .025, "sign": (-1, 1)[replica % 2],
            "schedule_factor": (1., 1.25)[(replica // 2) % 2],
            "hold_steps": 4, "ramp_steps": 2, "final_step": 12,
            "initial_unit": f"initial-{replica}", "asset_unit": f"asset-{angle:g}-{replica}",
            "stream_hash": f"stream-{schedule}-{angle:g}-{replica}",
            "metrics": rows, "completed": completed,
            "summary": {"learning_seconds": 10. - effect * 10, "online_wall_seconds": 12.,
                        "costs": {"solver_seconds": 3., "curvature_seconds": 4.,
                                  "controller_seconds": 1., "render_seconds": 1.}}}


def block(replica=0, **kwargs):
    return [trajectory(a, replica, **kwargs) for a in analysis.ARMS]


def png_bytes():
    output = io.BytesIO()
    image = Image.new("RGB", (24, 24), "#428579")
    image.putpixel((12, 12), (255, 255, 255))
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def historical(tmp_path):
    directory = tmp_path / "historical/report/generation-history"
    directory.mkdir(parents=True)
    png = png_bytes()
    cells = [nbformat.v4.new_markdown_cell("# Historical evidence", id="plan17-title")]
    for name in ("plan17", "plan17-retention", "plan17-zero-ridge", "plan17-registration"):
        cells.append(nbformat.v4.new_markdown_cell("Exact inherited conclusion", id=name + "-conclusion", metadata={"custom": [1, 2]}))
    for name, _ in report.historical_report.FIGURES:
        (directory / (name + ".png")).write_bytes(png)
        cells.append(nbformat.v4.new_code_cell("historical code; never executed", id="plan17-registration-" + name,
            execution_count=17, metadata={"preserve": True}, outputs=[nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png).decode()})]))
    while len(cells) < 52:
        cells.append(nbformat.v4.new_markdown_cell(f"Historical content {len(cells)}", id=f"prior-{len(cells)}"))
    sealed = {"schema": 1, "kind": "plan17-registration-snapshot", "study_id": "historical", "status": "completed"}
    data = report._json_bytes(sealed)
    book = nbformat.v4.new_notebook(cells=cells, metadata={"custom": {"all": [1, 2, 3]},
        "plan17_registration_report": {"schema": 1, "smoke": False, "status": "Completed Phase 6",
            "study_id": "historical", "generation": directory.name, "snapshot_sha256": report._hash(data)}})
    (directory / "report_snapshot.json").write_bytes(data)
    nbformat.write(book, directory / "report.ipynb")
    (directory / "analysis.json").write_text("{}")
    manifest = {"schema": 1, "kind": "plan17-registration-report", "study_id": "historical", "smoke": False,
        "report_dir": str(directory), "snapshot_sha256": report._hash(data),
        "files_sha256": {p.name: report._hash(p.read_bytes()) for p in directory.iterdir()}}
    encoded = report._json_bytes(manifest)
    (directory / "manifest.json").write_bytes(encoded)
    (directory / "COMPLETED").write_text(report._hash(encoded) + "\n")
    return directory


def snapshot(history=None, rows=None, selected=None, *, smoke=False):
    return {"schema": 1, "kind": report.KIND, "study_id": "discrepancy-v1", "status": "running",
            "phase": "7A: acceptance", "config": {"smoke": smoke, "hold_steps": 4, "ramp_steps": 2,
                "samples_per_step": 4, "bootstrap_draws": 32},
            "selection": selected, "trajectories": rows or [], "readiness": {}, "diagnostics": {},
            "progress": {"phase7_active_hours": .125, "active_hours": .125, "historical_active_hours": 18.619,
                "phase7_allocation_hours": 20., "heartbeat": "2026-11-05T10:00:00Z", "active_unit": "CPU-acceptance",
                "last_accepted_checkpoint": "accepted-3", "forecast_seconds": 3600,
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


def test_current_budget_integrates_squared_error_on_actual_arrivals_then_roots():
    rows = trajectory()["metrics"]
    value = analysis.summarize_path(rows, 4, 2)
    expected = sum((b["arrivals"] - a["arrivals"]) * (a["current_risk"] + b["current_risk"]) / 2
                   for a, b in zip(rows, rows[1:])) / rows[-1]["arrivals"]
    assert value["Acurrent"] == expected
    assert value["Ecurrent"] == value["current_rms"] == np.sqrt(expected)
    assert value["current_rms"] != np.mean([np.sqrt(r["current_risk"]) for r in rows])
    assert value["normalized_score"] == expected
    assert value["regimes"]["rise"]["current"] == 5.5
    assert value["Ashifted"] == analysis._area(rows, "shifted_risk")
    assert value["Afrozen_current"] == expected


def test_fixed_current_reference_is_a_curve_not_initialization_constant():
    t = trajectory()
    ax = Mock()
    report._curves(ax, [t], "frozen_current_risk", rms=True, frozen=True)
    x, y = ax.plot.call_args.args[:2]
    assert x == [r["arrivals"] for r in t["metrics"]]
    np.testing.assert_allclose(y, np.sqrt([4, 5, 6, 7, 8]))
    assert y[0] != y[-1]


def test_conclusion_reports_both_policies_and_keeps_full_contrasts_in_artifacts():
    rows = block(0) + block(1) + block(0, angle=45) + block(1, angle=45)
    value = snapshot(rows=rows, selected=selection())
    result = analysis.summarize(rows, selection(), bootstrap_draws=32)
    cell = report._conclusion(value, result)
    assert "Velocity Kalman improves whole-path tracking" in cell.source
    assert "Discrepancy Kalman improves whole-path tracking" in cell.source
    assert "and both ramps" in cell.source
    assert cell.source.count("| 25 deg:") == 3
    assert cell.source.count("| 45 deg:") == 3
    assert "Learning ratio [95% CI]" not in cell.source
    assert "full intervals and all 15 contrasts" in cell.source
    assert all(len(g["comparisons"]) == 5 for g in result["groups"])


def test_completed_progress_has_no_pending_active_checkpoint():
    rows = block(0) + block(1) + block(0, angle=45) + block(1, angle=45)
    value = snapshot(rows=rows, selected=selection())
    value["status"] = "completed"
    result = analysis.summarize(rows, selection(), bootstrap_draws=0)
    text = report._progress(value, result).source
    assert "none (all paths complete)" in text
    assert "accepted checkpoint: all trajectories sealed" in text
    value["status"] = "running"
    value["progress"].pop("last_accepted_checkpoint")
    value["progress"]["step"] = 3
    assert "accepted checkpoint: batch 3" in report._progress(value, result).source


def test_stationary_panels_do_not_draw_fictitious_drift():
    ax = Mock()
    report._regimes(ax, block(schedule="stationary"), snapshot())
    ax.axvspan.assert_not_called()
    ax.axvline.assert_not_called()
    report._regimes(ax, block(), snapshot(selected=selection()))
    assert ax.axvspan.call_count == 2 and ax.axvline.call_count == 3


def test_energy_plot_keeps_nonnegative_range_and_sparse_ticks(tmp_path, monkeypatch):
    from contextlib import contextmanager
    original = report._figure
    captured = {}

    @contextmanager
    def figure(**kwargs):
        with original(**kwargs) as (fig, axes):
            yield fig, axes
            captured["energy"] = [{"limits": ax.get_ylim(), "ticks": ax.get_yticks().tolist(),
                                   "label": ax.get_ylabel(), "scale": ax.get_yscale()}
                                  for ax in axes[:, 1]]

    monkeypatch.setattr(report, "_figure", figure)
    rows = block(0) + block(1) + block(0, angle=45) + block(1, angle=45) + block(schedule="stationary")
    for t in rows:
        for row in t["metrics"]:
            row["discrepancy_energy"] *= 10000
    value = snapshot(rows=rows, selected=selection(stationary=1))
    result = analysis.summarize(rows, value["selection"], bootstrap_draws=0)
    report._effects(value, result, tmp_path / "effects.png")
    for axis in captured["energy"]:
        assert axis["limits"][0] == 0 and axis["limits"][1] > 10000
        assert 1 <= len(axis["ticks"]) <= 6
        assert all(tick >= 0 for tick in axis["ticks"])
        assert axis["label"] == "Energy" and axis["scale"] == "symlog"


@pytest.mark.parametrize("mutate", [lambda t: t["metrics"].pop(2), lambda t: t["metrics"].pop(),
    lambda t: t["metrics"][1].update(arrivals=0), lambda t: t["metrics"][1].update(current_risk=-1),
    lambda t: t["metrics"][1].update(current_accuracy=1.01), lambda t: t["metrics"][1].update(applied_pi=.3),
    lambda t: t["metrics"][1].update(discrepancy_energy=-1), lambda t: t.update(angle_max=30),
    lambda t: t.update(c=.05), lambda t: t.update(final_step=13), lambda t: t.update(condition="other"),
    lambda t: t.update(sign=0)])
def test_invalid_or_incomplete_complete_paths_fail(mutate):
    t = trajectory()
    mutate(t)
    with pytest.raises(ValueError):
        analysis.summarize([t], selection())


def test_stationary_is_one_cohort_with_two_generalization_panels():
    rows = block(schedule="stationary")
    result = analysis.summarize(rows, selection(stationary=1), bootstrap_draws=0)
    assert [(g["schedule"], g["angle_max"], g["n_blocks"]) for g in result["groups"]] == [("stationary", 0., 1)]
    assert result["counts"]["paired_blocks"] == 1
    path = result["path_summaries"][rows[0]["unit_id"]]
    assert path["current_rms"] == path["old_rms"]
    assert path["shifted25_rms"] != path["shifted45_rms"]
    assert "shifted45_rms_gap" in result["groups"][0]["comparisons"][0]["effects"]
    rows[0]["angle_max"] = 25.
    with pytest.raises(ValueError, match="stationary"):
        analysis.summarize(rows, selection(stationary=1))


def test_partial_block_never_enters_inference_or_paired_amplitude_interaction():
    rows = block() + block(angle=45)
    rows[-1] = trajectory("discrepancy", angle=45, completed=False)
    result = analysis.summarize(rows, selection())
    assert [g["n_blocks"] for g in result["groups"]] == [1, 0]
    assert result["counts"] == {"complete": 7, "incomplete": 1, "paired_blocks": 1}
    assert len(result["awaiting_full_block_units"]) == 3
    assert not result["cross_angle"]


def test_pairing_rejects_assets_grids_and_initial_panel_mismatches():
    for name, value in (("sign", 1), ("schedule_factor", 1.25), ("stream_hash", "other"), ("asset_unit", "other")):
        rows = block()
        rows[-1][name] = value
        with pytest.raises(ValueError, match="paired block"):
            analysis.summarize(rows, selection())
    for field in ("current_risk", "frozen_current_accuracy"):
        rows = block()
        rows[-1]["metrics"][0][field] += .1
        with pytest.raises(ValueError, match="initialization"):
            analysis.summarize(rows, selection())
    rows = block()
    rows[-1]["metrics"][1]["arrivals"] += 1
    with pytest.raises(ValueError, match="grids"):
        analysis.summarize(rows, selection())
    with pytest.raises(ValueError, match="duplicate"):
        analysis.summarize([trajectory(), trajectory()], selection())


def test_cross_angle_pairing_allows_angle_specific_assets_but_preserves_initialization():
    rows = block() + block(angle=45)
    assert analysis.summarize(rows, selection())["cross_angle"][0]["n_blocks"] == 1
    rows[-1]["initial_unit"] = "different"
    # Make the whole 45-degree block internally consistent to isolate amplitude pairing.
    for t in rows[4:]:
        t["initial_unit"] = "different"
    with pytest.raises(ValueError, match="cross-angle"):
        analysis.summarize(rows, selection())


def test_contrasts_roots_within_replica_and_shared_cross_angle_bootstrap():
    rows = [t for angle in (25, 45) for r in range(8) for t in block(r, angle=angle)]
    result = analysis.summarize(rows, selection(n=8), bootstrap_draws=40, seed=12)
    assert [g["n_blocks"] for g in result["groups"]] == [8, 8]
    assert all(g["bootstrap_mode"] == "stratified_sign_factor" for g in result["groups"])
    assert result["groups"][0]["bootstrap_indices_sha256"] == result["groups"][1]["bootstrap_indices_sha256"]
    cross = result["cross_angle"][0]
    assert cross["n_blocks"] == 8
    assert cross["bootstrap_indices_sha256"] == result["groups"][0]["bootstrap_indices_sha256"]
    comparisons = [("covariance", "fixed"), ("velocity", "fixed"), ("discrepancy", "fixed"),
                   ("discrepancy", "velocity"), ("discrepancy", "covariance")]
    assert [(c["arm"], c["control"]) for c in result["groups"][0]["comparisons"]] == comparisons
    paths = result["path_summaries"]
    expected = np.mean([paths[f"fresh-main-25-{r}-discrepancy"]["current_rms"] - paths[f"fresh-main-25-{r}-fixed"]["current_rms"] for r in range(8)])
    c25, c45 = (g["comparisons"][2] for g in result["groups"])
    assert c25["effects"]["current_rms_gap"]["mean"] == expected
    assert c25["effects"]["current_risk_gain"]["mean"] > 0
    assert c25["effects"]["learning_seconds_ratio"]["mean"] == .7
    assert cross["comparisons"][2]["effects"]["current_rms_gap"]["mean"] == pytest.approx(c45["effects"]["current_rms_gap"]["mean"] - expected)
    reordered = analysis.summarize(list(reversed(rows)), selection(n=8), bootstrap_draws=40, seed=12)
    assert result["groups"] == reordered["groups"] and result["cross_angle"] == reordered["cross_angle"]


def test_missing_costs_are_not_zero_and_bootstrap_bounds_are_explicit():
    rows = block() + block(1)
    rows[-1]["summary"] = {}
    result = analysis.summarize(rows, selection())
    cost = result["groups"][0]["comparisons"][2]["effects"]["learning_seconds_ratio"]
    assert cost["mean"] is None and cost["ci95"] is None
    assert analysis.summarize([], bootstrap_draws=-1)["bootstrap"]["draws"] == 0
    assert analysis.summarize([], bootstrap_draws=999999)["bootstrap"]["draws"] == analysis.MAX_BOOTSTRAP_DRAWS
    for kwargs in ({"bootstrap_draws": True}, {"seed": -1}, {"seed": True}):
        with pytest.raises(ValueError):
            analysis.summarize([], **kwargs)


def test_pending_report_preserves_all_52_historical_cells_outputs_metadata(tmp_path, historical, fast_renderers):
    original = nbformat.read(historical / "report.ipynb", as_version=4)
    before = {p: p.read_bytes() for p in historical.iterdir()}
    root, target = tmp_path / "discrepancy", tmp_path / "canonical.ipynb"
    target.write_bytes((historical / "report.ipynb").read_bytes())
    write_snapshot(root, snapshot(historical))
    manifest = report.build_report(root, target)
    book = nbformat.read(target, as_version=4)
    assert book.cells[1].id == report.PREFIX + "progress"
    assert [c for c in book.cells if not c.id.startswith(report.PREFIX)] == original.cells
    assert len(original.cells) == 52
    assert {k: v for k, v in book.metadata.items() if k != report.MARKER} == original.metadata
    appendix = book.cells[53:]
    assert len([c for c in appendix if c.cell_type == "code"]) == 4
    assert sum("| ---" in c.source for c in appendix if c.cell_type == "markdown") == 2
    assert fast_renderers == [name + ".png" for name, _ in report.FIGURES]
    assert "independent post-batch holdout" in "\n".join(c.source for c in appendix)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert report._verify_generation(Path(manifest["report_dir"])) == manifest
    files = {p: p.read_bytes() for p in Path(manifest["report_dir"]).iterdir()}
    assert report.build_report(root, target) == manifest
    assert len(fast_renderers) == 4
    assert all(p.read_bytes() == data for p, data in files.items())


def test_smoke_pending_report_is_self_contained_and_cannot_replace_production(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    write_snapshot(root, snapshot(smoke=True))
    manifest = report.build_report(root)
    assert Path(manifest["notebook"]) == root / "discrepancy_smoke.ipynb"
    assert manifest["smoke"] is True
    with pytest.raises(ValueError, match="own root"):
        report.build_report(root, tmp_path / "canonical.ipynb")


def test_partial_preview_can_disappear_rollback_and_complete_without_rewriting_identity(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    partial = trajectory("discrepancy", stage="development", completed=False)
    first = snapshot(rows=[partial], smoke=True)
    write_snapshot(root, first)
    report.build_report(root)
    second = deepcopy(first)
    second["trajectories"] = []
    write_snapshot(root, second)
    report.build_report(root)
    second["trajectories"] = [partial]
    write_snapshot(root, second)
    report.build_report(root)
    third = deepcopy(second)
    third["trajectories"][0]["metrics"] = []
    third["trajectories"][0]["summary"] = {"online_wall_seconds": .1}
    write_snapshot(root, third)
    report.build_report(root)
    third["trajectories"] = [trajectory("discrepancy", stage="development")]
    write_snapshot(root, third)
    report.build_report(root)
    third["trajectories"][0]["metrics"][1]["current_risk"] += .1
    write_snapshot(root, third)
    with pytest.raises(ValueError, match="completed evidence"):
        report.build_report(root)


@pytest.mark.parametrize("change,match", [
    (lambda s: s["config"].update(ramp_steps=3), "configuration"),
    (lambda s: s["progress"].update(phase7_active_hours=0.), "compute"),
    (lambda s: s["progress"].update(historical_active_hours=0.), "historical compute"),
    (lambda s: s["trajectories"][0].update(asset_unit="other"), "provisional trajectory identity"),
])
def test_report_rejects_identity_or_ledger_regression(tmp_path, fast_renderers, change, match):
    root = tmp_path / "smoke"
    value = snapshot(rows=[trajectory(stage="development", completed=False)], smoke=True)
    write_snapshot(root, value)
    report.build_report(root)
    change(value)
    write_snapshot(root, value)
    with pytest.raises(ValueError, match=match):
        report.build_report(root)


def test_frozen_selection_cannot_change_after_publication(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    value = snapshot(selected=selection(), smoke=True)
    write_snapshot(root, value)
    report.build_report(root)
    value["selection"]["bootstrap_seed"] += 1
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="frozen selection"):
        report.build_report(root)


def test_unrelated_or_manually_modified_notebook_is_protected(tmp_path, historical, fast_renderers):
    root, target = tmp_path / "study", tmp_path / "canonical.ipynb"
    write_snapshot(root, snapshot(historical))
    target.write_text(nbformat.writes(nbformat.v4.new_notebook()))
    with pytest.raises(ValueError, match="unrelated"):
        report.build_report(root, target)
    target.unlink()
    report.build_report(root, target)
    book = nbformat.read(target, as_version=4)
    book.cells.append(nbformat.v4.new_markdown_cell("User annotation"))
    nbformat.write(book, target)
    before = target.read_bytes()
    with pytest.raises(ValueError, match="modified"):
        report.build_report(root, target)
    assert target.read_bytes() == before


def test_stale_snapshot_during_rendering_cannot_replace_notebook_or_latest(tmp_path, fast_renderers, monkeypatch):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    target = Path(first["notebook"])
    before = target.read_bytes(), (root / "report/latest.json").read_bytes()
    value["progress"]["phase7_active_hours"] += .01
    write_snapshot(root, value)
    render = report._effects
    def stale(snapshot, result, path):
        render(snapshot, result, path)
        newer = deepcopy(value)
        newer["status"] = "stopped"
        write_snapshot(root, newer)
    monkeypatch.setattr(report, "_effects", stale)
    with pytest.raises(ValueError, match="changed during rendering"):
        report.build_report(root)
    assert (target.read_bytes(), (root / "report/latest.json").read_bytes()) == before


def test_corrupt_analysis_and_nonfinite_snapshot_are_rejected():
    value = snapshot(smoke=True)
    value["analysis"] = {"incorrect": True}
    with pytest.raises(ValueError, match="snapshot.analysis"):
        report.validate_snapshot(value)
    value.pop("analysis")
    value["diagnostics"]["noise"] = float("nan")
    with pytest.raises(ValueError):
        report.validate_snapshot(value)


def test_action_curves_omit_only_initial_missing_action_and_never_interpolate_lengths():
    ax = Mock()
    paths = [trajectory(replica=0), trajectory(replica=2)]
    report._curves(ax, paths, "applied_pi")
    assert ax.plot.call_count == 2
    for call in ax.plot.call_args_list:
        assert call.args[0] == [8, 20, 36, 64]
        np.testing.assert_allclose(call.args[1], [.025] * 4)


def test_real_figures_use_recorded_contexts_and_render_nonblank_embedded_outputs(tmp_path):
    root = tmp_path / "smoke"
    rows = [t for angle in (25, 45) for r in range(2) for t in block(r, angle=angle)] + block(schedule="stationary")
    value = snapshot(rows=rows, selected=selection(stationary=1), smoke=True)
    value["status"] = "completed"
    image = np.zeros((40, 40))
    image[15:25, 17:23] = 1
    value["readiness"]["contexts"] = [{"angle_max": angle, "sign": 1, "reference": image.tolist(), "query": image.T.tolist(),
        "reference_landmarks": [[-6, -4], [6, -4], [0, 8]], "query_landmarks": [[-6, -4], [6, -4], [0, 8]],
        "schedule": [{"arrivals": n, "h": h, "mu": angle * h} for n, h in ((0, 0), (16, 0), (24, 1), (32, 0), (48, 0))],
        "angle_diagnostics": {f"shifted{angle}": {"angle_bin_edges": [-70, 0, 70], "accepted_angle_density": [.001, .01]}}}
        for angle in (25, 45)]
    write_snapshot(root, value)
    manifest = report.build_report(root)
    assert manifest["status"] == "Completed Phase 7"
    book = nbformat.read(manifest["notebook"], as_version=4)
    images = [o["data"]["image/png"] for c in book.cells for o in c.get("outputs", [])]
    assert len(images) == 4
    for encoded in images:
        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
        assert pixels.shape[0] >= 400 and pixels.shape[1] >= 600
        assert pixels.std() > 5
    conclusion = next(c.source for c in book.cells if c.id == report.PREFIX + "conclusion")
    assert "retention was not matched" in conclusion and "samples-saved or S statistic" in conclusion


def test_import_and_pending_analysis_do_not_import_torch_or_execution_modules():
    script = """
import sys
from mnist_experiment.kalman_pi.discrepancy import analysis, report
assert analysis.summarize([], None)['schema'] == 1
assert 'torch' not in sys.modules
assert not any(n.startswith('mnist_experiment.kalman_pi.discrepancy.' + part)
               for part in ('run', 'data', 'model', 'trajectory') for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
