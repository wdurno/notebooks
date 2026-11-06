"""Artifact-only Phase 5 publication, historical preservation, and isolation."""

import base64
from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
from PIL import Image
import pytest

from mnist_experiment.kalman_pi import report as historical_report
from mnist_experiment.kalman_pi.retention import analysis, report


@pytest.fixture(autouse=True)
def plotting_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))


def png_bytes():
    data = io.BytesIO()
    Image.new("RGB", (24, 24), "#428579").save(data, format="PNG")
    return data.getvalue()


@pytest.fixture
def historical(tmp_path):
    directory = tmp_path / "historical" / "report" / "generation-original"
    directory.mkdir(parents=True)
    cells = [nbformat.v4.new_markdown_cell("# Original report", id="plan17-title"),
             nbformat.v4.new_markdown_cell("## Progress\n\nOriginal complete summary.\n\n| Phase 5 follow-up | Drafted; not started |", id="plan17-progress"),
             nbformat.v4.new_markdown_cell("Original settings, costs and table.", id="plan17-settings")]
    for name, title in historical_report.FIGURES:
        png = png_bytes()
        (directory / (name + ".png")).write_bytes(png)
        cells.append(nbformat.v4.new_code_cell(
            f"from IPython.display import Image, display\ndisplay(Image(filename='original/{name}.png'))",
            id="plan17-" + name, execution_count=3,
            outputs=[nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png).decode(), "text/plain": title})],
            metadata={"original_metadata": {"preserve": True}}))
    cells.extend([nbformat.v4.new_markdown_cell("## Conclusion\n\nExact historical conclusion and its original intervals.", id="plan17-conclusion"),
                  nbformat.v4.new_markdown_cell("[Historical artifacts](original/analysis.json)", id="plan17-artifacts"),
                  historical_report._retention_followup()])
    book = nbformat.v4.new_notebook(cells=cells, metadata={"plan17_report": {
        "schema": 1, "study_id": "original-v4", "smoke": False,
        "snapshot_sha256": "original-snapshot", "generation": directory.name}, "original_custom": "untouched"})
    nbformat.write(book, directory / "report.ipynb")
    (directory / "manifest.json").write_text(json.dumps({"schema": 1, "report_dir": str(directory), "snapshot_sha256": "original-snapshot"}))
    return directory, nbformat.read(directory / "report.ipynb", as_version=4)


def trajectory(condition="fixed", replica=0, *, stage="fresh", schedule="main", ridge=None, completed=True, factor=1):
    ridge = {"fixed": .1, "covariance": .03, "ema_h16": .1, "kalman_eta0.0001": 0}[condition] if ridge is None else ridge
    delta = 0 if condition == "fixed" else .015 * (replica + 1)
    metrics = []
    for index, arrivals in enumerate((0, 4, 12, 20)):
        metrics.append(dict(step=arrivals // 4, arrivals=arrivals * factor, p=.005 if index < 2 else .1,
            segment="low_hold" if index < 2 else "rise", expected_nines=arrivals * factor / 10,
            observed_nines=int(arrivals * factor / 4), nine_recall=.2 + .1 * index + delta * index,
            old_accuracy=.9 - .02 * index + delta * index / 20, frozen_nine_recall=.2, frozen_old_accuracy=.9,
            wall_seconds=float(arrivals * (1 if condition == "fixed" else 2)), initial_nines=3))
    return dict(unit_id=f"{stage}-{schedule}-{replica}-{condition}-{ridge}", stage=stage, replica=replica,
        schedule=schedule, condition=condition, c=.025, ridge=ridge, completed=completed, metrics=metrics,
        schedule_factor=factor, stream_hash=f"{stage}-{schedule}-{replica}-stream", initial_unit=f"{stage}-{replica}-init",
        asset_unit=f"{stage}-{replica}-assets", summary={"costs": {"solver_seconds": 3., "curvature_seconds": 4.,
            "controller_seconds": 0. if condition == "fixed" else 1., "checkpoint_seconds": 1., "evaluation_seconds": 2.}})


def snapshot(historical_dir=None, trajectories=None, *, smoke=False, frozen=True):
    selection = dict(conditions=list(report.CONDITIONS), development_replicas={"main": 2, "stationary": 1},
        main_replicas=2, stationary_replicas=2, bootstrap_seed=837, ridges={}, development_selection={})
    if frozen:
        selection.update(frozen_at="2026-11-02T10:00:00Z",
            ridges={s: {"fixed": .1, "covariance": .03, "ema_h16": .1, "kalman_eta0.0001": 0.} for s in report.SCHEDULES},
            development_selection={s: {c: {"ridge": {"fixed": .1, "covariance": .03, "ema_h16": .1, "kalman_eta0.0001": 0.}[c],
                "old_gap_pp": .2, "old_gap_ci95": [-.1, .4], "matched": True} for c in report.CONDITIONS} for s in report.SCHEDULES})
    return dict(schema=1, kind=report.KIND, study_id="retention-study", status="running", phase="5B: calibration",
        config={"smoke": smoke, "bootstrap_draws": 30, "seed": 123},
        progress={"active_hours": 4.5, "allocation_hours": 20, "active_unit": "next-unit", "step": 3,
            "total_steps": 20, "last_checkpoint": "accepted-3", "forecast_seconds": 3600,
            "storage": {"used_bytes": 1e9, "external_bytes": 2e9, "reserved_bytes": 1e9, "max_bytes": 500e9, "filesystem_free_bytes": 30e9}},
        selection=selection, changes=[], trajectories=trajectories or [], analysis={},
        historical_report_dir=str(historical_dir) if historical_dir else None,
        **({"historical_notebook_sha256": report._hash((historical_dir / "report.ipynb").read_bytes())} if historical_dir else {}))


def write_snapshot(root, value):
    root.mkdir(parents=True, exist_ok=True)
    (root / "report_snapshot.json").write_text(json.dumps(value, allow_nan=True))


@pytest.fixture
def fast_renderers(monkeypatch):
    calls = []
    def render(snapshot, result, path):
        calls.append(path.name)
        path.write_bytes(png_bytes())
    for name in ("_development", "_fresh_accuracy", "_paired_effects"):
        monkeypatch.setattr(report, name, render)
    return calls


def read_book(path):
    return nbformat.read(path, as_version=4)


def prose(book):
    return "\n\n".join(c.source for c in book.cells if c.cell_type == "markdown")


def test_original_cells_outputs_conclusion_links_and_metadata_preserved_exactly(tmp_path, historical, fast_renderers, monkeypatch):
    directory, original = historical
    root, notebook = tmp_path / "phase5", tmp_path / "results.ipynb"
    value = snapshot(directory)
    write_snapshot(root, value)
    before = {p: p.read_bytes() for p in directory.iterdir()}
    def forbidden(*args, **kwargs):
        pytest.fail("historical figures must never be rerendered")
    for name in ("_shadow_calibration", "_development", "_fresh_accuracy", "_precision_exposure", "_policy", "_paired_cost"):
        monkeypatch.setattr(historical_report, name, forbidden)
    manifest = report.build_report(root, notebook)
    book = read_book(notebook)
    nbformat.validate(book)
    original_cells = [c for c in original.cells if c.id != "plan17-retention-followup"]
    combined_by_id = {c.id: c for c in book.cells}
    assert all(combined_by_id[c.id] == c for c in original_cells)
    assert [c.id for c in book.cells if c.id in {c.id for c in original_cells}] == [c.id for c in original_cells]
    assert combined_by_id["plan17-retention-followup"].source != original.cells[-1].source
    assert book.cells[1].id == "plan17-retention-progress"
    assert book.metadata.plan17_report == original.metadata.plan17_report
    assert book.metadata.original_custom == "untouched"
    assert book.metadata.plan17_retention_report.study_id == value["study_id"]
    assert len([c for c in book.cells if c.cell_type == "code"]) == 9
    assert book.cells.index(combined_by_id["plan17-conclusion"]) < next(i for i, c in enumerate(book.cells) if c.id == "plan17-retention-followup")
    assert fast_renderers == [name + ".png" for name, _ in report.FIGURES]
    assert all(p.read_bytes() == data for p, data in before.items())
    sealed = Path(manifest["report_dir"])
    assert report._verify_generation(sealed) == manifest
    assert json.loads((root / "report/latest.json").read_text()) == manifest
    assert (sealed / "report.ipynb").read_bytes() == notebook.read_bytes()
    assert json.loads((sealed / "report_snapshot.json").read_text()) == value
    saved_analysis = json.loads((sealed / "analysis.json").read_text())
    assert saved_analysis["bootstrap"]["seed"] == value["selection"]["bootstrap_seed"]
    assert str(Path(analysis.__file__).resolve()) in manifest["sources_sha256"]


def test_repeat_uses_validated_generation_without_rendering_and_new_snapshot_appends(tmp_path, historical, fast_renderers):
    directory, _ = historical
    root, notebook = tmp_path / "phase5", tmp_path / "results.ipynb"
    value = snapshot(directory)
    write_snapshot(root, value)
    first = report.build_report(root, notebook)
    first_files = {p: p.read_bytes() for p in Path(first["report_dir"]).iterdir()}
    assert report.build_report(root, notebook) == first
    assert len(fast_renderers) == 3
    value["progress"]["step"] = 4
    write_snapshot(root, value)
    second = report.build_report(root, notebook)
    assert second["report_dir"] != first["report_dir"]
    assert len(fast_renderers) == 6
    assert all(p.read_bytes() == data for p, data in first_files.items())


def test_helper_source_change_creates_new_generation(tmp_path, fast_renderers, monkeypatch):
    root = tmp_path / "smoke"
    write_snapshot(root, snapshot(smoke=True))
    first = report.build_report(root)
    read = Path.read_bytes
    helper = Path(report.__file__).resolve().parents[2] / "diagonal_lfu/report.py"
    monkeypatch.setattr(Path, "read_bytes", lambda p: read(p) + (b"\n# source probe\n" if p == helper else b""))
    second = report.build_report(root)
    assert second["report_dir"] != first["report_dir"]


def test_pending_prefreeze_no_data_is_explicit_and_preserves_original_summary(tmp_path, historical, fast_renderers):
    directory, _ = historical
    root, notebook = tmp_path / "phase5", tmp_path / "results.ipynb"
    value = snapshot(directory, frozen=False)
    value.update(status="drafted", phase="5A: planning")
    write_snapshot(root, value)
    report.build_report(root, notebook)
    text = prose(read_book(notebook))
    for fragment in ("Drafted, not started", "Pending: no complete fresh four-arm block", "Original complete summary",
                     "equal-weight", "+/-0.5 pp", "freeze ridge", "not evidence of equivalence", "both metrics with 95%",
                     "4.50 / 20.00", "next-unit / accepted-3", "1.00", "4.00 / 500.00"):
        assert fragment in text


@pytest.mark.parametrize("planned_development", (114, 150))
def test_progress_uses_planned_trajectory_denominators_and_visible_heartbeat(planned_development):
    rows = [trajectory("fixed", stage="development"), trajectory("covariance", stage="development", completed=False),
            trajectory("fixed")]
    value = snapshot(trajectories=rows)
    value["progress"].update(planned_development=planned_development, planned_fresh=48,
                             planned_trajectories=planned_development + 48, heartbeat="2026-11-02T12:34:56Z")
    result = analysis.summarize(rows, value["selection"], bootstrap_draws=0)
    text = report._progress(value, result).source
    assert "Trajectories (complete / planned)" in text
    assert f"Development 1 / {planned_development}; fresh 1 / 48; all 2 / {planned_development + 48}" in text
    assert "Heartbeat (snapshot freshness) | 2026-11-02T12:34:56Z" in text
    assert "original progress and summary below describe Phases 0--4" in text


@pytest.mark.parametrize("corruption", ("schema", "kind", "study", "nan", "inf", "selection", "analysis"))
def test_invalid_snapshot_preserves_previous_notebook_and_latest(tmp_path, fast_renderers, corruption):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    manifest = report.build_report(root)
    notebook = Path(manifest["notebook"])
    before_book, before_latest = notebook.read_bytes(), (root / "report/latest.json").read_bytes()
    if corruption == "schema":
        value["schema"] = True
    elif corruption == "kind":
        value["kind"] = "plan17-paired-analysis"
    elif corruption == "study":
        value["progress"]["study_id"] = "other"
    elif corruption == "nan":
        value["progress"]["active_hours"] = float("nan")
    elif corruption == "inf":
        value["config"]["opaque"] = 1e309
    elif corruption == "selection":
        value["selection"]["ridges"]["main"]["fixed"] = .3
    else:
        value["analysis"] = analysis.summarize([], value["selection"], bootstrap_draws=30, seed=837)
        value["analysis"]["counts"]["complete"] = 9
    write_snapshot(root, value)
    with pytest.raises(ValueError):
        report.build_report(root)
    assert notebook.read_bytes() == before_book
    assert (root / "report/latest.json").read_bytes() == before_latest
    assert not list((root / "report").glob(".building-*"))


@pytest.mark.parametrize("corruption", ("relative", "missing", "pinned_hash", "png", "placeholder", "combined", "smoke"))
def test_historical_corruption_cannot_replace_existing_report(tmp_path, historical, fast_renderers, corruption):
    directory, _ = historical
    root, notebook = tmp_path / "phase5", tmp_path / "results.ipynb"
    value = snapshot(directory)
    write_snapshot(root, value)
    report.build_report(root, notebook)
    before = notebook.read_bytes(), (root / "report/latest.json").read_bytes()
    if corruption == "relative":
        value["historical_report_dir"] = "relative/path"
    elif corruption == "missing":
        value["historical_report_dir"] = None
    elif corruption == "pinned_hash":
        value["historical_notebook_sha256"] = "wrong"
    elif corruption == "png":
        (directory / "01_shadow_calibration.png").write_bytes(b"corrupt")
    else:
        book = read_book(directory / "report.ipynb")
        if corruption == "placeholder":
            book.cells[-1].source = "Phase 5 results already exist"
        elif corruption == "combined":
            book.metadata[report.MARKER] = {"schema": 1}
        else:
            book.metadata.plan17_report.smoke = True
        nbformat.write(book, directory / "report.ipynb")
        value["historical_notebook_sha256"] = report._hash((directory / "report.ipynb").read_bytes())
    write_snapshot(root, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        report.build_report(root, notebook)
    assert (notebook.read_bytes(), (root / "report/latest.json").read_bytes()) == before


def test_corrupt_generation_is_not_reused_or_repaired(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    write_snapshot(root, snapshot(smoke=True))
    manifest = report.build_report(root)
    notebook, latest = Path(manifest["notebook"]), root / "report/latest.json"
    before = notebook.read_bytes(), latest.read_bytes()
    (Path(manifest["report_dir"]) / "analysis.json").write_text("corruption")
    with pytest.raises(ValueError, match="corrupt report generation"):
        report.build_report(root)
    assert (notebook.read_bytes(), latest.read_bytes()) == before
    assert (Path(manifest["report_dir"]) / "analysis.json").read_text() == "corruption"


def test_render_failure_leaves_previous_publication_and_no_temporary_generation(tmp_path, fast_renderers, monkeypatch):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    notebook, latest = Path(first["notebook"]), root / "report/latest.json"
    before = notebook.read_bytes(), latest.read_bytes()
    value["progress"]["step"] += 1
    write_snapshot(root, value)
    def fail(*args):
        raise RuntimeError("render failed")
    monkeypatch.setattr(report, "_fresh_accuracy", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        report.build_report(root)
    assert (notebook.read_bytes(), latest.read_bytes()) == before
    assert not list((root / "report").glob(".building-*"))


def test_canonical_write_failure_rolls_back_latest(tmp_path, fast_renderers, monkeypatch):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    notebook, latest = Path(first["notebook"]), root / "report/latest.json"
    before = notebook.read_bytes(), latest.read_bytes()
    value["progress"]["step"] += 1
    write_snapshot(root, value)
    atomic = report._atomic
    def fail(path, data):
        if path == notebook:
            raise OSError("publication interrupted")
        atomic(path, data)
    monkeypatch.setattr(report, "_atomic", fail)
    with pytest.raises(OSError, match="publication interrupted"):
        report.build_report(root)
    assert (notebook.read_bytes(), latest.read_bytes()) == before


def test_frozen_selection_change_rejected_even_with_recomputed_analysis(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    before = Path(first["notebook"]).read_bytes()
    value["selection"]["ridges"]["main"]["ema_h16"] = .3
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="frozen selection changed"):
        report.build_report(root)
    assert Path(first["notebook"]).read_bytes() == before


def test_smoke_self_contained_and_cannot_overwrite_production_or_other_study(tmp_path, historical, fast_renderers):
    directory, _ = historical
    root = tmp_path / "smoke"
    write_snapshot(root, snapshot(smoke=True))
    manifest = report.build_report(root)
    book = read_book(manifest["notebook"])
    assert book.metadata.plan17_retention_report.smoke is True
    assert "SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE" in book.cells[0].source
    assert "No production efficacy" in prose(book)
    assert len([c for c in book.cells if c.cell_type == "code"]) == 3
    assert "Original report" not in prose(book)
    protected = tmp_path / "production.ipynb"
    protected.write_bytes((directory / "report.ipynb").read_bytes())
    before = protected.read_bytes()
    with pytest.raises(ValueError, match="production"):
        report.build_report(root, protected)
    assert protected.read_bytes() == before
    production_root = tmp_path / "production"
    value = snapshot(directory)
    write_snapshot(production_root, value)
    with pytest.raises(ValueError, match="study or production/smoke"):
        report.build_report(production_root, Path(manifest["notebook"]))
    report.build_report(production_root, protected)
    value["study_id"] = "other-study"
    write_snapshot(production_root, value)
    with pytest.raises(ValueError, match="another study"):
        report.build_report(production_root, protected)


def test_immutable_historical_notebook_cannot_be_output(tmp_path, historical, fast_renderers):
    directory, _ = historical
    root = tmp_path / "phase5"
    value = snapshot(directory)
    write_snapshot(root, value)
    before = (directory / "report.ipynb").read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        report.build_report(root, directory / "report.ipynb")
    assert (directory / "report.ipynb").read_bytes() == before


@pytest.fixture
def captured_plots(monkeypatch):
    recorded = []
    figure = report._figure
    @contextmanager
    def capture(*args, **kwargs):
        with figure(*args, **kwargs) as (fig, axes):
            yield fig, axes
            recorded.append([{"title": ax.get_title(), "xlabel": ax.get_xlabel(), "xlim": ax.get_xlim(), "ylim": ax.get_ylim(),
                "patches": len(ax.patches), "texts": [t.get_text() for t in ax.texts],
                "lines": [{"label": line.get_label(), "x": np.asarray(line.get_xdata()).copy(), "y": np.asarray(line.get_ydata()).copy()} for line in ax.lines],
                "points": [np.asarray(c.get_offsets()).copy() for c in ax.collections]} for ax in axes.flat])
    monkeypatch.setattr(report, "_figure", capture)
    return recorded


def test_fresh_2x2_keeps_schedules_length_cohorts_references_exposure_and_partial_blocks_separate(tmp_path, captured_plots):
    rows = [trajectory(c, r, schedule=s, factor=1 if r == 0 else 1.25) for s in report.SCHEDULES for r in (0, 1) for c in report.CONDITIONS]
    for t in rows:
        if t["schedule"] == "stationary" and t["replica"] == 1 and t["condition"] == "ema_h16":
            t["completed"] = False
    value = snapshot(trajectories=rows)
    result = analysis.summarize(rows, value["selection"], bootstrap_draws=30, seed=837)
    report._fresh_accuracy(value, result, tmp_path / "fresh.png")
    axes = captured_plots[-1]
    assert len(axes) == 4
    assert [a["title"].split(":")[0] for a in axes] == ["Changing", "Changing", "Stationary", "Stationary"]
    assert all(a["ylim"] == (0, 100) for a in axes)
    assert axes[0]["xlim"] == axes[1]["xlim"] and axes[2]["xlim"] == axes[3]["xlim"]
    paths = [line for line in axes[0]["lines"] if not line["label"].startswith("Initial reference")]
    assert len(paths) == 8
    assert {tuple(p["x"]) for p in paths} == {(0, 4, 12, 20), (0, 5, 15, 25)}
    assert any("9 exp/obs=" in text for text in axes[0]["texts"])
    assert len([p for p in axes[2]["lines"] if not p["label"].startswith("Initial reference")]) == 4
    assert len([p for p in axes[0]["lines"] if p["label"].startswith("Initial reference")]) == 2


def test_frontier_displays_all_measured_candidates_band_ridges_uncertainty_and_selected_marks(tmp_path, captured_plots):
    rows = [trajectory("fixed", r, stage="development") for r in (0, 1)]
    rows += [trajectory(c, r, stage="development", ridge=ridge) for c in report.CONDITIONS[1:] for r in (0, 1) for ridge in (0, .03, .1, .3)]
    value = snapshot(trajectories=rows)
    result = analysis.summarize(rows, value["selection"], bootstrap_draws=30, seed=837)
    report._development(value, result, tmp_path / "development.png")
    changing, stationary = captured_plots[-1]
    assert changing["patches"] == stationary["patches"] == 1
    # Preserve every measured point and interval; annotate only selected/zero ridges.
    assert len(changing["points"]) == 13
    assert len(changing["texts"]) == 5
    assert "r=0.03" in changing["texts"] and "r=0.1" in changing["texts"]
    assert changing["texts"].count("r=0") == 3
    assert "r=0.3" not in changing["texts"]
    assert len(changing["lines"]) > 15  # Includes measured paired interval segments.
    assert "Pending complete development pairs" in stationary["texts"]


def test_primary_publisher_cannot_remove_supplemental_notebook(tmp_path, historical, fast_renderers):
    directory, original = historical
    root, notebook = tmp_path / "phase5", tmp_path / "results.ipynb"
    book = deepcopy(original)
    book.metadata["plan17_zero_ridge_report"] = {"schema": 1, "study_id": "supplement"}
    nbformat.write(book, notebook)
    before = notebook.read_bytes()
    write_snapshot(root, snapshot(directory))
    with pytest.raises(ValueError, match="Phase 5E"):
        report.build_report(root, notebook)
    assert notebook.read_bytes() == before
    assert not (root / "report").exists()


def test_effects_both_metrics_intervals_band_and_contrary_regimes(tmp_path, captured_plots):
    rows = [trajectory(c, r) for r in (0, 1) for c in report.CONDITIONS]
    value = snapshot(trajectories=rows)
    result = analysis.summarize(rows, value["selection"], bootstrap_draws=30, seed=837)
    report._paired_effects(value, result, tmp_path / "effects.png")
    axes = captured_plots[-1]
    assert len(axes) == 2 and axes[1]["patches"] == 1
    assert all("paired difference" in ax["xlabel"] for ax in axes)
    c = result["groups"][0]["comparisons"][0]
    c["segments"]["rise"]["observation_mean_pp"]["nine_recall"]["mean"] = -2.
    c["observation_mean_pp"]["old_accuracy"]["ci95"] = [-2., 2.]
    text = report._interpretation(value, result).source
    assert "Contrary regime effects: rise nine_recall" in text
    assert "comparable retention remains uncertain" in text
    assert "not evidence of equivalence" in text
    assert "actual samples saved" in text
    assert "development trajectory costs" in text
    assert "calibration_seconds=" not in text
    assert "must not be summed as additional compute" in text
    assert "95% CI" in report._results(result).source


def test_real_three_png_publication_nonblank_portable_and_does_not_touch_checkpoints(tmp_path):
    root = tmp_path / "smoke"
    rows = [trajectory(c, r, schedule=s, factor=1 if r == 0 else 1.25) for s in report.SCHEDULES for r in (0, 1) for c in report.CONDITIONS]
    value = snapshot(trajectories=rows, smoke=True)
    value["analysis"] = analysis.summarize(rows, value["selection"], bootstrap_draws=30, seed=837)
    write_snapshot(root, value)
    unit = root / "units" / "sealed"
    unit.mkdir(parents=True)
    (unit / "checkpoint.pt").write_bytes(b"never load this checkpoint")
    (unit / "COMPLETED").write_text("immutable scientific unit")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    manifest = report.build_report(root)
    for cell in read_book(manifest["notebook"]).cells:
        if cell.cell_type != "code":
            continue
        data = base64.b64decode(cell.outputs[0].data["image/png"])
        picture = Image.open(io.BytesIO(data)).convert("RGB")
        assert picture.width > 1000 and picture.height > 600
        assert np.asarray(picture).std() > 10
        assert (Path(manifest["report_dir"]) / (cell.id.removeprefix("plan17-retention-") + ".png")).read_bytes() == data
    assert all(p.read_bytes() == data for p, data in before.items())


def test_fresh_process_report_import_and_build_never_import_torch_or_execution_modules(tmp_path):
    root = tmp_path / "smoke"
    write_snapshot(root, snapshot(smoke=True))
    source = """
import importlib.abc
import sys
class NoCompute(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args, **kwargs):
        if fullname == 'torch' or fullname.startswith(('torch.', 'torchvision')) or fullname.rsplit('.', 1)[-1] in ('run', 'runtime', 'trajectory', 'observations', 'policies', 'data') and 'kalman_pi' in fullname:
            raise AssertionError('forbidden compute import: ' + fullname)
sys.meta_path.insert(0, NoCompute())
from pathlib import Path
from mnist_experiment.kalman_pi.retention.report import build_report
result = build_report(Path(sys.argv[1]))
assert result['smoke'] is True
assert 'torch' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", source, str(root)], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
