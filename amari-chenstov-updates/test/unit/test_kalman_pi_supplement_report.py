"""Scalar-only Phase 5E publication and immutable primary evidence tests."""

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

from mnist_experiment.kalman_pi.retention import report as primary_report
from mnist_experiment.kalman_pi.retention.supplement import analysis, report
from src.seeding import derive_component_seed


@pytest.fixture(autouse=True)
def plotting_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (24, 24), "#428579").save(output, format="PNG")
    return output.getvalue()


def seal_primary(directory, book, snapshot):
    snapshot_bytes = report._json_bytes(snapshot)
    if "plan17_retention_report" in book.metadata:
        book.metadata.plan17_retention_report.snapshot_sha256 = report._hash(snapshot_bytes)
    (directory / "report_snapshot.json").write_bytes(snapshot_bytes)
    nbformat.write(book, directory / "report.ipynb")
    manifest = {"schema": 1, "kind": "plan17-retention-report", "study_id": "primary-study", "smoke": False,
                "report_dir": str(directory), "snapshot_sha256": report._hash(snapshot_bytes),
                "files_sha256": {p.name: report._hash(p.read_bytes()) for p in directory.iterdir()
                                 if p.name not in ("manifest.json", "COMPLETED")}}
    (directory / "manifest.json").write_bytes(report._json_bytes(manifest))
    (directory / "COMPLETED").write_text(report._hash(report._json_bytes(manifest)) + "\n")


@pytest.fixture
def primary(tmp_path):
    directory = tmp_path / "primary" / "report" / "generation-primary"
    directory.mkdir(parents=True)
    cells = [nbformat.v4.new_markdown_cell("# Original report", id="plan17-title"),
             nbformat.v4.new_markdown_cell("## Original Progress\n\nOriginal complete summary; pending Phase 5 notice preserved.", id="plan17-progress"),
             nbformat.v4.new_markdown_cell("## Primary Progress\n\nCompleted Phase 5.", id="plan17-retention-progress")]
    for name, title, prefix in [(n, t, "plan17-") for n, t in primary_report.HISTORICAL_FIGURES] + [(n, t, "plan17-retention-") for n, t in primary_report.FIGURES]:
        png = png_bytes()
        (directory / (name + ".png")).write_bytes(png)
        cells.append(nbformat.v4.new_code_cell(
            f"from IPython.display import Image, display\ndisplay(Image(filename='preserved/{name}.png'))",
            id=prefix + name, execution_count=7, metadata={"preserve": [1, 2, 3]},
            outputs=[nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png).decode(), "text/plain": title})]))
    cells.extend([nbformat.v4.new_markdown_cell("## Original Conclusion\n\nExact historical conclusion, including old intervals.", id="plan17-conclusion"),
                  nbformat.v4.new_markdown_cell("## Phase 5 Conclusion\n\nExact primary conclusion and intervals.", id="plan17-retention-conclusion"),
                  nbformat.v4.new_markdown_cell("[Primary artifacts](preserved/analysis.json)", id="plan17-retention-artifacts")])
    book = nbformat.v4.new_notebook(cells=cells, metadata={"plan17_report": {"schema": 1, "study_id": "original-v4", "smoke": False},
        "plan17_retention_report": {"schema": 1, "study_id": "primary-study", "smoke": False, "status": "completed", "generation": directory.name},
        "custom_preserved": {"value": 17}})
    snapshot = {"schema": 1, "kind": "plan17-retention-snapshot", "status": "completed", "study_id": "primary-study"}
    (directory / "analysis.json").write_text("{}")
    seal_primary(directory, book, snapshot)
    return directory, nbformat.read(directory / "report.ipynb", as_version=4)


def selection(n=2):
    protocol, seed = "plan17-retention-calibration-v1", 171102
    return {"schema": 1, "protocol": "plan17-zero-ridge-confirmation-v1", "frozen": True,
        "frozen_at": "2026-11-02T12:00:00Z", "arms": deepcopy(analysis.ARMS), "main_replicas": n,
        "stationary_replicas": 0, "fresh_trajectory_target": 3 * n, "bootstrap_seed": 837,
        "config_hash": "scientific-config", "sources_hash": "scientific-sources",
        "data_partition": {"protocol": protocol, "seed": seed,
            **{split + "_seed": derive_component_seed(seed, f"{protocol}:study-partition:0:{split}") for split in ("train", "test")}},
        "schedule_factors": [1., 1.25]}


def trajectory(arm="fixed_ridge", replica=0, *, completed=True):
    spec = analysis.ARMS[arm]
    factor = (1., 1.25)[replica % 2]
    delta = {"fixed_ridge": 0, "fixed_zero": .015, "kalman_zero": .025}[arm] * (replica + 1)
    rows = [dict(step=arrivals // 4, arrivals=arrivals * factor, p=.005 if i < 2 else .1,
        segment="low_hold" if i < 2 else "rise", expected_nines=arrivals * factor / 10, observed_nines=int(arrivals * factor / 4),
        nine_recall=.2 + .1 * i + delta * i, old_accuracy=.9 - .02 * i + delta * i / 10,
        frozen_nine_recall=.2, frozen_old_accuracy=.9, initial_nines=3, wall_seconds=float(arrivals * (1 if arm == "fixed_ridge" else 2)))
        for i, arrivals in enumerate((0, 4, 12, 20))]
    if not completed:
        rows = rows[:2]
    return dict(unit_id=f"supplement-{replica}-{arm}", arm=arm, **spec, c=.025, stage="fresh", cohort="fresh", schedule="main",
        replica=replica, schedule_factor=factor, stream_hash=f"stream-{replica}", initial_unit=f"initial-{replica}", asset_unit=f"asset-{replica}",
        completed=completed, metrics=rows, summary={"costs": {"solver_seconds": 3., "curvature_seconds": 4.,
            "controller_seconds": 1. if arm == "kalman_zero" else 0., "checkpoint_seconds": 1., "evaluation_seconds": 2.}})


def snapshot(primary_dir=None, rows=None, *, smoke=False, n=2):
    return {"schema": 1, "kind": report.KIND, "study_id": "zero-ridge-study", "status": "running", "phase": "5E: supplemental confirmation",
        "config": {"smoke": smoke, "bootstrap_draws": 30, "seed": 171103}, "selection": selection(n), "trajectories": rows or [], "analysis": {},
        "progress": {"active_hours": 12.5, "allocation_hours": 20, "forecast_seconds": 3600,
            "active_unit": "next-unit", "last_accepted_checkpoint": "accepted-3", "step": 3, "total_steps": 20,
            "planned_trajectories": 3 * n, "heartbeat": "2026-11-02T12:34:56Z",
            "storage": {"used_bytes": 1e9, "external_bytes": 2e9, "reserved_bytes": 1e9, "max_bytes": 500e9, "filesystem_free_bytes": 30e9}},
        "historical_report_dir": str(primary_dir) if primary_dir else None,
        "historical_notebook_sha256": report._hash((primary_dir / "report.ipynb").read_bytes()) if primary_dir else None}


def write_snapshot(root, value):
    root.mkdir(parents=True, exist_ok=True)
    (root / "report_snapshot.json").write_text(json.dumps(value))


def read_book(path):
    return nbformat.read(path, as_version=4)


def prose(book):
    return "\n\n".join(c.source for c in book.cells if c.cell_type == "markdown")


@pytest.fixture
def fast_renderers(monkeypatch):
    calls = []
    def render(snapshot, result, path):
        calls.append(path.name)
        path.write_bytes(png_bytes())
    monkeypatch.setattr(report, "_effects", render)
    monkeypatch.setattr(report, "_accuracy", render)
    return calls


def test_preserve_all_primary_cells_images_conclusions_links_and_metadata(tmp_path, primary, fast_renderers, monkeypatch):
    directory, original = primary
    root, notebook = tmp_path / "supplement", tmp_path / "results.ipynb"
    notebook.write_bytes((directory / "report.ipynb").read_bytes())
    value = snapshot(directory)
    write_snapshot(root, value)
    before = {p: p.read_bytes() for p in directory.iterdir()}
    def forbidden(*args, **kwargs):
        pytest.fail("the primary report must never be regenerated")
    monkeypatch.setattr(primary_report, "build_report", forbidden)
    for name in ("_development", "_fresh_accuracy", "_paired_effects", "_historical"):
        monkeypatch.setattr(primary_report, name, forbidden)
    manifest = report.build_report(root, notebook)
    book = read_book(notebook)
    nbformat.validate(book)
    assert book.cells[1].id == "plan17-zero-ridge-progress"
    inherited = [c for c in book.cells if not c.id.startswith("plan17-zero-ridge-")]
    assert inherited == original.cells
    assert len([c for c in book.cells if c.cell_type == "code"]) == 11
    assert {k: v for k, v in book.metadata.items() if k != report.MARKER} == original.metadata
    assert book.metadata.plan17_zero_ridge_report.root == str(root)
    assert book.cells[len(original.cells) + 1].id == "plan17-zero-ridge-section"
    assert all(p.read_bytes() == data for p, data in before.items())
    assert fast_renderers == [name + ".png" for name, _ in report.FIGURES]
    assert report._verify_generation(Path(manifest["report_dir"])) == manifest
    assert json.loads((root / "report/latest.json").read_text()) == manifest
    assert json.loads((Path(manifest["report_dir"]) / "analysis.json").read_text())["bootstrap"]["seed"] == 837
    assert (Path(manifest["report_dir"]) / "report_snapshot.json").read_bytes() == (root / "report_snapshot.json").read_bytes()
    assert str(Path(analysis.__file__).resolve()) in manifest["sources_sha256"]


def test_pending_and_partial_blocks_remain_explicit_even_if_snapshot_says_completed(tmp_path, primary, fast_renderers):
    directory, _ = primary
    root, notebook = tmp_path / "supplement", tmp_path / "results.ipynb"
    rows = [trajectory("fixed_ridge"), trajectory("fixed_zero"), trajectory("kalman_zero", completed=False)]
    value = snapshot(directory, rows, n=8)
    value["status"] = "completed"
    write_snapshot(root, value)
    manifest = report.build_report(root, notebook)
    text = prose(read_book(notebook))
    assert manifest["status"] == "Partial / incomplete supplement"
    for fragment in ("0 / 8", "2 / 24; 1 incomplete", "Completed arms awaiting full block | 2", "next-unit / accepted-3",
                     "2026-11-02T12:34:56Z", "12.50 / 20.00", "4.00 / 500.00 / 30.00", "no complete fresh three-arm block",
                     "historical Phases 0--4 and primary Phase 5 snapshots"):
        assert fragment in text
    assert "| Kalman / r=0 - Fixed / r=.1 | 0 | pending" in text


def test_completed_primary_to_default_canonical_outside_primary_root_succeeds(tmp_path, primary, fast_renderers, monkeypatch):
    directory, original = primary
    root = tmp_path / "supplement"
    fake_source = tmp_path / "project/mnist_experiment/kalman_pi/retention/supplement/report.py"
    canonical = fake_source.parents[3] / "plan17_results.ipynb"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes((directory / "report.ipynb").read_bytes())
    sources = report._sources()
    monkeypatch.setattr(report, "_sources", lambda: sources)
    monkeypatch.setattr(report, "__file__", str(fake_source))
    value = snapshot(directory, [trajectory(a, r) for r in (0, 1) for a in report.ARMS])
    value["status"] = "completed"
    write_snapshot(root, value)
    before = {p: p.read_bytes() for p in directory.iterdir()}
    manifest = report.build_report(root)
    assert manifest["notebook"] == str(canonical)
    assert not canonical.is_relative_to(directory.parent.parent)
    assert manifest["status"] == "Completed supplement"
    assert [c for c in read_book(canonical).cells if not c.id.startswith("plan17-zero-ridge-")] == original.cells
    assert all(p.read_bytes() == data for p, data in before.items())


def test_completed_status_needs_every_intended_block(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    rows = [trajectory(a, r) for r in (0, 1) for a in report.ARMS]
    value = snapshot(rows=rows, smoke=True)
    value["status"] = "completed"
    write_snapshot(root, value)
    manifest = report.build_report(root)
    assert manifest["status"] == "Completed supplement"
    assert "2 / 2" in prose(read_book(manifest["notebook"]))


def test_idempotent_reuse_and_append_only_new_snapshot(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    old = {p: p.read_bytes() for p in Path(first["report_dir"]).iterdir()}
    assert report.build_report(root) == first
    assert len(fast_renderers) == 2
    value["trajectories"] = [trajectory(a) for a in report.ARMS]
    write_snapshot(root, value)
    second = report.build_report(root)
    assert second["report_dir"] != first["report_dir"]
    assert len(fast_renderers) == 4
    assert all(p.read_bytes() == data for p, data in old.items())


def test_source_hash_change_creates_new_generation_and_source_collision_is_rejected(tmp_path, fast_renderers, monkeypatch):
    root = tmp_path / "smoke"
    write_snapshot(root, snapshot(smoke=True))
    first = report.build_report(root)
    read = Path.read_bytes
    helper = Path(primary_report.__file__).resolve()
    monkeypatch.setattr(Path, "read_bytes", lambda path: read(path) + (b"\n# source identity probe\n" if path == helper else b""))
    monkeypatch.setattr(report, "_generation", lambda identity: Path(first["report_dir"]).name)
    before = Path(first["notebook"]).read_bytes(), (root / "report/latest.json").read_bytes()
    with pytest.raises(ValueError, match="source/identity collision"):
        report.build_report(root)
    assert (Path(first["notebook"]).read_bytes(), (root / "report/latest.json").read_bytes()) == before
    monkeypatch.setattr(report, "_generation", lambda identity: "generation-" + report._hash(report._json_bytes(identity))[:24])
    second = report.build_report(root)
    assert first["report_dir"] != second["report_dir"]


@pytest.mark.parametrize("corruption", ("schema", "kind", "seed", "nan", "cached_effect", "physical", "stationary", "development"))
def test_bad_snapshot_or_analysis_preserves_prior_publication(tmp_path, fast_renderers, corruption):
    root = tmp_path / "smoke"
    value = snapshot(rows=[trajectory(a, r) for r in (0, 1) for a in report.ARMS], smoke=True)
    write_snapshot(root, value)
    manifest = report.build_report(root)
    notebook, latest = Path(manifest["notebook"]), root / "report/latest.json"
    before = notebook.read_bytes(), latest.read_bytes()
    if corruption == "schema":
        value["schema"] = True
    elif corruption == "kind":
        value["kind"] = "plan17-retention-snapshot"
    elif corruption == "seed":
        value["selection"]["bootstrap_seed"] = None
    elif corruption == "nan":
        value["progress"]["active_hours"] = float("nan")
    elif corruption == "cached_effect":
        value["analysis"] = analysis.summarize(value["trajectories"], value["selection"], bootstrap_draws=30, seed=837)
        value["analysis"]["comparisons"][0]["observation_mean_pp"]["nine_recall"]["mean"] += 10
    elif corruption == "physical":
        value["trajectories"][1]["ridge"] = .1
    elif corruption == "stationary":
        value["trajectories"][0]["schedule"] = "stationary"
    else:
        value["trajectories"][0]["stage"] = "development"
    write_snapshot(root, value)
    with pytest.raises(ValueError):
        report.build_report(root)
    assert (notebook.read_bytes(), latest.read_bytes()) == before
    assert not list((root / "report").glob(".building-*"))


@pytest.mark.parametrize("corruption", ("not_completed", "smoke", "original_only", "already_supplement", "pinned_hash", "png", "relative"))
def test_history_must_pin_intact_completed_primary_not_original_or_other_phase(tmp_path, primary, fast_renderers, corruption):
    directory, original = primary
    root, notebook = tmp_path / "supplement", tmp_path / "results.ipynb"
    notebook.write_bytes((directory / "report.ipynb").read_bytes())
    value = snapshot(directory)
    if corruption in ("not_completed", "smoke", "original_only", "already_supplement"):
        book = deepcopy(original)
        sealed = json.loads((directory / "report_snapshot.json").read_text())
        if corruption == "not_completed":
            book.metadata.plan17_retention_report.status = "running"
        elif corruption == "smoke":
            book.metadata.plan17_retention_report.smoke = True
        elif corruption == "original_only":
            del book.metadata["plan17_retention_report"]
        else:
            book.metadata[report.MARKER] = {"schema": 1}
        # Resealing still cannot turn the wrong report phase into accepted history.
        seal_primary(directory, book, sealed)
        value["historical_notebook_sha256"] = report._hash((directory / "report.ipynb").read_bytes())
    elif corruption == "pinned_hash":
        value["historical_notebook_sha256"] = "wrong"
    elif corruption == "png":
        (directory / "08_retention_fresh.png").write_bytes(b"corrupt")
    else:
        value["historical_report_dir"] = "relative/path"
    write_snapshot(root, value)
    before = notebook.read_bytes()
    with pytest.raises((ValueError, KeyError)):
        report.build_report(root, notebook)
    assert notebook.read_bytes() == before
    assert not (root / "report/latest.json").exists()


@pytest.mark.parametrize("mutation", ("add_future_cell", "change_output", "another_owner"))
def test_canonical_guard_does_not_replace_newer_or_foreign_content(tmp_path, primary, fast_renderers, mutation):
    directory, _ = primary
    root, notebook = tmp_path / "supplement", tmp_path / "results.ipynb"
    write_snapshot(root, snapshot(directory))
    report.build_report(root, notebook)
    book = read_book(notebook)
    if mutation == "add_future_cell":
        book.cells.append(nbformat.v4.new_markdown_cell("Future phase evidence", id="future-phase"))
    elif mutation == "change_output":
        next(c for c in book.cells if c.cell_type == "code").outputs[0].data["text/plain"] = "newer revised result"
    else:
        book.metadata.plan17_zero_ridge_report.root = str(tmp_path / "other-owner")
    nbformat.write(book, notebook)
    before = notebook.read_bytes(), (root / "report/latest.json").read_bytes()
    with pytest.raises(ValueError, match="newer|owner"):
        report.build_report(root, notebook)
    assert (notebook.read_bytes(), (root / "report/latest.json").read_bytes()) == before


def test_changed_canonical_during_render_is_preserved(tmp_path, primary, fast_renderers, monkeypatch):
    directory, _ = primary
    root, notebook = tmp_path / "supplement", tmp_path / "results.ipynb"
    notebook.write_bytes((directory / "report.ipynb").read_bytes())
    write_snapshot(root, snapshot(directory))
    changed = []
    def render(snapshot, result, path):
        path.write_bytes(png_bytes())
        book = read_book(notebook)
        book.cells.append(nbformat.v4.new_markdown_cell("Newer content arrived during rendering", id="future-phase"))
        nbformat.write(book, notebook)
        changed.append(notebook.read_bytes())
    monkeypatch.setattr(report, "_effects", render)
    with pytest.raises(ValueError, match="newer"):
        report.build_report(root, notebook)
    assert notebook.read_bytes() == changed[0]
    assert not (root / "report/latest.json").exists()


def test_frozen_design_and_completed_evidence_cannot_regress(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    value = snapshot(rows=[trajectory(a) for a in report.ARMS], smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    original = deepcopy(value)
    value["selection"]["bootstrap_seed"] += 1
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="frozen supplemental selection changed"):
        report.build_report(root)
    original["trajectories"] = []
    write_snapshot(root, original)
    with pytest.raises(ValueError, match="regresses"):
        report.build_report(root)
    assert json.loads((root / "report/latest.json").read_text()) == first


def test_corrupt_generation_or_foreign_latest_not_repaired(tmp_path, fast_renderers):
    root = tmp_path / "smoke"
    write_snapshot(root, snapshot(smoke=True))
    first = report.build_report(root)
    notebook = Path(first["notebook"])
    before = notebook.read_bytes()
    (Path(first["report_dir"]) / "analysis.json").write_text("corrupt")
    with pytest.raises(ValueError, match="corrupt supplement"):
        report.build_report(root)
    assert notebook.read_bytes() == before
    first["report_dir"] = str(tmp_path / "another-root/generation")
    (root / "report/latest.json").write_text(json.dumps(first))
    with pytest.raises(ValueError, match="escapes"):
        report.build_report(root)


def test_publication_failure_rolls_back_latest_and_keeps_notebook(tmp_path, fast_renderers, monkeypatch):
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    notebook, latest = Path(first["notebook"]), root / "report/latest.json"
    before = notebook.read_bytes(), latest.read_bytes()
    value["progress"]["step"] += 1
    write_snapshot(root, value)
    atomic = primary_report._atomic
    def fail(path, data):
        if path == notebook:
            raise OSError("publication interrupted")
        atomic(path, data)
    monkeypatch.setattr(primary_report, "_atomic", fail)
    with pytest.raises(OSError, match="publication interrupted"):
        report.build_report(root)
    assert (notebook.read_bytes(), latest.read_bytes()) == before
    assert not list((root / "report").glob(".building-*"))


def test_smoke_is_root_only_self_contained_and_cannot_overwrite_primary(tmp_path, primary, fast_renderers):
    directory, _ = primary
    root = tmp_path / "smoke"
    value = snapshot(smoke=True)
    write_snapshot(root, value)
    first = report.build_report(root)
    book = read_book(first["notebook"])
    assert book.metadata.plan17_zero_ridge_report.smoke is True
    assert "SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE" in book.cells[0].source
    assert "support no production efficacy" in prose(book)
    assert "2 new changing replicas (6 paths)" in prose(book)
    assert len([c for c in book.cells if c.cell_type == "code"]) == 2
    outside = tmp_path / "outside.ipynb"
    with pytest.raises(ValueError, match="inside their supplemental root"):
        report.build_report(root, outside)
    assert not outside.exists()
    value["historical_report_dir"] = str(directory)
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="smoke must not use"):
        report.build_report(root)


def test_production_cannot_write_primary_root_or_overwrite_smoke(tmp_path, primary, fast_renderers):
    directory, _ = primary
    root = tmp_path / "supplement"
    write_snapshot(root, snapshot(directory))
    before = (directory / "report.ipynb").read_bytes()
    with pytest.raises(ValueError, match="primary root"):
        report.build_report(root, directory / "report.ipynb")
    assert (directory / "report.ipynb").read_bytes() == before
    smoke_root = tmp_path / "smoke"
    write_snapshot(smoke_root, snapshot(smoke=True))
    smoke = report.build_report(smoke_root)
    with pytest.raises(ValueError, match="another supplement owner"):
        report.build_report(root, Path(smoke["notebook"]))


@pytest.fixture
def captured_plots(monkeypatch):
    recorded = []
    figure = report._figure
    @contextmanager
    def capture(*args, **kwargs):
        with figure(*args, **kwargs) as (fig, axes):
            yield fig, axes
            recorded.append([{"title": ax.get_title(), "xlim": ax.get_xlim(), "ylim": ax.get_ylim(), "patches": len(ax.patches),
                "texts": [t.get_text() for t in ax.texts],
                "lines": [{"label": line.get_label(), "x": np.asarray(line.get_xdata()).copy(), "y": np.asarray(line.get_ydata()).copy()} for line in ax.lines]} for ax in axes.flat])
    monkeypatch.setattr(report, "_figure", capture)
    return recorded


def test_effect_panels_table_show_three_distinct_contrasts_two_metrics_and_measured_costs(tmp_path, captured_plots):
    rows = [trajectory(a, r) for r in (0, 1) for a in report.ARMS]
    value = snapshot(rows=rows)
    result = analysis.summarize(rows, value["selection"], bootstrap_draws=30, seed=837)
    report._effects(value, result, tmp_path / "effects.png")
    axes = captured_plots[-1]
    assert len(axes) == 2 and all(ax["patches"] == 0 for ax in axes)  # No retention-matching band.
    for ax, metric in zip(axes, ("nine_recall", "old_accuracy")):
        points = [line["x"][0] for line in ax["lines"] if len(line["x"]) == 1]
        assert points == pytest.approx([c["observation_mean_pp"][metric]["mean"] for c in result["comparisons"]])
    text = report._results(result).source
    assert "Kalman / r=0 - Fixed / r=.1 | 2" in text
    assert "Fixed / r=0 - Fixed / r=.1 | 2" in text
    assert "Kalman / r=0 - Fixed / r=0 | 2" in text
    assert "95% CI" in text and "Learning time ratio" in text
    assert "Online wall ratio" in text and "means of paired path ratios" in text


def test_retention_gain_with_uncertain_acquisition_is_not_called_joint_gain():
    rows = [trajectory(arm, replica) for replica in (0, 1) for arm in report.ARMS]
    value = snapshot(rows=rows, smoke=True)
    result = analysis.summarize(rows, value["selection"], bootstrap_draws=30, seed=837)
    comparison = next(c for c in result["comparisons"] if c["condition"] == "kalman_zero" and c["control"] == "fixed_zero")
    comparison["observation_mean_pp"]["nine_recall"].update(mean=.03, ci95=[-2.37, 2.28])
    comparison["observation_mean_pp"]["old_accuracy"].update(mean=2.11, ci95=[1.56, 2.58])
    text = report._interpretation(result)
    assert "old-accuracy interval favors the first arm" in text
    assert "9-recall gain is not established" in text
    assert "equivalence is not established" in text
    result["comparisons"][0]["segments"]["rise"]["observation_mean_pp"]["nine_recall"]["mean"] = -2
    assert "Contrary regime effects: rise: nine_recall" in report._interpretation(result)


def test_accuracy_panels_separate_length_cohorts_and_withhold_partial_blocks(tmp_path, captured_plots):
    rows = [trajectory(a, r, completed=not (a == "kalman_zero" and r == 1)) for r in (0, 1) for a in report.ARMS]
    value = snapshot(rows=rows)
    result = analysis.summarize(rows, value["selection"], bootstrap_draws=30, seed=837)
    report._accuracy(value, result, tmp_path / "partial.png")
    axes = captured_plots[-1]
    assert len(axes) == 2 and axes[0]["xlim"] == axes[1]["xlim"]
    assert all(ax["ylim"] == (0, 100) for ax in axes)
    paths = [line for line in axes[0]["lines"] if not line["label"].startswith("Initial")]
    assert len(paths) == 3 and all(line["x"][-1] == 20 for line in paths)
    assert any("online 9 exp/obs=" in text for text in axes[0]["texts"])
    rows[-1] = trajectory("kalman_zero", 1)
    result = analysis.summarize(rows, value["selection"], bootstrap_draws=30, seed=837)
    report._accuracy(value, result, tmp_path / "full.png")
    paths = [line for line in captured_plots[-1][0]["lines"] if not line["label"].startswith("Initial")]
    assert len(paths) == 6
    assert {tuple(line["x"]) for line in paths} == {(0, 4, 12, 20), (0, 5, 15, 25)}


def test_real_figures_nonblank_portable_and_scientific_units_untouched(tmp_path):
    root = tmp_path / "smoke"
    value = snapshot(rows=[trajectory(a, r) for r in (0, 1) for a in report.ARMS], smoke=True)
    value["analysis"] = analysis.summarize(value["trajectories"], value["selection"], bootstrap_draws=30, seed=837)
    write_snapshot(root, value)
    unit = root / "units/complete"
    unit.mkdir(parents=True)
    (unit / "checkpoint.pt").write_bytes(b"never load this checkpoint")
    (unit / "COMPLETED").write_text("immutable")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    manifest = report.build_report(root)
    for c in read_book(manifest["notebook"]).cells:
        if c.cell_type == "code":
            data = base64.b64decode(c.outputs[0].data["image/png"])
            picture = Image.open(io.BytesIO(data)).convert("RGB")
            assert picture.width > 1000 and picture.height > 600
            assert np.asarray(picture).std() > 10
            assert (Path(manifest["report_dir"]) / (c.id.removeprefix("plan17-zero-ridge-") + ".png")).read_bytes() == data
    assert all(p.read_bytes() == data for p, data in before.items())


def test_subprocess_publication_never_imports_torch_training_data_or_family_lock(tmp_path):
    root = tmp_path / "smoke"
    write_snapshot(root, snapshot(smoke=True))
    script = """
import importlib.abc
import sys
class NoCompute(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args, **kwargs):
        if fullname == 'torch' or fullname.startswith(('torch.', 'torchvision')) or fullname == 'mnist_experiment.fisher_memory.artifacts' or 'kalman_pi' in fullname and fullname.rsplit('.', 1)[-1] in ('run', 'runtime', 'trajectory', 'data', 'policies', 'observations'):
            raise AssertionError('forbidden compute import: ' + fullname)
sys.meta_path.insert(0, NoCompute())
from pathlib import Path
from mnist_experiment.kalman_pi.retention.supplement.report import build_report
assert build_report(Path(sys.argv[1]))['smoke'] is True
assert 'torch' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", script, str(root)], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
