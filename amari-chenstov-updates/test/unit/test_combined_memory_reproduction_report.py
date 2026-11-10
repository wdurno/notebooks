"""CPU-only scalar presentation and immutable Phase 8 publication contracts."""

import base64
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
from PIL import Image
import pytest

from mnist_experiment.combined_memory.reproduction import report


def test_native_fidelity_and_completed_closeout_are_not_labelled_pending():
    snapshot = dict(progress=dict(status="completed",forecast_hours=.5), fidelity=dict(status="passed", source_files=69,
        statistics=dict(status="passed", checks=218), same_seed_paths=[dict(status="passed",
            historical_unit="all_r2-trajectory-confirmation", checks=[dict(max_absolute=0.)])]))
    text = "\n".join(c["source"] for c in report.build_cells(snapshot))
    assert "69 frozen files" in text and "218 scalar checks" in text
    assert "R2 same-seed trajectory" in text and "max absolute discrepancy 0" in text
    assert "remaining forecast: 0 h" in text
    conclusion = next(c["source"] for c in report.build_cells(snapshot) if c["id"] == report.PREFIX+"conclusion")
    assert "driver verified" in conclusion and "no completed closeout" not in conclusion


class Store:
    def __init__(self, root):
        self.root, self.reservations = root, []

    @contextmanager
    def reserve(self, size):
        self.reservations.append(size)
        yield


def pending():
    return dict(progress=dict(stage="8A fidelity", status="pending", counts=dict(
        blurred=0, no_blur=0, fresh_blocks=0), charged_hours=.25, forecast_hours=7.5),
        fidelity=dict(status="pending"), settings=[], analysis=None)


def effect(mean, radius=.1, **extra):
    return dict(mean=mean, ci=[mean - radius, mean + radius], **extra)


def native_analysis():
    """The scalar boundary exported by Mill's analyze, without running bootstrap."""
    result = dict(protocol=report.PROTOCOL, schema_version=1, namespace="fresh", N=3,
        replicas=[0, 1], is_final=False, provisional=True,
        cohort=dict(partial=True, excluded_blur_blocks=[dict(replica=2, missing=["F1"])],
                    pending_no_blur_blocks=[]), curves={}, summaries={}, contrasts={}, efficiency={})
    for context in ("blur", "no_blur"):
        result["curves"][context], result["summaries"][context] = {}, {}
        for policy, final in (("R2", 8.), ("R1", 7.), ("F1", 4.)):
            if context == "no_blur" and policy == "R2":
                continue
            risk = [16., 12., 9., final] if context == "blur" else [4., 3., 2., 1.]
            curve = dict(observations=[0., 12., 25., 40.],
                query=dict(mean_mse=risk, ci_mse=[[v * .9 for v in risk], [v * 1.1 for v in risk]]),
                base=dict(mean_mse=[4., 4.1, 4.2, 4.3]))
            if context == "blur":
                curve["n"] = [0, 1, 2, 3]
            result["curves"][context][policy] = curve
            result["summaries"][context][policy] = dict(frozen_mse=effect(risk[0]), retention=effect(.07),
                retired_training_cohort=dict(n=[1, 2, 3], mean_mse=[4.5, 4.6, 4.7]))
    for name in ("F1-R2", "F1-R1"):
        result["contrasts"][name] = dict(acquisition_gain=effect(2., .5, favorable_replicas=2),
            endpoint_risk_difference=effect(-4.), retention_difference=effect(-.02))
    result["efficiency"]["F1"] = dict(point=dict(status="valid", saved_fraction=.414,
        learning_time_ratio=.684, S=.605, matching_budget=58.6, useful_target=True),
        equal_budget_learning_time_ratio=1.2, equal_budget_learning_time_ratio_ci=[1.1, 1.3],
        full_budget_retention_supported=True, matched_budget_retention_supported=False,
        bootstrap=dict(invalid=0, draws=20, ci=dict(saved_fraction=[.344, .44],
            learning_time_ratio=[.6, .8], S=[.5, .7])))
    result["blur_interaction"] = dict(status="available", delta_blur=effect(-2.), delta_no_blur=effect(-.2),
        interaction=effect(-1.8), endpoint_differences=dict(blur=effect(-3.), no_blur=effect(-.1)))
    return result


def completed():
    snapshot = pending()
    snapshot.update(analysis=native_analysis(), fidelity=dict(status="passed", checks={
        "Historical scalar headline": dict(status="passed", href="audit/statistics.json"),
        "Same-seed replay": dict(status="failed", href="audit/first-divergence.json")}),
        settings=[dict(policy="F1", ridge=1., strength=3.34056320604949)])
    snapshot["progress"].update(status="running", stage="8B fresh", counts=dict(blurred=6, no_blur=4, fresh_blocks=2))
    return snapshot


def test_final_outcome_reports_actual_savings_retention_and_blur_uncertainty():
    data = native_analysis()
    data["is_final"] = True
    text = report._outcome(data)
    assert "41.4 [34.4, 44]%" in text and "S=0.605 [0.5, 0.7]" in text
    assert "Retention qualification is not established" in text
    assert "relative risk effect is more favorable in the blurred context" in text
    data["blur_interaction"]["interaction"] = effect(0., .2)
    assert "blur dependence remains unresolved" in report._outcome(data)
    data["efficiency"]["F1"]["bootstrap"]["invalid"] = 1
    assert "41.4 [34.4, 44]" not in report._outcome(data)
    assert "inverse CI is withheld" in report._outcome(data)


def test_missing_inverse_bound_is_not_presented_as_a_retention_failure():
    data = native_analysis()
    data["efficiency"]["F1"]["bootstrap"]["invalid"] = 1
    text = report._efficiency(data)["source"]
    assert "supported / not estimable" in text
    assert "passed / failed" not in text
    data["efficiency"]["F1"]["point"]["target_mse_gain_fraction"] = .077
    assert "Reference improvement from initialization:** 7.7%" in report._efficiency(data)["source"]


def prose(cells):
    return "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in cells)


def write_book(path):
    history = [nbformat.v4.new_markdown_cell("# Plan 18", id="original-title"),
        nbformat.v4.new_code_cell("raise AssertionError('historical cell must not execute')", id="original-output",
            execution_count=4, metadata={"custom": [1, 2]}, outputs=[nbformat.v4.new_output(
                "stream", name="stdout", text="preserved output\n")]),
        nbformat.v4.new_markdown_cell("## Phase 7: Preserved", id="original-phase7")]
    history[1]["source"] = ["raise AssertionError('historical cell must not execute')\n"]
    history[2]["attachments"] = {"old.txt": {"text/plain": "historical attachment"}}
    book = dict(nbformat.v4.new_notebook(cells=history, metadata={"custom": {"nested": [1, 2]}}))
    path.write_text(json.dumps(book))
    return json.loads(path.read_bytes())


def generations(store):
    return store.root / "report/generations"


@pytest.fixture
def fast_figures(monkeypatch):
    def plot(result, view, href):
        return nbformat.v4.new_code_cell("# Scalar plot stub", id=report.PREFIX + view + "-plot", metadata={report.TAG: True})

    monkeypatch.setattr(report, "_plot_cell", plot)


def test_pending_is_safe_without_analysis_and_uses_exact_phase8_tag():
    snapshot = pending()
    before = deepcopy(snapshot)
    cells = report.build_cells(snapshot)
    text = prose(cells)
    for expected in ("Phase 8 Progress", "0/8", "0/24", "0/16", "0.25 h", "7.5 h", "Analysis Pending",
                     "No inference", "does not train", "Historical Fidelity", "F1", "different ridges"):
        assert expected in text
    assert all(c["cell_type"] == "markdown" and c["metadata"][report.TAG] is True for c in cells)
    assert snapshot == before
    assert report.owned(dict(metadata={report.TAG: True}))
    assert report.owned(dict(id=report.PREFIX + "intro"))
    assert not report.owned(dict(id="plan18-phase80-intro", metadata={report.TAG: False}))
    assert not report.owned(dict(id="plan18-phase7-intro", metadata={"plan18_phase7": True}))


@pytest.mark.parametrize("analysis", [None, {}])
def test_minimal_snapshot_can_show_progress_and_fidelity_alone(analysis):
    cells = report.build_cells(dict(progress=None, fidelity={"status": "failed"}, settings=None, analysis=analysis))
    assert "Analysis Pending" in prose(cells)
    assert "**failed**" in prose(cells)


def test_runtime_progress_aliases_and_settings_are_defensive():
    snapshot = pending()
    snapshot["progress"].update(counts=dict(blur=6, no_blur=2), fresh_blocks_completed=1,
        blurred_completed=6, no_blur_completed=2)
    snapshot["settings"] = [dict(policy="F1", ridge=1), ["Archive", "rank 64"], "frozen policy"]
    text = prose(report.build_cells(snapshot))
    assert all(value in text for value in ("1/8", "6/24", "2/16", "rank 64", "frozen policy"))
    snapshot["settings"] = {"Archive": "rank 64"}
    assert "rank 64" in prose(report.build_cells(snapshot))
    snapshot["fidelity"] = dict(status="passed", smoke_only=True)
    assert "not production acceptance" in prose(report.build_cells(snapshot))
    snapshot["progress"].update(counts=dict(blurred=6, no_blur=0, fresh_blocks=2), fresh_blocks_completed=0)
    assert "2/8" in prose(report.build_cells(snapshot))


def test_native_three_policy_analysis_renders_two_nonblank_inline_figures_and_correct_sign():
    snapshot = completed()
    before = deepcopy(snapshot)
    cells = report.build_cells(snapshot)
    text = prose(cells)
    assert snapshot == before
    assert "-2 [-2.5, -1.5]" in text
    assert "41.4 [34.4, 44]" in text
    assert "0.605 [0.5, 0.7]" in text
    assert "2/2" in text
    assert "failed" in text and "audit/first-divergence.json" in text
    assert "no novel-sample savings or S is calculated" in text
    assert "not a pure training-blur causal effect" in text
    assert "No blur: F1-R1" in text
    plots = [c for c in cells if c["cell_type"] == "code"]
    assert len(plots) == 2
    for cell in plots:
        assert "load_analysis" in cell["source"] and "expected_sha256" in cell["source"]
        assert cell["execution_count"] == 1
        png = base64.b64decode(cell["outputs"][0]["data"]["image/png"])
        pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        assert pixels.shape[0] > 300 and pixels.shape[1] > 900 and pixels.std() > 15


def test_figure_clocks_labels_legends_and_native_frozen_references_are_explicit():
    result = native_analysis()
    figure = report._figure(result, "blurred")
    learning, retention = figure.axes
    assert learning.get_xlabel() == "Designated novel grids"
    assert set(learning.get_legend_handles_labels()[1]) == set(report.LABELS.values()) | {"Frozen initialization"}
    assert list(learning.lines[0].get_xdata()) == [0, 1, 2, 3]
    assert list(learning.lines[-1].get_ydata()) == [16., 16.]
    assert list(retention.lines[-1].get_ydata()) == [2., 2.]
    figure.clear()
    figure = report._figure(result, "blur_contrast")
    assert len(figure.axes) == 4
    for axis in figure.axes:
        assert axis.get_xlabel() == "Total integrated grids"
        assert not any("R2" in label for label in axis.get_legend_handles_labels()[1])
        assert list(axis.lines[0].get_xdata()) == [0., 12., 25., 40.]
    assert "No-blur stationary control" in figure.axes[2].get_title()
    figure.clear()


def test_any_invalid_inverse_draw_withholds_only_inverse_intervals(fast_figures):
    snapshot = completed()
    snapshot["analysis"]["efficiency"]["F1"]["bootstrap"]["invalid"] = 1
    text = prose(report.build_cells(snapshot))
    assert "1/20; CI withheld" in text
    assert "41.4 [34.4, 44]" not in text
    assert "-2 [-2.5, -1.5]" in text
    assert "1.2 [1.1, 1.3]" in text
    snapshot["analysis"]["efficiency"]["F1"]["point"]["status"] = "target_not_reached_at_observed_endpoint"
    text = prose(report.build_cells(snapshot))
    assert "| F1-R2 headline | unresolved | unresolved | unresolved |" in text


@pytest.mark.parametrize("change", [dict(protocol="wrong"), dict(schema_version=2), dict(namespace="historical")])
def test_incompatible_or_historical_analysis_fails_clearly(change):
    with pytest.raises(ValueError):
        report.build_cells(dict(analysis={**native_analysis(), **change}))


@pytest.mark.parametrize("bad", [[1, 2], float("nan"), float("inf"), object()])
def test_nonfinite_or_nonmapping_analysis_is_rejected(bad):
    with pytest.raises(ValueError):
        report.build_cells(dict(analysis=bad))


def test_malformed_clocks_and_unmatched_summaries_are_not_silently_dropped():
    result = native_analysis()
    result["curves"]["blur"]["F1"]["n"] = [0, 1, 1, 3]
    with pytest.raises(ValueError, match="curve"):
        report._validate_analysis(result)
    with pytest.raises(ValueError, match="paired"):
        report._validate_analysis(dict(blurred=dict(pairing_complete=False)))
    result = native_analysis()
    del result["curves"]["blur"]["F1"]
    with pytest.raises(ValueError, match="paired"):
        report._validate_analysis(result)


def test_previous_analysis_protocol_is_readable_without_changing_shared_protocol(fast_figures):
    snapshot = completed()
    snapshot["analysis"]["protocol"] = report.ANALYSIS_PROTOCOL
    assert "-2 [-2.5, -1.5]" in prose(report.build_cells(snapshot))
    assert report.PROTOCOL == "plan18-stm-reproduction-v1"


def test_publication_preserves_all_history_and_later_user_cells_and_is_idempotent(tmp_path):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    before = write_book(notebook)
    first = report.publish(pending(), notebook, store)
    book = json.loads(notebook.read_bytes())
    assert [c for c in book["cells"] if not report.owned(c)] == before["cells"]
    assert book["metadata"] == before["metadata"]
    assert book["cells"][1]["id"] == report.PREFIX + "progress"
    assert next(i for i, c in enumerate(book["cells"]) if c["id"] == report.PREFIX + "intro") > 2
    generation = Path(first["path"])
    files = {str(p.relative_to(generation)): p.read_bytes() for p in generation.rglob("*") if p.is_file()}
    assert files["results.ipynb"] == notebook.read_bytes()
    assert json.loads(files["snapshot.json"]) == pending()
    assert json.loads(files["analysis.json"]) == {}
    assert (generation / "COMPLETED").is_file()
    assert sum(store.reservations) == sum(map(len, files.values())) + notebook.stat().st_size + (store.root / "report/latest.json").stat().st_size
    second = report.publish(pending(), notebook, store)
    assert second["changed"] is False and second["generation"] == first["generation"]
    assert len(list(generations(store).iterdir())) == 1
    assert json.loads(notebook.read_bytes()) == book
    later = nbformat.v4.new_markdown_cell("Later user notes", id="later-user")
    book["cells"].append(later)
    notebook.write_text(json.dumps(book))
    report.publish(pending(), notebook, store)
    assert json.loads(notebook.read_bytes())["cells"][-1] == later
    assert {str(p.relative_to(generation)): p.read_bytes() for p in generation.rglob("*") if p.is_file()} == files


def test_changed_snapshot_creates_new_generation_without_mutating_earlier_artifacts(tmp_path):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    old = report.publish(pending(), notebook, store)
    path = Path(old["path"])
    saved = {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    snapshot = pending()
    snapshot["progress"]["charged_hours"] += 1
    new = report.publish(snapshot, notebook, store)
    assert new["generation"] != old["generation"]
    assert len(list(generations(store).iterdir())) == 2
    assert {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()} == saved
    pointer = json.loads((store.root / "report/latest.json").read_bytes())
    assert pointer["path"] == f"generations/{new['generation']}"


def test_analysis_loader_validates_scalar_export_and_checksum(tmp_path, fast_figures):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    result = report.publish(completed(), notebook, store)
    analysis = Path(result["path"]) / "analysis.json"
    checksum = hashlib.sha256(analysis.read_bytes()).hexdigest()
    assert report.load_analysis(analysis, checksum) == native_analysis()
    with pytest.raises(ValueError, match="checksum"):
        report.load_analysis(analysis, "0" * 64)
    analysis.write_text('{"schema_version":2}')
    with pytest.raises(ValueError, match="schema"):
        report.load_analysis(analysis)


@pytest.mark.parametrize("filename", ["analysis.json", "snapshot.json", "results.ipynb", "report_source.py"])
def test_completed_generation_corruption_is_never_repaired(tmp_path, filename):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    result = report.publish(pending(), notebook, store)
    path = Path(result["path"]) / filename
    path.write_bytes(b"tampered\n")
    before = notebook.read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        report.publish(pending(), notebook, store)
    assert path.read_bytes() == b"tampered\n" and notebook.read_bytes() == before


@pytest.mark.parametrize("stage", ["staging", "public-notebook", "pointer"])
def test_interruption_can_resume_without_mutating_completed_generations(tmp_path, monkeypatch, stage):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    original = report._atomic

    def interrupt(path, data):
        if ((stage == "staging" and path.name == "analysis.json")
                or (stage == "public-notebook" and path == notebook)
                or (stage == "pointer" and path.name == "latest.json")):
            raise KeyboardInterrupt("simulated interruption")
        return original(path, data)

    monkeypatch.setattr(report, "_atomic", interrupt)
    with pytest.raises(KeyboardInterrupt):
        report.publish(pending(), notebook, store)
    assert not list(generations(store).glob(".tmp-*"))
    assert not (store.root / "report/latest.json").exists()
    sealed = {p.name: (p / "COMPLETED").read_bytes() for p in generations(store).iterdir()}
    monkeypatch.setattr(report, "_atomic", original)
    result = report.publish(pending(), notebook, store)
    assert result["changed"] == (stage == "staging")
    for name, content in sealed.items():
        assert (generations(store) / name / "COMPLETED").read_bytes() == content


def test_publisher_lease_bounds_symlinks_and_frozen_notebooks(tmp_path, monkeypatch):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    with report._lease(store.root / "report"):
        with pytest.raises(ValueError, match="lease"):
            report.publish(pending(), notebook, store)
    notebook.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        report.publish(pending(), notebook, store)
    notebook.unlink()
    notebook.write_text(json.dumps(dict(nbformat.v4.new_notebook(metadata={
        "plan15_phase8_presentation": {"role": "frozen_archive"}}))))
    before = notebook.read_bytes()
    with pytest.raises(ValueError, match="frozen"):
        report.publish(pending(), notebook, store)
    assert notebook.read_bytes() == before
    monkeypatch.setattr(report, "MAX_NOTEBOOK_BYTES", 2)
    with pytest.raises(ValueError, match="oversized"):
        report.publish(pending(), notebook, store)


def test_no_store_uses_ignored_default_root_and_size_guard_runs_before_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "DEFAULT_ROOT", tmp_path / "cache")
    report.publish(pending(), tmp_path / "results.ipynb")
    assert (tmp_path / "cache/report/latest.json").is_file()
    monkeypatch.setattr(report, "MAX_JSON_BYTES", 2)
    with pytest.raises(ValueError, match="size"):
        report.publish(pending(), tmp_path / "never-written.ipynb")
    assert not (tmp_path / "never-written.ipynb").exists()


def test_import_and_pending_publication_do_not_import_training_or_analysis(tmp_path):
    source = """
import sys
from pathlib import Path
from types import SimpleNamespace
from mnist_experiment.combined_memory.reproduction import report
report.publish({'progress': {}, 'fidelity': {}}, Path(sys.argv[1]), SimpleNamespace(root=Path(sys.argv[2])))
assert 'torch' not in sys.modules
assert 'mnist_experiment.combined_memory.reproduction.analysis' not in sys.modules
assert 'mnist_experiment.combined_memory.reproduction.runtime' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source, str(tmp_path / "results.ipynb"), str(tmp_path / "cache")],
                   check=True, capture_output=True, text=True)
