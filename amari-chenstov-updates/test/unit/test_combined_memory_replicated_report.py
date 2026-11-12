"""CPU-only Phase 9 scalar presentation and publication contracts."""

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

from mnist_experiment.combined_memory.replicated_combination import report


class Store:
    def __init__(self, root):
        self.root, self.reservations = root, []

    @contextmanager
    def reserve(self, size):
        self.reservations.append(size)
        yield


def effect(mean, radius=.01):
    return dict(mean=mean, ci=[mean - radius, mean + radius])


def pending():
    return dict(protocol=report.PROTOCOL, schema_version=1,
        progress=dict(stage="9A implementation", status="programming", current="report acceptance",
            fresh_completed=0, fresh_blocks_completed=0, charged_hours=.1, remaining_hours=19.9,
            forecast_hours=15., elapsed_hours=.15), fidelity=dict(status="pending"), analysis=None)


def analysis():
    """Native six-policy analyzer boundary, including its exact effect keys."""
    result = dict(protocol=report.PROTOCOL, schema_version=1, namespace="fresh", N=3,
        replicas=[0, 1], is_final=False, provisional=True, curves={}, summaries={}, contrasts={},
        efficiency={}, efficiency_vs_B={}, retention_comparisons={})
    for index, policy in enumerate(report.POLICIES):
        risk = [16., 13. - .2 * index, 11. - .2 * index, 9. - .2 * index]
        result["curves"][policy] = dict(n=[0, 1, 2, 3], observations=[0., 12., 25., 40.],
            query=dict(mean_mse=risk, ci_mse=[[v * .9 for v in risk], [v * 1.1 for v in risk]]),
            base=dict(mean_mse=[4., 4.01, 4.02, 4.03]))
        result["summaries"][policy] = dict(acquisition=effect(12. - .2 * index),
            retention=effect(.003), endpoint_retention=effect(.007), endpoint_mse=effect(risk[-1]),
            frozen_mse=effect(16.), frozen_old_rmse=effect(2.), endpoint_old_rmse=effect(2.007))
    for name in report.CONTRASTS:
        result["contrasts"][name] = dict(acquisition_risk_difference=effect(-.2, .1),
            acquisition_gain=dict(**effect(.2, .1), favorable_replicas=2),
            endpoint_risk_difference=effect(-.3), retention_difference=effect(.001, .002),
            endpoint_retention_difference=effect(.002, .003), favorable_replicas=2,
            retention_equivalent=True, endpoint_retention_equivalent=True)
    result["contrasts"]["interaction"] = dict(acquisition_risk_difference=effect(.05, .1))
    efficiency = dict(point=dict(status="valid", saved_fraction=.4, learning_time_ratio=.8,
                         S=.5, matching_budget=60., useful_target=True, target_mse_gain_fraction=.2),
        equal_budget_learning_time_ratio=1.4, equal_budget_learning_time_ratio_ci=[1.3, 1.5],
        full_budget_retention_supported=True, matched_budget_retention_supported=True,
        bootstrap=dict(draws=20, invalid=0, ci=dict(saved_fraction=[.2, .5],
            learning_time_ratio=[.7, .9], S=[.2, .6])))
    for policy in ("R", "B", "C", "D", "E"):
        result["efficiency"][policy] = deepcopy(efficiency)
    for policy in ("C", "D", "E"):
        result["efficiency_vs_B"][policy] = deepcopy(efficiency)
    return result


def ready():
    result = pending()
    result.update(analysis=analysis(), fidelity=dict(status="passed"), calibration=dict(status="passed"))
    result["progress"].update(stage="9C fresh", status="running", fresh_completed=13, fresh_blocks_completed=2)
    return result


def prose(cells):
    return "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in cells)


@pytest.fixture
def no_figures(monkeypatch):
    def plot(snapshot, view, href):
        return nbformat.v4.new_code_cell("# Scalar plot stub", id=report.PREFIX + view + "-plot", metadata={report.TAG: True})
    monkeypatch.setattr(report, "_plot_cell", plot)


def write_book(path):
    cells = [nbformat.v4.new_markdown_cell("# Plan 18", id="title"),
        nbformat.v4.new_code_cell("raise RuntimeError('never execute history')", id="old-output", execution_count=3,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="preserve this\n")]),
        nbformat.v4.new_markdown_cell("## Phase 8", id="plan18-phase8-intro", metadata={"plan18_phase8": True})]
    cells[1]["source"] = ["raise RuntimeError('never execute history')\n"]
    cells[2]["attachments"] = {"note.txt": {"text/plain": "keep attachment"}}
    book = dict(nbformat.v4.new_notebook(cells=cells, metadata={"custom": {"values": [1, 2]}}))
    path.write_text(json.dumps(book))
    return json.loads(path.read_bytes())


def test_pending_native_snapshot_is_usable_before_scientific_launch():
    snapshot = pending()
    before = deepcopy(snapshot)
    cells = report.build_cells(snapshot)
    text = prose(cells)
    assert snapshot == before
    for expected in ("Phase 9 Progress", "0/48", "0/8", "0.1 h", "19.9 h", "15 h", "report acceptance",
                     "Analysis Pending", "B-R", "A-E", "S", "does not establish matched retention"):
        assert expected in text
    assert all(c["cell_type"] == "markdown" for c in cells)
    assert all(report.owned(c) and c["metadata"][report.TAG] is True for c in cells)
    assert report.TAG == "plan18-phase9"
    assert not report.owned(dict(id="plan18-phase90-incorrect"))
    assert not report.owned(dict(id="plan18-phase8-intro", metadata={"plan18_phase8": True}))


def test_settings_uses_sealed_observer_scale_during_ridge_calibration():
    snapshot = pending()
    snapshot['settings'] = {p:dict(process_scale=1e-4) for p in ('D','E')}
    snapshot['calibration'] = dict(status='pending',process_scale=dict(process_scale=1e-5))
    result = report._settings(snapshot)['source']
    assert 'Selected Kalman process-noise scales: D=1e-05, E=1e-05.' in result


def test_native_counts_current_settings_and_recorded_changes(no_figures):
    snapshot = ready()
    snapshot["progress"].update(current=dict(policy="D", replica=2), development_completed=24, candidates=8)
    snapshot["settings"] = [dict(policy="C", ridge=3., representation="diagonal", transport=True)]
    snapshot["changes"] = [dict(summary="Corrected transaction recovery", version="v2")]
    text = prose(report.build_cells(snapshot))
    for expected in ("13/48", "2/8", "policy=D", "replica=2", "Development paths 24", "candidate recipes 8",
                     "Corrected transaction recovery", "| C: F1 + LFU | diagonal | 3.34056 | on | .025 | 3 |"):
        assert expected in text


def test_complete_native_schema_renders_effects_retention_and_both_efficiency_targets(no_figures):
    snapshot = ready()
    text = prose(report.build_cells(snapshot))
    assert "-0.2 [-0.3, -0.1]" in text and "0.001 [-0.001, 0.003]" in text
    assert "0.002 [-0.001, 0.005]" in text and "2/2" in text
    assert "| B:A | 40 [20, 50] | 0.8 [0.7, 0.9] | 0.5 [0.2, 0.6] |" in text
    assert "| C:B | 40 [20, 50]" in text
    assert "root of pooled mean MSE" in text
    assert "roots within each replica" in text
    assert "not familywise significance guarantees" in text
    assert "E-C-D+B" in text


def test_summary_aliases_match_analyzer_retention_contract():
    result = analysis()["summaries"]["B"]
    assert report._summary(result, "retention_avg") == result["retention"]
    assert report._summary(result, "retention_endpoint") == result["endpoint_retention"]


def test_signed_difference_and_legacy_gain_are_not_confused():
    item = dict(acquisition_gain=effect(.2, .1))
    assert report._contrast(item, "acquisition") == effect(-.2, .1)
    item["acquisition_risk_difference"] = effect(-.7, .2)
    assert report._contrast(item, "acquisition") == effect(-.7, .2)


@pytest.mark.parametrize("unavailable", [dict(status="target_unreached"),
    dict(point=dict(status="target_already_met_at_initialization", saved_fraction=.9, S=5.)),
    dict(point=None, bootstrap=None)])
def test_unavailable_efficiency_never_becomes_zero_or_an_invented_success(no_figures, unavailable):
    snapshot = ready()
    snapshot["analysis"]["efficiency"] = {"B": unavailable}
    snapshot["analysis"]["efficiency_vs_B"] = {}
    text = report._efficiency(snapshot["analysis"])["source"]
    assert "| B:A | unavailable | unavailable | unavailable |" in text
    assert "| B:A | 0" not in text and "90" not in text


def test_invalid_inverse_draws_withhold_only_inverse_ci_and_flag_retention(no_figures):
    snapshot = ready()
    item = snapshot["analysis"]["efficiency"]["B"]
    item["bootstrap"]["invalid"] = 1
    item["matched_budget_retention_supported"] = False
    snapshot["analysis"]["efficiency"] = {"B": item}
    snapshot["analysis"]["efficiency_vs_B"] = {}
    text = report._efficiency(snapshot["analysis"])["source"]
    assert "| B:A | 40 | 0.8 | 0.5 | 1.4 [1.3, 1.5] | supported / not estimable |" in text
    assert "CI withheld (1/20 invalid)" in text
    assert "40 [20, 50]" not in text
    assert "not established" not in text.split("\n\n")[0]
    item["point"]["useful_target"] = False
    assert "weak reference target" in report._efficiency(snapshot["analysis"])["source"]


def test_nested_efficiency_schema_does_not_assign_unknown_reference_meanings():
    item = analysis()["efficiency"]["B"]
    nested = dict(efficiency=dict(referenceA=dict(B=item), referenceB=dict(methods=dict(C=item))))
    assert [(p, r) for p, r, _ in report._efficiency_entries(nested)] == [("B", "A"), ("C", "B")]
    assert report._efficiency_entries(dict(efficiency=dict(unknown=dict(score=42.)))) == []


@pytest.mark.parametrize("change", [dict(protocol="phase8"), dict(schema_version=2),
    dict(namespace="development"), dict(namespace="acceptance"), dict(pairing_complete=False)])
def test_incompatible_or_nonfresh_analysis_rejected(change):
    with pytest.raises(ValueError):
        report.build_cells(dict(analysis={**analysis(), **change}))


@pytest.mark.parametrize("case", ["missing_policy", "missing_domain", "missing_summary", "clock", "ci", "duplicates", "final"])
def test_malformed_or_unmatched_analysis_fails_instead_of_omitting_it(case):
    result = analysis()
    if case == "missing_policy":
        del result["curves"]["E"]
    elif case == "missing_domain":
        del result["curves"]["B"]["base"]
    elif case == "missing_summary":
        del result["summaries"]["A"]
    elif case == "clock":
        result["curves"]["B"]["n"] = [0, 1, 1, 3]
    elif case == "ci":
        result["curves"]["C"]["query"]["ci_mse"] = [[2.], [1.]]
    elif case == "duplicates":
        result["replicas"] = [0, 0]
    elif case == "final":
        result.update(is_final=True, provisional=False)
    with pytest.raises(ValueError):
        report.build_cells(dict(analysis=result))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object(), [1, 2]])
def test_nonfinite_and_nonscalar_analysis_rejected(bad):
    with pytest.raises(ValueError):
        report.build_cells(dict(analysis=bad))


def test_final_outcome_uses_actual_effects_including_unfavorable_results(no_figures):
    snapshot = ready()
    result = snapshot["analysis"]
    result.update(replicas=list(range(8)), is_final=True, provisional=False)
    result["contrasts"]["E-B"]["acquisition_risk_difference"] = effect(.4, .1)
    snapshot["progress"].update(status="completed", fresh_completed=48, fresh_blocks_completed=8, forecast_hours=4.)
    cells = report.build_cells(snapshot)
    text = prose(cells)
    assert "forecast 0 h" in text and "forecast 4 h" not in text
    conclusion = next(c["source"] for c in cells if c["id"] == report.PREFIX + "conclusion")
    assert "Combined increment (E-B): **0.4 [0.3, 0.5]**, higher novel risk" in conclusion
    assert "no efficacy claim" not in conclusion
    snapshot["progress"]["status"] = "running"
    assert "Inference remains provisional" in report.build_cells(snapshot)[-1]["source"]


@pytest.mark.parametrize("status", ["calibration_infeasible", "calibration_budget_limited"])
def test_terminal_calibration_is_not_reported_as_pending(status):
    snapshot = pending()
    snapshot["progress"].update(status=status, stage="9B calibration closeout")
    snapshot["calibration"] = dict(status=status, selection=dict(matching=dict(
        ranges=dict(Havg=.01369, H100=.009))))
    text = prose(report.build_cells(snapshot))
    assert "forecast 0 h" in text
    assert "Fresh Confirmation Not Run" in text
    assert "Calibration closed without fresh confirmation" in text
    assert "not an efficacy rejection" in text
    assert "Proposed c" in text and "Selected c" not in text
    assert "average **0.01369 px**; endpoint **0.009 px**" in text
    assert "Analysis Pending" not in text
    assert "Inference remains provisional" not in text


def test_three_main_scalar_figures_are_nonblank_and_labels_mean_what_they_say():
    snapshot = ready()
    snapshot["calibration"]["candidates"] = [dict(policy=p, ridge=1., acquisition=effect(12. - i * .2),
        retention=effect(.003 * i), endpoint_retention=effect(.005 * i), selected=True)
        for i, p in enumerate(report.POLICIES)]
    snapshot["diagnostics"] = {p: dict(observations=[4, 8, 12], pi=[.02, .04, .03], gain=[.1, .2, .1],
        motion_energy=[.03, .02, .01], noise_scale=[.1, .1, .1],
        lfu_relative_correction=[.02, .03, .04], projection_fraction=[0., .001, 0.]) for p in ("D", "E")}
    cells = report.build_cells(snapshot)
    plots = [c for c in cells if c["cell_type"] == "code"]
    assert len(plots) == 3
    for plot in plots:
        assert "load_snapshot" in plot["source"] and "expected_sha256" in plot["source"]
        png = base64.b64decode(plot["outputs"][0]["data"]["image/png"])
        pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        assert pixels.shape[1] > 1000 and pixels.shape[0] > 300 and pixels.std() > 10
    figure = report._figure(snapshot, "fresh")
    new, old = figure.axes
    assert new.get_xlabel() == "Designated novel grids"
    assert new.get_ylabel() == "Mean held-out risk (px^2 / landmark)"
    assert old.get_ylabel() == "Root of mean old MSE (px)"
    assert set(new.get_legend_handles_labels()[1]) == set(report.LABELS.values()) | {"Frozen initialization"}
    assert list(old.lines[0].get_ydata()) == pytest.approx(np.sqrt([4., 4.01, 4.02, 4.03]))
    assert list(old.lines[-1].get_ydata()) == [2., 2.]
    figure.clear()


def test_selected_development_curves_remain_descriptive_and_paired():
    records = [dict(namespace="development", policy=p, replica=r, recipe=dict(ridge=1.),
        plan_hash=str(r), pairing_id=str(r), metrics=[dict(n=n, observations=n * 10,
            base_mse=4. + r + n * .01, novel_mse=9. + r - n * .1) for n in range(4)])
        for p in report.POLICIES for r in range(4)]
    ridges = dict.fromkeys(report.POLICIES, 1.)
    exported = report.development_curves(records, ridges)
    assert exported["replicas"] == [0, 1, 2, 3]
    assert exported["curves"]["B"]["query"]["mean_mse"] == pytest.approx([10.5, 10.4, 10.3, 10.2])
    snapshot = pending()
    snapshot["progress"]["status"] = "calibration_infeasible"
    snapshot["development"] = exported
    cells = report.build_cells(snapshot)
    text = prose(cells)
    assert "descriptive, not fresh inference" in text and "no confidence intervals" in text
    assert "Fresh Paired Effects" not in text
    figure = report._figure(snapshot, "development_paths")
    assert figure.axes[0].get_title() == "Development: New-data prediction"
    assert len(figure.axes[0].lines) == 7
    figure.clear()
    with pytest.raises(ValueError, match="four complete"):
        report.development_curves(records[:-1], ridges)
    records[0]["metrics"][1]["observations"] += 1
    with pytest.raises(ValueError, match="unpaired"):
        report.development_curves(records, ridges)


def test_frozen_line_not_invented_when_initial_checkpoint_missing():
    snapshot = ready()
    for curve in snapshot["analysis"]["curves"].values():
        curve["n"] = [1, 2, 3, 4]
    figure = report._figure(snapshot, "fresh")
    assert "Frozen initialization" not in figure.axes[0].get_legend_handles_labels()[1]
    figure.clear()


def test_native_row_diagnostics_and_unknown_metadata_do_not_invent_series():
    snapshot = pending()
    snapshot["diagnostics"] = [dict(policy="E", observations=4 * i, applied_pi=.02 + i * .001,
        kalman_velocity_gain_mean=.1, motion_energy=.01, noise_scale=.03,
        correction_norm=.02, negative_fraction=.001) for i in (1, 2, 3)]
    assert report._has_diagnostics(snapshot)
    normalized = report._diagnostics(snapshot)
    assert normalized["E"]["applied_pi"] == [.021, .022, .023]
    figure = report._figure(snapshot, "diagnostics")
    assert len(figure.axes) == 4
    assert set(figure.axes[0].get_legend_handles_labels()[1]) == {"E: pi", "E: Kalman velocity gain"}
    assert list(figure.axes[3].lines[0].get_ydata()) == [.001] * 3
    figure.clear()
    snapshot["diagnostics"] = {"status": "pending", "E": {"status": "not_yet_exported"}}
    assert not report._has_diagnostics(snapshot)


def test_repeated_row_clocks_are_not_pooled_into_unstated_replica_summaries():
    snapshot = pending()
    snapshot["diagnostics"] = [dict(policy="D", observations=4, applied_pi=.03),
                               dict(policy="D", observations=4, applied_pi=.02)]
    with pytest.raises(ValueError, match="diagnostic"):
        report.build_cells(snapshot)


def test_pairing_exposure_mismatch_and_missing_all_summaries_fail_clearly():
    data = analysis()
    data["curves"]["B"]["observations"][-1] += 1
    with pytest.raises(ValueError, match="exposure"):
        report.build_cells(dict(analysis=data))
    data = analysis()
    data["summaries"] = {}
    with pytest.raises(ValueError, match="summaries"):
        report.build_cells(dict(analysis=data))


def test_no_store_defaults_to_the_ignored_phase_root(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "DEFAULT_ROOT", tmp_path / "cache")
    result = report.publish(pending(), tmp_path / "results.ipynb")
    assert Path(result["path"]).is_relative_to(tmp_path / "cache")
    assert (tmp_path / "cache/report/latest.json").is_file()


def test_publication_preserves_old_cells_outputs_metadata_and_later_notes(tmp_path):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    before = write_book(notebook)
    first = report.publish(pending(), notebook, store)
    book = json.loads(notebook.read_bytes())
    assert [c for c in book["cells"] if not report.owned(c)] == before["cells"]
    assert book["metadata"] == before["metadata"]
    assert book["cells"][1]["id"] == report.PREFIX + "progress"
    assert next(i for i, c in enumerate(book["cells"]) if c["id"] == report.PREFIX + "intro") > 3
    path = Path(first["path"])
    files = {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    assert files["results.ipynb"] == notebook.read_bytes()
    assert files["report_source.py"] == Path(report.__file__).read_bytes()
    assert json.loads(files["snapshot.json"]) == pending()
    second = report.publish(pending(), notebook, store)
    assert second["changed"] is False and second["generation"] == first["generation"]
    assert json.loads(notebook.read_bytes()) == book
    later = nbformat.v4.new_markdown_cell("Future notes", id="later-user")
    book["cells"].append(later)
    notebook.write_text(json.dumps(book))
    report.publish(pending(), notebook, store)
    assert json.loads(notebook.read_bytes())["cells"][-1] == later
    assert {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()} == files
    assert all(n > 0 for n in store.reservations)


def test_new_generation_does_not_mutate_old_and_records_scalar_source_version(tmp_path):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    first = report.publish(pending(), notebook, store)
    path = Path(first["path"])
    saved = {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    snapshot = pending()
    snapshot["progress"]["charged_hours"] += .2
    second = report.publish(snapshot, notebook, store)
    assert first["generation"] != second["generation"]
    assert {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()} == saved
    pointer = json.loads((store.root / "report/latest.json").read_bytes())
    assert pointer["generation"] == second["generation"]
    inputs = json.loads(saved["inputs.json"])
    assert inputs["source_sha256"] == hashlib.sha256(saved["report_source.py"]).hexdigest()


@pytest.mark.parametrize("filename", ["snapshot.json", "analysis.json", "results.ipynb", "report_source.py"])
def test_corrupted_completed_generation_is_never_repaired(tmp_path, filename):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    result = report.publish(pending(), notebook, store)
    path = Path(result["path"]) / filename
    path.write_bytes(b"corrupted")
    before = notebook.read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        report.publish(pending(), notebook, store)
    assert path.read_bytes() == b"corrupted" and notebook.read_bytes() == before


@pytest.mark.parametrize("stage", ["staging", "notebook", "pointer"])
def test_interrupted_publication_is_resumable_and_seals_immutable(tmp_path, monkeypatch, stage):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    original = report._atomic
    def interrupt(path, data):
        if ((stage == "staging" and path.name == "analysis.json")
                or (stage == "notebook" and path == notebook)
                or (stage == "pointer" and path.name == "latest.json")):
            raise KeyboardInterrupt("simulated power loss")
        return original(path, data)
    monkeypatch.setattr(report, "_atomic", interrupt)
    with pytest.raises(KeyboardInterrupt):
        report.publish(pending(), notebook, store)
    directory = store.root / "report/generations"
    assert not list(directory.glob(".tmp-*"))
    seals = {p.name: (p / "COMPLETED").read_bytes() for p in directory.iterdir()}
    monkeypatch.setattr(report, "_atomic", original)
    result = report.publish(pending(), notebook, store)
    assert result["changed"] == (stage == "staging")
    for name, seal in seals.items():
        assert (directory / name / "COMPLETED").read_bytes() == seal


def test_scalar_loaders_bind_checksums_and_reject_schema_and_oversize(tmp_path, monkeypatch, no_figures):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    result = report.publish(ready(), notebook, store)
    path = Path(result["path"])
    for name, loader, expected in (("snapshot.json", report.load_snapshot, ready()),
                                   ("analysis.json", report.load_analysis, analysis())):
        scalar = path / name
        assert loader(scalar, hashlib.sha256(scalar.read_bytes()).hexdigest()) == expected
        with pytest.raises(ValueError, match="checksum"):
            loader(scalar, "0" * 64)
    monkeypatch.setattr(report, "MAX_JSON_BYTES", 2)
    with pytest.raises(ValueError, match="oversized"):
        report.load_analysis(path / "analysis.json")


def test_single_publisher_frozen_history_symlink_and_size_guards(tmp_path, monkeypatch):
    store, notebook = Store(tmp_path / "cache"), tmp_path / "results.ipynb"
    with report._lease(store.root / "report"):
        with pytest.raises(ValueError, match="lease"):
            report.publish(pending(), notebook, store)
    notebook.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        report.publish(pending(), notebook, store)
    notebook.unlink()
    notebook.write_text(json.dumps(dict(nbformat.v4.new_notebook(metadata={"historical": {"role": "frozen_archive"}}))))
    with pytest.raises(ValueError, match="frozen"):
        report.publish(pending(), notebook, store)
    monkeypatch.setattr(report, "MAX_JSON_BYTES", 2)
    with pytest.raises(ValueError, match="size"):
        report.publish(pending(), tmp_path / "not-written.ipynb", store)
    assert not (tmp_path / "not-written.ipynb").exists()


def test_import_and_publication_are_artifact_only_even_with_native_analysis(tmp_path):
    scalar = tmp_path / "input.json"
    scalar.write_text(json.dumps(ready()))
    source = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
from mnist_experiment.combined_memory.replicated_combination import report
snapshot = json.loads(Path(sys.argv[1]).read_bytes())
report.publish(snapshot, Path(sys.argv[2]), SimpleNamespace(root=Path(sys.argv[3])))
for name in ('torch', 'torchvision', 'mnist_experiment.combined_memory.replicated_combination.analysis',
             'mnist_experiment.combined_memory.replicated_combination.runtime',
             'mnist_experiment.combined_memory.replicated_combination.runner'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", source, str(scalar), str(tmp_path / "results.ipynb"), str(tmp_path / "cache")],
                   check=True, capture_output=True, text=True)
