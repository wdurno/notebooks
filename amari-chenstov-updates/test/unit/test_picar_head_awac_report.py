"""JSON-only Fix 5 reporting, without learners or canonical notebook writes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import nbformat
import pytest

from numerical_experiment.picar_head_awac import report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def progress(root, **fields):
    write_json(root / "progress.json", {
        "version": report.VERSION, "status": "running", "stage": "5B resource pilot",
        "charged_seconds": 360., "completed_units": 0, "planned_units": 6,
        "active_unit": "initialization", "forecast": {"remaining_seconds": [3600., 7200.]}, **fields,
    })


def unit(root, name="trajectory", *, completed=True, **fields):
    path = root / "units" / name
    operation = fields.pop("operation", "trajectory")
    summary = {"version": report.VERSION, "schema_version": 1,
               "operation": operation, "kind": "head_awac_" + operation,
               "settings": {"condition": "td_fd", "seed": 7, "ewc_rank": 3}, "active_seconds": 12., **fields}
    write_json(path / "summary.json", summary)
    if completed:
        (path / "COMPLETED").touch()
    return path


def episode(raw_return=2., **fields):
    return {"success": False, "raw_return": raw_return, "coded_return": 4.,
            "raw_qwen_mean": .25, "critic_mse": .5, "heldout_nll": 1.2, **fields}


def panel(n=0, environment="initial_A", episodes=None):
    return {"n": n, "environment": environment, "episodes": [episode()] if episodes is None else episodes}


def metric(n=1, **fields):
    return {"n": n, "data_nll": 2., "fd": .1, "ewc": 0., "post_step_ewc": .2, "update_seconds": .01,
            "replay_count": 8, "age_mean": 3., "age_saturated": .25,
            "ewc_proposal_difference": .002, "optimizer_exposures": 16, **fields}


def base_book(path):
    book = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Historical Plan 19", id="historical-title"),
        nbformat.v4.new_code_cell("print('preserve exactly')", id="plan19-phase3-fix5-progress",
            execution_count=17, outputs=[nbformat.v4.new_output("stream", name="stdout", text="historical output\n")],
            metadata={"report_section": "progress", "user_annotation": [1, 2]}),
        nbformat.v4.new_markdown_cell("User theory and conclusion", id="user-theory"),
    ], metadata={"user_metadata": {"retain": True}})
    # Multiline lists must remain lists, not normalized to a different representation.
    raw = json.loads(nbformat.writes(book))
    write_json(path, raw)
    return raw


def rendered(outputs):
    return "\n".join(o.get("data", {}).get("text/plain", "") for o in outputs)


def test_preserves_historical_cells_outputs_metadata_and_unique_ids_on_refresh(tmp_path):
    root, output, base = tmp_path / "run", tmp_path / "report.ipynb", tmp_path / "history.ipynb"
    history = base_book(base)
    progress(root)
    assert report.build_notebook(root, output, base) == output
    first = json.loads(output.read_text())
    assert [c for c in first["cells"] if not c.get("metadata", {}).get(report._TAG)] == history["cells"]
    assert first["metadata"] == history["metadata"]
    assert first["cells"][1]["metadata"][report._TAG]
    assert first["cells"][-1]["metadata"][report._TAG]
    assert len({c["id"] for c in first["cells"]}) == len(first["cells"])
    assert "Provisional / incomplete" in first["cells"][-1]["source"]
    progress(root, charged_seconds=720.)
    report.build_notebook(root, output)
    second = json.loads(output.read_text())
    assert [c for c in second["cells"] if not c.get("metadata", {}).get(report._TAG)] == history["cells"]
    assert len(second["cells"]) == len(first["cells"])
    assert [c["id"] for c in second["cells"]] == [c["id"] for c in first["cells"]]
    nbformat.validate(nbformat.from_dict(second))


def test_empty_and_incomplete_progress_is_concise_without_opening_unsealed_summary(tmp_path, monkeypatch):
    root = tmp_path / "run"
    path = unit(root, completed=False)
    (path / "summary.json").write_text("not JSON; must never be opened")
    progress(root, active_unit={"name": "trajectory", "acquisitions": 3, "accepted_fits": 7})
    opened = []
    original = report.shared._json

    def read(path, **kwargs):
        opened.append(Path(path).name)
        return original(path, **kwargs)

    monkeypatch.setattr(report.shared, "_json", read)
    snapshot = report.load_report(root)
    assert opened == ["progress.json"]
    assert not snapshot["complete"] and snapshot["units"] == []
    assert report.result_outputs(snapshot) == []
    text = rendered(report.progress_outputs(snapshot))
    assert "0 / 6" in text and "1.00--2.00 h remaining" in text
    assert "Acquisitions': '3'" in text and "Accepted fits': '7'" in text
    assert "No completed units yet" in report.conclusion(snapshot)


def test_nonexistent_root_is_read_only_and_report_can_be_built_early(tmp_path):
    root = tmp_path / "absent"
    base = tmp_path / "base.ipynb"
    base_book(base)
    report.build_notebook(root, tmp_path / "out.ipynb", base)
    assert not root.exists()
    assert "Not started / incomplete" in rendered(report.progress_outputs(report.load_report(root)))


def test_in_checkout_notebook_uses_portable_artifact_paths(tmp_path, monkeypatch):
    project = tmp_path / "project"
    root = project / "cache/run"
    output = project / "report.ipynb"
    monkeypatch.setattr(report, "__file__", str(project / "numerical_experiment/picar_head_awac/report.py"))
    progress(root)
    report.build_notebook(root, output)
    book = json.loads(output.read_text())
    code = "\n".join(c["source"] for c in book["cells"]
                     if c["cell_type"] == "code" and c.get("metadata", {}).get(report._TAG))
    assert "project_root / 'cache/run'" in code
    assert str(project) not in code
    assert "project_root = parent" in code


def test_completed_evaluations_average_trajectories_not_unequal_episode_counts(tmp_path):
    unit(tmp_path, "one", settings={"condition": "td_fd", "replica": 1},
         evaluations=[panel(0), panel(8, episodes=[episode(0.)])])
    unit(tmp_path, "two", settings={"condition": "td_fd", "replica": 2},
         evaluations=[panel(8, episodes=[episode(8., success=True)] * 3)])
    unit(tmp_path, "pilot", operation="timing", evaluations=[panel(8, episodes=[episode(999.)])])
    unit(tmp_path, "unsealed", completed=False, evaluations=[panel(8, episodes=[episode(999.)])])
    rows = report.evaluation_rows(report.load_report(tmp_path))
    assert [r["n"] for r in rows] == [0, 8]
    assert rows[-1]["raw_return"] == 4.
    assert rows[-1]["success"] == .5
    assert rows[-1]["Episodes"] == 4 and rows[-1]["Replicas"] == 2
    assert rows[-1]["raw_qwen_mean"] == .25 and rows[-1]["coded_return"] == 4.


def test_plot_values_raw_reward_and_missing_values_are_not_zero(tmp_path, monkeypatch):
    unit(tmp_path, metrics=[metric(), metric(8, data_nll=None)],
         evaluations=[panel(n, env, [episode(raw_return=-3. if n == 0 else 2.)])
                      for env in ("initial_A", "moderate_A") for n in (0, 8)],
         archive={"rank": 3, "eigenvalues": [1., 3., 2.], "trace": 12., "correlated_trace_fraction": .5})
    figures = []

    def capture(figure, description):
        figures.append(figure)
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", capture)
    outputs = report.result_outputs(report.load_report(tmp_path))
    assert len(figures) == 3
    line = figures[0].axes[0].lines[0]
    assert list(line.get_xdata()) == [0, 8] and list(line.get_ydata()) == [-3., 2.]
    assert list(figures[0].axes[1].lines[0].get_ydata()) == [0., 0.]
    assert figures[1].axes[0].lines[0].get_ydata()[1] != 0.
    assert list(figures[2].axes[0].lines[0].get_ydata()) == [3., 2., 1.]
    assert "not full archive" in figures[2].axes[0].get_title()
    assert "raw_qwen_mean" in rendered(outputs) and "whole-unit cost" in rendered(outputs)


def test_ewc_plot_uses_saved_post_fit_penalty_not_first_step_zeros(tmp_path, monkeypatch):
    unit(tmp_path, metrics=[metric(1, post_step_ewc=.02), metric(2, post_step_ewc=.01)])
    figures = []

    def capture(figure, description):
        figures.append(figure)
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", capture)
    report.result_outputs(report.load_report(tmp_path), section="training")
    axis = figures[0].axes[7]
    assert axis.get_title() == "ewc after fit"
    assert list(axis.lines[0].get_xdata()) == [1, 2]
    assert list(axis.lines[0].get_ydata()) == [.02, .01]


def test_missing_post_fit_ewc_is_not_replaced_by_first_step_zeros(tmp_path, monkeypatch):
    row = metric()
    row.pop("post_step_ewc")
    unit(tmp_path, metrics=[row])
    figures = []

    def capture(figure, description):
        figures.append(figure)
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", capture)
    report.result_outputs(report.load_report(tmp_path), section="training")
    assert len(figures[0].axes[7].lines) == 0
    assert "Not collected" in [t.get_text() for t in figures[0].axes[7].texts]


@pytest.mark.parametrize("active", [False, True])
def test_post_fit_ewc_schema_validates_completed_and_live_values(tmp_path, active):
    if active:
        progress(tmp_path, active_progress={"latest_fit": {"post_step_ewc": "bad"}})
    else:
        unit(tmp_path, metrics=[metric(post_step_ewc="bad")])
    with pytest.raises(report.ArtifactError, match="post_step_ewc"):
        report.load_report(tmp_path)


@pytest.mark.parametrize("has_visits", [True, False])
def test_fit_header_glossary_follows_plot_and_survives_refresh(tmp_path, monkeypatch, has_visits):
    root, output = tmp_path / "run", tmp_path / "report.ipynb"
    base = tmp_path / "history.ipynb"
    base_book(base)
    row = metric(visits=12.) if has_visits else metric()
    unit(root, metrics=[row])
    monkeypatch.setattr(report, "_figure", lambda figure, description:
                        nbformat.v4.new_output("display_data", data={"text/plain": description}))
    snapshot = report.load_report(root)
    outputs = report.result_outputs(snapshot)
    index = next(i for i, out in enumerate(outputs)
                 if out.get("data", {}).get("text/plain", "").startswith("Training data NLL"))
    glossary = outputs[index + 1]["data"]["text/markdown"]
    names = ("data nll", "replay count", "age mean", "age saturated",
             "visits" if has_visits else "optimizer exposures",
             "update seconds", "fd", "ewc after fit", "ewc proposal difference")
    assert len([line for line in glossary.splitlines() if line.startswith("- **")]) == 9
    assert all("- **" + name + "**:" in glossary for name in names)
    assert "not total buffer occupancy" in glossary
    assert "including this fit" in glossary if has_visits else "Batch size times optimizer steps" in glossary
    assert "square of each transition's age" in glossary
    assert "first optimizer step" in glossary and "not evidence of no EWC use" in glossary
    assert "exactly zero at every fit" in glossary
    assert "not the newly recentered one" in glossary and "not memory volume" in glossary
    assert "does not establish STM-to-LTM transfer" in glossary
    assert "not a fixed-reference test loss" in glossary
    assert "**optimizer exposures**" not in glossary if has_visits else "**visits**" not in glossary
    for refresh in range(2):
        report.build_notebook(root, output, base if refresh == 0 else None)
        book = nbformat.read(output, as_version=4)
        index = next(i for i, cell in enumerate(book.cells)
                     if cell.cell_type == "markdown" and cell.source == glossary)
        assert book.cells[index - 1].cell_type == "code"
        assert rendered(book.cells[index - 1].outputs).startswith("Training data NLL")
        assert sum(cell.cell_type == "markdown" and cell.source == glossary for cell in book.cells) == 1
        assert not [out for cell in book.cells if cell.cell_type == "code"
                    for out in cell.outputs if "text/markdown" in out.get("data", {})]


def test_recording_index_follows_success_plot_with_portable_links_and_true_outcomes(tmp_path, monkeypatch):
    from html.parser import HTMLParser
    from urllib.parse import unquote

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.hrefs = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.hrefs.extend(value for key, value in attrs if key == "href")

    root, output, base = tmp_path / "run", tmp_path / "notebooks/results.ipynb", tmp_path / "base.ipynb"
    base_book(base)
    shared = root / "initial & shared.mp4"
    win = root / "win <example>.mp4"
    loss = root / "loss.mp4"
    root.mkdir()
    for path in (shared, win, loss):
        path.touch()
    shared_panel = {**panel(0, "moderate_A", [episode(success=False, seed=5, videos=[{"path": str(shared)}]),
                                            episode(success=True)]),
                    "shared_initial_source": "shared-initial-unit"}
    for method in ("td_fd", "td_replay"):
        confirmation_unit(root, method, method, evaluations=[shared_panel,
            panel(4, "moderate_A", [episode(success=True)]),
            panel(8, "moderate_A", [
                episode(success=True, seed=6, videos=[{"path": win.name, "selection": "failure_gallery"}]),
                episode(seed=7, videos=[{"path": str(loss)}]),
                episode(success=True, videos=[{"path": str(root / "missing.mp4")}]),
            ])], metrics=[metric()])
    unit(root, "not-in-plot", settings={"phase": "development", "condition": "DEVELOPMENT"},
         evaluations=[panel(8, episodes=[episode(videos=[{"path": str(loss)}])])])
    unit(root, "not-sealed", completed=False,
         evaluations=[panel(8, episodes=[episode(videos=[{"path": str(loss)}])])])
    monkeypatch.setattr(report, "_figure", lambda figure, description:
                        nbformat.v4.new_output("display_data", data={"text/plain": description}))
    snapshot = report.load_report(root)
    text = report.recording_index(snapshot, output.parent)
    assert "Shared initialization (all methods)" in text and text.count("shared.mp4") == 1
    assert "TD + FD" in text and "TD replay" in text and "moderate_A" in text
    assert "Wins: 1/3 evaluated episodes recorded" in text
    assert "Losses: 1/1 evaluated episodes recorded" in text
    assert "Seed 7; n=4: 0/1 recorded. No saved video available." in text
    assert "missing.mp4" not in text and "DEVELOPMENT" not in text
    assert "not winners or representative episodes" in text
    parser = Links()
    parser.feed(text)
    assert len(parser.hrefs) == 5 and all(not href.startswith(("/", "file:", "http")) for href in parser.hrefs)
    assert all((output.parent / unquote(href)).resolve().is_file() for href in parser.hrefs)
    assert "win%20%3Cexample%3E.mp4" in text and "initial%20%26%20shared.mp4" in text
    # Renderer metadata says failure_gallery even for a real win; use episode.success.
    assert text.index("win%20%3Cexample%3E.mp4") < text.index("Losses: 1/1", text.index("TD + FD"))
    for refresh in range(2):
        report.build_notebook(root, output, base if refresh == 0 else None)
        book = nbformat.read(output, as_version=4)
        indexes = [i for i, cell in enumerate(book.cells) if cell.cell_type == "code"
                   and "display_recordings" in cell.source]
        assert len(indexes) == 1
        gallery = book.cells[indexes[0]]
        assert gallery.outputs[0]["data"]["text/html"] == text
        assert "text/markdown" not in gallery.outputs[0]["data"]
        assert gallery.metadata.jupyter.source_hidden
        assert not any(cell.cell_type == "markdown" and "<details>" in cell.source for cell in book.cells)
        previous = book.cells[indexes[0] - 1]
        assert previous.cell_type == "code"
        assert previous.outputs[-1]["data"]["text/plain"].startswith("Observed raw return and success")


def test_recording_html_has_no_raw_markdown_and_rendered_cell_reexecutes(tmp_path, monkeypatch):
    from IPython import display

    base = tmp_path / "history.ipynb"
    base_book(base)
    root, output = tmp_path / "run", tmp_path / "report.ipynb"
    unit(root, evaluations=[panel(8)])
    monkeypatch.setattr(report, "_figure", lambda figure, description:
                        nbformat.v4.new_output("display_data", data={"text/plain": description}))
    report.build_notebook(root, output, base)
    book = nbformat.read(output, as_version=4)
    cell = next(cell for cell in book.cells if cell.cell_type == "code" and "display_recordings" in cell.source)
    saved = cell.outputs[0]["data"]
    assert saved["text/html"].startswith("<h4>Episode Recordings</h4>")
    assert "#### Episode" not in saved["text/html"]
    assert saved["text/html"].count("<details>") == saved["text/html"].count("</details>")
    shown = []
    monkeypatch.setattr(display, "display", lambda data, **kwargs: shown.append(data))
    exec(cell.source, {})
    assert shown == [saved]


@pytest.mark.parametrize("videos", [None, ["clip.mp4"], [{"path": 5}], [{"path": ""}]])
def test_recording_metadata_schema_errors_are_clear(tmp_path, videos):
    unit(tmp_path, evaluations=[panel(episodes=[episode(videos=videos)])])
    with pytest.raises(report.ArtifactError, match="video"):
        report.load_report(tmp_path)


def test_archive_provenance_does_not_imply_nonzero_rank_fresh_outcomes(tmp_path, monkeypatch):
    unit(tmp_path, "initial", operation="initial_fisher", settings={"rank": 10},
         archive={"rank": 10, "eigenvalues": [5.]})
    confirmation_unit(tmp_path, "fresh", "td_fd", settings={"ewc_rank": 0},
                      evaluations=[panel(8)], metrics=[metric()], archive={"rank": 0, "eigenvalues": []})
    unit(tmp_path, "development", settings={"phase": "development", "ewc_rank": 10},
         evaluations=[panel(8, episodes=[episode(raw_return=999.)])], metrics=[metric(data_nll=999.)])
    figures = []

    def capture(figure, description):
        figures.append(figure)
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", capture)
    snapshot = report.load_report(tmp_path)
    outputs = report.result_outputs(snapshot)
    assert all("r=0" in label for label in figures[0].axes[0].get_legend_handles_labels()[1])
    assert list(figures[1].axes[0].lines[0].get_ydata()) == [2.]
    assert "Initial/geometry: Initial Fisher r=10" in figures[2].axes[0].get_legend_handles_labels()[1]
    assert "initialization / geometry" in rendered(outputs)
    assert "EWC rank r is not STM LoRA rank" in rendered(outputs)
    assert "not matched method timing comparisons" in rendered(outputs)


def test_no_completion_claim_until_all_planned_units_are_sealed(tmp_path):
    unit(tmp_path)
    progress(tmp_path, status="completed", completed_units=1, planned_units=2)
    assert "Provisional" in report.conclusion(report.load_report(tmp_path))
    progress(tmp_path, status="completed", completed_units=2, planned_units=2)
    assert not report.load_report(tmp_path)["complete"]
    unit(tmp_path, "selection", operation="selection", selected={"ewc_rank": 3})
    assert "Approved scope complete" in report.conclusion(report.load_report(tmp_path))
    assert "No acquired-sample equivalence" in report.conclusion(report.load_report(tmp_path))


def test_development_and_fresh_phases_and_ewc_ranks_are_not_pooled(tmp_path):
    for name, phase, rank in (("dev", "development", 3), ("fresh", "fresh", 3), ("rank", "fresh", 5)):
        unit(tmp_path, name, settings={"condition": "td_fd", "phase": phase, "ewc_rank": rank, "seed": 7},
             evaluations=[panel()])
    rows = report.evaluation_rows(report.load_report(tmp_path))
    assert len(rows) == 3
    assert all(r["Trajectories"] == 1 for r in rows)


def test_missing_metrics_remain_unavailable_with_explicit_coverage(tmp_path):
    unit(tmp_path, evaluations=[panel(episodes=[episode(None, heldout_nll=None, critic_mse=None)])])
    row = report.evaluation_rows(report.load_report(tmp_path))[0]
    assert row["raw_return"] is None and row["raw_return count"] == 0
    assert row["heldout_nll"] is None and row["heldout_nll count"] == 0
    assert row["critic_mse"] is None and row["critic_mse count"] == 0


def test_actual_acquisitions_are_distinguished_from_nominal_checkpoints(tmp_path):
    unit(tmp_path, "one", settings={"condition": "td_fd", "replica": 1},
         evaluations=[{**panel(64), "acquired_count": 68}])
    unit(tmp_path, "two", settings={"condition": "td_fd", "replica": 2},
         evaluations=[{**panel(64), "acquired_count": 72}])
    snapshot = report.load_report(tmp_path)
    row = report.evaluation_rows(snapshot)[0]
    assert row["n"] == 64 and row["Actual acquisition range"] == [68, 72]
    assert "actual acquisition ranges are tabulated" in rendered(report.result_outputs(snapshot))


def test_frozen_feature_health_is_visible_without_numerical_artifact_loading(tmp_path):
    unit(tmp_path, operation="feature_health", feature_dimension=64, centered_variance_trace=1.3,
         leading_variance_share=.99, median_channel_std=.09)
    text = rendered(report.result_outputs(report.load_report(tmp_path)))
    assert "Frozen Representation Health" in text
    assert "Leading component variance share" in text


def test_empty_present_progress_is_invalid_not_silently_treated_as_missing(tmp_path):
    write_json(tmp_path / "progress.json", {})
    with pytest.raises(report.ArtifactError, match="progress.status"):
        report.load_report(tmp_path)


@pytest.mark.parametrize("fields, match", [
    ({"kind": "curriculum_trajectory"}, "kind"),
    ({"version": "future"}, "version"),
    ({"schema_version": 2}, "schema_version"),
    ({"settings": []}, "settings"),
    ({"active_seconds": -1.}, "active_seconds"),
    ({"metrics": [metric(data_nll="bad")]}, "data_nll"),
    ({"metrics": [1]}, "objects"),
    ({"evaluations": [panel(episodes=[episode(success=1)])]}, "success"),
    ({"archive": {"eigenvalues": [1.] * 11}}, "eigenvalues"),
])
def test_summary_schema_errors_are_clear(tmp_path, fields, match):
    unit(tmp_path, **fields)
    with pytest.raises(report.ArtifactError, match="Fix 5 schema-incompatible:.*" + match):
        report.load_report(tmp_path)


@pytest.mark.parametrize("fields, match", [
    ({"charged_seconds": -1.}, "charged_seconds"),
    ({"completed_units": 7}, "counts"),
    ({"planned_units": "6"}, "planned_units"),
    ({"active_unit": []}, "active_unit"),
])
def test_progress_schema_errors_are_clear(tmp_path, fields, match):
    progress(tmp_path, **fields)
    with pytest.raises(report.ArtifactError, match=match):
        report.load_report(tmp_path)


def test_bounded_json_scan_tables_and_escaped_settings(tmp_path, monkeypatch):
    unit(tmp_path, settings={"condition": "<script>"})
    snapshot = report.load_report(tmp_path)
    assert "&lt;script&gt;" in "\n".join(o["data"].get("text/html", "") for o in report.result_outputs(snapshot))
    monkeypatch.setattr(report, "MAX_JSON_BYTES", 8)
    with pytest.raises(report.ArtifactError, match="size limit"):
        report.load_report(tmp_path)
    monkeypatch.setattr(report, "MAX_JSON_BYTES", 8192)
    monkeypatch.setattr(report, "MAX_SCAN_DIRS", 0)
    with pytest.raises(report.ArtifactError, match="scan limit"):
        report.load_report(tmp_path)
    rows = [{"Unit": str(i)} for i in range(30)]
    table = report._table(rows, "bounded")["data"]["text/html"]
    assert "Showing 18/30" in table and "<td>29</td>" not in table


def test_total_scalar_bound_counts_all_units_and_still_fails_above_capacity(tmp_path, monkeypatch):
    for index in range(28):
        unit(tmp_path, f"unit-{index}")
    total = sum(path.stat().st_size for path in tmp_path.glob("units/*/summary.json"))
    monkeypatch.setattr(report, "MAX_TOTAL_BYTES", total)
    assert len(report.load_report(tmp_path)["units"]) == 28
    monkeypatch.setattr(report, "MAX_TOTAL_BYTES", total - 1)
    with pytest.raises(report.ArtifactError, match="total scalar JSON size limit"):
        report.load_report(tmp_path)


def test_atomic_failure_preserves_output_and_cleans_temporary(tmp_path, monkeypatch):
    base, output = tmp_path / "base.ipynb", tmp_path / "out.ipynb"
    base_book(base)
    output.write_text("untouched original output")

    def fail(*args):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(report.os, "replace", fail)
    with pytest.raises(OSError, match="replace failure"):
        report.build_notebook(tmp_path / "missing", output, base)
    assert output.read_text() == "untouched original output"
    assert not list(tmp_path.glob(".head-awac-report-*"))


def test_schema_failure_preserves_output_and_completed_output_is_refused(tmp_path):
    output = tmp_path / "out.ipynb"
    output.write_text("unchanged")
    unit(tmp_path, active_seconds=-1.)
    with pytest.raises(report.ArtifactError, match="active_seconds"):
        report.build_notebook(tmp_path, output)
    assert output.read_text() == "unchanged"
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "COMPLETED").touch()
    with pytest.raises(report.ArtifactError, match="completed run"):
        report.build_notebook(tmp_path / "absent", sealed / "report.ipynb")


def test_builder_and_rendered_cells_never_import_training_or_unpickle(tmp_path):
    base = tmp_path / "base.ipynb"
    write_json(base, dict(nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# History")])))
    unit(tmp_path / "run", metrics=[metric()], evaluations=[panel()], archive={"eigenvalues": [1.], "rank": 1})
    script = """
import importlib.abc
import json
from pathlib import Path
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        forbidden = {'torch', 'transformers', 'safetensors', 'huggingface_hub'}
        if fullname.split('.')[0] in forbidden or fullname.startswith('numerical_experiment.picar_head_awac.') and fullname.rsplit('.', 1)[-1] in {'model', 'data', 'learner', 'likelihood', 'config', 'archive'}:
            raise AssertionError('forbidden training import: ' + fullname)
sys.meta_path.insert(0, Block())
import pickle
def forbidden(*args, **kwargs):
    raise AssertionError('unpickling forbidden')
pickle.load = pickle.loads = forbidden
from numerical_experiment.picar_head_awac import report
root, output, base = map(Path, sys.argv[1:])
report.build_notebook(root, output, base)
book = json.loads(output.read_text())
for cell in book['cells']:
    if cell['cell_type'] == 'code':
        exec(cell['source'], {})
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "run"), str(tmp_path / "out.ipynb"), str(base)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")})
    assert result.returncode == 0, result.stderr
    nbformat.validate(nbformat.read(tmp_path / "out.ipynb", as_version=4))


def test_factor_eigenvalues_alias_is_validated_and_plotted(tmp_path, monkeypatch):
    unit(tmp_path, archive={"factor_eigenvalues": [2., 5.], "rank": 2, "trace": 10.})
    spectra = []

    def capture(figure, description):
        spectra.append(list(figure.axes[0].lines[0].get_ydata()))
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", capture)
    report.result_outputs(report.load_report(tmp_path))
    assert spectra == [[5., 2.]]
    unit(tmp_path, "bad", archive={"factor_eigenvalues": ["not numeric"]})
    with pytest.raises(report.ArtifactError, match="eigenvalues"):
        report.load_report(tmp_path)


def test_live_preview_and_latest_fit_are_explicitly_incomplete_never_sealed_outcomes(tmp_path):
    unit(tmp_path, completed=False)
    (tmp_path / "units/trajectory/summary.json").write_text("must never be read")
    progress(tmp_path, active_progress={"n": 24, "fits": 7, "latest_fit": metric(24),
        "evaluation_preview": [{"n": 0, "environment": "initial_A", "success": .5,
                                "raw_return": 12., "episodes": 2}]})
    snapshot = report.load_report(tmp_path)
    outputs = report.progress_outputs(snapshot)
    assert len(outputs) == 3
    assert "Latest Fit: Incomplete Live Preview" in rendered(outputs)
    assert "excluded from sealed curves" in rendered(outputs)
    assert "raw_return': '12'" in rendered(outputs)
    assert "Acquisitions': '24'" in rendered(outputs)
    assert report.evaluation_rows(snapshot) == [] and report.result_outputs(snapshot) == []
    assert not snapshot["complete"]


def test_distinct_resource_and_fd_recipes_are_never_pooled_as_replicas(tmp_path):
    candidates = [("resource_development", {"capacity": 32, "batch_size": 8, "rounds": 1}),
                  ("resource_development", {"capacity": 128, "batch_size": 32, "rounds": 2}),
                  ("retention_calibration", {"fd_strength": .1, "ridge": .1}),
                  ("retention_calibration", {"fd_strength": .2, "ridge": .1}),
                  ("retention_calibration", {"fd_strength": .2, "ridge": 1.})]
    for i, (phase, recipe) in enumerate(candidates):
        unit(tmp_path, str(i), method="td_fd", seed=7,
             settings={"phase": phase, "ewc_rank": 3, "recipe": recipe}, evaluations=[panel()])
    rows = report.evaluation_rows(report.load_report(tmp_path))
    assert len(rows) == 5 and len({r["Condition"] for r in rows}) == 5
    assert all(r["Trajectories"] == 1 and r["Replicas"] == 1 for r in rows)
    assert "C=32" in rows[0]["Condition"] or any("C=32" in r["Condition"] for r in rows)


def test_exact_recipe_groups_pool_only_replicas_independent_of_dict_order(tmp_path):
    for i, recipe in enumerate(({"capacity": 32, "ridge": .1}, {"ridge": .1, "capacity": 32})):
        unit(tmp_path, str(i), method="td_replay", seed=i,
             settings={"phase": "confirmation", "recipe": recipe}, evaluations=[panel()])
    rows = report.evaluation_rows(report.load_report(tmp_path))
    assert len(rows) == 1 and rows[0]["Replicas"] == rows[0]["Trajectories"] == 2


def test_confirmation_preferred_resource_recipes_table_only_and_plot_count_bounded(tmp_path):
    for name, phase in (("resource", "resource_development"), ("dev", "development"), ("cal", "retention_calibration")):
        unit(tmp_path, name, settings={"method": "td_replay", "phase": phase, "recipe": {"capacity": 32}},
             evaluations=[panel()])
    snapshot = report.load_report(tmp_path)
    assert {u["name"] for u in report._plot_units(snapshot)} == {"dev", "cal"}
    unit(tmp_path, "fresh", settings={"method": "td_fd", "stage": "confirmation", "recipe": {"capacity": 32}},
         evaluations=[panel()])
    snapshot = report.load_report(tmp_path)
    assert [u["name"] for u in report._plot_units(snapshot)] == ["fresh"]
    for index in range(10):
        unit(tmp_path, f"fresh-{index}", settings={"method": "td_fd", "phase": "confirmation",
                                                "recipe": {"capacity": 32 + index}})
    assert len(report._plot_units(report.load_report(tmp_path))) <= report.MAX_PLOT_GROUPS


@pytest.mark.parametrize("row, match", [
    ({"n": 0, "environment": "initial_A", "episodes": 2, "success": 3.}, "fraction"),
    ({"n": 0, "environment": "initial_A", "episodes": "2"}, "episodes"),
])
def test_live_preview_schema_errors_are_clear(tmp_path, row, match):
    progress(tmp_path, active_progress={"evaluation_preview": [row]})
    with pytest.raises(report.ArtifactError, match=match):
        report.load_report(tmp_path)


def test_initial_fit_plot_shows_recorded_curves_minimum_not_final_weights_and_compact_inventory(tmp_path, monkeypatch):
    curves = [{"epoch": 1, "train_mse": 34.7, "validation_mse": 28.45},
              {"epoch": 6, "train_mse": 31.12, "validation_mse": 27.09},
              {"epoch": 60, "train_mse": 30.13, "validation_mse": 27.49}]
    unit(tmp_path, operation="initial_fit", train_count=515, validation_count=65,
         result={"curves": curves, "epoch": 60, "inventory": [
             {"bank": "ltm", "count": 20, "name": "weight-not-a-table-row"}, {"bank": "stm", "count": 3}]})
    figures = []

    def capture(figure, description):
        figures.append(figure)
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", capture)
    outputs = report.result_outputs(report.load_report(tmp_path))
    assert len(figures) == 1
    axis = figures[0].axes[0]
    assert list(axis.lines[0].get_xdata()) == [1, 6, 60]
    assert list(axis.lines[0].get_ydata()) == [34.7, 31.12, 30.13]
    assert list(axis.lines[1].get_ydata()) == [28.45, 27.09, 27.49]
    assert axis.collections[0].get_offsets().tolist() == [[6, 27.09]]
    text = rendered(outputs)
    assert "stopped at epoch 60" in axis.get_title()
    assert "lowest validation 27.09 at epoch 6" in text
    assert "LTM 20 / STM 3" in text and "515 train / 65 validation" in text
    assert "neither convergence nor navigation quality" in text
    assert "weight-not-a-table-row" not in text
    assert "Best-validation weights restored, not final-epoch weights" in figures[0].texts[0].get_text()


def test_incomplete_initializer_never_supplies_diagnostic_and_empty_curves_do_not_invent_one(tmp_path):
    unit(tmp_path, operation="initial_fit", completed=False, result={"curves": "uncommitted"})
    assert report.result_outputs(report.load_report(tmp_path)) == []
    unit(tmp_path, "empty-fit", operation="initial_fit", result={"epoch": 0, "curves": [], "inventory": []})
    assert report._initialization_outputs(report.load_report(tmp_path)) == []


@pytest.mark.parametrize("result, match", [
    ({"epoch": 1, "curves": [{"epoch": 2, "train_mse": 1., "validation_mse": 1.}]}, "epochs"),
    ({"epoch": 1, "curves": [{"epoch": 1, "train_mse": "bad", "validation_mse": 1.}]}, "MSE"),
    ({"epoch": 1, "inventory": [{"bank": "ltm", "count": -1}]}, "inventory"),
])
def test_initial_fit_diagnostic_schema_errors_are_clear(tmp_path, result, match):
    unit(tmp_path, operation="initial_fit", result=result)
    with pytest.raises(report.ArtifactError, match=match):
        report.load_report(tmp_path)


@pytest.mark.parametrize("operation, phase", [("trajectory", "resource_pilot"),
    ("trajectory", "resource_development"), ("resource_pilot", "unspecified"),
    ("resource_development", "unspecified")])
def test_resource_units_are_table_only_including_archive_diagnostics(tmp_path, monkeypatch, operation, phase):
    unit(tmp_path, operation=operation, settings={"method": "td_replay", "phase": phase},
         metrics=[metric()], evaluations=[panel()], archive={"rank": 3, "eigenvalues": [1.]})
    monkeypatch.setattr(report, "_figure", lambda *args: pytest.fail("resource pilot must not produce a figure"))
    snapshot = report.load_report(tmp_path)
    assert report._plot_units(snapshot) == []
    outputs = report.result_outputs(snapshot)
    assert len(outputs) == 1
    assert "Initial A endpoint" in rendered(outputs) and "raw_return" in rendered(outputs)


def test_compact_legends_keep_full_recipes_internal_and_never_merge_similar_labels(tmp_path, monkeypatch):
    for index, learning_rate in enumerate((.001, .002)):
        unit(tmp_path, f"long-unit-name-not-in-legend-{index}", settings={"method": "td_fd", "phase": "development",
             "ewc_rank": 3, "recipe": {"fd_strength": .02, "ridge": .1, "learning_rate": learning_rate}},
             evaluations=[panel(0, episodes=[episode(index + 1.)])], metrics=[metric()])
    snapshot = report.load_report(tmp_path)
    rows = report.evaluation_rows(snapshot)
    assert len(rows) == 2 and len({r["_group"] for r in rows}) == 2
    assert all(r["Trajectories"] == 1 for r in rows)
    legends = report._legend_labels(snapshot["units"])
    assert len(set(legends.values())) == 2
    assert all("TD + FD (dev)" in label and "r=3" in label and "FD=0.02 ridge=0.1" in label
               and len(label) < 65 and "long-unit" not in label for label in legends.values())
    assert all("learning_rate" not in label and "recipe=" not in label for label in legends.values())
    figures = []

    def capture(figure, description):
        figures.append(figure)
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", capture)
    report.result_outputs(snapshot)
    assert len(figures[0].axes[0].lines) == 2
    assert sorted(line.get_ydata()[0] for line in figures[0].axes[0].lines) == [1., 2.]
    for figure in figures:
        assert len(figure.legends) == 1
        assert all(axis.get_legend() is None for axis in figure.axes)


def test_eight_recipe_render_has_no_layout_collapse_and_legend_fits_canvas(tmp_path, monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    for index in range(report.MAX_PLOT_GROUPS):
        unit(tmp_path, f"very-long-artifact-unit-name-that-must-not-be-in-a-legend-{index}",
             settings={"method": "td_fd", "phase": "retention_calibration", "ewc_rank": index,
                       "recipe": {"fd_strength": 10. ** (index - 4), "ridge": .1}},
             metrics=[metric(n) for n in (1, 2)], evaluations=[panel(0), panel(2)],
             archive={"rank": index, "eigenvalues": [1.]})
    checked = []

    def render(figure, description):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            canvas = FigureCanvasAgg(figure)
            canvas.draw()
        assert not any("constrained_layout" in str(w.message) or "collapsed" in str(w.message) for w in caught)
        renderer = canvas.get_renderer()
        for legend in figure.legends:
            bounds = legend.get_window_extent(renderer)
            assert figure.bbox.contains(bounds.x0, bounds.y0)
            assert figure.bbox.contains(bounds.x1, bounds.y1)
        assert all(axis.get_window_extent(renderer).height > 75 for axis in figure.axes)
        checked.append(True)
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", render)
    report.result_outputs(report.load_report(tmp_path))
    assert len(checked) == 3


def confirmation_unit(root, name, method, seed=7, **fields):
    settings = {"method": method, "phase": "confirmation", "seed": seed, "total": 8,
                "recipe": {"capacity": 32, "fd_strength": .1 if method == "td_fd" else 0.}}
    settings.update(fields.pop("settings", {}))
    return unit(root, name, settings=settings, **fields)


def complete_report(root):
    count = len(list(root.glob("units/*/COMPLETED")))
    progress(root, status="completed", stage="confirmation", completed_units=count, planned_units=count,
             active_unit=None)
    return report.load_report(root)


def test_final_confirmation_gaps_pair_then_average_without_episode_weighting(tmp_path):
    outcomes = [
        (7, "mc_replay", [episode(2., success=True), episode(6.)], [episode(-2.)], 20.),
        (7, "td_replay", [episode(1.)], [episode(-4.)], 10.),
        (7, "td_fd", [episode(-1., success=True)], [episode(-3.)], 5.),
        (8, "mc_replay", [episode(12.)] * 5, [episode(4.)] * 2, 6.),
        (8, "td_replay", [episode(3., success=True)] * 3, [episode(0.)] * 4, 2.),
        (8, "td_fd", [episode(3., success=True)], [episode(2.)], 8.),
    ]
    for seed, method, initial, moderate, seconds in outcomes:
        confirmation_unit(tmp_path, f"{seed}-{method}", method, seed, active_seconds=seconds,
            evaluations=[panel(0, episodes=[episode(9999.)]), panel(8, episodes=initial),
                         panel(8, "moderate_A", moderate)])
    snapshot = complete_report(tmp_path)
    rows = report.paired_endpoint_rows(snapshot)
    assert len(rows) == 8 and all(row["Pairs"] == 2 and row["Nominal n"] == 8 for row in rows)
    assert {row["Metric"] for row in rows} == {"Raw observed return", "Success (pp)"}
    mc = [r for r in rows if r["Comparison"].startswith("MC replay") and r["Endpoint"] == "initial_A retention"]
    raw, success = mc
    assert raw["Mean gap"] == 6. and raw["Individual-gap range (not CI)"] == [3., 9.]
    assert success["Mean gap"] == -25. and success["Individual-gap range (not CI)"] == [-100., 50.]
    moderate = next(r for r in rows if r["Comparison"].startswith("MC replay")
                    and r["Endpoint"] == "moderate_A acquisition" and r["Metric"] == "Raw observed return")
    assert moderate["Mean gap"] == 3. and moderate["Individual-gap range (not CI)"] == [2., 4.]
    fd = next(r for r in rows if r["Comparison"].startswith("TD + FD")
              and r["Endpoint"] == "initial_A retention" and r["Metric"] == "Raw observed return")
    assert fd["Mean gap"] == -1. and fd["Individual-gap range (not CI)"] == [-2., 0.]
    costs = report.paired_cost_rows(snapshot)
    assert [r["Mean active-time ratio (left/TD)"] for r in costs] == [2.5, 2.25]
    assert costs[0]["Individual-ratio range (not CI)"] == [2., 3.]
    assert all(r["Pairs"] == 2 for r in costs)
    json.dumps({"endpoints": rows, "costs": costs}, allow_nan=False)
    text = rendered(report.result_outputs(snapshot))
    assert "Bellman control" in text and "Individual-gap range (not CI)" in text
    assert "labels differ" in text and "not confidence intervals" in text
    conclusion = report.conclusion(snapshot)
    assert "initial_A retention +6 raw return (2 pairs)" in conclusion
    assert "initial_A retention -25 pp success (2 pairs)" in conclusion
    assert "TD + FD - TD replay" in conclusion and "2.5 (2 pairs)" in conclusion
    assert "Two paired replicas do not establish significance" in conclusion
    assert "do not guarantee savings" in conclusion


def test_final_comparisons_are_withheld_until_entire_scope_is_complete(tmp_path):
    for method in ("mc_replay", "td_replay", "td_fd"):
        confirmation_unit(tmp_path, method, method, evaluations=[panel(8)])
    for fields in ({"status": "running", "completed_units": 3, "planned_units": 3},
                   {"status": "completed", "completed_units": 3, "planned_units": 4},
                   {"status": "completed", "completed_units": 4, "planned_units": 4}):
        progress(tmp_path, **fields)
        snapshot = report.load_report(tmp_path)
        assert not snapshot["complete"]
        assert report.paired_endpoint_rows(snapshot) == report.paired_cost_rows(snapshot) == []
        conclusion = report.conclusion(snapshot)
        assert "Provisional / incomplete" in conclusion and "await completion" in conclusion
        assert "Completed confirmation endpoint gaps" not in conclusion
        assert "Final Paired Endpoint Gaps" not in rendered(report.result_outputs(snapshot))
    snapshot = complete_report(tmp_path)
    assert len(report.paired_endpoint_rows(snapshot)) == 4


def test_missing_partners_never_turn_unpaired_means_into_paired_gaps(tmp_path):
    confirmation_unit(tmp_path, "mc-seven", "mc_replay", 7, evaluations=[panel(8, episodes=[episode(1000.)])])
    confirmation_unit(tmp_path, "td-eight", "td_replay", 8, evaluations=[panel(8, episodes=[episode(2.)])])
    confirmation_unit(tmp_path, "fd-eight", "td_fd", 8, evaluations=[panel(8, episodes=[episode(5.)])])
    confirmation_unit(tmp_path, "fd-nine", "td_fd", 9, evaluations=[panel(8, episodes=[episode(9999.)])])
    snapshot = complete_report(tmp_path)
    rows = report.paired_endpoint_rows(snapshot)
    assert len(rows) == 2 and all(r["Pairs"] == 1 and r["Comparison"] == "TD + FD - TD replay" for r in rows)
    assert rows[0]["Mean gap"] == 3. and rows[0]["Individual-gap range (not CI)"] == [3., 3.]
    assert len(report.paired_cost_rows(snapshot)) == 1


@pytest.mark.parametrize("left, right, paired", [
    ({"seed": 7}, {"seed": 7}, True),
    ({"replica": "one"}, {"replica": "one"}, True),
    ({"seed": 7}, {"seed": "7"}, False),
    ({"seed": 7, "replica": 1}, {"seed": 8, "replica": 1}, False),
    ({"seed": 7, "replica": 1}, {"seed": 7, "replica": 2}, False),
    ({"seed": 7, "replica": 1}, {"seed": 7}, False),
    ({"replica": 7}, {"seed": 7}, False),
    ({}, {}, False),
    ({"seed": None}, {"seed": None}, False),
])
def test_pairing_uses_exact_seed_and_replica_identity(tmp_path, left, right, paired):
    for method, identity in (("mc_replay", left), ("td_replay", right)):
        unit(tmp_path, method, settings={"method": method, "phase": "confirmation", "total": 8, **identity},
             evaluations=[panel(8)])
    snapshot = complete_report(tmp_path)
    rows = report.paired_endpoint_rows(snapshot)
    assert bool(rows) is paired
    assert bool(report.paired_cost_rows(snapshot)) is paired
    assert all(r["Pairs"] == 1 for r in rows)


def test_summary_seed_fallback_is_supported_but_conflicting_seed_is_omitted(tmp_path):
    for method in ("mc_replay", "td_replay"):
        unit(tmp_path, method, seed=7, settings={"method": method, "stage": "confirmation", "total": 8},
             evaluations=[panel(8)])
    assert len(report.paired_endpoint_rows(complete_report(tmp_path))) == 2
    unit(tmp_path, "conflicting-fd", seed=8,
         settings={"method": "td_fd", "phase": "confirmation", "seed": 7, "total": 8}, evaluations=[panel(8)])
    rows = report.paired_endpoint_rows(complete_report(tmp_path))
    assert len(rows) == 2 and all(r["Comparison"].startswith("MC replay") for r in rows)


def test_confirmation_pairs_exclude_development_resources_and_shared_initialization(tmp_path):
    for phase in ("development", "resource_pilot", "resource_development", "retention_calibration", "fresh"):
        for method in ("mc_replay", "td_replay", "td_fd"):
            confirmation_unit(tmp_path, phase + method, method, settings={"phase": phase},
                              evaluations=[panel(8, episodes=[episode(9999.)])])
    confirmation_unit(tmp_path, "shared", "td_replay", operation="initial_evaluation",
                      evaluations=[panel(0), panel(8, episodes=[episode(9999.)])])
    confirmation_unit(tmp_path, "td", "td_replay", evaluations=[panel(0), panel(8, episodes=[episode(1.)])])
    confirmation_unit(tmp_path, "fd", "td_fd", evaluations=[panel(0), panel(8, episodes=[episode(3.)])])
    snapshot = complete_report(tmp_path)
    rows = report.paired_endpoint_rows(snapshot)
    assert len(rows) == 2 and rows[0]["Mean gap"] == 2.
    assert all(r["Comparison"] == "TD + FD - TD replay" and r["Pairs"] == 1 for r in rows)
    assert len(report.paired_cost_rows(snapshot)) == 1


@pytest.mark.parametrize("differing_recipe", [False, True])
@pytest.mark.parametrize("ambiguous_method", ["mc_replay", "td_replay", "td_fd"])
def test_repeated_method_seed_is_omitted_instead_of_selected_or_counted_twice(tmp_path, differing_recipe, ambiguous_method):
    for method in ("mc_replay", "td_replay", "td_fd"):
        for seed in (7, 8):
            confirmation_unit(tmp_path, f"{method}-{seed}", method, seed, evaluations=[panel(8)])
    recipe = {"capacity": 64 if differing_recipe else 32, "fd_strength": .1 if ambiguous_method == "td_fd" else 0.}
    confirmation_unit(tmp_path, "ambiguous-seven", ambiguous_method, settings={"recipe": recipe},
                      evaluations=[panel(8, episodes=[episode(9999.)])])
    snapshot = complete_report(tmp_path)
    rows = report.paired_endpoint_rows(snapshot)
    mc_count = 1 if ambiguous_method in ("mc_replay", "td_replay") else 2
    fd_count = 1 if ambiguous_method in ("td_fd", "td_replay") else 2
    assert {r["Pairs"] for r in rows if r["Comparison"].startswith("MC replay")} == {mc_count}
    assert {r["Pairs"] for r in rows if r["Comparison"].startswith("TD + FD")} == {fd_count}
    assert [r["Pairs"] for r in report.paired_cost_rows(snapshot)] == [mc_count, fd_count]


def test_distinct_recipe_pairs_across_seeds_remain_separate_cohorts(tmp_path):
    for seed, capacity in ((7, 32), (8, 64)):
        confirmation_unit(tmp_path, f"mc-{seed}", "mc_replay", seed,
                          settings={"recipe": {"capacity": capacity}}, evaluations=[panel(8, episodes=[episode(seed)])])
        confirmation_unit(tmp_path, f"td-{seed}", "td_replay", seed, evaluations=[panel(8)])
    snapshot = complete_report(tmp_path)
    rows = report.paired_endpoint_rows(snapshot)
    assert len(rows) == 4 and all(r["Pairs"] == 1 for r in rows)
    assert len({r["Comparison"] for r in rows}) == 2
    assert all("recipe pair" in r["Comparison"] for r in rows)
    assert len(report.paired_cost_rows(snapshot)) == 2


@pytest.mark.parametrize("left_budget, right_budget, valid", [
    (8, 8, True), (8, 16, False), (None, None, False), (8, None, False),
    (8, "8", False), (0, 0, False), (8., 8., False),
])
def test_compute_ratios_require_equal_positive_declared_nominal_budgets(tmp_path, left_budget, right_budget, valid):
    for method, budget, acquired in (("mc_replay", left_budget, 11), ("td_replay", right_budget, 9)):
        confirmation_unit(tmp_path, method, method, settings={"total": budget}, n=acquired,
                          evaluations=[{**panel(8), "acquired_count": acquired}])
    snapshot = complete_report(tmp_path)
    assert bool(report.paired_endpoint_rows(snapshot)) is valid
    costs = report.paired_cost_rows(snapshot)
    assert bool(costs) is valid
    if valid:
        assert costs[0]["Nominal n"] == 8 and costs[0]["Mean active-time ratio (left/TD)"] == 1.


@pytest.mark.parametrize("zero_method", ["mc_replay", "td_replay"])
def test_compute_ratios_omit_zero_duration_pairs_without_hiding_performance(tmp_path, zero_method):
    for method in ("mc_replay", "td_replay"):
        for seed in (7, 8):
            confirmation_unit(tmp_path, f"{method}-{seed}", method, seed,
                              active_seconds=0. if seed == 7 and method == zero_method else 10., evaluations=[panel(8)])
    snapshot = complete_report(tmp_path)
    assert all(r["Pairs"] == 2 for r in report.paired_endpoint_rows(snapshot))
    cost = report.paired_cost_rows(snapshot)[0]
    assert cost["Pairs"] == 1 and cost["Individual-ratio range (not CI)"] == [1., 1.]


def test_final_endpoints_require_exact_budget_and_report_metric_specific_pair_counts(tmp_path):
    for seed in (7, 8):
        confirmation_unit(tmp_path, f"mc-{seed}", "mc_replay", seed,
                          evaluations=[panel(8, episodes=[episode(4. if seed == 7 else None, success=True)])])
        confirmation_unit(tmp_path, f"td-{seed}", "td_replay", seed,
                          evaluations=[panel(8, episodes=[episode(1.)]), panel(4, "moderate_A")])
    confirmation_unit(tmp_path, "shared-positive-panel", "td_fd",
                      evaluations=[{**panel(8), "shared_initial_source": "common-initial-evaluation"}])
    rows = report.paired_endpoint_rows(complete_report(tmp_path))
    assert len(rows) == 2 and all(r["Endpoint"] == "initial_A retention" for r in rows)
    assert rows[0]["Pairs"] == 1 and rows[0]["Mean gap"] == 3.
    assert rows[1]["Pairs"] == 2 and rows[1]["Mean gap"] == 100.


def test_completed_scope_without_confirmation_pairs_makes_no_comparative_claim(tmp_path):
    confirmation_unit(tmp_path, "only-mc", "mc_replay", evaluations=[panel(8)])
    snapshot = complete_report(tmp_path)
    assert report.paired_endpoint_rows(snapshot) == report.paired_cost_rows(snapshot) == []
    conclusion = report.conclusion(snapshot)
    assert "Approved scope complete" in conclusion
    assert "No unambiguous matched-budget confirmation endpoint pairs" in conclusion
    assert "Completed confirmation endpoint gaps" not in conclusion


def test_complete_paired_notebook_refresh_still_preserves_all_historical_cells(tmp_path):
    root, base, output = tmp_path / "run", tmp_path / "base.ipynb", tmp_path / "out.ipynb"
    history = base_book(base)
    for method in ("mc_replay", "td_replay", "td_fd"):
        confirmation_unit(root, method, method, evaluations=[panel(8), panel(8, "moderate_A")])
    complete_report(root)
    report.build_notebook(root, output, base)
    report.build_notebook(root, output)
    book = json.loads(output.read_text())
    assert [c for c in book["cells"] if not c.get("metadata", {}).get(report._TAG)] == history["cells"]
    assert book["metadata"] == history["metadata"]
    outputs = next(c["outputs"] for c in book["cells"] if c.get("metadata", {}).get(report._TAG)
                   and "display_results" in c.get("source", ""))
    assert "Final Paired Endpoint Gaps" in rendered(outputs)
    assert "Measured Whole-Trajectory Compute Ratios" in rendered(outputs)
    assert "Completed confirmation endpoint gaps" in book["cells"][-1]["source"]
    nbformat.validate(nbformat.from_dict(book))


def test_rank_report_exposes_development_without_pooling_fresh_or_calibration(tmp_path, monkeypatch):
    for rank in (0, 10):
        for method in ("td_replay", "td_fd"):
            unit(tmp_path, f"dev-{method}-{rank}", settings={
                "method": method, "seed": 7, "phase": "development", "ewc_rank": rank, "total": 8},
                evaluations=[panel(0), panel(8, episodes=[episode(2., success=True), episode(4.)]),
                             panel(8, "moderate_A", [episode(-5.), episode(-1.)])])
    for phase in ("confirmation", "retention_calibration", "resource_development"):
        unit(tmp_path, phase, settings={"method": "td_fd", "seed": 7, "phase": phase,
                                      "ewc_rank": 0, "total": 8}, evaluations=[panel(8, episodes=[episode(999.)])])
    unit(tmp_path, "pending", completed=False, settings={"phase": "development", "total": 8})
    unit(tmp_path, "selection", operation="selection", selected={
        "rank_scores": {"0": 123.26, "10": 122.55}, "recipe": {"rank": 0}})
    snapshot = report.load_report(tmp_path)
    rows = report.rank_development_rows(snapshot)
    assert len(rows) == 4
    assert all(r["Initial A return"] == 3. and r["Moderate A return"] == -3. for r in rows)
    assert all(r["Initial A wins"] == "1/2" and r["Moderate A wins"] == "0/2" for r in rows)
    figures = []

    def capture(figure, description):
        figures.append(figure)
        return nbformat.v4.new_output("display_data", data={"text/plain": description})

    monkeypatch.setattr(report, "_figure", capture)
    text = rendered(report.rank_outputs(snapshot))
    assert "Full-Curriculum Rank Development" in text
    assert len(figures) == 1 and all(len(a.lines) == 4 for a in figures[0].axes)
    assert {line.get_linestyle() for line in figures[0].axes[0].lines} == {"-", "--"}
    assert all(max(line.get_ydata()) < 999 for a in figures[0].axes for line in a.lines)
    notes = report.rank_selection_notes(snapshot)
    assert "123.2600" in notes and "122.5500" in notes
    assert "Retention refinement occurred after rank selection" in notes
    assert "not a sample-efficiency statistic" in notes


def test_rank_cost_table_is_matched_by_rank_and_clearly_a_cached_screen(tmp_path):
    unit(tmp_path, settings={"method": "td_replay", "phase": "development",
                           "ewc_rank": 0, "seed": 7, "total": 8}, evaluations=[panel(8)])
    candidates = [{"recipe": {"rank": rank, "capacity": 256, "batch_size": 64,
                               "rounds": 4, "fd_strength": fd},
                   "fit_seconds": .01 + .02 * fd + .001 * rank, "retirement_seconds": .006}
                  for rank in (0, 3, 5, 10) for fd in (0., 1.)]
    unit(tmp_path, "cached", operation="cached_screen", candidates=candidates)
    snapshot = report.load_report(tmp_path)
    output = report.rank_outputs(snapshot)[-1]
    text = rendered([output])
    assert "Cached Rank Cost" in text and "selected online strength 12.14" in text
    assert "'Fit with FD=1 (ms)': '40'" in text
    output_file = tmp_path / "report.ipynb"
    report.build_notebook(tmp_path, output_file)
    book = json.loads(output_file.read_text())
    ids = [c["id"] for c in book["cells"]]
    assert ids.index("plan19-phase3-fix5-rank-development") < ids.index("plan19-phase3-fix5-results")
    nbformat.validate(nbformat.from_dict(book))
