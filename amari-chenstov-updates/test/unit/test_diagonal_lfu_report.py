"""Synthetic scalar-only Plan 16 analysis and portable CPU report publication."""

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

from mnist_experiment.diagonal_lfu import analysis, report


@pytest.fixture(autouse=True)
def plotting_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))


def trajectory(condition="ema", replica=0, *, stage="fresh", schedule="main", c=.025,
               ridge=.1, completed=True, delta=.02):
    correction = {"ema": 0, "paired_fd": delta, "analytic_lfu": delta / 2}[condition]
    x = (0, 4, 12, 20)
    rows = []
    for i, arrivals in enumerate(x):
        rows.append(dict(step=arrivals // 4, arrivals=arrivals, p=.005 if i < 2 else .1,
            segment="low_hold" if i < 2 else "rise", expected_nines=.1 * arrivals, observed_nines=arrivals // 4,
            nine_recall=.2 + .1 * i + correction * i, old_accuracy=.9 - .02 * i - correction * i / 4,
            frozen_nine_recall=.2, frozen_old_accuracy=.9,
            nine_fpr=.01, nine_precision=None if i == 0 else .7, nll=.5, wall_seconds=float(arrivals * (2 if condition != "ema" else 1)),
            negative_fraction=.01 if condition != "ema" else 0, negative_mass=.001, closure_count=i + 1, grad_norm=.1 / (i + 1),
            initial_nines=3, unique_ids=arrivals))
    return dict(unit_id=f"{stage}-{schedule}-{condition}-{replica}-{c}-{ridge}", stage=stage,
                replica=replica, schedule=schedule, condition=condition, c=c, ridge=ridge,
                completed=completed, metrics=rows, summary={"costs": {"evaluation_seconds": 2.}, "initial_nines": 3})


def snapshot(trajectories=None):
    return dict(schema=1, study_id="synthetic-study", status="running", phase="3: fresh",
        config={"samples_per_step": 4, "dtype": "float32"},
        progress={"completed_blocks": 1, "planned_blocks": 8, "completed_fresh": 6,
                  "planned_fresh": 36, "active_hours": .5, "remaining_hours": 19.5, "allocation_hours": 20,
                  "active_unit": "pending-next", "step": 3, "heartbeat": "2026-11-01T10:00:00Z", "revision": "v1",
                  "storage": {"used_bytes": 1_000_000_000, "external_bytes": 2_000_000_000,
                              "max_bytes": 500_000_000_000, "filesystem_free_bytes": 30_000_000_000}},
        changes=[{"phase": "2", "revision": "v1", "description": "Common setting frozen", "validation": "unit tests"}],
        trajectories=trajectories or [], selection={"common_c": .025, "common_ridge": .1}, analysis={})


def write_snapshot(root, value):
    root.mkdir(parents=True, exist_ok=True)
    (root / "progress.json").write_text(json.dumps(value["progress"]))
    (root / "report_snapshot.json").write_text(json.dumps(value))


def contrast(result, condition="paired_fd", group=0):
    return next(c for c in result["groups"][group]["comparisons"] if c["condition"] == condition)


def test_observation_weighting_signs_endpoints_segments_and_costs():
    rows = [trajectory("ema"), trajectory("paired_fd")]
    result = analysis.summarize(rows, bootstrap_draws=20, seed=7)
    effect = contrast(result)
    # Nonuniform observation spacing: integral(0,.02,.04,.06) / 20 = .034.
    assert effect["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(3.4)
    assert effect["observation_mean_pp"]["old_accuracy"]["mean"] == pytest.approx(-.85)
    assert effect["endpoint_pp"]["nine_recall"]["mean"] == pytest.approx(6)
    assert effect["segments"]["low_hold"]["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(1)
    assert effect["segments"]["rise"]["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(4)
    assert effect["costs"]["wall_seconds"]["ratio"]["mean"] == 2
    assert effect["costs"]["wall_seconds"]["difference"]["mean"] == 20
    assert effect["costs"]["evaluation_seconds"]["difference"]["mean"] == 0
    assert effect["observation_mean_pp"]["nine_recall"]["ci95"] is None
    assert not effect["observation_mean_pp"]["nine_recall"]["interval_reliable"]
    assert contrast(result, "analytic_lfu")["n_pairs"] == 0
    json.dumps(result, allow_nan=False)


def test_complete_pairs_only_and_group_settings_schedule_stage_without_pooling():
    rows = [trajectory("ema"), trajectory("paired_fd"), trajectory("ema", 1),
            trajectory("paired_fd", 1, completed=False)]
    rows += [trajectory(arm, stage=stage, schedule=schedule, c=c, ridge=ridge)
             for stage, schedule, c, ridge in (("development", "main", .025, .1), ("fresh", "stationary", .025, .1),
                 ("fresh", "main", .05, .1), ("fresh", "main", .025, .01))
             for arm in ("ema", "paired_fd")]
    result = analysis.summarize(rows, bootstrap_draws=0)
    assert len(result["groups"]) == 5
    assert all(contrast({"groups": [g]})["n_pairs"] == 1 for g in result["groups"])
    assert result["counts"] == {"sealed_trajectories": 12, "complete": 11, "incomplete": 1}
    main = next(g for g in result["groups"] if g["stage"] == "fresh" and g["schedule"] == "main" and g["c"] == .025 and g["ridge"] == .1)
    assert contrast({"groups": [main]})["unpaired_control_units"] == [rows[2]["unit_id"]]
    assert main["incomplete_units"] == [rows[3]["unit_id"]]


def test_bootstrap_resamples_replica_scalars_and_persists_seed(monkeypatch):
    rows = [trajectory(arm, replica=r, delta=.01 * (r + 1)) for r in range(4) for arm in ("ema", "paired_fd", "analytic_lfu")]
    calls = []
    original = analysis._estimate
    def tracked(values, **kwargs):
        calls.append(len(values))
        return original(values, **kwargs)
    monkeypatch.setattr(analysis, "_estimate", tracked)
    result = analysis.summarize(rows, bootstrap_draws=500, seed=9)
    assert result == analysis.summarize(deepcopy(rows), bootstrap_draws=500, seed=9)
    assert result["bootstrap"]["seed"] == 9
    effect = contrast(result)["observation_mean_pp"]["nine_recall"]
    assert effect["individual"] == pytest.approx([1.7, 3.4, 5.1, 6.8])
    assert effect["mean"] == pytest.approx(4.25)
    assert effect["ci95"][0] < effect["mean"] < effect["ci95"][1]
    assert effect["interval_reliable"]
    assert set(calls) == {4}


def test_bootstrap_draw_bounds_and_small_pair_warning():
    empty = analysis.summarize([], bootstrap_draws=10**9)
    assert empty["bootstrap"]["draws"] == analysis.MAX_BOOTSTRAP_DRAWS
    assert analysis.summarize([], bootstrap_draws=-1)["bootstrap"]["draws"] == 0
    rows = [trajectory(arm, r) for r in range(3) for arm in ("ema", "paired_fd")]
    result = contrast(analysis.summarize(rows, bootstrap_draws=200))
    assert not result["observation_mean_pp"]["nine_recall"]["interval_reliable"]
    assert len(result["observation_mean_pp"]["nine_recall"]["individual"]) == 3


@pytest.mark.parametrize("name,value", [("arrivals", 99), ("p", .2), ("observed_nines", 19), ("segment", "different")])
def test_mismatched_complete_pair_is_not_truncated_or_interpolated(name, value):
    rows = [trajectory("ema"), trajectory("paired_fd")]
    rows[1]["metrics"][-1][name] = value
    with pytest.raises(ValueError, match="mismatched"):
        analysis.summarize(rows)


@pytest.mark.parametrize("mutate", [
    lambda t: t["metrics"][1].update(arrivals=0),
    lambda t: t["metrics"][0].update(arrivals=1),
    lambda t: t["metrics"][0].update(nine_recall=80),
    lambda t: t["metrics"][0].update(nine_recall=float("nan")),
    lambda t: t["metrics"][0].update(nine_recall=None),
    lambda t: t["metrics"][-1].update(wall_seconds=-1),
    lambda t: t["metrics"][-1].update(frozen_nine_recall=.5),
    lambda t: t.update(completed="true"),
    lambda t: t.update(metrics=[]),
    lambda t: t.update(c=0),
])
def test_corrupt_scalar_trajectories_fail_clearly(mutate):
    row = trajectory()
    mutate(row)
    with pytest.raises(ValueError):
        analysis.summarize([row])


def test_duplicate_alias_and_pairing_identity_rejected():
    row = trajectory("paired_fd")
    alias = deepcopy(row)
    alias.update(condition="fd", unit_id="alias")
    with pytest.raises(ValueError, match="duplicate"):
        analysis.summarize([row, alias])
    row["stream_hash"] = "one"
    control = trajectory("ema")
    control["stream_hash"] = "two"
    with pytest.raises(ValueError, match="stream_hash"):
        analysis.summarize([row, control])


def test_report_publishes_six_real_pngs_three_tables_portable_outputs_and_no_scientific_writes(tmp_path):
    root, notebook = tmp_path / "study", tmp_path / "mnist_experiment" / "plan16_results.ipynb"
    rows = [trajectory(arm, stage=stage, schedule=schedule)
            for stage in ("development", "fresh") for schedule in ("main", "stationary")
            for arm in ("ema", "paired_fd", "analytic_lfu")]
    rows += [trajectory("paired_fd", replica=5, completed=False)]
    value = snapshot(rows)
    value["analysis"] = analysis.summarize(rows, bootstrap_draws=20, seed=123)
    write_snapshot(root, value)
    sealed = root / "units" / "sealed"
    sealed.mkdir(parents=True)
    (sealed / "COMPLETED").write_text("immutable")
    (sealed / "checkpoint.pt").write_bytes(b"must never be read or loaded")
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = report.build_report(root, notebook)
    book = nbformat.read(notebook, as_version=4)
    nbformat.validate(book)
    assert result["notebook"] == str(notebook)
    assert result["counts"]["incomplete"] == 1
    assert result == json.loads((root / "report" / "latest.json").read_text())
    output = Path(result["report_dir"])
    assert len(list(output.glob("*.png"))) == 6
    code = [c for c in book.cells if c.cell_type == "code"]
    assert len(code) == 6
    for cell in code:
        assert cell.execution_count is not None
        png = base64.b64decode(cell.outputs[0].data["image/png"])
        pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        assert pixels.shape[0] > 600 and pixels.shape[1] > 1000
        assert pixels.std() > 5
        assert not any(word in cell.source for word in ("train", "torch", "summarize", "bootstrap", "download"))
        assert (notebook.parent / cell.source.split("filename=", 1)[1].split(")", 1)[0].strip("'\"")).is_file()
    text = "\n".join(c.source for c in book.cells if c.cell_type == "markdown")
    assert sum(line.startswith("| ---") for line in text.splitlines()) == 3
    for phrase in ("Partial / incomplete", "Development", "Fresh Main", "Fresh Stationary", "unreliable",
                   "20.00", "497.00", "0.50", "Frozen initialization", "not pooled", "persisted seed: 123", "No efficacy gate"):
        assert phrase in text
    assert "snapshot_sha256" in book.metadata.plan16_report
    for path, contents in before.items():
        assert Path(path).read_bytes() == contents
    assert not list((root / "report").glob(".building-*"))


def test_pending_cli_is_atomic_idempotent_and_reports_incomplete(tmp_path, capsys):
    root, notebook = tmp_path / "study", tmp_path / "plan16_results.ipynb"
    write_snapshot(root, snapshot())
    report.main(["--root", str(root), "--notebook", str(notebook)])
    result = json.loads(capsys.readouterr().out)
    before = notebook.read_bytes()
    assert result == report.build_report(root, notebook)
    assert notebook.read_bytes() == before
    assert len(list((root / "report").glob("generation-*"))) == 1
    book = nbformat.read(notebook, as_version=4)
    assert "No complete fresh paired evidence" in "\n".join(c.source for c in book.cells)
    assert "Partial / incomplete" in book.cells[0].source


def test_failed_render_preserves_previous_notebook_and_index(tmp_path, monkeypatch):
    root, notebook = tmp_path / "study", tmp_path / "plan16_results.ipynb"
    write_snapshot(root, snapshot())
    report.build_report(root, notebook)
    before = notebook.read_bytes(), (root / "report" / "latest.json").read_bytes()
    value = snapshot()
    value["phase"] = "changed"
    write_snapshot(root, value)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic renderer failure")
    monkeypatch.setattr(report, "_fresh", fail)
    with pytest.raises(RuntimeError, match="synthetic renderer"):
        report.build_report(root, notebook)
    assert before == (notebook.read_bytes(), (root / "report" / "latest.json").read_bytes())
    assert not list((root / "report").glob(".building-*"))


@pytest.mark.parametrize("corruption", ("schema", "missing", "stale", "nonfinite", "study_id"))
def test_invalid_snapshot_never_publishes_or_silently_repairs(tmp_path, corruption):
    root, notebook = tmp_path / "study", tmp_path / "plan16_results.ipynb"
    value = snapshot([trajectory("ema"), trajectory("paired_fd")])
    if corruption == "schema":
        value["schema"] = 2
    elif corruption == "missing":
        del value["config"]
    elif corruption == "stale":
        value["analysis"] = analysis.summarize([])
    elif corruption == "nonfinite":
        value["trajectories"][0]["metrics"][0]["nll"] = float("nan")
    else:
        value["progress"]["study_id"] = "wrong"
    write_snapshot(root, value)
    with pytest.raises(ValueError):
        report.build_report(root, notebook)
    assert not notebook.exists()
    assert not (root / "report").exists()


def test_missing_artifacts_fail_and_report_import_has_no_core_dependencies(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.build_report(tmp_path, tmp_path / "report.ipynb")
    subprocess.run([sys.executable, "-c", "import sys; from mnist_experiment.diagonal_lfu import report; "
        "assert 'mnist_experiment.diagonal_lfu.run' not in sys.modules; "
        "assert 'mnist_experiment.diagonal_lfu.data' not in sys.modules; assert 'torch' not in sys.modules"], check=True)


def test_settings_selection_stays_compact_and_progress_uses_parent_hour_and_step_fields():
    value = snapshot()
    value["selection"].update(common_c=.05, common_ridge=.01, main_replicas=8, stationary_replicas=4,
        conditions=["ema", "paired_fd"], forecast_fresh_seconds=3600, contingency=1.25,
        candidates=[{"large_unused_candidate": "verbose" * 1000}] * 100, seconds_by_condition={"ema": 123456})
    settings = report._settings(value, analysis.summarize([])).source
    assert len(settings) < 4200
    assert "large_unused_candidate" not in settings and "123456" not in settings
    for phrase in ("c=0.05; ridge=0.01", "main=8; stationary=4", "EMA, Paired FD", "1.00 h"):
        assert phrase in settings
    value["progress"].update(completed_trajectories=4, planned_trajectories=12, total_steps=20,
        completed_blocks=None, planned_blocks=None, revision=None, forecast_seconds=None)
    progress = report._progress(value, value["progress"], analysis.summarize([])).source
    for phrase in ("4 / 0", "step 3/20", "0.50 / 20.00", "6 / 36", "pending / calibrating"):
        assert phrase in progress
    assert "None" not in progress


def test_change_table_has_brief_sentences_and_links_instead_of_validation_dumps():
    value = snapshot()
    value["changes"] = [
        {"revision": "production_v2", "date": "2026-11-01",
         "change": "Consume recorded initialization model/order seeds; aggregate all development replicas; preserve stop after sealing.",
         "validation": "Long numerical validation details. " * 100},
        {"revision": "production_v3", "date": "2026-11-01",
         "change": "Solve the same physical NLL plus EWC/ridge objective using theta = anchor + scale*z. Report physical gradients.",
         "validation": "Saved-solve objective and closure counts. " * 100,
         "affected_results": "Superseded trajectories remain immutable. " * 100},
    ]
    before = deepcopy(value)
    link = "../cache/study/report_snapshot.json"
    source = report._settings(value, analysis.summarize([]), snapshot_link=link).source
    change_rows = [line for line in source.splitlines() if line.startswith("| Change |")]
    assert len(change_rows) == 2
    assert "Consume recorded initialization model/order seeds." in change_rows[0]
    assert "Solve the same physical NLL plus EWC/ridge objective." in change_rows[1]
    assert all(f"[Full event and validation]({link})" in row for row in change_rows)
    assert all(len(row.split(" | ")[4]) <= 120 for row in change_rows)
    assert "Long numerical validation" not in source and "Saved-solve objective" not in source
    assert "theta =" not in source and "Superseded trajectories" not in source
    assert value == before


def test_change_brief_honors_explicit_summary_and_bounds_long_single_sentences():
    assert report._change_brief({"summary": "Keep the scientific objective unchanged.", "change": "long details"}) == "Keep the scientific objective unchanged."
    brief = report._change_brief({"change": "Many detailed scientific and numerical qualifications " * 20})
    assert len(brief) <= 120 and brief.endswith("...")


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("step", (0, 37))
def test_progress_explicit_accepted_arrival_row_with_flat_or_nested_current(nested, step):
    value = snapshot()
    progress = value["progress"]
    if nested:
        progress.pop("step")
        progress.pop("active_unit")
        progress["current"] = {"active_unit": "online-main", "step": step, "total_steps": 1312}
    else:
        progress.update(step=step, total_steps=1312)
    progress["last_checkpoint"] = "checkpoint-present"
    source = report._progress(value, progress, analysis.summarize([])).source
    assert f"| Accepted incoming-batch step / total_steps | {step}/1312 |" in source
    assert "checkpoint-present" in source
    if nested:
        assert "online-main / checkpoint-present" in source
        assert '"total_steps"' not in source


def test_progress_pilot_unknown_totals_display_calibrating_without_none():
    value = snapshot()
    value["selection"].update(main_replicas=None, stationary_replicas=None)
    value["progress"].update(completed_trajectories=3, planned_trajectories=None,
        completed_blocks=0, planned_blocks=None, completed_fresh=0, planned_fresh=None,
        forecast_seconds=None, step=37, total_steps=None)
    source = report._progress(value, value["progress"], analysis.summarize([])).source
    for phrase in ("3 / 0", "0 / calibrating", "All trajectories planned | calibrating",
                   "37/calibrating", "Remaining forecast hours | calibrating"):
        assert phrase in source
    assert "None" not in source


@pytest.fixture
def captured_plots(monkeypatch):
    recorded = []
    original = report._figure
    @contextmanager
    def capture(*args, **kwargs):
        with original(*args, **kwargs) as (figure, axes):
            yield figure, axes
            recorded.append([dict(xlabel=ax.get_xlabel(), bands=len(ax.patches),
                xticklabels=[label.get_text() for label in ax.get_xticklabels()],
                legend=ax.get_legend() is not None,
                texts=[text.get_text() for text in ax.texts],
                lines=[dict(label=line.get_label(), style=line.get_linestyle(),
                            x=np.asarray(line.get_xdata()).copy()) for line in ax.lines],
                points=[np.asarray(collection.get_offsets()).copy() for collection in ax.collections]) for ax in axes.flat])
    monkeypatch.setattr(report, "_figure", capture)
    return recorded


def test_development_digit_units_compact_legends_regimes_and_schedule_styles(tmp_path, captured_plots):
    rows = [trajectory(arm, stage="development", schedule=schedule)
            for schedule in ("main", "stationary") for arm in ("ema", "paired_fd", "analytic_lfu")]
    for t in rows:
        if t["schedule"] == "stationary":
            for row in t["metrics"]:
                row.update(segment="stationary_low", p=.005, expected_nines=.005 * row["arrivals"], observed_nines=0)
    result = analysis.summarize(rows, bootstrap_draws=0)
    before = json.dumps(result, sort_keys=True), json.dumps(rows, sort_keys=True)
    value = snapshot(rows)
    value["selection"] = {}
    report._development(value, result, tmp_path / "development.png")
    axes = captured_plots[-1]
    for ax in axes[:2]:
        assert ax["xlabel"] == "Online digits observed"
        assert ax["bands"] == 2 and set(ax["texts"]) == {"Low hold", "Rise"}
        for line in ax["lines"]:
            assert "c=0.025" in line["label"] and "(n=1)" in line["label"]
            assert "development" not in line["label"] and "r=" not in line["label"]
            assert line["style"] == ("--" if "stationary" in line["label"] else "-")
            np.testing.assert_array_equal(line["x"], [0, 4, 12, 20])
    assert axes[3]["xlabel"] == "Seconds per online digit"
    assert axes[0]["legend"] and not axes[1]["legend"]
    np.testing.assert_array_equal([float(points[0, 0]) for points in axes[3]["points"]],
                                  [2, 1, 2, 2, 1, 2])
    assert before == (json.dumps(result, sort_keys=True), json.dumps(rows, sort_keys=True))
    settings = report._settings(snapshot(rows), result).source
    assert "one incoming batch" in settings and "`step * samples_per_step`" in settings
    assert "independent stratified holdout" in settings and "Predictive NLL excludes every regularizer" in settings


def test_selected_development_top_keeps_only_main_common_setting_and_all_lower_points(tmp_path, captured_plots):
    rows = [trajectory(arm, stage="development", c=c, ridge=ridge)
            for c, ridge in ((.025, .1), (.05, .1), (.025, .01)) for arm in ("ema", "paired_fd", "analytic_lfu")]
    rows += [trajectory(arm, stage="development", schedule="stationary") for arm in ("ema", "paired_fd", "analytic_lfu")]
    value = snapshot(rows)
    result = analysis.summarize(rows, bootstrap_draws=0)
    before = deepcopy(result), deepcopy(value)
    report._development(value, result, tmp_path / "selected-development.png")
    axes = captured_plots[-1]
    for ax in axes[:2]:
        assert len(ax["lines"]) == 3
        assert {line["label"] for line in ax["lines"]} == {
            "EMA, c=0.025, main (n=1)", "Paired FD, c=0.025, main (n=1)",
            "Analytic LFU, c=0.025, main (n=1)"}
        assert all(line["style"] == "-" for line in ax["lines"])
    assert len(axes[2]["points"]) == len(rows) == len(axes[3]["points"])
    settings = report._settings(value, result).source
    assert "c=0.05" in settings and "ridge=0.01" in settings and "stationary" in settings
    assert before == (result, value)


def test_selected_development_does_not_fall_back_to_other_settings_when_pending(tmp_path, captured_plots):
    rows = [trajectory(arm, stage="development", c=.05) for arm in ("ema", "paired_fd")]
    report._development(snapshot(rows), analysis.summarize(rows, bootstrap_draws=0), tmp_path / "pending-setting.png")
    axes = captured_plots[-1]
    assert not axes[0]["lines"] and not axes[1]["lines"]
    assert len(axes[2]["points"]) == len(rows) == len(axes[3]["points"])


def test_segment_panel_keeps_main_labels_and_distinguishes_methods_metrics(tmp_path, captured_plots):
    rows = [trajectory(arm, schedule=schedule) for schedule in ("main", "stationary")
            for arm in ("ema", "paired_fd", "analytic_lfu")]
    for t in rows:
        if t["schedule"] == "stationary":
            for row in t["metrics"]:
                row.update(segment="stationary_low", p=.005)
    value = snapshot(rows)
    result = analysis.summarize(rows, bootstrap_draws=0)
    report._paired_cost(value, result, tmp_path / "paired.png")
    panel = captured_plots[-1][3]
    assert panel["xticklabels"] == ["Low hold", "Rise"]
    assert {line["label"] for line in panel["lines"] if not line["label"].startswith("_")} == {
        "Paired FD: 9 recall", "Paired FD: 0--8 accuracy",
        "Analytic LFU: 9 recall", "Analytic LFU: 0--8 accuracy"}


def test_health_uses_only_selected_main_paths_and_paired_table_prefers_fresh(tmp_path, captured_plots):
    rows = [trajectory(arm, stage=stage, schedule=schedule, c=c)
            for stage in ("development", "fresh") for schedule in ("main", "stationary")
            for c in (.025, .05) for arm in ("ema", "paired_fd", "analytic_lfu")]
    value = snapshot(rows)
    result = analysis.summarize(rows, bootstrap_draws=0)
    report._health(value, result, tmp_path / "health.png")
    for panel in captured_plots[-1]:
        assert len(panel["lines"]) == 3
        assert all(" main c=0.025" in line["label"] for line in panel["lines"])
    table = report._paired_table(result).source
    assert "fresh / main" in table and "fresh / stationary" in table
    assert "development /" not in table


def test_paired_caption_omits_exact_array_dump_and_conclusion_keeps_one_short_caveat():
    rows = [trajectory(arm, delta=.01723456789) for arm in ("ema", "paired_fd", "analytic_lfu")]
    result = analysis.summarize(rows, bootstrap_draws=0)
    before = deepcopy(result)
    source = report._paired_table(result).source
    assert "Individual full-path differences" not in source
    assert "Exact differences" in source and "analysis.json" in source
    assert "unreliable" in source and "individual differences are plotted" in source
    individual = contrast(result)["observation_mean_pp"]["nine_recall"]["individual"]
    assert str(individual) not in source
    conclusion = report._conclusion(result).source
    assert "No efficacy gate is applied." in conclusion
    assert not any(phrase in conclusion for phrase in ("binary winner", "noninferiority", "familywise", "favorable interval"))
    assert result == before


def test_conclusion_separates_unpaired_development_controls_from_fresh_and_names_tradeoff():
    rows = [trajectory(arm, replica=r) for r in range(4) for arm in ("ema", "paired_fd", "analytic_lfu")]
    rows += [trajectory("ema", stage="development", c=.05)]
    result = analysis.summarize(rows, bootstrap_draws=200)
    source = report._conclusion(result).source
    assert "0 unmatched fresh arms" in source
    assert "higher 9 recall but lower 0--8 retention" in source
    assert "Unpaired development-screen controls remain descriptive" in source
    assert "trade-off, not an improvement in both" in source
    assert "do not establish sample savings" in source
    assert "finite-tolerance solves" in source


def test_undefined_precision_path_explained_without_pending_or_empty_legends(tmp_path, captured_plots):
    rows = [trajectory(arm, schedule="stationary") for arm in ("ema", "paired_fd", "analytic_lfu")]
    for t in rows:
        for row in t["metrics"]:
            row["nine_precision"] = None
    value = snapshot(rows)
    report._precision_exposure(value, analysis.summarize(rows, bootstrap_draws=0), tmp_path / "precision.png")
    panel = captured_plots[-1][1]
    assert not panel["lines"] and not panel["legend"]
    assert "No complete-cohort precision path" in panel["texts"][0]
    assert "EMA: 0/4" in panel["texts"][1]
    assert not any("Pending" in text for text in panel["texts"])


def test_completed_progress_displays_no_active_unit_instead_of_pending():
    value = snapshot()
    value["progress"].update(status="completed", active_unit=None)
    source = report._progress(value, value["progress"], analysis.summarize([])).source
    assert "none (completed)" in source


def test_fresh_single_setting_legend_only_method_and_replica_count(tmp_path, captured_plots):
    rows = [trajectory(arm) for arm in ("ema", "paired_fd", "analytic_lfu")]
    value = snapshot(rows)
    result = analysis.summarize(rows, bootstrap_draws=0)
    report._fresh(value, result, tmp_path / "fresh.png", stationary=False)
    axes = captured_plots[-1]
    assert {line["label"] for line in axes[0]["lines"]} == {
        "EMA (n=1)", "Paired FD (n=1)", "Frozen initial reference (n=1)"}
    assert {line["label"] for line in axes[2]["lines"]} == {
        "EMA (n=1)", "Analytic LFU (n=1)", "Frozen initial reference (n=1)"}
    report._precision_exposure(value, result, tmp_path / "precision.png")
    assert {line["label"] for line in captured_plots[-1][0]["lines"]} == {
        "EMA (n=1)", "Paired FD (n=1)", "Analytic LFU (n=1)"}


def test_fresh_multiple_settings_keep_necessary_legend_context(tmp_path, captured_plots):
    rows = [trajectory(arm, c=c, ridge=ridge) for c, ridge in ((.025, .1), (.05, .01))
            for arm in ("ema", "paired_fd")]
    report._fresh(snapshot(rows), analysis.summarize(rows, bootstrap_draws=0), tmp_path / "fresh.png", stationary=False)
    labels = [line["label"] for line in captured_plots[-1][0]["lines"] if not line["label"].startswith("Frozen")]
    assert len(set(labels)) == 4
    assert all("c=" in label and "r=" in label for label in labels)
    assert not any("fresh;" in label for label in labels)


def test_completed_smoke_notebook_cannot_be_mistaken_for_production(tmp_path):
    root, notebook = tmp_path / "smoke", tmp_path / "plan16_results.ipynb"
    value = snapshot([trajectory(arm) for arm in ("ema", "paired_fd", "analytic_lfu")])
    value.update(status="completed")
    value["config"]["smoke"] = True
    write_snapshot(root, value)
    report.build_report(root, notebook)
    book = nbformat.read(notebook, as_version=4)
    prose = "\n".join(c.source for c in book.cells if c.cell_type == "markdown")
    assert "SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE" in book.cells[0].source
    assert "## Smoke Main Trajectories" in prose and "## Smoke Stationary Trajectories" in prose
    assert "## Fresh Main Trajectories" not in prose
    assert "point estimates show" not in prose
    assert "## Smoke Acceptance Conclusion" in prose
    assert book.metadata.plan16_report.smoke is True
